/******************************************************************************
 * Copyright (c) 2019 Xin Wang
 * National Institute of Informatics, Japan
 *
 * This file is part of CURRENNT.
 *
 * CURRENNT is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * CURRENNT is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with CURRENNT.  If not, see <http://www.gnu.org/licenses/>.
 *****************************************************************************/

#ifdef _MSC_VER
#   pragma warning (disable: 4244) // thrust/iterator/iterator_adaptor.h(121): warning C4244: '+=' : conversion from '__int64' to 'int', possible loss of data
#endif

#include "SseCosPostOutputLayer.hpp"
#include "../helpers/JsonClasses.hpp"
#include "../helpers/getRawPointer.cuh"
#include "../Configuration.hpp"
#include "../MacroDefine.hpp"

#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/functional.h>

namespace internal {
namespace {

    
    struct calculateVectorNormAndCosSquare
    {
	
        int layerSize;
	bool pearsoncorr;
	
	real_t *a_buf;
	real_t *b_buf;
	real_t *output_buf;
	
        const char *patTypes;

        __host__ __device__ void operator() (const thrust::tuple<real_t&, int> &values) const
        {
	    int timeStep = values.get<1>();
	    
	    real_t ab = 0.0, aa = 0.0, bb = 0.0;
	    size_t data_ptr = 0;

	    real_t a_mean = 0.0, b_mean = 0.0;
	    
            if (patTypes[timeStep] == PATTYPE_NONE){
		// do nothing
		
	    }else{

		// calculate mean
		if (pearsoncorr){
		    for (int dim_idx = 0 ; dim_idx < layerSize ; dim_idx++){
			data_ptr = timeStep * layerSize + dim_idx;
			a_mean += a_buf[data_ptr];
			b_mean += b_buf[data_ptr];
		    }
		    a_mean = a_mean / (real_t)layerSize;
		    b_mean = b_mean / (real_t)layerSize;
		}
		for (int dim_idx = 0 ; dim_idx < layerSize ; dim_idx++){
		    data_ptr = timeStep * layerSize + dim_idx;
		    ab += (a_buf[data_ptr] - a_mean) * (b_buf[data_ptr] - b_mean);
		    aa += (a_buf[data_ptr] - a_mean) * (a_buf[data_ptr] - a_mean);
		    bb += (b_buf[data_ptr] - b_mean) * (b_buf[data_ptr] - b_mean);
		}
	    }
	    // [a.b, a.a, b.b, (a.b)^2/(a.a * b.b)]
            output_buf[timeStep * 4]     = ab;
	    output_buf[timeStep * 4 + 1] = aa;
	    output_buf[timeStep * 4 + 2] = bb;
	    output_buf[timeStep * 4 + 3] = (ab / aa) * (ab / bb);
        }
    };

    
    
    struct calculateCosSquareDistance
    {
	
	real_t *output_buf;	
        const char *patTypes;

        __host__ __device__ real_t operator() (const thrust::tuple<const real_t&, int> &values) const
        {
	    int timeStep = values.get<1>();
            if (patTypes[timeStep] == PATTYPE_NONE){
		return 0.0;
	    }else{
		return output_buf[timeStep * 4 + 3];
	    }
        }
    };
    


    struct calculateCosSquareGradient
    {
	int layerSize;
	real_t *output_buf;
	real_t cos_weight;
        const char *patTypes;

        __host__ __device__ void operator() (const thrust::tuple<real_t&, real_t&,
					     real_t&, int> &values) const
        {
	    int timeStep = values.get<3>() / layerSize;
	    
            if (patTypes[timeStep] == PATTYPE_NONE){
		values.get<0>() = 0;
	    }else{
		// gradient = cos^2(a,b) [b/a.b - a/a.a]
		values.get<0>() = cos_weight * output_buf[timeStep * 4 + 3] *
		    (values.get<2>() / output_buf[timeStep * 4 + 0] -
		     values.get<1>() / output_buf[timeStep * 4 + 1]);
	    }
        }
    };
    
    struct calculateErrorMean
    {
	int layerSize;
	real_t *grad_buf;
        const char *patTypes;

        __host__ __device__ void operator() (const thrust::tuple<real_t&, int> &values) const
        {
	    
	    int timeStep = values.get<1>();

	    values.get<0>() = 0;
            if (patTypes[timeStep] == PATTYPE_NONE){
		return;
	    }else{
		
		for (int dim_idx = 0; dim_idx < layerSize; dim_idx++){
		    values.get<0>() += grad_buf[timeStep * layerSize + dim_idx];
		}
		values.get<0>() /= layerSize;
	    }
        }
    };

    struct mse_cos_grad_merge
    {
	int     layerSize;
	bool    pearsoncorr;
	real_t *mean_grad;
	
        const char *patTypes;

        __host__ __device__ void operator() (const thrust::tuple<real_t&, real_t&, int> &values) const
        {
	    int timeStep = values.get<2>() / layerSize;
	    
            if (patTypes[timeStep] == PATTYPE_NONE){
		values.get<0>() = 0;
	    }else{
		if (pearsoncorr)
		    values.get<0>() = values.get<0>() + values.get<1>() - mean_grad[timeStep];
		else
		    values.get<0>() = values.get<0>() + values.get<1>();
	    }
        }
    };
    
} // anonymous namespace
} // namespace anonymous


namespace layers {

    template <typename TDevice>
    SseCosPostOutputLayer<TDevice>::SseCosPostOutputLayer(const helpers::JsonValue &layerChild,
							  Layer<TDevice> &precedingLayer,
							  int             maxSeqLength,
							  int             layerID)
        : SsePostOutputLayer<TDevice>(layerChild, precedingLayer, maxSeqLength, layerID)
    {

	/* ---- initialize the parameter ----- */
	m_cos_weight      = ((layerChild->HasMember("cos_weight")) ? 
			     static_cast<real_t>((*layerChild)["cos_weight"].GetDouble()) : 1.0);

	m_pearsoncorr     = ((layerChild->HasMember("cos_mean_norm")) ? 
			     (*layerChild)["cos_mean_norm"].GetBool() : true);

	/* ---- allocate memory space ----- */
	// memory for residuals
	m_residual = this->precedingLayer().outputs();

	// memory for vector norm, including a^Ta, b^Tb, a^Tb
	cpu_real_vector tmp(this->parallelSequences() * maxSeqLength * 4, 0.0);
	m_vector_norm = tmp;

	if (m_pearsoncorr)
	    printf("\n\tcos distance with mean normalized (pearson correlation)");
	
	// Done
    }

    template <typename TDevice>
    SseCosPostOutputLayer<TDevice>::~SseCosPostOutputLayer()
    {
    }

    template <typename TDevice>
    const std::string& SseCosPostOutputLayer<TDevice>::type() const
    {
        static const std::string s("sse_cos");
        return s;
    }

    template <typename TDevice>
    real_t SseCosPostOutputLayer<TDevice>::calculateError()
    {
	int seq_length = this->curMaxSeqLength() * this->parallelSequences();
	int data_num   = seq_length * this->size();

	// step1. computesseFn
	real_t mse = SsePostOutputLayer<TDevice>::calculateError();

	
	// step2. Cos distance
	real_t cos_dis = 0.0;
	{
	    // step 2.1, calculate residual a = generated - target
	    thrust::transform(this->_actualOutputs().begin(),
			      this->_actualOutputs().begin() + data_num,
			      this->_targets().begin(),
			      this->m_residual.begin(),
			      thrust::minus<real_t>());
	    
	    // step 2.2, let a = generated - target, b = target,
	    //    calculate vector norm a^Tb, a^Ta, b^Tb, and (a^Tb)^2 / (a^Ta * b^Tb)
	    {
		internal::calculateVectorNormAndCosSquare fn;
		fn.layerSize  = this->size();
		fn.pearsoncorr= m_pearsoncorr;
		fn.patTypes   = helpers::getRawPointer(this->patTypes());
		fn.a_buf      = helpers::getRawPointer(this->m_residual);
		fn.b_buf      = helpers::getRawPointer(this->_targets());
		fn.output_buf = helpers::getRawPointer(this->m_vector_norm);

		thrust::for_each(
                 thrust::make_zip_iterator(
		  thrust::make_tuple(this->m_vector_norm.begin(),
				     thrust::counting_iterator<int>(0))),
		 thrust::make_zip_iterator(
		  thrust::make_tuple(this->m_vector_norm.begin()       + seq_length,
				     thrust::counting_iterator<int>(0) + seq_length)),
		 fn);
	    }

	    // step 2.3, sum the cos^2 distance
	    {
		internal::calculateCosSquareDistance fn;
		fn.patTypes   = helpers::getRawPointer(this->patTypes());
		fn.output_buf = helpers::getRawPointer(this->m_vector_norm);
		
		cos_dis = thrust::transform_reduce(
                           thrust::make_zip_iterator(
				thrust::make_tuple(this->m_vector_norm.begin(),
						   thrust::counting_iterator<int>(0))),
			   thrust::make_zip_iterator(
		                thrust::make_tuple(this->m_vector_norm.begin()       + seq_length,
						   thrust::counting_iterator<int>(0) + seq_length)),
			   fn, (real_t)0.0, thrust::plus<real_t>());

	    }
	}
	// just print and return the results
	if (Configuration::instance().verboseLevel() == OP_VERBOSE_LEVEL_1)
	    std::cerr << mse << ", " << cos_dis << ", ";
	
	
	return mse + m_cos_weight * cos_dis;
    }

    template <typename TDevice>
    void SseCosPostOutputLayer<TDevice>::computeForwardPass(const int nnState)
    {
    }

    template <typename TDevice>
    void SseCosPostOutputLayer<TDevice>::computeForwardPass(const int timeStep,
							    const int nnState)
    {
    }

    template <typename TDevice>
    void SseCosPostOutputLayer<TDevice>::computeBackwardPass(const int nnState)
    {
	// step1. gradients w.r.t MSE
	SsePostOutputLayer<TDevice>::computeBackwardPass(nnState);

	// step2. gradients w.r.t cos distance

	
	// step 2.1,
	//   for common cos distance,
	//       a = generated - target, b = target
	//   for pearson correlation,
	//       \hat{a} = generated - target, \hat{b} = target
	//       a = \hat{a} - mean(\hat{a}), b = \hat{b} - mean(\hat{b})
	//   
	//    calculate \partial cos_E / \partial a_k for each dimensio
	//    \partial cos_E / \partial a_k = cos^2(a,b) [b_k /<a,b> - a_k /<a,a>]
	{
	    internal::calculateCosSquareGradient fn;
	    fn.layerSize  = this->size();
	    fn.cos_weight = this->m_cos_weight;
	    fn.output_buf = helpers::getRawPointer(this->m_vector_norm);
	    fn.patTypes   = helpers::getRawPointer(this->patTypes());

	    int n = this->curMaxSeqLength() * this->parallelSequences() * this->size();
	    
	    // note: we save cos_dis gradients to this->outputErrors()
	    thrust::for_each(
               thrust::make_zip_iterator(
		thrust::make_tuple(this->outputErrors().begin(),
				   this->m_residual.begin(),
				   this->_targets().begin(),
				   thrust::counting_iterator<int>(0))),
	       thrust::make_zip_iterator(
		thrust::make_tuple(this->outputErrors().begin()     + n,
				   this->m_residual.begin()          + n,
				   this->_targets().begin()          + n,
				   thrust::counting_iterator<int>(0) + n)),
	       fn);
	}
	
	// step2.2 if pearson correlation is used, calculate 
	//      \partial cos_E / \partial \hat{a}_k
	//           = sum_i \partial cos_E / \partial {a}_i * \partial {a}_i / \partial \hat{a}_k
	//           = \partial cos_E / \partial {a}_k - mean(\sum_i \partial cos_E / \partial {a}_i)
	if (m_pearsoncorr)
	{
	    // get the mean(\sum_i \partial cos_E / \partial {a}_i)
	    internal::calculateErrorMean fn1;
	    fn1.layerSize  = this->size();
	    fn1.grad_buf   = helpers::getRawPointer(this->outputErrors());
	    fn1.patTypes   = helpers::getRawPointer(this->patTypes());

	    int n = this->curMaxSeqLength() * this->parallelSequences();

	    thrust::for_each(
               thrust::make_zip_iterator(
		thrust::make_tuple(this->m_vector_norm.begin(),
				   thrust::counting_iterator<int>(0))),
	       thrust::make_zip_iterator(
		thrust::make_tuple(this->m_vector_norm.begin()       + n,
				   thrust::counting_iterator<int>(0) + n)),
	       fn1);   

	}

	// step 3. merge the cos_gradient with the MSE gradient
	// subtract the mean if pearsoncorr is used
	{
	    
	    internal::mse_cos_grad_merge fn2;
	    fn2.layerSize   = this->size();
	    fn2.pearsoncorr = m_pearsoncorr;
	    fn2.mean_grad   = helpers::getRawPointer(this->m_vector_norm);
	    fn2.patTypes    = helpers::getRawPointer(this->patTypes());

	    int n = this->curMaxSeqLength() * this->parallelSequences() * this->size();

	    thrust::for_each(
               thrust::make_zip_iterator(
		thrust::make_tuple(this->_outputErrors().begin(),
				   this->outputErrors().begin(),
				   thrust::counting_iterator<int>(0))),
	       thrust::make_zip_iterator(
		thrust::make_tuple(this->_outputErrors().begin()     + n,
				   this->outputErrors().begin()      + n,
				   thrust::counting_iterator<int>(0) + n)),
	       fn2);   
	}
    }


    template <typename TDevice>
    void SseCosPostOutputLayer<TDevice>::computeBackwardPass(const int timeStep, const int nnState)
    {
	if (timeStep == this->curMaxSeqLength()-1)
	    this->computeBackwardPass(nnState);
    }
    
    template <typename TDevice>
    void SseCosPostOutputLayer<TDevice>::exportLayer(const helpers::JsonValue     &layersArray, 
						     const helpers::JsonAllocator &allocator) const
    {
        SsePostOutputLayer<TDevice>::exportLayer(layersArray, allocator);
	(*layersArray)[layersArray->Size() - 1].AddMember("cos_weight", m_cos_weight, allocator);
	if (m_pearsoncorr)
	    (*layersArray)[layersArray->Size() - 1].AddMember("cos_mean_norm", m_pearsoncorr,
							      allocator);
    }
    
    // explicit template instantiations
    template class SseCosPostOutputLayer<Cpu>;
    template class SseCosPostOutputLayer<Gpu>;

} // namespace layers
