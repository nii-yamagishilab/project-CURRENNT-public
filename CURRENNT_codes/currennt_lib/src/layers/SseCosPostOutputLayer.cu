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
#include "../helpers/misFuncs.hpp"

#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/functional.h>


#define TEMP_VECTOR_NORM_DIM 6

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
	    
	    real_t ab_mean_normed = 0.0, aa_mean_normed = 0.0, bb_mean_normed = 0.0;
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

		    // We cannot change the actual output value here
		    // because it will be used by SsePostOutputLayer to calculate grad of MSE
		    
		    //a_buf[data_ptr] = (a_buf[data_ptr] - a_mean);
		    //b_buf[data_ptr] = (b_buf[data_ptr] - b_mean);
		    
		    //ab += a_buf[data_ptr] * b_buf[data_ptr];
		    //aa += a_buf[data_ptr] * a_buf[data_ptr];
		    //bb += b_buf[data_ptr] * b_buf[data_ptr];
		    
		    ab_mean_normed += (a_buf[data_ptr] - a_mean) * (b_buf[data_ptr] - b_mean);
		    aa_mean_normed += (a_buf[data_ptr] - a_mean) * (a_buf[data_ptr] - a_mean);
		    bb_mean_normed += (b_buf[data_ptr] - b_mean) * (b_buf[data_ptr] - b_mean);
		}
	    }
	    // <a_normed.b_normed>
	    output_buf[timeStep * TEMP_VECTOR_NORM_DIM]     = ab_mean_normed;
	    
	    // <a_normed.a_normed>,
	    output_buf[timeStep * TEMP_VECTOR_NORM_DIM + 1] = aa_mean_normed;

	    // <b_normed.b_normed>, 
	    output_buf[timeStep * TEMP_VECTOR_NORM_DIM + 2] = bb_mean_normed;
	    
	    // (<a_normed.b_normed>)^2/(<a_normed.a_normed * b_normed.b_normed>),
	    output_buf[timeStep * TEMP_VECTOR_NORM_DIM + 3] =
		(ab_mean_normed / aa_mean_normed) * (ab_mean_normed / bb_mean_normed);

	    // mean(a)
	    output_buf[timeStep * TEMP_VECTOR_NORM_DIM + 4] = a_mean;
	    
	    // mean(b)
	    output_buf[timeStep * TEMP_VECTOR_NORM_DIM + 5] = b_mean;
	    
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
		// (<a_normed.b_normed>)^2/(<a_normed.a_normed * b_normed.b_normed>),
		return output_buf[timeStep * TEMP_VECTOR_NORM_DIM + 3];
	    }
        }
    };

    /*
    struct calculateCosSquareGradient
    {
	int layerSize;
	bool corr_gen_residual;
	real_t *output_buf;
	real_t cos_weight;
        const char *patTypes;

        __host__ __device__ void operator() (const thrust::tuple<real_t&, real_t&,
					     real_t&, real_t&, int> &values) const
        {
	    // t<0>, grad
	    // t<1>, a = generated - target, not mean-normalized
	    // t<2>, b = target, not mean-normalized
	    // t<3>, b = generated, not mean-normalized
	    // t<4>, index
	    
	    int timeStep = values.get<4>() / layerSize;
	    
            if (patTypes[timeStep] == PATTYPE_NONE){
		values.get<0>() = 0;
	    }else{
		
		if (corr_gen_residual){
		    
		    // gradient = weight * pearson(a,b)^2 *
		    //             [(a_normed + b_normed) / <a_normed.b_normed> -
		    //              a_normed / <a_normed.a_normed> - b_normed / <b_normed.b_normed>]

		    values.get<0>() =
			cos_weight * output_buf[timeStep * TEMP_VECTOR_NORM_DIM + 3] *
			((values.get<1>() - output_buf[timeStep * TEMP_VECTOR_NORM_DIM + 4] +
			  values.get<3>() - output_buf[timeStep * TEMP_VECTOR_NORM_DIM + 5]) /
			 output_buf[timeStep * TEMP_VECTOR_NORM_DIM + 0] -
			 (values.get<1>() - output_buf[timeStep * TEMP_VECTOR_NORM_DIM + 4]) /
			 output_buf[timeStep * TEMP_VECTOR_NORM_DIM + 1] -
			 (values.get<3>() - output_buf[timeStep * TEMP_VECTOR_NORM_DIM + 5]) /
			 output_buf[timeStep * TEMP_VECTOR_NORM_DIM + 2]);
		}else{
		    
		    // gradient = weight * pearson(a,b)^2 *
		    //             [b_normed / <a_normed.b_normed> - a_normed / <a_normed.a_normed>]
		    values.get<0>() =
			cos_weight * output_buf[timeStep * TEMP_VECTOR_NORM_DIM + 3] *
			((values.get<2>() - output_buf[timeStep * TEMP_VECTOR_NORM_DIM + 5]) /
			 output_buf[timeStep * TEMP_VECTOR_NORM_DIM + 0] -
			 (values.get<1>() - output_buf[timeStep * TEMP_VECTOR_NORM_DIM + 4]) /
			 output_buf[timeStep * TEMP_VECTOR_NORM_DIM + 1]);
		}
	    }
        }
    };
    */

    struct calculateCosSquareGradient
    {
	int         layerSize;
	real_t     *output_buf;
	real_t      cos_weight;
	bool        grad_to_a;
        const char *patTypes;

        __host__ __device__ void operator() (const thrust::tuple<real_t&, real_t&,
					     real_t&, int> &values) const
        {
	    // t<0>, <a,b> grad w.r.t a
	    // t<1>, a = res
	    // t<2>, b = someinput
	    // t<3>, index
	    
	    int timeStep = values.get<3>() / layerSize;
	    
            if (patTypes[timeStep] == PATTYPE_NONE){
		values.get<0>() = 0;
	    }else{
		if (grad_to_a){
		    // grad_to_a = weight * pearson(a,b)^2 *
		    //             [b_normed / <a_normed.b_normed> - a_normed / <a_normed.a_normed>]
		    values.get<0>() =
			cos_weight * output_buf[timeStep * TEMP_VECTOR_NORM_DIM + 3] *
			((values.get<2>() - output_buf[timeStep * TEMP_VECTOR_NORM_DIM + 5]) /
			 output_buf[timeStep * TEMP_VECTOR_NORM_DIM + 0] -
			 (values.get<1>() - output_buf[timeStep * TEMP_VECTOR_NORM_DIM + 4]) /
			 output_buf[timeStep * TEMP_VECTOR_NORM_DIM + 1]);
		}else{
		    // grad_to_b = weight * pearson(a,b)^2 *
		    //             [a_normed / <a_normed.b_normed> - b_normed / <b_normed.b_normed>]
		    values.get<0>() =
			cos_weight * output_buf[timeStep * TEMP_VECTOR_NORM_DIM + 3] *
			((values.get<1>() - output_buf[timeStep * TEMP_VECTOR_NORM_DIM + 4]) /
			 output_buf[timeStep * TEMP_VECTOR_NORM_DIM + 0] -
			 (values.get<2>() - output_buf[timeStep * TEMP_VECTOR_NORM_DIM + 5]) /
			 output_buf[timeStep * TEMP_VECTOR_NORM_DIM + 1]);
		}
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
		// calculate the gradient mean (across dimensions)
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
		// Merge the pearson gradient with MSE gradient
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

	m_corr_gen_residual = ((layerChild->HasMember("cos_gen_residual")) ? 
			       (*layerChild)["cos_gen_residual"].GetBool() : true);
	
	m_sourceLayersStr = (layerChild->HasMember("cos_source_layer") ? 
			     ((*layerChild)["cos_source_layer"].GetString()) : "");
	m_sourceLayers.clear();
	m_sourceLayersNames.clear();	
	misFuncs::ParseStrOpt(m_sourceLayersStr, m_sourceLayersNames, ",");
	
	/* ---- allocate memory space ----- */
	// memory for residuals
	m_residual = this->precedingLayer().outputs();
	// memory to save b data
	m_b_data   = this->precedingLayer().outputs();
	
	// memory for vector norm, including a^Ta, b^Tb, a^Tb
	cpu_real_vector tmp(this->parallelSequences() * maxSeqLength * TEMP_VECTOR_NORM_DIM, 0.0);
	m_vector_norm = tmp;

	// memory for mean gradient
	cpu_real_vector tmp2(this->parallelSequences() * maxSeqLength, 0.0);
	m_grad_mean   = tmp2;
	
	if (m_pearsoncorr)
	    printf("\n\tcos distance with mean normalized (pearson correlation)");

	if (m_corr_gen_residual)
	    printf("\n\tcos distance between generated data and residual");
	else
	    printf("\n\tcos distance between natural data and residual");
	
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

	// step0. which data is used as b in <a,b>
	if (m_corr_gen_residual){
	    if (m_sourceLayers.size() == 0){
		// b is from the final output layer
		thrust::copy(this->_actualOutputs().begin(), this->_actualOutputs().end(),
			     this->m_b_data.begin());
	    }else{
		// b is from another hidden layer
		thrust::copy(m_sourceLayers[0]->outputs().begin(), m_sourceLayers[0]->outputs().end(),
			     this->m_b_data.begin());
	    }
	}else{
	    // b is from the actual target
	    thrust::copy(this->_targets().begin(), this->_targets().end(),
			 this->m_b_data.begin());
	}
		
	
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
	    
	    // step 2.2,
	    //   when m_corr_gen_residual == true,  a = generated - target, b = generated,
	    //   when m_corr_gen_residual == false, a = generated - target, b = target,
	    
	    //   then, calculate <a_normed.b_normed>, <a_normed.a_normed>,
	    //                   <b_normed.b_normed>,
	    //                   <a_normed.b_normed>^2/<a_normed.a_normed><b_normed.b_normed>
	    {
		internal::calculateVectorNormAndCosSquare fn;
		fn.layerSize  = this->size();
		fn.pearsoncorr= m_pearsoncorr;
		fn.patTypes   = helpers::getRawPointer(this->patTypes());
		fn.a_buf      = helpers::getRawPointer(this->m_residual);
		fn.b_buf      = helpers::getRawPointer(this->m_b_data);
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

	    // step 2.3, sum <a_normed.b_normed>^2/<a_normed.a_normed><b_normed.b_normed> across time
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
	
	// print and return the results
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

	int uttLength = this->curMaxSeqLength() * this->parallelSequences();
	int nElement  = uttLength * this->size();
	
	// step2. gradients w.r.t cos distance
	//   a = generated - target, b = some_input
	// 
	// step2.1, gradients w.r.t a
	{
	    {
	    internal::calculateCosSquareGradient fn;
	    fn.layerSize  = this->size();
	    fn.cos_weight = this->m_cos_weight;
	    fn.grad_to_a  = true;
	    
	    fn.output_buf = helpers::getRawPointer(this->m_vector_norm);
	    fn.patTypes   = helpers::getRawPointer(this->patTypes());

	    
	    // note: we save cos_dis gradients to this->outputErrors()
	    thrust::for_each(
               thrust::make_zip_iterator(
		thrust::make_tuple(this->outputErrors().begin(),
				   this->m_residual.begin(),
				   this->m_b_data.begin(),
				   thrust::counting_iterator<int>(0))),
	       thrust::make_zip_iterator(
		thrust::make_tuple(this->outputErrors().begin()      + nElement,
				   this->m_residual.begin()          + nElement,
				   this->m_b_data.begin()            + nElement,
				   thrust::counting_iterator<int>(0) + nElement)),
	       fn);
	    }
	
	    if (m_pearsoncorr){
	    // get the mean_over_i(\sum_i \partial cos_E / \partial {a}_i)
	    // or      mean_over_i([\partial cos_E/\partial {a}_i + \partial cos_E/\partial {b}_i])
	    internal::calculateErrorMean fn1;
	    fn1.layerSize  = this->size();
	    fn1.grad_buf   = helpers::getRawPointer(this->outputErrors());
	    fn1.patTypes   = helpers::getRawPointer(this->patTypes());

	    thrust::for_each(
               thrust::make_zip_iterator(
		thrust::make_tuple(this->m_grad_mean.begin(),
				   thrust::counting_iterator<int>(0))),
	       thrust::make_zip_iterator(
		thrust::make_tuple(this->m_grad_mean.begin()         + uttLength,
				   thrust::counting_iterator<int>(0) + uttLength)),
	       fn1);   
	    }

	    // Merge the cos_gradient with the MSE gradient
	    // subtract the mean if pearsoncorr is used
	    {
	    
	    internal::mse_cos_grad_merge fn2;
	    fn2.layerSize   = this->size();
	    fn2.pearsoncorr = m_pearsoncorr;
	    fn2.mean_grad   = helpers::getRawPointer(this->m_grad_mean);
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

	
	// step2.2, gradients w.r.t b is necessary
	if (m_corr_gen_residual){
	    {
	    internal::calculateCosSquareGradient fn;
	    fn.layerSize  = this->size();
	    fn.cos_weight = this->m_cos_weight;
	    fn.grad_to_a  = false;
	    
	    fn.output_buf = helpers::getRawPointer(this->m_vector_norm);
	    fn.patTypes   = helpers::getRawPointer(this->patTypes());

	    
	    // note: we save cos_dis gradients to this->outputErrors()
	    thrust::for_each(
               thrust::make_zip_iterator(
		thrust::make_tuple(this->outputErrors().begin(),
				   this->m_residual.begin(),
				   this->m_b_data.begin(),
				   thrust::counting_iterator<int>(0))),
	       thrust::make_zip_iterator(
		thrust::make_tuple(this->outputErrors().begin()      + nElement,
				   this->m_residual.begin()          + nElement,
				   this->m_b_data.begin()            + nElement,
				   thrust::counting_iterator<int>(0) + nElement)),
	       fn);
	    }
	
	    if (m_pearsoncorr){
	    internal::calculateErrorMean fn1;
	    fn1.layerSize  = this->size();
	    fn1.grad_buf   = helpers::getRawPointer(this->outputErrors());
	    fn1.patTypes   = helpers::getRawPointer(this->patTypes());

	    thrust::for_each(
               thrust::make_zip_iterator(
		thrust::make_tuple(this->m_grad_mean.begin(),
				   thrust::counting_iterator<int>(0))),
	       thrust::make_zip_iterator(
		thrust::make_tuple(this->m_grad_mean.begin()         + uttLength,
				   thrust::counting_iterator<int>(0) + uttLength)),
	       fn1);   
	    }

	    // Merge the cos_gradient with the MSE gradient
	    // subtract the mean if pearsoncorr is used
	    {	    
		internal::mse_cos_grad_merge fn2;
		fn2.layerSize   = this->size();
		fn2.pearsoncorr = m_pearsoncorr;
		fn2.mean_grad   = helpers::getRawPointer(this->m_grad_mean);
		fn2.patTypes    = helpers::getRawPointer(this->patTypes());
		if (m_sourceLayers.size() == 0){
		    thrust::for_each(
                      thrust::make_zip_iterator(
		        thrust::make_tuple(this->_outputErrors().begin(),
					   this->outputErrors().begin(),
					   thrust::counting_iterator<int>(0))),
		      thrust::make_zip_iterator(
			thrust::make_tuple(this->_outputErrors().begin()     + nElement,
					   this->outputErrors().begin()      + nElement,
					   thrust::counting_iterator<int>(0) + nElement)),
		      fn2);   
		}else{
		    thrust::for_each(
                      thrust::make_zip_iterator(
		        thrust::make_tuple(m_sourceLayers[0]->outputErrors().begin(),
					   this->outputErrors().begin(),
					   thrust::counting_iterator<int>(0))),
		      thrust::make_zip_iterator(
			thrust::make_tuple(m_sourceLayers[0]->outputErrors().begin() + nElement,
					   this->outputErrors().begin()            + nElement,
					   thrust::counting_iterator<int>(0)       + nElement)),
		      fn2);   
		    
		}
	    }

	}

	
	// step2. gradients w.r.t cos distance
	/*
	// step 2.1,
	{
	    //   if m_corr_gen_residual == true
	    //       a = generated - target, b = generated
	    //   for pearson correlation,
	    //       \hat{a} = generated - target, \hat{b} = generated
	    //       a = \hat{a} - mean(\hat{a}), b = \hat{b} - mean(\hat{b})
	    //   
	    //    calculate \partial cos_E / \partial a_k for each dimension
	    //    \partial cos_E / \partial a_k + \partial cos_E / \partial b_k
	    //            = cos^2(a,b) [(b_k + a_k)/<a,b> - a_k /<a,a> - b_k / <b,b>]
	    
	    
	    //   if m_corr_gen_residual == false
	    //       a = generated - target, b = target
	    //   for pearson correlation,
	    //       \hat{a} = generated - target, \hat{b} = target
	    //       a = \hat{a} - mean(\hat{a}), b = \hat{b} - mean(\hat{b})
	    //   
	    //    calculate \partial cos_E / \partial a_k for each dimensio
	    //    \partial cos_E / \partial a_k = cos^2(a,b) [b_k /<a,b> - a_k /<a,a>]

	    internal::calculateCosSquareGradient fn;
	    fn.layerSize  = this->size();
	    fn.cos_weight = this->m_cos_weight;
	    fn.corr_gen_residual = this->m_corr_gen_residual;
	    
	    fn.output_buf = helpers::getRawPointer(this->m_vector_norm);
	    fn.patTypes   = helpers::getRawPointer(this->patTypes());

	    int n = this->curMaxSeqLength() * this->parallelSequences() * this->size();
	    
	    // note: we save cos_dis gradients to this->outputErrors()
	    thrust::for_each(
               thrust::make_zip_iterator(
		thrust::make_tuple(this->outputErrors().begin(),
				   this->m_residual.begin(),
				   this->_targets().begin(),
				   this->_actualOutputs().begin(),
				   thrust::counting_iterator<int>(0))),
	       thrust::make_zip_iterator(
		thrust::make_tuple(this->outputErrors().begin()      + n,
				   this->m_residual.begin()          + n,
				   this->_targets().begin()          + n,
				   this->_actualOutputs().begin()    + n,
				   thrust::counting_iterator<int>(0) + n)),
	       fn);
	}
	*/
	
	// step2.2 if pearson correlation is used, calculate
	///  if m_corr_gen_residual == false
	//      \partial cos_E / \partial generated_mel_k
	//           = \partial cos_E / \partial \hat{a}_k
	//           = sum_i \partial cos_E / \partial {a}_i * \partial {a}_i / \partial \hat{a}_k
	//           = \partial cos_E / \partial {a}_k - mean_over_i(\partial cos_E / \partial {a}_i)
	//
	//   if m_corr_gen_residual == true
	//      \partial cos_E / \partial generated_mel_k
	//           = \partial cos_E / \partial \hat{a}_k + \partial cos_E / \partial \hat{b}_k
	//           = sum_i \partial cos_E / \partial {a}_i * \partial {a}_i / \partial \hat{a}_k
	//                  + \partial cos_E / \partial {b}_i * \partial {b}_i / \partial \hat{b}_k
	//           = sum_i [\partial cos_E / \partial {a}_i + \partial cos_E / \partial {b}_i] *
	//                \partial {a}_i / \partial \hat{a}_k
	//       note that \partial {a}_i / \partial \hat{a}_k = \partial {b}_i / \partial \hat{b}_k
	//           = [\partial cos_E / \partial {a}_k + \partial cos_E / \partial {b}_k]
	//             - mean_over_i([\partial cos_E/\partial {a}_i + \partial cos_E/\partial {b}_i])
	
	
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
	if (!m_pearsoncorr)
	    (*layersArray)[layersArray->Size() - 1].AddMember("cos_mean_norm", m_pearsoncorr,
							      allocator);
	if (!m_corr_gen_residual)
	    (*layersArray)[layersArray->Size() - 1].AddMember("cos_gen_residual", m_pearsoncorr,
							      allocator);

	if (m_sourceLayersStr.size())
	    (*layersArray)[layersArray->Size() - 1].AddMember("cos_source_layer",
							      m_sourceLayersStr.c_str(),
							      allocator);
    }

    template <typename TDevice>
    void SseCosPostOutputLayer<TDevice>::linkTargetLayer(Layer<TDevice> &targetLayer)
    {
	
	for (int i = 0; i<m_sourceLayersNames.size(); i++){
	    if (targetLayer.name() == m_sourceLayersNames[i] &&
		targetLayer.size() == this->size()){
		m_sourceLayers.push_back(&targetLayer);
		printf("\nLink for SseCosPostOutputLayer: %s", targetLayer.name().c_str());
		break;
	    }
	}
    }

    // explicit template instantiations
    template class SseCosPostOutputLayer<Cpu>;
    template class SseCosPostOutputLayer<Gpu>;

} // namespace layers
