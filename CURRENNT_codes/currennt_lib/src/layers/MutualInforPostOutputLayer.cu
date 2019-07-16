/******************************************************************************
 * This file is an addtional component of CURRENNT. 
 * Xin WANG
 * National Institute of Informatics, Japan
 * 2019
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

#include "MutualInforPostOutputLayer.hpp"
#include "../helpers/getRawPointer.cuh"
#include "../helpers/min.cuh"
#include "../helpers/max.cuh"
#include "../helpers/JsonClasses.hpp"

#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include "../activation_functions/Logistic.cuh"

#define MULTUALINFORMATION_MODE_BCE 0
#define MULTUALINFORMATION_MODE_DEFAULT 0

namespace internal {
namespace {

    struct CountPositive
    {
        __host__ __device__ real_t operator() (const thrust::tuple<real_t, int> &values) const
        {
	    
            if (values.get<0>() > 0.5)
                return 1;
	    else
		return 0;
        }
    };

    struct CountNegative
    {
        __host__ __device__ real_t operator() (const thrust::tuple<real_t, int> &values) const
        {
            if (values.get<0>() < 0.5 && values.get<0>() > -0.5)
                return 1;
	    else
		return 0;
        }
    };
    
    struct mutualInfor
    {
        real_t cntPositive;
	real_t cntNegative;

	real_t *sigmoid_buffer;
	
        __host__ __device__ real_t operator() (const thrust::tuple<real_t, real_t, int> &values) const
        {
	    real_t tmp_sig = 0.0;
	    
	    // get sigmoid(activation)
	    tmp_sig = activation_functions::Logistic::fn(values.get<0>());
	    
	    // save sigmoid(activation)
	    sigmoid_buffer[values.get<2>()] = tmp_sig;
	    
	    if (values.get<1>() > 0.5){
		// get log(sigmoid(activation))
		tmp_sig = log(helpers::max(helpers::NumericLimits<real_t>::min(), tmp_sig));
		return -1.0 * tmp_sig / cntPositive;
		
	    }else if (values.get<1>() > -0.5 && values.get<1>() < 0.5){
		tmp_sig = log(helpers::max(helpers::NumericLimits<real_t>::min(),
					   ((real_t)1.0 - tmp_sig)));
		return -1.0 * tmp_sig / cntNegative;
		
	    }else{
		return 0;
	    }
        }
    };

    struct mutualInforGrad
    {
	real_t cntPositive;
	real_t cntNegative;
	
    	__host__ __device__ real_t operator() (const thrust::tuple<const real_t&,
					       const real_t&> &values) const        
    	{

	    if (values.get<1>() > 0.5)		
		return (values.get<0>() - 1.0) / cntPositive;
	    else if (values.get<1>() > -0.5 && values.get<1>() < 0.5)
		return (values.get<0>() - 0.0) / cntNegative;
	    else
		return 0;
	}
    };
    
} // anonymous namespace
} // namespace anonymous


namespace layers {

    template <typename TDevice>
    MutualInforPostOutputLayer<TDevice>::MutualInforPostOutputLayer(
					const helpers::JsonValue &layerChild,
					Layer<TDevice> &precedingLayer,
					int             maxSeqLength,
					int             layerID)
        : PostOutputLayer<TDevice>(layerChild, precedingLayer, precedingLayer.size(),
				   maxSeqLength, layerID)
    {
	/* ------- load configuration ------- */
	m_mode = (layerChild->HasMember("mode") ? 
		  (*layerChild)["mode"].GetInt() : MULTUALINFORMATION_MODE_DEFAULT);

	// allocate memory
	m_sigmoid_output = this->outputs();
	
	/* ------- check ------- */
	if (this->size() != 1)
	    throw std::runtime_error("Error: mutual information layer size should be 1");
    }

    template <typename TDevice>
    MutualInforPostOutputLayer<TDevice>::~MutualInforPostOutputLayer()
    {
    }

    template <typename TDevice>
    const std::string& MutualInforPostOutputLayer<TDevice>::type() const
    {
        static const std::string s("mutual_infor");
        return s;
    }

    template <typename TDevice>
    real_t MutualInforPostOutputLayer<TDevice>::calculateError()
    {
	int n = this->curMaxSeqLength() * this->parallelSequences();
	
	// step1. count the number of positive and negative frames
	if (m_mode == MULTUALINFORMATION_MODE_BCE){
	    
	    // - Binary classification entropy (likelihood)
	    //    = -1/T_pos \sum_t=1^T(\delta(t = positive) log(sigmoid(discriminator(x1, x2))) +
	    //      -1/T_neg \sum_t=1^T(\delta(t = negative) log(1 - sigmoid(discriminator(x1, x_3)))
	    //  where x1, x2 are from the same utterance, x_3 from a different utterance
	    
	    {{

	    // note: this->_targets() will get a sequence of 1/0/-1 from InterWeaveLayer
	    // 1: the input to the discriminator is from the same utterance
	    // 0: the input to the discriminator is from different utterances
	    // -1: dummy slot
	    internal::CountPositive fn1;
	    this->m_positive_cnt = thrust::transform_reduce(
                thrust::make_zip_iterator(
			thrust::make_tuple(this->_targets().begin(),
					   thrust::counting_iterator<int>(0))),
		thrust::make_zip_iterator(
			thrust::make_tuple(this->_targets().begin()         +n,
					   thrust::counting_iterator<int>(0)+n)),
		fn1,
		(real_t)0,
		thrust::plus<real_t>());

	    internal::CountNegative fn2;
	    this->m_negative_cnt = thrust::transform_reduce(
                thrust::make_zip_iterator(
			thrust::make_tuple(this->_targets().begin(),
					   thrust::counting_iterator<int>(0))),
		thrust::make_zip_iterator(
			thrust::make_tuple(this->_targets().begin()         +n,
					   thrust::counting_iterator<int>(0)+n)),
		fn2,
		(real_t)0, 
		thrust::plus<real_t>());
	    }}

	    // step2. calcualte errors
	    {{
	    // calcualte mutual information
	    internal::mutualInfor fn;
	    fn.cntPositive = this->m_positive_cnt;
	    fn.cntNegative = this->m_negative_cnt;
	    fn.sigmoid_buffer = helpers::getRawPointer(this->m_sigmoid_output);
	    
	    real_t mi = thrust::transform_reduce(
                thrust::make_zip_iterator(
			thrust::make_tuple(this->_actualOutputs().begin(),
					   this->_targets().begin(),
					   thrust::counting_iterator<int>(0))),
		thrust::make_zip_iterator(
			thrust::make_tuple(this->_actualOutputs().begin() + n,
					   this->_targets().begin()       + n,
					   thrust::counting_iterator<int>(0) + n)),
		fn,
		(real_t)0,
		thrust::plus<real_t>());
	    return mi;
	    }}
	    
	}else{
	    throw std::runtime_error("Error: mutual information unimplemented mode");
	}
	
    }

    template <typename TDevice>
    void MutualInforPostOutputLayer<TDevice>::computeForwardPass(const int nnState)
    {
    }

    template <typename TDevice>
    void MutualInforPostOutputLayer<TDevice>::computeForwardPass(const int timeStep,
								 const int nnState)
    {
    }

    template <typename TDevice>
    void MutualInforPostOutputLayer<TDevice>::computeBackwardPass(const int nnState)
    {
	int n = this->curMaxSeqLength() * this->parallelSequences() * this->size();

	if (m_mode == MULTUALINFORMATION_MODE_BCE){
	    
	    // for time t = 1:T
	    //  if discriminator_t is from (x1, x2),
	    //    grad = 1/positive_cnt * (sigmoid(discriminator_t) - 1)
	    //  else
	    //    grad = 1/negative_cnt * (sigmoid(discriminator_t))
	    
	    {{
	    internal::mutualInforGrad fn;
	    fn.cntPositive = this->m_positive_cnt;
	    fn.cntNegative = this->m_negative_cnt;
	    
	    thrust::transform(
	       thrust::make_zip_iterator(thrust::make_tuple(this->m_sigmoid_output.begin(),
							    this->_targets().begin())),
	       thrust::make_zip_iterator(thrust::make_tuple(this->m_sigmoid_output.begin()+n,
							    this->_targets().begin()      +n)),
	       this->_outputErrors().begin(),
	       fn);
	    }}
	    
	}else{
	    throw std::runtime_error("Error: mutual information unimplemented mode");
	}
    }


    template <typename TDevice>
    void MutualInforPostOutputLayer<TDevice>::exportLayer(
					const helpers::JsonValue &layersArray,
					const helpers::JsonAllocator &allocator) const
    {
	PostOutputLayer<TDevice>::exportLayer(layersArray, allocator);
	(*layersArray)[layersArray->Size() - 1].AddMember("mode", m_mode, allocator);
    }


    template <typename TDevice>
    void MutualInforPostOutputLayer<TDevice>::computeBackwardPass(const int timeStep,
								  const int nnState)
    {
	if (timeStep == this->curMaxSeqLength()-1)
	    this->computeBackwardPass(nnState);
    }

    // explicit template instantiations
    template class MutualInforPostOutputLayer<Cpu>;
    template class MutualInforPostOutputLayer<Gpu>;

} // namespace layers
