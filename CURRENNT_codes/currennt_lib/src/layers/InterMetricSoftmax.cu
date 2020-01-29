/******************************************************************************
 * This file is an addtional component of CURRENNT. 
 * Copyright (c) 2019 Xin WANG
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

#include "InterMetricSoftmax.hpp"
#include "../helpers/getRawPointer.cuh"
#include "../helpers/min.cuh"
#include "../helpers/max.cuh"
#include "../helpers/safeExp.cuh"
#include "../helpers/JsonClasses.hpp"

#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>

#define INTERMETRIC_SOFTMAX_MODE1 1  // [target_indx, probability_vector]
#define INTERMETRIC_SOFTMAX_MODE2 2  // [target_one_hot, probability_vector]

namespace internal {
namespace {
    
    struct ComputeInterSoftmaxFn_mode1
    {
        int featDim;
	real_t *dataBuf;
	
        const char *patTypes;

        __host__ __device__ real_t operator() (const thrust::tuple<const real_t, int> &values) const
        {
            int patIdx = values.get<1>() / featDim;
	    int dimIdx = values.get<1>() % featDim;
	    
            if (patTypes[patIdx] == PATTYPE_NONE)
                return 0;

	    int targetIdx = (int)dataBuf[patIdx * (featDim + 1)];
	    dataBuf[patIdx * (featDim + 1)] = targetIdx;  // round the index if necessary
	    
	    if (targetIdx == dimIdx){
		real_t targetProb = helpers::max(helpers::NumericLimits<real_t>::min(),
						 dataBuf[patIdx * (featDim + 1) + targetIdx + 1]);
		return -1.0 * log(targetProb);
	    }else{
		return 0.0;
	    }
	}
    };

    struct ComputeInterSoftmaxFn_mode2
    {
        int featDim;
	real_t *dataBuf;
	
        const char *patTypes;

        __host__ __device__ real_t operator() (const thrust::tuple<const real_t, int> &values) const
        {
            int patIdx = values.get<1>() / featDim;
	    int dimIdx = values.get<1>() % featDim;
	    
            if (patTypes[patIdx] == PATTYPE_NONE)
                return 0;

	    real_t targetProb = helpers::max(helpers::NumericLimits<real_t>::min(),
					     dataBuf[patIdx * (featDim * 2) + dimIdx + featDim]);
	    return -1.0 * log(targetProb) * dataBuf[patIdx * (featDim * 2) + dimIdx];
	}
    };

    struct ComputeOutputInterSoftmaxFn_mode1
    {        
    	int featDim;
	real_t *dataBuf;
    	const char *patTypes;
	
    	__host__ __device__ real_t operator() (const thrust::tuple<const real_t&, int> &values) const
    	{
	    int layerSize = featDim + 1;
	    int patIdx = values.get<1>() / layerSize;
	    int dimIdx = values.get<1>() % layerSize;
	    
	    // check if the pattern is a dummy
	    if (patTypes[patIdx] == PATTYPE_NONE)
		return 0;
	    
	    if (dimIdx == 0){
		// index
		return 0.0;
	    }else{
		if (((dimIdx - 1) - dataBuf[patIdx * layerSize]) *
		    ((dimIdx - 1) - dataBuf[patIdx * layerSize]) < 0.0001){
		    // hit
		    return -1.0 / dataBuf[patIdx * layerSize + dimIdx];
		}else{
		    // otherwise
		    return 0.0;
		}
	    }
		    
	}
    };

    struct ComputeOutputInterSoftmaxFn_mode2
    {        
    	int featDim;
	real_t *dataBuf;
    	const char *patTypes;
	
    	__host__ __device__ real_t operator() (const thrust::tuple<const real_t&, int> &values) const
    	{
	    int layerSize = featDim * 2;
	    int patIdx = values.get<1>() / layerSize;
	    int dimIdx = values.get<1>() % layerSize;
	    
	    // check if the pattern is a dummy
	    if (patTypes[patIdx] == PATTYPE_NONE)
		return 0;
	    
	    if (dimIdx < featDim){
		// one-hot
		return 0.0;
	    }else{
		// probablity
		if (dataBuf[patIdx * layerSize + dimIdx - featDim] > 0){
		    // if current dimension is hit
		    return -1.0 / dataBuf[patIdx * layerSize + dimIdx];
		}else{
		    // otherwise
		    return 0.0;
		}
	    }
		    
	}
    };
    
} // anonymous namespace
} // namespace anonymous


namespace layers {

    template <typename TDevice>
    InterMetricLayer_softmax<TDevice>::InterMetricLayer_softmax(const helpers::JsonValue &layerChild,
						    Layer<TDevice> &precedingLayer,
						    int             maxSeqLength,
						    int             layerID)
        : InterMetricLayer<TDevice>(layerChild, precedingLayer, maxSeqLength, layerID)
    {
	// Assume the input is [target_indx, probability_vector]
	// or [target_one_hot, probability_vector]
	m_mode    = (layerChild->HasMember("softmax_input_mode") ? 
		     (*layerChild)["softmax_input_mode"].GetInt() :
		     INTERMETRIC_SOFTMAX_MODE1);
	
	
    }

    template <typename TDevice>
    InterMetricLayer_softmax<TDevice>::~InterMetricLayer_softmax()
    {
    }

    template <typename TDevice>
    const std::string& InterMetricLayer_softmax<TDevice>::type() const
    {
        static const std::string s("inter_softmax");
        return s;
    }

    template <typename TDevice>
    real_t InterMetricLayer_softmax<TDevice>::calculateError()
    {
	real_t mse = 0.0;
	
	if (m_mode == INTERMETRIC_SOFTMAX_MODE2){
	    internal::ComputeInterSoftmaxFn_mode2 fn;
	    fn.featDim   = this->precedingLayer().size()/2;
	    fn.dataBuf   = helpers::getRawPointer(this->precedingLayer().outputs());
	    fn.patTypes  = helpers::getRawPointer(this->patTypes());

	    int n = this->curMaxSeqLength() * this->parallelSequences() *
		this->precedingLayer().size() / 2;
	    
	    mse = thrust::transform_reduce(
                thrust::make_zip_iterator(
			thrust::make_tuple(this->precedingLayer().outputs().begin(),
					   thrust::counting_iterator<int>(0))),
		thrust::make_zip_iterator(
			thrust::make_tuple(this->precedingLayer().outputs().begin() + n,
					   thrust::counting_iterator<int>(0)        + n)),
		fn,
		(real_t)0,
		thrust::plus<real_t>());
	}else{
	    internal::ComputeInterSoftmaxFn_mode1 fn;
	    
	    fn.featDim   = (this->precedingLayer().size() - 1);
	    fn.dataBuf   = helpers::getRawPointer(this->precedingLayer().outputs());
	    fn.patTypes  = helpers::getRawPointer(this->patTypes());

	    int n = this->curMaxSeqLength() * this->parallelSequences() *
		(this->precedingLayer().size() - 1);
	    
	    mse = thrust::transform_reduce(
                thrust::make_zip_iterator(
			thrust::make_tuple(this->precedingLayer().outputs().begin(),
					   thrust::counting_iterator<int>(0))),
		thrust::make_zip_iterator(
			thrust::make_tuple(this->precedingLayer().outputs().begin() + n,
					   thrust::counting_iterator<int>(0)        + n)),
		fn,
		(real_t)0,
		thrust::plus<real_t>());
	}
	return mse;
	
    }

    template <typename TDevice>
    void InterMetricLayer_softmax<TDevice>::computeForwardPass(const int nnState)
    {
    }

    template <typename TDevice>
    void InterMetricLayer_softmax<TDevice>::computeForwardPass(const int timeStep,
							       const int nnState)
    {
    }

    template <typename TDevice>
    void InterMetricLayer_softmax<TDevice>::computeBackwardPass(const int nnState)
    {

	if (m_mode == INTERMETRIC_SOFTMAX_MODE2){
	    internal::ComputeOutputInterSoftmaxFn_mode2 fn;
	    fn.featDim   = this->precedingLayer().size() / 2;
	    fn.dataBuf   = helpers::getRawPointer(this->precedingLayer().outputs());
	    fn.patTypes  = helpers::getRawPointer(this->patTypes());

	    int n = this->curMaxSeqLength() * this->parallelSequences() *
		this->precedingLayer().size();

	    thrust::transform(
              thrust::make_zip_iterator(
		thrust::make_tuple(this->precedingLayer().outputs().begin(),
				   thrust::counting_iterator<int>(0))),
	      thrust::make_zip_iterator(
		thrust::make_tuple(this->precedingLayer().outputs().begin() + n,
				   thrust::counting_iterator<int>(0)        + n)),
	       this->precedingLayer().outputErrors().begin(),
	      fn);
	    
	}else{
	    
	    internal::ComputeOutputInterSoftmaxFn_mode1 fn;
	    fn.featDim   = this->precedingLayer().size() - 1;
	    fn.dataBuf   = helpers::getRawPointer(this->precedingLayer().outputs());
	    fn.patTypes  = helpers::getRawPointer(this->patTypes());

	    int n = this->curMaxSeqLength() * this->parallelSequences() *
		this->precedingLayer().size();

	    thrust::transform(
              thrust::make_zip_iterator(
		thrust::make_tuple(this->precedingLayer().outputs().begin(),
				   thrust::counting_iterator<int>(0))),
	      thrust::make_zip_iterator(
		thrust::make_tuple(this->precedingLayer().outputs().begin() + n,
				   thrust::counting_iterator<int>(0)        + n)),
	       this->precedingLayer().outputErrors().begin(),
	      fn);

	}
    }


    template <typename TDevice>
    void InterMetricLayer_softmax<TDevice>::computeBackwardPass(const int timeStep,
								const int nnState)
    {
	if (timeStep == this->curMaxSeqLength()-1)
	    this->computeBackwardPass(nnState);
    }

    template <typename TDevice>
    void InterMetricLayer_softmax<TDevice>::exportLayer(const helpers::JsonValue &layersArray, 
						    const helpers::JsonAllocator &allocator) const
    {
        Layer<TDevice>::exportLayer(layersArray, allocator);
	if (m_mode != INTERMETRIC_SOFTMAX_MODE1)
	    (*layersArray)[layersArray->Size() - 1].AddMember("softmax_input_mode",
							      m_mode, allocator);
		
    }

    template <typename TDevice>
    void InterMetricLayer_softmax<TDevice>::clearAllBuffers()
    {
    }

    template <typename TDevice>
    void InterMetricLayer_softmax<TDevice>::resizeAllBuffers(const int timeLength)
    {
    }

    template <typename TDevice>
    void InterMetricLayer_softmax<TDevice>::logAllBuffers(
		helpers::vecPoolManager<TDevice> &vecPoolMng, bool flag_add)
    {
    }

    template <typename TDevice>
    void InterMetricLayer_softmax<TDevice>::swapAllBuffers(
		helpers::vecPoolManager<TDevice> &vecPoolMng, bool flag_get)
    {
    }


    
    // explicit template instantiations
    template class InterMetricLayer_softmax<Cpu>;
    template class InterMetricLayer_softmax<Gpu>;

} // namespace layers
