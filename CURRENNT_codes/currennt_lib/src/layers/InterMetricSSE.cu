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

#include "InterMetricSSE.hpp"
#include "../helpers/getRawPointer.cuh"
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>


namespace internal {
namespace {
    
    struct ComputeInterSseFn
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

            // calculate the error
	    // assume [target, source] in each time frame
            real_t diff = dataBuf[patIdx * featDim * 2 + dimIdx] -
		dataBuf[patIdx * featDim * 2 + dimIdx + featDim];
            return (diff * diff);
        }
    };

    struct ComputeOutputInterErrorFn
    {        
    	int featDim;
	real_t  grad_scale;
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
	    
	    // for target signal, grad = target - source
	    // for source signal, grad = source - target
	    if (dimIdx < featDim)
		return (dataBuf[patIdx * layerSize + dimIdx] -
			dataBuf[patIdx * layerSize + dimIdx + featDim]) * grad_scale;
	    else
		return (dataBuf[patIdx * layerSize + dimIdx] -
			dataBuf[patIdx * layerSize + dimIdx - featDim]) * grad_scale;
		    
	}
    };
    
} // anonymous namespace
} // namespace anonymous


namespace layers {

    template <typename TDevice>
    InterMetricLayer_sse<TDevice>::InterMetricLayer_sse(const helpers::JsonValue &layerChild,
						    Layer<TDevice> &precedingLayer,
						    int             maxSeqLength,
						    int             layerID)
        : InterMetricLayer<TDevice>(layerChild, precedingLayer, maxSeqLength, layerID)
    {
	// Assume one input vector is [target, source]
	// check input dimension
	if (precedingLayer.size() % 2 != 0)
	    throw std::runtime_error("Error: input to inter_sse layer should be [target, source]");
    }

    template <typename TDevice>
    InterMetricLayer_sse<TDevice>::~InterMetricLayer_sse()
    {
    }

    template <typename TDevice>
    const std::string& InterMetricLayer_sse<TDevice>::type() const
    {
        static const std::string s("inter_sse");
        return s;
    }

    template <typename TDevice>
    real_t InterMetricLayer_sse<TDevice>::calculateError()
    {
	int featureDim   = this->precedingLayer().size() / 2;
	
	internal::ComputeInterSseFn fn;
	fn.featDim   = featureDim;
	fn.dataBuf   = helpers::getRawPointer(this->precedingLayer().outputs());
	fn.patTypes  = helpers::getRawPointer(this->patTypes());

	int n = this->curMaxSeqLength() * this->parallelSequences() * featureDim;
	    
	real_t mse = (real_t)0.5 * thrust::transform_reduce(
                thrust::make_zip_iterator(
			thrust::make_tuple(this->precedingLayer().outputs().begin(),
					   thrust::counting_iterator<int>(0))),
		thrust::make_zip_iterator(
			thrust::make_tuple(this->precedingLayer().outputs().begin() + n,
					   thrust::counting_iterator<int>(0)        + n)),
		fn,
		(real_t)0,
		thrust::plus<real_t>()) / featureDim;
	return mse;
	
    }

    template <typename TDevice>
    void InterMetricLayer_sse<TDevice>::computeForwardPass(const int nnState)
    {
    }

    template <typename TDevice>
    void InterMetricLayer_sse<TDevice>::computeForwardPass(const int timeStep,
							   const int nnState)
    {
    }

    template <typename TDevice>
    void InterMetricLayer_sse<TDevice>::computeBackwardPass(const int nnState)
    {
	int featureDim   = this->precedingLayer().size() / 2;

	internal::ComputeOutputInterErrorFn fn;
	fn.featDim   = featureDim;
	fn.dataBuf   = helpers::getRawPointer(this->precedingLayer().outputs());
	fn.patTypes  = helpers::getRawPointer(this->patTypes());
	fn.grad_scale= this->__grad_scale();
	
	int n = this->curMaxSeqLength() * this->parallelSequences() * featureDim * 2;

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


    template <typename TDevice>
    void InterMetricLayer_sse<TDevice>::computeBackwardPass(const int timeStep, const int nnState)
    {
	if (timeStep == this->curMaxSeqLength()-1)
	    this->computeBackwardPass(nnState);
    }

    template <typename TDevice>
    void InterMetricLayer_sse<TDevice>::exportLayer(const helpers::JsonValue &layersArray, 
						    const helpers::JsonAllocator &allocator) const
    {
        InterMetricLayer<TDevice>::exportLayer(layersArray, allocator);
    }

    template <typename TDevice>
    void InterMetricLayer_sse<TDevice>::clearAllBuffers()
    {
    }

    template <typename TDevice>
    void InterMetricLayer_sse<TDevice>::resizeAllBuffers(const int timeLength)
    {
    }

    template <typename TDevice>
    void InterMetricLayer_sse<TDevice>::logAllBuffers(
		helpers::vecPoolManager<TDevice> &vecPoolMng, bool flag_add)
    {
    }

    template <typename TDevice>
    void InterMetricLayer_sse<TDevice>::swapAllBuffers(
		helpers::vecPoolManager<TDevice> &vecPoolMng, bool flag_get)
    {
    }


    
    // explicit template instantiations
    template class InterMetricLayer_sse<Cpu>;
    template class InterMetricLayer_sse<Gpu>;

} // namespace layers
