/******************************************************************************
 * Copyright (c) 2013 Johannes Bergmann, Felix Weninger, Bjoern Schuller
 * Institute for Human-Machine Communication
 * Technische Universitaet Muenchen (TUM)
 * D-80290 Munich, Germany
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
#   pragma warning (disable: 4244) 
#endif

#include "SsePostOutputLayerForFeatTransform.hpp"
#include "../helpers/getRawPointer.cuh"

#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>


namespace internal {
namespace {
    
    struct ComputeSseFn
    {
        int layerSize;

        const char *patTypes;

        __host__ __device__ real_t operator() (const thrust::tuple<real_t, real_t, int> &values) const
        {
            // unpack the tuple
            real_t target = values.get<0>();
            real_t output = values.get<1>();
            int outputIdx = values.get<2>();

            // check if we have to skip this value
            int patIdx = outputIdx / layerSize;
            if (patTypes[patIdx] == PATTYPE_NONE)
                return 0;

            // calculate the error
            real_t diff = target - output;
            return (diff * diff);
        }
    };

    struct ComputeOutputErrorFn
    {        
    	int layerSize;
    	const char *patTypes;
    	__host__ __device__ real_t operator() (const thrust::tuple<const real_t&, const real_t&, int> &t) const        
    	{
	    // unpack the tuple
	    real_t actualOutput = t.get<0>();
	    real_t targetOutput = t.get<1>();
	    int    outputIdx    = t.get<2>();
	    
	    // calculate the pattern index
	    int patIdx = outputIdx / layerSize;
	    
	    // check if the pattern is a dummy
	    if (patTypes[patIdx] == PATTYPE_NONE)
		return 0;
	    
	    // calculate the error
	    real_t error = actualOutput - targetOutput;
	    return error;
	}
    };
    
} // anonymous namespace
} // namespace anonymous


namespace layers {

    template <typename TDevice>
    SsePostOutputLayerForFeatTrans<TDevice>::SsePostOutputLayerForFeatTrans(
					const helpers::JsonValue &layerChild,
					Layer<TDevice> &precedingLayer,
					int             maxSeqLength,
					int             layerID)
        : PostOutputLayer<TDevice>(layerChild, precedingLayer, -1,
				   maxSeqLength, layerID)
    {
	// feature dimension = precedingLayer size
	m_featSize = precedingLayer.size();

	// allocate memory space for feature buffer
	cpu_real_vector temp_data_buffer;
	temp_data_buffer.resize(this->outputs().size() / this->size() * m_featSize);
	thrust::fill(temp_data_buffer.begin(), temp_data_buffer.end(), 0.0);
	m_natDataFeat = temp_data_buffer;
	m_synDataFeat = temp_data_buffer;
	temp_data_buffer.clear();


	// a flag to switch between the buffer to store features
	m_loadNatFeat = false;
    }

    template <typename TDevice>
    SsePostOutputLayerForFeatTrans<TDevice>::~SsePostOutputLayerForFeatTrans()
    {
    }

    template <typename TDevice>
    const std::string& SsePostOutputLayerForFeatTrans<TDevice>::type() const
    {
        static const std::string s("featsse");
        return s;
    }

    template <typename TDevice>
    real_t SsePostOutputLayerForFeatTrans<TDevice>::calculateError()
    {
	internal::ComputeSseFn fn;
	fn.layerSize = m_featSize;
	fn.patTypes  = helpers::getRawPointer(this->patTypes());

	int n = this->curMaxSeqLength() * this->parallelSequences() * m_featSize;

	real_t mse = (real_t)0.5 * thrust::transform_reduce(
                thrust::make_zip_iterator(thrust::make_tuple(m_natDataFeat.begin(),
							     m_synDataFeat.begin(),
							     thrust::counting_iterator<int>(0))),
		thrust::make_zip_iterator(thrust::make_tuple(m_natDataFeat.begin()+n,
							     m_synDataFeat.begin()+n,
							     thrust::counting_iterator<int>(0)+n)),
		fn,
		(real_t)0,
		thrust::plus<real_t>());
	return mse;
    }

    template <typename TDevice>
    void SsePostOutputLayerForFeatTrans<TDevice>::computeForwardPass(const int nnState)
    {
	if (m_loadNatFeat == false){
	    thrust::copy(this->precedingLayer().outputs().begin(),
			 this->precedingLayer().outputs().end(),
			 m_natDataFeat.begin());
	    m_loadNatFeat = true;
	}else{
	    thrust::copy(this->precedingLayer().outputs().begin(),
			 this->precedingLayer().outputs().end(),
			 m_synDataFeat.begin());
	    m_loadNatFeat = false;
	}
	
    }

    template <typename TDevice>
    void SsePostOutputLayerForFeatTrans<TDevice>::computeForwardPass(const int timeStep,
							 const int nnState)
    {
    }

    template <typename TDevice>
    void SsePostOutputLayerForFeatTrans<TDevice>::computeBackwardPass(const int nnState)
    {

	internal::ComputeOutputErrorFn fn;
	fn.layerSize = m_featSize;
	fn.patTypes  = helpers::getRawPointer(this->patTypes());

	int n = this->curMaxSeqLength() * this->parallelSequences() * m_featSize;

	thrust::transform(
               thrust::make_zip_iterator(thrust::make_tuple(m_synDataFeat.begin(),
							    m_natDataFeat.begin(),
							    thrust::counting_iterator<int>(0))),
	       thrust::make_zip_iterator(thrust::make_tuple(m_synDataFeat.begin()+n,
							    m_natDataFeat.begin()+n,
							    thrust::counting_iterator<int>(0)+n)),
	       this->_outputErrors().begin(),
	       fn);
    }

    template <typename TDevice>
    void SsePostOutputLayerForFeatTrans<TDevice>::computeBackwardPass(const int timeStep,
								      const int nnState)
    {
	if (timeStep == this->curMaxSeqLength())
	    this->computeBackwardPass(nnState);
    }
    // explicit template instantiations
    template class SsePostOutputLayerForFeatTrans<Cpu>;
    template class SsePostOutputLayerForFeatTrans<Gpu>;

} // namespace layers
