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
#   pragma warning (disable: 4244) // thrust/iterator/iterator_adaptor.h(121): warning C4244: '+=' : conversion from '__int64' to 'int', possible loss of data
#endif

#include "SsePostOutputLayerMultiSource.hpp"
#include "../helpers/getRawPointer.cuh"
#include "../helpers/JsonClasses.hpp"
#include "../helpers/misFuncs.hpp"

#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <boost/foreach.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/algorithm/string.hpp>
#include <vector>



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
    	{            // unpack the tuple
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
    SsePostOutputMultiLayer<TDevice>::SsePostOutputMultiLayer(const helpers::JsonValue &layerChild,
							      Layer<TDevice> &precedingLayer,
							      int             maxSeqLength,
							      int             layerID)
        : PostOutputLayer<TDevice>(layerChild, precedingLayer, precedingLayer.size(),
				   maxSeqLength, layerID)
    {
	
	m_sourceLayersStr = (layerChild->HasMember("sourceLayers") ? 
			     ((*layerChild)["sourceLayers"].GetString()) : "");
	m_sourceLayers.clear();
	m_sourceLayersNames.clear();	
	misFuncs::ParseStrOpt(m_sourceLayersStr, m_sourceLayersNames, ",");
	

    }

    template <typename TDevice>
    SsePostOutputMultiLayer<TDevice>::~SsePostOutputMultiLayer()
    {
    }

    template <typename TDevice>
    const std::string& SsePostOutputMultiLayer<TDevice>::type() const
    {
        static const std::string s("sse_multi");
        return s;
    }
    
    template <typename TDevice>
    real_t SsePostOutputMultiLayer<TDevice>::calculateError()
    {
	real_t mse = 0.0;
	
	internal::ComputeSseFn fn;
	fn.layerSize = this->size();
	fn.patTypes  = helpers::getRawPointer(this->patTypes());

	int n = this->curMaxSeqLength() * this->parallelSequences() * this->size();

	BOOST_FOREACH (Layer<TDevice> *layer, this->m_sourceLayers) {
	    mse += (real_t)0.5 * thrust::transform_reduce(
                thrust::make_zip_iterator(
			thrust::make_tuple(this->_targets().begin(),
					   layer->outputs().begin(),
					   thrust::counting_iterator<int>(0))),
		thrust::make_zip_iterator(
			thrust::make_tuple(this->_targets().begin()+n,
					   layer->outputs().begin()+n,
					   thrust::counting_iterator<int>(0)+n)),
		fn,
		(real_t)0,
		thrust::plus<real_t>());
	}
	return mse;
    }

    template <typename TDevice>
    void SsePostOutputMultiLayer<TDevice>::computeForwardPass(const int nnState)
    {
    }

    template <typename TDevice>
    void SsePostOutputMultiLayer<TDevice>::computeForwardPass(const int timeStep,
							 const int nnState)
    {
    }

    template <typename TDevice>
    void SsePostOutputMultiLayer<TDevice>::computeBackwardPass(const int nnState)
    {
	internal::ComputeOutputErrorFn fn;
	fn.layerSize = this->size();
	fn.patTypes  = helpers::getRawPointer(this->patTypes());
	
	int n = this->curMaxSeqLength() * this->parallelSequences() * this->size();

	BOOST_FOREACH (Layer<TDevice> *layer, this->m_sourceLayers) {
	    thrust::transform(
               thrust::make_zip_iterator(
		thrust::make_tuple(layer->outputs().begin(),
				   this->_targets().begin(),
				   thrust::counting_iterator<int>(0))),
	       thrust::make_zip_iterator(
		thrust::make_tuple(layer->outputs().begin()+n,
				   this->_targets().begin()+n,
				   thrust::counting_iterator<int>(0)+n)),
	       layer->outputErrors().begin(),
	       fn);
	}
    }
	


    template <typename TDevice>
    void SsePostOutputMultiLayer<TDevice>::computeBackwardPass(const int timeStep, const int nnState)
    {
	if (timeStep == this->curMaxSeqLength() - 1)
	    this->computeBackwardPass(nnState);
    }

    template <typename TDevice>
    void SsePostOutputMultiLayer<TDevice>::linkTargetLayer(Layer<TDevice> &targetLayer)
    {
	
	for (int i = 0; i<m_sourceLayersNames.size(); i++){
	    if (targetLayer.name() == m_sourceLayersNames[i] &&
		targetLayer.size() == this->size()){
		m_sourceLayers.push_back(&targetLayer);
		printf("\nLink for SsePostOutputMultiLayer: %s", targetLayer.name().c_str());
		break;
	    }
	}
    }

    template <typename TDevice>
    void SsePostOutputMultiLayer<TDevice>::exportLayer(const helpers::JsonValue &layersArray,
					 const helpers::JsonAllocator &allocator) const
    {
	PostOutputLayer<TDevice>::exportLayer(layersArray, allocator);
	(*layersArray)[layersArray->Size() - 1].AddMember("sourceLayers", m_sourceLayersStr.c_str(),
							  allocator);
	
    }
    
    // explicit template instantiations
    template class SsePostOutputMultiLayer<Cpu>;
    template class SsePostOutputMultiLayer<Gpu>;

} // namespace layers
