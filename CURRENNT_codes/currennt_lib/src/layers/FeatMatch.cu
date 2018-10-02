/******************************************************************************
 * This file is an addtional component of CURRENNT. 
 * Xin WANG
 * National Institute of Informatics, Japan
 * 2016
 *
 * This file is part of CURRENNT. 
 * Copyright (c) 2013 Johannes Bergmann, Felix Weninger, Bjoern Schuller
 * Institute for Human-Machine Communication
 * Technische Universitaet Muenchen (TUM)
 * D-80290 Munich, Germany
 *
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
/*
 */

#ifdef _MSC_VER
#   pragma warning (disable: 4244) // thrust/iterator/iterator_adaptor.h(121): warning C4244: '+=' : conversion from '__int64' to 'int', possible loss of data
#endif

#include "FeatMatch.hpp"

#include "../helpers/getRawPointer.cuh"
#include "../helpers/Matrix.hpp"
#include "../helpers/min.cuh"
#include "../helpers/max.cuh"
#include "../helpers/safeExp.cuh"
#include "../helpers/JsonClasses.hpp"
#include "../helpers/misFuncs.hpp"

#include "../activation_functions/Tanh.cuh"
#include "../activation_functions/Logistic.cuh"
#include "../activation_functions/Identity.cuh"
#include "../activation_functions/Relu.cuh"

#include "../Configuration.hpp"
#include "../MacroDefine.hpp"

#include <boost/foreach.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>

#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>

#include <sstream>
#include <fstream>
#include <cmath>


namespace internal {
namespace {
    
    /*
    struct ComputeSseFeatMatch
    {
        int layerSize;

        const char *patTypes;

        __host__ __device__ real_t operator() (const thrust::tuple<real_t, 
	 real_t, int> &values) const
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
	}; */

    struct ComputeFeatMatchError
    {        
    	int         layerSize;
	real_t      ganRatio;
    	const char *patTypes;
	
    	__host__ __device__ real_t operator() (const thrust::tuple<const real_t&,
					       const real_t&, int> &t) const        
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
	    real_t error = (ganRatio * (actualOutput - targetOutput));
	    return error;
	}
    };
    
}
}

namespace layers{
    
    template <typename TDevice>
    FeatMatchLayer<TDevice>::FeatMatchLayer(const helpers::JsonValue &layerChild,
					    Layer<TDevice> &precedingLayer,
					    int             maxSeqLength,
					    int             layerID)
	: PostOutputLayer<TDevice> (layerChild, precedingLayer, precedingLayer.size(),
				    maxSeqLength, layerID, true)
	, m_ganRatio (1.0)
    {
	m_ganRatio  = (layerChild->HasMember("ganRatioGen") ? 
		       static_cast<real_t>((*layerChild)["ganRatioGen"].GetDouble()) :1.0);
	m_outputBuffer.resize(this->outputs().size(), 0.0);

	if (this->size() != precedingLayer.size()){
	    throw std::runtime_error("Error featMatch must be same size as previous layer");
	}
	
	if (this->precedingLayer().getSaveMemoryFlag())
	    throw std::runtime_error("layer before featmatch is reduced in mem");  

	
    }

    template <typename TDevice>
    FeatMatchLayer<TDevice>::~FeatMatchLayer()
    {
    }

    template <typename TDevice>
    void FeatMatchLayer<TDevice>::loadSequences(const data_sets::DataSetFraction &fraction,
						const int nnState)
    {
	Layer<TDevice>::loadSequences(fraction, nnState);
    }

    template <typename TDevice>
    real_t FeatMatchLayer<TDevice>::calculateError()
    {
	// No need to calculate error here
	//
	return 0.0;
    }

    template <typename TDevice>
    void FeatMatchLayer<TDevice>::computeForwardPass(const int nnState)
    {
	if (nnState == NN_STATE_GAN_GEN_FEATMAT){
	    // always copy the previous output
	    thrust::copy(this->precedingLayer().outputs().begin(),
			 this->precedingLayer().outputs().end(),
			 m_outputBuffer.begin());
	}else{
	    // always copy the previous output
	    thrust::copy(this->precedingLayer().outputs().begin(),
			 this->precedingLayer().outputs().end(),
			 this->outputs().begin());
	}
    }

    template <typename TDevice>
    void FeatMatchLayer<TDevice>::computeForwardPass(const int timeStep, const int nnState)
    {
	throw std::runtime_error("Single step computeForwardPass is not implemented featMatch");
    }

    template <typename TDevice>
    void FeatMatchLayer<TDevice>::computeBackwardPass(const int nnState)
    {
	if (nnState == NN_STATE_GAN_GEN){
	    // GAN generator phase
	    
	    internal::ComputeFeatMatchError fn;
	    fn.layerSize = this->size();
	    fn.patTypes  = helpers::getRawPointer(this->patTypes());
	    fn.ganRatio  = m_ganRatio;
	    int n = this->curMaxSeqLength() * this->parallelSequences() * this->size();
	    
	    thrust::transform(
               thrust::make_zip_iterator(
		thrust::make_tuple(this->precedingLayer().outputs().begin(),
				   m_outputBuffer.begin(),
				   thrust::counting_iterator<int>(0))),
	       thrust::make_zip_iterator(
		thrust::make_tuple(this->precedingLayer().outputs().begin() + n,
				   m_outputBuffer.begin()                   + n,
				   thrust::counting_iterator<int>(0)        + n)),
	       this->precedingLayer().outputErrors().begin(),
	       fn);
	    
	}else{
	    // other cases, copy the previous output
	    thrust::copy(this->outputErrors().begin(),
			 this->outputErrors().end(),
			 this->precedingLayer().outputErrors().begin());
	}
    }

    template <typename TDevice>
    const std::string& FeatMatchLayer<TDevice>::type() const
    {
        static const std::string s("featmatch");
        return s;
    }


    template <typename TDevice>
    typename FeatMatchLayer<TDevice>::real_vector& FeatMatchLayer<TDevice>::secondOutputs()
    {
	// This is not used at all;
	throw std::runtime_error("This function should not be used");
	//return m_feedBackOutput;
    }

    template <typename TDevice>
    void FeatMatchLayer<TDevice>::exportLayer(const helpers::JsonValue &layersArray, 
						 const helpers::JsonAllocator &allocator) const
    {
	PostOutputLayer<TDevice>::exportLayer(layersArray, allocator);
	(*layersArray)[layersArray->Size() - 1].AddMember("ganRatioGen",
							  m_ganRatio, allocator);

    }

    
    template class FeatMatchLayer<Cpu>;
    template class FeatMatchLayer<Gpu>;
    
}
