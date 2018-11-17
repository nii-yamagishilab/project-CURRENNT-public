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

#include "ExternalLoader.hpp"

#include "../helpers/getRawPointer.cuh"
#include "../helpers/Matrix.hpp"
#include "../helpers/JsonClasses.hpp"
#include "../helpers/misFuncs.hpp"
#include "../activation_functions/Logistic.cuh"
#include "../activation_functions/Tanh.cuh"
#include "../MacroDefine.hpp"

#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/for_each.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/fill.h>
#include <thrust/random.h>
#include <boost/foreach.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>
#include <vector>
#include <stdexcept>

#include "../Configuration.hpp"


namespace internal{
namespace {

    struct loadExternalData
    {
	int  featureDim;
	int  paralNum;
	int  maxFeatureLength;
	int  resolution;
	int  indexLoadM;
	
	const real_t *sourceData;
	const real_t *frameIndex;
	const real_t *contextMV;
	const char   *patTypes;
	
	__host__ __device__ void operator() (const thrust::tuple<real_t&, int> &t) const
	{
	    int dimIdx  = t.get<1>() % featureDim;
	    int timeIdx = t.get<1>() / featureDim;
	    int reTimeIdx = timeIdx  / paralNum;
	    int paralIdx  = timeIdx  % paralNum;

	    if (patTypes[timeIdx] == PATTYPE_NONE)
		t.get<0>() = 0.0;
	    else{
		// frameIndex is at the resolution of 1
		// frameIndex[timeIdx * resolution] is the value of index
		// featIdx is the value of index with parallel mode into consideration

		int frameIdx = frameIndex[reTimeIdx * resolution * paralNum + paralIdx];
		if (indexLoadM == NN_INDEXLOADMETHOD_1)
		    frameIdx = frameIdx / resolution;
		    
		int featIdx = frameIdx * paralNum + paralIdx;
		if (frameIdx >= maxFeatureLength){
		    t.get<0>() = 0.0;
		}else if(contextMV){
		    t.get<0>() = ((sourceData[featIdx * featureDim + dimIdx] - contextMV[dimIdx])/
				  ((contextMV[dimIdx + featureDim]<1e-5f) ?
				   (1.0): (contextMV[dimIdx + featureDim])));
		}else{
		    t.get<0>() = sourceData[featIdx * featureDim + dimIdx];
		}
		
	    }
	}
    };


}
}


namespace layers{


    template <typename TDevice>
    ExternalLoader<TDevice>::ExternalLoader(const helpers::JsonValue &layerChild,
					    const helpers::JsonValue &weightsSection,
					    Layer<TDevice>           &precedingLayer,
					    int                       maxSeqLength,
					    int                       layerID)
	: TrainableLayer<TDevice>(layerChild, weightsSection, 0, 0,
				  precedingLayer, maxSeqLength, layerID)
    {
	if (precedingLayer.type() != "input")
	    throw std::runtime_error("Externalloader is only implemented after the input layer");

	m_externalDataMV.clear();
	
	// Preparaing the mean and std of input data
	//  1. layerChild ['externalDataMVVec']
	//  2. --source_data_ms
	//  3. layerChild ['externalDataMV']
	if (layerChild->HasMember("externalDataMVVec")){
	    // Mean/std data vector has been provided
	    const rapidjson::Value &dataVec = (*layerChild)["externalDataMVVec"];
	    if (dataVec.Size() != this->size() * 2)
		throw std::runtime_error("externalDataMVVec dimension unmatched with layer size");
	    m_externalDataMV.reserve(this->size()*2);
	    for (rapidjson::Value::ConstValueIterator it = dataVec.Begin(); it!=dataVec.End(); ++it)
		m_externalDataMV.push_back(static_cast<real_t>(it->GetDouble()));              
	    printf("\n\tRead MV from network");
	    m_externalDataMVStr = "";
	}else{
	    // Path of mean/std data vector has been provided
	    if (Configuration::instance().dataSourcePath().size()){
		m_externalDataMVStr = Configuration::instance().dataSourcePath();	    
	    }else{
		m_externalDataMVStr = ((layerChild->HasMember("externalDataMV")) ?
				       ((*layerChild)["externalDataMV"].GetString()) : "");
	    }
	    cpu_real_vector tmp;
	    if (m_externalDataMVStr.size()){
		if (misFuncs::ReadRealData(m_externalDataMVStr, tmp) != this->size() * 2)
		    throw std::runtime_error("externalDataMV dimension unmatched with layer size");
		m_externalDataMV = tmp;
		printf("\n\tRead MV from %s", m_externalDataMVStr.c_str());
	    }else{
		printf("\n\tSkip reading MV");
	    }
	}
	
	
	// m_indexLoadMethod:
	//   default:  index [1, ... N], index n -> n-th data frame
	//             assume the number of data frame is N
	//   method 1: index [1, ... N], index n -> n/timeResolution -th data frame
	//             assume the number of data frame is N/timeResolution + N%timeResolution>0
	// 
	m_indexLoadMethod   = ((layerChild->HasMember("indexLoadMethod")) ?
			       ((*layerChild)["indexLoadMethod"].GetInt()) : 0);

    }

    template <typename TDevice>
    ExternalLoader<TDevice>::~ExternalLoader()
    {
    }

    template <typename TDevice>
    void ExternalLoader<TDevice>::exportLayer(const helpers::JsonValue     &layersArray, 
					      const helpers::JsonAllocator &allocator) const
    {
        TrainableLayer<TDevice>::exportLayer(layersArray, allocator);

	// save data mean/std directly into layerChild (since it is not trainable parameter)
	if (m_externalDataMV.size() == (this->size() * 2)){
	    rapidjson::Value dataMVArray(rapidjson::kArrayType);
	    dataMVArray.Reserve(this->size() * 2, allocator);
	    for (int i = 0; i < (this->size() * 2); ++i)
		dataMVArray.PushBack(m_externalDataMV[i], allocator);
	    (*layersArray)[layersArray->Size() - 1].AddMember("externalDataMVVec",
							      dataMVArray,
							      allocator);
	}else if (m_externalDataMVStr.size()){
	    (*layersArray)[layersArray->Size() - 1].AddMember("externalDataMV",
							      m_externalDataMVStr.c_str(),
							      allocator);
	}else{
	    
	}
	
	if (m_indexLoadMethod > 0)
	    (*layersArray)[layersArray->Size() - 1].AddMember("indexLoadMethod",
							      m_indexLoadMethod,
							      allocator);
    }

    template <typename TDevice>
    void ExternalLoader<TDevice>::loadSequences(const data_sets::DataSetFraction &fraction,
						const int nnState)
    {
	TrainableLayer<TDevice>::loadSequences(fraction, nnState);
	// Load the external linguistic features
	if (fraction.externalInputSize() != this->size()){
	    printf("external data dim %d %s", fraction.externalInputSize(), this->name().c_str());
	    throw std::runtime_error("external data dimension unmatched");
	}
	if (this->outputs().size()<1)
	    throw std::runtime_error("Fail to initialize output buffer");

	// Assume the fraciton.inputs has time resolution as 1
	m_dataBuffer.resize(fraction.inputs().size() + fraction.exInputData().size(), 0.0);
	thrust::copy(fraction.inputs().begin(),      fraction.inputs().end(), m_dataBuffer.begin());
	thrust::copy(fraction.exInputData().begin(), fraction.exInputData().end(),
		     m_dataBuffer.begin() + fraction.inputs().size());
	
	internal::loadExternalData fn1;
	fn1.featureDim = fraction.externalInputSize();
	fn1.paralNum   = this->parallelSequences();
	fn1.maxFeatureLength = fraction.maxExInputLength();
	fn1.sourceData = (helpers::getRawPointer(m_dataBuffer) + fraction.inputs().size());
	fn1.frameIndex = helpers::getRawPointer(m_dataBuffer);
	fn1.patTypes   = helpers::getRawPointer(this->patTypes());
	fn1.contextMV  = ((m_externalDataMV.size() ==this->size() * 2)?
			  helpers::getRawPointer(m_externalDataMV) : NULL);
	fn1.resolution = this->getResolution();
	fn1.indexLoadM = this->m_indexLoadMethod;
	
	int n = this->curMaxSeqLength() * this->parallelSequences() * this->size();
	thrust::for_each(
                 thrust::make_zip_iterator(
		  thrust::make_tuple(this->outputs().begin(),
				     thrust::counting_iterator<int>(0))),
	         thrust::make_zip_iterator(
		  thrust::make_tuple(this->outputs().begin()           + n,
				     thrust::counting_iterator<int>(0) + n)),
		 fn1);

	// 
    }
    
    template <typename TDevice>
    const std::string& ExternalLoader<TDevice>::type() const
    {
        static std::string s;
        if (s.empty()) s = "externalloader";
        return s;
    }

    template <typename TDevice>
    void ExternalLoader<TDevice>::computeForwardPass(const int nnState)
    {
    }

    template <typename TDevice>
    void ExternalLoader<TDevice>::computeForwardPass(const int timeStep, const int nnState)
    {
	throw std::runtime_error("ExternalLoader is not implemented after feedback layer");
    }

    template <typename TDevice>
    void ExternalLoader<TDevice>::computeBackwardPass(const int nnState)
    {	
    }

    template class ExternalLoader<Cpu>;
    template class ExternalLoader<Gpu>;
    
}
