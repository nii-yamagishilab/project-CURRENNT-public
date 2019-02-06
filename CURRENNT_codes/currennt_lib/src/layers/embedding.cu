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

#include "embedding.hpp"
#include "../helpers/getRawPointer.cuh"
#include "../helpers/Matrix.hpp"
#include "../activation_functions/Tanh.cuh"
#include "../activation_functions/Logistic.cuh"
#include "../activation_functions/Identity.cuh"
#include "../activation_functions/Relu.cuh"

#include "../helpers/JsonClasses.hpp"
#include "../Configuration.hpp"

#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/for_each.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>

#include <typeinfo>


namespace internal {
namespace {

    // copy segment
    struct CopyPart
    {
	real_t *source;
	real_t *target;
	
	int srcDim;
	int srcS;     // the first dimension to be copied in source stream

	int copyDim;  // dimension of the data to be copied
	int embeddingWhere;
	
	int tarDim;
	int tarS;     // the first dimension to store the copied data in target stream

	const char *patTypes;     // 
	
	__host__ __device__ void operator() (const thrust::tuple<real_t&, int> &t) const
	{
	    int outputIdx = t.get<1>();
	    int timeIdx   = outputIdx / copyDim;
	    int dimIdx    = outputIdx % copyDim;

	    // skip dummy frame (for parallel sentence processing)
	    if (patTypes != NULL && patTypes[timeIdx] == PATTYPE_NONE)
		return;
	    
	    // skip the dimension for embedding index
	    if ((tarS + dimIdx) == embeddingWhere)
		return;
	    
	    target[timeIdx * tarDim + tarS + dimIdx] = source[timeIdx * srcDim + srcS + dimIdx];
	}
    };

    struct GetEmbeddingIndex
    {
	real_t indexScale;
	real_t indexBias;
	int    embeddingNum;
	int    previousVecDim;
	int    embeddingWhere;
	
	int    shiftInTime;
	real_t *inputDataBuf;
	
	// for 0 : T
	__host__ __device__ void operator() (const thrust::tuple<real_t&, int> &t) const
	{
	    float inputIndex = inputDataBuf[(t.get<1>()-shiftInTime) * previousVecDim
					    + embeddingWhere];
	    int tmpIdx = (int)(inputIndex * indexScale + indexBias);
	    
	    tmpIdx = (tmpIdx < 0 ? 0: tmpIdx);
	    t.get<0>() = (tmpIdx < embeddingNum ? tmpIdx:(embeddingNum-1));
	}
	
    };
    
    struct LoadEmbeddings
    {
	int embeddingDim;    // dimension of embedding vectors
	int embeddingWhere;  // dimension index of the embedding index
	int previousVecDim;  // dimension of previous layer's output
	int thisLayerVecDim; // dimension of this layer's output
	
	real_t *inputDataBuf;
	real_t *index;
	real_t *embeddings;
	
	int shiftInTime;
	int shiftOutTime;
	
	const char *patTypes;

	// for 0 : T * featureDim
	__host__ __device__ void operator() (const thrust::tuple<real_t&, int> &t) const
	{
	    int timeIdxOut = t.get<1>() / thisLayerVecDim;
	    int featIdx    = t.get<1>() % thisLayerVecDim;
	    
	    if (patTypes[timeIdxOut] == PATTYPE_NONE){
		// dummy node
		t.get<0>() = 0;

	    }else{
		if (featIdx == embeddingWhere) {
		    // this is the dimension for index, for simplicity, set it to zero
		    t.get<0>() = 0;
		    
		}else if (featIdx < previousVecDim) {
		    // corresponding to the input data not used for embedding
		    t.get<0>() = inputDataBuf[(timeIdxOut - shiftInTime) * previousVecDim
					      + featIdx];
		}else{
		    // load the embedding vectorrs
		    t.get<0>() =
			embeddings[((int) index[timeIdxOut - shiftOutTime] * embeddingDim) +
				   featIdx - previousVecDim];
		}
	    }
	}
    };


    struct collectGradients
    {
	int embeddingDim;
	int embeddingStartDim;
	int errorBufDim;
	int totalTime;

	real_t *inputErrBuf;
	real_t *index;
	
	const char *patTypes;
	
	// for 0 : weight num
	__host__ __device__ void operator() (const thrust::tuple<real_t&, int> &t) const
	{
	    int thisEmIdx = t.get<1>() / embeddingDim;
	    int featDim   = t.get<1>() % embeddingDim;

	    real_t gradient = 0.0;
	    for (int timeIdx = 0; timeIdx < totalTime; timeIdx++){
		if (patTypes[timeIdx] == PATTYPE_NONE)
		    continue;
		if (((int)index[timeIdx]) == thisEmIdx)
		    gradient += inputErrBuf[timeIdx * errorBufDim + featDim + embeddingStartDim];
	    }
	    t.get<0>() = gradient;
	    
	}
    };

    
}
}

namespace embeddingTools {

    int getEmbeddingWeightNum(const helpers::JsonValue &layerChild, const int preLayerSize){
	int thisLayerSize = (layerChild->HasMember("size") ? (*layerChild)["size"].GetInt() : 0);
	int numEmbedding  = (layerChild->HasMember("embeddingNum") ?
			     (*layerChild)["embeddingNum"].GetInt() : 0);

	if (thisLayerSize == 0)
	    throw std::runtime_error("Error in network: embedding layer size is 0\n");
	if (numEmbedding == 0)
	    throw std::runtime_error("Error in network: number of embedding vectors is 0\n");
	if (thisLayerSize < preLayerSize)
	    throw std::runtime_error("Error in network: embedding layer size < pre layer size\n");
	
	return ((int)std::ceil(((float)((thisLayerSize - preLayerSize) * numEmbedding))
			       /((float)thisLayerSize)));
	
    }
    
}

namespace layers {

    template <typename TDevice>
    EmbeddingLayer<TDevice>::EmbeddingLayer(const helpers::JsonValue &layerChild,
					    const helpers::JsonValue &weightsSection,
					    Layer<TDevice>           &precedingLayer,
					    int                       maxSeqLength,
					    int                       layerID)
	: TrainableLayer<TDevice>(layerChild, weightsSection, 0,
				  embeddingTools::getEmbeddingWeightNum(layerChild,
									precedingLayer.size()),
				  precedingLayer, maxSeqLength, layerID)
	, m_numEmbedding   ((layerChild->HasMember("embeddingNum") ?
			     (*layerChild)["embeddingNum"].GetInt() : 0))
	, m_dimEmbedding   (0)
	, m_idxEmIdx       ((layerChild->HasMember("embeddingIndexWhere") ?
			     (*layerChild)["embeddingIndexWhere"].GetInt() : 0))
	, m_indexScale     ((layerChild->HasMember("indexScale") ? 
			     static_cast<real_t>((*layerChild)["indexScale"].GetDouble()) : 1.0))
	, m_indexBias      ((layerChild->HasMember("indexBias") ? 
			     static_cast<real_t>((*layerChild)["indexBias"].GetDouble()) : 0.0))
    {
	
	m_dimEmbedding = this->size() - precedingLayer.size();

	
	// allocate the memory for index vector
	cpu_real_vector temp(this->parallelSequences() * maxSeqLength, 0.0);
	m_embeddingIdxVec = temp;

	printf("\n\tEmbedding layer: # embedding vectors: %d, dim embedding vectors %d\n",
	       m_numEmbedding, m_dimEmbedding);
	printf("\n\tEmbedding index: input_index * %f + %f\n", m_indexScale, m_indexBias);
	if (m_idxEmIdx >= precedingLayer.size())
	    throw std::runtime_error("Error in network: embeddingIndexWhere > preious layer size");
	printf("\n\tEmbedding index is the %d-th dimension of input feature vector\n", m_idxEmIdx);

	
    }

    // Destructor
    template <typename TDevice>
    EmbeddingLayer<TDevice>::~EmbeddingLayer()
    {
    }

    template <typename TDevice>
    const std::string& EmbeddingLayer<TDevice>::type() const
    {
	static std::string s1("embedding");
        return s1;
    }

    template <typename TDevice>
    void EmbeddingLayer<TDevice>::exportLayer(const helpers::JsonValue &layersArray,
					      const helpers::JsonAllocator &allocator) const
    {
        TrainableLayer<TDevice>::exportLayer(layersArray, allocator);
	
	(*layersArray)[layersArray->Size() - 1].AddMember("embeddingNum",
							  m_numEmbedding, allocator);
	
	(*layersArray)[layersArray->Size() - 1].AddMember("embeddingIndexWhere",
							  m_idxEmIdx, allocator);
	
	(*layersArray)[layersArray->Size() - 1].AddMember("indexScale",
							  m_indexScale, allocator);
	
	(*layersArray)[layersArray->Size() - 1].AddMember("indexBias",
							  m_indexBias, allocator);	
    }

    template <typename TDevice>
    void EmbeddingLayer<TDevice>::reduceOutputBuffer()
    {
	// Only care about the output buffer
	// The gradients buffer is cleaned by the Layer.cu in initialization
	
	this->resizeOutputBuffer(this->parallelSequences() * this->size());
	this->setSaveMemoryFlag(true);

	// reduce the memory for index
	m_embeddingIdxVec.resize(this->parallelSequences(), 0);
	m_embeddingIdxVec.shrink_to_fit();
	
	printf("\t[mem saved]");
	
    }
    
    template <typename TDevice>
    int EmbeddingLayer<TDevice>::outputBufPtrBias(const int timeStepTimesParallel,
						  const int nnState)
    {
	if (this->getSaveMemoryFlag()){
	    return timeStepTimesParallel * this->size();
	}else{
	    return 0;
	}
    }	

    template <typename TDevice>
    void EmbeddingLayer<TDevice>::computeForwardPass(const int nnState)
    {
	if (this->getSaveMemoryFlag())
	    throw std::runtime_error("Memory save mode should be turned off");
	
	int uttLength = this->curMaxSeqLength() * this->parallelSequences();
	
	{{
	    // Load the embedding index
	    internal::GetEmbeddingIndex fn1;
	    fn1.indexScale     = this->m_indexScale;
	    fn1.indexBias      = this->m_indexBias;
	    fn1.embeddingNum   = this->m_numEmbedding;
	    fn1.previousVecDim = this->precedingLayer().size();
	    fn1.embeddingWhere = this->m_idxEmIdx;
	    
	    fn1.inputDataBuf   = helpers::getRawPointer(this->precedingLayer().outputs());
	    fn1.shiftInTime    = 0;
	    thrust::for_each(
		thrust::make_zip_iterator(
			thrust::make_tuple(this->m_embeddingIdxVec.begin(),
					   thrust::counting_iterator<int>(0))),
		thrust::make_zip_iterator(
			thrust::make_tuple(this->m_embeddingIdxVec.begin()   + uttLength,
					   thrust::counting_iterator<int>(0) + uttLength)),
		fn1);  
	}}

	{{
	    // load the embedding data into the output buffer
	    internal::LoadEmbeddings fn2;
	    fn2.embeddingDim   = this->m_dimEmbedding;
	    fn2.embeddingWhere = this->m_idxEmIdx;
	    fn2.previousVecDim = this->precedingLayer().size();
	    fn2.thisLayerVecDim = this->size();
	    
	    fn2.inputDataBuf   = helpers::getRawPointer(this->precedingLayer().outputs());
	    fn2.index          = helpers::getRawPointer(this->m_embeddingIdxVec);
	    fn2.embeddings     = helpers::getRawPointer(this->weights());

	    fn2.patTypes       = helpers::getRawPointer(this->patTypes());
	    
	    fn2.shiftInTime    = 0;
	    fn2.shiftOutTime   = 0;
	    
	    int n1 = uttLength * this->size();
	    
	    thrust::for_each(
		thrust::make_zip_iterator(
			thrust::make_tuple(this->outputs().begin(),
					   thrust::counting_iterator<int>(0))),
		thrust::make_zip_iterator(
			thrust::make_tuple(this->outputs().begin()           + n1,
					   thrust::counting_iterator<int>(0) + n1)),
		fn2);  
	}}
	
    }

    
    template <typename TDevice>
    void EmbeddingLayer<TDevice>::computeForwardPass(const int timeStep, const int nnState)
    {
	int effTimeStart = timeStep * this->parallelSequences();
	int effTimeEnd   = (timeStep+1) * this->parallelSequences();
	
	// Pointer to the output of previous layer (input buffer)
	int shiftIn  = this->precedingLayer().outputBufPtrBias(timeStep * this->parallelSequences(),
							       nnState);
	// Pointer to the output of this layer
	int shiftOut = this->outputBufPtrBias(timeStep * this->parallelSequences(), nnState);

	int shiftInTime = shiftIn / this->precedingLayer().size();
	int shiftOutTime= shiftOut / this->size();

	
	
	{{
	    // Load the embedding index
	    internal::GetEmbeddingIndex fn1;
	    fn1.indexScale     = this->m_indexScale;
	    fn1.indexBias      = this->m_indexBias;
	    fn1.embeddingNum   = this->m_numEmbedding;
	    fn1.previousVecDim = this->precedingLayer().size();
	    fn1.embeddingWhere = this->m_idxEmIdx;
	    
	    fn1.inputDataBuf   = helpers::getRawPointer(this->precedingLayer().outputs());
	    fn1.shiftInTime    = shiftInTime;
	    
	    thrust::for_each(
		thrust::make_zip_iterator(
		   thrust::make_tuple(
			this->m_embeddingIdxVec.begin() + effTimeStart - shiftOutTime,
			thrust::counting_iterator<int>(0) + effTimeStart)),
		thrust::make_zip_iterator(
		   thrust::make_tuple(
			this->m_embeddingIdxVec.begin() + effTimeEnd - shiftOutTime,
			thrust::counting_iterator<int>(0) + effTimeEnd)),
		fn1);  
	}}

	{{
	    // load the embedding data into the output buffer
	    internal::LoadEmbeddings fn2;
	    fn2.embeddingDim   = this->m_dimEmbedding;
	    fn2.embeddingWhere = this->m_idxEmIdx;
	    fn2.previousVecDim = this->precedingLayer().size();
	    fn2.thisLayerVecDim = this->size();
	    
	    fn2.inputDataBuf   = helpers::getRawPointer(this->precedingLayer().outputs());
	    fn2.index          = helpers::getRawPointer(this->m_embeddingIdxVec);
	    fn2.embeddings     = helpers::getRawPointer(this->weights());

	    fn2.patTypes       = helpers::getRawPointer(this->patTypes());
	    
	    fn2.shiftOutTime   = shiftOutTime;
	    fn2.shiftInTime    = shiftInTime;
	    
	    thrust::for_each(
	     thrust::make_zip_iterator(
	      thrust::make_tuple(
		this->outputs().begin()           + effTimeStart * this->size() - shiftOut,
		thrust::counting_iterator<int>(0) + effTimeStart * this->size())),
	     thrust::make_zip_iterator(
	      thrust::make_tuple(
		this->outputs().begin()           + effTimeEnd * this->size() - shiftOut,
		thrust::counting_iterator<int>(0) + effTimeEnd * this->size())),
	     fn2);  
	}}
	
    }

    template <typename TDevice>
    void EmbeddingLayer<TDevice>::computeBackwardPass(const int nnState)
    {
	if (this->getSaveMemoryFlag())
	    throw std::runtime_error("Memory save mode should be turned off");
	
	thrust::fill(this->precedingLayer().outputErrors().begin(),
		     this->precedingLayer().outputErrors().end(), 0.0);
	thrust::fill(this->_weightUpdates().begin(), this->_weightUpdates().end(), 0.0);
	
	int uttLength = this->curMaxSeqLength() * this->parallelSequences();
	
	{{
	    // copy the gradients for the unembedded part
	    internal::CopyPart fn1;
	    fn1.source = helpers::getRawPointer(this->outputErrors());
	    fn1.target = helpers::getRawPointer(this->precedingLayer().outputErrors());

	    fn1.srcDim = this->size();
	    fn1.srcS   = 0;
	    fn1.tarDim = this->precedingLayer().size();
	    fn1.tarS   = 0;
	    
	    fn1.copyDim        = this->precedingLayer().size();

	    fn1.patTypes       = helpers::getRawPointer(this->patTypes());
	    fn1.embeddingWhere = this->m_idxEmIdx;

	    int n = uttLength * this->precedingLayer().size();
	    thrust::for_each(
	     thrust::make_zip_iterator(
	       thrust::make_tuple(this->precedingLayer().outputErrors().begin(),
				  thrust::counting_iterator<int>(0))),
	     thrust::make_zip_iterator(
	       thrust::make_tuple(this->precedingLayer().outputErrors().begin() + n,
				  thrust::counting_iterator<int>(0)             + n)),
	     fn1);  
	}}
	
	{{
	    // collect gradients from this->outputErrors()
	    internal::collectGradients fn2;
	    fn2.embeddingDim      = this->m_dimEmbedding;
	    fn2.embeddingStartDim = this->precedingLayer().size();
	    fn2.errorBufDim       = this->size();
	    fn2.totalTime         = uttLength;
	    fn2.inputErrBuf       = helpers::getRawPointer(this->outputErrors());
	    fn2.index             = helpers::getRawPointer(this->m_embeddingIdxVec);
	    fn2.patTypes          = helpers::getRawPointer(this->patTypes());

	    thrust::for_each(
	     thrust::make_zip_iterator(
	       thrust::make_tuple(this->_weightUpdates().begin(),
				  thrust::counting_iterator<int>(0))),
	     thrust::make_zip_iterator(
		thrust::make_tuple(this->_weightUpdates().end(),
				   thrust::counting_iterator<int>(0) + this->weights().size())),
	     fn2);  
	}}
	
    }

    
    template <typename TDevice>
    void EmbeddingLayer<TDevice>::computeBackwardPass(const int timeStep, const int nnState)
    {
	throw std::runtime_error("EmbeddingLayer computeBackwardPass(timeStep) not implemented");
    }
    
    template class EmbeddingLayer<Cpu>;
    template class EmbeddingLayer<Gpu>;
}
