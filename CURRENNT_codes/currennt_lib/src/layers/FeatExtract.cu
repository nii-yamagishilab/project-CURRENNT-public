/******************************************************************************
 * This file is an addtional component of CURRENNT. 
 * Xin WANG
 * National Institute of Informatics, Japan
 * 2016 - 2020
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

#include "FeatExtract.hpp"

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


// CUDA functions
namespace internal{
namespace {

    struct calculateSegMeanVar
    {
	int     featureDim;
	int     paralSeqNm;
	
	int    *segBoundary;
	real_t *sourceData;
	
	const char *patTypes;
	
	__host__ __device__ void operator() (const thrust::tuple<real_t&, real_t&, int> &t) const
	{
	    int outputIdx = t.get<2>();
	    int dimIdx    = outputIdx % featureDim;
	    int timeIdx   = outputIdx / featureDim;

	    if (patTypes[timeIdx] == PATTYPE_NONE){
		t.get<0>() = 0.0;
		return;
	    }
	    
	    int paralBlk  = timeIdx / paralSeqNm;
	    int utteranceIdx   = timeIdx % paralSeqNm;

	    
	    t.get<0>() = 0.0;
	    t.get<1>() = 1.0;

	    real_t delta;
	    real_t delta2;
	    
	    int boundary = segBoundary[timeIdx * 2];
	    
	    if (boundary > 0){
		// If this is the first frame of one segment
		//  calculate the mean/var
		for (int step = 1 ; step <= boundary; step++){
		    delta = sourceData[((paralBlk + step - 1) * paralSeqNm +
					utteranceIdx) * featureDim + dimIdx] - t.get<0>();
		    t.get<0>() += delta / ((real_t)step);
		    delta2 = sourceData[((paralBlk + step - 1) * paralSeqNm +
					 utteranceIdx) * featureDim + dimIdx] - t.get<0>();
		    t.get<1>() += delta * delta2;
		}
		t.get<1>() = t.get<1>()/((real_t)boundary);
	    }
	}
    };

    struct propagateMeanVar
    {
	int     featureDim;
	int     outBufferDim;
	int     paralSeqNm;
	
	int    *segBoundary;
	real_t *meanData;
	real_t *varData;
	
	const char *patTypes;
	
	__host__ __device__ void operator() (const thrust::tuple<real_t&, int> &t) const
	{
	    int outputIdx = t.get<1>();
	    int dimIdx    = outputIdx % outBufferDim;
	    int timeIdx   = outputIdx / outBufferDim;

	    if (patTypes[timeIdx] == PATTYPE_NONE){
		t.get<0>() = 0.0;
		return;
	    }
	    
	    int paralBlk  = timeIdx / paralSeqNm;
	    int utteranceIdx   = timeIdx % paralSeqNm;
	    
	    t.get<0>() = 0.0;
	    
	    real_t shift = 0;      // distance from 1st frame in the segment
	    real_t segLength = 1;  // length of the segment

	    
	    if (segBoundary[timeIdx * 2] > 0){
		// If this is the first frame of one segment
		shift = 0;
		segLength = segBoundary[timeIdx * 2];
	    }else{
		// If this is not the first frame in the segment
		shift = segBoundary[timeIdx * 2]; // shift will be a negative number (look backwards)
		segLength = segBoundary[((paralBlk + (int)shift) * paralSeqNm + utteranceIdx) * 2];
	    }
		    
	    if (dimIdx < featureDim){
		// mean part
		t.get<0>() = meanData[((paralBlk+(int)shift)*paralSeqNm+utteranceIdx)*featureDim +
				      dimIdx] / sqrt(segLength);
	    }else{
		// var part
		t.get<0>() = varData[((paralBlk+(int)shift)*paralSeqNm+utteranceIdx)*featureDim +
				      dimIdx - featureDim] / sqrt(segLength);
	    }
	}
    };    


    struct propagateMeanVarGrad
    {
	int     featureDim;
	int     outBufferDim;
	int     paralSeqNm;
	
	int    *segBoundary;
	
	real_t *data;
	real_t *meanData;
	real_t *varData;
	
	const char *patTypes;
	
	__host__ __device__ void operator() (const thrust::tuple<real_t&, int> &t) const
	{
	    int outputIdx = t.get<1>();
	    int dimIdx    = outputIdx % outBufferDim;
	    int timeIdx   = outputIdx / outBufferDim;

	    if (patTypes[timeIdx] == PATTYPE_NONE){
		t.get<0>() = 0.0;
		return;
	    }
	    
	    int paralBlk  = timeIdx / paralSeqNm;
	    int utteranceIdx   = timeIdx % paralSeqNm;
	    
	    t.get<0>() = 0.0;

	    real_t shift = 0;
	    real_t segLength = 1;
	    
	    if (segBoundary[timeIdx * 2] > 0){
		shift = 0;
		segLength = segBoundary[timeIdx * 2];
	    }else{
		shift = segBoundary[timeIdx * 2];
		segLength = segBoundary[((paralBlk + (int)shift) * paralSeqNm + utteranceIdx) * 2];
	    }

	    // grad from mean and var
	    t.get<0>() = 2.0 / segLength *
		(data[outputIdx] -
		 meanData[((paralBlk + (int)shift) * paralSeqNm + utteranceIdx) * featureDim + dimIdx])
		+ 1.0 / segLength;
	}
    };
    
}
}


namespace layers{
	
    
    template <typename TDevice>
    FeatExtract<TDevice>::FeatExtract(const helpers::JsonValue &layerChild,
					    const helpers::JsonValue &weightsSection,
					    Layer<TDevice>           &precedingLayer,
					    int                       maxSeqLength,
					    int                       layerID)
	: TrainableLayer<TDevice>(layerChild, weightsSection, 0, 0,
				  precedingLayer, maxSeqLength, layerID)
	, m_segLevel    (-1)
    {
	
	m_segLevel       = (layerChild->HasMember("segLevel")?
			    static_cast<int>((*layerChild)["segLevel"].GetInt()) : -1);

	m_featExtractRes = (layerChild->HasMember("featExtractRate")?
			    static_cast<int>((*layerChild)["featExtractRate"].GetInt()) : -1);
	
	m_featConfig     = ((layerChild->HasMember("featExtractConfig")) ? 
			    ((*layerChild)["featExtractConfig"].GetString()) : (""));
	
	/* Not implemented here
	if (m_featConfig.size()){
	    m_featConfigVecH.clear();
	    misFuncs::ParseIntOpt(m_featConfig, m_featConfigVecH);
	    m_featConfigVecD = m_featConfigVecH;
	}else{
	    throw std::runtime_error("Error in network.jsn: featextract needs featExtractConfig");
	    }*/
	// Assume segment-level mean/var will be extracted
	if (this->size() != this->precedingLayer().size() * 2)
	    throw std::runtime_error("featextract layer size != precedingLayer().size * 2");
	
	if (m_segLevel){
	    m_segBoundaryH.resize(this->maxSeqLength() * this->parallelSequences() * 2, 0);
	    m_segBoundaryD = m_segBoundaryH;	
	}else{
	    throw std::runtime_error("Unknown lastShot option number");
	}

	m_featMeanVec = this->precedingLayer().outputs();
	m_featVarVec  = this->precedingLayer().outputs();

    }

    template <typename TDevice>
    FeatExtract<TDevice>::~FeatExtract()
    {
    }

    template <typename TDevice>
    void FeatExtract<TDevice>::exportLayer(const helpers::JsonValue     &layersArray, 
					   const helpers::JsonAllocator &allocator) const
    {
        TrainableLayer<TDevice>::exportLayer(layersArray, allocator);

	(*layersArray)[layersArray->Size() - 1].AddMember("segLevel", m_segLevel, allocator);
	(*layersArray)[layersArray->Size() - 1].AddMember("featExtractRate", m_featExtractRes,
							  allocator);
	(*layersArray)[layersArray->Size() - 1].AddMember("featExtractConfig", m_featConfig.c_str(),
							  allocator);	
    }

    template <typename TDevice>
    void FeatExtract<TDevice>::loadSequences(const data_sets::DataSetFraction &fraction,
						const int nnState)
    {
	TrainableLayer<TDevice>::loadSequences(fraction, nnState);

	// load the segments length
	if (m_segLevel > CHAR_BIT)
	    throw std::runtime_error("featextract: segLevel is larger than expected");
	    
	if (fraction.auxPattypeData().size() == 0)
	    throw std::runtime_error("featextract: requires segment boundary (auxData)");
    
	int pos;
	    
	// boundary in forward / backward direction
	int boundaryR, segmentLength;
	    
	// bit operator, 2^m_segLevel
	char bitOp = (0b01 << m_segLevel);

	// 
	for (int i = 0; i < fraction.numSequences(); i++){
		
	    // the last segment boundry is the end of utterance
	    //boundary  = (fraction.seqInfo(i).length - 1);
	    boundaryR = 0;
	    segmentLength = 0;

	    // decide the lingusitic segment boundary 
	    for (int time = fraction.seqInfo(i).length - 1; time>=0; time--){

		// Abosolute position in the data sequence
		pos = time * this->parallelSequences() + i;  

		// -------
		// Forward direction
		//     N N+1 N+2 ... N+K-1 | N+K      abosolute time (for each single sentence)
		// ... K -   -       -     |          m_segBoundaryH[2*pos]
		// ... K-1 K-2 K-3   0     |          m_segBoundaryH[2*pos + 1]

		m_segBoundaryH[2 * pos + 1] =  boundaryR;
		
		segmentLength++;
		if (fraction.auxPattypeData()[pos] & bitOp){
		    m_segBoundaryH[2 * pos] = segmentLength;
		    segmentLength = 0;
		    boundaryR = 0;
		}else{
		    m_segBoundaryH[2 * pos] = -1;
		    boundaryR++;
		}
	    }

	    segmentLength = 0;
	    // decide the lingusitic segment boundary 
	    for (int time = 0; time<=fraction.seqInfo(i).length - 1; time++){

		// Abosolute position in the data sequence
		pos = time * this->parallelSequences() + i;  

		// -------
		// Forward direction
		//     N N+1 N+2 ... N+K-1 | N+K      abosolute time (for each single sentence)
		// ... K -1  -2      K-1   |          m_segBoundaryH boundary
		// ... K-1 K-2 K-3   0     |          m_segBoundaryH[2*pos + 1]
		segmentLength++;
		if (m_segBoundaryH[2 * pos] >= 0){
		    segmentLength = 0;
		}else{
		    m_segBoundaryH[2 * pos] = -1 * segmentLength;
		}
	    }
	}
	
	m_segBoundaryD = m_segBoundaryH;
	
    }
    
    template <typename TDevice>
    const std::string& FeatExtract<TDevice>::type() const
    {
        static std::string s;
        if (s.empty()) s = "featextract";
        return s;
    }

    template <typename TDevice>
    void FeatExtract<TDevice>::computeForwardPass(const int nnState)
    {
	int timeLength = this->curMaxSeqLength() * this->parallelSequences();


	// step1. get mean / var
	{{	    
	    // use last shot mode based on segmental boundary
	    internal::calculateSegMeanVar fn1;
	    fn1.featureDim  = this->precedingLayer().size();
	    fn1.paralSeqNm  = this->parallelSequences();
	    fn1.segBoundary = helpers::getRawPointer(m_segBoundaryD);
	    fn1.sourceData  = helpers::getRawPointer(this->precedingLayer().outputs());
	    fn1.patTypes    = helpers::getRawPointer(this->patTypes());
	    
	    int n = timeLength * this->precedingLayer().size();
	    thrust::for_each(
               thrust::make_zip_iterator(
		  thrust::make_tuple(m_featMeanVec.begin(),
				     m_featVarVec.begin(),
				     thrust::counting_iterator<int>(0))),
	       thrust::make_zip_iterator(
		  thrust::make_tuple(m_featMeanVec.begin()             + n,
				     m_featVarVec.begin()              + n,
				     thrust::counting_iterator<int>(0) + n)),
	       fn1);	    
	}}

	// step2. propagate mean / std to other time steps
	{{
	    // use last shot mode based on segmental boundary
	    internal::propagateMeanVar fn1;
	    fn1.featureDim  = this->precedingLayer().size();
	    fn1.outBufferDim= this->size();
	    fn1.paralSeqNm  = this->parallelSequences();
	    fn1.segBoundary = helpers::getRawPointer(m_segBoundaryD);
	    fn1.meanData    = helpers::getRawPointer(m_featMeanVec);
	    fn1.varData     = helpers::getRawPointer(m_featVarVec);
	    fn1.patTypes    = helpers::getRawPointer(this->patTypes());
	    
	    int n = timeLength * this->size();
	    
	    thrust::for_each(
               thrust::make_zip_iterator(
		  thrust::make_tuple(this->outputs().begin(),
				     thrust::counting_iterator<int>(0))),
	       thrust::make_zip_iterator(
		  thrust::make_tuple(this->outputs().begin()           + n,
				     thrust::counting_iterator<int>(0) + n)),
	       fn1);
	}}
	

    }

    template <typename TDevice>
    void FeatExtract<TDevice>::computeForwardPass(const int timeStep, const int nnState)
    {
	throw std::runtime_error("featextract not implemented for computeForward(timestep)");
    }

    template <typename TDevice>
    void FeatExtract<TDevice>::computeBackwardPass(const int nnState)
    {
	int timeLength = this->curMaxSeqLength() * this->parallelSequences();
	
	// step1. propagate mean / std to other time steps
	{{
	    // use last shot mode based on segmental boundary
	    internal::propagateMeanVarGrad fn1;
	    fn1.featureDim  = this->precedingLayer().size();
	    fn1.outBufferDim= this->size();
	    fn1.paralSeqNm  = this->parallelSequences();
	    fn1.segBoundary = helpers::getRawPointer(m_segBoundaryD);
	    fn1.data        = helpers::getRawPointer(this->precedingLayer().outputs());
	    fn1.meanData    = helpers::getRawPointer(m_featMeanVec);
	    fn1.varData     = helpers::getRawPointer(m_featVarVec);
	    fn1.patTypes    = helpers::getRawPointer(this->patTypes());
	    
	    int n = timeLength * this->precedingLayer().size();
	    
	    thrust::for_each(
              thrust::make_zip_iterator(
		thrust::make_tuple(this->precedingLayer().outputErrors().begin(),
				     thrust::counting_iterator<int>(0))),
	      thrust::make_zip_iterator(
		thrust::make_tuple(this->precedingLayer().outputErrors().begin() + n,
				     thrust::counting_iterator<int>(0)           + n)),
	       fn1);	    
	    
	}}	

    }

    template <typename TDevice>
    void FeatExtract<TDevice>::computeBackwardPass(const int timeStep, const int nnState)
    {
	throw std::runtime_error("featextract not implemented for computeBackward(timestep)");
    }

    template <typename TDevice>
    void FeatExtract<TDevice>::reduceOutputBuffer()
    {
	throw std::runtime_error("This operator layer doesn't support memory reduce mode");
    }
    
    template <typename TDevice>
    int FeatExtract<TDevice>::outputBufPtrBias(const int timeStepTimesParallel,
						  const int nnState)
    {
	if (this->getSaveMemoryFlag()){
	    return timeStepTimesParallel * this->size();
	}else{
	    return 0;
	}
    }	

    template <typename TDevice>
    void FeatExtract<TDevice>::clearAllBuffers()
    {
	this->clearOutputBuffer();
    }

    template <typename TDevice>
    void FeatExtract<TDevice>::resizeAllBuffers(const int timeLength)
    {
	this->resizeOutputBuffer(timeLength * this->size());
    }


    
    template class FeatExtract<Cpu>;
    template class FeatExtract<Gpu>;
    
}
