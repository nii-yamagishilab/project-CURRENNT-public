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

#include "MiddleOutputLayer.hpp"

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
    
    /******************************************
     Utilities
    ******************************************/
    // copy segment
    /*
    struct CopyPartMiddleOutput
    {
	const real_t *source;
	real_t *target;

	int srcDim;
	int srcS;     // the first dimension to be copied in source stream

	int copyDim;  // dimension of the data to be copied

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
	    
	    target[timeIdx * tarDim + tarS + dimIdx] = source[timeIdx * srcDim + srcS + dimIdx];
	}
	}; */


    struct InterweaveOutput
    {
	int     featDim;
	real_t *src1;
	real_t *src2;
	real_t *tart;
	real_t *random;

	bool    flagForFeatMatch; // dirty part 
	const char *patTypes;

	__host__ __device__ void operator() (const thrust::tuple<real_t&, int> &t) const
	{
	    int outputIdx = t.get<1>();
	    int timeIdx   = outputIdx / featDim;

	    // skip dummy frame (for parallel sentence processing)
	    if (patTypes != NULL && patTypes[timeIdx] == PATTYPE_NONE)
		tart[outputIdx] = 0.0;

	    if (src2){
		// this is used in computeForwardPass
		//  1. Natural data is used in discrminator training or feature matching
		//  2. Otherwise, use the generated data
		if (random[timeIdx] > 0.5 || flagForFeatMatch){
		    tart[outputIdx] = src2[outputIdx];
		}else{
		    tart[outputIdx] = src1[outputIdx];		
		}
	    }else{
		// this is used in computeBackwardPass (no longer used)
		if (random[timeIdx] > 0.5){
		    tart[outputIdx] = 0;
		}else{
		    tart[outputIdx] = src1[outputIdx]; // * -1.0; 
		}
	    }
	}
    };
    
    struct ComputeSseMiddleOutput
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

    struct ComputeOutputErrorWithGAN
    {        
    	int         layerSize;
	real_t      ganRatio;
	real_t      ganGradMag;
	
    	const char *patTypes;
	
    	__host__ __device__ real_t operator() (const thrust::tuple<const real_t&,
					       const real_t&, const real_t&, int> &t) const        
    	{
	    // unpack the tuple
	    real_t actualOutput = t.get<0>();
	    real_t targetOutput = t.get<1>();
	    real_t disGradient  = t.get<2>(); // gradient from the discriminator
	    int    outputIdx    = t.get<3>();
	    // calculate the pattern index
	    int patIdx = outputIdx / layerSize;
	    // check if the pattern is a dummy
	    if (patTypes[patIdx] == PATTYPE_NONE)
		return 0;
	    // calculate the error
	    real_t error = ((1.0 - ganRatio) * (actualOutput - targetOutput) +
			    ganGradMag * ganRatio * disGradient);
	    return error;
	}
    };


    struct loadExternalData
    {
	int  featureDim;
	int  paralNum;
	int  maxFeatureLength;
	int  resolution;
	
	const real_t *sourceData;
	const real_t *frameIndex;
	const real_t *contextMV;
	const char   *patTypes;
	
	__host__ __device__ void operator() (const thrust::tuple<real_t&, int> &t) const
	{
	    int dimIdx  = t.get<1>() % featureDim;
	    int timeIdx = t.get<1>() / featureDim;
	    int paralIdx= timeIdx    % paralNum;

	    if (patTypes[timeIdx] == PATTYPE_NONE)
		t.get<0>() = 0.0;
	    else{
		int featIdx = frameIndex[timeIdx * resolution] * paralNum + paralIdx;
		if (frameIndex[timeIdx * resolution] >= maxFeatureLength){
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
    MiddleOutputLayer<TDevice>::MiddleOutputLayer(const helpers::JsonValue &layerChild,
						  Layer<TDevice> &precedingLayer,
						  int maxSeqLength,
						  int layerID)
	: PostOutputLayer<TDevice> (layerChild, precedingLayer, precedingLayer.size(),
				    maxSeqLength, layerID, true)
	, m_natPriDim (-1)
	, m_natSecDim (-1)
	, m_state     (UNKNOWN)
	, m_ganRatio  (0.001)  
    {
	m_dataOutputDim     = (layerChild->HasMember("dataOutputDim")) ? 
			       (*layerChild)["dataOutputDim"].GetInt() : (-1);
	m_ganRatio          = (layerChild->HasMember("ganRatio") ? 
			       static_cast<real_t>((*layerChild)["ganRatio"].GetDouble()) :0.8);
	m_ganGradEnhance    = (layerChild->HasMember("ganGradMag") ? 
			       static_cast<real_t>((*layerChild)["ganGradMag"].GetDouble()) :10.0);

	/*m_generatorEpoch    = (layerChild->HasMember("generator_only")) ? 
			       (*layerChild)["generator_only"].GetInt() : (-1);
	m_discriminatorEpoch= (layerChild->HasMember("discriminator_only")) ? 
	(*layerChild)["discriminator_only"].GetInt() : (-1);*/

	if (m_ganRatio > 1.0005 || m_ganRatio < 0.0000){
	    throw std::runtime_error("ganRatio must be within (0, 1]");
	}
	
	if (m_dataOutputDim > 0 && m_dataOutputDim != this->size()){
	    throw std::runtime_error("Error dataOutputDim in middleoutput layer");
	}	
	
	m_natPriDim     = this->size();
	if (m_dataOutputDim > 0) {
	    m_natSecDim     = m_dataOutputDim - m_natPriDim;
	    if (m_natSecDim){
		// This part is here secondary output. It is not used now
		m_natSecTarget.reserve(this->outputs().size() / m_natPriDim * m_natSecDim);
		thrust::fill(m_natSecTarget.begin(), m_natSecTarget.end(), 0.0);
	    }else
		m_natSecTarget.clear();
	}
	m_natPriTarget  = this->outputs();
	
	m_stateRandom.resize(this->outputs().size() / m_natPriDim, 0.0);

	// print the information
	printf("\n\tGAN configure: ganRatio %f, ganGradMag %f\n", m_ganRatio, m_ganGradEnhance);
	printf("\n\tGAN criterion: (1 - ganRatio) * 0.5 * (synthetic - natural) ^ 2 + ");
	printf("ganRatio * ganGradMag * Loss_of_discriminator\n");
	
	if (this->precedingLayer().getSaveMemoryFlag())
	    throw std::runtime_error("layer before MiddleOutput is reduced in mem");

    }

    template <typename TDevice>
    MiddleOutputLayer<TDevice>::~MiddleOutputLayer()
    {
    }

    template <typename TDevice>
    void MiddleOutputLayer<TDevice>::loadSequences(const data_sets::DataSetFraction &fraction,
						   const int nnState)
    {
	Layer<TDevice>::loadSequences(fraction, nnState);

	/*
	if (this->getCurrTrainingEpoch() <= m_generatorEpoch){
	    m_state = GENERATOR_ONLY;
	}else if (this->getCurrTrainingEpoch() <= (m_generatorEpoch + m_discriminatorEpoch)){
	    m_state = DISCRIMINATOR_ONLY;
	}else{
	    m_state = JOINT_TRAIN;
	}*/

	cpu_real_vector tmp(m_stateRandom);
	if (nnState == NN_STATE_GAN_DIS_GENDATA || nnState == NN_STATE_GAN_DIS_NATDATA){
	    // This is discriminator-training epoch
	    m_state = DISCRIMINATOR_ONLY;

	    // Set the random 1/0 sequence to mix the natural and generated output
	    // P > 0.5: use natural   output
	    //    else: use generated output
	    if (nnState == NN_STATE_GAN_DIS_NATDATA){
		thrust::fill(tmp.begin(), tmp.end(), 1.0);
	    }else{
		thrust::fill(tmp.begin(), tmp.end(), 0.0);
	    }
	    //for (int i = 0; i < (this->curMaxSeqLength() * this->parallelSequences()); i++)
	    //tmp[i] = (GetRandomNumber() > 0.5) ? (1.0) : (0.0);
	    
	}else if (nnState == NN_STATE_GAN_GEN || nnState == NN_STATE_GAN_GEN_FEATMAT){
	    
	    // This is the genrator-training epoch
	    m_state = GENERATOR_ONLY;
	    // use all generated output
	    thrust::fill(tmp.begin(), tmp.end(), 0.0);
	    
	}else if (nnState == NN_STATE_GENERATION_STAGE){
	    printf(" for generation");
	    m_state = GENERATOR_ONLY;                  // any value is OK
	    thrust::fill(tmp.begin(), tmp.end(), 0.0); // any value is OK
	    
	}else{
	    throw std::runtime_error("Error unknown nnState");
	}
	m_stateRandom = tmp;
	
	if (this->_outputs().empty() || m_natPriTarget.empty())
	    throw std::runtime_error("Error no output buffer in middleoutput");

	if (fraction.outputPatternSize() == this->size()){
	    thrust::copy(fraction.outputs().begin(), fraction.outputs().end(),
			 m_natPriTarget.begin());
	}else if (fraction.externalOutputSize() == this->size()){
	    this->_dataBuffer().resize(fraction.outputs().size() +
				       fraction.exOutputData().size(), 0.0);
	    thrust::copy(fraction.outputs().begin(),
			 fraction.outputs().end(),
			 this->_dataBuffer().begin());
	    thrust::copy(fraction.exOutputData().begin(),
			 fraction.exOutputData().end(),
			 this->_dataBuffer().begin() + fraction.outputs().size());
	
	    internal::loadExternalData fn1;
	    fn1.featureDim = fraction.externalOutputSize();
	    fn1.paralNum   = this->parallelSequences();
	    fn1.maxFeatureLength = fraction.maxExOutputLength();
	    fn1.sourceData = (helpers::getRawPointer(this->_dataBuffer()) +
			      fraction.outputs().size());
	    fn1.frameIndex = helpers::getRawPointer(this->_dataBuffer());
	    fn1.patTypes   = helpers::getRawPointer(this->patTypes());
	    fn1.contextMV  = ((this->_mvVector().size() ==this->size() * 2)?
			      helpers::getRawPointer(this->_mvVector()) : NULL);
	    fn1.resolution = this->getResolution();
		    
	    int n = this->curMaxSeqLength() * this->parallelSequences() * this->size();
	    thrust::for_each(
			thrust::make_zip_iterator(
				thrust::make_tuple(this->_outputs().begin(),
						   thrust::counting_iterator<int>(0))),
			thrust::make_zip_iterator(
				thrust::make_tuple(this->_outputs().begin()           + n,
						   thrust::counting_iterator<int>(0) + n)),
			fn1);

	}else{
	    throw std::runtime_error("MidlleOutputLayer size is not equal to the data dim");
	}
	
	/*
	// copy the output corresponding to this layer
	{{
	   
	   internal::CopyPartMiddleOutput fn;
	   fn.source   = helpers::getRawPointer(fraction.outputs());
	   fn.target   = helpers::getRawPointer(m_natPriTarget);
	   fn.srcDim   = this->m_dataOutputDim;
	   fn.srcS     = 0;
	   fn.copyDim  = m_natPriDim;
	   fn.tarDim   = m_natPriDim;
	   fn.tarS     = 0;
	   fn.patTypes = helpers::getRawPointer(this->patTypes());
	   
	   int n = this->curMaxSeqLength() * this->parallelSequences() * m_natPriDim;
	   thrust::for_each(
		thrust::make_zip_iterator(
			thrust::make_tuple(this->m_natPriTarget.begin(), 
					   thrust::counting_iterator<int>(0))),
		thrust::make_zip_iterator(
			thrust::make_tuple(this->m_natPriTarget.begin() + n, 
					   thrust::counting_iterator<int>(0) + n)),
		fn);
	   
	   //
	   if (m_natSecDim > 0){
	       fn.source   = helpers::getRawPointer(fraction.outputs());
	       fn.target   = helpers::getRawPointer(m_natSecTarget);
	       fn.srcDim   = this->m_dataOutputDim;
	       fn.srcS     = m_natPriDim;
	       fn.copyDim  = m_natSecDim;
	       fn.tarDim   = m_natSecDim;
	       fn.tarS     = 0;
	       fn.patTypes = helpers::getRawPointer(this->patTypes());
	       
	       n = this->curMaxSeqLength() * this->parallelSequences() * m_natSecDim;
	       thrust::for_each(
		thrust::make_zip_iterator(
			thrust::make_tuple(this->m_natSecTarget.begin(), 
					   thrust::counting_iterator<int>(0))),
		thrust::make_zip_iterator(
			thrust::make_tuple(this->m_natSecTarget.begin() + n, 
					   thrust::counting_iterator<int>(0) + n)),
		fn);
	   }
	   }}
	*/
	
	// 
    }

    template <typename TDevice>
    real_t MiddleOutputLayer<TDevice>::calculateError()
    {
	// by default, use the sse criterion
	//if (m_state == GEN_OUTPUT_EVALUATE){
	{{
	    internal::ComputeSseMiddleOutput fn;
	    fn.layerSize = this->size();
	    fn.patTypes  = helpers::getRawPointer(this->patTypes());

	    int n = this->curMaxSeqLength() * this->parallelSequences() * this->size();

	    real_t mse = (real_t)0.5 * thrust::transform_reduce(
                thrust::make_zip_iterator(
			thrust::make_tuple(m_natPriTarget.begin(),
					   this->_actualOutputs().begin(),
					   thrust::counting_iterator<int>(0))),
		thrust::make_zip_iterator(
			thrust::make_tuple(m_natPriTarget.begin()+n,
					   this->_actualOutputs().begin()+n,
					   thrust::counting_iterator<int>(0)+n)),
		fn,
		(real_t)0,
		thrust::plus<real_t>());
	    return mse;
	}}
	
    }

    template <typename TDevice>
    void MiddleOutputLayer<TDevice>::computeForwardPass(const int nnState)
    {
	internal::InterweaveOutput fn;
	fn.src1     = helpers::getRawPointer(this->precedingLayer().outputs());
	fn.src2     = helpers::getRawPointer(m_natPriTarget);
	fn.featDim  = m_natPriDim;
	    
	fn.tart     = helpers::getRawPointer(this->outputs());
	fn.random   = helpers::getRawPointer(m_stateRandom);
	fn.patTypes = helpers::getRawPointer(this->patTypes());

	// Dirty code: when feature matching is used,
	// this function should load natural daa
	fn.flagForFeatMatch = (nnState == NN_STATE_GAN_GEN_FEATMAT);
	
	int n = this->curMaxSeqLength() * this->parallelSequences() * m_natPriDim;
	thrust::for_each(
		thrust::make_zip_iterator(
			thrust::make_tuple(this->outputs().begin(), 
					   thrust::counting_iterator<int>(0))),
		thrust::make_zip_iterator(
			thrust::make_tuple(this->outputs().begin() + n, 
					   thrust::counting_iterator<int>(0) + n)),
		fn);
	/*
	if (m_state == JOINT_TRAIN || m_state == DISCRIMINATOR_ONLY){
	    internal::InterweaveOutput fn;
	    fn.src1     = helpers::getRawPointer(this->precedingLayer().outputs());
	    fn.src2     = helpers::getRawPointer(m_natPriTarget);
	    fn.featDim  = m_natPriDim;
	    
	    fn.tart     = helpers::getRawPointer(this->outputs());
	    fn.random   = helpers::getRawPointer(m_stateRandom);
	    fn.patTypes = helpers::getRawPointer(this->patTypes());
	    
	    int n = this->curMaxSeqLength() * this->parallelSequences() * m_natPriDim;
	    thrust::for_each(
		thrust::make_zip_iterator(
			thrust::make_tuple(this->outputs().begin(), 
					   thrust::counting_iterator<int>(0))),
		thrust::make_zip_iterator(
			thrust::make_tuple(this->outputs().begin() + n, 
					   thrust::counting_iterator<int>(0) + n)),
		fn);
	}else if(m_state == GENERATOR_ONLY){
	    thrust::fill(this->outputs().begin(), this->outputs().end(), 0.0);
	}else{
	    throw std::runtime_error("Unknown trainng state of middleoutputlayer");
	}
	*/
    }

    template <typename TDevice>
    void MiddleOutputLayer<TDevice>::computeForwardPass(const int timeStep, const int nnState)
    {
	/*
	int st = timeStep * this->parallelSequences() * this->size();
	int et = st + this->parallelSequences() * this->size();
	*/
	throw std::runtime_error("GAN is not implemented for feedback structure");
    }

    template <typename TDevice>
    void MiddleOutputLayer<TDevice>::computeBackwardPass(const int nnState)
    {
	/*
	if (m_state == JOINT_TRAIN){
	    
	    internal::InterweaveOutput fn;
	    fn.src1     = helpers::getRawPointer(this->outputErrors());
	    fn.src2     = NULL;
	    fn.featDim  = m_natPriDim;
	    
	    fn.tart     = helpers::getRawPointer(this->precedingLayer().outputErrors());
	    fn.random   = helpers::getRawPointer(m_stateRandom);
	    fn.patTypes = helpers::getRawPointer(this->patTypes());
	    
	    int n = this->curMaxSeqLength() * this->parallelSequences() * m_natPriDim;
	    thrust::for_each(
		thrust::make_zip_iterator(
			thrust::make_tuple(this->outputs().begin(), 
					   thrust::counting_iterator<int>(0))),
		thrust::make_zip_iterator(
			thrust::make_tuple(this->outputs().begin() + n, 
					   thrust::counting_iterator<int>(0) + n)),
		fn);
	    
	}else if (m_state == GENERATOR_ONLY){
	    
	    internal::ComputeOutputErrorFnMiddleOutput fn;
	    fn.layerSize = this->size();
	    fn.patTypes  = helpers::getRawPointer(this->patTypes());

	    int n = this->curMaxSeqLength() * this->parallelSequences() * this->size();

	    thrust::transform(
               thrust::make_zip_iterator(
		thrust::make_tuple(this->precedingLayer().outputs().begin(),
				   this->m_natPriTarget.begin(),
				   thrust::counting_iterator<int>(0))),
	       thrust::make_zip_iterator(
		thrust::make_tuple(this->precedingLayer().outputs().begin() + n,
				   this->m_natPriTarget.begin() + n,
				   thrust::counting_iterator<int>(0) + n)),
	       this->precedingLayer().outputErrors().begin(),
	       fn);
	    
	}else if (m_state == DISCRIMINATOR_ONLY){
	    thrust::fill(this->precedingLayer().outputErrors().begin(),
			 this->precedingLayer().outputErrors().end(), 0.0);
	    // undefined state
	}else{
	    throw std::runtime_error("Unknown trainng state of middleoutputlayer");	    
	    }*/

	if (m_state == DISCRIMINATOR_ONLY){
	    thrust::fill(this->precedingLayer().outputErrors().begin(),
			 this->precedingLayer().outputErrors().end(), 0.0);
	}else if (m_state == GENERATOR_ONLY){
	    internal::ComputeOutputErrorWithGAN fn;
	    fn.layerSize = this->size();
	    fn.patTypes  = helpers::getRawPointer(this->patTypes());
	    fn.ganRatio  = m_ganRatio;
	    fn.ganGradMag= m_ganGradEnhance;
	    int n = this->curMaxSeqLength() * this->parallelSequences() * this->size();
	    
	    thrust::transform(
               thrust::make_zip_iterator(
		thrust::make_tuple(this->precedingLayer().outputs().begin(),
				   this->m_natPriTarget.begin(),
				   this->outputErrors().begin(),
				   thrust::counting_iterator<int>(0))),
	       thrust::make_zip_iterator(
		thrust::make_tuple(this->precedingLayer().outputs().begin() + n,
				   this->m_natPriTarget.begin()             + n,
				   this->outputErrors().begin()             + n,
				   thrust::counting_iterator<int>(0)        + n)),
	       this->precedingLayer().outputErrors().begin(),
	       fn);
	}else{
	    throw std::runtime_error("Unknown trainng state of middleoutputlayer");	    
	}   
    }

    template <typename TDevice>
    const std::string& MiddleOutputLayer<TDevice>::type() const
    {
        static const std::string s("middleoutput");
        return s;
    }


    template <typename TDevice>
    typename MiddleOutputLayer<TDevice>::real_vector& MiddleOutputLayer<TDevice>::secondOutputs()
    {
	// m_stateRandom should be named as 1/0 tags
	// m_stateRandom logs down the true/false of each frame in the current utterance
	// previously, m_stateRandom can be a mixture of 1/0
	// now, m_stateRandom only contains 1 or 0 for one utterance,
	//  without mixing natural and fake data
	return m_stateRandom;
    }

    template <typename TDevice>
    void MiddleOutputLayer<TDevice>::exportLayer(const helpers::JsonValue &layersArray, 
						 const helpers::JsonAllocator &allocator) const
    {
	Layer<TDevice>::exportLayer(layersArray, allocator);
	(*layersArray)[layersArray->Size() - 1].AddMember("dataOutputDim",
							  m_dataOutputDim, allocator);
	(*layersArray)[layersArray->Size() - 1].AddMember("ganRatio",
							  m_ganRatio, allocator);
	(*layersArray)[layersArray->Size() - 1].AddMember("ganGradMag",
							  m_ganGradEnhance, allocator);
    }

    
    template class MiddleOutputLayer<Cpu>;
    template class MiddleOutputLayer<Gpu>;
    
}
