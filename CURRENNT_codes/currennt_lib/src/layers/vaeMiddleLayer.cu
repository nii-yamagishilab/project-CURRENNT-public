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

#ifdef _MSC_VER
#   pragma warning (disable: 4244) // thrust/iterator/iterator_adaptor.h(121): warning C4244: '+=' : conversion from '__int64' to 'int', possible loss of data
#endif

#include "vaeMiddleLayer.hpp"

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
#include <thrust/random.h>

#include <sstream>
#include <fstream>
#include <cmath>

namespace internal{
namespace {

    struct FrameNum
    {
	int  featureDim;
	const char *patTypes;
	const real_t *nPara;
	__host__ __device__ real_t operator() (const thrust::tuple<const real_t&, const int&> &t) const
	{
	    const int timeStep = t.get<1>() / featureDim;
	    const int dimIndex = t.get<1>() % featureDim;
	    
	    if (patTypes[timeStep] == PATTYPE_NONE){
		return 0;		
	    }else{
		
		real_t std  = nPara[timeStep * featureDim * 2 + dimIndex + featureDim];
		real_t mean = nPara[timeStep * featureDim * 2 + dimIndex];
		if (std == 0 && mean == 0){
		    // When the input is set to zero, this timestep is assumed to be dummy
		    return 0;
		}else{
		    return 1;
		}
	    }
	}
    };

    /*
    struct CopyManiFoldData
    {
	const char   *patTypes;
	const real_t *dataBuffer;
	int           outDim;
	__host__ __device__ void operator() (const thrust::tuple<real_t&, const int&> &t) const
	{
	    const int timeStep = t.get<1>() / 2;
	    const int dimIndex = t.get<1>() % 2;
	    
	    if (patTypes[timeStep] == PATTYPE_NONE)
		t.get<0>() = 0.0;
	    else
		t.get<0>() = dataBuffer[outDim * timeStep + dimIndex];
	}
	};*/


    struct noiseRepeat
    {
	int           noiseDim;
	real_t       *noiseSrc;
	const char   *patTypes;
	
	__host__ __device__ void operator() (const thrust::tuple<real_t&, const int&> &t) const
	{
	    const int timeStep = t.get<1>() / noiseDim;
	    const int dimIndex = t.get<1>() % noiseDim;
	    
	    if (patTypes[timeStep] == PATTYPE_NONE)
		t.get<0>() = 0.0;
	    else
		t.get<0>() = noiseSrc[dimIndex];
	}
    };

    
    struct genNoise
    {
	float a, b;
	int   seed;
	
	__host__ __device__
	genNoise(float _a=0.f, float _b=1.f, int _seed=123) : a(_a), b(_b), seed(_seed) {};

	__host__ __device__
	float operator()(const unsigned int n) const
	{
	    thrust::default_random_engine rng(seed);
	    thrust::normal_distribution<float> dist(a, b);
	    rng.discard(n);
	    return dist(rng);
	}
    };

    struct vaeNoiseNontransform
    {
	int     noiseDim;
	real_t *nPara;     // mean and std of each frame
	
	const real_t *stdNoise;  // noise from N(0, 1)
	const char   *patTypes;  
	
	__host__ __device__ void operator() (const thrust::tuple<real_t&, const int&> &t) const
	{
	    const int timeStep = t.get<1>() / noiseDim;
	    const int dimIndex = t.get<1>() % noiseDim;
	    if (patTypes[timeStep] == PATTYPE_NONE){
		t.get<0>() = 0;
	    }else{
		if (nPara[timeStep*noiseDim*2+dimIndex+noiseDim] == 0 &&
		    nPara[timeStep * noiseDim * 2 + dimIndex]    == 0)
		    t.get<0>() = 0.0;
		else
		    t.get<0>() = stdNoise[timeStep * noiseDim + dimIndex];
		
	    }
	}
    };

    struct vaeNoiseTransform
    {
	int     noiseDim;
	real_t *nPara;     // mean and std of each frame
	
	const real_t *stdNoise;  // noise from N(0, 1)
	const char   *patTypes;  
	
	__host__ __device__ void operator() (const thrust::tuple<real_t&, const int&> &t) const
	{
	    const int timeStep = t.get<1>() / noiseDim;
	    const int dimIndex = t.get<1>() % noiseDim;
	    if (patTypes[timeStep] == PATTYPE_NONE){
		t.get<0>() = 0;
	    }else{
		// retrieve mean and std
		// Note: std = log(exp(raw_std) + 1), softplus function for std
		real_t std  =  helpers::safeLog(
				helpers::safeExp(nPara[timeStep*noiseDim*2+dimIndex+noiseDim])+1.0);
		real_t mean = nPara[timeStep * noiseDim * 2 + dimIndex];

		if (nPara[timeStep*noiseDim*2+dimIndex+noiseDim] == 0 &&
		    mean == 0){
		    // the previous input is a all zero vector
		    // (probably from the Operator last shot)
		    t.get<0>() = 0.0;
		}else{
		    // transform noise
		    t.get<0>() = std * stdNoise[timeStep * noiseDim + dimIndex] + mean;
		}
		// save std
		// nPara[timeStep * noiseDim * 2 + dimIndex + noiseDim] = std;
	    }
	}
    };

    struct vaeKLDivergence
    {
	int featureDim;
	const real_t *nPara;     // mean and std of each frame
	const char   *patTypes;  
	
	__host__ __device__ real_t operator() (const thrust::tuple<const real_t&, const int&> &t) const
	{
	    const int timeStep = t.get<1>() / featureDim;
	    const int dimIndex = t.get<1>() % featureDim;
	    
	    if (patTypes[timeStep] == PATTYPE_NONE){
		return 0;		
	    }else{
		
		real_t std  = nPara[timeStep * featureDim * 2 + dimIndex + featureDim];
		real_t mean = nPara[timeStep * featureDim * 2 + dimIndex];
		if (std == 0 && mean == 0){
		    // When the input is set to zero, this timestep is assumed to be dummy
		    return 0;
		}else{
		    std = helpers::safeLog(helpers::safeExp(std) + 1.0);
		    return (helpers::safeLog(std) * 2.0 + 1.0 - mean * mean - std * std);
		}
	    }
	}

    };

    struct vaeKLGradient
    {
	int noiseDim;
	const char   *patTypes;
	const real_t *noise;
	const real_t *nPara;
	const real_t *grad;
	
	__host__ __device__ void operator() (const thrust::tuple<real_t&, const int&> &t) const
	{
	    const int numIndex = t.get<1>();
	    const int timeStep = numIndex / (2 * noiseDim);
	    const int dimIndex = numIndex % (2 * noiseDim);
	    
	    if (patTypes[timeStep] == PATTYPE_NONE){
		t.get<0>() = 0.0;
		
	    }else{
		
		if (dimIndex >= noiseDim){
		    // std part
		    real_t std  = nPara[numIndex];
		    real_t mean = nPara[numIndex - noiseDim];
		    
		    if (std == 0 && mean == 0){
			t.get<0>() = 0;
		    }else{
			int    noiseIndex =  timeStep * noiseDim + dimIndex - noiseDim;
			std = helpers::safeLog(helpers::safeExp(std)+1.0);
			t.get<0>()  = (grad[noiseIndex] * noise[noiseIndex] + std - 1.0 / std) /
			    (1.0+helpers::safeExp(-1 * nPara[numIndex]));
		    }
		}else{
		    // mean part
		    real_t mean  = nPara[numIndex];
		    real_t std   = nPara[numIndex + noiseDim];
		    
		    if (std == 0 && mean == 0){
			t.get<0>() = 0;
		    }else{
			int    noiseIndex =  timeStep * noiseDim + dimIndex;
			t.get<0>()  = grad[noiseIndex]  + nPara[numIndex];
		    }
		}
	    }
	}
    };
    

}
}



namespace layers{
    
    template <typename TDevice>
    VaeMiddleLayer<TDevice>::VaeMiddleLayer(const helpers::JsonValue &layerChild,
					    const helpers::JsonValue &weightsSection,
					    Layer<TDevice> &precedingLayer,
					    int             maxSeqLength,
					    int             layerID)
	: TrainableLayer<TDevice> (layerChild, weightsSection, 0, 0,
			      precedingLayer, 
			      maxSeqLength,
			      layerID)
	, m_noiseStd    (1.0)
	, m_noiseMean   (0.0)
	, m_noiseRepeat (0)
    {
	m_noiseStd    = (layerChild->HasMember("noiseStd") ? 
			 static_cast<real_t>((*layerChild)["noiseStd"].GetDouble()) :  1.0);
	m_noiseMean   = (layerChild->HasMember("noiseMean") ? 
			 static_cast<real_t>((*layerChild)["noiseMean"].GetDouble()) : 0.0);
	m_noiseRepeat = (layerChild->HasMember("noiseRepeat") ? 
			 static_cast<real_t>((*layerChild)["noiseRepeat"].GetInt()) : 0);

	m_noiseInput.resize(this->size() * (this->precedingLayer().outputs().size() /
					    this->precedingLayer().size()), 0.0);


	// Option to use the vae during inference time (not used anymore)
	m_vaeUsageOpt = Configuration::instance().vaePlotManifold();

	if (m_vaeUsageOpt == 1){
	    printf("\n\tTurn off sampling from VAE distribution");
	    m_noiseStd = 0.0;
	}
	
	// Check validity
	if (precedingLayer.size() != (this->size() * 2)){
	    printf("Layer size: previous %d, this layer %d", precedingLayer.size(), this->size());
	    throw std::runtime_error("vae's layer size should = 1/2 of previous layer size");
	}	
	printf("\n\tVAE interface between encoder and decoder");

	if (m_noiseRepeat) printf("\n\tVAE noise repeat across frames");
	
	// Note: in vaeNoiseTransform, it is assumed that, when the input mean and variance
	//       parameters are zero, the output noise should be set to zero
    }

    template <typename TDevice>
    VaeMiddleLayer<TDevice>::~VaeMiddleLayer()
    {
    }

    template <typename TDevice>
    void VaeMiddleLayer<TDevice>::loadSequences(const data_sets::DataSetFraction &fraction,
						const int nnState)
    {
	Layer<TDevice>::loadSequences(fraction, nnState);
		
	// Block #1
	// Directly load the mean & std from output
	if (Configuration::instance().vaeCodeInputDir().size()){
	    if (fraction.numSequences() > 1)
		throw std::runtime_error("Please turn off parallel mode");
	    std::string fileName = Configuration::instance().vaeCodeInputDir() + "/" +
		fraction.seqInfo(0).seqTag + ".bin";
	    cpu_real_vector latentData;
	    int numEle = misFuncs::ReadRealData(fileName, latentData);
	    if (numEle % this->size() == 0 && numEle / this->size() == this->curMaxSeqLength()){
		thrust::copy(latentData.begin(), latentData.begin() + numEle,
			     this->outputs().begin());
	    }else{
		printf("\n\tUnable to load external code from %s ",
		       Configuration::instance().vaeCodeInputDir().c_str());
	    }
	}
    }
    
    template <typename TDevice>
    real_t VaeMiddleLayer<TDevice>::vaeKLD()
    {
	// calculate the KL divergence
	real_t kld = 0.0;
	{{
	    int n =this->curMaxSeqLength() * this->parallelSequences() * this->size();
	    
	    // count the number of valid frames
	    int frameNum = 0;
	    //if (this->parallelSequences()>1){
	    internal::FrameNum fn3;
	    fn3.featureDim= this->size();
	    fn3.patTypes  = helpers::getRawPointer(this->patTypes());
	    fn3.nPara     = helpers::getRawPointer(this->precedingLayer().outputs());
	    frameNum      = thrust::transform_reduce(
				thrust::make_zip_iterator(
				   thrust::make_tuple(this->precedingLayer().outputs().begin(), 
						      thrust::counting_iterator<int>(0))),
				thrust::make_zip_iterator(
				   thrust::make_tuple(this->precedingLayer().outputs().begin() + n, 
						      thrust::counting_iterator<int>(0) + n)),
				fn3,
				(real_t)0,
				thrust::plus<real_t>());
	    frameNum = frameNum / this->size();
	    //}else{
	    //frameNum = this->curMaxSeqLength();
	    //}

	    // compute the KLD divergence
	    internal::vaeKLDivergence fn2;
	    fn2.featureDim = this->size();
	    fn2.nPara      = helpers::getRawPointer(this->precedingLayer().outputs());
	    fn2.patTypes   = helpers::getRawPointer(this->patTypes());
	    	    
	    kld = thrust::transform_reduce(
		         thrust::make_zip_iterator(
			     thrust::make_tuple(this->m_noiseInput.begin(), 
						thrust::counting_iterator<int>(0))),
		         thrust::make_zip_iterator(
			     thrust::make_tuple(this->m_noiseInput.begin() + n, 
						thrust::counting_iterator<int>(0) + n)),
			 fn2,
			 (real_t)0,
			 thrust::plus<real_t>());

	    kld = kld * (-0.5) / frameNum;	    
	}}
	return kld;
    }

    template <typename TDevice>
    void VaeMiddleLayer<TDevice>::computeForwardPass(const int nnState)
    {
	// Block #2
	if (Configuration::instance().vaeCodeInputDir().size())
	    return;

	// Generating standard noise
	int timeLength = this->curMaxSeqLength() * this->parallelSequences();
	
	thrust::counting_iterator<unsigned int> index_sequence_begin(0);
	thrust::transform(index_sequence_begin,
			  index_sequence_begin + timeLength * this->size(),
			  m_noiseInput.begin(),
			  internal::genNoise(m_noiseMean, m_noiseStd,
					     (int)(misFuncs::GetRandomNumber()*10000.0)));
	if (m_noiseRepeat){
	    internal::noiseRepeat fn;
	    fn.noiseDim = this->size();
	    fn.noiseSrc = helpers::getRawPointer(m_noiseInput);
	    fn.patTypes = helpers::getRawPointer(this->patTypes());
	    
	    int n = timeLength * this->size();
	    thrust::for_each(
               thrust::make_zip_iterator(
		  thrust::make_tuple(m_noiseInput.begin(),
				     thrust::counting_iterator<int>(0))),
	       thrust::make_zip_iterator(
		  thrust::make_tuple(m_noiseInput.begin()              + n,
				     thrust::counting_iterator<int>(0) + n)),
	       fn);
	}

	// Generate the output noise
	{{
	    internal::vaeNoiseTransform fn1;
	    fn1.noiseDim   = this->size();
	    fn1.nPara      = helpers::getRawPointer(this->precedingLayer().outputs());
	    fn1.stdNoise   = helpers::getRawPointer(this->m_noiseInput);
	    fn1.patTypes   = helpers::getRawPointer(this->patTypes());
	
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
	
	// Block #3
    }

    template <typename TDevice>
    void VaeMiddleLayer<TDevice>::computeForwardPass(const int timeStep, const int nnState)
    {
	if (this->precedingLayer().getSaveMemoryFlag()){
	    throw std::runtime_error("The layer before vae is reduced in mem");
	}

	// Block #2
	
	if (timeStep == 0 && Configuration::instance().vaeCodeInputDir().size() == 0){
	    // If this is the generation stage and external latent variable is not provided
	    
	    // Generating standard noise ()
	    int timeLength = this->curMaxSeqLength() * this->parallelSequences();
	
	    thrust::counting_iterator<unsigned int> index_sequence_begin(0);
	    thrust::transform(index_sequence_begin,
			      index_sequence_begin + timeLength * this->size(),
			      m_noiseInput.begin(),
			      internal::genNoise(m_noiseMean, m_noiseStd,
						 (int)(misFuncs::GetRandomNumber()*10000.0)));

	    if (m_noiseRepeat){
		internal::noiseRepeat fn;
		fn.noiseDim = this->size();
		fn.noiseSrc = helpers::getRawPointer(m_noiseInput);
		fn.patTypes = helpers::getRawPointer(this->patTypes());
		
		int n = timeLength * this->size();
		thrust::for_each(
		  thrust::make_zip_iterator(
		    thrust::make_tuple(m_noiseInput.begin(),
				       thrust::counting_iterator<int>(0))),
		  thrust::make_zip_iterator(
		    thrust::make_tuple(m_noiseInput.begin()              + n,
				       thrust::counting_iterator<int>(0) + n)),
		  fn);
	    }
	    
	    {{
	    	internal::vaeNoiseNontransform fn1;
		fn1.noiseDim   = this->size();
		fn1.nPara      = helpers::getRawPointer(this->precedingLayer().outputs());
		fn1.stdNoise   = helpers::getRawPointer(this->m_noiseInput);
		fn1.patTypes   = helpers::getRawPointer(this->patTypes());
		
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

	// Block #4
    }

    template <typename TDevice>
    void VaeMiddleLayer<TDevice>::computeBackwardPass(const int nnState)
    {

	// Generating standard noise
	int timeLength = this->curMaxSeqLength() * this->parallelSequences();

	{{
	    internal::vaeKLGradient fn1;
	    fn1.noiseDim   = this->size();
	    fn1.patTypes   = helpers::getRawPointer(this->patTypes());
	    fn1.noise      = helpers::getRawPointer(this->m_noiseInput);
	    fn1.nPara      = helpers::getRawPointer(this->precedingLayer().outputs());
	    fn1.grad       = helpers::getRawPointer(this->outputErrors());

	    int n = timeLength * this->size() * 2;
	    thrust::for_each(
               thrust::make_zip_iterator(
		thrust::make_tuple(this->precedingLayer().outputErrors().begin(),
				   thrust::counting_iterator<int>(0))),
	       thrust::make_zip_iterator(
		thrust::make_tuple(this->precedingLayer().outputErrors().begin() + n,
				   thrust::counting_iterator<int>(0) + n)),
	       fn1);
	}}
    }

    template <typename TDevice>
    const std::string& VaeMiddleLayer<TDevice>::type() const
    {
        static const std::string s("vae");
        return s;
    }

    template <typename TDevice>
    void VaeMiddleLayer<TDevice>::exportLayer(const helpers::JsonValue &layersArray, 
						 const helpers::JsonAllocator &allocator) const
    {
	TrainableLayer<TDevice>::exportLayer(layersArray, allocator);

	(*layersArray)[layersArray->Size() - 1].AddMember("noiseStd",    m_noiseStd, allocator);
	(*layersArray)[layersArray->Size() - 1].AddMember("noiseMean",   m_noiseMean, allocator);
	(*layersArray)[layersArray->Size() - 1].AddMember("noiseRepeat", m_noiseRepeat, allocator);
	
    }


    
    template class VaeMiddleLayer<Cpu>;
    template class VaeMiddleLayer<Gpu>;

}


/*
Block #1
	if (nnState == NN_STATE_GENERATION_STAGE && this->size() == 2 && m_vaeUsageOpt==1){
	    printf("Plot manifold");
	    throw std::runtime_error("Not supported now");
	    
	    real_vector tmp(fraction.outputs().size());
	    thrust::copy(fraction.outputs().begin(), fraction.outputs().end(), tmp.begin());

	    // This part is dirty
	    // Manifold code (2-dim) is put in first 2 dimensions of output vectors
	    // So, just copy the first 2-dim from the fraction.outputs to this->outputs()
	    // 
	    internal::CopyManiFoldData fn1;
	    fn1.patTypes   = helpers::getRawPointer(this->patTypes());
	    
	    fn1.dataBuffer = helpers::getRawPointer(tmp);
	    fn1.outDim     = fraction.outputPatternSize();
	    
	    int n = this->curMaxSeqLength() * this->parallelSequences() * this->size();
	    
	    thrust::for_each(
               thrust::make_zip_iterator(
		  thrust::make_tuple(this->outputs().begin(),
				     thrust::counting_iterator<int>(0))),
	       thrust::make_zip_iterator(
		  thrust::make_tuple(this->outputs().begin()           + n,
				     thrust::counting_iterator<int>(0) + n)),
	       fn1);
	    
	}

Block #2
	if (nnState == NN_STATE_GENERATION_STAGE && this->size() == 2 && m_vaeUsageOpt == 1){
	    printf("Plot manifold");
	    // No need to generate anything; data has been loaded in loadSequences()
	    return;
	}

	if (nnState == NN_STATE_GENERATION_STAGE && this->size() == 2 && m_vaeUsageOpt == 1){
	    printf("Plot manifold");
	    // No need to generate anything; data has been loaded in loadSequences()
	    return;
	}


Block #3
	if (nnState == NN_STATE_GENERATION_STAGE && m_vaeUsageOpt == 2){
	    // no transformation
	    internal::vaeNoiseNontransform fn1;
	    fn1.noiseDim   = this->size();
	    fn1.nPara      = helpers::getRawPointer(this->precedingLayer().outputs());
	    fn1.stdNoise   = helpers::getRawPointer(this->m_noiseInput);
	    fn1.patTypes   = helpers::getRawPointer(this->patTypes());

	    int n = timeLength * this->size();
	    thrust::for_each(
               thrust::make_zip_iterator(
		  thrust::make_tuple(this->outputs().begin(),
				     thrust::counting_iterator<int>(0))),
	       thrust::make_zip_iterator(
		  thrust::make_tuple(this->outputs().begin()           + n,
				     thrust::counting_iterator<int>(0) + n)),
	       fn1);
	    
	}else{
	    // Transform and generate the noise
	    internal::vaeNoiseTransform fn1;
	    fn1.noiseDim   = this->size();
	    fn1.nPara      = helpers::getRawPointer(this->precedingLayer().outputs());
	    fn1.stdNoise   = helpers::getRawPointer(this->m_noiseInput);
	    fn1.patTypes   = helpers::getRawPointer(this->patTypes());

	    int n = timeLength * this->size();
	    thrust::for_each(
               thrust::make_zip_iterator(
		  thrust::make_tuple(this->outputs().begin(),
				     thrust::counting_iterator<int>(0))),
	       thrust::make_zip_iterator(
		  thrust::make_tuple(this->outputs().begin()           + n,
				     thrust::counting_iterator<int>(0) + n)),
	       fn1);
	}


Block #4


	if (nnState == NN_STATE_GENERATION_STAGE && m_vaeUsageOpt == 2){
	    // copy the noise (all frames) to the output
	    // no need to loop over time to generate noise frame by frame
	    {{
	    internal::vaeNoiseNontransform fn1;
	    fn1.noiseDim   = this->size();
	    fn1.nPara      = helpers::getRawPointer(this->precedingLayer().outputs());
	    fn1.stdNoise   = helpers::getRawPointer(this->m_noiseInput);
	    fn1.patTypes   = helpers::getRawPointer(this->patTypes());

	    //int n = timeLength * this->size();
	    thrust::for_each(
               thrust::make_zip_iterator(
		thrust::make_tuple(this->outputs().begin()           + timeStep * this->size(),
				   thrust::counting_iterator<int>(0) + timeStep * this->size())),
	       thrust::make_zip_iterator(
	        thrust::make_tuple(this->outputs().begin()           +(timeStep+1) * this->size(),
				   thrust::counting_iterator<int>(0) +(timeStep+1) * this->size())),
	       fn1);
	    }}
	}else{
	    // Transform and generate the noise
	    {{
	    internal::vaeNoiseTransform fn1;
	    fn1.noiseDim   = this->size();
	    fn1.nPara      = helpers::getRawPointer(this->precedingLayer().outputs());
	    fn1.stdNoise   = helpers::getRawPointer(this->m_noiseInput);
	    fn1.patTypes   = helpers::getRawPointer(this->patTypes());

	    //int n = timeLength * this->size();
	    thrust::for_each(
               thrust::make_zip_iterator(
		thrust::make_tuple(this->outputs().begin()           + timeStep * this->size(),
				   thrust::counting_iterator<int>(0) + timeStep * this->size())),
	       thrust::make_zip_iterator(
	        thrust::make_tuple(this->outputs().begin()           +(timeStep+1) * this->size(),
				   thrust::counting_iterator<int>(0) +(timeStep+1) * this->size())),
	       fn1);
	    }}
	}

*/
