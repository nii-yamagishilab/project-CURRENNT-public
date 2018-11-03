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

#include "SignalGenLayer.hpp"

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


// for since wave generator
#define NN_OPE_FREGEN_F0_LF0  1   // input is mel-logF0 = 1127 * log (1 + F0 / 700)
#define NN_OPE_FREGEN_F0_QF0  2   // input is qF0 (after mel)
#define NN_OPE_FREGEN_F0_PF0  3   // input is normal f0 linear domain

#define NN_SIGGEN_LAYER_MODE_NOISE_ONLY  0   // generate only noise
#define NN_SIGGEN_LAYER_MODE_SINE_SIMPLE 1   // generate a sine without phase match & harmonics
#define NN_SIGGEN_LAYER_MODE_SINE_PHASE  2   // generate a sine with phase match, not no harmonics
#define NN_SIGGEN_LAYER_MODE_SINE_HARMO  3   // generate a sine with phase match and harmonics

#define PI_DEFINITION 3.141592654f

#define NN_SIGGEN_LAYER_NOISE_UNIFORM  1 // uniform noise
#define NN_SIGGEN_LAYER_NOISE_GAUSSIAN 2 // Gaussian noise

#define NN_SIGGEN_NOISE_FLOOR 0.000001
namespace internal{
namespace{
    
    struct genNoise
    {
	float a, b;
	int   seed;
	
	__host__ __device__
	genNoise(float _a=-1.f, float _b=1.f, int _seed=123) : a(_a), b(_b), seed(_seed) {};

	__host__ __device__
	float operator()(const unsigned int n) const
	{
	    thrust::default_random_engine rng(seed);
	    thrust::uniform_real_distribution<float> dist(a, b);
	    rng.discard(n);
	    return dist(rng);
	}
    };

    struct genNoise_Gaussian
    {
	float a, b;
	int   seed;
	
	__host__ __device__
	genNoise_Gaussian(float _a=-1.f, float _b=1.f, int _seed=123) : a(_a), b(_b), seed(_seed) {};

	__host__ __device__
	float operator()(const unsigned int n) const
	{
	    thrust::default_random_engine rng(seed);
	    thrust::normal_distribution<float> dist(a, b);
	    rng.discard(n);
	    return dist(rng);
	}
    };
        
    struct sinWaveGenerator_accum
    {
	int inputRes;
	int outputRes;
	int signalDim;
	int hnmNum;
	int prelayerSize;
	
	int freqDim;
	int freqOpt;
	int seqLength;

	int equalNoiseSinePower;
	
	real_t freqSR;
	real_t qf0Max;
	real_t qf0Min;
	real_t qf0Lev;
	real_t f0M;
	real_t f0S;

	real_t f0Mag;
	real_t addtiveNoiseMag;
	
	int inputShiftTime;
	int outputShiftTime;
	
	int parallel;
	int HmnNum;
	int noNoiseInSince;
	
	real_t *targetData;
	real_t *sourceData;
	real_t *genSignalBuff;
	real_t *signalStatic;
	real_t *HmnBuff;
	real_t *phaseNoise;
	real_t *addtiveNoise;
	
	const char *patTypes;
	
	__host__ __device__ void operator() (const thrust::tuple<real_t&, int> &t) const
	{
	    int dimIdx    = t.get<1>() % signalDim;
	    int paraBlock = t.get<1>() / signalDim;

	    // there are signalDim signals to be generated, among which hnmNum are harmonics
	    int phaseCandNum  = signalDim - hnmNum; 
		
	    int timeReso   = inputRes / outputRes;
	    int timeIdxPhy = 0;

	    // the first phaseCandNum use different initial phase
	    // the rest signals use zero initial phase but different frequency (harmonics)
	    
	    real_t initPhase = ((dimIdx<phaseCandNum)?
				(2.0 * PI_DEFINITION * ((real_t)(dimIdx)/phaseCandNum)) : 0.0);
	    
	    real_t spPhase = initPhase;
	    real_t sigValue = 0.0;

	    real_t delta_1 = 0.0;
	    real_t delta_2 = 0.0;
	    real_t phaseNoiseValue = 0.0;
	    
	    for (int timeIdx = 0; timeIdx < seqLength; timeIdx++){

		timeIdxPhy = timeIdx * parallel + paraBlock;
		
		if (patTypes[timeIdxPhy] == PATTYPE_NONE)
		    continue;
	
		// read and denormalize the raw data
		real_t freq = sourceData[(timeIdxPhy / timeReso - inputShiftTime) * prelayerSize
					 + freqDim] * f0S + f0M;
		
		// for harmonics, multiply the freq
		if (dimIdx >= phaseCandNum)    
		    freq = freq * (dimIdx - phaseCandNum + 2);
		

		// for voiced segment
		if ((freqOpt == NN_OPE_FREGEN_F0_QF0 && freq > 1.0)  ||
		    (freqOpt == NN_OPE_FREGEN_F0_LF0 && freq > 10.0) ||
		    (freqOpt == NN_OPE_FREGEN_F0_PF0 && freq > 10.0)){

		    // convert F0 
		    if (freqOpt == NN_OPE_FREGEN_F0_LF0){
			// LF0 -> F0
			freq = (exp(freq / 1127.0) -1.0 ) * 700.0;
		    }else if (freqOpt == NN_OPE_FREGEN_F0_QF0){
			// QF0 -> LF0
			freq = (freq - 1.0) / (qf0Lev-2.0) * (qf0Max - qf0Min) + qf0Min;
			// LF0 -> F0
			freq = (exp(freq / 1127.0) -1.0 ) * 700.0;
		    }else if (freqOpt == NN_OPE_FREGEN_F0_PF0){
			// input is linear
			//freq = freq;
		    }
		    
		    // get the phase noise
		    if (dimIdx < phaseCandNum){
			phaseNoiseValue = phaseNoise[timeIdxPhy*(hnmNum+1)];
		    }else{
			phaseNoiseValue = phaseNoise[timeIdxPhy*(hnmNum+1) +dimIdx -phaseCandNum];
		    }

		    // accumulate the phase + phase nosise
		    spPhase += (2.0 * PI_DEFINITION * freq / freqSR + phaseNoiseValue);		    
		    if (spPhase > 2.0 * PI_DEFINITION)
			spPhase = spPhase - 2.0 * PI_DEFINITION;
		    else if (spPhase < -2.0 * PI_DEFINITION)
			spPhase = spPhase + 2.0 * PI_DEFINITION;

		    // store the signal value
		    sigValue = sin(spPhase) * f0Mag;
		    if (dimIdx < phaseCandNum)
			genSignalBuff[timeIdxPhy * phaseCandNum + dimIdx] = sigValue;
		    else
			HmnBuff[timeIdxPhy * hnmNum + (dimIdx - phaseCandNum)] = sigValue;

		    // if necessary, set the noise in voiced regions to zero
		    if (noNoiseInSince)
			if (dimIdx < (hnmNum + 1))
			    addtiveNoise[timeIdxPhy * (hnmNum + 1) + dimIdx] = 0.0;
			
		}else{
		    // for unvoiced segment, keep the initial phase, set data = 0
		    spPhase = initPhase;

		    // if necessary adjust the amplitude of noise in unvoiced segment
		    if (equalNoiseSinePower){
			if (dimIdx < (hnmNum + 1)){
			    addtiveNoise[timeIdxPhy * (hnmNum + 1) + dimIdx] =
				addtiveNoise[timeIdxPhy * (hnmNum + 1) + dimIdx] /
				(addtiveNoiseMag) * (f0Mag + addtiveNoiseMag);
			}
		    }
		}

		// accumulate the static for each generated signal
		if (signalStatic && dimIdx < phaseCandNum){
		    // mean
		    delta_1 = sigValue - signalStatic[t.get<1>() * 2];
		    signalStatic[t.get<1>() * 2]   += delta_1 / (timeIdx + 1);
		    // std
		    delta_2 = sigValue - signalStatic[t.get<1>() * 2];
		    signalStatic[t.get<1>() * 2+1] += delta_1 * delta_2;
		    
		    // accumulate the static for target signal (trick, only accumulate in
		    //  the first thread)
		    // and assume target layer dimension is 1
		    if (targetData && dimIdx == 0){

			// mean of target
			delta_1 = targetData[timeIdxPhy] -
			    signalStatic[(signalDim * parallel + paraBlock) * 2];
			signalStatic[(signalDim*parallel+paraBlock)*2] += delta_1/(timeIdx+1);
		    
			// std of target
			delta_2 = targetData[timeIdxPhy] -
			    signalStatic[(signalDim * parallel + paraBlock) * 2];
			signalStatic[(signalDim*parallel+paraBlock)*2+1] += delta_1 * delta_2;
		    }
		}		    
	    }

	    // sqrt
	    if (signalStatic && dimIdx < phaseCandNum){
		signalStatic[t.get<1>() * 2 + 1] = sqrt(signalStatic[t.get<1>() * 2 + 1] /seqLength);
		if (targetData && dimIdx == 0){
		    signalStatic[(signalDim * parallel + paraBlock) * 2 + 1] =
			sqrt(signalStatic[(signalDim * parallel + paraBlock) * 2 + 1] / seqLength);
		}
	    }
	    
	}
	
    };


    struct signalCorr
    {
	int signalDim;
	int hnmNum;
	int seqLength;
	int parallel;
	
	real_t *targetData;
	real_t *genSignalBuff;
	real_t *signalStatic;
	
	const char *patTypes;
	
	__host__ __device__ void operator() (const thrust::tuple<real_t&, int> &t) const
	{
	    int dimIdx    = t.get<1>() % signalDim;
	    int paraBlock = t.get<1>() / signalDim;
	    int phaseCandNum = signalDim - hnmNum;
	    
	    int tmpPos    = signalDim * parallel;
	    int timeIdxPhy = 0;
	    	    
	    t.get<0>() = 0.0;

	    if (dimIdx < phaseCandNum){
		for (int timeIdx = 0; timeIdx < seqLength; timeIdx++){
		    timeIdxPhy = timeIdx * parallel + paraBlock;	
		    if (patTypes[timeIdxPhy] == PATTYPE_NONE)
			continue;
		
		    t.get<0>() += (((genSignalBuff[timeIdxPhy * phaseCandNum + dimIdx] -
				     signalStatic[t.get<1>() * 2])  /
				    signalStatic[t.get<1>() * 2+1]) *
				   ((targetData[timeIdxPhy] -
				     signalStatic[tmpPos * 2 + paraBlock * 2]) /
				    signalStatic[tmpPos * 2 + paraBlock * 2 + 1]))/seqLength;
		}
	    }else{

		// for harmonics, make a fake value 
		t.get<0>() = -2.0;
	    }
	}
    };
    
    struct getBestIndex
    {
	real_t *errorBuff;
	int    signalDim;
	int    hnmNum;
	int    parallel;

	__host__ __device__ void operator() (const thrust::tuple<real_t&, int> &t) const
	{
	    real_t maxValue = -2.0;
	    real_t dimFlag = 0;
	    int phaseCandNum = signalDim - hnmNum;
	    
	    for (int dimIdx = 0; dimIdx < phaseCandNum; dimIdx++){
		if (errorBuff[t.get<1>() * signalDim + dimIdx] > maxValue){
		    maxValue = errorBuff[t.get<1>() * signalDim + dimIdx];
		    dimFlag  = dimIdx;
		}
	    }
	    // store the best index in the first dimension
	    errorBuff[t.get<1>() * signalDim + 0] = dimFlag;
	}
    };

    struct LoadSignal
    {
	int phaseCandNum;
	int hnmNum;
	int parallel;
	
	real_t *genSignalBuff;
	real_t *errorBuff;
	real_t *noise;
	real_t *HmnBuff;
	const char *patTypes;
	
	__host__ __device__ void operator() (const thrust::tuple<real_t&, int> &t) const
	{
	    int signalDim = phaseCandNum + hnmNum;
	    
	    int dimIdx    = t.get<1>() % (1 + hnmNum);
	    int timeBlock = t.get<1>() / (1 + hnmNum);
	    int paraBlock = timeBlock  % parallel;

	    if (patTypes[timeBlock] == PATTYPE_NONE){
		t.get<0>() = 0.0;
	    }else{
		int bestDimIdx = (int)errorBuff[paraBlock * signalDim];		
		if (dimIdx == 0){
		    t.get<0>() = noise[t.get<1>()] +
			genSignalBuff[timeBlock * phaseCandNum + bestDimIdx];
		}else{
		    t.get<0>() = noise[t.get<1>()] +
			HmnBuff[timeBlock * hnmNum + dimIdx - 1];
		}
	    }
	    
	}
    };
	
}    
}


namespace layers{

	
    
    template <typename TDevice>
    SignalGenLayer<TDevice>::SignalGenLayer(const helpers::JsonValue &layerChild,
					    const helpers::JsonValue &weightsSection,
					    Layer<TDevice>           &precedingLayer,
					    int                       maxSeqLength,
					    int                       layerID)
	: TrainableLayer<TDevice>(layerChild, weightsSection, 0, 0,
				  precedingLayer, maxSeqLength, layerID)
	, m_noiseMag   (1.0)
	, m_freqDim    (-1)
	, m_freqOpt    (NN_OPE_FREGEN_F0_LF0)
	, m_freqBins   (0)
	, m_freqHmn    (0)
	, m_freqSR     (-1)
	, m_freqQF0min (0)
	, m_freqQF0max (0)
	, m_freqQF0Lev (0)
	, m_freqDataM  (0)
	, m_freqDataS  (0)
	, m_targetLayer(NULL)
	, m_equalNoiseSinePower (0)
	, m_noiseType  (NN_SIGGEN_LAYER_NOISE_UNIFORM)
    {
	// load options
	this->__loadOpts(layerChild);
	
	// set layer mode
	this->__setLayerMode();

	// allocate memory
	this->__allocateLocalMem();
    }

    template <typename TDevice>
    SignalGenLayer<TDevice>::~SignalGenLayer()
    {
    }
    
    template <typename TDevice>
    void SignalGenLayer<TDevice>::__loadOpts(const helpers::JsonValue &layerChild)
    {
	// Read flags
	m_freqDim = (layerChild->HasMember("frequencyDim")?
		     static_cast<real_t>((*layerChild)["frequencyDim"].GetInt()) :
		     this->precedingLayer().size()-1);
	m_freqOpt = (layerChild->HasMember("frequencyOpt")?
		     static_cast<real_t>((*layerChild)["frequencyOpt"].GetInt()) :
		     NN_OPE_FREGEN_F0_PF0);
	m_freqSR  = (layerChild->HasMember("frequencySR")?
		     static_cast<real_t>((*layerChild)["frequencySR"].GetDouble()) : -1);
	m_freqQF0min = (layerChild->HasMember("frequencyQF0min")?
			static_cast<real_t>((*layerChild)["frequencyQF0min"].GetDouble()):-1);
	m_freqQF0max = (layerChild->HasMember("frequencyQF0max")?
			static_cast<real_t>((*layerChild)["frequencyQF0max"].GetDouble()):-1);
	m_freqQF0Lev = (layerChild->HasMember("frequencyQF0Lev")?
			static_cast<real_t>((*layerChild)["frequencyQF0Lev"].GetDouble()):-1);
	m_freqDataM = (layerChild->HasMember("frequencyF0Mean")?
		       static_cast<real_t>((*layerChild)["frequencyF0Mean"].GetDouble()):0);
	m_freqDataS = (layerChild->HasMember("frequencyF0Std")?
		       static_cast<real_t>((*layerChild)["frequencyF0Std"].GetDouble()):1);
	m_freqSignalMag = (layerChild->HasMember("frequencyF0Mag")?
			   static_cast<real_t>((*layerChild)["frequencyF0Mag"].GetDouble()):1);
	m_freqBins = (layerChild->HasMember("frequencyBins")?
		      static_cast<real_t>((*layerChild)["frequencyBins"].GetInt()) : 0);
	m_freqHmn = (layerChild->HasMember("frequencyHarmonics")?
		     static_cast<real_t>((*layerChild)["frequencyHarmonics"].GetInt()) : 0);
	m_noiseMag = (layerChild->HasMember("frequencyNoiseMag") ? 
		      static_cast<real_t>((*layerChild)["frequencyNoiseMag"].GetDouble()):1.0);
	m_phaseNoiseMag = (layerChild->HasMember("phaseNoiseMag") ? 
			   static_cast<real_t>((*layerChild)["phaseNoiseMag"].GetDouble()):0.0);
	m_equalNoiseSinePower = (layerChild->HasMember("equalNoiseSinePower") ? 
			   static_cast<real_t>((*layerChild)["equalNoiseSinePower"].GetInt()):1.0);
	m_noiseType = (layerChild->HasMember("noiseType") ? 
			   static_cast<real_t>((*layerChild)["noiseType"].GetInt()):
		       NN_SIGGEN_LAYER_NOISE_UNIFORM);

	m_noNoiseInSine = (layerChild->HasMember("noNoiseInSine") ? 
			   static_cast<real_t>((*layerChild)["noNoiseInSine"].GetInt()):0);

	
	const Configuration &config = Configuration::instance();
	if (config.f0dataMean_signalgen() > 0)
	    m_freqDataM = config.f0dataMean_signalgen();
	if (config.f0dataStd_signalgen() > 0)
	    m_freqDataS = config.f0dataStd_signalgen();

	printf("\n\tSource module info:\n");
	printf("\n\tTake the %d-th dimension of previous layer's output as F0:\n", m_freqDim);
	if (m_freqOpt == NN_OPE_FREGEN_F0_LF0)
	    printf("\n\tInput F0 is log-F0. ");
	else if (m_freqOpt == NN_OPE_FREGEN_F0_QF0)
	    printf("\n\tInput F0 is quantized F0. ");
	else if (m_freqOpt == NN_OPE_FREGEN_F0_PF0)
	    printf("\n\tInput F0 is linear F0. ");
	else
	    throw std::runtime_error("Unknown F0 input type (frequencyOpt)");
	printf("Denormalize F0 using mean/std: %f %f.", m_freqDataM, m_freqDataS);
	printf("\n\tSine wave magnitude %f", m_freqSignalMag);
	printf("\n\tSine wave harmonics %d", m_freqHmn);
	if (m_freqBins)
	    printf("\n\tSine wave used phase match in training");
	if (m_noNoiseInSine)
	    printf("\n\tSine wave will have no additive noise");

	// otherwise, m_equalNoiseSinePower will lead to nan in sinWaveGenerator_accum
	if (m_noiseMag < NN_SIGGEN_NOISE_FLOOR && m_equalNoiseSinePower){
	    m_noiseMag = NN_SIGGEN_NOISE_FLOOR;
	    printf("\n\tNoise magnitude is floored to %f in voiced region", m_noiseMag);
	}
	
    }

    template <typename TDevice>
    void SignalGenLayer<TDevice>::__setLayerMode()
    {
	// check layer mode
	if (m_freqDim >= 0){
	    if (m_freqDim >= this->precedingLayer().size())
		throw std::runtime_error("frequencyDim is larger than previous layer size");    
	    if (m_freqDim >= 0 && m_freqSR < 1)
		throw std::runtime_error("frequencySR is not specified");
	    if (m_freqBins > 0){	
		if (m_freqHmn > 0)
		    this->setLayerMode(NN_SIGGEN_LAYER_MODE_SINE_HARMO);
		else
		    this->setLayerMode(NN_SIGGEN_LAYER_MODE_SINE_PHASE);
		
		if (this->size() != (1+m_freqHmn)){
		    printf("Error: if frequencyBins>0, ");
		    throw std::runtime_error("signalgen laye size should be 1 + m_freqHmn");
		}
	    }else{
		this->setLayerMode(NN_SIGGEN_LAYER_MODE_SINE_SIMPLE);
	    }
	}else{
	    this->setLayerMode(NN_SIGGEN_LAYER_MODE_NOISE_ONLY);
	    printf("\nSignalGen: generating noise");
	}
	
    }
    
    template <typename TDevice>
    void SignalGenLayer<TDevice>::__allocateLocalMem()
    {
	m_noiseInput.resize(this->outputs().size(), 0.0);
	m_phaseNoise.resize(this->outputs().size(), 0.0);
	
	switch (this->getLayerMode()) {
	case NN_SIGGEN_LAYER_MODE_NOISE_ONLY:
	case NN_SIGGEN_LAYER_MODE_SINE_SIMPLE:
	    m_freqSignalBuff.clear();
	    m_freqErrorBuff.clear();
	    m_signalStatic.clear();
	    m_freqHmnBuff.clear();
	    break;
	case NN_SIGGEN_LAYER_MODE_SINE_HARMO:
	    m_freqHmnBuff.resize(this->outputs().size()/this->size() * m_freqHmn, 0.0);
	case NN_SIGGEN_LAYER_MODE_SINE_PHASE:
	    m_freqSignalBuff.resize(this->outputs().size()/this->size() * m_freqBins, 0.0);
	    m_freqErrorBuff.resize((m_freqBins + m_freqHmn) * this->parallelSequences(), 0.0);
	    m_signalStatic.resize((m_freqBins + m_freqHmn + 1) * this->parallelSequences() * 2,
				  0.0);
	    break;
	default:
	    break;
	}	
    }

    template <typename TDevice>
    void SignalGenLayer<TDevice>::__clearLocalMem()
    {
	m_freqSignalBuff.clear(); m_freqSignalBuff.shrink_to_fit();
	m_freqErrorBuff.clear();  m_freqErrorBuff.shrink_to_fit();
	m_signalStatic.clear();   m_signalStatic.shrink_to_fit();
	m_freqHmnBuff.clear();    m_freqHmnBuff.shrink_to_fit();
	m_noiseInput.clear();     m_noiseInput.shrink_to_fit();
	m_phaseNoise.clear();     m_phaseNoise.shrink_to_fit();
    }

    template <typename TDevice>
    void SignalGenLayer<TDevice>::exportLayer(const helpers::JsonValue     &layersArray, 
					      const helpers::JsonAllocator &allocator) const
    {
        TrainableLayer<TDevice>::exportLayer(layersArray, allocator);

	if (m_freqDim >= 0){
	    (*layersArray)[layersArray->Size() - 1].AddMember("frequencyOpt", m_freqOpt,
							      allocator);
	    (*layersArray)[layersArray->Size() - 1].AddMember("frequencySR",  m_freqSR,
							      allocator);
	    (*layersArray)[layersArray->Size() - 1].AddMember("frequencyQF0min", m_freqQF0min,
							      allocator);
	    (*layersArray)[layersArray->Size() - 1].AddMember("frequencyQF0max", m_freqQF0max,
							      allocator);
	    (*layersArray)[layersArray->Size() - 1].AddMember("frequencyQF0Lev", m_freqQF0Lev,
							      allocator);
	    (*layersArray)[layersArray->Size() - 1].AddMember("frequencyF0Mean", m_freqDataM,
							      allocator);
	    (*layersArray)[layersArray->Size() - 1].AddMember("frequencyF0Std",  m_freqDataS,
							      allocator);
	    
	    (*layersArray)[layersArray->Size() - 1].AddMember("frequencyF0Mag",  m_freqSignalMag,
							      allocator);
	    if (m_freqBins > 0){
		(*layersArray)[layersArray->Size() - 1].AddMember("frequencyBins", m_freqBins,
								  allocator);
		(*layersArray)[layersArray->Size() - 1].AddMember("frequencyHarmonics", m_freqHmn,
								  allocator);
	    }
	    if (m_noNoiseInSine)
		(*layersArray)[layersArray->Size() - 1].AddMember("noNoiseInSine", m_noNoiseInSine,
								  allocator);

	}
	(*layersArray)[layersArray->Size() - 1].AddMember("frequencyDim",      m_freqDim,
							  allocator);
	
	(*layersArray)[layersArray->Size() - 1].AddMember("frequencyNoiseMag", m_noiseMag,
							  allocator);

	if (m_noiseType != NN_SIGGEN_LAYER_NOISE_UNIFORM)
	    (*layersArray)[layersArray->Size() - 1].AddMember("noiseType", m_noiseType,
							      allocator);

	(*layersArray)[layersArray->Size() - 1].AddMember("phaseNoiseMag",     m_phaseNoiseMag,
							  allocator);

	if (m_equalNoiseSinePower){
	    (*layersArray)[layersArray->Size() - 1].AddMember("equalNoiseSinePower",
							      m_equalNoiseSinePower,
							      allocator);
	    
	}
    }

    template <typename TDevice>
    void SignalGenLayer<TDevice>::loadSequences(const data_sets::DataSetFraction &fraction,
						const int nnState)
    {
	TrainableLayer<TDevice>::loadSequences(fraction, nnState);
	
    }
    
    template <typename TDevice>
    const std::string& SignalGenLayer<TDevice>::type() const
    {
        static std::string s;
        if (s.empty()) s = "signalgen";
        return s;
    }

    template <typename TDevice>
    void SignalGenLayer<TDevice>::computeForwardPass(const int nnState)
    {
	int timeLength      = this->curMaxSeqLength();
	int timeLengthTotal = timeLength * this->parallelSequences();
	int signalDimTotal  = this->size() * this->parallelSequences();

	thrust::fill(this->outputs().begin(), this->outputs().end(), 0.0);
		
	// generate noise
	thrust::counting_iterator<unsigned int> index_sequence_begin(0);
	if (m_noiseType == NN_SIGGEN_LAYER_NOISE_GAUSSIAN){
	    thrust::transform(index_sequence_begin,
			      index_sequence_begin + timeLength * signalDimTotal,
			      m_noiseInput.begin(),
			      internal::genNoise_Gaussian(0.0, m_noiseMag/3.0,
				(int)(misFuncs::GetRandomNumber()*10000.0)));
	}else{
	    thrust::transform(index_sequence_begin,
			      index_sequence_begin + timeLength * signalDimTotal,
			      m_noiseInput.begin(),
			      internal::genNoise(-1.0 * m_noiseMag, m_noiseMag,
						 (int)(misFuncs::GetRandomNumber()*10000.0)));
	}


	if (m_phaseNoiseMag > 0.0){
	    thrust::transform(index_sequence_begin,
			      index_sequence_begin + timeLength * signalDimTotal,
			      m_phaseNoise.begin(),
			      internal::genNoise(-1.0 * m_phaseNoiseMag, m_phaseNoiseMag,
						 (int)(misFuncs::GetRandomNumber()*10000.0)));
	}else{
	    thrust::fill(m_phaseNoise.begin(), m_phaseNoise.end(), 0.0);
	    
	}	

	// generate signal
	if (m_freqDim < 0){
	    
	    // generating only random noise
	    thrust::copy(m_noiseInput.begin(),
			 m_noiseInput.begin() + timeLength * this->size(),
			 this->outputs().begin());
	    
	}else{
	
	    if (m_freqBins == 0){
	    
		// Directly generate multiple signals with different initial phase
	    
		internal::sinWaveGenerator_accum fn1;
		fn1.inputRes      = this->precedingLayer().getResolution();
		fn1.outputRes     = this->getResolution();
		fn1.signalDim     = this->size();
		fn1.hnmNum        = 0;
		fn1.prelayerSize  = this->precedingLayer().size();
		
		fn1.inputShiftTime  = 0;
		fn1.outputShiftTime = 0;
		
		fn1.freqDim       = this->m_freqDim;
		fn1.freqOpt       = this->m_freqOpt;
		fn1.freqSR        = (real_t)this->m_freqSR;
		fn1.qf0Max        = this->m_freqQF0max;
		fn1.qf0Min        = this->m_freqQF0min;
		fn1.qf0Lev        = this->m_freqQF0Lev;
		fn1.f0M           = this->m_freqDataM;
		fn1.f0S           = this->m_freqDataS;

		fn1.f0Mag         = this->m_freqSignalMag;
		fn1.addtiveNoiseMag= this->m_noiseMag;
		fn1.noNoiseInSince = m_noNoiseInSine;
		fn1.parallel       = this->parallelSequences();
		fn1.genSignalBuff  = helpers::getRawPointer(this->outputs());
		fn1.sourceData     = helpers::getRawPointer(this->precedingLayer().outputs());
		fn1.patTypes       = helpers::getRawPointer(this->patTypes());
		fn1.phaseNoise     = helpers::getRawPointer(this->m_phaseNoise);
		fn1.addtiveNoise   = helpers::getRawPointer(this->m_noiseInput);
		fn1.signalStatic   = NULL;
		fn1.targetData     = NULL;
		fn1.HmnBuff        = NULL;

		fn1.equalNoiseSinePower = m_equalNoiseSinePower;
		fn1.seqLength      = timeLength;

		
		thrust::for_each(
		    thrust::make_zip_iterator(
			thrust::make_tuple(this->outputs().begin(),
					   thrust::counting_iterator<int>(0))),
		    thrust::make_zip_iterator(
			thrust::make_tuple(this->outputs().begin()           + signalDimTotal,
					   thrust::counting_iterator<int>(0) + signalDimTotal)),
		    fn1);
	       
	       
	    
		thrust::transform(m_noiseInput.begin(),
				  m_noiseInput.begin() + timeLengthTotal * this->size(),
				  this->outputs().begin(),
				  this->outputs().begin(),
				  thrust::plus<real_t>());
	    
	    
	    }else{

		// Generate the phase-matched signal as output
		
		// Step1. generate candidate signals
		
		internal::sinWaveGenerator_accum fn1;
		fn1.inputRes      = this->precedingLayer().getResolution();
		fn1.outputRes     = this->getResolution();
		fn1.signalDim     = m_freqBins + m_freqHmn;
		fn1.hnmNum        = m_freqHmn;
		fn1.prelayerSize  = this->precedingLayer().size();

		fn1.equalNoiseSinePower = m_equalNoiseSinePower;
		fn1.inputShiftTime  = 0;
		fn1.outputShiftTime = 0;
	    
		fn1.freqDim       = this->m_freqDim;
		fn1.freqOpt       = this->m_freqOpt;
		fn1.freqSR        = (real_t)this->m_freqSR;
		fn1.qf0Max        = this->m_freqQF0max;
		fn1.qf0Min        = this->m_freqQF0min;
		fn1.qf0Lev        = this->m_freqQF0Lev;
		fn1.f0M           = this->m_freqDataM;
		fn1.f0S           = this->m_freqDataS;

		fn1.f0Mag         = this->m_freqSignalMag;
		fn1.addtiveNoiseMag= this->m_noiseMag;
		fn1.noNoiseInSince = m_noNoiseInSine;		
		fn1.HmnNum         = this->m_freqHmn;
		
		fn1.parallel       = this->parallelSequences();
		fn1.genSignalBuff  = helpers::getRawPointer(m_freqSignalBuff);
		fn1.sourceData     = helpers::getRawPointer(this->precedingLayer().outputs());
		fn1.patTypes       = helpers::getRawPointer(this->patTypes());
		fn1.phaseNoise     = helpers::getRawPointer(this->m_phaseNoise);
		fn1.HmnBuff        = helpers::getRawPointer(m_freqHmnBuff);
		fn1.addtiveNoise   = helpers::getRawPointer(this->m_noiseInput);
		
		if (this->flagTrainingMode()){
		    fn1.targetData     = helpers::getRawPointer(this->m_targetLayer->outputs());
		    fn1.signalStatic   = helpers::getRawPointer(m_signalStatic);
		}else{
		    fn1.targetData = NULL;
		    fn1.signalStatic = NULL;
		}

		fn1.seqLength      = timeLength;
	    
		thrust::for_each(
                 thrust::make_zip_iterator(
		  thrust::make_tuple(m_freqErrorBuff.begin(),
				     thrust::counting_iterator<int>(0))),
		 thrust::make_zip_iterator(
		  thrust::make_tuple(m_freqErrorBuff.begin()           + m_freqErrorBuff.size(),
				     thrust::counting_iterator<int>(0) + m_freqErrorBuff.size())),
		 fn1);

		if (this->flagTrainingMode()){
		    // Step2. calculate correlation
		    internal::signalCorr fn2;
		    fn2.signalDim = m_freqBins + m_freqHmn;
		    fn2.hnmNum    = m_freqHmn;
		    fn2.seqLength = timeLength;
		    fn2.parallel  = this->parallelSequences();
	    
		    fn2.genSignalBuff  = helpers::getRawPointer(m_freqSignalBuff);
		    fn2.targetData     = helpers::getRawPointer(this->m_targetLayer->outputs());
		    fn2.signalStatic   = helpers::getRawPointer(m_signalStatic);
		
		    fn2.patTypes       = helpers::getRawPointer(this->patTypes());
		    thrust::for_each(
                     thrust::make_zip_iterator(
		      thrust::make_tuple(m_freqErrorBuff.begin(),
					 thrust::counting_iterator<int>(0))),
		     thrust::make_zip_iterator(
		      thrust::make_tuple(m_freqErrorBuff.begin()           + m_freqErrorBuff.size(),
					 thrust::counting_iterator<int>(0) + m_freqErrorBuff.size())),
		     fn2);
	    
		    // step3. select the matched signal
		    internal::getBestIndex fn3;
		    fn3.errorBuff = helpers::getRawPointer(m_freqErrorBuff);
		    fn3.signalDim = m_freqBins + m_freqHmn;
		    fn3.hnmNum    = m_freqHmn;
		    fn3.parallel  = this->parallelSequences();
		    thrust::for_each(
                     thrust::make_zip_iterator(
		      thrust::make_tuple(
			m_freqErrorBuff.begin(),
			thrust::counting_iterator<int>(0))),
		     thrust::make_zip_iterator(
		      thrust::make_tuple(
			m_freqErrorBuff.begin()           + this->parallelSequences(),
			thrust::counting_iterator<int>(0) + this->parallelSequences())),
		     fn3);
		}else{
		    
		    // During generation, just use the first candidate
		    thrust::fill(m_freqErrorBuff.begin(), m_freqErrorBuff.end(), 0.0);
		}
	    
		// step4. put the target signal into the buffer
		internal::LoadSignal fn4;

		fn4.phaseCandNum = m_freqBins;
		fn4.hnmNum       = m_freqHmn;
	    
		fn4.parallel  = this->parallelSequences();
		
		fn4.errorBuff      = helpers::getRawPointer(m_freqErrorBuff);
		fn4.genSignalBuff  = helpers::getRawPointer(m_freqSignalBuff);
		fn4.patTypes       = helpers::getRawPointer(this->patTypes());
		fn4.noise          = helpers::getRawPointer(m_noiseInput);
		fn4.HmnBuff        = helpers::getRawPointer(m_freqHmnBuff);
	    
		thrust::for_each(
                 thrust::make_zip_iterator(
		   thrust::make_tuple(
			this->outputs().begin(),
			thrust::counting_iterator<int>(0))),
		 thrust::make_zip_iterator(
	           thrust::make_tuple(
			this->outputs().begin() + timeLengthTotal * this->size(),
			thrust::counting_iterator<int>(0)+timeLengthTotal * this->size())),
		 fn4);
	    
	    }
	}

    }

    template <typename TDevice>
    void SignalGenLayer<TDevice>::computeForwardPass(const int timeStep, const int nnState)
    {
	int timeLength = this->curMaxSeqLength() * this->parallelSequences();

	int effTimeStart = timeStep * this->parallelSequences();
	int effTimeEnd   = (timeStep+1) * this->parallelSequences();
	
	// Pointer to the output of previous layer (input buffer)
	int shiftIn  = this->precedingLayer().outputBufPtrBias(timeStep*this->parallelSequences(),
							       nnState);
	// Pointer to the output of this layer
	int shiftOut = this->outputBufPtrBias(timeStep * this->parallelSequences(), nnState);

	throw std::runtime_error("signalgen not implemented");

    }

    template <typename TDevice>
    void SignalGenLayer<TDevice>::computeBackwardPass(const int nnState)
    {	
	// do nothing
    }

    template <typename TDevice>
    void SignalGenLayer<TDevice>::reduceOutputBuffer()
    {
	throw std::runtime_error("signalgen not implemented");
    }
    
    template <typename TDevice>
    int SignalGenLayer<TDevice>::outputBufPtrBias(const int timeStepTimesParallel,
						  const int nnState)
    {
	if (this->getSaveMemoryFlag()){
	    return timeStepTimesParallel * this->size();
	}else{
	    return 0;
	}
    }	


    template <typename TDevice>
    void SignalGenLayer<TDevice>::linkTargetLayer(Layer<TDevice> &targetLayer)
    {
	if (m_freqBins > 0){
	    m_targetLayer = &targetLayer;
	    if (m_targetLayer->size() != 1)
		throw std::runtime_error("signalgen not implemented for target size > 1");
	}else{
	    m_targetLayer = NULL;
	}
    }


    template <typename TDevice>
    std::vector<int> SignalGenLayer<TDevice>::dependLayerIDs()
    {
	std::vector<int> tmp;
	tmp.clear();
	return tmp;
    }
        
    template <typename TDevice>
    void SignalGenLayer<TDevice>::clearAllBuffers()
    {
	this->clearOutputBuffer();
	this->__clearLocalMem();
    }

    template <typename TDevice>
    void SignalGenLayer<TDevice>::resizeAllBuffers(const int timeLength)
    {
	this->resizeOutputBuffer(timeLength * this->parallelSequences() * this->size());
	this->__allocateLocalMem();
    }

    
    template class SignalGenLayer<Cpu>;
    template class SignalGenLayer<Gpu>;
    
}

