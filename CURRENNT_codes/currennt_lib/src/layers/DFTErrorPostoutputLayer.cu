/******************************************************************************
 * This file is an addtional component of CURRENNT. 
 * Xin WANG
 * National Institute of Informatics, Japan
 * 2018
 *
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


#include "DFTErrorPostoutputLayer.hpp"
#include "../helpers/getRawPointer.cuh"
#include "../helpers/Matrix.hpp"
#include "../helpers/FFTMat.hpp"
#include "../helpers/JsonClasses.hpp"
#include "../helpers/misFuncs.hpp"

#include "../Configuration.hpp"
#include "../MacroDefine.hpp"

#include <thrust/functional.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/for_each.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/fill.h>
#include <boost/foreach.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/algorithm/string.hpp>
#include <vector>

#define DFTERRORPOST_HNM_MODEL_1 1
#define DFTERRORPOST_HNM_MODEL_2 2

// to be integrated with signalgen F0 U/V
#define DFTERRORUV       10
#define DFTERRORUVSEARCH 2
#define DFTERROR_PI      3.141215

#define DFTMODEFORMULTIDIMSIGNAL_NONE    0
#define DFTMODEFORMULTIDIMSIGNAL_CONCATE 1

#define DFTPREEMPHASISCOEFF 0.94

namespace internal{
namespace{

    struct ComputeMseWaveform
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

    struct ComputeMseWaveformGrad
    {
	real_t *preError;
	real_t *preData;
	real_t *realTargetData;

	int     preDim;
        int     featDim;
	real_t  beta;
	bool    accumulate;
	
        const char *patTypes;

        __host__ __device__ void operator() (const thrust::tuple<real_t, int> &values) const
        {
            // unpack the tuple
            int timeIdx = values.get<1>() / featDim;
	    int dimIdx  = values.get<1>() % featDim;

            if (patTypes[timeIdx] == PATTYPE_NONE){
		preError[timeIdx * preDim + dimIdx] = 0.0;
	    }else{
		if (accumulate)
		    preError[timeIdx * preDim + dimIdx] +=
			(preData[timeIdx * preDim + dimIdx] -
			 realTargetData[timeIdx * featDim + dimIdx]) * beta;
		else
		    preError[timeIdx * preDim + dimIdx] =
			(preData[timeIdx * preDim + dimIdx] -
			 realTargetData[timeIdx * featDim + dimIdx]) * beta;
	    }
        }
    };

    struct TimeDomainRemoveWaveformVoiced
    {
	real_t  f0TimeResolution;
	real_t *f0DataBuffer;

	real_t  f0DataM;
	real_t  f0DataS;
	
	int     f0InputLayerDim;
	int     waveformLength;
	int     featDim;
	
	const char *patTypes;
	
	__host__ __device__ void operator() (const thrust::tuple<real_t&, int> &values) const
	{
	    
            int timeIdx = values.get<1>() / featDim;
	    int f0TimeIdx = timeIdx / f0TimeResolution;
	    
            if (patTypes[timeIdx] == PATTYPE_NONE){
		values.get<0>() = 0.0;
	    }else{
		
		// remove voiced waveforms through time domain approach
		int distance = -1;
		real_t weight = 0.0;

		// if this point is in voiced frame
		if ((f0DataBuffer[f0TimeIdx * f0InputLayerDim + f0InputLayerDim - 1] *
		     f0DataS + f0DataM) > DFTERRORUV){

		    // first, look ahead, find the preceding unvoiced frame
		    for (int lookhead = 1;
			 (f0TimeIdx - lookhead) >=0 && lookhead <= DFTERRORUVSEARCH;
			 lookhead++){
			
			// if the frameframe is unvoiced
			//  make the boundary of unvoiced/voiced
			if ((f0DataBuffer[(f0TimeIdx - lookhead) * f0InputLayerDim +
					  f0InputLayerDim - 1] * f0DataS + f0DataM)
			    < DFTERRORUV){
			    distance = timeIdx - ((f0TimeIdx - lookhead + 1) * f0TimeResolution -1);
			    break;
			}
		    }

		    // second, look back, find the following unvoiced voices 
		    if (distance < 0){
			// first look ahead
			for (int lookback=1;
			     (f0TimeIdx + lookback) < waveformLength && lookback <= DFTERRORUVSEARCH;
			     lookback++){
			    if ((f0DataBuffer[(f0TimeIdx + lookback) * f0InputLayerDim +
					      f0InputLayerDim - 1] * f0DataS + f0DataM)
				< DFTERRORUV){
				distance = (f0TimeIdx + lookback) * f0TimeResolution - timeIdx;
				break;
			    }
			}	
		    }

		    if (distance < 0){
			// this time step is very inside a voiced frame, set it to zero directly
			values.get<0>() = 0.0;
		    }else{
			// if this time step is near the boundary of U/V change,
			//  set the voiced region based on Hann window
			weight = 0.5 * (1.0 + cos(2.0 * DFTERROR_PI * distance /
						  (2.0 * f0TimeResolution * DFTERRORUVSEARCH - 1)));
			values.get<0>() = weight * values.get<0>();
		    }
		}
	    }
	}
    };

    
    struct multiDimSignaltoOneDim
    {
	
	real_t *sourceData;
	int     sourceDim;
	int     maxLength;
	
        const char *patTypes;

        __host__ __device__ void operator() (const thrust::tuple<real_t&, int> &values) const
        {
            // unpack the tuple
            int dimIdx   = values.get<1>() / maxLength;
	    int timeIdx  = values.get<1>() % maxLength;

            if (patTypes[timeIdx] == PATTYPE_NONE){
		// dummy data point
		values.get<0>() = 0.0;
	    }else{
		values.get<0>() = sourceData[timeIdx * sourceDim + dimIdx];
	    }
        }
    };

    struct multiDimSignaltoOneDimGrad
    {
	
	real_t *sourceData;
	int     sourceDim;
	int     maxLength;
	
        const char *patTypes;

        __host__ __device__ void operator() (const thrust::tuple<real_t&, int> &values) const
        {
            // unpack the tuple
            int timeIdx = values.get<1>() / sourceDim;
	    int dimIdx  = values.get<1>() % sourceDim;

            if (patTypes[timeIdx] == PATTYPE_NONE){
		// dummy data point
		values.get<0>() = 0.0;
	    }else{
		values.get<0>() = sourceData[dimIdx * maxLength + timeIdx];
	    }
        }
    };

    struct preemphasis
    {
	
	real_t *sourceData;
	int     parallel;
	
        const char *patTypes;

        __host__ __device__ void operator() (const thrust::tuple<real_t&, int> &values) const
        {
            // unpack the tuple
            int timeIdx      = values.get<1>();
	    int blockIdx     = values.get<1>() / parallel;
	    int blockIntIdx  = values.get<1>() % parallel;

            if (patTypes[timeIdx] == PATTYPE_NONE){
		// dummy data point
		values.get<0>() = 0.0;
	    }else{
		// o[n] = s[n] - 0.94 * s[n-1]
		// o[n]: values.get<0>()
		// s[n]: sourceData[timeIdx]
		// s[n-1]: sourceData[(blockIdx-1) * parallel + blockIntIdx];
		
		if ((blockIdx - 1)>=0)
		    values.get<0>() = sourceData[timeIdx] -
			DFTPREEMPHASISCOEFF * sourceData[(blockIdx-1) * parallel + blockIntIdx];
		else
		    values.get<0>() = sourceData[timeIdx];
	    }
        }
    };

    struct deemphasis
    {
	
	real_t     *sourceData;
	int         parallel;
	int         maxWaveLength;
        const char *patTypes;

        __host__ __device__ void operator() (const thrust::tuple<real_t&, int> &values) const
        {
	    // de-emphasis is autoregressive, cannot be parallelized
	    // but can be parallelized cross different waveforms in the batch
	    
	    int blockIdx = values.get<1>();
	    for (int timeIdx = 0; timeIdx < maxWaveLength; timeIdx++){
		if (timeIdx > 0 && patTypes[timeIdx * parallel + blockIdx] != PATTYPE_NONE){
		    sourceData[timeIdx * parallel + blockIdx] =
			sourceData[timeIdx * parallel + blockIdx] +
			DFTPREEMPHASISCOEFF * sourceData[(timeIdx - 1) * parallel + blockIdx];
		}
		
	    }
        }
    };
    
}
}


namespace layers{

    // Construct the layer
    template <typename TDevice>
    DFTPostoutputLayer<TDevice>::DFTPostoutputLayer(
				const helpers::JsonValue &layerChild,
				Layer<TDevice> &precedingLayer,
				int maxSeqLength,
				int layerID)
        : PostOutputLayer<TDevice>(layerChild, precedingLayer,
				   precedingLayer.size(), maxSeqLength, layerID)
	, m_mseError       (0.0)
	, m_specError      (0.0)
	, m_specError2     (0.0)
	, m_specError3     (0.0)
	, m_beta           (0.0)
	, m_gamma          (0.0)
	, m_zeta           (0.0)
	, m_fftLength      (0)
	, m_fftBinsNum     (0)
	, m_frameLength    (0)
	, m_frameShift     (0)
	, m_frameNum       (0)
	, m_windowType     (FFTMAT_WINDOW_HANN)
	, m_windowType2    (FFTMAT_WINDOW_HANN)
	, m_windowType3    (FFTMAT_WINDOW_HANN)
	, m_windowTypePhase  (FFTMAT_WINDOW_HANN)
	, m_windowTypePhase2 (FFTMAT_WINDOW_HANN)
	, m_windowTypePhase3 (FFTMAT_WINDOW_HANN)
	, m_noiseOutputLayer (NULL)
	, m_f0InputLayer     (NULL)
	, m_noiseTrain_epoch (-1)
	, m_modeMultiDimSignal (DFTMODEFORMULTIDIMSIGNAL_NONE)
    {
	
	if (precedingLayer.size() != this->size())
	    throw std::runtime_error("\nDFT Error previous layer size != DFT layer size");
		
	this->__loadOpts(layerChild);

	if (m_modeMultiDimSignal == DFTMODEFORMULTIDIMSIGNAL_NONE){
	    if (this->size() > 1){
		printf("\nError in dft layer: DFT layer size > 1, but multiDimSignalMode is off");
		throw std::runtime_error("\nError in network");
	    }
	    if (this->parallelSequences() > 1)
		throw std::runtime_error("\nDF TError is not implemented for parallel mode");
	}
    }	


    template <typename TDevice>
    void DFTPostoutputLayer<TDevice>::__loadOpts(const helpers::JsonValue &layerChild)
    {

	// Training criterion E =
	//    m_beta * waveform_MSE + m_gamma * spectral_amplitude_dis + m_zeta * phase_dis
	m_beta          = (layerChild->HasMember("beta") ? 
			   static_cast<real_t>((*layerChild)["beta"].GetDouble()) : 0.0);

	m_gamma         = (layerChild->HasMember("gamma") ? 
			   static_cast<real_t>((*layerChild)["gamma"].GetDouble()) : 0.0);

	m_zeta         = (layerChild->HasMember("zeta") ? 
			  static_cast<real_t>((*layerChild)["zeta"].GetDouble()) : 0.0);

	// Type of spectral amplitude distance (see ../helpers/FFTMat.hpp):
	//  FFTMAT_SPECTYPE_MSE: MSE of log-spectra
	//  FFTMAT_SPECTYPE_KLD: KLD of spectra
	m_specDisType   = (layerChild->HasMember("specDisType") ? 
			   static_cast<real_t>((*layerChild)["specDisType"].GetInt()) :
			   FFTMAT_SPECTYPE_MSE);

	// Reserved option
	//  if the target signal is multi-dimensional, we can convert the multi-dimensional
	//  signal into one-dimensional signal as a waveform, then calcualte FFT distance
	m_modeMultiDimSignal = (layerChild->HasMember("multiDimSignalMode") ? 
				static_cast<int>((*layerChild)["multiDimSignalMode"].GetInt()) :
				DFTMODEFORMULTIDIMSIGNAL_NONE);

	// Reserved option
	//  If the natural waveform has to be pre-emphasized, we can use this option to
	//  do pre-emphasis, rather than preparing new training data
	//  Generated waveform will be de-emphasized in function computeForward()
	m_preEmphasis = (layerChild->HasMember("preEmphasisNaturalWav") ? 
			 static_cast<int>((*layerChild)["preEmphasisNaturalWav"].GetInt()) : 0);

	// Maximum length of the waveforms in the training/test set
	int maxSeqLength = this->__vMaxSeqLength();
	
	
	// Configuration for DFT frameing/windowing
	if (m_gamma > 0.0 || m_zeta > 0.0){

	    // N in FFT
	    m_fftLength   = (layerChild->HasMember("fftLength") ? 
			     static_cast<int>((*layerChild)["fftLength"].GetInt()) : 512);
	    
	    //  make m_fftLength an even number
	    m_fftLength   = (m_fftLength / 2) * 2;
	    
	    // actual length of one frame, which must be smaller than m_fftLength
	    m_frameLength = (layerChild->HasMember("frameLength") ? 
			     static_cast<int>((*layerChild)["frameLength"].GetInt()) : 320);
	    if (m_frameLength > m_fftLength)
		throw std::runtime_error("\nFFT length should be > frame length");

	    // frame shift
	    m_frameShift  = (layerChild->HasMember("frameShift") ? 
			     static_cast<int>((*layerChild)["frameShift"].GetInt()) : 80);

	    // number of fftBins (dimension of FFT feature) actually stored
	    m_fftBinsNum  = helpers::fftTools::fftBinsNum(m_fftLength);

	    // how many frames at maximum
	    m_frameNum    = helpers::fftTools::fftFrameNum(maxSeqLength, m_frameLength,
							   m_frameShift);

	    m_windowType  = (layerChild->HasMember("windowType") ? 
			     static_cast<real_t>((*layerChild)["windowType"].GetInt()) :
			     FFTMAT_WINDOW_HANN);

	    m_windowTypePhase  = (layerChild->HasMember("windowTypePhase") ? 
				  static_cast<real_t>((*layerChild)["windowTypePhase"].GetInt()) :
				  FFTMAT_WINDOW_HANN);

	    complex_t tmp;
	    tmp.x = 0;
	    tmp.y = 0;

	    // framed signal buffer (with zero padding)
	    m_fftSourceFramed.resize(m_frameNum * m_fftLength, 0.0);
	    m_fftTargetFramed.resize(m_frameNum * m_fftLength, 0.0);
	    m_fftDiffFramed = m_fftTargetFramed;

	    // FFT buffer (complex number)
	    m_fftSourceSigFFT.resize(m_frameNum * m_fftBinsNum, tmp);
	    m_fftTargetSigFFT.resize(m_frameNum * m_fftBinsNum, tmp);
	    m_fftDiffSigFFT = m_fftTargetSigFFT;

	    // A temporary buffer to store the difference between target and source
	    m_fftDiffData = this->outputs();
	    if (m_zeta > 0.0)
		m_fftDiffDataPhase = m_fftDiffData;

	    
	    // Second FFT part
	    m_fftLength2   = (layerChild->HasMember("fftLength2") ? 
			     static_cast<int>((*layerChild)["fftLength2"].GetInt()) : 0);
	    m_frameLength2 = (layerChild->HasMember("frameLength2") ? 
			      static_cast<int>((*layerChild)["frameLength2"].GetInt()) : 80);
	    m_frameShift2  = (layerChild->HasMember("frameShift2") ? 
			      static_cast<int>((*layerChild)["frameShift2"].GetInt()) : 40);
	    m_windowType2  = (layerChild->HasMember("windowType2") ? 
			      static_cast<real_t>((*layerChild)["windowType2"].GetInt()) :
			      FFTMAT_WINDOW_HANN);
	    m_windowTypePhase2  = (layerChild->HasMember("windowTypePhase2") ? 
				  static_cast<real_t>((*layerChild)["windowTypePhase2"].GetInt()) :
				  FFTMAT_WINDOW_HANN);
	    m_fftBinsNum2  = helpers::fftTools::fftBinsNum(m_fftLength2);
	    m_frameNum2    = helpers::fftTools::fftFrameNum(maxSeqLength, m_frameLength2,
							   m_frameShift2);
	    m_fftSourceFramed2.resize(m_frameNum2 * m_fftLength2, 0.0);
	    m_fftTargetFramed2.resize(m_frameNum2 * m_fftLength2, 0.0);
	    m_fftDiffFramed2 = m_fftTargetFramed2;
	    m_fftSourceSigFFT2.resize(m_frameNum2 * m_fftBinsNum2, tmp);
	    m_fftTargetSigFFT2.resize(m_frameNum2 * m_fftBinsNum2, tmp);
	    m_fftDiffSigFFT2 = m_fftTargetSigFFT2;
	    m_fftDiffData2 = this->outputs();
	    if (m_zeta > 0.0)
		m_fftDiffDataPhase2 = m_fftDiffData2;

	    
	    // third FFT part
	    m_fftLength3   = (layerChild->HasMember("fftLength3") ? 
			     static_cast<int>((*layerChild)["fftLength3"].GetInt()) : 0);
	    m_frameLength3 = (layerChild->HasMember("frameLength3") ? 
			      static_cast<int>((*layerChild)["frameLength3"].GetInt()) : 80);
	    m_frameShift3  = (layerChild->HasMember("frameShift3") ? 
			      static_cast<int>((*layerChild)["frameShift3"].GetInt()) : 40);
	    m_windowType3  = (layerChild->HasMember("windowType3") ? 
			      static_cast<real_t>((*layerChild)["windowType3"].GetInt()) :
			      FFTMAT_WINDOW_HANN);
	    m_windowTypePhase3  = (layerChild->HasMember("windowTypePhase3") ? 
				  static_cast<real_t>((*layerChild)["windowTypePhase3"].GetInt()) :
				  FFTMAT_WINDOW_HANN);

	    m_fftBinsNum3  = helpers::fftTools::fftBinsNum(m_fftLength3);
	    m_frameNum3    = helpers::fftTools::fftFrameNum(maxSeqLength, m_frameLength3,
							   m_frameShift3);
	    m_fftSourceFramed3.resize(m_frameNum3 * m_fftLength3, 0.0);
	    m_fftTargetFramed3.resize(m_frameNum3 * m_fftLength3, 0.0);
	    m_fftDiffFramed3 = m_fftTargetFramed3;
	    m_fftSourceSigFFT3.resize(m_frameNum3 * m_fftBinsNum3, tmp);
	    m_fftTargetSigFFT3.resize(m_frameNum3 * m_fftBinsNum3, tmp);
	    m_fftDiffSigFFT3 = m_fftTargetSigFFT3;
	    m_fftDiffData3 = this->outputs();
	    if (m_zeta > 0.0)
		m_fftDiffDataPhase3 = m_fftDiffData3;
	    
	}else{
	    m_fftSourceFramed.clear();
	    m_fftTargetFramed.clear();
	    m_fftSourceSigFFT.clear();
	    m_fftTargetSigFFT.clear();
	    m_fftDiffSigFFT.clear();
	    m_fftDiffFramed.clear();
	    m_fftDiffData.clear();
	    m_fftDiffDataPhase.clear();
	    
	    m_fftSourceFramed2.clear();
	    m_fftTargetFramed2.clear();
	    m_fftSourceSigFFT2.clear();
	    m_fftTargetSigFFT2.clear();
	    m_fftDiffSigFFT2.clear();
	    m_fftDiffFramed2.clear();
	    m_fftDiffData2.clear();
	    m_fftDiffDataPhase2.clear();
	    
	    m_fftSourceFramed3.clear();
	    m_fftTargetFramed3.clear();
	    m_fftSourceSigFFT3.clear();
	    m_fftTargetSigFFT3.clear();
	    m_fftDiffSigFFT3.clear();
	    m_fftDiffFramed3.clear();
	    m_fftDiffData3.clear();
	    m_fftDiffDataPhase3.clear();   
	}

	// Reserved option for a special training strategy on h-NSF.
	//  not used anymore
	m_hnm_flag   = (layerChild->HasMember("hnmMode") ? 
			static_cast<int>((*layerChild)["hnmMode"].GetInt()) : 0);
	if (m_hnm_flag > 0){
	    m_noiseTrain_epoch = (layerChild->HasMember("noisePartTrainEpochNum") ? 
		static_cast<int>((*layerChild)["noisePartTrainEpochNum"].GetInt()) : 15);
	    
	    m_noiseOutputLayerName = (layerChild->HasMember("noiseOutputLayerName") ? 
				      ((*layerChild)["noiseOutputLayerName"].GetString()) : "");
	    
	    m_f0InputLayerName = (layerChild->HasMember("f0InputLayerName") ? 
				  ((*layerChild)["f0InputLayerName"].GetString()) : "");
	    printf("\n\tDFT error layers turns on HNM model mode [%d]", m_hnm_flag);

	    m_f0DataM = (layerChild->HasMember("f0DataMean")?
			 static_cast<real_t>((*layerChild)["f0DataMean"].GetDouble()):0);
	    m_f0DataS = (layerChild->HasMember("f0DataStd")?
			 static_cast<real_t>((*layerChild)["f0DataStd"].GetDouble()):1);

	    const Configuration &config = Configuration::instance();
	    if (config.f0dataMean_signalgen() > 0)
		m_f0DataM = config.f0dataMean_signalgen();
	    if (config.f0dataStd_signalgen() > 0)
		m_f0DataS = config.f0dataStd_signalgen();
	    printf("\n\tDFT errir layers receives F0 mean-%f std-%f", m_f0DataM, m_f0DataS);
	}
	
	if (m_modeMultiDimSignal != DFTMODEFORMULTIDIMSIGNAL_NONE)
	    m_modeChangeDataBuf = this->outputs();
	else
	    m_modeChangeDataBuf.clear();
	
	if (m_preEmphasis)
	    printf("\n\tNatural waveform will be pre-emphasis before evaluating.");
    
    }

    
    // Destructor
    template <typename TDevice>
    DFTPostoutputLayer<TDevice>::~DFTPostoutputLayer()
    {
    }
    
    // NN forward
    template <typename TDevice>
    void DFTPostoutputLayer<TDevice>::computeForwardPass(const int nnState)
    {
	// Checking 
	if (this->getSaveMemoryFlag())
	    throw std::runtime_error("Memory save mode should be turned off");

	// Length of the currennt waveform (or maximum length in the batch)
	int timeLength = this->__vCurMaxSeqLength();
		
	
	if (!this->flagTrainingMode()){
	    // Testing mode

	    // De-emphasis on the generated waveform
	    if (m_preEmphasis){
		{
		internal::deemphasis fn1;
		fn1.sourceData = helpers::getRawPointer(this->precedingLayer().outputs());
		fn1.parallel   = this->parallelSequences();
		fn1.patTypes   = helpers::getRawPointer(this->patTypes());
		fn1.maxWaveLength = timeLength;
		
		// use the buffer of outputErrors to store results
		thrust::for_each(
		   thrust::make_zip_iterator(
			thrust::make_tuple(
				this->outputs().begin(),
				thrust::counting_iterator<int>(0))),
		   thrust::make_zip_iterator(
			thrust::make_tuple(
				this->outputs().begin() + this->parallelSequences(),
				thrust::counting_iterator<int>(0)  + this->parallelSequences())),
		   fn1);
		}
	    }
	    return;
	}else{
	    // Training mode
	    
	    // If preemphasis is necessary
	    if (m_preEmphasis){
		// pre-emphasis the natural waveform before evaluating
		{
		internal::preemphasis fn1;
		fn1.sourceData = helpers::getRawPointer(this->outputs());
		fn1.parallel   = this->parallelSequences();
		fn1.patTypes   = helpers::getRawPointer(this->patTypes());

		// use the buffer of outputErrors to store results
		thrust::for_each(
		   thrust::make_zip_iterator(
			thrust::make_tuple(this->outputErrors().begin(),
					   thrust::counting_iterator<int>(0))),
		   thrust::make_zip_iterator(
			thrust::make_tuple(this->outputErrors().begin()            + timeLength,
					   thrust::counting_iterator<int>(0)  + timeLength)),
		   fn1);
		this->outputs() = this->outputErrors();
		thrust::fill(this->outputErrors().begin(), this->outputErrors().end(), 0.0);
		}
	    }

	    // If target is multi-dimensional signal
	    if (m_modeMultiDimSignal == DFTMODEFORMULTIDIMSIGNAL_CONCATE){
		// If the output of previous layer is multi-dimensional signal,
		// convert the N * T input data matrix into a 1-dim signal of length NT.

		// convert target data
		{
		    internal::multiDimSignaltoOneDim fn1;
		    fn1.sourceData  = helpers::getRawPointer(this->outputs());
		    fn1.patTypes    = helpers::getRawPointer(this->patTypes());
		    fn1.maxLength   = this->precedingLayer().curMaxSeqLength();
		    fn1.sourceDim   = this->precedingLayer().parallelSequences() * this->size();
		    
		    thrust::for_each(
		      thrust::make_zip_iterator(
		       thrust::make_tuple(
			 m_modeChangeDataBuf.begin(),
			 thrust::counting_iterator<int>(0))),
		      thrust::make_zip_iterator(
		       thrust::make_tuple(
			 m_modeChangeDataBuf.begin()        + timeLength,
			 thrust::counting_iterator<int>(0)  + timeLength)),
		      fn1);
		}
		
		thrust::copy(m_modeChangeDataBuf.begin(), m_modeChangeDataBuf.end(),
			     this->outputs().begin());

		// convert target data
		{
		    internal::multiDimSignaltoOneDim fn1;
		    fn1.sourceData  = helpers::getRawPointer(this->precedingLayer().outputs());
		    fn1.patTypes    = helpers::getRawPointer(this->patTypes());
		    fn1.maxLength   = this->precedingLayer().curMaxSeqLength();
		    fn1.sourceDim   = this->precedingLayer().parallelSequences() * this->size();
		    
		    thrust::for_each(
		      thrust::make_zip_iterator(
		       thrust::make_tuple(
			 m_modeChangeDataBuf.begin(),
			 thrust::counting_iterator<int>(0))),
		      thrust::make_zip_iterator(
		       thrust::make_tuple(
			 m_modeChangeDataBuf.begin()        + timeLength,
			 thrust::counting_iterator<int>(0)  + timeLength)),
		      fn1);
		}
		m_modeChangeDataBuf.swap(this->precedingLayer().outputs());
	    }
	
	
	    // for HNM special training mode (not used anymore)
	    if (m_hnm_flag == DFTERRORPOST_HNM_MODEL_1 || m_hnm_flag == DFTERRORPOST_HNM_MODEL_2){
		// if this layer is valid for HNM special mode
		if (m_noiseOutputLayer && m_f0InputLayer &&
		    this->getCurrTrainingEpoch() < m_noiseTrain_epoch &&
		    nnState == NN_STATE_GAN_NOGAN_TRAIN) {
		
		    // remove the voiced part in target waveforms based on U/V infor
		    {
		    internal::TimeDomainRemoveWaveformVoiced fn1;
		    fn1.f0TimeResolution = m_f0InputLayer->getResolution();
		    fn1.f0InputLayerDim  = m_f0InputLayer->size();
		    fn1.f0DataBuffer     = helpers::getRawPointer(m_f0InputLayer->outputs());
		    fn1.waveformLength   = timeLength;
		    fn1.f0DataM          = m_f0DataM;
		    fn1.f0DataS          = m_f0DataS;
		    fn1.featDim          = this->__vSize();
		    fn1.patTypes         = helpers::getRawPointer(this->patTypes());
		    thrust::for_each(
		      thrust::make_zip_iterator(
		       thrust::make_tuple(
			 this->outputs().begin(),
			 thrust::counting_iterator<int>(0))),
		      thrust::make_zip_iterator(
		       thrust::make_tuple(
			 this->outputs().begin() + timeLength * this->__vSize(),
			 thrust::counting_iterator<int>(0)+timeLength*this->__vSize())),
		      fn1);
		    }
		
		    // copy the noise output as the generated waveforms
		    //  (in order to calculate the error on the unvoiced regions only)
		    thrust::copy(m_noiseOutputLayer->outputs().begin(),
				 m_noiseOutputLayer->outputs().end(),
				 this->precedingLayer().outputs().begin());
		}
	    }
	
	    // Compute waveform MSE if necessary 
	    if (m_beta > 0.0){
	    
		internal::ComputeMseWaveform fn;
		fn.layerSize = this->__vSize();
		fn.patTypes  = helpers::getRawPointer(this->patTypes());

		m_mseError =
		(real_t)thrust::transform_reduce(
		   thrust::make_zip_iterator(
		      thrust::make_tuple(
			 this->outputs().begin(),
			 this->precedingLayer().outputs().begin(),   
			 thrust::counting_iterator<int>(0))),
		   thrust::make_zip_iterator(
		     thrust::make_tuple(
			 this->outputs().begin() + timeLength * this->__vSize(),
			 this->precedingLayer().outputs().begin() + timeLength*this->__vSize(),
			 thrust::counting_iterator<int>(0) + timeLength * this->__vSize())),
		   fn, (real_t)0, thrust::plus<real_t>()) / timeLength;	
	    }else{
		m_mseError = 0.0;
	    }

	
	    // Compute DFT amplitute and phase distance if necessary
	    if (m_gamma > 0.0 || m_zeta > 0.0){
   
		// FFT 1
		// step0. build the data structure
		helpers::FFTMat<TDevice> sourceSig(
			&this->_actualOutputs(), &this->m_fftSourceFramed,
			&this->m_fftSourceSigFFT,
			m_frameLength, m_frameShift, m_windowType, m_fftLength, m_fftBinsNum,
			m_frameNum, this->__vMaxSeqLength(), timeLength,
			m_specDisType);

		helpers::FFTMat<TDevice> targetSig(
			&this->_targets(), &this->m_fftTargetFramed,
			&this->m_fftTargetSigFFT,
			m_frameLength, m_frameShift, m_windowType, m_fftLength, m_fftBinsNum,
			m_frameNum, this->__vMaxSeqLength(), timeLength,
			m_specDisType);

		helpers::FFTMat<TDevice> fftDiffSig(
			&this->m_fftDiffData, &this->m_fftDiffFramed,
			&this->m_fftDiffSigFFT,
			m_frameLength, m_frameShift, m_windowType, m_fftLength, m_fftBinsNum,
			m_frameNum, this->__vMaxSeqLength(), timeLength,
			m_specDisType);
		
		// step1. framing and windowing
		sourceSig.frameSignal();
		targetSig.frameSignal();
		
		// step2. fft
		sourceSig.FFT();
		targetSig.FFT();

		/* When phase and amplitude use the same FFTMats
		// -- phase part
		if (m_zeta > 0.0){
		// calculate phase distortion
		m_phaseError = sourceSig.specPhaseDistance(targetSig, fftDiffSig);
		// compute complex-valued grad vector
		fftDiffSig.specPhaseGrad(sourceSig, targetSig);
		// inverse DFT
		fftDiffSig.iFFT();
		// de-framing/windowing
		fftDiffSig.collectGrad(m_zeta);
		// copy the gradients to the phase grad buffer
		m_fftDiffDataPhase = m_fftDiffData;
		}else{
		m_phaseError = 0;
		}
		*/
		
		// -- amplitude part
		m_specError = sourceSig.specAmpDistance(targetSig, fftDiffSig);
		// compute complex-valued grad vector
		fftDiffSig.specAmpGrad(sourceSig, targetSig);
		// inverse DFT
		fftDiffSig.iFFT();
		// de-framing/windowing
		fftDiffSig.collectGrad(m_gamma);
		// Gradients should be in m_fftDiffData		    

	    
		// -- phase part
		if (m_zeta > 0.0){
		    // FFT 1
		    // step0. build the data structure
		    helpers::FFTMat<TDevice> sourceSigPhase(
			&this->_actualOutputs(), &this->m_fftSourceFramed,
			&this->m_fftSourceSigFFT,
			m_frameLength, m_frameShift, m_windowTypePhase, m_fftLength, m_fftBinsNum,
			m_frameNum, this->__vMaxSeqLength(), timeLength,
			m_specDisType);

		    helpers::FFTMat<TDevice> targetSigPhase(
			&this->_targets(), &this->m_fftTargetFramed,
			&this->m_fftTargetSigFFT,
			m_frameLength, m_frameShift, m_windowTypePhase, m_fftLength, m_fftBinsNum,
			m_frameNum, this->__vMaxSeqLength(), timeLength,
			m_specDisType);

		    helpers::FFTMat<TDevice> fftDiffSigPhase(
			&this->m_fftDiffDataPhase, &this->m_fftDiffFramed,
			&this->m_fftDiffSigFFT,
			m_frameLength, m_frameShift, m_windowTypePhase, m_fftLength, m_fftBinsNum,
			m_frameNum, this->__vMaxSeqLength(), timeLength,
			m_specDisType);
		
		    // step1. framing and windowing
		    sourceSigPhase.frameSignal();
		    targetSigPhase.frameSignal();
		
		    // step2. fft
		    sourceSigPhase.FFT();
		    targetSigPhase.FFT();

		    // calculate phase distortion
		    m_phaseError = sourceSigPhase.specPhaseDistance(targetSigPhase,
								    fftDiffSigPhase);
		    // compute complex-valued grad vector
		    fftDiffSigPhase.specPhaseGrad(sourceSigPhase, targetSigPhase);
		    // inverse DFT
		    fftDiffSigPhase.iFFT();
		    // de-framing/windowing, grad will be in m_fftDiffDataPhase
		    fftDiffSigPhase.collectGrad(m_zeta);
		}else{
		    m_phaseError = 0;
		}
		    
		
		// FFT 2
		if (m_fftLength2){
		    helpers::FFTMat<TDevice> sourceSig2(
			&this->_actualOutputs(), &this->m_fftSourceFramed2,
			&this->m_fftSourceSigFFT2,
			m_frameLength2, m_frameShift2, m_windowType2, m_fftLength2, m_fftBinsNum2,
			m_frameNum2, this->__vMaxSeqLength(), timeLength,
			m_specDisType);

		    helpers::FFTMat<TDevice> targetSig2(
			&this->_targets(), &this->m_fftTargetFramed2,
			&this->m_fftTargetSigFFT2,
			m_frameLength2, m_frameShift2, m_windowType2, m_fftLength2, m_fftBinsNum2,
			m_frameNum2, this->__vMaxSeqLength(), timeLength,
			m_specDisType);
		    
		    helpers::FFTMat<TDevice> fftDiffSig2(
			&this->m_fftDiffData2, &this->m_fftDiffFramed2,
			&this->m_fftDiffSigFFT2,
			m_frameLength2, m_frameShift2, m_windowType2, m_fftLength2, m_fftBinsNum2,
			m_frameNum2, this->__vMaxSeqLength(), timeLength,
			m_specDisType);
		
		    sourceSig2.frameSignal();
		    targetSig2.frameSignal();
		    sourceSig2.FFT();
		    targetSig2.FFT();

		    /*
		    // -- phase part
		    if (m_zeta > 0.0){
		    m_phaseError2 = sourceSig2.specPhaseDistance(targetSig2, fftDiffSig2);
		    fftDiffSig2.specPhaseGrad(sourceSig2, targetSig2);
		    fftDiffSig2.iFFT();
		    fftDiffSig2.collectGrad(m_zeta);
		    m_fftDiffDataPhase2 = m_fftDiffData2;
		    }else{
		    m_phaseError2 = 0;
		    }*/
		    
		    m_specError2 = sourceSig2.specAmpDistance(targetSig2, fftDiffSig2);
		    fftDiffSig2.specAmpGrad(sourceSig2, targetSig2);
		    fftDiffSig2.iFFT();
		    fftDiffSig2.collectGrad(m_gamma);

		    if (m_zeta > 0.0){
			helpers::FFTMat<TDevice> sourceSigPhase2(
			  &this->_actualOutputs(), &this->m_fftSourceFramed2,
			  &this->m_fftSourceSigFFT2,
			  m_frameLength2, m_frameShift2, m_windowTypePhase2,
			  m_fftLength2, m_fftBinsNum2,
			  m_frameNum2, this->__vMaxSeqLength(), timeLength,
			  m_specDisType);

			helpers::FFTMat<TDevice> targetSigPhase2(
			  &this->_targets(), &this->m_fftTargetFramed2,
			  &this->m_fftTargetSigFFT2,
			  m_frameLength2, m_frameShift2, m_windowTypePhase2,
			  m_fftLength2, m_fftBinsNum2,
			  m_frameNum2, this->__vMaxSeqLength(), timeLength,
			  m_specDisType);
		    
			helpers::FFTMat<TDevice> fftDiffSigPhase2(
			  &this->m_fftDiffDataPhase2, &this->m_fftDiffFramed2,
			  &this->m_fftDiffSigFFT2,
			  m_frameLength2, m_frameShift2, m_windowTypePhase2,
			  m_fftLength2, m_fftBinsNum2,
			  m_frameNum2, this->__vMaxSeqLength(), timeLength,
			  m_specDisType);
		
			sourceSigPhase2.frameSignal();
			targetSigPhase2.frameSignal();
			sourceSigPhase2.FFT();
			targetSigPhase2.FFT();

			m_phaseError2 = sourceSigPhase2.specPhaseDistance(targetSigPhase2,
									  fftDiffSigPhase2);
			fftDiffSigPhase2.specPhaseGrad(sourceSigPhase2, targetSigPhase2);
			fftDiffSigPhase2.iFFT();
			fftDiffSigPhase2.collectGrad(m_zeta);
		    }else{
			m_phaseError2 = 0;
		    }
		}else{
		    m_specError2 = 0.0;
		    m_phaseError2 = 0.0;
		}

		// FFT 3
		if (m_fftLength3){
		
		    helpers::FFTMat<TDevice> sourceSig3(
			&this->_actualOutputs(), &this->m_fftSourceFramed3,
			&this->m_fftSourceSigFFT3,
			m_frameLength3, m_frameShift3, m_windowType3, m_fftLength3, m_fftBinsNum3,
			m_frameNum3, this->__vMaxSeqLength(), timeLength,
			m_specDisType);

		    helpers::FFTMat<TDevice> targetSig3(
			&this->_targets(), &this->m_fftTargetFramed3,
			&this->m_fftTargetSigFFT3,
			m_frameLength3, m_frameShift3, m_windowType3, m_fftLength3, m_fftBinsNum3,
			m_frameNum3, this->__vMaxSeqLength(), timeLength,
			m_specDisType);
		    
		    helpers::FFTMat<TDevice> fftDiffSig3(
			&this->m_fftDiffData3, &this->m_fftDiffFramed3,
			&this->m_fftDiffSigFFT3,
			m_frameLength3, m_frameShift3, m_windowType3, m_fftLength3, m_fftBinsNum3,
			m_frameNum3, this->__vMaxSeqLength(), timeLength,
			m_specDisType);
		    
		    sourceSig3.frameSignal();
		    targetSig3.frameSignal();
		    sourceSig3.FFT();
		    targetSig3.FFT();

		    /*
		    // -- phase part
		    if (m_zeta > 0.0){
		    m_phaseError3 = sourceSig3.specPhaseDistance(targetSig3, fftDiffSig3);
		    fftDiffSig3.specPhaseGrad(sourceSig3, targetSig3);
		    fftDiffSig3.iFFT();
		    fftDiffSig3.collectGrad(m_zeta);
		    m_fftDiffDataPhase3 = m_fftDiffData3;
		    }else{
		    m_phaseError3 = 0;
		    }*/
		    
		    m_specError3 = sourceSig3.specAmpDistance(targetSig3, fftDiffSig3);
		    fftDiffSig3.specAmpGrad(sourceSig3, targetSig3);
		    fftDiffSig3.iFFT();
		    fftDiffSig3.collectGrad(m_gamma);

		
		    if (m_zeta > 0.0){
			helpers::FFTMat<TDevice> sourceSigPhase3(
			  &this->_actualOutputs(), &this->m_fftSourceFramed3,
			  &this->m_fftSourceSigFFT3,
			  m_frameLength3, m_frameShift3, m_windowTypePhase3,
			  m_fftLength3, m_fftBinsNum3,
			  m_frameNum3, this->__vMaxSeqLength(), timeLength,
			  m_specDisType);

			helpers::FFTMat<TDevice> targetSigPhase3(
			  &this->_targets(), &this->m_fftTargetFramed3,
			  &this->m_fftTargetSigFFT3,
			  m_frameLength3, m_frameShift3, m_windowTypePhase3,
			  m_fftLength3, m_fftBinsNum3,
			  m_frameNum3, this->__vMaxSeqLength(), timeLength,
			  m_specDisType);
		    
			helpers::FFTMat<TDevice> fftDiffSigPhase3(
			  &this->m_fftDiffDataPhase3, &this->m_fftDiffFramed3,
			  &this->m_fftDiffSigFFT3,
			  m_frameLength3, m_frameShift3, m_windowTypePhase3,
			  m_fftLength3, m_fftBinsNum3,
			  m_frameNum3, this->__vMaxSeqLength(), timeLength,
			  m_specDisType);
		    
			sourceSigPhase3.frameSignal();
			targetSigPhase3.frameSignal();
			sourceSigPhase3.FFT();
			targetSigPhase3.FFT();

			m_phaseError3 = sourceSigPhase3.specPhaseDistance(targetSigPhase3,
								      fftDiffSigPhase3);
			fftDiffSigPhase3.specPhaseGrad(sourceSigPhase3, targetSigPhase3);
			fftDiffSigPhase3.iFFT();
			fftDiffSigPhase3.collectGrad(m_zeta);

		    }else{
			m_phaseError3 = 0;
		    }
		}else{
		    m_specError3 = 0.0;
		    m_phaseError3 = 0.0;
		}
	    }else{
		m_specError = 0.0;
		m_phaseError = 0.0;
		m_specError2 = 0.0;
		m_phaseError2 = 0.0;
		m_specError3 = 0.0;
		m_phaseError3 = 0.0;
	    }
	    // Done
	    return;
	}
	// Done
    }

    template <typename TDevice>
    void DFTPostoutputLayer<TDevice>::computeForwardPass(const int timeStep, const int nnState)
    {
	// Not implemented yet
	// throw std::runtime_error("Not implemented DFTErrorPostoutput computeForwardPass(timeStep)");
    }


    template <typename TDevice>
    void DFTPostoutputLayer<TDevice>::computeBackwardPass(const int nnState)
    {
	if (this->getSaveMemoryFlag())
	    throw std::runtime_error("Memory save mode should be turned off");

	// length of the training utterance
	int timeLength = this->__vCurMaxSeqLength();
	
	// initialize the gradients buffer
	thrust::fill(this->precedingLayer().outputErrors().begin(),
		     this->precedingLayer().outputErrors().end(), 0.0);
	    
	// Gradients from waveform MSE
	if (m_beta > 0.0){
	    {{
		internal::ComputeMseWaveformGrad fn2;
		fn2.preData  = helpers::getRawPointer(this->precedingLayer().outputs());
		fn2.realTargetData = helpers::getRawPointer(this->_targets());
		fn2.preDim   = this->__vSize();
		fn2.featDim  = this->__vSize();
		fn2.beta     = m_beta;
		fn2.patTypes = helpers::getRawPointer(this->patTypes());
	    
		fn2.accumulate = false;
		fn2.preError   = helpers::getRawPointer(this->precedingLayer().outputErrors());
		
	    
		thrust::for_each(
		  thrust::make_zip_iterator(
		   thrust::make_tuple(
		     this->outputs().begin(),
		     thrust::counting_iterator<int>(0))),
		  thrust::make_zip_iterator(
		   thrust::make_tuple(
		     this->outputs().begin()   + timeLength * this->__vSize(),
		     thrust::counting_iterator<int>(0)  + timeLength * this->__vSize())),
		  fn2);
	     }}	
	}

	// Gradients from spectral amplitude and phase
	//  gradients have been calculated in computeForwardPass()
	//  here, gradients are simply accumulated into the gradient buffer
	if (m_gamma > 0.0){
	    
	    // FFT1
	    thrust::transform(m_fftDiffData.begin(),
			      m_fftDiffData.begin() + timeLength * this->__vSize(),
			      this->precedingLayer().outputErrors().begin(),
			      this->precedingLayer().outputErrors().begin(),
			      thrust::plus<real_t>());
	    if (m_zeta > 0.0)
		thrust::transform(m_fftDiffDataPhase.begin(),
				  m_fftDiffDataPhase.begin() + timeLength * this->__vSize(),
				  this->precedingLayer().outputErrors().begin(),
				  this->precedingLayer().outputErrors().begin(),
				  thrust::plus<real_t>());
	    
	    // FFT2
	    if (m_fftLength2){
		thrust::transform(m_fftDiffData2.begin(),
			      m_fftDiffData2.begin() + timeLength * this->__vSize(),
			      this->precedingLayer().outputErrors().begin(),
			      this->precedingLayer().outputErrors().begin(),
			      thrust::plus<real_t>());
		if (m_zeta > 0.0)
		    thrust::transform(m_fftDiffDataPhase2.begin(),
				  m_fftDiffDataPhase2.begin() + timeLength * this->__vSize(),
				  this->precedingLayer().outputErrors().begin(),
				  this->precedingLayer().outputErrors().begin(),
				  thrust::plus<real_t>());
	    }

	    // FFT3
	    if (m_fftLength3){
		thrust::transform(m_fftDiffData3.begin(),
			      m_fftDiffData3.begin() + timeLength * this->__vSize(),
			      this->precedingLayer().outputErrors().begin(),
			      this->precedingLayer().outputErrors().begin(),
			      thrust::plus<real_t>());
		if (m_zeta > 0.0)
		    thrust::transform(m_fftDiffDataPhase3.begin(),
				  m_fftDiffDataPhase3.begin() + timeLength * this->__vSize(),
				  this->precedingLayer().outputErrors().begin(),
				  this->precedingLayer().outputErrors().begin(),
				  thrust::plus<real_t>());
	    }
	}
	
	// For HNM special training mode
	if (m_hnm_flag == DFTERRORPOST_HNM_MODEL_1 || m_hnm_flag == DFTERRORPOST_HNM_MODEL_2){
	    
	    if (m_noiseOutputLayer && m_f0InputLayer &&
		this->getCurrTrainingEpoch() < m_noiseTrain_epoch &&
		nnState == NN_STATE_GAN_NOGAN_TRAIN) {
		
		// copy the gradients w.r.t noise component to the noise output layer
		thrust::copy(this->precedingLayer().outputErrors().begin(),
			     this->precedingLayer().outputErrors().end(),
			     m_noiseOutputLayer->outputErrors().begin());

		// Set the gradients w.r.t harmonic to zero (by setting this->precedingLayer())
		//  because m_noiseOutputLayer will be a skip-layer,
		//  this->precedingLayer send 0 to m_noiseOutputLayer.outputErrorFromSkipLayers,
		//  the gradients from DFT will be kept in m_noiseOutputLayer.outputErrors()
		thrust::fill(this->precedingLayer().outputErrors().begin(),
			     this->precedingLayer().outputErrors().end(),
			     0.0);
	    }

	    // for special mode 2
	    //  Harmonic and noise part are trained separatedly
	    //  When training on harmonic part, the gradients w.r.t noise component will be zero
	    if (m_noiseOutputLayer && m_f0InputLayer &&
		this->getCurrTrainingEpoch() >= m_noiseTrain_epoch &&
		nnState == NN_STATE_GAN_NOGAN_TRAIN && m_hnm_flag == DFTERRORPOST_HNM_MODEL_2){
		// copy the gradients w.r.t noise component to the noise output layer
		thrust::copy(this->precedingLayer().outputErrors().begin(),
			     this->precedingLayer().outputErrors().end(),
			     m_noiseOutputLayer->outputErrors().begin());

		// set the gradients w.r.t noise component to zero
		//  this is done by the simple trick below
		//  this trick cancles the gradients when outputSkipErrors + outputErrors for
		//  noise component
		thrust::negate<real_t> op;
		thrust::transform(m_noiseOutputLayer->outputErrors().begin(),
				  m_noiseOutputLayer->outputErrors().end(),
				  m_noiseOutputLayer->outputErrors().begin(),
				  op);
	    }
	}

	// For multi-dim signal case
	if (m_modeMultiDimSignal == DFTMODEFORMULTIDIMSIGNAL_CONCATE){
	    // re-store the generated data
	    m_modeChangeDataBuf.swap(this->precedingLayer().outputs());

	    // gradient of multiDimSignaltoOneDim
	    // convert target data
		{
		    internal::multiDimSignaltoOneDimGrad fn1;
		    fn1.sourceData  = helpers::getRawPointer(this->precedingLayer().outputErrors());
		    fn1.patTypes    = helpers::getRawPointer(this->patTypes());
		    fn1.maxLength   = this->precedingLayer().curMaxSeqLength();
		    fn1.sourceDim   = this->precedingLayer().parallelSequences() * this->size();
		    
		    thrust::for_each(
		      thrust::make_zip_iterator(
		       thrust::make_tuple(
			 m_modeChangeDataBuf.begin(),
			 thrust::counting_iterator<int>(0))),
		      thrust::make_zip_iterator(
		       thrust::make_tuple(
			 m_modeChangeDataBuf.begin()        + timeLength,
			 thrust::counting_iterator<int>(0)  + timeLength)),
		      fn1);
		}
		
		thrust::copy(m_modeChangeDataBuf.begin(), m_modeChangeDataBuf.end(),
			     this->precedingLayer().outputErrors().begin());

	}
	
    }
	
    
    template <typename TDevice>
    const std::string& DFTPostoutputLayer<TDevice>::type() const
    {
	static std::string s1("dft");
        return s1;
    }

    template <typename TDevice>
    void DFTPostoutputLayer<TDevice>::exportLayer(const helpers::JsonValue &layersArray,
						  const helpers::JsonAllocator &allocator) const
    {
	PostOutputLayer<TDevice>::exportLayer(layersArray, allocator);
	    
	if (m_beta > 0.0)
	    (*layersArray)[layersArray->Size() - 1].AddMember("beta", m_beta, allocator);

	if (m_specDisType != FFTMAT_SPECTYPE_MSE)
	    (*layersArray)[layersArray->Size() - 1].AddMember("specDisType", m_specDisType,
							      allocator);
	if (m_preEmphasis)
	    (*layersArray)[layersArray->Size() - 1].AddMember("preEmphasisNaturalWav", m_preEmphasis,
							      allocator);
	
	if (m_gamma > 0.0){
	    (*layersArray)[layersArray->Size() - 1].AddMember("gamma", m_gamma, allocator);
	    (*layersArray)[layersArray->Size() - 1].AddMember("zeta", m_zeta, allocator);
	    (*layersArray)[layersArray->Size() - 1].AddMember("fftLength", m_fftLength,
							      allocator);
	    (*layersArray)[layersArray->Size() - 1].AddMember("frameLength", m_frameLength,
							      allocator);
	    (*layersArray)[layersArray->Size() - 1].AddMember("frameShift", m_frameShift,
							      allocator);

	    (*layersArray)[layersArray->Size() - 1].AddMember("windowType", m_windowType,
							      allocator);
	    
	    (*layersArray)[layersArray->Size() - 1].AddMember("windowTypePhase",
							      m_windowTypePhase,
							      allocator);

	    
	    if (m_fftLength2){
		(*layersArray)[layersArray->Size() - 1].AddMember("fftLength2", m_fftLength2,
								  allocator);
		(*layersArray)[layersArray->Size() - 1].AddMember("frameLength2", m_frameLength2,
								  allocator);
		(*layersArray)[layersArray->Size() - 1].AddMember("frameShift2", m_frameShift2,
								  allocator);
		(*layersArray)[layersArray->Size() - 1].AddMember("windowType2", m_windowType2,
								  allocator);
		(*layersArray)[layersArray->Size() - 1].AddMember("windowTypePhase2",
								  m_windowTypePhase2,
								  allocator);

	    }
	    
	    if (m_fftLength3){
		(*layersArray)[layersArray->Size() - 1].AddMember("fftLength3", m_fftLength3,
								  allocator);
		(*layersArray)[layersArray->Size() - 1].AddMember("frameLength3", m_frameLength3,
								  allocator);
		(*layersArray)[layersArray->Size() - 1].AddMember("frameShift3", m_frameShift3,
								  allocator);
		(*layersArray)[layersArray->Size() - 1].AddMember("windowType3", m_windowType3,
								  allocator);
		(*layersArray)[layersArray->Size() - 1].AddMember("windowTypePhase3",
								  m_windowTypePhase3,
								  allocator);

	    }

	    if (m_hnm_flag > 0 ){
		(*layersArray)[layersArray->Size() - 1].AddMember("hnmMode", m_hnm_flag, allocator);
		(*layersArray)[layersArray->Size() - 1].AddMember("noisePartTrainEpochNum",
								  m_noiseTrain_epoch, allocator);
		(*layersArray)[layersArray->Size() - 1].AddMember("noiseOutputLayerName",
								  m_noiseOutputLayerName.c_str(),
								  allocator);
		(*layersArray)[layersArray->Size() - 1].AddMember("f0InputLayerName",
								  m_f0InputLayerName.c_str(),
								  allocator);
		(*layersArray)[layersArray->Size() - 1].AddMember("f0DataMean",
								  m_f0DataM, allocator);
		(*layersArray)[layersArray->Size() - 1].AddMember("f0DataStd",
								  m_f0DataS, allocator);
		
	    }

	    if (m_modeMultiDimSignal != DFTMODEFORMULTIDIMSIGNAL_NONE){
		(*layersArray)[layersArray->Size() - 1].AddMember("multiDimSignalMode",
								  m_modeMultiDimSignal, allocator);
	
	    }
	}   
    }
        
    template <typename TDevice>
    real_t DFTPostoutputLayer<TDevice>::calculateError()
    {
	// calculation has been down in forward pass
	// just print and return the results
	if (Configuration::instance().verboseLevel() == OP_VERBOSE_LEVEL_1){
	    std::cerr << m_specError << ", " << m_specError2 << ", " << m_specError3;
	    std::cerr << ", " << m_mseError << ", 0" << ", ";
	    if (m_zeta > 0.0){
		std::cerr << m_phaseError << ", " << m_phaseError2 << ", ";
		std::cerr << m_phaseError3 << ", ";
	    }
	}
	return (m_specError + m_specError2 + m_specError3 +
		m_phaseError + m_phaseError2 + m_phaseError3);
	    
    }

    template <typename TDevice>
    void DFTPostoutputLayer<TDevice>::linkTargetLayer(Layer<TDevice> &targetLayer)
    {
	// for HNM model, link the noise outputlayer and F0 input layer
	if (m_hnm_flag > 0){
	    // for noise output layer
	    if (targetLayer.name() == m_noiseOutputLayerName){
		// to be done: assert that it is a skip-layer
		if (targetLayer.type() != "skipini" && targetLayer.type() != "skipadd")
		    throw std::runtime_error("noiseOutputLayer is not a skipini/skipadd layer");
		// assert layer size
		if (targetLayer.size() != this->__vSize())
		    throw std::runtime_error("noiseOutputLayer layer size != DFT layer size");
		
		m_noiseOutputLayer = &targetLayer;
		printf("\n\tDFT layer get noise output from %s", m_noiseOutputLayer->name().c_str());
	    }
	    
	    // for F0 input layer
	    if (targetLayer.name() == m_f0InputLayerName){
		m_f0InputLayer = &targetLayer;
		printf("\n\tDFT layer get F0 infor from %s", m_f0InputLayer->name().c_str());
		printf(", assume last dimension of its output as F0");
	    }
	}
	return;
    }

    template <typename TDevice>
    void DFTPostoutputLayer<TDevice>::computeBackwardPass(const int timeStep, const int nnState)
    {
	if (timeStep == this->curMaxSeqLength())
	    this->computeBackwardPass(nnState);
    }
    
    template <typename TDevice>
    int  DFTPostoutputLayer<TDevice>::__vSize()
    {
	if (m_modeMultiDimSignal == DFTMODEFORMULTIDIMSIGNAL_CONCATE)
	    return 1;
	else
	    return this->size();
    }

    template <typename TDevice>
    int  DFTPostoutputLayer<TDevice>::__vMaxSeqLength()
    {
	if (m_modeMultiDimSignal == DFTMODEFORMULTIDIMSIGNAL_CONCATE)
	    return this->maxSeqLength() * this->size() * this->parallelSequences();
	else
	    return this->maxSeqLength() * this->parallelSequences();
    }


    template <typename TDevice>
    int  DFTPostoutputLayer<TDevice>::__vCurMaxSeqLength()
    {
	if (m_modeMultiDimSignal == DFTMODEFORMULTIDIMSIGNAL_CONCATE)
	    return this->curMaxSeqLength() * this->size()  * this->parallelSequences();
	else
	    return this->curMaxSeqLength() * this->parallelSequences();
    } 
    
    template class DFTPostoutputLayer<Cpu>;
    template class DFTPostoutputLayer<Gpu>;
    
}
