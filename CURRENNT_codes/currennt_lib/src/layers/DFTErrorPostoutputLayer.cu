/******************************************************************************
 * This file is an addtional component of CURRENNT. 
 * Xin WANG
 * National Institute of Informatics, Japan
 * 2016 - 2019
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
#include "../helpers/sigProcess.hpp"
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

// Configuration for HNM training (obsolete)
#define DFTERRORPOST_HNM_MODEL_1 1
#define DFTERRORPOST_HNM_MODEL_2 2
#define DFTERRORUV       10
#define DFTERRORUVSEARCH 2
#define DFTERROR_PI      3.141215

// Configuration for Multi-dimensional signal analysis
#define DFTMODEFORMULTIDIMSIGNAL_NONE    0
#define DFTMODEFORMULTIDIMSIGNAL_CONCATE 1

// Configuration for waveform pre-emphasis & de-emphasis
#define DFTPREEMPHASISCOEFF 0.94 // emphasis factor

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

    struct cleanImagPart
    {
	__host__ __device__ void operator() (complex_t &t) const
	{
	    t.y = 0;
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
      , m_beta           (0.0)
      , m_gamma          (0.0)
      , m_zeta           (0.0)
      , m_eta            (0.0)
      , m_kappa          (0.0)
      , m_tau            (0.0)
      , m_iota           (0.0)
      , m_mseError       (0.0)
      , m_noiseOutputLayer (NULL)
      , m_f0InputLayer     (NULL)
      , m_sineInputLayer_ptr (NULL)
      , m_f0InputLayer_ptr (NULL)
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
    void DFTPostoutputLayer<TDevice>::__cleanDFTError(struct_DFTData &dftBuf)
    {
	// clean the errors
	dftBuf.m_specError = 0.0;
	dftBuf.m_phaseError = 0.0;
	dftBuf.m_resError = 0.0;
	dftBuf.m_realSpecError = 0.0;
	dftBuf.m_lpcError = 0.0;
	dftBuf.m_specErrorOthers = 0.0;
    }
    
    template <typename TDevice>
    void DFTPostoutputLayer<TDevice>::__initDFTBuffer(struct_DFTData &dftBuf)
    {
	// clean the buffers for short-time analysis
	dftBuf.m_valid_flag = false;
	
	this->__cleanDFTError(dftBuf);
	
	dftBuf.m_frameNum = 0;
	dftBuf.m_fftBinsNum = 0;
	dftBuf.m_frameLength = 0;
	dftBuf.m_frameShift = 0;
	dftBuf.m_windowType = 0;
	dftBuf.m_windowTypePhase = 0;
	dftBuf.m_fftLength = 0;

	dftBuf.m_fftLengthRealSpec = 0;
	dftBuf.m_fftBinsNumRealSpec = 0;

	dftBuf.m_lpcOrder = 0;
	
	dftBuf.m_fftSourceFramed.clear();
	dftBuf.m_fftTargetFramed.clear();
	dftBuf.m_fftSourceSigFFT.clear();
	dftBuf.m_fftTargetSigFFT.clear();
	
	dftBuf.m_fftDiffSigFFT.clear();
	dftBuf.m_fftDiffFramed.clear();
	dftBuf.m_fftDiffData.clear();
	dftBuf.m_fftDiffDataPhase.clear();
	dftBuf.m_fftResData.clear();
	
	dftBuf.m_fftSourceFramedRealSpec.clear();
	dftBuf.m_fftSourceSigFFTRealSpec.clear();
	dftBuf.m_fftTargetFramedRealSpec.clear();
	dftBuf.m_fftTargetSigFFTRealSpec.clear();
	
	dftBuf.m_fftDiffDataRealSpec.clear();
	dftBuf.m_fftDiffFramedRealSpec.clear();
	dftBuf.m_fftDiffSigFFTRealSpec.clear();

	dftBuf.m_fftMaskFFT.clear();
	dftBuf.m_fftMaskSignalFramed.clear();
	dftBuf.m_specWeight.clear();
	
	dftBuf.m_autoCorrSrc.clear();
	dftBuf.m_lpcCoefSrc.clear();
	dftBuf.m_lpcErrSrc.clear();
	dftBuf.m_refCoefSrc.clear();
	
	dftBuf.m_autoCorrTar.clear();
	dftBuf.m_lpcCoefTar.clear();
	dftBuf.m_lpcErrTar.clear();
	dftBuf.m_refCoefTar.clear();

	dftBuf.m_lpcGrad.clear();
	
	dftBuf.m_specGrad_others.clear();
	dftBuf.m_specGrad_tmpBuf.clear();
    }
    
    template <typename TDevice>
    void DFTPostoutputLayer<TDevice>::__configDFTBuffer(struct_DFTData &dftBuf,
							const int fftLength,
							const int frameLength,
							const int frameShift,
							const int windowType,
							const int windowTypePhase,
							const int lpcOrder,
							const std::string specWeightVec)
    {
	
	//  make m_fftLength an even number
	int tmp_fftLength = (fftLength / 2) * 2;

	if (tmp_fftLength <= 0){
	    
	    // invalid DFT buffer
	    dftBuf.m_valid_flag = false;
	    return;
	    
	}else{
	    
	    // valid DFT buffer
	    dftBuf.m_valid_flag = true;

	    // configuration
	    dftBuf.m_fftLength = tmp_fftLength;
	    dftBuf.m_frameLength = frameLength;
	    dftBuf.m_frameShift = frameShift;
	    dftBuf.m_windowType = windowType;
	    dftBuf.m_lpcOrder = lpcOrder;

	    printf("\n\tDFT analysis: fft_point: %d, frame_length: %d, frame_shift: %d",
		   dftBuf.m_fftLength, dftBuf.m_frameLength, dftBuf.m_frameShift);
	    printf("\n\t\tspectral amplitude distance [weight: %.3f]", m_gamma);
	    // check
	    if (frameLength > tmp_fftLength)
		throw std::runtime_error("\nFFT length should be > frame length");

	    if (lpcOrder > frameLength)
		throw std::runtime_error("\nLPC order is too large");

	    // Maximum length of the waveforms in the training/test set
	    int maxSeqLength = this->__vMaxSeqLength();
	    
	    // number of fftBins (dimension of FFT feature) actually stored
	    dftBuf.m_fftBinsNum  = helpers::fftTools::fftBinsNum(dftBuf.m_fftLength);

	    // how many frames at maximum
	    dftBuf.m_frameNum = helpers::fftTools::fftFrameNum(maxSeqLength,
							       dftBuf.m_frameLength,
							       dftBuf.m_frameShift);

	    // allocate memory for FFT buffers
	    complex_t tmp;
	    tmp.x = 0;
	    tmp.y = 0;
	    
	    // framed signal buffer (with zero padding)
	    //   buffer for framed source signal
	    dftBuf.m_fftSourceFramed.resize(dftBuf.m_frameNum * dftBuf.m_fftLength, 0.0);
	    //   buffer for framed target signal
	    dftBuf.m_fftTargetFramed.resize(dftBuf.m_frameNum * dftBuf.m_fftLength, 0.0);
	    //   buffer for framed gradients
	    dftBuf.m_fftDiffFramed = dftBuf.m_fftTargetFramed;

	    //   buffer for STFT of source signal (complex number)
	    dftBuf.m_fftSourceSigFFT.resize(dftBuf.m_frameNum * dftBuf.m_fftBinsNum, tmp);
	    //   buffer for STFT of target signal
	    dftBuf.m_fftTargetSigFFT.resize(dftBuf.m_frameNum * dftBuf.m_fftBinsNum, tmp);
	    //   buffer for gradients (which is to be iSTFT into the 'framed gradients')
	    dftBuf.m_fftDiffSigFFT = dftBuf.m_fftTargetSigFFT;

	    // A temporary buffer to store the gradients
	    dftBuf.m_fftDiffData = this->outputs();

	    
	    // gradient buffer phase distance
	    //  other data buffers are shared with specAmp distance
	    if (this->m_zeta > 0.0){
		dftBuf.m_fftDiffDataPhase = dftBuf.m_fftDiffData;
		printf("\n\t\tphase distance [weight: %f]", m_zeta);
	    }
	    
	    // buffer for complex-valued spectral distance
	    //  other data buffers are shared with specAmp distance	    
	    if (this->m_eta > 0.0){
		dftBuf.m_fftResData = dftBuf.m_fftDiffData;
		printf("\n\t\tcomplex-valued spectral distance [weight: %f]", m_eta);
	    }
	    // buffer for real-valued spectral distance
	    if (this->m_kappa > 0.0){
		dftBuf.m_fftLengthRealSpec  = dftBuf.m_fftLength * 2; 
		dftBuf.m_fftBinsNumRealSpec = helpers::fftTools::fftBinsNum(
				dftBuf.m_fftLengthRealSpec);
		dftBuf.m_fftSourceFramedRealSpec.resize(
				dftBuf.m_frameNum * dftBuf.m_fftLengthRealSpec, 0.0);
		dftBuf.m_fftTargetFramedRealSpec.resize(
				dftBuf.m_frameNum * dftBuf.m_fftLengthRealSpec, 0.0);
		dftBuf.m_fftDiffFramedRealSpec = dftBuf.m_fftTargetFramedRealSpec;
		dftBuf.m_fftSourceSigFFTRealSpec.resize(
				dftBuf.m_frameNum * dftBuf.m_fftBinsNumRealSpec, tmp);
		dftBuf.m_fftTargetSigFFTRealSpec.resize(
				dftBuf.m_frameNum * dftBuf.m_fftBinsNumRealSpec, tmp);
		dftBuf.m_fftDiffSigFFTRealSpec = dftBuf.m_fftTargetSigFFTRealSpec;
		dftBuf.m_fftDiffDataRealSpec   = this->outputs();
		printf("\n\t\treal-valued spectral distance [weight: %f]", m_kappa);
	    }

	    // buffer for LPC analysis
	    if (this->m_tau > 0.0){
		
		//if (dftBuf.m_lpcOrder == 0)
		// throw std::runtime_error("\nLPC order should be larger than 0");
		
		dftBuf.m_autoCorrSrc.resize(dftBuf.m_frameNum * (dftBuf.m_lpcOrder + 1), 0.0);
		dftBuf.m_lpcCoefSrc.resize(dftBuf.m_frameNum * (dftBuf.m_lpcOrder + 1) * 2, 0.0);
		dftBuf.m_lpcErrSrc.resize(dftBuf.m_frameNum * (dftBuf.m_lpcOrder + 1), 0.0);
		dftBuf.m_refCoefSrc.resize(dftBuf.m_frameNum * (dftBuf.m_lpcOrder + 1), 0.0);
		dftBuf.m_lpcResSrc.resize(dftBuf.m_frameNum * dftBuf.m_fftLength, 0.0);
		
		dftBuf.m_autoCorrTar = dftBuf.m_autoCorrSrc;
		dftBuf.m_lpcCoefTar = dftBuf.m_lpcCoefSrc;
		dftBuf.m_lpcErrTar  = dftBuf.m_lpcErrSrc;
		dftBuf.m_refCoefTar = dftBuf.m_refCoefSrc;
		dftBuf.m_lpcResTar = dftBuf.m_lpcResSrc;
		
		dftBuf.m_lpcGrad = dftBuf.m_fftDiffData;
		thrust::fill(dftBuf.m_lpcGrad.begin(), dftBuf.m_lpcGrad.end(), 0.0);
		printf("\n\t\tlpc-based error [weight: %f, lpc-order: %d]", m_tau, dftBuf.m_lpcOrder);
	    }


	    // buffer for additional spectral loss
	    if (this->m_iota > 0.0){
		// waveform grad w.r.t spec-amp-distances for all hidden layers
		// one dimension for each hidden lauer
		dftBuf.m_specGrad_others.resize(this->outputs().size() * m_otherSignalInputLayer_num, 0.0);
		
		// this buffer is used to actually store the gradients
		// it will be copied to dftBuf.m_specGrad_others (in __specAmpDistanceOthers())
		dftBuf.m_specGrad_tmpBuf.resize(this->outputs().size(), 0.0);

		
		if (m_sineInputLayer_str.size()){
		    // framed signal buffer for sine source excitation
		    dftBuf.m_fftMaskSignalFramed.resize(dftBuf.m_frameNum * dftBuf.m_fftLength, 0.0);
		    
		    // FFT buffer for sine source excitation
		    dftBuf.m_fftMaskFFT.resize(dftBuf.m_frameNum * dftBuf.m_fftBinsNum, tmp);
		}
		    
		
		printf("\n\t\tspectral amplitude distance on hidden features [weight: %f]", m_iota);
	    }

	    dftBuf.m_specWeightStr = specWeightVec;
	    if (dftBuf.m_specWeightStr.size()){
		cpu_real_vector tmp;
		misFuncs::ParseFloatOpt(dftBuf.m_specWeightStr, tmp);
		dftBuf.m_specWeight = tmp;
		if (dftBuf.m_specWeight.size() != dftBuf.m_fftBinsNum){
		    printf("\n\tSpectral weight vec %s length != %d\n",
			   specWeightVec.c_str(), dftBuf.m_fftBinsNum);
		    throw std::runtime_error("Error in CURRENNT network.jsn");
		}
	    }else{
		dftBuf.m_specWeight.clear();
	    }
	    
	}		
    }
    
    template <typename TDevice>
    void DFTPostoutputLayer<TDevice>::__loadOpts(const helpers::JsonValue &layerChild)
    {
	/* ------ Load general configuration ----- 
	  Error =  m_beta * waveform_MSE + m_gamma * spectral_amplitude_MSE + 
	           m_zeta * phase_MSE    + m_eta  * residual_signal_spectral_amplitude + 
		   m_kappa * real_spectrum_amp
	*/	
	m_beta         = (layerChild->HasMember("beta") ? 
			   static_cast<real_t>((*layerChild)["beta"].GetDouble()) : 0.0);

	m_gamma        = (layerChild->HasMember("gamma") ? 
			   static_cast<real_t>((*layerChild)["gamma"].GetDouble()) : 0.0);

	m_zeta         = (layerChild->HasMember("zeta") ? 
			  static_cast<real_t>((*layerChild)["zeta"].GetDouble()) : 0.0);

	m_eta          = (layerChild->HasMember("eta") ? 
			  static_cast<real_t>((*layerChild)["eta"].GetDouble()) : 0.0);

	m_kappa        = (layerChild->HasMember("kappa") ? 
			  static_cast<real_t>((*layerChild)["kappa"].GetDouble()) : 0.0);
	
	m_tau        = (layerChild->HasMember("tau") ? 
			  static_cast<real_t>((*layerChild)["tau"].GetDouble()) : 0.0);
	
	m_iota        = (layerChild->HasMember("iota") ? 
			  static_cast<real_t>((*layerChild)["iota"].GetDouble()) : 0.0);

	// Type of spectral amplitude distance (see ../helpers/FFTMat.hpp):
	m_specDisType   = (layerChild->HasMember("specDisType") ? 
			   static_cast<real_t>((*layerChild)["specDisType"].GetInt()) :
			   FFTMAT_SPECTYPE_AMP_LOG_MSE);
	
	m_floor_log_spec_amp = (layerChild->HasMember("logSpecAmpFloor") ? 
				static_cast<real_t>((*layerChild)["logSpecAmpFloor"].GetDouble()) :
				FFTMAT_LOG_AMP_FLOOR);
	
	m_phaseDisType  = (layerChild->HasMember("phaseDisType") ? 
			   static_cast<real_t>((*layerChild)["phaseDisType"].GetInt()) :
			   FFTMAT_PHASETYPE_COS);
	// Type of real-valued spectrum
	m_realSpecType    = (layerChild->HasMember("kappa_realspec_type") ? 
			     ((*layerChild)["kappa_realspec_type"].GetInt()) :
			     FFTMAT_REALSPEC_TYPE_NORMAL);
	
	m_realSpecDisType = (layerChild->HasMember("realSpecDisType") ? 
			   static_cast<real_t>((*layerChild)["realSpecDisType"].GetInt()) :
			   FFTMAT_SPECTYPE_AMP_LOG_MSE);
	
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


	// LPC related configuration
	m_lpcErrorType = (layerChild->HasMember("lpcErrorType") ? 
			  static_cast<int>((*layerChild)["lpcErrorType"].GetInt()) :
			  SIGPROCESS_LPC_ERR_TYPE_WAV_MSE);
	m_lpcGain = (layerChild->HasMember("lpcCalculateGain") ? 
		     static_cast<int>((*layerChild)["lpcCalculateGain"].GetInt()) : 1);

	
	// sine-input layer
	m_sineInputLayer_str = (layerChild->HasMember("sineSourceLayer") ? 
				((*layerChild)["sineSourceLayer"].GetString()) : "");

	// f0 input layer
	m_f0InputLayer_str = (layerChild->HasMember("f0InputLayer") ? 
				((*layerChild)["f0InputLayer"].GetString()) : "");
	
	// ------- for additional signals from other hidden layers of the network
	m_separate_excitation_loss = (layerChild->HasMember("lpcExcitationLoss") ? 
				      static_cast<int>((*layerChild)["lpcExcitationLoss"].GetInt()):0);
	if (m_tau > 0.0){

	    // m_tau --> residual errors between natural and waveforms
	    //           |-- FFT loss: m_separate_excitation_loss = 1, m_lpcError = FFTLOSS
	    //           |-- MSE     : m_separate_excitation_loss = 1, m_lpcError = LOSS
	    //       --> natural and generated waveforms LPC residuals
	    //           |-- MSE     : m_separate_excitation_loss = 0, m_lpcError = WAV_MSE
	    //           |-- RES MSE : m_separate_excitation_loss = 0, m_lpcError = RES_MSE
	    if (m_separate_excitation_loss){
		// m_separate_excitation_loss = 1
		// select a default LPC loss when external excitation is used
		if (m_lpcErrorType != SIGPROCESS_LPC_ERR_TYPE_EXCIT_LOSS &&
		    m_lpcErrorType != SIGPROCESS_LPC_ERR_TYPE_EXCIT_FFTLOSS){
		    m_lpcErrorType = SIGPROCESS_LPC_ERR_TYPE_EXCIT_LOSS;
		}
		printf("\n\tLPC error between natural residual and generated excitation,");
		if (m_lpcErrorType == SIGPROCESS_LPC_ERR_TYPE_EXCIT_FFTLOSS)
		    printf(" spectral-amplitude-loss");
		else
		    printf(" waveform MSE");
	    }else{
		// m_separate_excitation_loss = 0
		if (m_lpcErrorType != SIGPROCESS_LPC_ERR_TYPE_RES_MSE &&
		    m_lpcErrorType != SIGPROCESS_LPC_ERR_TYPE_WAV_MSE){
		    m_lpcErrorType = SIGPROCESS_LPC_ERR_TYPE_WAV_MSE;
		}
		printf("\n\tLPC error between residuals of natural and generated waveforms");
		if (m_lpcErrorType == SIGPROCESS_LPC_ERR_TYPE_WAV_MSE)
		    printf(" MSE (o_{1:T} - iLPC(LPC(hat{o}_{1:T})))^2");
		else
		    printf(" (LPC(o_{1:T}) - LPC(hat{o}_{1:T}))^2");
	    }

	    if (m_lpcGain)
		printf("\n\tLPC analysis will calculate Gain");
	    else
		printf("\n\tLPC analysis ignores Gain");
	    
	}else{
	    m_separate_excitation_loss = 0;
	}

	if (m_iota > 0.0){
	    if (m_sineInputLayer_str.size())
		printf("\n\tEvaluate DFT errors with sine spectral mask");
	    else
		printf("\n\tEvaluate DFT errors without sine spectral mask");
	}
	
	// ------ read additional output layers for DFT errors
	m_otherSignalInputLayers_names.clear();
	m_otherSignalInputLayers_ptr.clear();
	m_otherSignalInputLayer_num = 0;
	m_otherSignalInputLayers_str = (layerChild->HasMember("otherLayersTobeImported") ? 
					((*layerChild)["otherLayersTobeImported"].GetString()) : "");
	if (m_otherSignalInputLayers_str.size()){
	    misFuncs::ParseStrOpt(m_otherSignalInputLayers_str, m_otherSignalInputLayers_names, ",");
	    m_otherSignalInputLayer_num = m_otherSignalInputLayers_names.size();
	}
	
	if (m_tau > 0.0 && m_separate_excitation_loss && m_otherSignalInputLayer_num < 1)
	    throw std::runtime_error("Error: lpcExcitationLoss is on but otherLayersForDFTError is not provided");
	
	if (m_iota > 0.0 && m_otherSignalInputLayer_num < 1)
	    throw std::runtime_error("Error: iota is on but otherLayersForDFTError is not provided");
		
	/* ------ Load DFT configurations ----- */
	if (m_gamma > 0.0 || m_zeta > 0.0 || m_eta > 0.0 || m_kappa > 0.0 || m_tau > 0.0 || m_iota > 0.0){

	    // -------- First DFT configuration ----
	    
	    // N-points FFT
	    int tmp_fftLength = (layerChild->HasMember("fftLength") ? 
				 static_cast<int>((*layerChild)["fftLength"].GetInt()) : 512);
	    
	    // actual length of one frame, which must be smaller than m_fftLength
	    int tmp_frameLength = (layerChild->HasMember("frameLength") ? 
				   static_cast<int>((*layerChild)["frameLength"].GetInt()) : 320);
	    
	    // frame shift
	    int tmp_frameShift = (layerChild->HasMember("frameShift") ? 
				  static_cast<int>((*layerChild)["frameShift"].GetInt()) : 80);
	    
	    // type of window for STFT framing (used to calculate spectral amplitude)
	    int tmp_windowType  = (layerChild->HasMember("windowType") ? 
				   static_cast<real_t>((*layerChild)["windowType"].GetInt()) :
				   FFTMAT_WINDOW_HANN);
	    
	    // type of window for STFT framing (used to calculate phase)
	    int tmp_windowTypePhase = layerChild->HasMember("windowTypePhase") ? 
		static_cast<real_t>((*layerChild)["windowTypePhase"].GetInt()):FFTMAT_WINDOW_HANN;

	    // Order of LPC (if configured)
	    int tmp_lpcOrder  = (layerChild->HasMember("lpcOrder") ? 
			  static_cast<int>((*layerChild)["lpcOrder"].GetInt()) : 0);


	    std::string tmp_specWeight = (layerChild->HasMember("specWeightVec") ? 
					  ((*layerChild)["specWeightVec"].GetString()) : "");
	    
	    
	    struct_DFTData dftBuf_1;
	    this->__initDFTBuffer(dftBuf_1);
	    this->__configDFTBuffer(dftBuf_1, tmp_fftLength, tmp_frameLength, tmp_frameShift,
				    tmp_windowType, tmp_windowTypePhase, tmp_lpcOrder,
				    tmp_specWeight);
	    this->m_DFTDataBuf.push_back(dftBuf_1);

	    if (this->m_DFTDataBuf[0].m_valid_flag == false)
		throw std::runtime_error("\nFFT configuration error in DFT layer");
	    
	    // -------- Second DFT configuration ----
	    tmp_fftLength        = (layerChild->HasMember("fftLength2") ? 
				   static_cast<int>((*layerChild)["fftLength2"].GetInt()) : 0);
	    tmp_frameLength      = (layerChild->HasMember("frameLength2") ? 
				   static_cast<int>((*layerChild)["frameLength2"].GetInt()) : 80);
	    tmp_frameShift       = (layerChild->HasMember("frameShift2") ? 
				   static_cast<int>((*layerChild)["frameShift2"].GetInt()) : 40);
	    tmp_windowType       = (layerChild->HasMember("windowType2") ? 
				    static_cast<real_t>((*layerChild)["windowType2"].GetInt()) :
				    FFTMAT_WINDOW_HANN);
	    tmp_lpcOrder         = (layerChild->HasMember("lpcOrder2") ? 
				    static_cast<int>((*layerChild)["lpcOrder2"].GetInt()) : 0);
	    tmp_windowTypePhase  = layerChild->HasMember("windowTypePhase2") ? 
		static_cast<real_t>((*layerChild)["windowTypePhase2"].GetInt()) :
		FFTMAT_WINDOW_HANN;

	    tmp_specWeight = (layerChild->HasMember("specWeightVec2") ? 
			      ((*layerChild)["specWeightVec2"].GetString()) : "");
	    struct_DFTData dftBuf_2;
	    this->__initDFTBuffer(dftBuf_2);
	    this->__configDFTBuffer(dftBuf_2, tmp_fftLength, tmp_frameLength, tmp_frameShift,
				    tmp_windowType, tmp_windowTypePhase, tmp_lpcOrder,
				    tmp_specWeight);
	    this->m_DFTDataBuf.push_back(dftBuf_2);
	    

	    
	    /* --------   third FFT ------------ */
	    tmp_fftLength        = (layerChild->HasMember("fftLength3") ? 
				   static_cast<int>((*layerChild)["fftLength3"].GetInt()) : 0);
	    tmp_frameLength      = (layerChild->HasMember("frameLength3") ? 
				   static_cast<int>((*layerChild)["frameLength3"].GetInt()) : 80);
	    tmp_frameShift       = (layerChild->HasMember("frameShift3") ? 
				   static_cast<int>((*layerChild)["frameShift3"].GetInt()) : 40);
	    tmp_windowType       = (layerChild->HasMember("windowType3") ? 
				    static_cast<real_t>((*layerChild)["windowType3"].GetInt()) :
				    FFTMAT_WINDOW_HANN);
	    tmp_lpcOrder         = (layerChild->HasMember("lpcOrder3") ? 
				    static_cast<int>((*layerChild)["lpcOrder3"].GetInt()) : 0);
	    tmp_windowTypePhase  = layerChild->HasMember("windowTypePhase3") ? 
		static_cast<real_t>((*layerChild)["windowTypePhase3"].GetInt()) :
		FFTMAT_WINDOW_HANN;
	    tmp_specWeight = (layerChild->HasMember("specWeightVec3") ? 
			      ((*layerChild)["specWeightVec3"].GetString()) : "");
	    
	    struct_DFTData dftBuf_3;
	    this->__initDFTBuffer(dftBuf_3);
	    this->__configDFTBuffer(dftBuf_3, tmp_fftLength, tmp_frameLength, tmp_frameShift,
				    tmp_windowType, tmp_windowTypePhase, tmp_lpcOrder,
				    tmp_specWeight);
	    this->m_DFTDataBuf.push_back(dftBuf_3);

	    // Note: here we only used three DFT configurations
	    // If more DFT configurations are to be used,
	    // Please also modify FFTMat.cu getCuFFTHandle_fft to return multiple FFT handlers
	}

	// -------  Obsolete (not used anymore)
	// for a special training strategy on h-NSF
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

	// multiple dimensional signal
	if (m_modeMultiDimSignal != DFTMODEFORMULTIDIMSIGNAL_NONE)
	    m_modeChangeDataBuf = this->outputs();
	else
	    m_modeChangeDataBuf.clear();

	// pre-emmphasis
	if (m_preEmphasis)
	    printf("\n\tNatural waveform will be pre-emphasis before evaluating.");

	if (m_floor_log_spec_amp != FFTMAT_LOG_AMP_FLOOR &&
	    m_specDisType == FFTMAT_SPECTYPE_AMP_LOG_MSE)
	    printf("\n\tLog-spectral-amplitude is floored by %f.", m_floor_log_spec_amp);
	// Done
    }

    
    // Destructor
    template <typename TDevice>
    DFTPostoutputLayer<TDevice>::~DFTPostoutputLayer()
    {
    }


    // Deemphasis during generation
    template <typename TDevice>
    void DFTPostoutputLayer<TDevice>::__deEmphasis(const int timeLength)
    {
	
	{{
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
				this->outputs().begin()            + this->parallelSequences(),
				thrust::counting_iterator<int>(0)  + this->parallelSequences())),
		   fn1);
	}}
	return;
    }

    // Pre-emphasis during training
    template <typename TDevice>
    void DFTPostoutputLayer<TDevice>::__preEmphasis(const int timeLength)
    {
	// pre-emphasis the natural waveform before evaluating
	{{
		internal::preemphasis fn1;
		fn1.sourceData = helpers::getRawPointer(this->outputs());
		fn1.parallel   = this->parallelSequences();
		fn1.patTypes   = helpers::getRawPointer(this->patTypes());

		// dirty code: use the buffer of outputErrors to store results
		thrust::for_each(
		   thrust::make_zip_iterator(
			thrust::make_tuple(this->outputErrors().begin(),
					   thrust::counting_iterator<int>(0))),
		   thrust::make_zip_iterator(
			thrust::make_tuple(this->outputErrors().begin()       + timeLength,
					   thrust::counting_iterator<int>(0)  + timeLength)),
		   fn1);
		this->outputs() = this->outputErrors();
		thrust::fill(this->outputErrors().begin(), this->outputErrors().end(), 0.0);
	    }}
    }


    // Flatten the multi-dimensional signal into a one-dimensional signal
    template <typename TDevice>
    void DFTPostoutputLayer<TDevice>::__flattenMultiDimSignalForward(const int timeLength)
    {
	// If the output of previous layer is multi-dimensional signal,
	// convert the N * T input data matrix into a 1-dim signal of length NT.

	// convert target data
	{{
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
	}}
	    
	thrust::copy(m_modeChangeDataBuf.begin(), m_modeChangeDataBuf.end(),
		     this->outputs().begin());

	// convert target data
	{{
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
	}}
	m_modeChangeDataBuf.swap(this->precedingLayer().outputs());
    }


    // Flatten the multi-dimensional signal into a one-dimensional signal
    template <typename TDevice>
    void DFTPostoutputLayer<TDevice>::__flattenMultiDimSignalBackward(const int timeLength)
    {

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


    // Obsolete
    // A special training mode for HNM, not used anymore
    template <typename TDevice>
    void DFTPostoutputLayer<TDevice>::__hnmSpecialForward(const int timeLength, const int nnState)
    {
	// if this layer is valid for HNM special mode
	if (m_noiseOutputLayer && m_f0InputLayer &&
	    this->getCurrTrainingEpoch() < m_noiseTrain_epoch &&
	    nnState == NN_STATE_GAN_NOGAN_TRAIN) {
		
	    // remove the voiced part in target waveforms based on U/V infor
	    {{
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
	     }}
		
	    // copy the noise output as the generated waveforms
	    //  (in order to calculate the error on the unvoiced regions only)
	    thrust::copy(m_noiseOutputLayer->outputs().begin(),
			 m_noiseOutputLayer->outputs().end(),
			 this->precedingLayer().outputs().begin());
	}
    }
    
    // Obsolete
    template <typename TDevice>
    void DFTPostoutputLayer<TDevice>::__hnmSpecialBackward(const int timeLength, const int nnState)
    {
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


    // Waveform MSE error
    template <typename TDevice>
    real_t DFTPostoutputLayer<TDevice>::__waveformMseForward(const int timeLength)
    {
	internal::ComputeMseWaveform fn;
	fn.layerSize = this->__vSize();
	fn.patTypes  = helpers::getRawPointer(this->patTypes());

	real_t mseError =
		(real_t)thrust::transform_reduce(
		   thrust::make_zip_iterator(
		      thrust::make_tuple(
			 this->outputs().begin(),
			 this->precedingLayer().outputs().begin(),   
			 thrust::counting_iterator<int>(0))),
		   thrust::make_zip_iterator(
		     thrust::make_tuple(
			 this->outputs().begin()                  + timeLength * this->__vSize(),
			 this->precedingLayer().outputs().begin() + timeLength * this->__vSize(),
			 thrust::counting_iterator<int>(0)        + timeLength * this->__vSize())),
		   fn, (real_t)0, thrust::plus<real_t>()) / timeLength;
	return mseError;
    }

    // Waveform MSE error gradients
    template <typename TDevice>
    void DFTPostoutputLayer<TDevice>::__waveformMseBackward(const int timeLength)
    {
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
		     this->outputs().begin()            + timeLength * this->__vSize(),
		     thrust::counting_iterator<int>(0)  + timeLength * this->__vSize())),
		  fn2);
	}}	
    }


    template <typename TDevice>
    real_t DFTPostoutputLayer<TDevice>::__specAmpDistance(struct_DFTData &dftBuf,
							const int timeLength)
    {
	// FFT 
	// step0. build the data structure
	// FFTMat(raw_signal, framed_signal, fft_buffer,
	//        frame_length, frame_shift, window_type, N_points_FFT, N_points_FFT/2+1,
	//        num_of_frame, maximum_signal_length, this_signal_length,
	//        special_distance_type)
	helpers::FFTMat<TDevice> sourceSig(
			&this->_actualOutputs(), &dftBuf.m_fftSourceFramed,
			&dftBuf.m_fftSourceSigFFT,
			dftBuf.m_frameLength, dftBuf.m_frameShift,
			dftBuf.m_windowType, dftBuf.m_fftLength, dftBuf.m_fftBinsNum,
			dftBuf.m_frameNum, this->__vMaxSeqLength(), timeLength,
			this->m_specDisType, this->m_floor_log_spec_amp);

	helpers::FFTMat<TDevice> targetSig(
			&this->_targets(), &dftBuf.m_fftTargetFramed,
			&dftBuf.m_fftTargetSigFFT,
			dftBuf.m_frameLength, dftBuf.m_frameShift,
			dftBuf.m_windowType, dftBuf.m_fftLength, dftBuf.m_fftBinsNum,
			dftBuf.m_frameNum, this->__vMaxSeqLength(), timeLength,
			this->m_specDisType, this->m_floor_log_spec_amp);

	helpers::FFTMat<TDevice> fftDiffSig(
			&dftBuf.m_fftDiffData, &dftBuf.m_fftDiffFramed,
			&dftBuf.m_fftDiffSigFFT,
			dftBuf.m_frameLength, dftBuf.m_frameShift,
			dftBuf.m_windowType, dftBuf.m_fftLength, dftBuf.m_fftBinsNum,
			dftBuf.m_frameNum, this->__vMaxSeqLength(), timeLength,
			this->m_specDisType, this->m_floor_log_spec_amp,
			(dftBuf.m_specWeight.size()?(&dftBuf.m_specWeight):NULL));
	
	// step1. framing and windowing
	sourceSig.frameSignal();
	targetSig.frameSignal();
		
	// step2. fft
	sourceSig.FFT();
	targetSig.FFT();

	// Dustbin #1
		
	// amplitude distance
	dftBuf.m_specError = sourceSig.specAmpDistance(targetSig, fftDiffSig);
	
	// compute complex-valued grad vector
	fftDiffSig.specAmpGrad(sourceSig, targetSig);
	// inverse DFT
	fftDiffSig.iFFT();
	// de-framing/windowing
	fftDiffSig.collectGrad(m_gamma);
	
	// Gradients should be in m_fftDiffData		    
	// Done
	return dftBuf.m_specError;
    }

    template <typename TDevice>
    real_t DFTPostoutputLayer<TDevice>::__specPhaDistance(struct_DFTData &dftBuf,
							const int timeLength)
    {
	// FFT 1
	// step0. build the data structure
	helpers::FFTMat<TDevice> sourceSigPhase(
			&this->_actualOutputs(), &dftBuf.m_fftSourceFramed,
			&dftBuf.m_fftSourceSigFFT,
			dftBuf.m_frameLength, dftBuf.m_frameShift,
			dftBuf.m_windowType, dftBuf.m_fftLength, dftBuf.m_fftBinsNum,
			dftBuf.m_frameNum, this->__vMaxSeqLength(), timeLength,
			this->m_phaseDisType);

	helpers::FFTMat<TDevice> targetSigPhase(
			&this->_targets(), &dftBuf.m_fftTargetFramed,
			&dftBuf.m_fftTargetSigFFT,
			dftBuf.m_frameLength, dftBuf.m_frameShift,
			dftBuf.m_windowType, dftBuf.m_fftLength, dftBuf.m_fftBinsNum,
			dftBuf.m_frameNum, this->__vMaxSeqLength(), timeLength,
			this->m_phaseDisType);

	helpers::FFTMat<TDevice> fftDiffSigPhase(
			&dftBuf.m_fftDiffDataPhase, &dftBuf.m_fftDiffFramed,
			&dftBuf.m_fftDiffSigFFT,
			dftBuf.m_frameLength, dftBuf.m_frameShift,
			dftBuf.m_windowType, dftBuf.m_fftLength, dftBuf.m_fftBinsNum,
			dftBuf.m_frameNum, this->__vMaxSeqLength(), timeLength,
			this->m_phaseDisType);
		
	// step1. framing and windowing
	sourceSigPhase.frameSignal();
	targetSigPhase.frameSignal();
		
	// step2. fft
	sourceSigPhase.FFT();
	targetSigPhase.FFT();

	// calculate phase distortion
	dftBuf.m_phaseError = sourceSigPhase.specPhaseDistance(targetSigPhase,
							       fftDiffSigPhase);
	// compute complex-valued grad vector
	fftDiffSigPhase.specPhaseGrad(sourceSigPhase, targetSigPhase);
	// inverse DFT
	fftDiffSigPhase.iFFT();
	// de-framing/windowing, grad will be in m_fftDiffDataPhase
	fftDiffSigPhase.collectGrad(m_zeta);

	return dftBuf.m_phaseError;
    }

    template <typename TDevice>
    real_t DFTPostoutputLayer<TDevice>::__specResAmpDistance(struct_DFTData &dftBuf,
							   const int timeLength)
    {
	// FFT 1	
	// step 1, calculate residual a = generated - target
	thrust::transform(this->_actualOutputs().begin(),
			  this->_actualOutputs().begin() + timeLength * this->__vSize(),
			  this->_targets().begin(),
			  dftBuf.m_fftResData.begin(),
			  thrust::minus<real_t>());
	    
	helpers::FFTMat<TDevice> fftResSig(
			&dftBuf.m_fftResData, &dftBuf.m_fftDiffFramed,
			&dftBuf.m_fftDiffSigFFT,
			dftBuf.m_frameLength, dftBuf.m_frameShift, dftBuf.m_windowType,
			dftBuf.m_fftLength, dftBuf.m_fftBinsNum,
			dftBuf.m_frameNum, this->__vMaxSeqLength(), timeLength,
			this->m_specDisType);
	
	// step 2. framing and windowing
	fftResSig.frameSignal();
	// step 3. fft
	fftResSig.FFT();		
	// amplitude distance
	dftBuf.m_resError = fftResSig.specAmpToZeroDistance();
	// compute complex-valued grad vector
	fftResSig.specAmpToZeroGrad();
	// inverse DFT
	fftResSig.iFFT();
	// de-framing/windowing
	fftResSig.collectGrad(m_eta);
	
	return dftBuf.m_resError;
    }

    template <typename TDevice>
    real_t DFTPostoutputLayer<TDevice>::__specRealAmpDistance(struct_DFTData &dftBuf,
							    const int timeLength)
    {
	
	helpers::FFTMat<TDevice> sourceSig(
			&this->_actualOutputs(), &dftBuf.m_fftSourceFramedRealSpec,
			&dftBuf.m_fftSourceSigFFTRealSpec,
			dftBuf.m_frameLength, dftBuf.m_frameShift, dftBuf.m_windowType,
			dftBuf.m_fftLengthRealSpec, dftBuf.m_fftBinsNumRealSpec,
			dftBuf.m_frameNum, this->__vMaxSeqLength(), timeLength,
			this->m_realSpecDisType);

	helpers::FFTMat<TDevice> targetSig(
			&this->_targets(), &dftBuf.m_fftTargetFramedRealSpec,
			&dftBuf.m_fftTargetSigFFTRealSpec,
			dftBuf.m_frameLength, dftBuf.m_frameShift, dftBuf.m_windowType,
			dftBuf.m_fftLengthRealSpec, dftBuf.m_fftBinsNumRealSpec,
			dftBuf.m_frameNum, this->__vMaxSeqLength(), timeLength,
			this->m_realSpecDisType);

	helpers::FFTMat<TDevice> fftDiffSig(
			&dftBuf.m_fftDiffDataRealSpec, &dftBuf.m_fftDiffFramedRealSpec,
			&dftBuf.m_fftDiffSigFFTRealSpec,
			dftBuf.m_frameLength, dftBuf.m_frameShift, dftBuf.m_windowType,
			dftBuf.m_fftLengthRealSpec, dftBuf.m_fftBinsNumRealSpec,
			dftBuf.m_frameNum, this->__vMaxSeqLength(), timeLength,
			this->m_realSpecDisType);
	
	// step1. framing and windowing
	sourceSig.frameSignalRealSpec(m_realSpecType);
	targetSig.frameSignalRealSpec(m_realSpecType);
		
	// step2. fft
	sourceSig.FFT();
	targetSig.FFT();

	// theoretically, imaginary part should be zero
	thrust::for_each(dftBuf.m_fftSourceSigFFTRealSpec.begin(),
			 dftBuf.m_fftSourceSigFFTRealSpec.end(),
			 internal::cleanImagPart());
	thrust::for_each(dftBuf.m_fftTargetSigFFTRealSpec.begin(),
			 dftBuf.m_fftTargetSigFFTRealSpec.end(),
			 internal::cleanImagPart());
	
	// amplitude distance
	dftBuf.m_realSpecError = sourceSig.specAmpDistance(targetSig, fftDiffSig);
	// compute complex-valued grad vector
	fftDiffSig.specAmpGrad(sourceSig, targetSig);
	// inverse DFT
	fftDiffSig.iFFT();
	// de-framing/windowing
	fftDiffSig.collectGradRealSpec(m_kappa, m_realSpecType, sourceSig);
	
	// Gradients should be in m_fftDiffData		    
	// Done
	return dftBuf.m_realSpecError;
    }


    template <typename TDevice>
    real_t DFTPostoutputLayer<TDevice>::__lpcError(struct_DFTData &dftBuf,
						 const int timeLength)
    {
	// number of frames for this utterance
	int validFrameNum =  helpers::fftTools::fftFrameNum(timeLength,
							    dftBuf.m_frameLength,
							    dftBuf.m_frameShift);
	if (dftBuf.m_lpcOrder < 1){
	    
	    // no need to use lpcError
	    dftBuf.m_lpcError = 0.0;
	}else{
	    
	    // when LPC error is calculated
	    if (m_separate_excitation_loss){
		// when LPC excitation loss is evaluated on excitation waveform

		if (m_otherSignalInputLayers_ptr.size() < 1 ||
		    m_otherSignalInputLayers_ptr[0] == NULL)
		    throw std::runtime_error("Error: no excitation layer loaded in DFTlayer");
		
		// step1. frame the excitation signal
		//  here we use dftBuf.m_fftSourceFramed to store the framed excitation signals
		//  
		helpers::FFTMat<TDevice> excitSig(
			&m_otherSignalInputLayers_ptr[0]->outputs(),
			&dftBuf.m_fftSourceFramed,
			&dftBuf.m_fftSourceSigFFT,
			dftBuf.m_frameLength, dftBuf.m_frameShift,
			dftBuf.m_windowType, dftBuf.m_fftLength, dftBuf.m_fftBinsNum,
			dftBuf.m_frameNum, this->__vMaxSeqLength(), timeLength,
			this->m_specDisType); 
		excitSig.frameSignal();

		// step2. do LPC analysis on target signal
		helpers::lpcWarpper<TDevice> lpcAnalysizer(
		&dftBuf.m_fftSourceFramed, &dftBuf.m_fftTargetFramed,
		&dftBuf.m_autoCorrSrc, &dftBuf.m_autoCorrTar,
		&dftBuf.m_lpcCoefSrc, &dftBuf.m_lpcCoefTar,
		&dftBuf.m_lpcErrSrc, &dftBuf.m_lpcErrTar,
		&dftBuf.m_refCoefSrc, &dftBuf.m_refCoefTar,
		&dftBuf.m_lpcResSrc, &dftBuf.m_lpcResTar,
		&dftBuf.m_lpcGrad,
		dftBuf.m_lpcOrder,
		this->m_lpcErrorType,
		dftBuf.m_frameLength,
		dftBuf.m_frameShift,
		dftBuf.m_fftLength,
		validFrameNum,
		this->__vMaxSeqLength(), timeLength,
		this->m_lpcGain);

		lpcAnalysizer.lpcAnalysisTargetSignal();


		// step3. calculate distance
		if (this->m_lpcErrorType == SIGPROCESS_LPC_ERR_TYPE_EXCIT_LOSS){
		    // MSE distance between excitation and natural residuals waveforms

		    // Calculate LPC domain Errors
		    dftBuf.m_lpcError = lpcAnalysizer.lpcError();

		    // Collect gradients (which will be in dftBuf.m_lpcGrad)
		    lpcAnalysizer.lpcGradCollect(m_tau);
		    
		}else if (this->m_lpcErrorType == SIGPROCESS_LPC_ERR_TYPE_EXCIT_FFTLOSS){
		    // FFT distance between excitation and natural residual waveforms
		    // Note: this method is a bad idea
		    // 
		    
		    // For excitation signals have been created
		    // it has been framed
		    // do STFT
		    excitSig.FFT();

		    
		    // here, we need to re-use the buffer for waveform spectral loss
		    // wrap around residual signals of natural waveforms (m_lpcResTar)
		    helpers::FFTMat<TDevice> ResTarSig(
			&this->_targets(), &dftBuf.m_lpcResTar,
			&dftBuf.m_fftTargetSigFFT,
			dftBuf.m_frameLength, dftBuf.m_frameShift,
			dftBuf.m_windowType, dftBuf.m_fftLength, dftBuf.m_fftBinsNum,
			dftBuf.m_frameNum, this->__vMaxSeqLength(), timeLength,
			this->m_specDisType);
		    // residual signals have been framed
		    // do STFT
		    ResTarSig.FFT();
		    
		    // wrap around the gradient buffer for LPC DFT loss (m_lpcGrad)
		    helpers::FFTMat<TDevice> lpcFFTDiffSig(
			&dftBuf.m_lpcGrad, &dftBuf.m_fftDiffFramed,
			&dftBuf.m_fftDiffSigFFT,
			dftBuf.m_frameLength, dftBuf.m_frameShift,
			dftBuf.m_windowType, dftBuf.m_fftLength, dftBuf.m_fftBinsNum,
			dftBuf.m_frameNum, this->__vMaxSeqLength(), timeLength,
			this->m_specDisType);
	
		    // amplitude distance
		    dftBuf.m_lpcError = excitSig.specAmpDistance(ResTarSig, lpcFFTDiffSig);
		    // compute complex-valued grad vector
		    lpcFFTDiffSig.specAmpGrad(excitSig, ResTarSig);
		    // inverse DFT
		    lpcFFTDiffSig.iFFT();
		    // de-framing/windowing
		    lpcFFTDiffSig.collectGrad(m_tau);
		}else{
		    throw std::runtime_error("Error: cannot reach here in DFT layer with external signal loss");
		}
		
	    }else{

		// when LPC loss is evaluated on generated waveform
		// Warp the data structure
		// Assume m_fftSourceFramed and m_fftTargetFramed have stored the framed data.
		// (after __specAmpDistance())
		
		helpers::lpcWarpper<TDevice> lpcAnalysizer(
		&dftBuf.m_fftSourceFramed, &dftBuf.m_fftTargetFramed,
		&dftBuf.m_autoCorrSrc, &dftBuf.m_autoCorrTar,
		&dftBuf.m_lpcCoefSrc, &dftBuf.m_lpcCoefTar,
		&dftBuf.m_lpcErrSrc, &dftBuf.m_lpcErrTar,
		&dftBuf.m_refCoefSrc, &dftBuf.m_refCoefTar,
		&dftBuf.m_lpcResSrc, &dftBuf.m_lpcResTar,
		&dftBuf.m_lpcGrad,
		dftBuf.m_lpcOrder,
		this->m_lpcErrorType,
		dftBuf.m_frameLength,
		dftBuf.m_frameShift,
		dftBuf.m_fftLength,
		validFrameNum,
		this->__vMaxSeqLength(), timeLength,
		this->m_lpcGain);

		// Do LPC analysis
		lpcAnalysizer.lpcAnalysisSourceSignal();
		lpcAnalysizer.lpcAnalysisTargetSignal();		

		// Calculate LPC domain Errors
		dftBuf.m_lpcError = lpcAnalysizer.lpcError();

		// Collect gradients (which will be in dftBuf.m_lpcGrad)
		lpcAnalysizer.lpcGradCollect(m_tau);
	    }
	}
	
	return dftBuf.m_lpcError;
    }


    template <typename TDevice>
    real_t DFTPostoutputLayer<TDevice>::__specAmpDistanceOthers(struct_DFTData &dftBuf,
								const int timeLength)
    {

	//
	dftBuf.m_specErrorOthers = 0.0;
	
	// Do anlysis on target signal at first
	helpers::FFTMat<TDevice> targetSig(
			&this->_targets(), &dftBuf.m_fftTargetFramed,
			&dftBuf.m_fftTargetSigFFT,
			dftBuf.m_frameLength, dftBuf.m_frameShift,
			dftBuf.m_windowType, dftBuf.m_fftLength, dftBuf.m_fftBinsNum,
			dftBuf.m_frameNum, this->__vMaxSeqLength(), timeLength,
			this->m_specDisType, this->m_floor_log_spec_amp);
	// specAmpDistance should have done the two steps on target signals
	//targetSig.frameSignal();
	//targetSig.FFT();

	if (m_otherSignalInputLayers_ptr.size()){
	    
	    for (int layerIndex = 0; layerIndex < m_otherSignalInputLayers_ptr.size(); layerIndex++){
		// for each external input layer
		helpers::FFTMat<TDevice> sourceSig(
			&m_otherSignalInputLayers_ptr[layerIndex]->outputs(),
			&dftBuf.m_fftSourceFramed,
			&dftBuf.m_fftSourceSigFFT,
			dftBuf.m_frameLength, dftBuf.m_frameShift,
			dftBuf.m_windowType, dftBuf.m_fftLength, dftBuf.m_fftBinsNum,
			dftBuf.m_frameNum, this->__vMaxSeqLength(), timeLength,
			this->m_specDisType, this->m_floor_log_spec_amp);

		sourceSig.frameSignal();
		sourceSig.FFT();

		// if mask signal is provided, use mask
		if (m_sineInputLayer_str.size()){
		    if (m_sineInputLayer_ptr == NULL)
			throw std::runtime_error("Error: sine source layer is not linked");
		    
		    helpers::FFTMat<TDevice> maskSig(
			&m_sineInputLayer_ptr->outputs(),
			&dftBuf.m_fftMaskSignalFramed,
			&dftBuf.m_fftMaskFFT,
			dftBuf.m_frameLength, dftBuf.m_frameShift,
			dftBuf.m_windowType, dftBuf.m_fftLength, dftBuf.m_fftBinsNum,
			dftBuf.m_frameNum, this->__vMaxSeqLength(), timeLength,
			this->m_specDisType, this->m_floor_log_spec_amp);

		    if (layerIndex == 0){
			// only do framing and FFT once
			maskSig.frameSignal();
			maskSig.FFT();
			maskSig.spec2mask();
			
			// mask the target signal
			targetSig.spectralMask(maskSig);
		    }
		    // mask the source spectral
		    sourceSig.spectralMask(maskSig);
		}
	
		// Calculate the distance and gradietns over each hidden layer
		helpers::FFTMat<TDevice> fftDiffSig(
			&dftBuf.m_specGrad_tmpBuf, &dftBuf.m_fftDiffFramed,
			&dftBuf.m_fftDiffSigFFT,
			dftBuf.m_frameLength, dftBuf.m_frameShift,
			dftBuf.m_windowType, dftBuf.m_fftLength, dftBuf.m_fftBinsNum,
			dftBuf.m_frameNum, this->__vMaxSeqLength(), timeLength,
			this->m_specDisType, this->m_floor_log_spec_amp);
	
		// calculate the error
		dftBuf.m_specErrorOthers += sourceSig.specAmpDistance(targetSig, fftDiffSig);
		// calculate grad w.r.t masked spectral
		fftDiffSig.specAmpGrad(sourceSig, targetSig);
		
		// grad through the mask
		if (m_sineInputLayer_str.size()){
		    helpers::FFTMat<TDevice> maskSig(
			&m_sineInputLayer_ptr->outputs(),
			&dftBuf.m_fftMaskSignalFramed,
			&dftBuf.m_fftMaskFFT,
			dftBuf.m_frameLength, dftBuf.m_frameShift,
			dftBuf.m_windowType, dftBuf.m_fftLength, dftBuf.m_fftBinsNum,
			dftBuf.m_frameNum, this->__vMaxSeqLength(), timeLength,
			this->m_specDisType, this->m_floor_log_spec_amp);
		    fftDiffSig.spectralMask(maskSig);
		}

		// grad from spectral domain to framed signal
		fftDiffSig.iFFT();
		// grad from framed signal to original waveform
		fftDiffSig.collectGrad(m_iota);
		
		// collect gradients to m_specGrad_others
		thrust::copy(dftBuf.m_specGrad_tmpBuf.begin(), dftBuf.m_specGrad_tmpBuf.end(),
			     dftBuf.m_specGrad_others.begin() + layerIndex * dftBuf.m_specGrad_tmpBuf.size());
	    }
	    
	}else{
	    throw std::runtime_error("Error: iota is on but otherLayersForDFTError is not provided");
	}
	return dftBuf.m_specErrorOthers;
    }


    
    template <typename TDevice>
    real_t DFTPostoutputLayer<TDevice>::__specDistance_wrapper(struct_DFTData &dftBuf,
							     const int timeLength)
    {
	real_t specAmpError = 0.0;
	real_t phaseError = 0.0;
	real_t specResError = 0.0;
	real_t specRealError = 0.0;
	real_t lpcError = 0.0;
	real_t specErrorsOthers = 0.0;
	
	this->__cleanDFTError(dftBuf);
	
	// FFT configuration 1
	if (dftBuf.m_valid_flag){
		    
	    // calculate the spectral amplitude distance1
	    specAmpError = this->__specAmpDistance(dftBuf, timeLength);
	    
	    // calculate the phase distance
	    if (m_zeta > 0.0)
		phaseError = this->__specPhaDistance(dftBuf, timeLength);

	    // calculate the spectrum distance in complex-domain (equivalent to waveform MSE)
	    if (m_eta > 0.0)
		specResError = this->__specResAmpDistance(dftBuf, timeLength);

	    // calculate the real-valued spectrum distances
	    if (m_kappa > 0.0)
		specRealError = this->__specRealAmpDistance(dftBuf, timeLength);

	    // calculate the spectral distance for other hidden layers
	    //  note that, this will re-use the buffers for short-time-analysis
	    if (m_iota > 0.0)
		specErrorsOthers = this->__specAmpDistanceOthers(dftBuf, timeLength);
	    
	    // calculate the distances in LPC domain
	    //  note that, LPC analysis requires framing,
	    //  thus, we put the lpcError here so that it can re-use the data structure
	    //  for framing in FFTMat
	    if (m_tau > 0.0)
		lpcError = this->__lpcError(dftBuf, timeLength);
		    
	}
	return specAmpError + phaseError + specResError + specRealError + lpcError + specErrorsOthers;
    }

    template <typename TDevice>
    void DFTPostoutputLayer<TDevice>::__specAccumulateGrad(struct_DFTData &dftBuf,
							   const int timeLength)
    {

	// Note: all the gradients have been computed in ComputeForwardPass
	// here, we only accumulate the gradients
	
	if (dftBuf.m_valid_flag){
	    
	    // grad of spec amplitude distance
	    thrust::transform(dftBuf.m_fftDiffData.begin(),
			      dftBuf.m_fftDiffData.begin() + timeLength * this->__vSize(),
			      this->precedingLayer().outputErrors().begin(),
			      this->precedingLayer().outputErrors().begin(),
			      thrust::plus<real_t>());
	    
	    // grad of phase amplitude distance
	    if (m_zeta > 0.0)
		thrust::transform(dftBuf.m_fftDiffDataPhase.begin(),
				  dftBuf.m_fftDiffDataPhase.begin() + timeLength * this->__vSize(),
				  this->precedingLayer().outputErrors().begin(),
				  this->precedingLayer().outputErrors().begin(),
				  thrust::plus<real_t>());

	    // grad of complex-valued spectral distance
	    if (m_eta > 0.0)
		thrust::transform(dftBuf.m_fftResData.begin(),
				  dftBuf.m_fftResData.begin() + timeLength * this->__vSize(),
				  this->precedingLayer().outputErrors().begin(),
				  this->precedingLayer().outputErrors().begin(),
				  thrust::plus<real_t>());

	    // grad of real-valued spectral distance
	    if (m_kappa > 0.0)
		thrust::transform(dftBuf.m_fftDiffDataRealSpec.begin(),
				  dftBuf.m_fftDiffDataRealSpec.begin() + timeLength*this->__vSize(),
				  this->precedingLayer().outputErrors().begin(),
				  this->precedingLayer().outputErrors().begin(),
				  thrust::plus<real_t>());

	    // grad of LPC error
	    if (m_tau > 0.0){
		if (m_separate_excitation_loss){
		    // when lpc error is calculated on excitation and natural residuals
		    // copy the gradients to the excitation layer
		    // Note: here we copy the gradients directly to the outputErrors
		    //  this requires that the excitation layer is a skipini layer,
		    //  also, the layer after the excitation layer should be a skipini layer
		    thrust::transform(dftBuf.m_lpcGrad.begin(),
				      dftBuf.m_lpcGrad.begin() + timeLength * this->__vSize(),
				      m_otherSignalInputLayers_ptr[0]->outputErrors().begin(),
				      m_otherSignalInputLayers_ptr[0]->outputErrors().begin(),
				      thrust::plus<real_t>());
		}else{
		    // when lpc error is calculated on generated and natural waveforms 
		    // copy the gradients to the output of neural-filter-module
		    thrust::transform(dftBuf.m_lpcGrad.begin(),
				      dftBuf.m_lpcGrad.begin() + timeLength * this->__vSize(),
				      this->precedingLayer().outputErrors().begin(),
				      this->precedingLayer().outputErrors().begin(),
				      thrust::plus<real_t>());
		}
	    }
	    
	    // gradients for other hidden layers
	    if (m_iota > 0.0){
		for (int layerIndex = 0; layerIndex < m_otherSignalInputLayers_ptr.size(); layerIndex++){
		    thrust::transform(
			dftBuf.m_specGrad_others.begin() + layerIndex * this->__vMaxSeqLength(),
			dftBuf.m_specGrad_others.begin() + (layerIndex+1) * this->__vMaxSeqLength(),
			m_otherSignalInputLayers_ptr[layerIndex]->outputErrors().begin(),
			m_otherSignalInputLayers_ptr[layerIndex]->outputErrors().begin(),
			thrust::plus<real_t>());
		}		
	    }
	    
	}

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
	    if (m_preEmphasis)
		this->__deEmphasis(timeLength);
	    
	    return;
	    
	}else{
	    
	    // Training mode
	    
	    // If preemphasis is necessary
	    if (m_preEmphasis)
		this->__preEmphasis(timeLength);
	    	    
	    // If target is multi-dimensional signal
	    if (m_modeMultiDimSignal == DFTMODEFORMULTIDIMSIGNAL_CONCATE)
		this->__flattenMultiDimSignalForward(timeLength);
	
	    // for HNM special training mode (obsolete)
	    if (m_hnm_flag == DFTERRORPOST_HNM_MODEL_1 || m_hnm_flag == DFTERRORPOST_HNM_MODEL_2)
		this->__hnmSpecialForward(timeLength, nnState);
	    	
	    // Compute waveform MSE if necessary 
	    if (m_beta > 0.0)
		m_mseError = this->__waveformMseForward(timeLength);
	    else
		m_mseError = 0.0;

	    // Compute DFT amplitute and phase distance if necessary
	    // This is the core part of this layer
	    // Note: these methods will
	    //   1. conduct signal_framing, STFT
	    //   2. calculate distances
	    //   3. calculate gradients and save them to the buffer
	    // 
	    if (m_gamma > 0.0 || m_zeta > 0.0 || m_eta > 0.0 || m_kappa > 0.0 || m_tau > 0.0 || m_iota > 0.0){
		// calculate distances
		for (int dftBufIndex = 0; dftBufIndex < this->m_DFTDataBuf.size(); dftBufIndex++)
		    this->__specDistance_wrapper(this->m_DFTDataBuf[dftBufIndex], timeLength);
	    }
	}
	return;
	// Done
    }

    template <typename TDevice>
    void DFTPostoutputLayer<TDevice>::computeForwardPass(const int timeStep, const int nnState)
    {
	// Not implemented yet
	//throw std::runtime_error("Not implemented DFTError computeForwardPass(timeStep)");
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
	if (m_beta > 0.0)
	    this->__waveformMseBackward(timeLength);

	// Gradients from spectral amplitude and phase
	//  gradients have been calculated in computeForwardPass()
	//  here, gradients are simply accumulated into the gradient buffer
	if (m_gamma > 0.0 || m_zeta > 0.0 || m_eta > 0.0 || m_kappa > 0.0 || m_tau > 0.0 || m_iota > 0.0){
	    // Accumulate gradients from each DFT buffer
	    for (int dftBufIndex = 0; dftBufIndex < this->m_DFTDataBuf.size(); dftBufIndex++)
		this->__specAccumulateGrad(this->m_DFTDataBuf[dftBufIndex], timeLength);
	}
	
	// For HNM special training mode
	if (m_hnm_flag == DFTERRORPOST_HNM_MODEL_1 || m_hnm_flag == DFTERRORPOST_HNM_MODEL_2)
	    this->__hnmSpecialBackward(timeLength, nnState);
	    
	// For multi-dim signal case
	if (m_modeMultiDimSignal == DFTMODEFORMULTIDIMSIGNAL_CONCATE)
	    this->__flattenMultiDimSignalBackward(timeLength);

	   
	// done
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

	if (m_specDisType != FFTMAT_SPECTYPE_AMP_LOG_MSE)
	    (*layersArray)[layersArray->Size() - 1].AddMember("specDisType", m_specDisType,
							      allocator);
	if (m_floor_log_spec_amp != FFTMAT_LOG_AMP_FLOOR)
	    (*layersArray)[layersArray->Size() - 1].AddMember("logSpecAmpFloor", m_floor_log_spec_amp,
							      allocator);
	if (m_phaseDisType != FFTMAT_PHASETYPE_COS)
	    (*layersArray)[layersArray->Size() - 1].AddMember("phaseDisType", m_phaseDisType,
							      allocator);
	if (m_preEmphasis)
	    (*layersArray)[layersArray->Size() - 1].AddMember("preEmphasisNaturalWav",
							      m_preEmphasis,
							      allocator);
	if (m_separate_excitation_loss)
	    (*layersArray)[layersArray->Size() - 1].AddMember(
			"lpcExcitationLoss", m_separate_excitation_loss, allocator);
	

	if (m_otherSignalInputLayers_str.size())
	    (*layersArray)[layersArray->Size() - 1].AddMember(
			"otherLayersTobeImported", m_otherSignalInputLayers_str.c_str(), allocator);
	
	
	if (m_gamma > 0.0){
	    (*layersArray)[layersArray->Size() - 1].AddMember("gamma", m_gamma, allocator);
	    
	    if (m_zeta > 0.0)
		(*layersArray)[layersArray->Size() - 1].AddMember("zeta", m_zeta, allocator);
	    
	    if (m_eta > 0.0)
		(*layersArray)[layersArray->Size() - 1].AddMember("eta", m_eta, allocator);
	    
	    if (m_kappa > 0.0){
		(*layersArray)[layersArray->Size() - 1].AddMember("kappa", m_kappa, allocator);

		if (m_realSpecType != FFTMAT_REALSPEC_TYPE_NORMAL)
		    (*layersArray)[layersArray->Size() - 1].AddMember("kappa_realspec_type",
								      m_realSpecType, allocator);
		
		if (m_realSpecDisType != FFTMAT_SPECTYPE_AMP_LOG_MSE)
		    (*layersArray)[layersArray->Size() - 1].AddMember("realSpecDisType",
								      m_specDisType, allocator);
	    }

	    if (m_tau > 0.0){
		(*layersArray)[layersArray->Size() - 1].AddMember("tau", m_tau, allocator);
		(*layersArray)[layersArray->Size() - 1].AddMember("lpcErrorType", m_lpcErrorType,
								  allocator);
		(*layersArray)[layersArray->Size() - 1].AddMember("lpcCalculateGain", m_lpcGain,
								  allocator);
	    }

	    if (m_iota > 0.0){
		(*layersArray)[layersArray->Size() - 1].AddMember("iota", m_iota, allocator);
		if (m_sineInputLayer_str.size())
		    (*layersArray)[layersArray->Size() - 1].AddMember("sineSourceLayer",
								      m_sineInputLayer_str.c_str(),
								      allocator);
	    }

	    if (m_f0InputLayer_str.size())
		(*layersArray)[layersArray->Size() - 1].AddMember("f0InputLayer",
								  m_f0InputLayer_str.c_str(),
								  allocator);		
	    
	    for (int dftBufIndex = 0; dftBufIndex < this->m_DFTDataBuf.size(); dftBufIndex++){
		if (dftBufIndex == 0 && m_DFTDataBuf[dftBufIndex].m_valid_flag){
		    (*layersArray)[layersArray->Size() - 1].AddMember(
			"fftLength",
			m_DFTDataBuf[dftBufIndex].m_fftLength,
			allocator);
		    (*layersArray)[layersArray->Size() - 1].AddMember(
			"frameLength",
			m_DFTDataBuf[dftBufIndex].m_frameLength,
			allocator);
		    (*layersArray)[layersArray->Size() - 1].AddMember(
			"frameShift",
			m_DFTDataBuf[dftBufIndex].m_frameShift,
			allocator);
		    if (m_DFTDataBuf[dftBufIndex].m_windowType != FFTMAT_WINDOW_HANN)
			(*layersArray)[layersArray->Size() - 1].AddMember(
			 "windowType",
			 m_DFTDataBuf[dftBufIndex].m_windowType,
			 allocator);
		    if (m_DFTDataBuf[dftBufIndex].m_windowTypePhase != FFTMAT_WINDOW_HANN)
			(*layersArray)[layersArray->Size() - 1].AddMember(
			 "windowTypePhase",
			 m_DFTDataBuf[dftBufIndex].m_windowTypePhase,
			 allocator);
		    (*layersArray)[layersArray->Size() - 1].AddMember(
			"lpcOrder",
			m_DFTDataBuf[dftBufIndex].m_lpcOrder,
			allocator);

		    if (m_DFTDataBuf[dftBufIndex].m_specWeightStr.size())
			(*layersArray)[layersArray->Size() - 1].AddMember(
			"specWeightVec",
			m_DFTDataBuf[dftBufIndex].m_specWeightStr.c_str(),
			allocator);
		    
		}else if (dftBufIndex == 1 && m_DFTDataBuf[dftBufIndex].m_valid_flag){
		    (*layersArray)[layersArray->Size() - 1].AddMember(
			"fftLength2",
			m_DFTDataBuf[dftBufIndex].m_fftLength,
			allocator);
		    (*layersArray)[layersArray->Size() - 1].AddMember(
			"frameLength2",
			m_DFTDataBuf[dftBufIndex].m_frameLength,
			allocator);
		    (*layersArray)[layersArray->Size() - 1].AddMember(
			"frameShift2",
			m_DFTDataBuf[dftBufIndex].m_frameShift,
			allocator);
		    if (m_DFTDataBuf[dftBufIndex].m_windowType != FFTMAT_WINDOW_HANN)
			(*layersArray)[layersArray->Size() - 1].AddMember(
			 "windowType2",
			 m_DFTDataBuf[dftBufIndex].m_windowType,
			 allocator);
		    if (m_DFTDataBuf[dftBufIndex].m_windowTypePhase != FFTMAT_WINDOW_HANN)
			(*layersArray)[layersArray->Size() - 1].AddMember(
			 "windowTypePhase2",
			 m_DFTDataBuf[dftBufIndex].m_windowTypePhase,
			 allocator);
		    (*layersArray)[layersArray->Size() - 1].AddMember(
			"lpcOrder2",
			m_DFTDataBuf[dftBufIndex].m_lpcOrder,
			allocator);

		    if (m_DFTDataBuf[dftBufIndex].m_specWeightStr.size())
			(*layersArray)[layersArray->Size() - 1].AddMember(
			"specWeightVec2",
			m_DFTDataBuf[dftBufIndex].m_specWeightStr.c_str(),
			allocator);
		    
		}else if (dftBufIndex == 2 && m_DFTDataBuf[dftBufIndex].m_valid_flag){
		    (*layersArray)[layersArray->Size() - 1].AddMember(
			"fftLength3",
			m_DFTDataBuf[dftBufIndex].m_fftLength,
			allocator);
		    (*layersArray)[layersArray->Size() - 1].AddMember(
			"frameLength3",
			m_DFTDataBuf[dftBufIndex].m_frameLength,
			allocator);
		    (*layersArray)[layersArray->Size() - 1].AddMember(
			"frameShift3",
			m_DFTDataBuf[dftBufIndex].m_frameShift,
			allocator);
		    if (m_DFTDataBuf[dftBufIndex].m_windowType != FFTMAT_WINDOW_HANN)
			(*layersArray)[layersArray->Size() - 1].AddMember(
			 "windowType3",
			 m_DFTDataBuf[dftBufIndex].m_windowType,
			 allocator);
		    if (m_DFTDataBuf[dftBufIndex].m_windowTypePhase != FFTMAT_WINDOW_HANN)
			(*layersArray)[layersArray->Size() - 1].AddMember(
			 "windowTypePhase3",
			 m_DFTDataBuf[dftBufIndex].m_windowTypePhase,
			 allocator);
		    (*layersArray)[layersArray->Size() - 1].AddMember(
			"lpcOrder3",
			m_DFTDataBuf[dftBufIndex].m_lpcOrder,
			allocator);

		    if (m_DFTDataBuf[dftBufIndex].m_specWeightStr.size())
			(*layersArray)[layersArray->Size() - 1].AddMember(
			"specWeightVec3",
			m_DFTDataBuf[dftBufIndex].m_specWeightStr.c_str(),
			allocator);
		    
		}else{
		    // pass
		}
	    }
	    
	    
	    if (m_hnm_flag > 0 ){
		(*layersArray)[layersArray->Size() - 1].AddMember("hnmMode", m_hnm_flag,
								  allocator);
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

	    for (int dftBufIndex = 0; dftBufIndex < this->m_DFTDataBuf.size(); dftBufIndex++)
		std::cerr << m_DFTDataBuf[dftBufIndex].m_specError << ", ";
	    
	    std::cerr << m_mseError << ", 0" << ", ";
	    if (m_zeta > 0.0){
		for (int dftBufIndex = 0; dftBufIndex < this->m_DFTDataBuf.size(); dftBufIndex++)
		    std::cerr << m_DFTDataBuf[dftBufIndex].m_phaseError << ", ";
	    }
	    if (m_eta > 0.0){
		for (int dftBufIndex = 0; dftBufIndex < this->m_DFTDataBuf.size(); dftBufIndex++)
		    std::cerr << m_DFTDataBuf[dftBufIndex].m_resError << ", ";
	    }
	    if (m_kappa > 0.0){
		for (int dftBufIndex = 0; dftBufIndex < this->m_DFTDataBuf.size(); dftBufIndex++)
		    std::cerr << m_DFTDataBuf[dftBufIndex].m_realSpecError << ", ";		
	    }
	    
	    if (m_tau > 0.0){
		for (int dftBufIndex = 0; dftBufIndex < this->m_DFTDataBuf.size(); dftBufIndex++)
		    std::cerr << m_DFTDataBuf[dftBufIndex].m_lpcError << ", ";		
	    }
	    if (m_iota > 0.0){
		for (int dftBufIndex = 0; dftBufIndex < this->m_DFTDataBuf.size(); dftBufIndex++)
		    std::cerr << m_DFTDataBuf[dftBufIndex].m_specErrorOthers << ", ";		
	    }
	}

	real_t sum_error = 0.0;
	for (int dftBufIndex = 0; dftBufIndex < this->m_DFTDataBuf.size(); dftBufIndex++){
	    sum_error += m_DFTDataBuf[dftBufIndex].m_specError;
	    sum_error += m_DFTDataBuf[dftBufIndex].m_phaseError;
	    sum_error += m_DFTDataBuf[dftBufIndex].m_resError;
	    sum_error += m_DFTDataBuf[dftBufIndex].m_realSpecError;
	    sum_error += m_DFTDataBuf[dftBufIndex].m_lpcError;
	    sum_error += m_DFTDataBuf[dftBufIndex].m_specErrorOthers;
	}
	return sum_error;
	    
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
		printf("\n\tDFTlayer get noise output from %s",m_noiseOutputLayer->name().c_str());
	    }
	    
	    // for F0 input layer
	    if (targetLayer.name() == m_f0InputLayerName){
		m_f0InputLayer = &targetLayer;
		printf("\n\tDFT layer get F0 infor from %s", m_f0InputLayer->name().c_str());
		printf(", assume last dimension of its output as F0");
	    }
	}

	// link layers for DFT losss
	if (m_otherSignalInputLayers_names.size()){
	    for (int layerIndex = 0; layerIndex < m_otherSignalInputLayers_names.size(); layerIndex++){
		if (targetLayer.name() == m_otherSignalInputLayers_names[layerIndex]){
		    m_otherSignalInputLayers_ptr.push_back(&targetLayer);
		    printf("\n\tDFT layer catches excitation layer %s",
			   m_otherSignalInputLayers_ptr[layerIndex]->name().c_str());
		}
	    }
	}
	
	// link layer of sine input
	if (m_sineInputLayer_str.size()){
	    if (targetLayer.name() == m_sineInputLayer_str){
		m_sineInputLayer_ptr = &targetLayer;
		printf("\n\tDFT layer catches sine source layer %s", m_sineInputLayer_str.c_str());
	    }
	}
	
	// link layer of F0 input
	if (m_f0InputLayer_str.size()){
	    if (targetLayer.name() == m_f0InputLayer_str){
		m_f0InputLayer_ptr = &targetLayer;
		printf("\n\tDFT layer catches F0 from %s", m_f0InputLayer_str.c_str());
	    }
	}

	return;
    }

    template <typename TDevice>
    std::vector<int> DFTPostoutputLayer<TDevice>::dependLayerIDs()
    {
	std::vector<int> tmp;
	if (m_otherSignalInputLayers_ptr.size())
	    for (int layerIndex = 0; layerIndex < m_otherSignalInputLayers_ptr.size(); layerIndex++)
		tmp.push_back(m_otherSignalInputLayers_ptr[layerIndex]->getLayerID());
	if (m_sineInputLayer_ptr)
	    tmp.push_back(m_sineInputLayer_ptr->getLayerID());
	if (m_f0InputLayer_ptr)
	    tmp.push_back(m_f0InputLayer_ptr->getLayerID());
	tmp.push_back(this->precedingLayer().getLayerID());
	return tmp;
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


/* Dustbin #1
   // When phase and amplitude use the same FFTMats
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
	

  Dustbin #2
	    // -- phase part
		    if (m_zeta > 0.0){
		    m_phaseError2 = sourceSig2.specPhaseDistance(targetSig2, fftDiffSig2);
		    fftDiffSig2.specPhaseGrad(sourceSig2, targetSig2);
		    fftDiffSig2.iFFT();
		    fftDiffSig2.collectGrad(m_zeta);
		    m_fftDiffDataPhase2 = m_fftDiffData2;
		    }else{
		    m_phaseError2 = 0;
		    }

  Dustbin #3
	    
		    // -- phase part
		    if (m_zeta > 0.0){
		    m_phaseError3 = sourceSig3.specPhaseDistance(targetSig3, fftDiffSig3);
		    fftDiffSig3.specPhaseGrad(sourceSig3, targetSig3);
		    fftDiffSig3.iFFT();
		    fftDiffSig3.collectGrad(m_zeta);
		    m_fftDiffDataPhase3 = m_fftDiffData3;
		    }else{
		    m_phaseError3 = 0;
		    }
	
*/
