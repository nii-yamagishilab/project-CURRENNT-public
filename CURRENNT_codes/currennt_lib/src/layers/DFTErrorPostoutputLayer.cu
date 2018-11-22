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
	// use preLayers[0] as fake preceding layers
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
    {
	if (this->parallelSequences() > 1)
	    throw std::runtime_error("\nDFTError is not implemented for parallel mode");

	if (this->size() > 1)
	    throw std::runtime_error("\nDFTError only allows output data dimension = 1");

	this->__loadOpts(layerChild);	
    }	


    template <typename TDevice>
    void DFTPostoutputLayer<TDevice>::__loadOpts(const helpers::JsonValue &layerChild)
    {
	int maxSeqLength = this->maxSeqLength();
	
	// basic configuration
	m_beta          = (layerChild->HasMember("beta") ? 
			   static_cast<real_t>((*layerChild)["beta"].GetDouble()) : 0.0);

	m_gamma         = (layerChild->HasMember("gamma") ? 
			   static_cast<real_t>((*layerChild)["gamma"].GetDouble()) : 0.0);

	m_zeta         = (layerChild->HasMember("zeta") ? 
			  static_cast<real_t>((*layerChild)["zeta"].GetDouble()) : 0.0);
	
	m_specDisType   = (layerChild->HasMember("specDisType") ? 
			   static_cast<real_t>((*layerChild)["specDisType"].GetInt()) :
			   FFTMAT_SPECTYPE_MSE);

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
	
	if (this->getSaveMemoryFlag())
	    throw std::runtime_error("Memory save mode should be turned off");
	
	
	int timeLength = (this->precedingLayer().curMaxSeqLength() *
			  this->precedingLayer().parallelSequences());

	
	// Compute waveform MSE
	if (m_beta > 0.0){
	    
	    internal::ComputeMseWaveform fn;
	    fn.layerSize = this->size();
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
			    this->outputs().begin()   + timeLength * this->size(),
			    this->precedingLayer().outputs().begin()  + timeLength * this->size(),
			    thrust::counting_iterator<int>(0) + timeLength * this->size())),
		     fn, (real_t)0, thrust::plus<real_t>()) / timeLength;	
	}else{
	    m_mseError = 0.0;
	}

	// Compute DFT amplitute and phase distance
	if (m_gamma > 0.0 || m_zeta > 0.0){
   
	    // FFT 1
	    // step0. build the data structure
	    helpers::FFTMat<TDevice> sourceSig(
			&this->_actualOutputs(), &this->m_fftSourceFramed,
			&this->m_fftSourceSigFFT,
			m_frameLength, m_frameShift, m_windowType, m_fftLength, m_fftBinsNum,
			m_frameNum, this->maxSeqLength(), timeLength,
			m_specDisType);

	    helpers::FFTMat<TDevice> targetSig(
			&this->_targets(), &this->m_fftTargetFramed,
			&this->m_fftTargetSigFFT,
			m_frameLength, m_frameShift, m_windowType, m_fftLength, m_fftBinsNum,
			m_frameNum, this->maxSeqLength(), timeLength,
			m_specDisType);

	    helpers::FFTMat<TDevice> fftDiffSig(
			&this->m_fftDiffData, &this->m_fftDiffFramed,
			&this->m_fftDiffSigFFT,
			m_frameLength, m_frameShift, m_windowType, m_fftLength, m_fftBinsNum,
			m_frameNum, this->maxSeqLength(), timeLength,
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
			m_frameNum, this->maxSeqLength(), timeLength,
			m_specDisType);

		helpers::FFTMat<TDevice> targetSigPhase(
			&this->_targets(), &this->m_fftTargetFramed,
			&this->m_fftTargetSigFFT,
			m_frameLength, m_frameShift, m_windowTypePhase, m_fftLength, m_fftBinsNum,
			m_frameNum, this->maxSeqLength(), timeLength,
			m_specDisType);

		helpers::FFTMat<TDevice> fftDiffSigPhase(
			&this->m_fftDiffDataPhase, &this->m_fftDiffFramed,
			&this->m_fftDiffSigFFT,
			m_frameLength, m_frameShift, m_windowTypePhase, m_fftLength, m_fftBinsNum,
			m_frameNum, this->maxSeqLength(), timeLength,
			m_specDisType);
		
		// step1. framing and windowing
		sourceSigPhase.frameSignal();
		targetSigPhase.frameSignal();
		
		// step2. fft
		sourceSigPhase.FFT();
		targetSigPhase.FFT();

		// calculate phase distortion
		m_phaseError = sourceSigPhase.specPhaseDistance(targetSigPhase, fftDiffSigPhase);
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
			m_frameNum2, this->maxSeqLength(), timeLength,
			m_specDisType);

		helpers::FFTMat<TDevice> targetSig2(
			&this->_targets(), &this->m_fftTargetFramed2,
			&this->m_fftTargetSigFFT2,
			m_frameLength2, m_frameShift2, m_windowType2, m_fftLength2, m_fftBinsNum2,
			m_frameNum2, this->maxSeqLength(), timeLength,
			m_specDisType);
		    
		helpers::FFTMat<TDevice> fftDiffSig2(
			&this->m_fftDiffData2, &this->m_fftDiffFramed2,
			&this->m_fftDiffSigFFT2,
			m_frameLength2, m_frameShift2, m_windowType2, m_fftLength2, m_fftBinsNum2,
			m_frameNum2, this->maxSeqLength(), timeLength,
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
			m_frameNum2, this->maxSeqLength(), timeLength,
			m_specDisType);

		    helpers::FFTMat<TDevice> targetSigPhase2(
			&this->_targets(), &this->m_fftTargetFramed2,
			&this->m_fftTargetSigFFT2,
			m_frameLength2, m_frameShift2, m_windowTypePhase2,
			m_fftLength2, m_fftBinsNum2,
			m_frameNum2, this->maxSeqLength(), timeLength,
			m_specDisType);
		    
		    helpers::FFTMat<TDevice> fftDiffSigPhase2(
			&this->m_fftDiffDataPhase2, &this->m_fftDiffFramed2,
			&this->m_fftDiffSigFFT2,
			m_frameLength2, m_frameShift2, m_windowTypePhase2,
			m_fftLength2, m_fftBinsNum2,
			m_frameNum2, this->maxSeqLength(), timeLength,
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
			m_frameNum3, this->maxSeqLength(), timeLength,
			m_specDisType);

		helpers::FFTMat<TDevice> targetSig3(
			&this->_targets(), &this->m_fftTargetFramed3,
			&this->m_fftTargetSigFFT3,
			m_frameLength3, m_frameShift3, m_windowType3, m_fftLength3, m_fftBinsNum3,
			m_frameNum3, this->maxSeqLength(), timeLength,
			m_specDisType);
		    
		helpers::FFTMat<TDevice> fftDiffSig3(
			&this->m_fftDiffData3, &this->m_fftDiffFramed3,
			&this->m_fftDiffSigFFT3,
			m_frameLength3, m_frameShift3, m_windowType3, m_fftLength3, m_fftBinsNum3,
			m_frameNum3, this->maxSeqLength(), timeLength,
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
			m_frameNum3, this->maxSeqLength(), timeLength,
			m_specDisType);

		    helpers::FFTMat<TDevice> targetSigPhase3(
			&this->_targets(), &this->m_fftTargetFramed3,
			&this->m_fftTargetSigFFT3,
			m_frameLength3, m_frameShift3, m_windowTypePhase3,
			m_fftLength3, m_fftBinsNum3,
			m_frameNum3, this->maxSeqLength(), timeLength,
			m_specDisType);
		    
		    helpers::FFTMat<TDevice> fftDiffSigPhase3(
			&this->m_fftDiffDataPhase3, &this->m_fftDiffFramed3,
			&this->m_fftDiffSigFFT3,
			m_frameLength3, m_frameShift3, m_windowTypePhase3,
			m_fftLength3, m_fftBinsNum3,
			m_frameNum3, this->maxSeqLength(), timeLength,
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
	
    }

    template <typename TDevice>
    void DFTPostoutputLayer<TDevice>::computeForwardPass(const int timeStep, const int nnState)
    {
	// Not implemented yet
	throw std::runtime_error("Not implemented DFTErrorPostoutput computeForwardPass(timeStep)");
    }


    template <typename TDevice>
    void DFTPostoutputLayer<TDevice>::computeBackwardPass(const int nnState)
    {
	if (this->getSaveMemoryFlag())
	    throw std::runtime_error("Memory save mode should be turned off");

	int timeLength = (this->precedingLayer().curMaxSeqLength() *
			  this->precedingLayer().parallelSequences());

	thrust::fill(this->precedingLayer().outputErrors().begin(),
		     this->precedingLayer().outputErrors().end(), 0.0);
	    
	// Gradients from waveform MSE
	if (m_beta > 0.0){
	    {{
		internal::ComputeMseWaveformGrad fn2;
		fn2.preData  = helpers::getRawPointer(this->precedingLayer().outputs());
		fn2.realTargetData = helpers::getRawPointer(this->_targets());
		fn2.preDim   = this->precedingLayer().size();
		fn2.featDim  = this->size();
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
		     this->outputs().begin()   + timeLength * this->size(),
		     thrust::counting_iterator<int>(0)  + timeLength * this->size())),
		  fn2);
	     }}	
	}

	// Gradients from spectral amplitude and phase
	if (m_gamma > 0.0){
	    
	    // FFT1
	    thrust::transform(m_fftDiffData.begin(),
			      m_fftDiffData.begin() + timeLength * this->size(),
			      this->precedingLayer().outputErrors().begin(),
			      this->precedingLayer().outputErrors().begin(),
			      thrust::plus<real_t>());
	    if (m_zeta > 0.0)
		thrust::transform(m_fftDiffDataPhase.begin(),
				  m_fftDiffDataPhase.begin() + timeLength * this->size(),
				  this->precedingLayer().outputErrors().begin(),
				  this->precedingLayer().outputErrors().begin(),
				  thrust::plus<real_t>());
	    
	    // FFT2
	    if (m_fftLength2){
		thrust::transform(m_fftDiffData2.begin(),
			      m_fftDiffData2.begin() + timeLength * this->size(),
			      this->precedingLayer().outputErrors().begin(),
			      this->precedingLayer().outputErrors().begin(),
			      thrust::plus<real_t>());
		if (m_zeta > 0.0)
		    thrust::transform(m_fftDiffDataPhase2.begin(),
				  m_fftDiffDataPhase2.begin() + timeLength * this->size(),
				  this->precedingLayer().outputErrors().begin(),
				  this->precedingLayer().outputErrors().begin(),
				  thrust::plus<real_t>());
	    }

	    // FFT3
	    if (m_fftLength3){
		thrust::transform(m_fftDiffData3.begin(),
			      m_fftDiffData3.begin() + timeLength * this->size(),
			      this->precedingLayer().outputErrors().begin(),
			      this->precedingLayer().outputErrors().begin(),
			      thrust::plus<real_t>());
		if (m_zeta > 0.0)
		    thrust::transform(m_fftDiffDataPhase3.begin(),
				  m_fftDiffDataPhase3.begin() + timeLength * this->size(),
				  this->precedingLayer().outputErrors().begin(),
				  this->precedingLayer().outputErrors().begin(),
				  thrust::plus<real_t>());
	    }
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
	}   
    }
    
	    
    
    template <typename TDevice>
    real_t DFTPostoutputLayer<TDevice>::calculateError()
    {
	
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
    
    template class DFTPostoutputLayer<Cpu>;
    template class DFTPostoutputLayer<Gpu>;
    
}
