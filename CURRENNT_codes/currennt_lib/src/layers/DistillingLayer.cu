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


#include "DistillingLayer.hpp"
#include "../helpers/getRawPointer.cuh"
#include "../helpers/Matrix.hpp"
#include "../helpers/FFTMat.hpp"
#include "../helpers/JsonClasses.hpp"
#include "../helpers/misFuncs.hpp"

#include "MDNLayer.hpp"

#include "../Configuration.hpp"
#include "../MacroDefine.hpp"

#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/for_each.h>
#include <thrust/reduce.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/fill.h>
#include <thrust/random.h>
#include <boost/foreach.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/algorithm/string.hpp>
#include <vector>

// Assume Gaussian, only consider the mean
#define DISTILLING_TYPE_LIKELIHOOD      0
// TYPE 0 + beta * MSE
#define DISTILLING_TYPE_KLD_GAUSSIAN    1
// 
#define DISTILLING_TYPE_AFTER_TRAINING -1 

namespace internal{
namespace{
    
    struct reverseGrad
    {
	int dataDim;
	int gradDim;
	bool accumulate;
	real_t  beta;
	int    *gradPtr;
	real_t *gradMDN;
	
	real_t *realTarget;
	real_t *genData;
	
	const char *patTypes;
	
	__host__ __device__ void operator() (const thrust::tuple<real_t&, int> &t) const
	{
	    int timeIdx = t.get<1>() / dataDim;
	    int dimIdx  = t.get<1>() % dataDim;
	    
	    real_t tmpGrad = 0.0;
	    
	    // skip dummy frame (for parallel sentence processing)
	    if (patTypes != NULL && patTypes[timeIdx] == PATTYPE_NONE){
		t.get<0>() = 0.0;
		
	    }else{
		int mixNum  = gradPtr[dimIdx * 3];      // how many mixtures
		int featDim = gradPtr[dimIdx * 3 + 1];  // dimension of feature
		int startD  = gradPtr[dimIdx * 3 + 2];  // position of the first dimension of mean
		
		for (int i = 0; i < mixNum; i++)
		    tmpGrad += (gradMDN[gradDim * timeIdx + startD + i * featDim]);
		
		if (accumulate)
		    t.get<0>() += tmpGrad * -1.0;
		else
		    t.get<0>() = tmpGrad * -1.0;

		if (beta > 0 && realTarget && genData)
		    t.get<0>() += beta * (genData[t.get<1>()] - realTarget[t.get<1>()]);
		
	    }
	}
    };

    
    struct ComputeMse
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


    // copy segment
    struct CopyPartDistillingLayer
    {
	real_t *source;
	real_t *target;

	int srcDim;
	int srcS;     // the first dimension to be copied in source stream

	int copyDim;  // dimension of the data to be copied

	int tarDim;
	int tarS;     // the first dimension to store the copied data in target stream

	bool accumulate;
	
	const char *patTypes;     // 
	
	__host__ __device__ void operator() (const thrust::tuple<real_t&, int> &t) const
	{
	    int outputIdx = t.get<1>();
	    int timeIdx   = outputIdx / copyDim;
	    int dimIdx    = outputIdx % copyDim;

	    // skip dummy frame (for parallel sentence processing)
	    if (patTypes != NULL && patTypes[timeIdx] == PATTYPE_NONE)
		return;
	    if (accumulate){
		target[timeIdx * tarDim + tarS + dimIdx] +=
		    source[timeIdx * srcDim + srcS + dimIdx];
	    }else{
		target[timeIdx * tarDim + tarS + dimIdx] =
		    source[timeIdx * srcDim + srcS + dimIdx];
	    }
	}
    };

    // copy segment
    struct CopyPartDistillingLayer_alpha
    {
	real_t *source;
	real_t *target;
	real_t alpha;
	
	int srcDim;
	int srcS;     // the first dimension to be copied in source stream

	int copyDim;  // dimension of the data to be copied

	int tarDim;
	int tarS;     // the first dimension to store the copied data in target stream

	bool accumulate;
	
	const char *patTypes;     // 
	
	__host__ __device__ void operator() (const thrust::tuple<real_t&, int> &t) const
	{
	    int outputIdx = t.get<1>();
	    int timeIdx   = outputIdx / copyDim;
	    int dimIdx    = outputIdx % copyDim;

	    // skip dummy frame (for parallel sentence processing)
	    if (patTypes != NULL && patTypes[timeIdx] == PATTYPE_NONE)
		return;
	    if (accumulate){
		target[timeIdx * tarDim + tarS + dimIdx] +=
		    source[timeIdx * srcDim + srcS + dimIdx] * alpha;
	    }else{
		target[timeIdx * tarDim + tarS + dimIdx] =
		    source[timeIdx * srcDim + srcS + dimIdx] * alpha;
	    }
	}
    };


    // calculate KLD
    struct calculateKLDDistillingLayer
    {
	
	real_t *prePara;
	int preParaDim;
	real_t *preGrad;
	
	real_t *targetPara;
	int targetParaDim;
	real_t *targetGrad;

	real_t alpha;
	int featDim;

	const char *patTypes;     // 
	__host__ __device__ void operator() (const thrust::tuple<real_t&, int> &t) const
	{
	    int timeIdx = t.get<1>() / featDim;
	    int featIdx = t.get<1>() % featDim;

	    if (patTypes != NULL && patTypes[timeIdx] == PATTYPE_NONE){
		t.get<0>() = 0.0;
		return;
	    }

	    
	    // assume the parameters are for Gaussian,
	    // prePara [x1, ... xN, mu_1, .., mu_N, s1, ..., s_N], N * 3 dimensions
	    // targetPara       [w, mu_1, ... mu_N, s1, ..., s_N], N * 2 + 1 dims
	    // std  = exp(s)

	    // log(std_teacher) - log(std_student)
	    real_t kld = targetPara[timeIdx * targetParaDim + 1 + featDim + featIdx] -
		prePara[timeIdx * preParaDim + featDim + featDim + featIdx];

	    real_t std_T = exp(targetPara[timeIdx * targetParaDim + 1 + featDim + featIdx]);
	    real_t std_S = exp(prePara[timeIdx * preParaDim + featDim + featDim + featIdx]);
	    real_t mu_T  = targetPara[timeIdx * targetParaDim + 1 + featIdx];
	    real_t mu_S  = prePara[timeIdx * preParaDim + featDim + featIdx];
	    real_t var_T = std_T * std_T;
	    real_t var_S = std_S * std_S;
	    
	    // KLD = ...
	    kld += (var_S - var_T + (mu_S - mu_T) * (mu_S - mu_T)) / var_T / 2.0;

	    // By the way, calculate the gradients
	    // teacher gradient, w, mu, s
	    targetGrad[timeIdx * targetParaDim ] = 0;
	    targetGrad[timeIdx * targetParaDim + 1 + featIdx] = (mu_T - mu_S) / var_T * alpha;
	    targetGrad[timeIdx * targetParaDim + 1 + featIdx + featDim] =
		(1.0 - var_S / var_T -  (mu_T - mu_S) * (mu_T - mu_S) / var_T) * alpha;

	    // student, mu, s
	    preGrad[timeIdx * preParaDim + featDim + featIdx] =
		targetGrad[timeIdx * targetParaDim + 1 + featIdx] * -1.0 * alpha;
	    preGrad[timeIdx * preParaDim + featDim + featDim + featIdx] =
		(var_S / var_T - 1.0) * alpha;
	    
	    t.get<0>() = kld * alpha;
		
	    
	}
    };


    struct ComputeMseGrad
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
    DistillingLayer<TDevice>::DistillingLayer(const helpers::JsonValue &layerChild,
					      const helpers::JsonValue &weightsSection,
					      std::vector<Layer<TDevice>*> &precedingLayers,
					      int maxSeqLength,
					      int layerID)
	// use preLayers[0] as fake preceding layers
	: SkipLayer<TDevice>(layerChild, weightsSection, precedingLayers,
			     maxSeqLength, layerID, false)
	, m_mseError       (0.0)
	, m_kld            (0.0)
	, m_specError      (0.0)
	, m_specError2     (0.0)
	, m_specError3     (0.0)
	, m_distillingOpt  (DISTILLING_TYPE_LIKELIHOOD)
	, m_beta           (0.0)
	, m_gamma          (0.0)
	, m_fftLength      (0)
	, m_fftBinsNum     (0)
	, m_frameLength    (0)
	, m_frameShift     (0)
	, m_frameNum       (0)	
    {

	m_distillingOpt = (layerChild->HasMember("distillingOpt") ? 
			   ((*layerChild)["distillingOpt"].GetInt()) : DISTILLING_TYPE_LIKELIHOOD);
		
	if (m_distillingOpt == DISTILLING_TYPE_LIKELIHOOD &&
	    this->size() != this->precedingLayer().size())
	    throw std::runtime_error("\nError Distilling layer size != preceding layer size");
	
	if (m_distillingOpt == DISTILLING_TYPE_KLD_GAUSSIAN &&
	    this->size() != this->precedingLayer().size() / 3)
	    throw std::runtime_error("\nError Distilling layer size != preceding layer size/3");
	
	m_beta          = (layerChild->HasMember("beta") ? 
			   static_cast<real_t>((*layerChild)["beta"].GetDouble()) : 0.0);

	m_gamma         = (layerChild->HasMember("gamma") ? 
			   static_cast<real_t>((*layerChild)["gamma"].GetDouble()) : 0.0);

	m_alpha         = (layerChild->HasMember("alpha") ? 
			   static_cast<real_t>((*layerChild)["alpha"].GetDouble()) : 1.0);

	m_specDisType   = (layerChild->HasMember("specDisType") ? 
			   static_cast<real_t>((*layerChild)["specDisType"].GetInt()) :
			   FFTMAT_SPECTYPE_AMP_LOG_MSE);

	if (m_gamma > 0.0){

	    // only use phase when spectral ditance is used
	    m_zeta         = (layerChild->HasMember("zeta") ? 
			      static_cast<real_t>((*layerChild)["zeta"].GetDouble()) : 0.0);

	    m_realTargetVector = this->outputs();
	    
	    if (this->parallelSequences() > 1)
		throw std::runtime_error("\nFFT is not implemented for parallel mode");

	    if (this->size() > 1)
		throw std::runtime_error("\nFFT in distilling layer only allows 1 dim signal");
	    
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

	if (m_beta > 0.0){	    
	    m_realTargetVector = this->outputs();
	}
	
	
	// to collect information about the gradients in MDN Layer
	m_gradPtr.clear();
    }	
    
    // Destructor
    template <typename TDevice>
    DistillingLayer<TDevice>::~DistillingLayer()
    {
    }
    
    // NN forward
    template <typename TDevice>
    void DistillingLayer<TDevice>::computeForwardPass(const int nnState)
    {

	if (this->getSaveMemoryFlag())
	    throw std::runtime_error("Memory save mode should be turned off");
	
	// initialization for backward pass
	// (because gradients will be accumulated from multiple layers)
	if (this->flagTrainingMode()){
	    thrust::fill(this->outputErrors().begin(), 
			 (this->outputErrors().begin() + 
			  this->curMaxSeqLength() * this->parallelSequences() * this->size()),
			 0.0);

	    thrust::fill(this->outputErrorsFromSkipLayer().begin(),
			 (this->outputErrorsFromSkipLayer().begin() + 
			  this->curMaxSeqLength() * this->parallelSequences() * this->size()),
			 0.0);
	}
	
	int timeLength = this->precedingLayer().curMaxSeqLength();
	timeLength = timeLength * this->precedingLayer().parallelSequences();



	// 
	if (this->m_distillingOpt == DISTILLING_TYPE_LIKELIHOOD){

	    // copy the real target 
	    thrust::copy(m_targetLayer->outputs().begin(),
			 m_targetLayer->outputs().begin() + timeLength * this->size(),
			 m_realTargetVector.begin());
	    
	    if (m_beta > 0.0){
	
		// compute MSE
		internal::ComputeMse fn;
		fn.layerSize = this->size();
		fn.patTypes  = helpers::getRawPointer(this->patTypes());

		m_mseError =
		    (real_t)thrust::transform_reduce(
		     thrust::make_zip_iterator(
			thrust::make_tuple(
			    this->precedingLayer().outputs().begin(),
			    m_realTargetVector.begin(),
			    thrust::counting_iterator<int>(0))),
		     thrust::make_zip_iterator(
			thrust::make_tuple(
			    this->precedingLayer().outputs().begin() + timeLength * this->size(),
			    m_realTargetVector.begin()               + timeLength * this->size(),
			    thrust::counting_iterator<int>(0)        + timeLength * this->size())),
		     fn,
		     (real_t)0,
		     thrust::plus<real_t>()) / timeLength;
	    }
	    
	    // directly copy the output of previous layer as the target of the teacher network
	    thrust::copy(this->precedingLayer().outputs().begin(),
			 this->precedingLayer().outputs().begin() + timeLength * this->size(),
			 m_targetLayer->outputs().begin());
	    
	}else if (this->m_distillingOpt == DISTILLING_TYPE_KLD_GAUSSIAN){

	    // 1. keep the real target
	    // 2. copy the generated targets into the target layer
	    // 3. calculate MSE or other error terms that doesn't require teacher model
	    
	    // keep the real target 
	    thrust::copy(m_targetLayer->outputs().begin(),
			 m_targetLayer->outputs().begin() + timeLength * this->size(),
			 m_realTargetVector.begin());

	    // step2.
	    {{
		    // assume the first dimension of the previous layer stores the
		    // generated signal
		internal::CopyPartDistillingLayer fn1;
		fn1.source = helpers::getRawPointer(this->precedingLayer().outputs());
		fn1.target = helpers::getRawPointer(this->m_targetLayer->outputs());
		fn1.srcDim = this->precedingLayer().size();
		fn1.srcS   = 0;
		fn1.copyDim = this->m_targetLayer->size();
		fn1.tarDim  = this->m_targetLayer->size();
		fn1.tarS    = 0;
		fn1.patTypes = helpers::getRawPointer(this->patTypes());
		fn1.accumulate = false;
		
		thrust::for_each(
		thrust::make_zip_iterator(
		    thrust::make_tuple(
			this->m_targetLayer->outputs().begin(), 
			thrust::counting_iterator<int>(0))),
		thrust::make_zip_iterator(
		    thrust::make_tuple(
			this->m_targetLayer->outputs().begin() + timeLength * this->size(), 
			thrust::counting_iterator<int>(0)      + timeLength * this->size())),
		fn1);

	    }}

	    // step3.
	    if (m_beta > 0.0){
		
		// compute MSE
		internal::ComputeMse fn;
		fn.layerSize = this->size();
		fn.patTypes  = helpers::getRawPointer(this->patTypes());

		m_mseError =
		    (real_t)thrust::transform_reduce(
		     thrust::make_zip_iterator(
			thrust::make_tuple(
			    this->m_targetLayer->outputs().begin(),
			    m_realTargetVector.begin(),
			    thrust::counting_iterator<int>(0))),
		     thrust::make_zip_iterator(
			thrust::make_tuple(
			    this->m_targetLayer->outputs().begin()   + timeLength * this->size(),
			    m_realTargetVector.begin()               + timeLength * this->size(),
			    thrust::counting_iterator<int>(0)        + timeLength * this->size())),
		     fn,
		     (real_t)0,
		     thrust::plus<real_t>()) / timeLength;
		
	    }

	    if (m_gamma > 0.0){

		// compute FFT lose

		// FFT 1
		//  here m_targetLayer->outputs() has loaded the source signal
		helpers::FFTMat<TDevice> sourceSig(
			&this->m_targetLayer->outputs(), &this->m_fftSourceFramed,
			&this->m_fftSourceSigFFT,
			m_frameLength, m_frameShift, 0, m_fftLength, m_fftBinsNum,
			m_frameNum, this->maxSeqLength(), timeLength,
			m_specDisType);

		helpers::FFTMat<TDevice> targetSig(
			&m_realTargetVector, &this->m_fftTargetFramed,
			&this->m_fftTargetSigFFT,
			m_frameLength, m_frameShift, 0, m_fftLength, m_fftBinsNum,
			m_frameNum, this->maxSeqLength(), timeLength,
			m_specDisType);

		helpers::FFTMat<TDevice> fftDiffSig(
			&this->m_fftDiffData, &this->m_fftDiffFramed,
			&this->m_fftDiffSigFFT,
			m_frameLength, m_frameShift, 0, m_fftLength, m_fftBinsNum,
			m_frameNum, this->maxSeqLength(), timeLength,
			m_specDisType);

		
		// FFT 
		// step1. frame
		sourceSig.frameSignal();
		targetSig.frameSignal();
		
		// step2. fft
		sourceSig.FFT();
		targetSig.FFT();

		// -- phase part
		if (m_zeta > 0.0){
		    // calculate phase distortion
		    m_phaseError = sourceSig.specPhaseDistance(targetSig, fftDiffSig);
		    fftDiffSig.specPhaseGrad(sourceSig, targetSig);
		    fftDiffSig.iFFT();
		    fftDiffSig.collectGrad(m_zeta);
		    m_fftDiffDataPhase = m_fftDiffData;
		}
		
		// step3. FFT distance
		m_specError = sourceSig.specAmpDistance(targetSig, fftDiffSig);
		
		// step4. Prepare Gradients
		fftDiffSig.specAmpGrad(sourceSig, targetSig);

		// step5. use iFFT to collect gradients w.r.t framed signal
		fftDiffSig.iFFT();

		// step6. collect gradients to the real signal
		fftDiffSig.collectGrad(m_gamma);
		// Gradients should be in m_fftDiffData		    
		
		
		// FFT 2
		if (m_fftLength2){
		    //  here m_targetLayer->outputs() has loaded the source signal
		    helpers::FFTMat<TDevice> sourceSig2(
			&this->m_targetLayer->outputs(), &this->m_fftSourceFramed2,
			&this->m_fftSourceSigFFT2,
			m_frameLength2, m_frameShift2, 0, m_fftLength2, m_fftBinsNum2,
			m_frameNum2, this->maxSeqLength(), timeLength,
			m_specDisType);

		    helpers::FFTMat<TDevice> targetSig2(
			&m_realTargetVector, &this->m_fftTargetFramed2,
			&this->m_fftTargetSigFFT2,
			m_frameLength2, m_frameShift2, 0, m_fftLength2, m_fftBinsNum2,
			m_frameNum2, this->maxSeqLength(), timeLength,
			m_specDisType);
		    
		    helpers::FFTMat<TDevice> fftDiffSig2(
			&this->m_fftDiffData2, &this->m_fftDiffFramed2,
			&this->m_fftDiffSigFFT2,
			m_frameLength2, m_frameShift2, 0, m_fftLength2, m_fftBinsNum2,
			m_frameNum2, this->maxSeqLength(), timeLength,
			m_specDisType);
		
		    sourceSig2.frameSignal();
		    targetSig2.frameSignal();
		    sourceSig2.FFT();
		    targetSig2.FFT();
		    
		    // -- phase part
		    if (m_zeta > 0.0){
			// calculate phase distortion
			m_phaseError2 = sourceSig2.specPhaseDistance(targetSig2, fftDiffSig2);
			fftDiffSig2.specPhaseGrad(sourceSig2, targetSig2);
			fftDiffSig2.iFFT();
			fftDiffSig2.collectGrad(m_zeta);
			m_fftDiffDataPhase2 = m_fftDiffData2;
		    }
		    
		    m_specError2 = sourceSig2.specAmpDistance(targetSig2, fftDiffSig2);
		    fftDiffSig2.specAmpGrad(sourceSig2, targetSig2);
		    fftDiffSig2.iFFT();
		    fftDiffSig2.collectGrad(m_gamma);
		}

		if (m_fftLength3){
		    //  here m_targetLayer->outputs() has loaded the source signal
		    helpers::FFTMat<TDevice> sourceSig3(
			&this->m_targetLayer->outputs(), &this->m_fftSourceFramed3,
			&this->m_fftSourceSigFFT3,
			m_frameLength3, m_frameShift3, 0, m_fftLength3, m_fftBinsNum3,
			m_frameNum3, this->maxSeqLength(), timeLength,
			m_specDisType);

		    helpers::FFTMat<TDevice> targetSig3(
			&m_realTargetVector, &this->m_fftTargetFramed3,
			&this->m_fftTargetSigFFT3,
			m_frameLength3, m_frameShift3, 0, m_fftLength3, m_fftBinsNum3,
			m_frameNum3, this->maxSeqLength(), timeLength,
			m_specDisType);
		    
		    helpers::FFTMat<TDevice> fftDiffSig3(
			&this->m_fftDiffData3, &this->m_fftDiffFramed3,
			&this->m_fftDiffSigFFT3,
			m_frameLength3, m_frameShift3, 0, m_fftLength3, m_fftBinsNum3,
			m_frameNum3, this->maxSeqLength(), timeLength,
			m_specDisType);

		    
		    sourceSig3.frameSignal();
		    targetSig3.frameSignal();
		    sourceSig3.FFT();
		    targetSig3.FFT();

		    // -- phase part
		    if (m_zeta > 0.0){
			// calculate phase distortion
			m_phaseError3 = sourceSig3.specPhaseDistance(targetSig3, fftDiffSig3);
			fftDiffSig3.specPhaseGrad(sourceSig3, targetSig3);
			fftDiffSig3.iFFT();
			fftDiffSig3.collectGrad(m_zeta);
			m_fftDiffDataPhase3 = m_fftDiffData3;
		    }
		    
		    m_specError3 = sourceSig3.specAmpDistance(targetSig3, fftDiffSig3);
		    fftDiffSig3.specAmpGrad(sourceSig3, targetSig3);
		    fftDiffSig3.iFFT();
		    fftDiffSig3.collectGrad(m_gamma);		    
		}
	    }
	    
	    
	}else if (m_distillingOpt == DISTILLING_TYPE_AFTER_TRAINING){
	    
	    // in the generation mode
	    
	    // directly copy the output to the postoutputlayer
	    
	    // assume the first dimension of the previous layer stores the
	    // generated signal
	    {{
		internal::CopyPartDistillingLayer fn1;
		fn1.source = helpers::getRawPointer(this->precedingLayer().outputs());
		fn1.target = helpers::getRawPointer(this->outputs());
		fn1.srcDim = this->precedingLayer().size();
		fn1.srcS   = 0;
		fn1.copyDim = this->m_targetLayer->size();
		fn1.tarDim  = this->m_targetLayer->size();
		fn1.tarS    = 0;
		fn1.patTypes = helpers::getRawPointer(this->patTypes());
		fn1.accumulate = false;
		
		thrust::for_each(
		thrust::make_zip_iterator(
		    thrust::make_tuple(
			this->outputs().begin(), 
			thrust::counting_iterator<int>(0))),
		thrust::make_zip_iterator(
		    thrust::make_tuple(
			this->outputs().begin() + timeLength * this->size(), 
			thrust::counting_iterator<int>(0)      + timeLength * this->size())),
		fn1);

	    }}
	}else{
	    throw std::runtime_error("\nUnknown distillingOpt type");
	}

    }

    // NN forward
    template <typename TDevice>
    void DistillingLayer<TDevice>::computeForwardPass(const int timeStep, const int nnState)
    {
	// Not implemented yet
	throw std::runtime_error("Not implemented distillingLayer computeForward(timeStep)");
	
    }


    // NN backward
    template <typename TDevice>
    void DistillingLayer<TDevice>::computeBackwardPass(const int nnState)
    {
	if (this->getSaveMemoryFlag())
	    throw std::runtime_error("Memory save mode should be turned off");

	int timeLength = this->precedingLayer().curMaxSeqLength();
	timeLength = timeLength * this->precedingLayer().parallelSequences();

	if (this->m_distillingOpt == DISTILLING_TYPE_LIKELIHOOD){

	    // only lieklihood 
	    internal::reverseGrad fn1;
	    
	    fn1.dataDim  = this->size();
	    fn1.gradDim  = m_targetBeforeLayer->size();
	    fn1.gradPtr  = helpers::getRawPointer(this->m_gradPtr);
	    fn1.gradMDN  = helpers::getRawPointer(m_targetBeforeLayer->outputErrors());
	    fn1.patTypes = helpers::getRawPointer(this->patTypes());
	    
	    fn1.beta     = m_beta;
	    if (m_beta > 0.0){
		fn1.realTarget = helpers::getRawPointer(this->m_realTargetVector);
		fn1.genData    = helpers::getRawPointer(m_targetLayer->outputs());
	    }else{
		fn1.realTarget = NULL;
		fn1.genData    = NULL;
	    }
	    SkipLayer<TDevice>* tempLayerAb =
		dynamic_cast<SkipLayer<TDevice>*>(&this->precedingLayer());

	    if (tempLayerAb){
		
		fn1.accumulate = true;
		thrust::for_each(
		 thrust::make_zip_iterator(
		   thrust::make_tuple(
		     tempLayerAb->outputErrorsFromSkipLayer().begin(),
		     thrust::counting_iterator<int>(0))),
		 thrust::make_zip_iterator(
		  thrust::make_tuple(
		     tempLayerAb->outputErrorsFromSkipLayer().begin() + timeLength * this->size(),
		     thrust::counting_iterator<int>(0)                + timeLength * this->size())),
		 fn1);
		
	    }else{
		
		fn1.accumulate = false;
		thrust::for_each(
		 thrust::make_zip_iterator(
		   thrust::make_tuple(
		     this->precedingLayer().outputErrors().begin(),
		     thrust::counting_iterator<int>(0))),
		 thrust::make_zip_iterator(
		  thrust::make_tuple(
		     this->precedingLayer().outputErrors().begin() + timeLength * this->size(),
		     thrust::counting_iterator<int>(0)             + timeLength * this->size())),
		 fn1);
		
	    }
	    
	}else if (this->m_distillingOpt == DISTILLING_TYPE_KLD_GAUSSIAN){
	    
	    SkipLayer<TDevice>* tempLayer =
		dynamic_cast<SkipLayer<TDevice>*>(&this->precedingLayer());

	    // KLD (possibly with MSE)
	    // Gradients w.r.t teacher have been sent in intermediaterror()
	    // Gradients w.r.t student mu, std have been sent in intermediateError()

	    
	    // This part only receive the gradients w.r.t x from teacher 
	    //
	    if (m_beta > 0.0){
		{{
		internal::ComputeMseGrad fn2;
		fn2.preData  = helpers::getRawPointer(this->precedingLayer().outputs());
		fn2.realTargetData = helpers::getRawPointer(this->m_realTargetVector);
		fn2.preDim   = this->precedingLayer().size();
		fn2.featDim  = this->size();
		fn2.beta     = m_beta;
		fn2.patTypes = helpers::getRawPointer(this->patTypes());
	    
		if (tempLayer){
		    fn2.accumulate = true;
		    fn2.preError   = helpers::getRawPointer(tempLayer->outputErrorsFromSkipLayer());
		}else{
		    fn2.accumulate = false;
		    fn2.preError   = helpers::getRawPointer(this->precedingLayer().outputErrors());
		}
	    
		thrust::for_each(
		  thrust::make_zip_iterator(
		   thrust::make_tuple(
		     this->m_realTargetVector.begin(),
		     thrust::counting_iterator<int>(0))),
		  thrust::make_zip_iterator(
		  thrust::make_tuple(
		     this->m_realTargetVector.begin()   + timeLength * this->size(),
		     thrust::counting_iterator<int>(0)  + timeLength * this->size())),
		  fn2);
		}}
		
	    }

	    if (m_gamma > 0.0){
		
		{{
		    internal::CopyPartDistillingLayer fn1;
		    fn1.source  = helpers::getRawPointer(m_fftDiffData);
		    fn1.srcDim  = this->size();
		    fn1.srcS    = 0;
		    fn1.copyDim = this->size();
		
		    if (tempLayer){
			fn1.target = helpers::getRawPointer(tempLayer->outputErrorsFromSkipLayer());
			fn1.accumulate = true;
		    }else{
			fn1.target = helpers::getRawPointer(this->precedingLayer().outputErrors());
			fn1.accumulate = false;
		    }
		    fn1.tarDim = this->precedingLayer().size();
		    fn1.tarS    = 0;
		    fn1.patTypes = helpers::getRawPointer(this->patTypes());
		
		    thrust::for_each(
		      thrust::make_zip_iterator(
		        thrust::make_tuple(
			  this->outputs().begin(), 
			  thrust::counting_iterator<int>(0))),
		      thrust::make_zip_iterator(
		        thrust::make_tuple(
			  this->outputs().begin()           + timeLength * this->size(), 
			  thrust::counting_iterator<int>(0) + timeLength * this->size())),
		      fn1);


		    if (m_zeta > 0.0){
			fn1.source  = helpers::getRawPointer(m_fftDiffDataPhase);
			fn1.accumulate = true;    
			thrust::for_each(
		          thrust::make_zip_iterator(
		            thrust::make_tuple(
				  this->outputs().begin(), 
				  thrust::counting_iterator<int>(0))),
			  thrust::make_zip_iterator(
		            thrust::make_tuple(
				  this->outputs().begin()           + timeLength * this->size(), 
				  thrust::counting_iterator<int>(0) + timeLength * this->size())),
			  fn1);
		    }

		    
		    if (m_fftLength2){
			fn1.source  = helpers::getRawPointer(m_fftDiffData2);
			fn1.accumulate = true;    
			thrust::for_each(
		          thrust::make_zip_iterator(
		            thrust::make_tuple(
				  this->outputs().begin(), 
				  thrust::counting_iterator<int>(0))),
			  thrust::make_zip_iterator(
		            thrust::make_tuple(
				  this->outputs().begin()           + timeLength * this->size(), 
				  thrust::counting_iterator<int>(0) + timeLength * this->size())),
			  fn1);
			
			if (m_zeta > 0.0){
			fn1.source  = helpers::getRawPointer(m_fftDiffDataPhase2);
			fn1.accumulate = true;    
			thrust::for_each(
		          thrust::make_zip_iterator(
		            thrust::make_tuple(
				  this->outputs().begin(), 
				  thrust::counting_iterator<int>(0))),
			  thrust::make_zip_iterator(
		            thrust::make_tuple(
				  this->outputs().begin()           + timeLength * this->size(), 
				  thrust::counting_iterator<int>(0) + timeLength * this->size())),
			  fn1);
			}

		    
		    }

		    if (m_fftLength3){
			fn1.source  = helpers::getRawPointer(m_fftDiffData3);
			fn1.accumulate = true;    
			thrust::for_each(
		          thrust::make_zip_iterator(
		            thrust::make_tuple(
				  this->outputs().begin(), 
				  thrust::counting_iterator<int>(0))),
			  thrust::make_zip_iterator(
		            thrust::make_tuple(
				  this->outputs().begin()           + timeLength * this->size(), 
				  thrust::counting_iterator<int>(0) + timeLength * this->size())),
			  fn1);
			if (m_zeta > 0.0){
			fn1.source  = helpers::getRawPointer(m_fftDiffDataPhase3);
			fn1.accumulate = true;    
			thrust::for_each(
		          thrust::make_zip_iterator(
		            thrust::make_tuple(
				  this->outputs().begin(), 
				  thrust::counting_iterator<int>(0))),
			  thrust::make_zip_iterator(
		            thrust::make_tuple(
				  this->outputs().begin()           + timeLength * this->size(), 
				  thrust::counting_iterator<int>(0) + timeLength * this->size())),
			  fn1);
			}

		    
		    }
		}}
		
		
	    }
	    
	    // Receive Gradients from the input layer of teacher model
	    //  if that layer is a feedback layer, shift the gradients by 1 step in time
	    if (this->followingLayer().type() == "feedback"){
		if (this->m_alpha > 0){{

		internal::CopyPartDistillingLayer_alpha fn1;
		fn1.source = helpers::getRawPointer(this->followingLayer().outputErrors());
		fn1.target = helpers::getRawPointer(this->precedingLayer().outputErrors()) +
		    this->parallelSequences() * this->precedingLayer().size();
		fn1.accumulate = false;
		fn1.srcDim = this->followingLayer().size();
		fn1.srcS   = 0;
		fn1.alpha  = this->m_alpha;
		fn1.copyDim = this->size();
		fn1.tarDim  = this->precedingLayer().size();
		fn1.tarS    = 0;
		fn1.patTypes = helpers::getRawPointer(this->patTypes());
		
		thrust::for_each(
		thrust::make_zip_iterator(
		    thrust::make_tuple(
			this->followingLayer().outputErrors().begin(), 
			thrust::counting_iterator<int>(0))),
		thrust::make_zip_iterator(
		    thrust::make_tuple(
			this->followingLayer().outputErrors().begin() +
			(timeLength - this->parallelSequences()) * this->size(), 
			thrust::counting_iterator<int>(0)      +
			(timeLength - this->parallelSequences()) * this->size())),
		fn1);

		}}
		
	    }else{
		throw std::runtime_error("Not implemented if the following layer is not feedback");
	    }
	
	}else if (m_distillingOpt == DISTILLING_TYPE_AFTER_TRAINING){
	    printf("\nDistilling layer is in generation mode.");
	    throw std::runtime_error("Please check distilling layer, or postoutput layer type");
	    
	}else{
	    // nothing
	}
		
    }
	
   // NN backward
    template <typename TDevice>
    void DistillingLayer<TDevice>::computeBackwardPass(const int timeStep, const int nnState)
    {
	throw std::runtime_error("Not implemented distillingLayer computeBackward(timeStep)");
    }
    
    template <typename TDevice>
    const std::string& DistillingLayer<TDevice>::type() const
    {
	static std::string s1("distilling");
        return s1;
    }

    template <typename TDevice>
    void DistillingLayer<TDevice>::exportLayer(const helpers::JsonValue &layersArray,
					 const helpers::JsonAllocator &allocator) const
    {
	SkipLayer<TDevice>::exportLayer(layersArray, allocator);
	(*layersArray)[layersArray->Size() - 1].AddMember("distillingOpt",
							  m_distillingOpt,
							  allocator);

	(*layersArray)[layersArray->Size() - 1].AddMember("alpha", m_alpha, allocator);
	    
	if (m_beta > 0.0)
	    (*layersArray)[layersArray->Size() - 1].AddMember("beta", m_beta, allocator);

	if (m_specDisType != FFTMAT_SPECTYPE_AMP_LOG_MSE)
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

	    if (m_fftLength2){
		(*layersArray)[layersArray->Size() - 1].AddMember("fftLength2", m_fftLength2,
								  allocator);
		(*layersArray)[layersArray->Size() - 1].AddMember("frameLength2", m_frameLength2,
								  allocator);
		(*layersArray)[layersArray->Size() - 1].AddMember("frameShift2", m_frameShift2,
								  allocator);
	    }
	    
	    if (m_fftLength3){
		(*layersArray)[layersArray->Size() - 1].AddMember("fftLength3", m_fftLength3,
								  allocator);
		(*layersArray)[layersArray->Size() - 1].AddMember("frameLength3", m_frameLength3,
								  allocator);
		(*layersArray)[layersArray->Size() - 1].AddMember("frameShift3", m_frameShift3,
								  allocator);
	    }
	     
	    
	}

	    
    }
    
    template <typename TDevice>
    void DistillingLayer<TDevice>::reduceOutputBuffer()
    {
	this->resizeOutputBuffer(this->parallelSequences() * this->size());
	this->setSaveMemoryFlag(true);
	printf("\t[mem saved]");
    }

    template <typename TDevice>
    int DistillingLayer<TDevice>::outputBufPtrBias(const int timeStepTimesParallel,
						    const int nnState)
    {
	if (this->getSaveMemoryFlag()){
	    return timeStepTimesParallel * this->size();
	}else{
	    return 0;
	}
    }

    template <typename TDevice>
    void DistillingLayer<TDevice>::linkTargetLayer(Layer<TDevice> &targetLayer)
    {

	m_targetLayer = &targetLayer;
	
	if (m_targetLayer->getResolution() != this->getResolution()){
	    printf("Error: %s vs %s, ", targetLayer.name().c_str(), this->name().c_str());
	    throw std::runtime_error("Resolution mismatch");
	}
		
	if (m_targetLayer->type() == "mdn"){
	    // when the target is MDN
	
	    // We need to know which dimension is the mean and which dimension is the std    
	    layers::MDNLayer<TDevice>* tmpPtr =
		dynamic_cast<layers::MDNLayer<TDevice>*>(m_targetLayer);
	
	    if (tmpPtr){
		Cpu::real_vector mdnConfigVec = tmpPtr->getMdnConfigVec();
		Cpu::int_vector tmpGradPtr  = Cpu::int_vector(this->size() * 3, 0.0);
	    
		if (!mdnConfigVec.empty()){
		    for (int x = 0; x < (mdnConfigVec.size()-1)/5; x++){

			int mdnType  = (int)mdnConfigVec[5+x*5];
			int unitSOut = (int)mdnConfigVec[3+x*5];
			int unitEOut = (int)mdnConfigVec[4+x*5];

			if (mdnType != 1)
			    throw std::runtime_error("distilling is only implemented for Gaussian");
		    
			if (mdnType == MDN_TYPE_SIGMOID || mdnType == MDN_TYPE_SOFTMAX){
			    // If the target distribution is binar/multinomial,
			    // normalizing flow cannot be used
			    throw std::runtime_error("distilling can only be used for GMM/Gaussian");
			}else{
			    // If the target distribution is Gaussian/GMM
			    int paraDimS = (int)mdnConfigVec[1+x*5];
			    for (int y = unitSOut; y < unitEOut; y++){
				tmpGradPtr[y * 3 + 0] = mdnType;           // mixture number
				tmpGradPtr[y * 3 + 1] = unitEOut-unitSOut; // feature dim
				tmpGradPtr[y * 3 + 2] = paraDimS + mdnType + (y - unitSOut);
				// startD
			    }
			}
		    }
		}
		
		m_gradPtr  = tmpGradPtr;
		
	    }else{
		throw std::runtime_error("the target layer is not a MDN a layer");
	    }
	
	    m_targetBeforeLayer = &(targetLayer.precedingLayer());


	    // check the layer size
	    if (m_targetLayer->size() != this->size()){
		printf("Error: %s vs %s, ", m_targetLayer->name().c_str(), this->name().c_str());
		throw std::runtime_error("unequal layer size");
	    }

	    if (m_distillingOpt == DISTILLING_TYPE_KLD_GAUSSIAN &&
		(m_targetBeforeLayer->size() / 3 * 2) != (this->precedingLayer().size()-1)){
		printf("Error: %s vs %s, ",
		       m_targetBeforeLayer->name().c_str(),
		       this->precedingLayer().name().c_str());
		throw std::runtime_error("layer size before target should be = previous layer size");
	    }

	    // another check
	    if (this->followingLayer().type() != "feedback"){
		throw std::runtime_error("Distilling is not implemented for non-feedback model");
	    }else{
		if (this->followingLayer().size() != this->size())
		    throw std::runtime_error("Feedback layer size != distilling layer size");
	    }
	}else{
	    
	    // when the target layer is not MDN (only used during generation)
	    m_gradPtr.clear();
	    m_targetBeforeLayer = NULL;
	    m_distillingOpt = DISTILLING_TYPE_AFTER_TRAINING;

	    printf("\nDistilling: network in generation mode as target layer is not MDN layer\n");
	    
	}
    }

    template <typename TDevice>
    std::vector<int> DistillingLayer<TDevice>::dependLayerIDs()
    {
	// by default, only the previous layer
	std::vector<int> depend_layerIDs;
	depend_layerIDs.push_back(this->precedingLayer().getLayerID());
	return depend_layerIDs;
    }

    template <typename TDevice>
    void DistillingLayer<TDevice>::clearAllBuffers()
    {
	// do nothing for distilling layer
    }

    template <typename TDevice>
    void DistillingLayer<TDevice>::resizeAllBuffers(const int timeLength)
    {
	// do nothing for distilling layer
    }
    
    template <typename TDevice>
    void DistillingLayer<TDevice>::logAllBuffers(
		helpers::vecPoolManager<TDevice> &vecPoolMng,
		bool flag_add)
    {
    }
    
    template <typename TDevice>
    void DistillingLayer<TDevice>::swapAllBuffers(
		helpers::vecPoolManager<TDevice> &vecPoolMng,
		bool flag_get)
    {
    }
	    
    
    template <typename TDevice>
    real_t DistillingLayer<TDevice>::intermediateError()
    {
	int timeLength = this->precedingLayer().curMaxSeqLength();
	timeLength = timeLength * this->precedingLayer().parallelSequences();

	if (m_distillingOpt == DISTILLING_TYPE_LIKELIHOOD){
	    //
	    return m_mseError;
	    
	}else if (m_distillingOpt == DISTILLING_TYPE_KLD_GAUSSIAN){
	    // mseError calculation should be moved here
	    
	    // KLD calculation
	    //  meanwhile, calculate the gradients (maybe move the gradients calculation
	    //  to another place)
	    if (m_alpha > 0){{
		    
		internal::calculateKLDDistillingLayer fn;
		fn.prePara = helpers::getRawPointer(this->precedingLayer().outputs());
		fn.preParaDim = this->precedingLayer().size();

		fn.alpha    = this->m_alpha;
		SkipLayer<TDevice>* tempLayer =
		    dynamic_cast<SkipLayer<TDevice>*>(&this->precedingLayer());

		if (tempLayer){
		    fn.preGrad = helpers::getRawPointer(tempLayer->outputErrorsFromSkipLayer());
		}else{
		    fn.preGrad = helpers::getRawPointer(this->precedingLayer().outputErrors());
		}
		    
		fn.targetPara = helpers::getRawPointer(this->m_targetBeforeLayer->outputs());
		fn.targetParaDim = this->m_targetBeforeLayer->size();
		fn.targetGrad = helpers::getRawPointer(this->m_targetBeforeLayer->outputErrors());
		
		fn.featDim = this->size();
		fn.patTypes  = helpers::getRawPointer(this->patTypes());

		thrust::for_each(
		     thrust::make_zip_iterator(
		       thrust::make_tuple(
			this->outputs().begin(),
			thrust::counting_iterator<int>(0))),
		     thrust::make_zip_iterator(
		       thrust::make_tuple(
			this->outputs().begin()             + timeLength * this->size(),
			thrust::counting_iterator<int>(0)   + timeLength * this->size())),
		     fn);
		
		m_kld = thrust::reduce(this->outputs().begin(),
				       this->outputs().begin() + timeLength * this->size());
		m_kld = m_kld / timeLength;
		
	    }}

	    if (Configuration::instance().verboseLevel() == OP_VERBOSE_LEVEL_1){
		std::cerr << m_specError << ", " << m_specError2 << ", " << m_specError3;
		std::cerr << ", " << m_mseError << ", " << m_kld << ", ";
		if (m_zeta > 0.0){
		    std::cerr << m_phaseError << ", " << m_phaseError2 << ", ";
		    std::cerr << m_phaseError3 << ", ";
		}
	    }
	    return m_specError + m_specError2 + m_specError3;
	    
	}else{
	    return 0.0;
	}
    }
    
    template class DistillingLayer<Cpu>;
    template class DistillingLayer<Gpu>;
    
}
