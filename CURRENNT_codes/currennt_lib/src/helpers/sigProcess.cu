/******************************************************************************
 * This file is an addtional component of CURRENNT. 
 * Xin WANG
 * National Institute of Informatics, Japan
 * 2018 - 2019
 *
 * This file is part of CURRENNT. *
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

#include "sigProcess.hpp"
#include "getRawPointer.cuh"
#include "safeExp.cuh"
#include "FFTMat.hpp"

#include <stdexcept>
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/random.h>
#include <thrust/transform_reduce.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>

// Used to adjust the amplitude of waveform into [-128, 128]
#define SIGPROCESS_WAVEFORM_LEVEL 128.0
#define SIGPROCESS_AUTOCORR_THESHOLD 0.000000000000001
#define SIGPROCESS_LPC_0ORDER_THESHOLD 0.00001

namespace internal{
namespace {

    struct scaleError
    {
	real_t factor;
	
	__host__ __device__ real_t operator() (const real_t &t) const
	{
	    return t / factor;
	}
    };

    struct lpcAutoCorr
    {
	int polyOrder;     // maximum number of auto-correlation-order
	int frameLength;   // length of one frame
	int frameBufLen;   // length of the frame buffer allocated to each frame
	real_t *frameData; // framed speech data

	// t.get<0>(): array of auto-correlation [frames, auto-correlation-order]
	// t.get<1>(): index of the data cell in auto-correlation array
	// 
	// Parallelized over frames and auto-correlation order 
	__host__ __device__ void operator() (const thrust::tuple<real_t &, int> &t) const
	{
	    // which frame?
	    int frameIdx = t.get<1>() / polyOrder;
	    // which auto-correlation coefficient is to be calculated?
	    int shiftValue = t.get<1>() % polyOrder;

	    // temporary buffer
	    real_t tmp_sum = 0.0;

	    // calculate the auto-correlation as auto_corr(k) = \sum_{t=-inf}^{inf} x[n]x[n+k]
	    for (int index = 0; index < frameLength; index++){
		if (index + shiftValue >= frameLength)
		    break;
		tmp_sum += (frameData[frameIdx * frameBufLen + index] *
			    frameData[frameIdx * frameBufLen + index + shiftValue]);
	    }
	    // save auto-corr(k) for the frameIdx-th frame
	    t.get<0>() = tmp_sum;
	}
    };

    struct lpcLevinsonDurbinRecursion
    {
	int polyOrder;     // order of LPC + 1, for example polyOrder = k+1 for [1, a_1, ..., a_k]
	                   
	// input
	real_t *autoCorr;  // matrix of auto-correlation values, [frame, polyOrder]

	// output
	real_t *refCoef;   // reflection coefficients array, [frame, polyOrder]
	real_t *lpcErr;    // LPC analysis error array, [frame, polyOrder]
	real_t *lpcCoef;   // LPC coefficients, [frame, polyOrder]

	// Calcualte the LPC coeffient for one frame, assume LPC model (in z-domain)
	//    1 + \sum_k=1^K a_k z^-k
	
	__host__ __device__ void operator() (const thrust::tuple<real_t &, int> &t) const
	{
	    // frame index
	    int frameIdx = t.get<1>();

	    // a temporary polyOrder, which may be changed if the reflection coefficiets is unstable
	    int tmp_polyOrder = polyOrder;
		
	    // pointer that points to autoCorr, refCoef, lpcErr of current frame
	    long int dataPtr = frameIdx * polyOrder;
	    // pointer that points to lpcCoef of current frame
	    long int lpcCoefPtr = frameIdx * polyOrder * 2;

	    // reflection coef
	    real_t tmp_refCoef = 0.0;

	    // temporary value for swapping
	    real_t tmp_lpcCoef = 0.0;

	    
	    // if the frame is all 0.0, return
	    if (autoCorr[dataPtr + 0] < SIGPROCESS_AUTOCORR_THESHOLD)
		return;
	    
	    // initialization
	    lpcErr[dataPtr] = autoCorr[dataPtr + 0];  // 0-order LPC, error = auto-corr[0]
	    lpcCoef[lpcCoefPtr] = 1.0;                // 0-order LPC, only coefficient 1	    
	    
	    // lpcCoef[lpcCoefPtr] ->   [a_p^p, a_p-1^p, ... 1.0, 0, 0]
	    // lpcCoef[lpcCoefPtr + polyOrder] -> [a_p+1^p+1, a_p^p+1, ... 1.0, 0]

	    // from order 1 to order polyOrder - 1 = lpcOrder
	    for (int order = 1; order < polyOrder; order ++){
		// set 1.0 in LPC coefficients
		lpcCoef[lpcCoefPtr + polyOrder + order] = 1.0;

		// calculate reflection coef
		tmp_refCoef = 0.0;
		for (int order2 = 0; order2 < order; order2++)
		    tmp_refCoef += (lpcCoef[lpcCoefPtr + order2] *
				    autoCorr[dataPtr + 1 + order2]);
		tmp_refCoef /= lpcErr[dataPtr + order - 1];
		
		refCoef[dataPtr + order] = tmp_refCoef;

		// update LPC coefficients
		lpcCoef[lpcCoefPtr + polyOrder] = -1.0 * tmp_refCoef;

		// for iteration in higher order
		if (order > 1){
		    for (int order2 = 1; order2 < order; order2++)
			lpcCoef[lpcCoefPtr + polyOrder + order2] =
			    lpcCoef[lpcCoefPtr + order2 - 1] -
			    tmp_refCoef * lpcCoef[lpcCoefPtr + order - 1 - order2];
		}
		
		// lpc error
		lpcErr[dataPtr + order] = lpcErr[dataPtr + order - 1] *
		    (1 - tmp_refCoef * tmp_refCoef);

		// check the reflection coefficients
		//  if |tmp_refCoef| > 1.0, return with the LPC coefficients of order - 1
		if (lpcErr[dataPtr + order] < 0.0){
		    tmp_polyOrder = order;
		    break;
		}

		// copy the lpc coeff (including the 1 of 0-order LPC)
		for (int order2 = 0; order2 <= order; order2++)
		    lpcCoef[lpcCoefPtr + order2] = lpcCoef[lpcCoefPtr + polyOrder + order2];
	    }

	    // Reverse the order of LPC coef [a_p^p, a_p^p-1, ... 1.0]  -> [1.0, a_p^1, ... a_p^p]
	    for (int order = 0; order < (tmp_polyOrder / 2); order++){
		tmp_lpcCoef = lpcCoef[lpcCoefPtr + tmp_polyOrder - 1 - order];
		lpcCoef[lpcCoefPtr + tmp_polyOrder - 1 - order] = lpcCoef[lpcCoefPtr + order];
		lpcCoef[lpcCoefPtr + order] = tmp_lpcCoef;
	    }
	}
    };

    
    struct lpcResidual
    {
	int polyOrder;     // maximum number of auto-correlation-order
	int frameLength;   // length of one frame
	int frameBufLen;   // length of the frame buffer allocated to each frame
	
	real_t *framedData; // framed speech data
	real_t *lpcCoef;   // LPC coefficients
	
	// Calcualte the LPC residual for one frame, assume LPC model (in z-domain)
	//    r_t = o_t + \sum_k=1^K a_k o_t-k
	// 
	// t.get<0>(): array of residual [frames, frameBuflength]
	// t.get<1>(): index of the data cell in residual array
	// 
	// Parallelized over frames and frame_length
	__host__ __device__ void operator() (const thrust::tuple<real_t &, int> &t) const
	{
	    // which frame?
	    int frameIdx = t.get<1>() / frameBufLen;
	    // which sampling point in the frame
	    int posInFrame = t.get<1>() % frameBufLen;

	    if (posInFrame >= frameLength ||
		lpcCoef[frameIdx * polyOrder * 2] < SIGPROCESS_LPC_0ORDER_THESHOLD){
		// note, the residual signal buffer has the same dimension as framedData
		// number of data in one frame is frameBufLen  > frameLength
		// I just keep the shape of residual signal buffer be identical to framedData
		
		t.get<0>() = 0;
	    }else{
		// temporary buffer
		real_t tmp_sum = framedData[frameIdx * frameBufLen + posInFrame];
		real_t lpc_coef = 0.0;
	    
		for (int order = 1; order < polyOrder; order++){
		    if (posInFrame - order < 0)
			break;
		    
		    lpc_coef = lpcCoef[frameIdx * polyOrder * 2 + order];
		    tmp_sum += framedData[frameIdx * frameBufLen + posInFrame - order] * lpc_coef;
		
		}
		
		// save r_k for frameIdx-th frame
		t.get<0>() = tmp_sum;
	    }
	}
    };


    struct lpcSynthesis
    {
	int polyOrder;     // maximum number of auto-correlation-order
	int frameLength;   // length of one frame
	int frameBufLen;   // length of the frame buffer allocated to each frame

	real_t *residualData;
	real_t *waveData;     // framed speech data
	real_t *lpcCoef;      // LPC coefficients
	
	// Calcualte the LPC residual for one frame, assume LPC model (in z-domain)
	//    o_t = r_t - \sum_k=1^K a_k o_t-k
	// 
	// t.get<0>(): not used
	// t.get<1>(): index of frame
	// 
	// Parallelized over frames
	__host__ __device__ void operator() (const thrust::tuple<real_t &, int> &t) const
	{
	    // which frame?
	    int frameIdx = t.get<1>();

	    real_t tmpValue = 0.0;
	    
	    if (lpcCoef[frameIdx * polyOrder * 2] < SIGPROCESS_LPC_0ORDER_THESHOLD){
		// void frame
		return;
		
	    }else{
		for (int timeStep = 0; timeStep < frameBufLen; timeStep++){

		    if (timeStep < frameLength){
			
			// time range within the frame length
			tmpValue = residualData[frameIdx * frameBufLen + timeStep];
			
			for (int order = 1; order < polyOrder; order++){
			    if (timeStep - order >= 0){

				if (waveData == NULL){
				    tmpValue = tmpValue - lpcCoef[frameIdx * polyOrder * 2 + order] *
					residualData[frameIdx * frameBufLen + timeStep - order];
				}else{
				    tmpValue = tmpValue - lpcCoef[frameIdx * polyOrder * 2 + order] *
					waveData[frameIdx * frameBufLen + timeStep - order];
				}
			    }else{
				break;
			    }
			}
			
			if (waveData == NULL)
			    residualData[frameIdx * frameBufLen + timeStep] = tmpValue;
			else
			    waveData[frameIdx * frameBufLen + timeStep] = tmpValue;
		    }else{
			// time range beyond the frame length
			if (waveData == NULL){
			    residualData[frameIdx * frameBufLen + timeStep] = 0.0;
			}else{
			    waveData[frameIdx * frameBufLen + timeStep] = 0.0;
			}
			
		    }
		}
	    }
	    
	}
    };


    struct lpcResidualDifference
    {
	int frameLength;   // length of one frame
	int frameBufLen;   // length of the frame buffer allocated to each frame
	int frameNum;
	int polyOrder;
	
	real_t *lpcCoefSrc;
	real_t *lpcCoefTar;	
	real_t *resTar;    // r_t, residual of target signal
	real_t *resSrc;    // \hat_r_t, residual of source signal
	
	// save [\hat_r_t - r_t]^2 in resSrc
	// save [\hat_r_t - r_t]   in resTar

	__host__ __device__ void operator() (const thrust::tuple<real_t &, int> &t) const
	{
	    
	    // which frame?
	    int frameIdx = t.get<1>() / frameBufLen;

	    // which sampling point in the frame
	    int posInFrame = t.get<1>() % frameBufLen;

	    if (posInFrame >= frameLength ||
		lpcCoefSrc[frameIdx * polyOrder * 2] < SIGPROCESS_LPC_0ORDER_THESHOLD ||
		lpcCoefTar[frameIdx * polyOrder * 2] < SIGPROCESS_LPC_0ORDER_THESHOLD){
		// if this time step is beyond the frame length
		// or if this frame is all 0 (dummy frame)
		resTar[t.get<1>()] = 0.0;
		resSrc[t.get<1>()] = 0.0;
	    }else{
		// [\hat_r_t - r_t]   in resTar
		resTar[t.get<1>()] =  resSrc[t.get<1>()] - resTar[t.get<1>()];
		// [\hat_r_t - r_t]^2   in resSrc
		resSrc[t.get<1>()] = (resTar[t.get<1>()] * resTar[t.get<1>()]);
	    }
	}
    };



    struct lpcResidualDifferenceGrad
    {
	int frameLength;   // length of one frame
	int frameBufLen;   // length of the frame buffer allocated to each frame
	int polyOrder;

	real_t *resDif;     // [\hat_r_t - r_t] * 2 * weight / framenum / framelength
	real_t *lpcCoefSrc;
	real_t *lpcCoefTar;
	
	// t.get<0>() \partial E / \partial \widehat_o_t, framed waveform
	// \partial E / \partial \widehat_o_t =
	//    \partial E / \partial \widehat_r_t + \sum_k=1^K a_k \partial E / \partial \widehat_r_t+k
	// Here we ignore the gradients through \partial a_k / \partial \widehat_o_t
	//  which is too complicated to calculate
	__host__ __device__ void operator() (const thrust::tuple<real_t &, int> &t) const
	{
	    
	    // which frame?
	    int frameIdx = t.get<1>() / frameBufLen;
	    
	    // which sampling point in the frame
	    int posInFrame = t.get<1>() % frameBufLen;
	    
	    if (posInFrame >= frameLength ||
		lpcCoefSrc[frameIdx * polyOrder * 2] < SIGPROCESS_LPC_0ORDER_THESHOLD ||
		lpcCoefTar[frameIdx * polyOrder * 2] < SIGPROCESS_LPC_0ORDER_THESHOLD){

		// if this time step is beyond the frame length
		// or if this frame is dummy frame (all 0.0)
		t.get<0>() = 0.0;
		
	    }else{
		real_t tmp_grad = resDif[t.get<1>()];
		for (int order = 1; order < polyOrder; order++){
		    if (posInFrame + order >= frameLength)
			break;
		    tmp_grad += lpcCoefSrc[frameIdx * polyOrder * 2 + order] *
			resDif[frameIdx * frameBufLen + posInFrame + order];
		}
		t.get<0>() = tmp_grad;
	    }
	}
    };



    struct lpcWaveformDifference
    {
	int frameLength;   // length of one frame
	int frameBufLen;   // length of the frame buffer allocated to each frame
	int frameNum;
	int polyOrder;
	
	real_t *lpcCoefSrc;
	real_t *lpcCoefTar;	
	real_t *resTar;    // r_t, residual of target signal
	real_t *resSrc;    // \hat_r_t, residual of source signal
	
	// save [\hat_r_t - r_t]^2 in resSrc
	// save [\hat_r_t - r_t]   in resTar

	__host__ __device__ void operator() (const thrust::tuple<real_t &, int> &t) const
	{
	    
	    // which frame?
	    int frameIdx = t.get<1>() / frameBufLen;

	    // which sampling point in the frame
	    int posInFrame = t.get<1>() % frameBufLen;

	    if (posInFrame >= frameLength ||
		lpcCoefSrc[frameIdx * polyOrder * 2] < SIGPROCESS_LPC_0ORDER_THESHOLD ||
		lpcCoefTar[frameIdx * polyOrder * 2] < SIGPROCESS_LPC_0ORDER_THESHOLD){
		// if this time step is beyond the frame length
		// or if this frame is all 0 (dummy frame)
		resTar[t.get<1>()] = 0.0;
		resSrc[t.get<1>()] = 0.0;
	    }else{
		// [\hat_r_t - r_t]   in resSrc
		resSrc[t.get<1>()] =  resSrc[t.get<1>()] - resTar[t.get<1>()];
		// [\hat_r_t - r_t]^2   in resTar
		resTar[t.get<1>()] = (resSrc[t.get<1>()] * resSrc[t.get<1>()]);
	    }
	}
    };


    struct lpcMseResExcitAndGrad
    {
	int frameLength;   // length of one frame
	int frameBufLen;   // length of the frame buffer allocated to each frame
	int frameNum;
	int polyOrder;
	
	real_t *lpcCoefTar;	
	real_t *resTar;    // r_t, residual of target signal
	real_t *resSrc;    // \hat_r_t, residual of source signal
	
	// save [\hat_r_t - r_t]^2 in resSrc
	// save [\hat_r_t - r_t]   in resTar

	__host__ __device__ void operator() (const thrust::tuple<real_t &, int> &t) const
	{
	    
	    // which frame?
	    int frameIdx = t.get<1>() / frameBufLen;

	    // which sampling point in the frame
	    int posInFrame = t.get<1>() % frameBufLen;

	    if (posInFrame >= frameLength ||
		lpcCoefTar[frameIdx * polyOrder * 2] < SIGPROCESS_LPC_0ORDER_THESHOLD){
		// if this time step is beyond the frame length
		// or if this frame is all 0 (dummy frame)
		resTar[t.get<1>()] = 0.0;
		resSrc[t.get<1>()] = 0.0;
	    }else{
		// [\hat_r_t - r_t]   in resSrc
		resSrc[t.get<1>()] =  resSrc[t.get<1>()] - resTar[t.get<1>()];
		// [\hat_r_t - r_t]^2   in resTar
		resTar[t.get<1>()] = (resSrc[t.get<1>()] * resSrc[t.get<1>()]);
	    }
	}
    };

    
    
}
}


namespace helpers {

    
    template <typename TDevice>
    lpcWarpper<TDevice>::lpcWarpper(real_vector *framedDataSrc,
				    real_vector *framedDataTar,
				    real_vector *autoCorrSrc,
				    real_vector *autoCorrTar,
				    real_vector *lpcCoefSrc,
				    real_vector *lpcCoefTar,				    
				    real_vector *lpcErrSrc,
				    real_vector *lpcErrTar,
				    real_vector *refCoefSrc,
				    real_vector *refCoefTar,
				    real_vector *lpcResSrc,
				    real_vector *lpcResTar,				    
				    real_vector *lpcGrad,
				    int lpcOrder,
				    int lpcErrorType,
				    int frameLength,
				    int frameShift,
				    int frameBufferLength, 
				    int frameNum,
				    int signalBufLength,
				    int signalLength)
    : m_framedDataSrc (framedDataSrc)
    , m_autoCorrSrc   (autoCorrSrc)
    , m_lpcCoefSrc    (lpcCoefSrc)
    , m_lpcResSrc     (lpcResSrc)
    , m_refCoefSrc    (refCoefSrc)
    , m_lpcErrSrc     (lpcErrSrc)
    , m_framedDataTar (framedDataTar)
    , m_autoCorrTar   (autoCorrTar)
    , m_lpcCoefTar    (lpcCoefTar)
    , m_lpcResTar     (lpcResTar)
    , m_refCoefTar    (refCoefTar)
    , m_lpcErrTar     (lpcErrTar)
    , m_lpcGrad       (lpcGrad)
    , m_lpcOrder      (lpcOrder)
    , m_lpcErrorType  (lpcErrorType)
    , m_frameLength   (frameLength)
    , m_frameShift    (frameShift)
    , m_frameNum      (frameNum)
    , m_frameBufLength (frameBufferLength)
    , m_signalBufLength (signalBufLength)
    , m_signalLength  (signalLength)
    {

	if (signalBufLength != this->m_lpcGrad->size())
	    throw std::runtime_error("Error: mismatch size of signalLength and lpcGrad size");

	if (m_signalBufLength < m_signalLength)
	    throw std::runtime_error("Error: signal Buff length < signal length");

	if ((m_frameNum * m_frameBufLength > m_framedDataSrc->size()) ||
	    (m_frameNum * m_frameBufLength > m_framedDataTar->size()))
	    throw std::runtime_error("Error: framed buffer size < frameNum * frameLength");

	m_polyOrder = m_lpcOrder + 1;
	if (m_polyOrder >= m_frameLength)
	    throw std::runtime_error("Error: LPC order > frame length");

	if ((m_frameNum * m_polyOrder > m_autoCorrSrc->size()) ||
	    (m_frameNum * m_polyOrder > m_autoCorrTar->size()))
	    throw std::runtime_error("Error: autoCorr buffer size is insufficient");

	if ((m_frameNum * m_polyOrder > m_refCoefSrc->size()) ||
	    (m_frameNum * m_polyOrder > m_refCoefTar->size()))
	    throw std::runtime_error("Error: reflection coef buffer size is insufficient");
	
	if ((m_frameNum * m_polyOrder * 2 > m_lpcCoefSrc->size()) ||
	    (m_frameNum * m_polyOrder * 2 > m_lpcCoefTar->size()))
	    throw std::runtime_error("Error: LPC coef buffer size is insufficient");
	
	if ((m_frameNum * m_polyOrder > m_lpcErrSrc->size()) ||
	    (m_frameNum * m_polyOrder > m_lpcErrTar->size()))
	    throw std::runtime_error("Error: LPC error buffer size is insufficient");

	if ((m_frameNum * m_frameLength > m_lpcResSrc->size()) ||
	    (m_frameNum * m_frameLength > m_lpcResTar->size()))
	    throw std::runtime_error("Error: LPC residual signal buffer size is insufficient");
	
	// clean the buffer cells that are beyond the length of current sentence
	thrust::fill((*m_lpcGrad).begin() + m_signalLength,
		     (*m_lpcGrad).end(), 0.0);
	
	
    }

    template <typename TDevice>
    lpcWarpper<TDevice>::~lpcWarpper()
    {
    }

    template <typename TDevice>
    void lpcWarpper<TDevice>::__autoCorr(real_vector *framedData, real_vector *autoCorr)
    {
	if (framedData == NULL || autoCorr == NULL)
	    throw std::runtime_error("framedData and autoCorr not initialized");

	{{
	    internal::lpcAutoCorr fn1;
	    fn1.frameLength = m_frameLength;
	    fn1.frameBufLen = m_frameBufLength;
	    fn1.frameData   = getRawPointer(*framedData);
	    fn1.polyOrder   = m_polyOrder;
	    
	    thrust::for_each(
		thrust::make_zip_iterator(
			thrust::make_tuple(
				(*autoCorr).begin(), 
				thrust::counting_iterator<int>(0))),
		thrust::make_zip_iterator(
			thrust::make_tuple(
				(*autoCorr).begin() + m_frameNum * m_polyOrder,
				thrust::counting_iterator<int>(0) + m_frameNum * m_polyOrder)),
		fn1);
	}}
    }

    template <typename TDevice>
    void lpcWarpper<TDevice>::__levinsonDurbinRecursion(real_vector *autoCorr,
							real_vector *lpcCoef,
							real_vector *refCoef,
							real_vector *lpcErr)
    {
	if (autoCorr == NULL || lpcCoef == NULL || refCoef == NULL || lpcErr == NULL)
	    throw std::runtime_error("autoCorr, lpcCoef, refCoef, lpcErr not initialized");

	thrust::fill((*lpcCoef).begin(), (*lpcCoef).end(), 0.0);
	thrust::fill((*refCoef).begin(), (*refCoef).end(), 0.0);
	thrust::fill((*lpcErr).begin(),  (*lpcErr).end(), 0.0);
	
	{{
	    internal::lpcLevinsonDurbinRecursion fn1;
	    fn1.polyOrder   = m_polyOrder;
	    
	    fn1.autoCorr   = getRawPointer(*autoCorr);
	    fn1.refCoef    = getRawPointer(*refCoef);
	    fn1.lpcErr     = getRawPointer(*lpcErr);
	    fn1.lpcCoef    = getRawPointer(*lpcCoef);
	    
	    thrust::for_each(
		thrust::make_zip_iterator(
			thrust::make_tuple(
				(*autoCorr).begin(), 
				thrust::counting_iterator<int>(0))),
		thrust::make_zip_iterator(
			thrust::make_tuple(
				(*autoCorr).begin() + m_frameNum,
				thrust::counting_iterator<int>(0) + m_frameNum)),
		fn1);
	}}
    }

    template <typename TDevice>
    void lpcWarpper<TDevice>::__lpcSynthesis(real_vector *lpcRes, real_vector *lpcCoef,
					     real_vector *framedData)
    {
	if (lpcCoef == NULL || lpcRes == NULL)
	    throw std::runtime_error("lpcCoef, lpcRes not initialized");

	
	{{
	    internal::lpcSynthesis fn1;
	    fn1.polyOrder   = m_polyOrder;
	    fn1.frameLength = m_frameLength;    
	    fn1.frameBufLen = m_frameBufLength;
	    
	    fn1.residualData= getRawPointer(*lpcRes);
	    fn1.lpcCoef     = getRawPointer(*lpcCoef);
	    if (framedData == NULL)
		fn1.waveData = NULL;
	    else
		fn1.waveData = getRawPointer(*framedData);
	    
	    int tmp_num_data = m_frameNum;
	    thrust::for_each(
		thrust::make_zip_iterator(
			thrust::make_tuple(
			    (*lpcRes).begin(), 
			    thrust::counting_iterator<int>(0))),
		thrust::make_zip_iterator(
			thrust::make_tuple(
			    (*lpcRes).begin() + tmp_num_data,
			    thrust::counting_iterator<int>(0) + tmp_num_data)),
		fn1);
	}}  

    }

    template <typename TDevice>
    void lpcWarpper<TDevice>::__lpcResidual(real_vector *framedData,
					    real_vector *lpcCoef,
					    real_vector *lpcRes)
    {
	if (framedData == NULL || lpcCoef == NULL || lpcRes == NULL)
	    throw std::runtime_error("framedData, lpcCoef, lpcRes not initialized");

	
	thrust::fill((*lpcRes).begin(),  (*lpcRes).end(), 0.0);	
	{{
	    internal::lpcResidual fn1;
	    fn1.polyOrder   = m_polyOrder;
	    fn1.frameLength = m_frameLength;    
	    fn1.frameBufLen = m_frameBufLength;
	    
	    fn1.framedData  = getRawPointer(*framedData);
	    fn1.lpcCoef     = getRawPointer(*lpcCoef);
	    
	    long int tmp_num_data = m_frameNum * m_frameBufLength;
	    thrust::for_each(
		thrust::make_zip_iterator(
			thrust::make_tuple(
			    (*lpcRes).begin(), 
			    thrust::counting_iterator<int>(0))),
		thrust::make_zip_iterator(
			thrust::make_tuple(
			    (*lpcRes).begin() + tmp_num_data,
			    thrust::counting_iterator<int>(0) + tmp_num_data)),
		fn1);
	}}
    }

    template <typename TDevice>
    real_t lpcWarpper<TDevice>::__lpcWaveformMseAndGrad(real_vector *framedDataSrc,
							real_vector *lpcResSrc,  real_vector *lpcResTar,
							real_vector *lpcCoefSrc, real_vector *lpcCoefTar)
    {
	if (framedDataSrc == NULL || lpcResTar == NULL)
	    throw std::runtime_error("framedDataSrc and lpcResTar not initialized");

	// total number of data points in all frames
	long int tmp_num_data = m_frameNum * m_frameBufLength;


	// get the LPC synthesized signal, using natural residual and
	// generated LPC coefficients
	// The output will be saved to lpcResTar
	this->__lpcSynthesis(lpcResTar, lpcCoefSrc, NULL);
	
	// copy the generated waveforms to lpcResSrc
	thrust::copy((*framedDataSrc).begin(), (*framedDataSrc).end(),
		     (*lpcResSrc).begin());

	// Note: we are using lpcResSrc and lpcResTar to store the
	// generated waveforms and LPC-synthesized waveforms.
	// Just for convenience
	
	// borrow the function 
	{{
	    internal::lpcWaveformDifference fn1;
	    fn1.frameLength = m_frameLength;    
	    fn1.frameBufLen = m_frameBufLength;
	    fn1.frameNum    = m_frameNum;
	    fn1.polyOrder   = m_polyOrder;

	    // lpcCoeff are used to judge whether this frame is dummy or not
	    fn1.lpcCoefSrc = getRawPointer(*lpcCoefSrc);
	    fn1.lpcCoefTar = getRawPointer(*lpcCoefTar);
	    fn1.resTar = getRawPointer(*lpcResTar);
	    fn1.resSrc = getRawPointer(*lpcResSrc);
	    
	    thrust::for_each(
		thrust::make_zip_iterator(
			thrust::make_tuple(
			    (*lpcResSrc).begin(), 
			    thrust::counting_iterator<int>(0))),
		thrust::make_zip_iterator(
			thrust::make_tuple(
			    (*lpcResSrc).begin() + tmp_num_data,
			    thrust::counting_iterator<int>(0) + tmp_num_data)),
		fn1);
	}}
	// [\hat_waveform_t - waveform_t]^2 is saved in lpcResTar
	// weight * 2 / NM * [\hat_waveform_t - waveform_t] is saved in lpcResSrc

	real_t distance = 0.0;
	{{
	   internal::scaleError fn2;
	   fn2.factor = 1.0;
	   distance = thrust::transform_reduce((*(lpcResTar)).begin(),
					       (*(lpcResTar)).begin() + tmp_num_data,
					       fn2,
					       (real_t)0,
					       thrust::plus<real_t>());
	   
	   // the MSE of residual signal can be quite small
	   // so, we do the averaging after sum
	   distance = distance / (m_frameNum * m_frameLength);
	}}

	// gradients have been to saved to lpcResSrc;
	return distance;
    }

    template <typename TDevice>
    real_t lpcWarpper<TDevice>::__lpcResidualMseAndGrad(real_vector *lpcResSrc, real_vector *lpcResTar,
							real_vector *lpcCoefSrc, real_vector *lpcCoefTar)
    {
	if (lpcResSrc == NULL || lpcResTar == NULL)
	    throw std::runtime_error("lpcResSrc and lpcResTar not initialized");

	// total number of data points in all frames
	long int tmp_num_data = m_frameNum * m_frameBufLength;
	
	{{
	    internal::lpcResidualDifference fn1;
	    fn1.frameLength = m_frameLength;    
	    fn1.frameBufLen = m_frameBufLength;
	    fn1.frameNum    = m_frameNum;
	    fn1.polyOrder   = m_polyOrder;
		
	    fn1.lpcCoefSrc = getRawPointer(*lpcCoefSrc);
	    fn1.lpcCoefTar = getRawPointer(*lpcCoefTar);
	    fn1.resTar = getRawPointer(*lpcResTar);
	    fn1.resSrc = getRawPointer(*lpcResSrc);
	    
	    thrust::for_each(
		thrust::make_zip_iterator(
			thrust::make_tuple(
			    (*lpcResSrc).begin(), 
			    thrust::counting_iterator<int>(0))),
		thrust::make_zip_iterator(
			thrust::make_tuple(
			    (*lpcResSrc).begin() + tmp_num_data,
			    thrust::counting_iterator<int>(0) + tmp_num_data)),
		fn1);
	}}
	// [\hat_residual_t - residual_t]^2 is saved in lpcResSrc
	// weight * 2 / NM * [\hat_residual_t - residual_t] is saved in lpcResTar

	real_t distance = 0.0;
	{{
	   internal::scaleError fn2;
	   fn2.factor = 1.0;
	   distance = thrust::transform_reduce((*(lpcResSrc)).begin(),
					       (*(lpcResSrc)).begin() + tmp_num_data,
					       fn2,
					       (real_t)0,
					       thrust::plus<real_t>());
	   
	   // the MSE of residual signal can be quite small
	   // so, we do the averaging after sum
	   distance = distance / (m_frameNum * m_frameLength);
	}}


	// calcualte the gradients w.r.t framed_signal
	// Results will be saved to lpcResSrc
	{{
	    internal::lpcResidualDifferenceGrad fn1;
	    fn1.frameLength = m_frameLength;    
	    fn1.frameBufLen = m_frameBufLength;
	    fn1.polyOrder   = m_polyOrder;
	    
	    fn1.resDif  = getRawPointer(*lpcResTar);
	    fn1.lpcCoefSrc = getRawPointer(*lpcCoefSrc);
	    fn1.lpcCoefTar = getRawPointer(*lpcCoefTar);
	    
	    thrust::for_each(
		thrust::make_zip_iterator(
			thrust::make_tuple(
			    (*lpcResSrc).begin(), 
			    thrust::counting_iterator<int>(0))),
		thrust::make_zip_iterator(
			thrust::make_tuple(
			    (*lpcResSrc).begin() + tmp_num_data,
			    thrust::counting_iterator<int>(0) + tmp_num_data)),
		fn1);
	}}
	
	return distance;
    }


    template <typename TDevice>
    real_t lpcWarpper<TDevice>::__lpcMseResExcitAndGrad(real_vector *framedDataSrc,
							real_vector *lpcResSrc,  real_vector *lpcResTar,
							real_vector *lpcCoefSrc, real_vector *lpcCoefTar)
    {
	if (framedDataSrc == NULL || lpcResTar == NULL)
	    throw std::runtime_error("framedDataSrc and lpcResTar not initialized");

	// total number of data points in all frames
	long int tmp_num_data = m_frameNum * m_frameBufLength;

	
	// copy the generated waveforms to lpcResSrc
	thrust::copy((*framedDataSrc).begin(), (*framedDataSrc).end(),
		     (*lpcResSrc).begin());

	// Note: we are using lpcResSrc and lpcResTar to store the
	// generated waveforms and LPC-synthesized waveforms.
	// Just for convenience
	
	// borrow the function 
	{{
	    internal::lpcMseResExcitAndGrad fn1;
	    fn1.frameLength = m_frameLength;    
	    fn1.frameBufLen = m_frameBufLength;
	    fn1.frameNum    = m_frameNum;
	    fn1.polyOrder   = m_polyOrder;

	    // lpcCoeff are used to judge whether this frame is dummy or not
	    fn1.lpcCoefTar = getRawPointer(*lpcCoefTar);
	    fn1.resTar = getRawPointer(*lpcResTar);
	    fn1.resSrc = getRawPointer(*lpcResSrc);
	    
	    thrust::for_each(
		thrust::make_zip_iterator(
			thrust::make_tuple(
			    (*lpcResSrc).begin(), 
			    thrust::counting_iterator<int>(0))),
		thrust::make_zip_iterator(
			thrust::make_tuple(
			    (*lpcResSrc).begin() + tmp_num_data,
			    thrust::counting_iterator<int>(0) + tmp_num_data)),
		fn1);
	}}
	// [\hat_waveform_t - waveform_t]^2 is saved in lpcResTar
	// weight * 2 / NM * [\hat_waveform_t - waveform_t] is saved in lpcResSrc

	real_t distance = 0.0;
	{{
	   internal::scaleError fn2;
	   fn2.factor = 1.0;
	   distance = thrust::transform_reduce((*(lpcResTar)).begin(),
					       (*(lpcResTar)).begin() + tmp_num_data,
					       fn2,
					       (real_t)0,
					       thrust::plus<real_t>());
	   
	   // the MSE of residual signal can be quite small
	   // so, we do the averaging after sum
	   distance = distance / (m_frameNum * m_frameLength);
	}}

	// gradients have been to saved to lpcResSrc;
	return distance;
    }

    template <typename TDevice>
    void lpcWarpper<TDevice>::lpcAnalysisTargetSignal()
    {
	// auto-correlation
	this->__autoCorr(m_framedDataTar, m_autoCorrTar);
	
	// Levinson-durbin
	this->__levinsonDurbinRecursion(m_autoCorrTar, m_lpcCoefTar,
					m_refCoefTar, m_lpcErrTar);

	// LPC residual 
	this->__lpcResidual(m_framedDataTar, m_lpcCoefTar, m_lpcResTar);	
    }

    
    template <typename TDevice>
    void lpcWarpper<TDevice>::lpcAnalysisSourceSignal()
    {
	// autoCorr
	this->__autoCorr(m_framedDataSrc, m_autoCorrSrc);
		    
	// Levison-durbin recursion
	this->__levinsonDurbinRecursion(m_autoCorrSrc, m_lpcCoefSrc,
					m_refCoefSrc, m_lpcErrSrc);
	// Calculate residual signals
	this->__lpcResidual(m_framedDataSrc, m_lpcCoefSrc, m_lpcResSrc);
    }

    
    template <typename TDevice>
    real_t lpcWarpper<TDevice>::lpcError()
    {
	real_t lpcError = 0.0;
	
	if (m_lpcErrorType == SIGPROCESS_LPC_ERR_TYPE_RES_MSE){
	    
	    
	    // Calculate residual MSE
	    // Calculate gradients and save to m_lpcResSrc
	    // Note, m_lpcResSrc mus be de-framed
	    lpcError = this->__lpcResidualMseAndGrad(m_lpcResSrc, m_lpcResTar,
						     m_lpcCoefSrc, m_lpcCoefTar);	    

	    // The gradients have been saved to m_lpcResSrc
	    // The gradients will be de-framed and saved to m_lpcGrad in lpcGradCollect()
	    
	}else if (m_lpcErrorType == SIGPROCESS_LPC_ERR_TYPE_WAV_MSE){
	    
	    lpcError = this->__lpcWaveformMseAndGrad(m_framedDataSrc,
						     m_lpcResSrc, m_lpcResTar,
						     m_lpcCoefSrc, m_lpcCoefTar);	    

	}else if (m_lpcErrorType == SIGPROCESS_LPC_ERR_TYPE_EXCIT_LOSS){

	    lpcError = this->__lpcMseResExcitAndGrad(m_framedDataSrc,
						     m_lpcResSrc, m_lpcResTar,
						     m_lpcCoefSrc, m_lpcCoefTar);	    

	    
	}else{
	    throw std::runtime_error("Error: lpcErrorType undefined");
	}
	
	return lpcError;
    }

    template <typename TDevice>
    void lpcWarpper<TDevice>::lpcGradCollect(const real_t weight)
    {
	int maxWaveLength = m_signalBufLength;
	int maxFrameNum = fftTools::fftFrameNum(maxWaveLength,m_frameLength,m_frameShift);
	
	// borrow the function from FFTMat for de-framing
	helpers::FFTMat<TDevice> tmpStructure(
		m_lpcGrad, m_lpcResSrc, NULL,
		m_frameLength, m_frameShift, FFTMAT_WINDOW_HANN, m_frameBufLength,
		fftTools::fftBinsNum(m_frameBufLength), maxFrameNum, maxWaveLength,
		m_signalLength, FFTMAT_SPECTYPE_AMP_LOG_MSE);

	tmpStructure.collectGrad(weight);
    }
    
    template class lpcWarpper<Cpu>;
    template class lpcWarpper<Gpu>;
}
