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

#include <stdexcept>
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/random.h>
#include <thrust/transform_reduce.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>

// Used to adjust the amplitude of waveform into [-128, 128]
#define SIGPROCESS_WAVEFORM_LEVEL 128.0

namespace internal{
namespace {

    struct lpcAutoCorr
    {
	int polyOrder;
	int frameLength;   // length of one frame
	int frameBufLen;   // length of the frame buffer allocated to each frame
	real_t *frameData;

	__host__ __device__ void operator() (const thrust::tuple<real_t &, int> &t) const
	{
	    int frameIdx = t.get<1>() / polyOrder;
	    int shiftValue = t.get<1>() % polyOrder;
	    
	    real_t tmp_sum = 0.0;
	    
	    for (int index = 0; index < frameLength; index++){
		if (index + shiftValue >= frameLength)
		    break;
		tmp_sum += (frameData[frameIdx * frameBufLen + index] * SIGPROCESS_WAVEFORM_LEVEL
			    * frameData[frameIdx * frameBufLen + index + shiftValue] *
			    (SIGPROCESS_WAVEFORM_LEVEL / frameLength));
	    }
	    t.get<0>() = tmp_sum;
	}
    };

    struct lpcLevinsonDurbinRecursion
    {
	int polyOrder;
	int frameLength;
	
	real_t *autoCorr;
	real_t *refCoef;
	real_t *lpcErr;
	real_t *lpcCoef;

	// for 1 to N frames
	__host__ __device__ void operator() (const thrust::tuple<real_t &, int> &t) const
	{
	    int frameIdx = t.get<1>();

	    // pointer that points to autoCorr, refCoef, lpcErr of current frame
	    long int dataPtr = frameIdx * polyOrder;
	    // pointer that points to lpcCoef of current frame
	    long int lpcCoefPtr = frameIdx * polyOrder * 2;

	    // reflection coef
	    real_t tmp_refCoef = 0.0;
	    
	    // initialization
	    lpcErr[dataPtr] = autoCorr[dataPtr + 0];  // 0-order LPC, error = auto-corr[0]
	    lpcCoef[lpcCoefPtr] = 1.0; // 0-order LPC, only coefficient 1

	    // lpcCoef[lpcCoefPtr] ->   [a_p^p, a_p-1^p, ... 1.0, 0, 0]
	    // lpcCoef[lpcCoefPtr + polyOrder] -> [a_p+1^p+1, a_p^p+1, ... 1.0, 0]

	    // from order 1 to order polyOrder - 1 = lpcOrder
	    for (int order = 1; order < polyOrder; order ++){
		// set 1.0 as the 1st LPC coefficient
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

		// for higher order
		if (order > 1){
		    for (int order2 = 1; order2 < order; order2++)
			lpcCoef[lpcCoefPtr + polyOrder + order2] =
			    lpcCoef[lpcCoefPtr + order2 - 1] -
			    tmp_refCoef * lpcCoef[lpcCoefPtr + order - 1 - order2];
		}
		
		// lpc error
		lpcErr[dataPtr + order] = lpcErr[dataPtr + order - 1] *
		    (1 - tmp_refCoef * tmp_refCoef);

		// copy the lpc coeff (including the 1 of 0-order LPC)
		for (int order2 = 0; order2 <= order; order2++)
		    lpcCoef[lpcCoefPtr + order2] = lpcCoef[lpcCoefPtr + polyOrder + order2];
	    }

	    // Reverse the order of LPC coef
	    for (int order = 0; order < polyOrder; order++)
		lpcCoef[lpcCoefPtr + order] =
		    lpcCoef[lpcCoefPtr + polyOrder + polyOrder - 1 - order];
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

	if ((m_frameNum * m_frameBufLength >= m_framedDataSrc->size()) ||
	    (m_frameNum * m_frameBufLength >= m_framedDataTar->size()))
	    throw std::runtime_error("Error: framed buffer size < frameNum * frameLength");

	m_polyOrder = m_lpcOrder + 1;
	if (m_polyOrder >= m_frameLength)
	    throw std::runtime_error("Error: LPC order > frame length");

	if ((m_frameNum * m_polyOrder >= m_autoCorrSrc->size()) ||
	    (m_frameNum * m_polyOrder >= m_autoCorrTar->size()))
	    throw std::runtime_error("Error: autoCorr buffer size is insufficient");

	if ((m_frameNum * m_polyOrder >= m_refCoefSrc->size()) ||
	    (m_frameNum * m_polyOrder >= m_refCoefTar->size()))
	    throw std::runtime_error("Error: reflection coef buffer size is insufficient");
	
	if ((m_frameNum * m_polyOrder * 2 >= m_lpcCoefSrc->size()) ||
	    (m_frameNum * m_polyOrder * 2 >= m_lpcCoefTar->size()))
	    throw std::runtime_error("Error: LPC coef buffer size is insufficient");
	
	if ((m_frameNum * m_polyOrder >= m_lpcErrSrc->size()) ||
	    (m_frameNum * m_polyOrder >= m_lpcErrTar->size()))
	    throw std::runtime_error("Error: LPC error buffer size is insufficient");

	if ((m_frameNum * m_frameLength >= m_lpcResSrc->size()) ||
	    (m_frameNum * m_frameLength >= m_lpcResTar->size()))
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
	    fn1.frameLength = m_frameLength;
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
    void lpcWarpper<TDevice>::lpcAnalysis()
    {
	// autoCorr
	this->__autoCorr(m_framedDataSrc, m_autoCorrSrc);
	this->__autoCorr(m_framedDataTar, m_autoCorrTar);

	// Levison-durbin recursion
	this->__levinsonDurbinRecursion(m_autoCorrSrc, m_lpcCoefSrc,
					m_refCoefSrc, m_lpcErrSrc);
	this->__levinsonDurbinRecursion(m_autoCorrTar, m_lpcCoefTar,
					m_refCoefTar, m_lpcErrTar);
    }

    template <typename TDevice>
    real_t lpcWarpper<TDevice>::lpcError()
    {
	return 0.0;
    }

    template <typename TDevice>
    void lpcWarpper<TDevice>::lpcGradCollect()
    {

    }
    
    template class lpcWarpper<Cpu>;
    template class lpcWarpper<Gpu>;
}
