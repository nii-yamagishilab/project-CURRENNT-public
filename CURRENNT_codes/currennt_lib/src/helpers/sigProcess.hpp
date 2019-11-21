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

#ifndef HELPERS_SIGPROCESS_HPP
#define HELPERS_SIGPROCESS_HPP

#include "../Types.hpp"

#define SIGPROCESS_LPC_ERR_TYPE_WAV_MSE 0       // waveform MSE (o_{1:T} - iLPC(LPC(\hat{o}_{1:T})))^2
#define SIGPROCESS_LPC_ERR_TYPE_RES_MSE 1       // residual MSE (LPC(o_{1:T}) - LPC(\hat{o}_{1:T}))^2
#define SIGPROCESS_LPC_ERR_TYPE_EXCIT_LOSS 2    // loss between excitation and natural-waveform residual
#define SIGPROCESS_LPC_ERR_TYPE_EXCIT_FFTLOSS 3 // FFT loss between excitation and natural-waveform residual

namespace helpers {

    // Warpper of signal processing methods
    
    template <typename TDevice>
    class lpcWarpper
    {
	typedef typename TDevice::complex_vector complex_vector;
	typedef typename TDevice::real_vector real_vector;
	
    private:
	real_vector    *m_framedDataSrc;
	real_vector    *m_autoCorrSrc;
	real_vector    *m_lpcCoefSrc;
	real_vector    *m_lpcResSrc;
	real_vector    *m_refCoefSrc;
	real_vector    *m_lpcErrSrc;
	
	real_vector    *m_framedDataTar;
	real_vector    *m_autoCorrTar;
	real_vector    *m_lpcCoefTar;
	real_vector    *m_lpcResTar;
	real_vector    *m_refCoefTar;
	real_vector    *m_lpcErrTar;

	real_vector    *m_lpcGrad;

	int m_lpcErrorType;
	int m_frameLength;
	int m_frameShift;
	
	int m_frameNum;          // valid number of frame
	int m_frameBufLength; // length of the memory allocated to each frame
	int m_signalBufLength;  // maxSeqLength, maximum length of the buffer
	int m_signalLength;     // curMaxSeqLength, length of the current utterance

	int m_lpcOrder;
	int m_polyOrder;  // m_polyOrder = m_lpcOrder + 1
	
    public:
	lpcWarpper(real_vector *framedDataSrc,
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
		   int frameLength, int frameShift,
		   int frameBufferLength,
		   int frameNum,
		   int signalBufLength,
		   int signalLength);

	~lpcWarpper();

	// calculate auto-correlation coefficients
	void __autoCorr(real_vector *framedData, real_vector *autoCorr);

	// LPC analysis using Levinson-Durbin
	void __levinsonDurbinRecursion(real_vector *autoCorr, real_vector *lpcCoef,
				       real_vector *refCoef, real_vector *lpcErr);

	// calculate residual signal, given LPC coefficients
	void __lpcResidual(real_vector *framedData, real_vector *lpcCoef,
			   real_vector *lpcRes);
	
	void __lpcSynthesis(real_vector *lpcRes, real_vector *lpcCoef,
			    real_vector *framedData);
	
	// calculate residual MSE and gradients
	real_t __lpcResidualMseAndGrad(real_vector *lpcResSrc, real_vector *lpcResTar,
				       real_vector *lpcCoefSrc, real_vector *lpcCoefTar);

	// calculate waveform MSE and gradients
	real_t __lpcWaveformMseAndGrad(real_vector *framedDataSrc,
				       real_vector *lpcResSrc, real_vector *lpcResTar,
				       real_vector *lpcCoefSrc, real_vector *lpcCoefTar);

	// calculate MSE between natural residual and explicit excitation
	real_t __lpcMseResExcitAndGrad(real_vector *framedDataSrc,
				       real_vector *lpcResSrc, real_vector *lpcResTar,
				       real_vector *lpcCoefSrc, real_vector *lpcCoefTar);

	// warpper functions
	void lpcAnalysisSourceSignal();
	void lpcAnalysisTargetSignal();	
	real_t lpcError();
	void lpcGradCollect(const real_t weight);
    };

    
}

#endif
