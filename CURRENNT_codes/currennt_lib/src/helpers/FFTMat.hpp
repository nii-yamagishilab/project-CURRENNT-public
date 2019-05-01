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

#ifndef HELPERS_FFTMAT_HPP
#define HELPERS_FFTMAT_HPP

#include "../Types.hpp"


#define FFTMAT_SPECTYPE_MSE 0
#define FFTMAT_SPECTYPE_KLD 1

#define FFTMAT_PHASETYPE_COS 0
#define FFTMAT_PHASETYPE_INTANT 1

#define FFTMAT_WINDOW_HANN 0
#define FFTMAT_WINDOW_SQUARE 1

namespace helpers {

    template <typename TDevice>
    class FFTMat
    {
	typedef typename TDevice::complex_vector complex_vector;
	typedef typename TDevice::real_vector real_vector;
	
    private:
	real_vector    *m_rawData;
	real_vector    *m_framedData;
	complex_vector *m_fftData;

	int m_frameLength;
	int m_frameShift;
	int m_windowType;
	int m_fftLength;
	
	int m_fftBins;          // dimenions of the FFT feature vector
	int m_batchSize;        // frame number

	int m_signalBufLength;  // maxSeqLength
	int m_signalLength;     // curMaxSeqLength

	int m_disType;      //
	int m_validFrameNum;    //
	int m_validDataPointNum;//
	
    public:
	FFTMat(real_vector *rawData, real_vector *framedData, complex_vector *fftData,
	       int frameLength, int frameShift, int windowType,
	       int fftLength,   int fftBins,  int batchSize,
	       int signalBufLength,
	       int signsignalLength,
	       int disType);

	FFTMat(real_vector *framedDAta, complex_vector *fftData,
	       int fftLength, int fftBins, int batchSize);
	~FFTMat();

	void FFT();
	void iFFT();
	void frameSignal();
	void collectGrad(real_t gradScaleFactor);

	void specAmpGrad(FFTMat<TDevice> &source, FFTMat<TDevice> &target);
	real_t specAmpDistance(FFTMat<TDevice> &target, FFTMat<TDevice> &diff);

	void specPhaseGrad(FFTMat<TDevice> &source, FFTMat<TDevice> &target);
	real_t specPhaseDistance(FFTMat<TDevice> &target, FFTMat<TDevice> &diff);

	void changeWindowType(const int windowType);
    };


    namespace fftTools{
	// 
	int fftFrameNum(int signalLength, int frameLength, int frameShift);
	
	// 
	int fftBinsNum(int fftLength);
    }
    
}

#endif
