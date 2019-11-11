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

#include "FFTMat.hpp"
#include "getRawPointer.cuh"
#include "safeExp.cuh"
#include <cufft.h>

#include <stdexcept>
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/random.h>
#include <thrust/transform_reduce.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>

typedef cufftHandle *cufftHandle_t;

#define FFT_PI_DEFINITION 3.141215
//#define FFT_KLD_FLOOR_NUM 0.0000000001
#define FFT_KLD_FLOOR_NUM 0.00001
#define FFT_IS_FLOOR_NUM 0.00001
#define FFT_AMP_MIN_NUM   0.0000001
#define FFT_REAL_SPECTRUM_SHIFT 0.00001
#define FFT_SPEC_LOG_MSE_CONST 1.0

namespace internal{
namespace {

    // cuFFT plan Handle
    //  currently, only allows 3 different FFT plans to be used
    //  this part should be modified
    
    cufftHandle_t getCuFFTHandle_fft(int fftSize, int batchSize)
    {
	static cufftHandle cufftHandle_fft = 0;
	static int fftSize_local = 0;
	static int batchSize_local = 0;
	
	static cufftHandle cufftHandle_fft2 = 0;
	static int fftSize_local2 = 0;
	static int batchSize_local2 = 0;

	static cufftHandle cufftHandle_fft3 = 0;
	static int fftSize_local3 = 0;
	static int batchSize_local3 = 0;

	static cufftHandle cufftHandle_fft4 = 0;
	static int fftSize_local4 = 0;
	static int batchSize_local4 = 0;
	
	static cufftHandle cufftHandle_fft5 = 0;
	static int fftSize_local5 = 0;
	static int batchSize_local5 = 0;

	static cufftHandle cufftHandle_fft6 = 0;
	static int fftSize_local6 = 0;
	static int batchSize_local6 = 0;

	
	if (!cufftHandle_fft){
	    fftSize_local = fftSize;
	    batchSize_local = batchSize;
	    cufftResult res = cufftPlan1d(&cufftHandle_fft, fftSize_local,
					  CUFFT_R2C, batchSize_local);
	    if (res != CUFFT_SUCCESS)
		throw std::runtime_error("Error: could not create cufft plan");
	    return &cufftHandle_fft;
	    
	}else if (fftSize_local == fftSize && batchSize_local == batchSize){
	    return &cufftHandle_fft;
	    
	}else if (!cufftHandle_fft2){
	    fftSize_local2 = fftSize;
	    batchSize_local2 = batchSize;
	    cufftResult res = cufftPlan1d(&cufftHandle_fft2, fftSize_local2,
					  CUFFT_R2C, batchSize_local2);
	    if (res != CUFFT_SUCCESS)
		throw std::runtime_error("Error: could not create cufft plan");
	    return &cufftHandle_fft2;
	    
	}else if (fftSize_local2 == fftSize && batchSize_local2 == batchSize){
	    return &cufftHandle_fft2;
	    
	}else if (!cufftHandle_fft3){
	    fftSize_local3 = fftSize;
	    batchSize_local3 = batchSize;
	    cufftResult res = cufftPlan1d(&cufftHandle_fft3, fftSize_local3,
					  CUFFT_R2C, batchSize_local3);
	    if (res != CUFFT_SUCCESS)
		throw std::runtime_error("Error: could not create cufft plan");
	    return &cufftHandle_fft3;
	    
	}else if (fftSize_local3 == fftSize && batchSize_local3 == batchSize){
	    return &cufftHandle_fft3;
	    
	}else if (!cufftHandle_fft4){
	    fftSize_local4 = fftSize;
	    batchSize_local4 = batchSize;
	    cufftResult res = cufftPlan1d(&cufftHandle_fft4, fftSize_local4,
					  CUFFT_R2C, batchSize_local4);
	    if (res != CUFFT_SUCCESS)
		throw std::runtime_error("Error: could not create cufft plan");
	    return &cufftHandle_fft4;
	    
	}else if (fftSize_local4 == fftSize && batchSize_local4 == batchSize){
	    return &cufftHandle_fft4;
	    
	}else if (!cufftHandle_fft5){
	    fftSize_local5 = fftSize;
	    batchSize_local5 = batchSize;
	    cufftResult res = cufftPlan1d(&cufftHandle_fft5, fftSize_local5,
					  CUFFT_R2C, batchSize_local5);
	    if (res != CUFFT_SUCCESS)
		throw std::runtime_error("Error: could not create cufft plan");
	    return &cufftHandle_fft5;
	    
	}else if (fftSize_local5 == fftSize && batchSize_local5 == batchSize){
	    return &cufftHandle_fft5;
	    
	}else if (!cufftHandle_fft6){
	    fftSize_local6 = fftSize;
	    batchSize_local6 = batchSize;
	    cufftResult res = cufftPlan1d(&cufftHandle_fft6, fftSize_local6,
					  CUFFT_R2C, batchSize_local6);
	    if (res != CUFFT_SUCCESS)
		throw std::runtime_error("Error: could not create cufft plan");
	    return &cufftHandle_fft6;
	    
	}else if (fftSize_local6 == fftSize && batchSize_local6 == batchSize){
	    return &cufftHandle_fft6;
	    
	}else{
	    throw std::runtime_error("Impossible");
	}
	
	/*
	if (!cufftHandle_fft || fftSize_local != fftSize || batchSize_local != batchSize){
	    fftSize_local = fftSize;
	    batchSize_local = batchSize;
	    cufftResult res = cufftPlan1d(&cufftHandle_fft, fftSize_local,
					  CUFFT_R2C, batchSize_local);
	    if (res != CUFFT_SUCCESS)
		throw std::runtime_error("Error: could not create cufft plan");
	    
		}*/
	    
	
    }

    cufftHandle_t getCuFFTHandle_ifft(int fftSize, int batchSize)
    {
	static cufftHandle cufftHandle_ifft = 0;
	static int fftSize_local = 0;
	static int batchSize_local = 0;
	
	static cufftHandle cufftHandle_ifft2 = 0;
	static int fftSize_local2 = 0;
	static int batchSize_local2 = 0;

	static cufftHandle cufftHandle_ifft3 = 0;
	static int fftSize_local3 = 0;
	static int batchSize_local3 = 0;

	static cufftHandle cufftHandle_ifft4 = 0;
	static int fftSize_local4 = 0;
	static int batchSize_local4 = 0;
	
	static cufftHandle cufftHandle_ifft5 = 0;
	static int fftSize_local5 = 0;
	static int batchSize_local5 = 0;

	static cufftHandle cufftHandle_ifft6 = 0;
	static int fftSize_local6 = 0;
	static int batchSize_local6 = 0;

	
	if (!cufftHandle_ifft){
	    fftSize_local = fftSize;
	    batchSize_local = batchSize;
	    cufftResult res = cufftPlan1d(&cufftHandle_ifft, fftSize_local,
					  CUFFT_C2R, batchSize_local);
	    if (res != CUFFT_SUCCESS)
		throw std::runtime_error("Error: could not create cufft plan");
	    return &cufftHandle_ifft;
	    
	}else if (fftSize_local == fftSize && batchSize_local == batchSize){
	    return &cufftHandle_ifft;
	    
	}else if (!cufftHandle_ifft2){
	    fftSize_local2 = fftSize;
	    batchSize_local2 = batchSize;
	    cufftResult res = cufftPlan1d(&cufftHandle_ifft2, fftSize_local2,
					  CUFFT_C2R, batchSize_local2);
	    if (res != CUFFT_SUCCESS)
		throw std::runtime_error("Error: could not create cufft plan");
	    return &cufftHandle_ifft2;
	    
	}else if (fftSize_local2 == fftSize && batchSize_local2 == batchSize){
	    return &cufftHandle_ifft2;
	    
	}else if (!cufftHandle_ifft3){
	    fftSize_local3 = fftSize;
	    batchSize_local3 = batchSize;
	    cufftResult res = cufftPlan1d(&cufftHandle_ifft3, fftSize_local3,
					  CUFFT_C2R, batchSize_local3);
	    if (res != CUFFT_SUCCESS)
		throw std::runtime_error("Error: could not create cufft plan");
	    return &cufftHandle_ifft3;
	    
	}else if (fftSize_local3 == fftSize && batchSize_local3 == batchSize){
	    return &cufftHandle_ifft3;
	    
	}else if (!cufftHandle_ifft4){
	    fftSize_local4 = fftSize;
	    batchSize_local4 = batchSize;
	    cufftResult res = cufftPlan1d(&cufftHandle_ifft4, fftSize_local4,
					  CUFFT_C2R, batchSize_local4);
	    if (res != CUFFT_SUCCESS)
		throw std::runtime_error("Error: could not create cufft plan");
	    return &cufftHandle_ifft4;
	    
	}else if (fftSize_local4 == fftSize && batchSize_local4 == batchSize){
	    return &cufftHandle_ifft4;
	    
	}else if (!cufftHandle_ifft5){
	    fftSize_local5 = fftSize;
	    batchSize_local5 = batchSize;
	    cufftResult res = cufftPlan1d(&cufftHandle_ifft5, fftSize_local5,
					  CUFFT_C2R, batchSize_local5);
	    if (res != CUFFT_SUCCESS)
		throw std::runtime_error("Error: could not create cufft plan");
	    return &cufftHandle_ifft5;
	    
	}else if (fftSize_local5 == fftSize && batchSize_local5 == batchSize){
	    return &cufftHandle_ifft5;
	    
	}else if (!cufftHandle_ifft6){
	    fftSize_local6 = fftSize;
	    batchSize_local6 = batchSize;
	    cufftResult res = cufftPlan1d(&cufftHandle_ifft6, fftSize_local6,
					  CUFFT_C2R, batchSize_local6);
	    if (res != CUFFT_SUCCESS)
		throw std::runtime_error("Error: could not create cufft plan");
	    return &cufftHandle_ifft6;
	    
	}else if (fftSize_local6 == fftSize && batchSize_local6 == batchSize){
	    return &cufftHandle_ifft6;
	    
	}else{
	    throw std::runtime_error("Impossible");
	}
	

    }

    struct FrameSignal
    {
	int fftLength;   // dimension of frame (padded)
	int frameLength;
	int frameShift;
	int windowType;  // now only implemented the hann window
	
	real_t *rawData;

	__host__ __device__ void operator() (const thrust::tuple<real_t &, int> &t) const
	{
	    int frameIdx = t.get<1>() / fftLength;
	    int framePos = t.get<1>() % fftLength;

	    if (framePos < frameLength){
		// real data in each frame

		// Hann window by default
		if (windowType == FFTMAT_WINDOW_SQUARE)
		    t.get<0>() = rawData[frameIdx * frameShift + framePos];
		else
		    t.get<0>() = rawData[frameIdx * frameShift + framePos] *
			0.5 * (1 - cos(2.0 * FFT_PI_DEFINITION * framePos / (frameLength - 1)));
	    }else{
		// Since FFT length should be >= signal length (frame length)
		//  we need zero paddings after the frame length
		t.get<0>() = 0.0;
	    }
	    
	}
    };


    struct FrameMakeEvenSignal
    {
	int fftLength;   // dimension of frame (padded)
	int frameLength;
	int frameShift;
	int windowType;  // now only implemented the hann window
	
	real_t *rawData;

	__host__ __device__ void operator() (const thrust::tuple<real_t &, int> &t) const
	{
	    int frameIdx = t.get<1>() / fftLength;
	    int framePos = t.get<1>() % fftLength;

	    if (framePos < frameLength){
		// real data in each frame

		// Hann window by default
		if (windowType == FFTMAT_WINDOW_SQUARE)
		    t.get<0>() = rawData[frameIdx * frameShift + framePos];
		else
		    t.get<0>() = rawData[frameIdx * frameShift + framePos] *
			0.5 * (1 - cos(2.0 * FFT_PI_DEFINITION * framePos / (frameLength - 1)));

		if (framePos > 0)
		    t.get<0>() = t.get<0>() * 0.5;
		
	    }else if (framePos < fftLength && framePos > (fftLength - frameLength)){

		// mirror the signal
		framePos = fftLength - framePos;
		if (windowType == FFTMAT_WINDOW_SQUARE)
		    t.get<0>() = rawData[frameIdx * frameShift + framePos];
		else
		    t.get<0>() = rawData[frameIdx * frameShift + framePos] *
			0.5 * (1 - cos(2.0 * FFT_PI_DEFINITION * framePos / (frameLength - 1)));
		
		t.get<0>() = t.get<0>() * 0.5;
		
	    }else{
		t.get<0>() = 0.0;
	    }
	    
	}
    };


    struct FrameDeltaShift
    {
	int fftLength;   // dimension of frame (padded)	
	real_t *frameData;

	// for 1 to N frames
	__host__ __device__ void operator() (const thrust::tuple<real_t &, int> &t) const
	{
	    int frameIdx = t.get<1>();
	    real_t tmp_sum = 0.0;

	    // force x[0] = 0 (assume the left side of the analysis window is 0)
	    frameData[frameIdx * fftLength] = 0.0;
	    
	    // sum_n=1^M abs(x[n])
	    for (int frame_idx = 0; frame_idx < fftLength; frame_idx ++)
		tmp_sum += abs(frameData[frameIdx * fftLength + frame_idx]);
	    
	    // shift the x[0] + const
	    frameData[frameIdx * fftLength] += (tmp_sum + FFT_REAL_SPECTRUM_SHIFT);
	}
    };
    

    
    struct GradFromMirrorToOriginal
    {
	int fftLength;   // dimension of frame (padded)
	int frameLength;
	int frameShift;
	int windowType;  // now only implemented the hann window
	
	real_t *rawGradData;

	__host__ __device__ void operator() (const thrust::tuple<real_t &, int> &t) const
	{
	    int frameIdx = t.get<1>() / fftLength;
	    int framePos = t.get<1>() % fftLength;

	    if (framePos > 0 && framePos < frameLength){

		// dimension of the mirrored part
		int mirFramePos = fftLength - framePos;

		t.get<0>() = (t.get<0>() + rawGradData[frameIdx * fftLength + mirFramePos]) * 0.5;
		
	    }else{
		// do nothing for no-mirrored part
		// note that the mirror gradients will be cleaned later by cleanDummyGradients
	    }
	    
	}
    };


    struct GradFromDeltaShiftedSignal_nonZeroPosition
    {
	int fftLength;   // dimension of frame (padded)
	real_t *framedData;
	real_t *rawGradData;

	__host__ __device__ void operator() (const thrust::tuple<real_t &, int> &t) const
	{
	    int frameIdx = t.get<1>() / fftLength;
	    int framePos = t.get<1>() % fftLength;

	    if (framePos > 0){
		// for x[n>0], the gradients depends on the sign
		if (framedData[t.get<1>()] > 0.0){
		    t.get<0>() = t.get<0>() + rawGradData[frameIdx * fftLength];
		}else if (framedData[t.get<1>()] < 0.0){
		    t.get<0>() = t.get<0>() - rawGradData[frameIdx * fftLength];
		}else{
		    // do nothing
		}
	    }
	}
    };

    struct GradFromDeltaShiftedSignal_zeroPosition
    {
	int fftLength;   // dimension of frame (padded)
	real_t *rawGradData;

	__host__ __device__ void operator() (const thrust::tuple<real_t &, int> &t) const
	{
	    int frameIdx = t.get<1>();
	    // set the gradients for x[0] = 0 because x[0] is set to 0
	    rawGradData[frameIdx * fftLength] = 0.0;
	}
    };


    
    struct cleanDummyGradients
    {
	int fftLength;   // dimension of frame (padded)
	int frameLength;
	int frameShift;
	int validFrameNum;

	__host__ __device__ void operator() (const thrust::tuple<real_t &, int> &t) const
	{
	    int framePos = t.get<1>() % fftLength;
	    int frameIdx = t.get<1>() / fftLength;
	    
	    if (framePos > frameLength || frameIdx >= validFrameNum)
		t.get<0>() = 0.0;
	}
    };

    struct SpecAmpDistance_AMP_LOG_MSE
    {
	__host__ __device__ void operator() (const thrust::tuple<complex_t &,
					     complex_t &, complex_t &> &t) const
	{
	    // t.get<0>() source
	    // t.get<1>() target
	    // t.get<2>() buffer to store diff

	    // To save memory space, t.get<2>().x to store diff^2
	    // t.get<2>().y = [ log(Re(src)^2 + Im(src)^2) - log(Re(tar)^2 + Im(tar)^2) ]
	    t.get<2>().y = (helpers::safeLog(t.get<0>().x * t.get<0>().x +
					     t.get<0>().y * t.get<0>().y +
					     FFT_KLD_FLOOR_NUM) -
			    helpers::safeLog(t.get<1>().x * t.get<1>().x +
					     t.get<1>().y * t.get<1>().y +
					     FFT_KLD_FLOOR_NUM));
	    // t.get<2>().x = [ log(Re(src)^2 + Im(src)^2) - log(Re(tar)^2 + Im(tar)^2) ] ^ 2
	    t.get<2>().x = t.get<2>().y * t.get<2>().y;

	}
    };

    struct SpecAmpDistance_AMP_RATIO
    {
	__host__ __device__ void operator() (const thrust::tuple<complex_t &,
					     complex_t &, complex_t &> &t) const
	{
	    // t.get<0>() source
	    // t.get<1>() target
	    // t.get<2>() buffer to store diff

	    // To save memory space, t.get<2>().x to store diff^2
	    // t.get<2>().y = [ (Re(src)^2 + Im(src)^2) / (Re(tar)^2 + Im(tar)^2)-1.0 ]
	    t.get<2>().y = ((t.get<0>().x * t.get<0>().x +
			     t.get<0>().y * t.get<0>().y +
			     FFT_KLD_FLOOR_NUM) /
			    (t.get<1>().x * t.get<1>().x +
			     t.get<1>().y * t.get<1>().y +
			     FFT_KLD_FLOOR_NUM) - 1.0);
	    t.get<2>().x = t.get<2>().y * t.get<2>().y;

	}
    };


    struct SpecAmpDistance_AMP_IS
    {
	__host__ __device__ void operator() (const thrust::tuple<complex_t &,
					     complex_t &, complex_t &> &t) const
	{
	    // t.get<0>() source
	    // t.get<1>() target
	    // t.get<2>() buffer to store diff

	    real_t source_amp = sqrt(t.get<0>().x * t.get<0>().x +
				     t.get<0>().y * t.get<0>().y +
				     FFT_IS_FLOOR_NUM);
	    
	    real_t target_amp = sqrt(t.get<1>().x * t.get<1>().x +
				     t.get<1>().y * t.get<1>().y +
				     FFT_IS_FLOOR_NUM);
	    
	    // To save memory space, t.get<2>().x to store diff^2
	    // t.get<2>().y=sqrt(Re(src)^2 + Im(src)^2)/sqrt(Re(tar)^2 + Im(tar)^2)
	    t.get<2>().y = target_amp / source_amp;
	    t.get<2>().x = t.get<2>().y - helpers::safeLog(t.get<2>().y) - 1.0;

	}
    };

    
    struct SpecAmpDistance_AMP_KLD
    {
	__host__ __device__ void operator() (const thrust::tuple<complex_t &,
					     complex_t &, complex_t &> &t) const
	{
	    // t.get<0>() source
	    // t.get<1>() target
	    // t.get<2>() buffer to store diff

	    real_t source = sqrt(t.get<0>().x * t.get<0>().x + t.get<0>().y * t.get<0>().y +
				 FFT_KLD_FLOOR_NUM);
	    real_t target = sqrt(t.get<1>().x * t.get<1>().x + t.get<1>().y * t.get<1>().y +
				 FFT_KLD_FLOOR_NUM);
	    
	    // t.x save the KLD divergence
	    t.get<2>().x = target * helpers::safeLog(target / source) - target + source;
	    // t.y save the intermediate results for gradient calculation
	    t.get<2>().y = 1.0 - target / source;

	}
    };

    struct SpecAmpDistance_AMP_MSE
    {
	__host__ __device__ void operator() (const thrust::tuple<complex_t &,
					     complex_t &, complex_t &> &t) const
	{
	    // t.get<0>() source
	    // t.get<1>() target
	    // t.get<2>() buffer to store diff

	    // To save memory space, t.get<2>().x to store diff^2
	    // t.get<2>().y = [ (Re(src)^2 + Im(src)^2) - (Re(tar)^2 + Im(tar)^2) ]
	    t.get<2>().y = ((t.get<0>().x * t.get<0>().x +
			     t.get<0>().y * t.get<0>().y) -
			    (t.get<1>().x * t.get<1>().x +
			     t.get<1>().y * t.get<1>().y));
	    
	    // t.get<2>().x = [ (Re(src)^2 + Im(src)^2) - (Re(tar)^2 + Im(tar)^2) ] ^ 2
	    t.get<2>().x = t.get<2>().y * t.get<2>().y;

	}
    };

    struct SpecAmpDistance_SPEC_MSE
    {
	__host__ __device__ void operator() (const thrust::tuple<complex_t &,
					     complex_t &, complex_t &> &t) const
	{
	    // t.get<0>() source
	    // t.get<1>() target
	    // t.get<2>() buffer to store diff

	    // t.get<2>().y is not used to store intermediate results
	    t.get<2>().y = 0;

	    // [Re(src) - Re(tar)]^2 + [Im(src) - Im(tar)]^2 
	    t.get<2>().x = (t.get<0>().x - t.get<1>().x) * (t.get<0>().x - t.get<1>().x) +
		(t.get<0>().y - t.get<1>().y) * (t.get<0>().y - t.get<1>().y);

	}
    };

    struct SpecAmpDistance_SPEC_LOG_MSE
    {
	__host__ __device__ void operator() (const thrust::tuple<complex_t &,
					     complex_t &, complex_t &> &t) const
	{
	    // t.get<0>() source
	    // t.get<1>() target
	    // t.get<2>() buffer to store diff

	    // [Re(src) - Re(tar)]^2 + [Im(src) - Im(tar)]^2 + const
	    t.get<2>().y = (t.get<0>().x - t.get<1>().x) * (t.get<0>().x - t.get<1>().x) +
		(t.get<0>().y - t.get<1>().y) * (t.get<0>().y - t.get<1>().y) + FFT_SPEC_LOG_MSE_CONST;

	    // log[ [Re(src) - Re(tar)]^2 + [Im(src) - Im(tar)]^2 + const ]
	    t.get<2>().x = helpers::safeLog(t.get<2>().y);
	    
	}
    };
    
    
    struct phaseDistance
    {
	__host__ __device__ void operator() (const thrust::tuple<complex_t &,
					     complex_t &, complex_t &> &t) const
	{
	    // t.get<0>() source
	    // t.get<1>() target
	    // t.get<2>() buffer to store diff

	    // To save memory space, t.get<2>().x stores
	    //   Amp_source * Amp_target
	    t.get<2>().y = (sqrt(t.get<0>().x * t.get<0>().x +
				 t.get<0>().y * t.get<0>().y) *
			    sqrt(t.get<1>().x * t.get<1>().x +
				 t.get<1>().y * t.get<1>().y));

	    if (t.get<2>().y < FFT_AMP_MIN_NUM){
		t.get<2>().x = 0.0;
		t.get<2>().y = FFT_AMP_MIN_NUM; // floor the amplitutde
	    }else{
		// phaseDistance = 1 - Re_s / Amp_s * Re_t / Amp_t - Im_s / Amp_s * Im_t / Amp_t
		t.get<2>().x = 1.0 -
		    (t.get<0>().x * t.get<1>().x + t.get<0>().y * t.get<1>().y) / t.get<2>().y;
	    }

	}
    };

    struct instantaneousPhaseDistance_step1
    {
	int fftBins;
	complex_t* source;
	complex_t* target;
	
	__host__ __device__ void operator() (const thrust::tuple<complex_t &,
					     complex_t &, complex_t &, int> &t) const
	{
	    // Calcualte and store the phase
	    // t.get<0>() source
	    // t.get<1>() target
	    // t.get<2>() buffer to store diff
	    // t.get<3>() index
	    
	    real_t src_phase_diff = atan2(source[t.get<3>()].y, source[t.get<3>()].x) -
		atan2(source[t.get<3>() - fftBins].y, source[t.get<3>() - fftBins].x);
	    real_t tar_phase_diff = atan2(target[t.get<3>()].y, target[t.get<3>()].x) -
		atan2(target[t.get<3>() - fftBins].y, target[t.get<3>() - fftBins].x);
	    
	    real_t ddmod_src = fmod(src_phase_diff + FFT_PI_DEFINITION, 2.0 * FFT_PI_DEFINITION) -
		FFT_PI_DEFINITION;
	    real_t ddmod_tar = fmod(tar_phase_diff + FFT_PI_DEFINITION, 2.0 * FFT_PI_DEFINITION) -
		FFT_PI_DEFINITION;

	    if (abs(ddmod_src + FFT_PI_DEFINITION) < FFT_AMP_MIN_NUM && src_phase_diff > 0)
		ddmod_src = FFT_PI_DEFINITION;
	    if (abs(ddmod_tar + FFT_PI_DEFINITION) < FFT_AMP_MIN_NUM && tar_phase_diff > 0)
		ddmod_tar = FFT_PI_DEFINITION;
			    
	    t.get<2>().x = ddmod_src - src_phase_diff;
	    t.get<2>().y = ddmod_tar - tar_phase_diff;
	    
	    if (src_phase_diff < 0)
		t.get<2>().x = 0;
	    if (tar_phase_diff < 0)
		t.get<2>().y = 0;
	    
	}
    };

    
    
    struct instantaneousPhaseDistance_step2
    {
	int fftBins;
	int validFrameNum;
	complex_t* phaseOff;
	
	__host__ __device__ void operator() (const thrust::tuple<complex_t &, int> &t) const
	{
	    
	    for (int i = 1; i<validFrameNum; i++){
		phaseOff[ i * fftBins + t.get<1>()].x = phaseOff[ i * fftBins + t.get<1>()].x +
		    phaseOff[ (i-1) * fftBins + t.get<1>()].x;
		phaseOff[ i * fftBins + t.get<1>()].y = phaseOff[ i * fftBins + t.get<1>()].y +
		    phaseOff[ (i-1) * fftBins + t.get<1>()].y;
	    }
	    
	}
    };

    struct instantaneousPhaseDistance_step3
    {
	int fftBins;	
	complex_t* source;
	complex_t* target;
	complex_t* phaseOff;
	
	__host__ __device__ void operator() (const thrust::tuple<complex_t &, complex_t &,
					     complex_t &, int> &t) const
	{
	    // Calcualte and store the phase
	    // t.get<0>() diff

	    // source - target
	    real_t src_phase_off_cur = phaseOff[t.get<3>()].x;
	    real_t tar_phase_off_cur = phaseOff[t.get<3>()].y;
	    
	    real_t src_phase_off_pre = phaseOff[t.get<3>() - fftBins].x;
	    real_t tar_phase_off_pre = phaseOff[t.get<3>() - fftBins].y;

	    // source
	    t.get<2>().y = 
		atan2(source[t.get<3>()].y, source[t.get<3>()].x) + src_phase_off_cur -
		atan2(source[t.get<3>() - fftBins].y, source[t.get<3>() - fftBins].x) -
		src_phase_off_pre;
	    
	    // target
	    t.get<2>().x =
		atan2(target[t.get<3>()].y, target[t.get<3>()].x) + tar_phase_off_cur -
		atan2(target[t.get<3>() - fftBins].y, target[t.get<3>() - fftBins].x) -
		tar_phase_off_pre;

	    //return;
	    t.get<2>().y = t.get<2>().y - t.get<2>().x;
	    
	    // (source - target) ^ 2
	    t.get<2>().x = t.get<2>().y * t.get<2>().y;
	}
    };

    struct SpecAmpDistance_AMP_LOG_MSE_Grad
    {
	__host__ __device__ void operator() (const thrust::tuple<complex_t &,
					     complex_t &, complex_t &> &t) const
	{
	    // t.get<0>() source
	    // t.get<1>() target
	    // t.get<2>() buffer to store diff
	    // Note: before calculation, t.get<2>().y is set by SpecAmpDistance_AMP_LOG_MSE
	    //   t.get<2>().y = [ log(Re(src)^2 + Im(src)^2) - log(Re(tar)^2 + Im(tar)^2) ]
	    // 
	    real_t spec = (t.get<0>().x * t.get<0>().x +
			   t.get<0>().y * t.get<0>().y +
			   FFT_KLD_FLOOR_NUM);
	    
	    t.get<2>().x = t.get<2>().y * t.get<0>().x / spec;
	    t.get<2>().y = t.get<2>().y * t.get<0>().y / spec;


	}
    };


    struct SpecAmpDistance_AMP_RATIO_Grad
    {
	__host__ __device__ void operator() (const thrust::tuple<complex_t &,
					     complex_t &, complex_t &> &t) const
	{
	    // t.get<0>() source
	    // t.get<1>() target
	    // t.get<2>() buffer to store diff

	    real_t spec = (t.get<1>().x * t.get<1>().x +
			   t.get<1>().y * t.get<1>().y +
			   FFT_KLD_FLOOR_NUM);
	    
	    t.get<2>().x = t.get<2>().y * t.get<0>().x / spec;
	    t.get<2>().y = t.get<2>().y * t.get<0>().y / spec;


	}
    };

    struct SpecAmpDistance_AMP_IS_Grad
    {
	__host__ __device__ void operator() (const thrust::tuple<complex_t &,
					     complex_t &, complex_t &> &t) const
	{
	    // t.get<0>() source
	    // t.get<1>() target
	    // t.get<2>() buffer to store diff

	    real_t spec = t.get<0>().x * t.get<0>().x +
		t.get<0>().y * t.get<0>().y + FFT_IS_FLOOR_NUM;
	    
	    t.get<2>().x = t.get<0>().x / spec * (1.0 - t.get<2>().y);
	    t.get<2>().y = t.get<0>().y / spec * (1.0 - t.get<2>().y);

	}
    };

    struct SpecAmpDistance_AMP_KLD_Grad
    {
	__host__ __device__ void operator() (const thrust::tuple<complex_t &,
					     complex_t &, complex_t &> &t) const
	{
	    // t.get<0>() source
	    // t.get<1>() target
	    // t.get<2>() buffer to store diff
	    real_t spec = sqrt(t.get<0>().x * t.get<0>().x +
			       t.get<0>().y * t.get<0>().y +
			       FFT_KLD_FLOOR_NUM);
	    
	    t.get<2>().x = t.get<2>().y * t.get<0>().x / spec;
	    t.get<2>().y = t.get<2>().y * t.get<0>().y / spec;

	}
    };


    struct SpecAmpDistance_AMP_MSE_Grad
    {
	__host__ __device__ void operator() (const thrust::tuple<complex_t &,
					     complex_t &, complex_t &> &t) const
	{
	    // t.get<0>() source
	    // t.get<1>() target
	    // t.get<2>().y to store diff

	    // t.get<2>().y stores (Re(src)^2 + Im(src)^2) - (Re(tar)^2 + Im(tar)^2)
	    // [ (Re(src)^2 + Im(src)^2) - (Re(tar)^2 + Im(tar)^2) ] * Re(src)
	    // [ (Re(src)^2 + Im(src)^2) - (Re(tar)^2 + Im(tar)^2) ] * Im(src)
	    t.get<2>().x = t.get<2>().y * t.get<0>().x;
	    t.get<2>().y = t.get<2>().y * t.get<0>().y;
	}
    };

    
    struct SpecAmpDistance_SPEC_MSE_Grad
    {
	__host__ __device__ void operator() (const thrust::tuple<complex_t &,
					     complex_t &, complex_t &> &t) const
	{
	    // t.get<0>() source
	    // t.get<1>() target
	    // t.get<2>().y to store diff

	    // Re(src) - Re(tar)
	    t.get<2>().x = t.get<0>().x - t.get<1>().x;
	    // Im(src) - Im(tar)
	    t.get<2>().y = t.get<0>().y - t.get<1>().y;
	}
    };

    
    struct SpecAmpDistance_SPEC_LOG_MSE_Grad
    {
	__host__ __device__ void operator() (const thrust::tuple<complex_t &,
					     complex_t &, complex_t &> &t) const
	{
	    // t.get<0>() source
	    // t.get<1>() target
	    // t.get<2>().y to store diff

	    // t.get<2>().y stores [Re(src) - Re(tar)]^2 + [Im(src) - Im(tar)]^2 + const
	    // Re(src) - Re(tar) /  (Re(src)^2 + Im(src)^2) - (Re(tar)^2 + Im(tar)^2)
	    t.get<2>().x = (t.get<0>().x - t.get<1>().x) / t.get<2>().y;
	    // Im(src) - Im(tar) / (Re(src)^2 + Im(src)^2) - (Re(tar)^2 + Im(tar)^2)
	    t.get<2>().y = (t.get<0>().y - t.get<1>().y) / t.get<2>().y;
	}
    };
    
    
    struct PhaseGrad
    {
	__host__ __device__ void operator() (const thrust::tuple<complex_t &,
					     complex_t &, complex_t &> &t) const
	{
	    // t.get<0>() source
	    // t.get<1>() target
	    // t.get<2>() buffer to store diff
	    real_t spec = (t.get<0>().x * t.get<0>().x +
			   t.get<0>().y * t.get<0>().y +
			   FFT_KLD_FLOOR_NUM);
	    
	    real_t fac  = (t.get<0>().y * t.get<1>().x - t.get<0>().x * t.get<1>().y) / t.get<2>().y;
	    t.get<2>().x = fac * (-1.0) * t.get<0>().y / spec;
	    t.get<2>().y = fac * t.get<0>().x / spec;
	    
	}
    };

    struct instantaneousPhaseGrad
    {
	int fftBins;
	complex_t* diffData;

	__host__ __device__ void operator() (const thrust::tuple<complex_t &,
					     complex_t &, complex_t &, int> &t) const
	{
	    // t.get<0>() source
	    // t.get<1>() target
	    // t.get<2>() buffer to store diff
	    // t.get<3>() data index
	    
	    real_t spec = (t.get<0>().x * t.get<0>().x +
			   t.get<0>().y * t.get<0>().y +
			   FFT_KLD_FLOOR_NUM);
	    
	    //   \widehat{Phase}(m,n) - \widehat{Phase}(m-1,n) - {Phase}(m,n) + {Phase}(m-1,n)
	    // - (\widehat{Phase}(m+1,n) - \widehat{Phase}(m,n) - {Phase}(m+1,n) + {Phase}(m,n))
	    real_t fac  = diffData[t.get<3>()].y - diffData[t.get<3>() + fftBins].y;
	    
	    t.get<2>().x = fac * (-1.0) * t.get<0>().y / spec;
	    t.get<2>().y = fac * t.get<0>().x / spec;
	    
	}
    };

    
    struct getError
    {
	real_t factor;
	
	__host__ __device__ real_t operator() (const complex_t &t) const
	{
	    return (real_t)t.x / factor;
	}
    };


    struct spectralAmpToZeroDistance
    {
	real_t factor;
	
	__host__ __device__ real_t operator() (const complex_t &t) const
	{
	    return (real_t)(t.x * t.x / factor + t.y * t.y / factor);
	}
    };

    

    struct collectGrad
    {
	int fftLength;
	int frameLength;
	int frameShift;
	int windowType;
	int frameNum;

	real_t  gradScale;
	real_t *gradData;
	
	
	__host__ __device__ void operator() (const thrust::tuple<real_t &, int> &t) const
	{
	    // starting index 0
	    // for the t-th waveform point, it is used in frames whose frame_idx staties
	    //  frame_idx * frame_shift <= t < frame_idx * frame_shift + frame_length
	    // so, frame_idx that for t would be
	    //  [ max[0, 1 + (t-frame_length)/frame_shift] , min[frame_num, t/frame_shift+1] )

	    int start_frame_idx = ((t.get<1>() >= frameLength) ?
				   (1 + (t.get<1>() - frameLength) / frameShift) :
				   0);
	    int end_frame_idx = t.get<1>() / frameShift + 1;
	    end_frame_idx = ((end_frame_idx < frameNum)?end_frame_idx:frameNum);

	    int framePos = 0;
	    
	    t.get<0>() = 0.0;
	    for (int frameIdx = start_frame_idx; frameIdx < end_frame_idx; frameIdx++){
		framePos = t.get<1>() - frameIdx * frameShift;
		// Hann window by default
		if (windowType == FFTMAT_WINDOW_SQUARE)
		    t.get<0>() += (gradData[frameIdx * fftLength + framePos]);
		else
		    t.get<0>() += (gradData[frameIdx * fftLength + framePos]
				   * 0.5 * (1 - cos(2.0 * FFT_PI_DEFINITION * framePos /
						    (frameLength - 1))));
	    }
	    t.get<0>() = t.get<0>() * gradScale;
	}
    };

}
}


namespace helpers {


    namespace fftTools{

	// calculate the number of frames
	int fftFrameNum(int signalLength, int frameLength, int frameShift){
	    if (signalLength < frameLength)
		return 1;
	    else
		return 1 + (signalLength - frameLength) / frameShift;
	}

	// number of fft bins (assume Hermitian symmetry)
	int fftBinsNum(int fftLength){
	    return fftLength / 2 + 1;
	}

    }

    template <typename TDevice>
    FFTMat<TDevice>::FFTMat(real_vector *rawData,
			    real_vector *framedData,
			    complex_vector *fftData,
			    int frameLength, int frameShift,
			    int windowType,  int fftLength,
			    int fftBins,
			    int batchSize,
			    int signalBufLength,
			    int signalLength,
			    int specDisType)
	
	: m_rawData         (rawData)
	, m_framedData      (framedData)
	, m_fftData         (fftData)
	, m_frameLength     (frameLength)
	, m_frameShift      (frameShift)
	, m_windowType      (windowType)
	, m_fftLength       (fftLength)
	, m_fftBins         (fftBins)
	, m_batchSize       (batchSize)
	, m_signalBufLength (signalBufLength)
	, m_signalLength    (signalLength)
	, m_disType         (specDisType)
	  
    {
	// m_windowType is not implemented for windows other than Hann
	// m_signalLength    (length of the current input signal)
	// m_signalBufLength (max length of all input signals)
	
	if (fftTools::fftFrameNum(m_rawData->size(), m_frameLength, m_frameShift) * m_fftLength !=
	    framedData->size() || m_rawData->size() != m_signalBufLength)
	    throw std::runtime_error("Error: mismatch size of rawData and framedData buffer");
	
	if (m_signalBufLength < m_signalLength)
	    throw std::runtime_error("Error: signal Buff length < signal length");

	// clean the buffer cells that are beyond the length of current sentence
	thrust::fill((*m_rawData).begin() + m_signalLength,
		     (*m_rawData).end(), 0.0);

	m_validFrameNum     = fftTools::fftFrameNum(m_signalLength, m_frameLength, m_frameShift);
	m_validDataPointNum = m_validFrameNum * m_frameLength;
    }

    template <typename TDevice>
    FFTMat<TDevice>::FFTMat(real_vector *framedData, complex_vector *fftData,
			    int fftLength, int fftBins, int batchSize)
	: m_rawData      (NULL)
	, m_framedData   (framedData)
	, m_fftData      (fftData)
	, m_fftLength    (fftLength)
	, m_fftBins      (fftBins)
	, m_frameShift   (-1)
	, m_windowType   (-1)
	, m_frameLength  (-1)
	, m_signalLength (-1)
	, m_batchSize    (batchSize)
	, m_disType      (0)
    {
	// check?
    }
	    
    template <typename TDevice>
    FFTMat<TDevice>::~FFTMat()
    {	
    }

    template <>
    void FFTMat<Cpu>::FFT()
    {
	throw std::runtime_error("FFT using Cpu is not implemented");
    }

    template <>
    void FFTMat<Cpu>::iFFT()
    {
	throw std::runtime_error("iFFT using Cpu is not implemented");
    }
    
    template <typename TDevice>
    void FFTMat<TDevice>::changeWindowType(const int windowType)
    {
	// change window type
	this->m_windowType = windowType;
    }

    template <typename TDevice>
    void FFTMat<TDevice>::frameSignal()
    {
	// Frame and window the original signal (m_rawData)
	//  here Hann window is used.
	// Framing is conducted on the entire m_rawData buffer.
	//  Since m_rawData is zero padded after m_signalLength, it is safe
	if (m_rawData == NULL || m_framedData == NULL)
	    throw std::runtime_error("signal and framedSignal buffer not initialized");

	{{
	    internal::FrameSignal fn1;
	    fn1.fftLength   = m_fftLength;
	    fn1.frameLength = m_frameLength;
	    fn1.frameShift  = m_frameShift;
	    fn1.windowType  = m_windowType;
	    fn1.rawData     = getRawPointer(*m_rawData);
	    thrust::for_each(
		thrust::make_zip_iterator(
			thrust::make_tuple((*m_framedData).begin(), 
					   thrust::counting_iterator<int>(0))),
		thrust::make_zip_iterator(
			thrust::make_tuple((*m_framedData).end(),  
					   thrust::counting_iterator<int>(0) +
					   m_framedData->size())),
		fn1);
	}}
	    
    }

    template <typename TDevice>
    void FFTMat<TDevice>::frameSignalRealSpec(const int realspec_type)
    {
	if (m_fftLength < 2 * m_frameLength)
	    throw std::runtime_error("FFTLength < 2*frameLength for real-valued spectrum");
	
	// calculate the even part of the signal
	{{
	    internal::FrameMakeEvenSignal fn1;
	    fn1.fftLength   = m_fftLength;
	    fn1.frameLength = m_frameLength;
	    fn1.frameShift  = m_frameShift;
	    fn1.windowType  = m_windowType;
	    fn1.rawData     = getRawPointer(*m_rawData);

	    thrust::for_each(
		thrust::make_zip_iterator(
			thrust::make_tuple((*m_framedData).begin(), 
					   thrust::counting_iterator<int>(0))),
		thrust::make_zip_iterator(
			thrust::make_tuple((*m_framedData).end(),  
					   thrust::counting_iterator<int>(0) +
					   m_framedData->size())),
		fn1);
	}}

	// if this is delta-shifted real-spectrum
	if (realspec_type == FFTMAT_REALSPEC_TYPE_SHIFT){{
	    internal::FrameDeltaShift fn1;
	    fn1.fftLength   = m_fftLength;
	    fn1.frameData   = getRawPointer(*m_framedData);
	    
	    thrust::for_each(
		thrust::make_zip_iterator(
			thrust::make_tuple((*m_framedData).begin(), 
					   thrust::counting_iterator<int>(0))),
		thrust::make_zip_iterator(
			thrust::make_tuple((*m_framedData).begin() + m_validFrameNum,  
					   thrust::counting_iterator<int>(0) +
					   m_validFrameNum)),
		fn1);	

	}}
	
    }

    
    
    template <>
    void FFTMat<Gpu>::FFT()
    {
	// conduct FFT
	cufftResult res;
	res = cufftExecR2C((*internal::getCuFFTHandle_fft(m_fftLength, m_batchSize)),
			   (cufftReal *)getRawPointer(*m_framedData),
			   (cufftComplex *)getRawPointer(*m_fftData));
	if (res != CUFFT_SUCCESS)
	    throw std::runtime_error("Error: could not create cufft plan");

    }

    template <>
    void FFTMat<Gpu>::iFFT()
    {
	// conduct iFFT
	cufftResult res;
	res = cufftExecC2R((*internal::getCuFFTHandle_ifft(m_fftLength, m_batchSize)),
			   (cufftComplex *)getRawPointer(*m_fftData),
			   (cufftReal *)getRawPointer(*m_framedData));
	if (res != CUFFT_SUCCESS)
	    throw std::runtime_error("Error: could not create cufft plan");
	
    }

    template <typename TDevice>
    real_t FFTMat<TDevice>::specAmpDistance(FFTMat<TDevice> &target, FFTMat<TDevice> &diff)
    {
	// calculate the spectral-distrince
	if (this->m_fftData->size() != target.m_fftData->size())
	    throw std::runtime_error("Spectral distance error: input fft data unequal size");

	if (m_disType == FFTMAT_SPECTYPE_AMP_IS){
	    {
		internal::SpecAmpDistance_AMP_IS fn1;
		thrust::for_each(
		thrust::make_zip_iterator(
			thrust::make_tuple((*m_fftData).begin(),
					   (*(target.m_fftData)).begin(),
					   (*(diff.m_fftData)).begin())),
		thrust::make_zip_iterator(
			thrust::make_tuple((*m_fftData).end(),
					   (*(target.m_fftData)).end(),
					   (*(diff.m_fftData)).end())),
		fn1);
	    }
	    
	}else if (m_disType == FFTMAT_SPECTYPE_AMP_RATIO){
	    {
		internal::SpecAmpDistance_AMP_RATIO fn1;
		thrust::for_each(
		thrust::make_zip_iterator(
			thrust::make_tuple((*m_fftData).begin(),
					   (*(target.m_fftData)).begin(),
					   (*(diff.m_fftData)).begin())),
		thrust::make_zip_iterator(
			thrust::make_tuple((*m_fftData).end(),
					   (*(target.m_fftData)).end(),
					   (*(diff.m_fftData)).end())),
		fn1);
	    }
	    
	}else if (m_disType == FFTMAT_SPECTYPE_SPEC_MSE){
	    {
		internal::SpecAmpDistance_SPEC_MSE fn1;
		thrust::for_each(
		thrust::make_zip_iterator(
			thrust::make_tuple((*m_fftData).begin(),
					   (*(target.m_fftData)).begin(),
					   (*(diff.m_fftData)).begin())),
		thrust::make_zip_iterator(
			thrust::make_tuple((*m_fftData).end(),
					   (*(target.m_fftData)).end(),
					   (*(diff.m_fftData)).end())),
		fn1);
	    }

	}else if (m_disType == FFTMAT_SPECTYPE_SPEC_LOG_MSE){
	    {
		internal::SpecAmpDistance_SPEC_LOG_MSE fn1;
		thrust::for_each(
		thrust::make_zip_iterator(
			thrust::make_tuple((*m_fftData).begin(),
					   (*(target.m_fftData)).begin(),
					   (*(diff.m_fftData)).begin())),
		thrust::make_zip_iterator(
			thrust::make_tuple((*m_fftData).end(),
					   (*(target.m_fftData)).end(),
					   (*(diff.m_fftData)).end())),
		fn1);
	    }
	    
	}else if (m_disType == FFTMAT_SPECTYPE_AMP_MSE){
	    {
		internal::SpecAmpDistance_AMP_MSE fn1;
		thrust::for_each(
		thrust::make_zip_iterator(
			thrust::make_tuple((*m_fftData).begin(),
					   (*(target.m_fftData)).begin(),
					   (*(diff.m_fftData)).begin())),
		thrust::make_zip_iterator(
			thrust::make_tuple((*m_fftData).end(),
					   (*(target.m_fftData)).end(),
					   (*(diff.m_fftData)).end())),
		fn1);
	    }	    
	}else if (m_disType == FFTMAT_SPECTYPE_AMP_KLD){
	    {
		internal::SpecAmpDistance_AMP_KLD fn1;
		thrust::for_each(
		thrust::make_zip_iterator(
			thrust::make_tuple((*m_fftData).begin(),
					   (*(target.m_fftData)).begin(),
					   (*(diff.m_fftData)).begin())),
		thrust::make_zip_iterator(
			thrust::make_tuple((*m_fftData).end(),
					   (*(target.m_fftData)).end(),
					   (*(diff.m_fftData)).end())),
		fn1);
	    }
	}else{
	    {
		internal::SpecAmpDistance_AMP_LOG_MSE fn1;
		thrust::for_each(
		thrust::make_zip_iterator(
			thrust::make_tuple((*m_fftData).begin(),
					   (*(target.m_fftData)).begin(),
					   (*(diff.m_fftData)).begin())),
		thrust::make_zip_iterator(
			thrust::make_tuple((*m_fftData).end(),
					   (*(target.m_fftData)).end(),
					   (*(diff.m_fftData)).end())),
		fn1);
	    }
	}

	// Spectral-distance averaged over (frameNum * fftBins)
	real_t distance = 0.0;
	{{
		internal::getError fn2;
		//fn2.factor = 1.0/this->m_fftData->size();
		fn2.factor = fftTools::fftBinsNum(m_fftLength) * m_validFrameNum;
		distance = thrust::transform_reduce((*(diff.m_fftData)).begin(),
						    (*(diff.m_fftData)).begin() + (int)fn2.factor,
						    fn2,
						    (real_t)0,
						    thrust::plus<real_t>());
	}}
	return distance;
    }



    template <typename TDevice>
    real_t FFTMat<TDevice>::specAmpToZeroDistance()
    {
	
	// Spectral-distance averaged over (frameNum * fftBins)
	real_t distance = 0.0;
	{{
		internal::spectralAmpToZeroDistance fn2;
		
		fn2.factor = fftTools::fftBinsNum(m_fftLength) * m_validFrameNum;
		distance = thrust::transform_reduce((*m_fftData).begin(),
						    (*m_fftData).begin() + (int)fn2.factor,
						    fn2,
						    (real_t)0,
						    thrust::plus<real_t>());
	}}
	return distance;
    }


    
    template <typename TDevice>
    real_t FFTMat<TDevice>::specPhaseDistance(FFTMat<TDevice> &target, FFTMat<TDevice> &diff)
    {
	// calculate the spectral-distrince
	if (this->m_fftData->size() != target.m_fftData->size())
	    throw std::runtime_error("Phase distance error: input fft data unequal size");
	
	// Phase-distance averaged over (frameNum * fftBins)
	real_t distance = 0.0;
	
	
	if (m_disType == FFTMAT_PHASETYPE_INTANT){
	    if (m_validFrameNum <= 2)
		return distance;
	    
	    complex_t tmp;
	    tmp.x = 0;
	    tmp.y = 0;
	    thrust::fill((*diff.m_fftData).begin(), (*diff.m_fftData).end(), tmp);
	    
	    // Step1. calculate phase_shift for unwrap
	    {
		internal::instantaneousPhaseDistance_step1 fn1;
		fn1.fftBins = m_fftBins;
		fn1.source = helpers::getRawPointer(*m_fftData);
		fn1.target = helpers::getRawPointer(*(target.m_fftData));
		
		thrust::for_each(
			thrust::make_zip_iterator(
				thrust::make_tuple(
					(*m_fftData).begin()          + m_fftBins,
					(*(target.m_fftData)).begin() + m_fftBins,
					(*(diff.m_fftData)).begin()   + m_fftBins,
					thrust::counting_iterator<int>(0) + m_fftBins)),
			thrust::make_zip_iterator(
				thrust::make_tuple(
					(*m_fftData).end(),
					(*(target.m_fftData)).end(),
					(*(diff.m_fftData)).end(),
					thrust::counting_iterator<int>(0) + this->m_fftData->size())),
			fn1);
	    }
	    
	    // Step2. calculate phase_shift.cumsum
	    {
		internal::instantaneousPhaseDistance_step2 fn1;
		fn1.fftBins  = m_fftBins;
		fn1.validFrameNum = m_validFrameNum;
		fn1.phaseOff = helpers::getRawPointer(*(diff.m_fftData));
		
		thrust::for_each(
			thrust::make_zip_iterator(
				thrust::make_tuple(
					(*(diff.m_fftData)).begin(),
					thrust::counting_iterator<int>(0))),
			thrust::make_zip_iterator(
				thrust::make_tuple(
					(*(diff.m_fftData)).begin()       + m_fftBins,
					thrust::counting_iterator<int>(0) + m_fftBins)),
			fn1);
	    }

	    complex_vector phaseOff = (*diff.m_fftData);
	    
	    // Step3. unwrap and calculate diff(phase) in diff.m_fftData.x y
	    {
		internal::instantaneousPhaseDistance_step3 fn1;
		fn1.fftBins = m_fftBins;
		fn1.source = helpers::getRawPointer(*m_fftData);
		fn1.target = helpers::getRawPointer(*(target.m_fftData));
		fn1.phaseOff = helpers::getRawPointer(phaseOff);
		
		thrust::for_each(
		     thrust::make_zip_iterator(
			  thrust::make_tuple(
				(*m_fftData).begin()              + m_fftBins,
				(*(target.m_fftData)).begin()     + m_fftBins,
				(*(diff.m_fftData)).begin()       + m_fftBins,
				thrust::counting_iterator<int>(0) + m_fftBins)),
		     thrust::make_zip_iterator(
			  thrust::make_tuple(
				(*m_fftData).begin()              + m_validFrameNum * m_fftBins,
				(*(target.m_fftData)).begin()     + m_validFrameNum * m_fftBins,
				(*(diff.m_fftData)).begin()       + m_validFrameNum * m_fftBins,
				thrust::counting_iterator<int>(0) + m_validFrameNum * m_fftBins)),
			fn1);
		
	    }	    
	    
	    // sum the distance
	    {
		internal::getError fn2;
		fn2.factor = fftTools::fftBinsNum(m_fftLength) * (m_validFrameNum - 2);
		distance = thrust::transform_reduce(
				(*(diff.m_fftData)).begin() + m_fftBins,
				(*(diff.m_fftData)).begin() + m_fftBins + (int)fn2.factor,
				fn2,
				(real_t)0,
				thrust::plus<real_t>());
		if (distance != distance)
		    return 0.0;
	    }
	    
	}else{
	    
	    // default cos distance
	    {
		internal::phaseDistance fn1;
		thrust::for_each(
		thrust::make_zip_iterator(
			thrust::make_tuple((*m_fftData).begin(),
					   (*(target.m_fftData)).begin(),
					   (*(diff.m_fftData)).begin())),
		thrust::make_zip_iterator(
			thrust::make_tuple((*m_fftData).end(),
					   (*(target.m_fftData)).end(),
					   (*(diff.m_fftData)).end())),
		fn1);
	    }
	    
	    // sum the distance
	    {
		internal::getError fn2;
		//fn2.factor = 1.0/this->m_fftData->size();
		fn2.factor = fftTools::fftBinsNum(m_fftLength) * m_validFrameNum;
		distance = thrust::transform_reduce((*(diff.m_fftData)).begin(),
						    (*(diff.m_fftData)).begin() + (int)fn2.factor,
						    fn2,
						    (real_t)0,
						    thrust::plus<real_t>());
	    }
	}

	return distance;
    }

    
    template <typename TDevice>
    void FFTMat<TDevice>::specAmpGrad(FFTMat<TDevice> &source, FFTMat<TDevice> &target)
    {

	// calculate the gradient at each frequency bin and each time step
	if (m_disType == FFTMAT_SPECTYPE_AMP_IS){
	    {
		internal::SpecAmpDistance_AMP_IS_Grad fn1;
		thrust::for_each(
		thrust::make_zip_iterator(
			thrust::make_tuple((*(source.m_fftData)).begin(),
					   (*(target.m_fftData)).begin(),
					   (*m_fftData).begin())),
		thrust::make_zip_iterator(
			thrust::make_tuple((*(source.m_fftData)).end(),
					   (*(target.m_fftData)).end(),
					   (*m_fftData).end())),
		fn1);
	    }
	    
	}else if (m_disType == FFTMAT_SPECTYPE_AMP_RATIO){
	    {
		internal::SpecAmpDistance_AMP_RATIO_Grad fn1;
		thrust::for_each(
		thrust::make_zip_iterator(
			thrust::make_tuple((*(source.m_fftData)).begin(),
					   (*(target.m_fftData)).begin(),
					   (*m_fftData).begin())),
		thrust::make_zip_iterator(
			thrust::make_tuple((*(source.m_fftData)).end(),
					   (*(target.m_fftData)).end(),
					   (*m_fftData).end())),
		fn1);
	    }

	}else if (m_disType == FFTMAT_SPECTYPE_SPEC_MSE){
	    {
		internal::SpecAmpDistance_SPEC_MSE_Grad fn1;
		thrust::for_each(
		thrust::make_zip_iterator(
			thrust::make_tuple((*(source.m_fftData)).begin(),
					   (*(target.m_fftData)).begin(),
					   (*m_fftData).begin())),
		thrust::make_zip_iterator(
			thrust::make_tuple((*(source.m_fftData)).end(),
					   (*(target.m_fftData)).end(),
					   (*m_fftData).end())),
		fn1);
	    }
	    
	}else if (m_disType == FFTMAT_SPECTYPE_SPEC_LOG_MSE){
	    {
		internal::SpecAmpDistance_SPEC_LOG_MSE_Grad fn1;
		thrust::for_each(
		thrust::make_zip_iterator(
			thrust::make_tuple((*(source.m_fftData)).begin(),
					   (*(target.m_fftData)).begin(),
					   (*m_fftData).begin())),
		thrust::make_zip_iterator(
			thrust::make_tuple((*(source.m_fftData)).end(),
					   (*(target.m_fftData)).end(),
					   (*m_fftData).end())),
		fn1);
	    }
	    
	}else if (m_disType == FFTMAT_SPECTYPE_AMP_MSE){
	    {
		internal::SpecAmpDistance_AMP_MSE_Grad fn1;
		thrust::for_each(
		thrust::make_zip_iterator(
			thrust::make_tuple((*(source.m_fftData)).begin(),
					   (*(target.m_fftData)).begin(),
					   (*m_fftData).begin())),
		thrust::make_zip_iterator(
			thrust::make_tuple((*(source.m_fftData)).end(),
					   (*(target.m_fftData)).end(),
					   (*m_fftData).end())),
		fn1);
	    } 

	
	}else if (m_disType == FFTMAT_SPECTYPE_AMP_KLD){
	    {
		internal::SpecAmpDistance_AMP_KLD_Grad fn1;
		thrust::for_each(
		thrust::make_zip_iterator(
			thrust::make_tuple((*(source.m_fftData)).begin(),
					   (*(target.m_fftData)).begin(),
					   (*m_fftData).begin())),
		thrust::make_zip_iterator(
			thrust::make_tuple((*(source.m_fftData)).end(),
					   (*(target.m_fftData)).end(),
					   (*m_fftData).end())),
		fn1);
	    }
	}else{
	      {
		  internal::SpecAmpDistance_AMP_LOG_MSE_Grad fn1;
		  thrust::for_each(
		thrust::make_zip_iterator(
			thrust::make_tuple((*(source.m_fftData)).begin(),
					   (*(target.m_fftData)).begin(),
					   (*m_fftData).begin())),
		thrust::make_zip_iterator(
			thrust::make_tuple((*(source.m_fftData)).end(),
					   (*(target.m_fftData)).end(),
					   (*m_fftData).end())),
		fn1);
	      }
	}
	
    }

    template <typename TDevice>
    void FFTMat<TDevice>::specAmpToZeroGrad()
    {
	// do nothing because the gradients of spectralAmpToZeroDistance is the FFT(resi)
	return;
    }

    template <typename TDevice>
    void FFTMat<TDevice>::specPhaseGrad(FFTMat<TDevice> &source, FFTMat<TDevice> &target)
    {
	
	if (m_disType == FFTMAT_PHASETYPE_INTANT){
	    if (m_validFrameNum <= 2)
		return;

	    {
		internal::instantaneousPhaseGrad fn1;
		fn1.fftBins = m_fftBins;
		fn1.diffData = helpers::getRawPointer(*m_fftData);
		
		thrust::for_each(
			thrust::make_zip_iterator(
				thrust::make_tuple(
					(*(source.m_fftData)).begin()     + m_fftBins,
					(*(target.m_fftData)).begin()     + m_fftBins,
					(*m_fftData).begin()              + m_fftBins,
					thrust::counting_iterator<int>(0) + m_fftBins)),
			thrust::make_zip_iterator(
				thrust::make_tuple(
					(*(source.m_fftData)).end()       - m_fftBins,
					(*(target.m_fftData)).end()       - m_fftBins,
					(*m_fftData).end()                - m_fftBins,
					thrust::counting_iterator<int>(0) +
					this->m_fftData->size() - m_fftBins)),
			fn1);		
	    }
	    complex_t tmp;
	    tmp.x = 0;
	    tmp.y = 0;
	    thrust::fill((*m_fftData).begin(), (*m_fftData).begin()+m_fftBins, tmp);
	    thrust::fill((*m_fftData).begin()+ (m_validFrameNum -1) * m_fftBins,
			 (*m_fftData).end(), tmp);

	}else{
	    {
		internal::PhaseGrad fn1;
		thrust::for_each(
		thrust::make_zip_iterator(
			thrust::make_tuple((*(source.m_fftData)).begin(),
					   (*(target.m_fftData)).begin(),
					   (*m_fftData).begin())),
		thrust::make_zip_iterator(
			thrust::make_tuple((*(source.m_fftData)).end(),
					   (*(target.m_fftData)).end(),
					   (*m_fftData).end())),
		fn1);
	    }
	}
	
    }
    
    template <typename TDevice>
    void FFTMat<TDevice>::collectGrad(real_t gradScaleFactor)
    {

	// Fatal Error
	// *** this line of code sets the frames near the end of the memory buff to 0.0,
	//     note the end waveform points in each frame !!!
	// set the gradients of dummy time point to zero
	// thrust::fill((*m_framedData).begin() + m_validDataPointNum, (*m_framedData).end(), 0.0);
	//
	// now, clean the gradients correspoding to the zero-padded values in each frame
	// maybe this step is unnecessary if we carefully gather the gradients in collectGrad
	{{
	    internal::cleanDummyGradients fn1;
	    fn1.fftLength   = m_fftLength;
	    fn1.frameLength = m_frameLength;
	    fn1.frameShift  = m_frameShift;
	    fn1.validFrameNum = m_validFrameNum;
	    thrust::for_each(
		thrust::make_zip_iterator(
			thrust::make_tuple((*m_framedData).begin(), 
					   thrust::counting_iterator<int>(0))),
		thrust::make_zip_iterator(
			thrust::make_tuple((*m_framedData).end(),  
					   thrust::counting_iterator<int>(0) +
					   m_framedData->size())),
		fn1);
	}}

	
	// calculate the gradient w.r.t original signal by accumulating
	// gradients w.r.t framed signals
	{{
	    internal::collectGrad fn1;
	    fn1.fftLength  = m_fftLength;
	    fn1.frameLength= m_frameLength;
	    fn1.frameShift = m_frameShift;
	    fn1.frameNum   = m_framedData->size() / m_fftLength;
	    fn1.windowType = m_windowType;
	    fn1.gradData   = getRawPointer(*m_framedData);
	    fn1.gradScale  = gradScaleFactor / (fftTools::fftBinsNum(m_fftLength) * m_validFrameNum);
		
	    thrust::for_each(
		thrust::make_zip_iterator(
			thrust::make_tuple((*m_rawData).begin(), 
					   thrust::counting_iterator<int>(0))),
		thrust::make_zip_iterator(
			thrust::make_tuple((*m_rawData).end(),  
					   thrust::counting_iterator<int>(0) +
					   m_rawData->size())),
		fn1);

	}}
    }


    template <typename TDevice>
    void FFTMat<TDevice>::collectGradRealSpec(real_t gradScaleFactor, const int realspec_type,
					      FFTMat<TDevice> &source)
    {

	// collect gradients caused by the delta-shifted operation
	if (realspec_type == FFTMAT_REALSPEC_TYPE_SHIFT){{

	    {{
	    internal::GradFromDeltaShiftedSignal_nonZeroPosition fn1;
	    fn1.fftLength   = m_fftLength;
	    fn1.rawGradData = getRawPointer(*m_framedData);
	    fn1.framedData  = getRawPointer(*(source.m_framedData));
	    
	    thrust::for_each(
		thrust::make_zip_iterator(
			thrust::make_tuple((*m_framedData).begin(), 
					   thrust::counting_iterator<int>(0))),
		thrust::make_zip_iterator(
			thrust::make_tuple((*m_framedData).end(),  
					   thrust::counting_iterator<int>(0) +
					   m_framedData->size())),
		fn1);
	    }}
	    
	    {{
	    internal::GradFromDeltaShiftedSignal_zeroPosition fn2;
	    fn2.fftLength   = m_fftLength;
	    fn2.rawGradData = getRawPointer(*m_framedData);
	    thrust::for_each(
		thrust::make_zip_iterator(
			thrust::make_tuple((*m_framedData).begin(), 
					   thrust::counting_iterator<int>(0))),
		thrust::make_zip_iterator(
			thrust::make_tuple((*m_framedData).begin() + m_validFrameNum,
					   thrust::counting_iterator<int>(0) +
					   m_validFrameNum)),
		fn2);
	    }}
	}}
	
	// collect gradients from the mirrorred signal to the original signal
	{{
	    internal::GradFromMirrorToOriginal fn1;
	    fn1.fftLength   = m_fftLength;
	    fn1.frameLength = m_frameLength;
	    fn1.frameShift  = m_frameShift;
	    fn1.windowType  = m_windowType;
	    fn1.rawGradData = getRawPointer(*m_framedData);
	    thrust::for_each(
		thrust::make_zip_iterator(
			thrust::make_tuple((*m_framedData).begin(), 
					   thrust::counting_iterator<int>(0))),
		thrust::make_zip_iterator(
			thrust::make_tuple((*m_framedData).end(),  
					   thrust::counting_iterator<int>(0) +
					   m_framedData->size())),
		fn1);
	}}
	
	// do normal gradients collection
	this->collectGrad(gradScaleFactor);
    }

    
    template class FFTMat<Cpu>;
    template class FFTMat<Gpu>;
}
