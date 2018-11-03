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
/****
 *
 *
 *
 ****/

#ifndef LAYERS_DISTILLING_HPP
#define LAYERS_DISTILLING_HPP

#include "SkipLayer.hpp"
#include <boost/foreach.hpp>
#include <boost/shared_ptr.hpp>
#include <vector>

namespace layers {
    
    /**********************************************************************
     An interface layer for distilling

     **********************************************************************/
    
    // class definition
    template <typename TDevice>
    class DistillingLayer : public SkipLayer<TDevice>
    {
	typedef typename TDevice::real_vector     real_vector;
	typedef typename TDevice::complex_vector  fft_vector;
	typedef typename Cpu::real_vector         cpu_real_vector;
	typedef typename TDevice::int_vector      int_vector;
	typedef typename TDevice::pattype_vector  pattype_vector;
	typedef typename Cpu::int_vector          cpu_int_vector;
        
    private:
	Layer<TDevice>    *m_targetLayer;       // target layer (where likelihood is evaluated)
	Layer<TDevice>    *m_targetBeforeLayer; //
	
	int                m_distillingOpt;     //

	real_t             m_alpha;
	
	real_t             m_beta;
	real_t             m_mseError;
	real_t             m_kld;

	real_t             m_gamma;
	real_t             m_zeta;
	
	// Short time STFT part (configurable))
	real_t             m_specError;
	real_t             m_phaseError;
	int                m_fftLength;
	int                m_fftBinsNum;
	int                m_frameLength;
	int                m_frameShift;
	int                m_frameNum;
	
	real_vector        m_fftSourceFramed;
	fft_vector         m_fftSourceSigFFT;

	real_vector        m_fftTargetFramed;
	fft_vector         m_fftTargetSigFFT;

	real_vector        m_fftDiffData;
	real_vector        m_fftDiffFramed;
	fft_vector         m_fftDiffSigFFT;

	real_vector         m_fftDiffDataPhase;

	// Long time STFT 
	real_t             m_specError2;
	real_t             m_phaseError2;
	int                m_fftLength2;
	int                m_fftBinsNum2;
	int                m_frameLength2;
	int                m_frameShift2;
	int                m_frameNum2;

	real_vector        m_fftSourceFramed2;
	fft_vector         m_fftSourceSigFFT2;

	real_vector        m_fftTargetFramed2;
	fft_vector         m_fftTargetSigFFT2;

	real_vector        m_fftDiffData2;
	real_vector        m_fftDiffFramed2;
	fft_vector         m_fftDiffSigFFT2;

	real_vector         m_fftDiffDataPhase2;

	// Long time STFT 
	real_t             m_specError3;
	real_t             m_phaseError3;
	int                m_fftLength3;
	int                m_fftBinsNum3;
	int                m_frameLength3;
	int                m_frameShift3;
	int                m_frameNum3;

	real_vector        m_fftSourceFramed3;
	fft_vector         m_fftSourceSigFFT3;

	real_vector        m_fftTargetFramed3;
	fft_vector         m_fftTargetSigFFT3;

	real_vector        m_fftDiffData3;
	real_vector        m_fftDiffFramed3;
	fft_vector         m_fftDiffSigFFT3;

	real_vector         m_fftDiffDataPhase3;
	
	
	int_vector         m_gradPtr;           // pointers to collect gradients from MDN
	real_vector        m_realTargetVector;  // real target data

	int                m_specDisType;
    public:
	
	
	// Construct the layer
	DistillingLayer(const helpers::JsonValue &layerChild,
			 const helpers::JsonValue &weightsSection,
			 std::vector<Layer<TDevice>*> &precedingLayers,
			 int maxSeqLength,
			 int layerID);

	// Destructor
	virtual ~DistillingLayer();
	
	// void 
	virtual const std::string& type() const;

	// NN forward
	virtual void computeForwardPass(const int nnState);

	// NN forward
	virtual void computeForwardPass(const int timeStep, const int nnState);
	
	// NN backward
	virtual void computeBackwardPass(const int nnState);

	
	// output the layer configuration
	virtual void exportLayer(const helpers::JsonValue &layersArray,
				 const helpers::JsonAllocator &allocator) const;

	virtual void reduceOutputBuffer();

	virtual int outputBufPtrBias(const int timeStepTimesParallel, const int nnState);

	// load the target data from the target layer
	void linkTargetLayer(Layer<TDevice> &targetLayer);

	real_t intermediateError();

	std::vector<int> dependLayerIDs();
	
	void clearAllBuffers();

	void resizeAllBuffers(const int timeLength);
    };

}


#endif 


