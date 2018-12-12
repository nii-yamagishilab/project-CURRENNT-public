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

#ifndef LAYERS_DFTERRORPOSTOUTPUT_HPP
#define LAYERS_DFTERRORPOSTOUTPUT_HPP

#include "PostOutputLayer.hpp"


namespace layers {

    template <typename TDevice>
    class DFTPostoutputLayer : public PostOutputLayer<TDevice>
    {
	typedef typename TDevice::real_vector     real_vector;
	typedef typename TDevice::complex_vector  fft_vector;
	typedef typename Cpu::real_vector         cpu_real_vector;
	typedef typename TDevice::int_vector      int_vector;
	typedef typename TDevice::pattype_vector  pattype_vector;
	typedef typename Cpu::int_vector          cpu_int_vector;
	
    private:
	
	real_t             m_beta;              // Weight for waveform MSE
	real_t             m_mseError;         
	
	real_t             m_gamma;             // Weight for DFT amplitude 
	real_t             m_zeta;              // Weight for DFT phase

	int                m_specDisType;       // Type of spectral amplitude distance
	
	// Short time STFT part (configurable))
	real_t             m_specError;
	real_t             m_phaseError;
	int                m_fftLength;
	int                m_fftBinsNum;
	int                m_frameLength;
	int                m_frameShift;
	int                m_frameNum;
	int                m_windowType;
	int                m_windowTypePhase;
	
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
	int                m_windowType2;
	int                m_windowTypePhase2;
	
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
	int                m_windowType3;
	int                m_windowTypePhase3;
	
	real_vector        m_fftSourceFramed3;
	fft_vector         m_fftSourceSigFFT3;

	real_vector        m_fftTargetFramed3;
	fft_vector         m_fftTargetSigFFT3;

	real_vector        m_fftDiffData3;
	real_vector        m_fftDiffFramed3;
	fft_vector         m_fftDiffSigFFT3;

	real_vector        m_fftDiffDataPhase3;


	// support for harmonic + noise model
	int                m_hnm_flag;
	int                m_noiseTrain_epoch;
	Layer<TDevice>*    m_noiseOutputLayer;
	Layer<TDevice>*    m_f0InputLayer;
	std::string        m_noiseOutputLayerName;
	std::string        m_f0InputLayerName;
	real_t             m_f0DataM;
	real_t             m_f0DataS;
	
	void __loadOpts(const helpers::JsonValue &layerChild);
	
    public:
        /**
         * Constructs the Layer
         *
         * @param layerChild     The layer child of the JSON configuration for this layer
         * @param precedingLayer The layer preceding this one
         */
        DFTPostoutputLayer(
            const helpers::JsonValue &layerChild, 
            Layer<TDevice> &precedingLayer,
	    int             maxSeqLength,
	    int             layerID);

        /**
         * Destructs the Layer
         */
        virtual ~DFTPostoutputLayer();

        /**
         * @see Layer::type()
         */
        virtual const std::string& type() const;

        /**
         * @see PostOutputLayer::calculateError()
         */
        virtual real_t calculateError();

        /**
         * @see Layer::computeForwardPass()
         */
        virtual void computeForwardPass(const int nnState);

        /**
         * @see Layer::computeForwardPass()
         */
        virtual void computeForwardPass(const int timeStep, const int nnState);

         /**
         * @see Layer::computeBackwardPass()
         */
        virtual void computeBackwardPass(const int nnState);

	virtual void exportLayer(const helpers::JsonValue &layersArray, 
				 const helpers::JsonAllocator &allocator) const;


	// load the target data from the target layer
	virtual void linkTargetLayer(Layer<TDevice> &targetLayer);

    };

} // namespace layers

#endif
