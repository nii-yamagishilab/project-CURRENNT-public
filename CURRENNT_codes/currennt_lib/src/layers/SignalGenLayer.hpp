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
/*
 */

#ifndef LAYERS_SIGNALGENLAYER_HPP
#define LAYERS_SIGNALGENLAYER_HPP


#include "TrainableLayer.hpp"

namespace layers{
    
    template <typename TDevice>
    class SignalGenLayer : public TrainableLayer<TDevice>
    {
	typedef typename TDevice::real_vector     real_vector;
	typedef typename Cpu::real_vector         cpu_real_vector;
	typedef typename TDevice::int_vector      int_vector;
	typedef typename TDevice::pattype_vector  pattype_vector;
	typedef typename Cpu::int_vector          cpu_int_vector;

    private:
	int             m_freqDim;         // which dimension of input specifies the F0
	int             m_freqOpt;         // what kind of F0 is provided?
	int             m_freqBins;        // when phrase-lock is used, specify number of candidates
	int             m_freqHmn;         // number of harmonics
	real_t          m_freqSR;          // sampling rate of the signal
	real_t          m_freqQF0min;
	real_t          m_freqQF0max;
	real_t          m_freqQF0Lev;
	real_t          m_freqDataM;       // input F0 data mean
	real_t          m_freqDataS;       // input F0 data var

	real_t          m_freqSignalMag;
	real_t          m_noiseMag;
	real_t          m_phaseNoiseMag;
	
	Layer<TDevice> *m_targetLayer;     // target layer to load the target sinal

	real_vector     m_freqSignalBuff;
	real_vector     m_freqHmnBuff;
	real_vector     m_noiseInput;
	real_vector     m_phaseNoise;
	
	real_vector     m_freqErrorBuff;
	real_vector     m_signalStatic;	

	int             m_equalNoiseSinePower;

	void __loadOpts(const helpers::JsonValue &layerChild);
	void __setLayerMode();
	void __allocateLocalMem();
	void __clearLocalMem();
	
    public:
	SignalGenLayer(
	    const helpers::JsonValue &layerChild,
	    const helpers::JsonValue &weightsSection,
            Layer<TDevice>           &precedingLayer,
	    int                       maxSeqLength,
	    int                       layerID
	);

	virtual ~SignalGenLayer();
	
	
	virtual const std::string& type() const;
	
	// NN forward
	virtual void computeForwardPass(const int nnState);
	
	// NN forward, per frame
	virtual void computeForwardPass(const int timeStep, const int nnState);
	
	// NN backward
	virtual void computeBackwardPass(const int nnState);

	// export
	virtual void exportLayer(const helpers::JsonValue &layersArray, 
				 const helpers::JsonAllocator &allocator) const;
	//
	virtual void loadSequences(const data_sets::DataSetFraction &fraction, const int nnState);

	virtual void reduceOutputBuffer();
	
	virtual int outputBufPtrBias(const int timeStepTimesParallel, const int nnState);

	virtual void linkTargetLayer(Layer<TDevice> &targetLayer);

	std::vector<int> dependLayerIDs();

	void clearAllBuffers();

	void resizeAllBuffers(const int timeLength);
	
    };
    
}

#endif
