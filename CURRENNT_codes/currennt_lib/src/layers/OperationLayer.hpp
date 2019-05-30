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

#ifndef LAYERS_OPERATIONLAYER_HPP
#define LAYERS_OPERATIONLAYER_HPP


#include "TrainableLayer.hpp"

namespace layers{
    /*
      Modes of OperationLayer
      
      if LastShotMode is on
         do lastShotMode 1 2 3 4

      else if TimeResolutionMode is on
         options is decided by the time resolution of the previous and follow layers
	 option1: down samping 
	 option2: up sampling 

      else

         option1: concatenate noise vector
	 option2: scale the input vector
	 option3: duplicate input vector to N vectors
      
     */
    template <typename TDevice>
    class OperationLayer : public TrainableLayer<TDevice>
    {
	typedef typename TDevice::real_vector     real_vector;
	typedef typename Cpu::real_vector         cpu_real_vector;
	typedef typename TDevice::int_vector      int_vector;
	typedef typename TDevice::pattype_vector  pattype_vector;
	typedef typename Cpu::int_vector          cpu_int_vector;

    public:
	real_vector     m_setZeroVec_D;
	cpu_real_vector m_setZeroVec_H;
	std::string     m_setZeroStr;

	real_vector     m_setBiasVec_D;
	cpu_real_vector m_setBiasVec_H;
	std::string     m_setBiasStr;

	real_vector     m_stopGradVec_D;
	cpu_real_vector m_stopGradVec_H;
	std::string     m_stopGradStr;

	real_vector     m_setScaleVec_D;
	cpu_real_vector m_setScaleVec_H;
	std::string     m_setScaleStr;

	real_vector     m_setBinarizeVec_D;
	cpu_real_vector m_setBinarizeVec_H;
	std::string     m_setBinarizeStr;
	
	int             m_noiseSize;
	real_t          m_noiseMag;
	real_vector     m_noiseInput;
	int             m_noiseRepeat;

	int             m_outDupRate;

	real_vector     m_oneVec;
	int             m_lastShot;
	
	cpu_int_vector  m_seqLengthBuffH;  // the length of each sequence
	int_vector      m_seqLengthBuffD;  // the length of each sequence
	
	cpu_int_vector  m_segBoundaryH;    // position of the end of segment (for each frame)
	int_vector      m_segBoundaryD;
	int             m_segLevel;        // which level to be used ?

	int             m_changeTimeRes;   // whether to turn of time resolution change

	real_t          m_dropoutRate;

	int             m_freqDim;         // which dimension of input specifies the F0
	int             m_freqOpt;         // what kind of F0 is provided?
	real_t          m_freqSR;          // sampling rate of the signal
	real_t          m_freqQF0min;
	real_t          m_freqQF0max;
	real_t          m_freqQF0Lev;
	real_t          m_freqDataM;       // input F0 data mean
	real_t          m_freqDataS;       // input F0 data var

	int             m_F02UV;
	real_t          m_F0DataMean;
	real_t          m_F0DataStd;

	int             m_positional_code_mode;

	real_t          m_reverse_grad;
	
	OperationLayer(
	    const helpers::JsonValue &layerChild,
	    const helpers::JsonValue &weightsSection,
            Layer<TDevice>           &precedingLayer,
	    int                       maxSeqLength,
	    int                       layerID
	);

	virtual ~OperationLayer();
	
	
	virtual const std::string& type() const;
	
	// NN forward
	virtual void computeForwardPass(const int nnState);
	
	// NN forward, step by step
	virtual void computeForwardPass(const int timeStep, const int nnState);
	
	// NN backward
	virtual void computeBackwardPass(const int nnState);

	// NN backward, step by step
	virtual void computeBackwardPass(const int timeStep, const int nnState);

	// export
	virtual void exportLayer(const helpers::JsonValue &layersArray, 
				 const helpers::JsonAllocator &allocator) const;
	//
	virtual void loadSequences(const data_sets::DataSetFraction &fraction, const int nnState);

	virtual void reduceOutputBuffer();
	
	virtual int outputBufPtrBias(const int timeStepTimesParallel, const int nnState);

	void clearAllBuffers();

	void resizeAllBuffers(const int timeLength);

    };
    
}

#endif
