/******************************************************************************
 * This file is an addtional component of CURRENNT. 
 * Xin WANG
 * National Institute of Informatics, Japan
 * 2016
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

#ifndef LAYERS_BATCHNORMNEWLAYER_HPP
#define LAYERS_BATCHNORMNEWLAYER_HPP

#include "TrainableLayer.hpp"


namespace layers {

    /******************************************************************************************//**
     * Represents a feed forward layer in the neural network
     *
     * @param TDevice The computation device (Cpu or Gpu)
     * @param TActFn  The activation function to use
     * 
     * This code is directly built based on FeedForwardLayer.hpp and FeedForwardLayer.cu
     * Different from BatchNorm: this BatchNormNew layer has activation function
     *********************************************************************************************/
    template <typename TDevice, typename TActFn>
    class BatchNormNewLayer : public TrainableLayer<TDevice>
    {
	typedef typename TDevice::real_vector real_vector;
	typedef typename TDevice::int_vector  int_vector;
	typedef typename TDevice::bool_vector bool_vector;
	typedef typename TDevice::pattype_vector pattype_vector;

	int  m_batchNormGenUseTrainMV;    // whether to use mean and std of training data to normalize
	
	real_vector m_stats;              // buffer to store mean and std
	real_vector m_outNormed;          // normed data output without being scaled
	
	real_t      m_stdConst;           // const floor for the var
	real_t      m_batchCnt;           // counter for batch normalization
	
	
	int         m_preEpoch;           // epoch number counter
	real_t      m_batchSize;          // number of data (frames) in a mini-batch

	real_vector m_oneVector;          // temporary buffer to store a all-one vector
	real_vector m_buff;               // temporary buffer



	// common methods to allocate / release memory
	void __allocateLocalMem();
	void __clearLocalMem();

	// initialization methods for different operation mode
	void __batchnorm_ini(const helpers::JsonValue &weightsSection);
	
	// forward computation (normal case, given one sequence of input)
	void __batchnorm_computeForwardPass(const int nnState);
	
	// forward computation (online case, given one frame of input data per step)
	void __batchnorm_computeForwardPass(const int timeStep, const int nnState,
					    const int effTimeStart, const int effTimeEnd,
					    const int shiftIn,      const int shiftOut);
	

	// backward computation sub-routines
	void __batchnorm_computeBackwardPass(const int nnState);
	
	
    public:
        /**
         * Constructs the Layer
         *
         * @param layerChild     The layer child of the JSON configuration for this layer
         * @param weightsSection The weights section of the JSON configuration
         * @param precedingLayer The layer preceding this one
         */
        BatchNormNewLayer(
            const helpers::JsonValue &layerChild, 
            const helpers::JsonValue &weightsSection,
            Layer<TDevice>           &precedingLayer,
	    int                       maxSeqLength,
	    int                       layerID
            );

        /**
         * Destructs the Layer
         */
        virtual ~BatchNormNewLayer();

        /**
         * @see Layer::type()
         */
        virtual const std::string& type() const;

        /**
         * @see Layer::computeForwardPass()
         */
        virtual void computeForwardPass(const int nnState);

         /**
         * @see Layer::computeBackwardPass()
         */
        virtual void computeBackwardPass(const int nnState);

	
	/***
	 * 
	 */
	virtual void computeForwardPass(const int timeStep, const int nnState);

	virtual void computeBackwardPass(const int timeStep, const int nnState);
	

	// export
	virtual void exportLayer(const helpers::JsonValue &layersArray, 
				 const helpers::JsonAllocator &allocator) const;

	/*
	 * to optimize the memory usage
	 */
	virtual void reduceOutputBuffer();

	virtual int outputBufPtrBias(const int timeStepTimesParallel, const int nnState);

	void clearAllBuffers();

	void resizeAllBuffers(const int timeLength);

	
    };

} // namespace layers


#endif
