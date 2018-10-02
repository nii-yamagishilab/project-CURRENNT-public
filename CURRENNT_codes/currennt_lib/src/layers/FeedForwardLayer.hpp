/******************************************************************************
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

#ifndef LAYERS_FEEDFORWARDLAYER_HPP
#define LAYERS_FEEDFORWARDLAYER_HPP

#include "TrainableLayer.hpp"


namespace layers {

    /******************************************************************************************//**
     * Represents a feed forward layer in the neural network
     *
     * @param TDevice The computation device (Cpu or Gpu)
     * @param TActFn  The activation function to use
     *********************************************************************************************/
    template <typename TDevice, typename TActFn>
    class FeedForwardLayer : public TrainableLayer<TDevice>
    {
	typedef typename TDevice::real_vector real_vector;
	typedef typename TDevice::int_vector  int_vector;
	typedef typename TDevice::bool_vector bool_vector;
	typedef typename TDevice::pattype_vector pattype_vector;

	bool m_batchNorm;            // whether to use batch normalization
	real_vector m_stats;         // mean and variance of each batch
	real_vector m_outNormed;     // normed data output without being scaled
	
	real_t      m_stdConst;      // const floor for the var
	real_t      m_batchCnt;
	// bool        m_trainFlag;  // replaced by m_flagTraining
	int         m_preEpoch;
	real_t      m_batchSize;     //

	real_vector m_oneVector;     // all-one vector
	real_vector m_buff;

	void __allocateLocalMem();
	void __clearLocalMem();
	
    public:
        /**
         * Constructs the Layer
         *
         * @param layerChild     The layer child of the JSON configuration for this layer
         * @param weightsSection The weights section of the JSON configuration
         * @param precedingLayer The layer preceding this one
         */
        FeedForwardLayer(
            const helpers::JsonValue &layerChild, 
            const helpers::JsonValue &weightsSection,
            Layer<TDevice>           &precedingLayer,
	    int                       maxSeqLength,
	    int                       layerID
            );

        /**
         * Destructs the Layer
         */
        virtual ~FeedForwardLayer();

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


#endif // LAYERS_FEEDFORWARDLAYER_HPP
