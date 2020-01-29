/******************************************************************************
 * This file is an addtional component of CURRENNT. 
 * Xin WANG
 * National Institute of Informatics, Japan
 * 2016 - 2020
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

#ifndef LAYERS_FEEDFORWARDLAYER_HPP
#define LAYERS_FEEDFORWARDLAYER_HPP

#include "TrainableLayer.hpp"


namespace layers {

    /*********************************************************************//**
     * Represents a feed forward layer in the neural network
     *
     * @param TDevice The computation device (Cpu or Gpu)
     * @param TActFn  The activation function to use
     ************************************************************************/
    template <typename TDevice, typename TActFn>
    class FeedForwardLayer : public TrainableLayer<TDevice>
    {
	// declarition of data type
	typedef typename TDevice::real_vector real_vector;
	typedef typename TDevice::int_vector  int_vector;
	typedef typename TDevice::bool_vector bool_vector;
	typedef typename TDevice::pattype_vector pattype_vector;

	/**
	 * Flags for batch-normalization
	 */
	// whether to use batch normalization
	bool m_batchNorm;
	
	// whether to use mean and std of training data to normalize
	// the input test data?
	int  m_batchNormGenUseTrainMV;
	
	// counter of training batches
	real_t      m_batchCnt;           

	// number of data (frames) in a training batch
	real_t      m_batchSize;

	
	/***
	 * Flags for weight-normalization (obsolete)
	 */
	// whether to use weight normalization (not fully test)
	bool     m_weightNorm;
	
	real_t   m_weightNormVal;
	
	real_t   m_weightNormGrad;


	/**
	 * Flags for layer normalization
	 */
	// whether to use layer normalization
	bool m_layerNorm;                 

	// buffer to store mean and std
	real_vector m_stats;
	
	// normed data output without being scaled
	real_vector m_outNormed;          

	
	/**
	 * Shared data buffer or statistics
	 */
	// const floor for the var
	real_t      m_stdConst;
	
	// epoch number counter
	int         m_preEpoch;           

	// temporary buffer to store a all-one vector
	real_vector m_oneVector;
	
	// temporary buffer
	real_vector m_buff;               


	
	/** 
	 * common methods to allocate / release memory 
	 */
	void __allocateLocalMem();
	void __clearLocalMem();

	/**
	 * initialization methods for different operation mode
	 */
	void __batchnorm_ini(const helpers::JsonValue &weightsSection);
	void __layernorm_ini(const helpers::JsonValue &weightsSection);
	void __weightnorm_ini(const helpers::JsonValue &weightsSection);

	/**
	 * forward computation (normal case, given one sequence of input)
	 */
	void __batchnorm_computeForwardPass(const int nnState);
	void __weightnorm_computeForwardPass(const int nnState);
	void __layernorm_computeForwardPass(const int nnState);
	void __computeForwardPass(const int nnState);

	/**
	 * forward computation (online case, given one frame of input data per step)
	 */
	void __batchnorm_computeForwardPass(const int timeStep, const int nnState,
					    const int effTimeStart, const int effTimeEnd,
					    const int shiftIn,      const int shiftOut);
	void __weightnorm_computeForwardPass(const int timeStep, const int nnState,
					     const int effTimeStart, const int effTimeEnd,
					     const int shiftIn,      const int shiftOut);
	void __layernorm_computeForwardPass(const int timeStep, const int nnState,
					    const int effTimeStart, const int effTimeEnd,
					    const int shiftIn,      const int shiftOut);
	void __computeForwardPass(const int timeStep, const int nnState,
				  const int effTimeStart, const int effTimeEnd,
				  const int shiftIn,      const int shiftOut);

	/**
	 * backward computation sub-routines
	 */
	void __batchnorm_computeBackwardPass(const int nnState);
	void __weightnorm_computeBackwardPass_p1(const int nnState);
	void __weightnorm_computeBackwardPass_p2(const int nnState);
	void __layernorm_computeBackwardPass(const int nnState);
	void __computeBackwardPass_bias(const int nnState);

	
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

	virtual void computeBackwardPass(const int timeStep, const int nnState);
	

	// export
	virtual void exportLayer(const helpers::JsonValue &layersArray, 
				 const helpers::JsonAllocator &allocator) const;

	/***
	 * to optimize the memory usage
	 */
	virtual void reduceOutputBuffer();

	virtual int outputBufPtrBias(const int timeStepTimesParallel, const int nnState);

	void clearAllBuffers();

	void resizeAllBuffers(const int timeLength);
	
	void logAllBuffers(helpers::vecPoolManager<TDevice> &vecPoolMng,
			   bool flag_add);
	
	void swapAllBuffers(helpers::vecPoolManager<TDevice> &vecPoolMng,
			    bool flag_get);
	
    };

} // namespace layers


#endif // LAYERS_FEEDFORWARDLAYER_HPP
