/******************************************************************************
 * This file is an addtional component of CURRENNT. 
 * Xin WANG
 * National Institute of Informatics, Japan
 * 2019
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

#ifndef LAYERS_SELF_ATTENTION_LAYER_HPP
#define LAYERS_SELF_ATTENTION_LAYER_HPP

#include "TrainableLayer.hpp"


namespace layers {

    template <typename TDevice>
    class SelfAttentionLayer : public TrainableLayer<TDevice>
    {
	typedef typename TDevice::real_vector real_vector;
	typedef typename TDevice::int_vector  int_vector;
	typedef typename TDevice::bool_vector bool_vector;
	typedef typename TDevice::pattype_vector pattype_vector;

	real_vector m_mat_v;   // matrix for value
	real_vector m_mat_k;   // matrix for key
	real_vector m_mat_q;   // matrix for query

	real_vector m_align;      // matrix for alignment
	real_vector m_align_grad; // matrix to store gradients of alignment

	real_vector m_grad_buf;      // a temporary buffer to store grad statistics
	real_vector m_softmax_buf;   // a temporary buffer for softmax

	real_vector m_one_vector;

	real_t      m_align_prior_w;

	void __loadOpts(const helpers::JsonValue &layerChild);
	void __allocateLocalMem();
	void __clearLocalMem();
	
    public:
	
        SelfAttentionLayer(
            const helpers::JsonValue &layerChild, 
            const helpers::JsonValue &weightsSection,
            Layer<TDevice>           &precedingLayer,
	    int                       maxSeqLength,
	    int                       layerID
            );

        /**
         * Destructs the Layer
         */
        virtual ~SelfAttentionLayer();

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


	virtual void logAllBuffers(helpers::vecPoolManager<TDevice> &vecPoolMng,
				   bool flag_add);
	
	virtual void swapAllBuffers(helpers::vecPoolManager<TDevice> &vecPoolMng,
				    bool flag_get);	
    };

} // namespace layers


#endif

