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

#ifndef LAYERS_TRAINABLELAYER_HPP
#define LAYERS_TRAINABLELAYER_HPP

#include "Layer.hpp"


namespace layers {

    /******************************************************************************************//**
     * Represents a layer with weights that can be trained
     *
     * @param TDevice The computation device (Cpu or Gpu)
     *********************************************************************************************/
    template <typename TDevice>
    class TrainableLayer : public Layer<TDevice>
    {
        typedef typename TDevice::real_vector real_vector;

    private:
        // Layer<TDevice> &m_precedingLayer;
	
        const int    m_inputWeightsPerBlock;
        const int    m_internalWeightsPerBlock;
        const real_t m_bias;
        const real_t m_learningRate;

        //real_vector m_outputErrors;
        real_vector  m_weights;
        real_vector  m_weightUpdates;
	
	// Add 04-13 for weight Mask
	real_vector  m_weightMask;  // the float data to store weight mask	
	bool         m_weightMaskFlag;     // whether to use the weight Mask
	
	int          m_weightNum;          // the number of weights
	unsigned     m_optOpt;             // Note: this is used for Average Gradient
	                                   // but now, it is used as a controller to fix the
	                                   // the weight of one layer (by setting gradients to 0)

	int          m_specialInitialization; // special initialization strategy
	
    protected:
        real_vector&    _weightUpdates();
	//const unsigned& _optOpt() const;
	
    public:
        /**
         * Constructs the Layer
         *
         * @param layerChild              The layer child of the JSON configuration for this layer
         * @param weightsSection          The weights section of the JSON configuration
         * @param inputWeightsPerBlock    The number of input weights per block
         * @param internalWeightsPerBlock The number of internal weights per block
         * @param precedingLayer          The layer preceding this one
         */
        TrainableLayer(
            const helpers::JsonValue &layerChild,
            const helpers::JsonValue &weightsSection,
            int                       inputWeightsPerBlock, 
            int                       internalWeightsPerBlock,
            Layer<TDevice>           &precedingLayer,
	    int                       maxSeqLength,
	    int                       layerID
            );

        /**
         * Destructs the Layer
         */
        virtual ~TrainableLayer();

        /**
         * Returns the preceding layer
         *
         * @return The preceding layer
         */
        Layer<TDevice>& precedingLayer();

        /**
         * Returns the preceding layer
         *
         * @return The preceding layer
         */
        const Layer<TDevice>& precedingLayer() const;

	/**
         * Returns the bias
         *
         * @return The bias
         */
        real_t bias() const;

        /**
         * Returns the learning rate used for this layer
         *
         * @return the learning rate used for this layer
         */
        real_t learningRate() const;

        /**
         * Calculates the output errors of the layer
         *
         * @return The output error
         */
        //real_vector& outputErrors();

        /**
         * Returns the current weights
         *
         * @return The current weights
         */
        real_vector& weights();

        /**
         * Returns the current weights
         *
         * @return The current weights
         */
        const real_vector& weights() const;

        /**
         * Returns the current weight updates
         *
         * @return The current weight updates
         */
        const real_vector& weightUpdates() const;
	
	// Add 0413 weight matrix
	/* Return the number of weight elements*/
	const int& weightNum() const; 
	// 
	const real_vector& weightMask() const;
	
	/* Read in the weight mask */
	void readWeightMask(std::vector<real_t>::iterator b, std::vector<real_t>::iterator e);

	const bool& flagUseWeightMask() const;

	const bool& adaGrad() const;

	virtual void maskWeight();
	
	
        /**
         * Adds Gaussian weight noise with the given standard deviation.
         * 
         * @param sigma the standard deviation of the Gaussian weight noise
         *
         * @return void
         */
        void injectWeightNoise(real_t sigma);

        /**
         * Stores the weights of the layer in a JSON object
         *
         * @param weightsObject The object containing the weights
         * @param allocator     The allocator to use
         */
        virtual void exportWeights(const helpers::JsonValue &weightsObject,
				   const helpers::JsonAllocator &allocator) const;

        /**
         * @see Layer::exportLayer()
         */
        virtual void exportLayer(const helpers::JsonValue &layersArray,
				 const helpers::JsonAllocator &allocator) const;

	/**
	 * Re-initialize the network
	   only defines for Trainable Layers
	 */
	virtual void reInitWeight();
	
	/**
	 * Read the weight from a layer of a trained network
	 */
	virtual void reReadWeight(const helpers::JsonValue &weightsSection, const int layerSize, 
				  const int readCtrFlag);
	

	/**
	 * return the weight configuration
	 */
	int inputWeightsPerBlock();
        int internalWeightsPerBlock();

	virtual void cleanGradidents();	
	const unsigned& optOpt() const;

	
    };

} // namespace layers


#endif // LAYERS_TRAINABLELAYER_HPP
