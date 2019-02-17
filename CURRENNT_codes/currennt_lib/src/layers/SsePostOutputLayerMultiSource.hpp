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


#ifndef LAYERS_SSEPOSTOUTPUTMULTILAYER_HPP
#define LAYERS_SSEPOSTOUTPUTMULTILAYER_HPP

#include "PostOutputLayer.hpp"

namespace layers {

    /******************************************************************************************//**
     * This layer is used as the very last layer which store the target outputs and calculates the
     * error between the target outputs and the actual outputs
     *
     * @param TDevice The computation device (Cpu or Gpu)
     *********************************************************************************************/
    template <typename TDevice>
    class SsePostOutputMultiLayer : public PostOutputLayer<TDevice>
    {
    private:
	std::vector<Layer<TDevice>*> m_sourceLayers;
	std::string                  m_sourceLayersStr;
	std::vector<std::string>     m_sourceLayersNames;
	
    public:
        /**
         * Constructs the Layer
         *
         * @param layerChild     The layer child of the JSON configuration for this layer
         * @param precedingLayer The layer preceding this one
         */
        SsePostOutputMultiLayer(
            const helpers::JsonValue &layerChild, 
            Layer<TDevice> &precedingLayer,
	    int             maxSeqLength,
	    int             layerID
            );

        /**
         * Destructs the Layer
         */
        virtual ~SsePostOutputMultiLayer();

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

	/**
         * @see Layer::computeBackwardPass()
         */
        virtual void computeBackwardPass(const int timeStep, const int nnState);

	// output the layer configuration
	virtual void exportLayer(const helpers::JsonValue &layersArray,
				 const helpers::JsonAllocator &allocator) const;

	/*
	 * @ linkTargetLayer
	 */
	void linkTargetLayer(Layer<TDevice> &targetLayer);
    };

} // namespace layers


#endif // LAYERS_SSEPOSTOUTPUTMULTILAYER_HPP
