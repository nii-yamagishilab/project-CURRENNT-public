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

#ifndef LAYER_FACTORY_HPP
#define LAYER_FACTORY_HPP

#include "layers/Layer.hpp"

#include <boost/shared_ptr.hpp>
#include <vector>

/******************************************************************************************//**
 * Factory class for creating layers
 *
 * @param TDevice The computation device (Cpu or Gpu)
 *********************************************************************************************/
template <typename TDevice>
class LayerFactory
{
public:
    /**
     * Instantiates a layer
     *
     * @param layerType         The layer type (e.g. "feedforward_tanh")
     * @param layerChild        The layer child of the JSON configuration for this layer
     * @param weightsSection    The weights section of the JSON configuration
     * @param parallelSequences The maximum number of sequences that shall be computed in parallel
     * @param maxSeqLength      The maximum length of a sequence
     * @param precedingLayer    The layer preceding this one
     * @return The constructed layer
     */
    static layers::Layer<TDevice>* createLayer(
            const std::string        &layerType,
	    const helpers::JsonValue &layerChild,
	    const helpers::JsonValue &weightsSection,
	    int                       parallelSequences, 
	    int                       maxSeqLength,
	    int                       layerID,
	    layers::Layer<TDevice>   *precedingLayer = NULL
        );

    static layers::Layer<TDevice>* createSkipNonParaLayer(
            const std::string        &layerType,
	    const helpers::JsonValue &layerChild,
	    const helpers::JsonValue &weightsSection,
	    int                       parallelSequences, 
	    int                       maxSeqLength,
	    int                       layerID,
	    std::vector<layers::Layer<TDevice>*> &precedingLayers
        );

    static layers::Layer<TDevice>* createSkipParaLayer(
            const std::string        &layerType,
	    const helpers::JsonValue &layerChild,
	    const helpers::JsonValue &weightsSection,
	    int                       parallelSequences, 
	    int                       maxSeqLength,
	    int                       layerID,
	    std::vector<layers::Layer<TDevice>*> &precedingLayers
        );

};


#endif // LAYER_FACTORY_HPP
