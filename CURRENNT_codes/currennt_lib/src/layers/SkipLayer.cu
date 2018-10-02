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


#include "SkipLayer.hpp"
#include "../helpers/getRawPointer.cuh"
#include "../helpers/Matrix.hpp"

namespace layers{

    // Construct the layer
    template <typename TDevice>
    SkipLayer<TDevice>::SkipLayer(const helpers::JsonValue &layerChild,
				  const helpers::JsonValue &weightsSection,
				  std::vector<Layer<TDevice>*> precedingLayers,
				  int maxSeqLength,
				  int layerID,
				  bool trainable)
	// use preLayers[0] as fake preceding layers
	: TrainableLayer<TDevice>(layerChild, weightsSection,
				  (trainable ? 1 : 0), 0, *(precedingLayers.back()),
				  maxSeqLength,
				  layerID)
    {
	if (this->flagTrainingMode())
	    m_outputErrorsFromSkipLayer = Cpu::real_vector(this->outputs().size(), (real_t)0.0);
    }	

    // Destructor
    template <typename TDevice>
    SkipLayer<TDevice>::~SkipLayer()
    {
    }
    
    template <typename TDevice>
    typename SkipLayer<TDevice>::real_vector& SkipLayer<TDevice>::outputFromGate()
    {
	printf("WARNING: Output from SkipLayer, WRONG!\n");
	return m_outputErrorsFromSkipLayer;
    }
    
    template <typename TDevice>
    typename SkipLayer<TDevice>::real_vector& SkipLayer<TDevice>::outputErrorsFromSkipLayer()
    {
        return m_outputErrorsFromSkipLayer;
    }

    // return all the preceding layers
    template <typename TDevice>
    std::vector<Layer<TDevice>*>& SkipLayer<TDevice>::PreLayers()
    {
    	return m_preLayers;
    }
    
    template <typename TDevice>
    void SkipLayer<TDevice>::exportLayer(const helpers::JsonValue &layersArray,
					 const helpers::JsonAllocator &allocator) const
    {
	TrainableLayer<TDevice>::exportLayer(layersArray, allocator);
    }

    template <typename TDevice>
    std::vector<int> SkipLayer<TDevice>::dependLayerIDs()
    {
	std::vector<int> tmp;
	for (size_t layerIdx = 0; layerIdx < m_preLayers.size(); layerIdx++)
	    tmp.push_back(m_preLayers[layerIdx]->getLayerID());
	return tmp;
    }

    template <typename TDevice>
    void SkipLayer<TDevice>::clearAllBuffers()
    {
    }

    template <typename TDevice>
    void SkipLayer<TDevice>::resizeAllBuffers(const int timeLength)
    {
    }    
    
    template class SkipLayer<Cpu>;
    template class SkipLayer<Gpu>;
    
}
