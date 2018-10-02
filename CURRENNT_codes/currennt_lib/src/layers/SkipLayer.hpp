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
/****
 *
 *
 *
 ****/

#ifndef LAYERS_SKIPLAYER_HPP
#define LAYERS_SKIPLAYER_HPP

#include "TrainableLayer.hpp"
#include <boost/foreach.hpp>
#include <boost/shared_ptr.hpp>
#include <vector>

namespace layers {
    
    /**********************************************************************
	    Definition of the Skip layer
     A base class for SkipAdd and SkipPara layers

     **********************************************************************/
    
    // class definition
    template <typename TDevice>
    class SkipLayer : public TrainableLayer<TDevice>
    {
	typedef typename TDevice::real_vector    real_vector;
        
    private:
	// all the preceding skipping layers
	// std::vector<Layer<TDevice>*> m_preLayers;
	// to receive the errors directly from next skip add layer
	real_vector       m_outputErrorsFromSkipLayer;

	// all the preceding skipping layers
	std::vector<Layer<TDevice>*> m_preLayers;
	// to receive the errors directly from next skip add layer
	// real_vector       m_outputErrorsFromSkipLayer;
        
    public:
	
	
	// Construct the layer
	SkipLayer(const helpers::JsonValue &layerChild,
		  const helpers::JsonValue &weightsSection,
		  std::vector<Layer<TDevice>*> precedingLayers,
		  int maxSeqLength,
		  int layerID,
		  bool trainable);

	// Destructor
	virtual ~SkipLayer();
	
	// void 
	//virtual const std::string& type() const;

	// NN forward
	//virtual void computeForwardPass();
	
	// NN backward
	//virtual void computeBackwardPass();
	
	// return all the preceding layers
	virtual std::vector<Layer<TDevice>*>& PreLayers();
	
	virtual real_vector& outputFromGate();
	
	// return reference to the m_outputErrorsFromSkipLayer
	real_vector& outputErrorsFromSkipLayer();

	// output the layer configuration
	virtual void exportLayer(const helpers::JsonValue &layersArray,
				 const helpers::JsonAllocator &allocator) const;

	virtual std::vector<int> dependLayerIDs();

	virtual void clearAllBuffers();

	virtual void resizeAllBuffers(const int timeLength);

    };

}


#endif 


