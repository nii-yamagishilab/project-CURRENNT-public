/******************************************************************************
 * This file is an addtional component of CURRENNT. 
 * Xin WANG
 * National Institute of Informatics, Japan
 * 2019
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

#ifndef LAYERS_SPECIALFEEDBACK_LAYER_HPP
#define LAYERS_SPECIALFEEDBACK_LAYER_HPP

#include "TrainableLayer.hpp"
#include <boost/foreach.hpp>
#include <boost/shared_ptr.hpp>
#include <vector>

namespace layers {
    
    /*
     * Special feedback layer
     *
     * Compared with FeedBackLayer, this special feedback layer is not for
     * autoregressive models. It only takes the network's target as output
     * of this layer during training.
     * 
     * The motivation is to simplify the network construction for networks
     * similar to auto-encoder
     */
    
    // class definition
    template <typename TDevice>
    class SpecialFeedBackLayer : public TrainableLayer<TDevice>
    {
	typedef typename TDevice::real_vector    real_vector;

    private:
	// the name of the layer that gives its output as the input of this layer
	std::string        m_feedback_src_str;
	Layer<TDevice>*    m_feedback_src_ptr;

	
	
    public:
		
	// Construct the layer
	SpecialFeedBackLayer(const helpers::JsonValue &layerChild,
			     const helpers::JsonValue &weightsSection,
			     Layer<TDevice>           &precedingLayer,
			     int                       maxSeqLength,
			     int                       layerID);

	// Destructor
	virtual ~SpecialFeedBackLayer();

	void linkTargetLayer(Layer<TDevice> &targetLayer);
	
	virtual const std::string& type() const;
	
	// NN forward
	virtual void computeForwardPass(const int nnState);
	
	// NN forward, per frame
	virtual void computeForwardPass(const int timeStep, const int nnState);
	
	// NN backward
	virtual void computeBackwardPass(const int nnState);

	// NN backward
	virtual void computeBackwardPass(const int timeStep, const int nnState);

	// export
	virtual void exportLayer(const helpers::JsonValue &layersArray, 
				 const helpers::JsonAllocator &allocator) const;

    };

}


#endif 


