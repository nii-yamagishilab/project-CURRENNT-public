/******************************************************************************
 * This file is an addtional component of CURRENNT. 
 * Xin WANG
 * National Institute of Informatics, Japan
 * 2019
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

#ifndef LAYERS_RANDOMSHUFFLE_HPP
#define LAYERS_RANDOMSHUFFLE_HPP

#include "SkipLayer.hpp"
#include <boost/foreach.hpp>
#include <boost/shared_ptr.hpp>
#include <vector>

namespace layers {
    
    
    // class definition
    template <typename TDevice>
    class RandomShuffleLayer : public SkipLayer <TDevice>
    {
	typedef typename TDevice::real_vector real_vector;
        
    private:

	std::string                      m_preShuffleLayerName;  // name of previoius shuffle layer
	RandomShuffleLayer<TDevice>     *m_preShuffleLayer;     

	real_vector                      m_dataIndex;            // index of data after shuffling
	real_vector                      m_randomValue;          // random numbers for shuffling
	real_vector                      m_dataIndexRev;         // index to recover from shuffling

    protected:
       
	real_vector& __dataIndex();
	real_vector& __dataIndexRev();
	
    public:
	
	// Construct the layer
	RandomShuffleLayer(const helpers::JsonValue &layerChild,
			   const helpers::JsonValue &weightsSection,
			   std::vector<Layer<TDevice>*> &precedingLayers,
			   int maxSeqLength,
			   int layerID);

	// Destructor
	virtual ~RandomShuffleLayer();
	
	// void 
	virtual const std::string& type() const;

		// NN forward
	virtual void computeForwardPass(const int nnState);

	// NN forward
	virtual void computeForwardPass(const int timeStep, const int nnState);
	
	// NN backward
	virtual void computeBackwardPass(const int nnState);

	// NN backward
	virtual void computeBackwardPass(const int timeStep, const int nnState);
		
	// output the layer configuration
	virtual void exportLayer(const helpers::JsonValue &layersArray,
				 const helpers::JsonAllocator &allocator) const;

	virtual std::vector<int> dependLayerIDs();
	
	virtual void clearAllBuffers();

	virtual void resizeAllBuffers(const int timeLength);

    };

}


#endif 


