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

#ifndef LAYERS_FEEDBACKHIDDENLAYER_HPP
#define LAYERS_FEEDBACKHIDDENLAYER_HPP

#include "Layer.hpp"
#include "TrainableLayer.hpp"
#include <boost/foreach.hpp>
#include <boost/shared_ptr.hpp>
#include <vector>

namespace layers {
    template <typename TDevice>
    class FeedBackHiddenLayer : public TrainableLayer<TDevice>
    {
	typedef typename TDevice::real_vector     real_vector;
	typedef typename TDevice::int_vector      int_vector;
	typedef typename TDevice::pattype_vector  pattype_vector;
	typedef typename Cpu::int_vector          cpu_int_vector;
	
    private:
	int             m_targetDim;      // dimension of the target data
	int             m_targetDimStart; // the 1st dimension of the target vector to be fed back
	int             m_targetDimEnd;   // the last dim of the target vector to be fed back
	real_vector     m_targetBuffer;   // buffer for the target data
	Layer<TDevice> *m_targetLayer;    // target layer to be fed back

	std::string     m_targetLayerName;

	// configuration of the previous layer
	int             m_prevDimStart;
	int             m_prevDimEnd;
	
    public:
	
	FeedBackHiddenLayer(
	    const helpers::JsonValue &layerChild,
	    const helpers::JsonValue &weightsSection,
            Layer<TDevice>           &precedingLayer,
	    int                       maxSeqLength,
	    int                       layerID
	);

	virtual ~FeedBackHiddenLayer();
	
	// load the target data from the target layer
	void linkTargetLayer(Layer<TDevice> &targetLayer);

	int returnTargetLayerID();
	
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

	// load sequences
	virtual void loadSequences(const data_sets::DataSetFraction &fraction, const int nnState);
    };

}


#endif
