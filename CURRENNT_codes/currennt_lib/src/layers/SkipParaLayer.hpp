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

#ifndef LAYERS_SKIPPARALAYER_HPP
#define LAYERS_SKIPPARALAYER_HPP

#include "SkipLayer.hpp"
#include <boost/foreach.hpp>
#include <boost/shared_ptr.hpp>
#include <vector>

namespace layers {
    
    /**********************************************************************
	    Definition of the Skip Add layer
     

     **********************************************************************/
    
    // class definition
    template <typename TDevice, typename TActFn>
    class SkipParaLayer : public SkipLayer<TDevice>
    {
	typedef typename TDevice::real_vector    real_vector;
        
    private:
	// the preceding skipping layer
	//  currently, only one previous skip layer is allowed 
	//  this is different from skipaddlayer, because I don't want to assign another
	//  vector to accumulate the input from multiple previous skip layers
	Layer<TDevice>    *m_preSkipLayer;

	// to receive the errors directly from next skip add layer
	// real_vector       m_outputErrorsFromSkipLayer;

	// the gate 
	real_vector       m_gateOutput;
        // the gate error
	real_vector       m_gateErrors; // error before the actFn of gate unit

	
	// return all the preceding layers
	Layer<TDevice>* preSkipLayer();

    public:
	
	
	// Construct the layer
	SkipParaLayer(
		     const helpers::JsonValue &layerChild,
		     const helpers::JsonValue &weightsSection,
		     std::vector<Layer<TDevice>*> &precedingLayers,
		     int maxSeqLength,
		     int layerID
		     );

	// Destructor
	virtual ~SkipParaLayer();
	
	// void 
	virtual const std::string& type() const;

	// NN forward
	virtual void computeForwardPass(const int nnState);

	// NN forward
	virtual void computeForwardPass(const int timeStep, const int nnState);
	
	// NN backward
	virtual void computeBackwardPass(const int nnState);
	
	// NN forward
	virtual void computeBackwardPass(const int timeStep, const int nnState);

	// Gate output
	real_vector& outputFromGate();
	
       
	// output of the gate unit
	real_vector& gateOutput();
	
	// 
	real_vector& gateErrors();
	
	virtual void clearAllBuffers();

	virtual void resizeAllBuffers(const int timeLength);

    };

}


#endif 


