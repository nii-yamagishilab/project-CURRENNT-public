/******************************************************************************
 * This file is an addtional component of CURRENNT. 
 * Copyright (c) 2019 Xin WANG
 * National Institute of Informatics, Japan
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

#ifndef LAYERS_INTERMETRIC_SSE_HPP
#define LAYERS_INTERMETRIC_SSE_HPP

#include "InterMetric.hpp"
#include <boost/foreach.hpp>
#include <boost/shared_ptr.hpp>
#include <vector>

namespace layers {
    
    // class definition
    template <typename TDevice>
    class InterMetricLayer_sse : public InterMetricLayer<TDevice>
    {
	typedef typename TDevice::real_vector    real_vector;
    private:
	//
	
    public:
	
	InterMetricLayer_sse(
            const helpers::JsonValue &layerChild, 
            Layer<TDevice>  &precedingLayer,
	    int maxSeqLength,
	    int layerID
            );

	// Destructor
	virtual ~InterMetricLayer_sse();

	virtual real_t calculateError();

	virtual void computeForwardPass(const int nnState);
	
	virtual void computeForwardPass(const int timeStep, const int nnState);

        virtual const std::string& type() const;
	
	virtual void computeBackwardPass(const int nnState);
	
	virtual void computeBackwardPass(const int timeStep, const int nnState);

	
	// output the layer configuration
	virtual void exportLayer(const helpers::JsonValue &layersArray,
				 const helpers::JsonAllocator &allocator) const;

	virtual void clearAllBuffers();

	virtual void resizeAllBuffers(const int timeLength);


    };

}


#endif 


