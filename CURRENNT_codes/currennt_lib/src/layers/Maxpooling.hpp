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

#ifndef LAYERS_MAXPOOLING_HPP
#define LAYERS_MAXPOOLING_HPP

#include "TrainableLayer.hpp"

namespace layers {

    template <typename TDevice>
    class MaxPoolingLayer : public TrainableLayer<TDevice>
    {
	typedef typename TDevice::real_vector real_vector;
	typedef typename TDevice::int_vector  int_vector;
	typedef typename Cpu::int_vector      cpu_int_vector;
	typedef typename TDevice::bool_vector bool_vector;
	typedef typename TDevice::pattype_vector pattype_vector;

	std::string    m_width;
	int_vector     m_width_D;
	cpu_int_vector m_width_H;
	
	std::string    m_stride;
	int_vector     m_stride_D;
	cpu_int_vector m_stride_H;

	int            m_maxWidth;
	int_vector m_maxPos;

    public:
	MaxPoolingLayer(
			const helpers::JsonValue &layerChild,
			const helpers::JsonValue &weightsSection,
			Layer<TDevice>           &precedingLayer,
			int                       maxSeqLength,
			int                       layerID);
	
	virtual ~MaxPoolingLayer();

	virtual const std::string& type() const;

	virtual void computeForwardPass(const int nnState);

	virtual void computeBackwardPass(const int nnState);

	virtual void computeForwardPass(const int timeStep, const int nnState);

	virtual void exportLayer(const helpers::JsonValue     &layersArray,
				 const helpers::JsonAllocator &allocator) const;
	
    };

}
#endif
