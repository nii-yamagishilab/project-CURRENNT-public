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

#ifndef LAYERS_STRUCT_TRANSFORM_HPP
#define LAYERS_STRUCT_TRANSFORM_HPP

#include "SkipLayer.hpp"
#include <boost/foreach.hpp>
#include <boost/shared_ptr.hpp>
#include <vector>

namespace layers {
    
    /**********************************************************************
     Structured transformation layer

     **********************************************************************/
    
    // class definition
    template <typename TDevice>
    class StructTransLayer : public SkipLayer<TDevice>
    {
	typedef typename TDevice::real_vector     real_vector;
	typedef typename Cpu::real_vector         cpu_real_vector;
	typedef typename TDevice::int_vector      int_vector;
	typedef typename TDevice::pattype_vector  pattype_vector;
	typedef typename Cpu::int_vector          cpu_int_vector;
        
    private:
	int                m_structTransType;  // which type of transformation is
	int                m_structReverse;    // 
	std::string        m_previousSkipStr;  //
	
	int_vector         m_seqLength;        // length of utterance in frac
	cpu_int_vector     m_seqLengthCpu;     // length of utterance in frac

	real_vector        m_tempOutput;

	void __allocateLocalMem();
	void __clearLocalMem();
	
    public:
	
	
	// Construct the layer
	StructTransLayer(const helpers::JsonValue &layerChild,
			 const helpers::JsonValue &weightsSection,
			 std::vector<Layer<TDevice>*> &precedingLayers,
			 int maxSeqLength,
			 int layerID);

	// Destructor
	virtual ~StructTransLayer();
	
	// void 
	virtual const std::string& type() const;

	// NN forward
	virtual void computeForwardPass(const int nnState);

	// NN forward
	virtual void computeForwardPass(const int timeStep, const int nnState);
	
	// NN backward
	virtual void computeBackwardPass(const int nnState);

	
	// output the layer configuration
	virtual void exportLayer(const helpers::JsonValue &layersArray,
				 const helpers::JsonAllocator &allocator) const;

	virtual void reduceOutputBuffer();

	virtual int outputBufPtrBias(const int timeStepTimesParallel, const int nnState);

	void clearAllBuffers();

	void resizeAllBuffers(const int timeLength);

	// load sequences
        virtual void loadSequences(const data_sets::DataSetFraction &fraction, const int nnState);

	
    };

}


#endif 


