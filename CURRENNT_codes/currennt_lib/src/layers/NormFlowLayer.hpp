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

#ifndef LAYERS_NORMFLOW_HPP
#define LAYERS_NORMFLOW_HPP

#include "SkipLayer.hpp"
#include <boost/foreach.hpp>
#include <boost/shared_ptr.hpp>
#include <vector>

namespace layers {
    
    /**********************************************************************
     Normalization flow layer

     **********************************************************************/
    
    // class definition
    template <typename TDevice>
    class NormFlowLayer : public SkipLayer<TDevice>
    {
	typedef typename TDevice::real_vector     real_vector;
	typedef typename Cpu::real_vector         cpu_real_vector;
	typedef typename TDevice::int_vector      int_vector;
	typedef typename TDevice::pattype_vector  pattype_vector;
	typedef typename Cpu::int_vector          cpu_int_vector;
        
    private:
	Layer<TDevice>    *m_targetLayer;   // target layer (where likelihood is evaluated)
	NormFlowLayer<TDevice>    *m_preFlowLayer;  // previous norm flow layer

	std::string        m_previousFlowStr;  //
	std::string        m_skipDim;          // 
	bool               m_initFlowFlag;
	bool               m_lastFlowFlag;
	int                m_flowDirection;    //  
	
	real_vector        m_paraBuffer;
	real_vector        m_tempOutput;  
	
	int_vector         m_skipFlag;         // not all target feature dims should be changed
	int_vector         m_gradPtr;          // pointers to collect gradients from MDN
	int_vector         m_seqLength;        // length of utterance in frac
	cpu_int_vector     m_seqLengthCpu;     // length of utterance in frac
	
	int                m_normType;         // 1: default: (1-sig(std)) * mu + sig(std) * o
	int                m_frameNum;
	real_t             m_likelihoodJac;    // likelihood for the Jacobian part

    public:
	
	
	// Construct the layer
	NormFlowLayer(const helpers::JsonValue &layerChild,
		      const helpers::JsonValue &weightsSection,
		      std::vector<Layer<TDevice>*> &precedingLayers,
		      int maxSeqLength,
		      int layerID);

	// Destructor
	virtual ~NormFlowLayer();
	
	// void 
	virtual const std::string& type() const;

	// NN forward
	virtual void computeForwardPass(const int nnState);

	// NN forward
	virtual void computeForwardPass(const int timeStep, const int nnState);
	
	// NN backward
	virtual void computeBackwardPass(const int nnState);

	// load the target data from the target layer
	void linkTargetLayer(Layer<TDevice> &targetLayer);

	const bool& flagInitFlow();

	const bool& flagLastFlow();
	
	// output the layer configuration
	virtual void exportLayer(const helpers::JsonValue &layersArray,
				 const helpers::JsonAllocator &allocator) const;

	// load sequences
        virtual void loadSequences(const data_sets::DataSetFraction &fraction, const int nnState);

	// set flag for the last flow
	void noticeLastFlow();

	// return the likelihood of jacobian part
	real_t likelihoodJac();

	// load the generated results from MDN
	void loadNormedOutput();

	// initialize the previous layer
	void initPreOutput();

	// direction of the dependency
	int flagCausalDependency() const;

	bool flagChangeFlowDirection();
	
    };

}


#endif 


