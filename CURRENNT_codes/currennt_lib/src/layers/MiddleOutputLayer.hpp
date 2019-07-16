/******************************************************************************
 * This file is an addtional component of CURRENNT. 
 * Xin WANG
 * National Institute of Informatics, Japan
 * 2016
 *
 * This file is part of CURRENNT. 
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
/*
 */

#ifndef LAYERS_MIDDLEOUTPUTLAYER_HPP
#define LAYERS_MIDDLEOUTPUTLAYER_HPP

#include "PostOutputLayer.hpp"


namespace layers {

    /*
      MiddleOutputLayer:
      This layer is exclusively used for General Adversarial Network:
      1. It is the interface between Generator and Discriminator
      2. It loads either natural or generated data sequences
      3. It prepares a target sequence (with 1 or 0) for the discriminator's target layer

      Note: 
      In additiona to MiddleOutputLayer, GAN requires many methods defined in NeuralNetwork,
      PostOutputLayer, and MDNLayer

      GAN criterion:
      (1 - ganRatio) * 0.5 * (generated - natural) ^ 2 + ganRatio * ganGradMag * Loss_discriminator

     */

    
    template <typename TDevice>
    class MiddleOutputLayer : public PostOutputLayer<TDevice>
    {
	typedef typename TDevice::real_vector real_vector;
	typedef typename Cpu::real_vector cpu_real_vector;

	int          m_dataOutputDim;    // dimension of the total output of this layer
	int          m_natPriDim;        // dimension of natural data
	int          m_natSecDim;        // (obsolete)
	
	int          m_state;            // state of this layer: see layerStatus below
	
	real_t       m_ganRatio;         // see GAN criterion above
	real_t       m_ganGradEnhance;   // see GAN criterion above
	
	real_vector  m_natPriTarget;     // buffer to load natural data
	real_vector  m_natSecTarget;     // obsolete
	real_vector  m_stateRandom;      // a sequence of 1/0, which will be used as the target
	                                 //  of discriminator, also used as the flag to
	                                 //  load either natural or generated data as the output
	                                 //  of this layer

	//int          m_generatorEpoch;      // obsolete
	//int          m_discriminatorEpoch;  // obsolete
	
    protected:

	// layerStatus
	//   GENERATOR_ONLY:   one training step where only generator is trained
	//   DISCRMINATOR_ONLY:one training step where only discriminator is trained
	//   JOINT_TRAIN:      (obsolete)
	
	enum layerStatus{
	    GENERATOR_ONLY = 1,    
	    DISCRIMINATOR_ONLY,
	    JOINT_TRAIN,
	    UNKNOWN
	};
	
    public:
	MiddleOutputLayer(const helpers::JsonValue &layerChild, 
			  Layer<TDevice>  &precedingLayer,
			  int maxSeqLength,
			  int layerID);
	
	virtual ~MiddleOutputLayer();

	virtual void loadSequences(const data_sets::DataSetFraction &fraction, const int nnState);

	virtual real_t calculateError();

	virtual const std::string& type() const;
	
	virtual void computeForwardPass(const int nnState);

	virtual void computeForwardPass(const int timeStep, const int nnState);
	
	virtual void computeBackwardPass(const int nnState);

	virtual void computeBackwardPass(const int timeStep, const int nnState);

	/*
	 * secondOutputs: 
	   This function will be the interface which provides the
	   1/0 sequence as the target of the discriminator.
           It will be called by PostOutputLayer->loadSequences()
	 */
	virtual real_vector& secondOutputs();
	
	virtual void exportLayer(const helpers::JsonValue &layersArray, 
				 const helpers::JsonAllocator &allocator) const;

	
    };
}


#endif
