/******************************************************************************
 * Copyright (c) 2013 Johannes Bergmann, Felix Weninger, Bjoern Schuller
 * Institute for Human-Machine Communication
 * Technische Universitaet Muenchen (TUM)
 * D-80290 Munich, Germany
 *
 * This file is part of CURRENNT.
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

#ifndef LAYERS_PARALAYER_HPP
#define LAYERS_PARALAYER_HPP

#include "FeedForwardLayer.hpp"


namespace layers {

    /******************************************************************************************//**
     * Represents a feed forward layer in the neural network
     *
     * @param TDevice The computation device (Cpu or Gpu)
     * @param TActFn  The activation function to use
     *********************************************************************************************/
    template <typename TDevice, typename TActFn>
    class ParaLayer : public FeedForwardLayer<TDevice, TActFn>
    {
	typedef typename TDevice::real_vector real_vector;
	typedef typename TDevice::int_vector  int_vector;
	typedef typename TDevice::bool_vector bool_vector;
	typedef typename TDevice::pattype_vector pattype_vector;
    private:
	std::string              m_paraConStr;    // configuration of the parametric form
	Cpu::int_vector          m_paraConfig;    // configuration of the parametric form
	int_vector               m_paraConfigDev; // 
	pattype_vector           m_timeStep;      // 
	int_vector               m_paraConfig2;   // another representation of the parameter config
		
	std::string              m_crStepStr;     // the same as in Clock Rnn Layer
	Cpu::int_vector          m_crStep;        // a vector of [start1,...,endN]
	int_vector               m_crStepDevice;
	real_vector               m_relativeTime;  // the relative time from the boundary
	real_vector              m_weightMask;    // the weight mask for parameter function
	bool                     m_brnn;          // is the preceding layer a bidirecitonal RNN?
	
    public:
        /**
         * Constructs the Layer
         *
         * @param layerChild     The layer child of the JSON configuration for this layer
         * @param weightsSection The weights section of the JSON configuration
         * @param precedingLayer The layer preceding this one
         */
        ParaLayer(
            const helpers::JsonValue &layerChild, 
            const helpers::JsonValue &weightsSection,
            Layer<TDevice>           &precedingLayer,
	    int                       maxSeqLength,
	    int                       layerID
            );

        /**
         * Destructs the Layer
         */
        virtual ~ParaLayer();

        /**
         * @see Layer::type()
         */
        virtual const std::string& type() const;
	
        /**
         * @see Layer::loadSequences
         */
        virtual void loadSequences(const data_sets::DataSetFraction &fraction, const int nnState);
	
        /**
         * @see Layer::computeForwardPass()
         */
        virtual void computeForwardPass(const int nnState);
	
	/**
         * @see Layer::exportLayer()
         */
	virtual void exportLayer(const helpers::JsonValue &layersArray, 
				 const helpers::JsonAllocator &allocator) const;

         /**
         * @see Layer::computeBackwardPass()
         */
        virtual void computeBackwardPass(const int nnState);

	virtual void computeBackwardPass(const int timeStep, const int nnState);
	/**
	 * @see Layer::computeForwardPass()
         */
        virtual void computeForwardPass(const int timeStep, const int nnState);


    };

} // namespace layers


#endif // LAYERS_FEEDFORWARDLAYER_HPP
