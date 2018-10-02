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

#ifndef LAYERS_INPUTLAYER_HPP
#define LAYERS_INPUTLAYER_HPP

#include "Layer.hpp"


namespace layers {

    /******************************************************************************************//**
     * Represents the input layer of the neural network
     *
     * @param TDevice The computation device (Cpu or Gpu)
     *********************************************************************************************/
    template <typename TDevice>
    class InputLayer : public Layer<TDevice>
    {
	typedef typename TDevice::real_vector    real_vector;

    private:
	/* Add 16-02-22 Wang: for WE updating */
	/* The vector to store the readWeBank */
	Cpu::real_vector  m_weBank;
	Cpu::real_vector  m_weIdx;
	unsigned int      m_weDim;
	unsigned int      m_weIDDim;
	bool              m_flagWeUpdate;

	
	/* Add 17/01/29 */
	Cpu::real_vector  m_weMask;
	bool              m_weMaskFlag;
	
	/* Add 20160902 Wang: add noise to the WE */
	// Because input_noise_sigma must be turned off when reading the WE index,
	// noise can only be added in the input layer when WE is used
	int               m_weNoiseStartDim;
	int               m_weNoiseEndDim;
	real_t            m_weNoiseDev;
		
    public:
        /**
         * Constructs the Layer
         *
         * @param layerChild        The layer section of the JSON configuration
         * @param parallelSequences The maximum number of sequences that shall be computed 
	 *                          in parallel
         * @param maxSeqLength      The maximum length of a sequence
         */
        InputLayer(const helpers::JsonValue &layerChild,
		   int parallelSequences,
		   int maxSeqLength,
		   int layerID);

	/* Add 16-02-22 Wang: for WE updating */
	bool readWeBank(const std::string weBankPath, const unsigned dim, 
			const unsigned dimidx, const unsigned maxLength);
	
	int  readWeMask(std::vector<real_t>::iterator b);

	void maskWe();
	
	Cpu::real_vector& _weBank();
	
	Cpu::real_vector& _weIdx();
	
	unsigned int&     _weDim();
	
	unsigned int&     _weIDDim();
	
	bool              flagInputWeUpdate();
	
	bool              saveWe(const std::string weFile);

	bool initWeNoiseOpt(const int weNoiseStartDim, const int weNoiseEndDim,
			    const real_t weNoiseDev);

	Cpu::real_vector& _weMask();
	
	bool              flagWeMask();
	
        /**
         * Destructs the Layer
         */
        virtual ~InputLayer();

        /**
         * @see Layer::type()
         */
        virtual const std::string& type() const;

        /**
         * @see Layer::loadSequences()
         */
        virtual void loadSequences(const data_sets::DataSetFraction &fraction, const int nnState);

        /**
         * @see Layer::computeForwardPass()
         */
        virtual void computeForwardPass(const int nnState);

         /**
         * @see Layer::computeBackwardPass()
         */
        virtual void computeBackwardPass(const int nnState);

	/**
	 * Re-initialize the network
	   only defines for Trainable Layers, here do nothing
	 */
	virtual void reInitWeight();

	virtual void computeForwardPass(const int timeStep, const int nnState);
	
    };

} // namespace layers


#endif // LAYERS_INPUTLAYER_HPP
