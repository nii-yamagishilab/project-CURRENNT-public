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

#ifndef RNNLAYER_HPP
#define RNNLAYER_HPP

#include "TrainableLayer.hpp"
#include "../helpers/Matrix.hpp"

namespace layers{

    /************************************************************************
     * the simple recurrent layer 
     * weights: P = preceidingLayer().size() and L =size()
     * Uni-directional case:
     *    ~ weights from preceding layer:
     *      [0 PL-1]: input link 
     *    ~ bias
     *      [PL+0 PL+L-1]: bias
     *    ~ weights from preceding step:
     *      [PL+L+0 PL+L+LL-1]: recurrent link
     *
     * Bi-directional case:
     *    ~ weights from preceding layer:
     *      [0      PL/2-1]: input link forward
     *      [PL/2+0 PL-1]: input link backward
     *    ~ bias
     *      [PL+0     PL+L/2-1]: bias forward
     *      [PL+L/2+0 PL+L-1]: bias backward
     *    ~ weights from preceding step:
     *      [PL+L+0   PL+L+LL/4-1]: recurrent link forward
     *      [PL+L+LL/4+0   PL+L+LL/2-1]: recurrent link backward
     * Other intermediate buffer
     *    ~ a buffer to store the results 
     *      [0 this->_outputs().size()-1]: (size * (parallel*maxLength))
     *    ~ a buffer to store the gradients
     *      [0 this->_outputs().size()-1]: (size * (parallel*maxLength))
     *    ~ a vector mask for ClockRNN
     *      [0 this->size() * parallel]: size * parallel 
     *    ~ matrix mask for ClockRNN
     *      same as the weights from preceding layer (triangle matrix)
     ************************************************************************/

    template <typename TDevice>
    class RnnLayer : public TrainableLayer<TDevice>
    {
	typedef typename TDevice::real_vector real_vector;
	typedef typename TDevice::int_vector  int_vector;
	typedef typename TDevice::bool_vector bool_vector;
	typedef typename TDevice::pattype_vector pattype_vector;
	// Matrix wrappers of the weights
	//  the raw data is stored as vector
	//  wrap it up so that matrix multiplication can be launched
        struct weight_matrices_t {
            helpers::Matrix<TDevice> InputToHiddenWrap;
            helpers::Matrix<TDevice> HiddenToHiddenWrap;
	    helpers::Matrix<TDevice> BiasWrap;
        };

	// Matrix wrappers of intermediate results per time step
        struct timestep_matrices_t {
            helpers::Matrix<TDevice> tmpOutputsWrapT;      // for bi-directional
            helpers::Matrix<TDevice> tmpOutputErrorsWrapT; // for bi-directional
            helpers::Matrix<TDevice> unitActsWrapT;        // activation 
            helpers::Matrix<TDevice> unitDeltasWrapT;      // gradients before activation function
	    helpers::Matrix<TDevice> unitActsBufWrapT;     // 
	    
	    // For ClockRNN
	    helpers::Matrix<TDevice> h2hWrap;        // a simple wrapper for clock RNN in BP
	    // bool_vector              skipCR;      // whether a dimension should be skippped
	    int                      skipCRPos;      // where should I found the skipCR ?
	    int                      h2hIdx;         //
	    int_vector               m_crS;                // 
	    int_vector               m_crE;                //
	    real_t *                 unitDeltaP;
        };

	// Data Structure for easy forward-backward operation
        struct forward_backward_info_t {
            real_vector tmpOutputs;                   // data buffer
            real_vector tmpOutputErrors;              // gradients buffer
            real_vector unitActs;
            real_vector unitDeltas;
	    real_vector unitActsBuf;                  // similar to unitActs, but this buffer
	                                              // stores the result from previous step
	    bool_vector skipCR;                       // 

	    // the wrapper of the same data vector over all the time step,
	    // different from WrapT that wraps for each time step
            helpers::Matrix<TDevice> unitActsWrapA;   // wrap over unitActs, but over all time
            helpers::Matrix<TDevice> unitDeltasWrapA; // wrap over unitDeltas, but over all time

	    // vector of weight wrappers (the weights are defined in TrainableLayer->weights())
            weight_matrices_t                weightMatrices;

	    // vector of weightUpdates wrappers (the weights are defined in TrainableLayer)
            weight_matrices_t                weightUpdateMatrices;

	    // vector of the timestep operator
            std::vector<timestep_matrices_t> timestepMatrices;

        };
	
    private:
        const bool               m_isBidirectional;
        
        real_t                   *_rawBiasWeights;
	
	real_vector              m_onesVec;
	helpers::Matrix<TDevice> m_onesVecWrap;

        forward_backward_info_t  m_fw;
        forward_backward_info_t  m_bw;

	// wrappers over the output of preceding layer
        helpers::Matrix<TDevice> m_precLayerOutputsWrapA;
	
	// For ClockRNN
	bool                     m_clockRNN;        // whether use clock RNN
	std::string              m_crStepStr;       
	Cpu::int_vector          m_crStep;          // a vector of [start1,end1,...,end1,endN]
	int_vector               m_crStepDevice;    //
	real_vector              m_h2hClockRNN;     // duplicated H2H matrices for each possible
	                                            // updating schedule
	int                      m_numH2Hmat;       // number of possible updating schedule

	int                      m_iterUpdate;      //

	// wrappers over the error buffer of preceding layer
	// This wrap is not prepared, because we need to know whether the previous layer
	// is trainable or not
	// helpers::Matrix<TDevice> m_precLayerOutputsErrorWrapA;

    public:
        /**
         * Constructs the Layer
         *
         * @param layerChild     The layer child of the JSON configuration for this layer
         * @param weightsSection The weights section of the JSON configuration
         * @param precedingLayer The layer preceding this one
         * @param bidirectional  Wether the layer should be bidirectional or unidirectional
         */
        RnnLayer(
            const helpers::JsonValue &layerChild, 
            const helpers::JsonValue &weightsSection,
            Layer<TDevice>           &precedingLayer,
	    int                       maxSeqLength,
	    int                       layerID,
            bool                      bidirectional = false
            );

        /**
         * Destructs the Layer
         */
        virtual ~RnnLayer();

        /**
         * @see Layer::type()
         */
        virtual const std::string& type() const;

        /**
         * Returns true if the layer is bidirectional
         *
         * @return True if the layer is bidirectional
         */
        bool isBidirectional() const;

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

         /**
         * @see Layer::computeBackwardPass()
         */
        virtual void computeBackwardPass(const int timeStep, const int nnState);

        /**
         * @see Layer::loadSequences
         */
        virtual void prepareStepGeneration(const int timeStep);

        /**
         * @see Layer::computeForwardPass()
         */
        virtual void computeForwardPass(const int timeStep, const int nnState);

    };

} // namespace layers

#endif //
