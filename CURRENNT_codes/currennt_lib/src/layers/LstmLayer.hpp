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

#ifndef LAYERS_LSTMLAYER_HPP
#define LAYERS_LSTMLAYER_HPP

#include "TrainableLayer.hpp"
#include "../helpers/Matrix.hpp"


namespace layers {

    /******************************************************************************************//**
     * Represents a fully connected layer which uses LSTM cells with forget gates, peephole
     * connections and one cell per block
     *
     * weights; with P = precedingLayer().size() and L = size():
     *    ~ weights from preceding layer:
     *        - [0 .. PL-1]:    net input
     *        - [PL .. 2PL-1]:  input gate
     *        - [2PL .. 3PL-1]: forget gate
     *        - [3PL .. 4PL-1]: output gate
     *    ~ bias weights:
     *        - [4PL + 0  .. 4PL + L-1]:  net input
     *        - [4PL + L  .. 4PL + 2L-1]: input gate
     *        - [4PL + 2L .. 4PL + 3L-1]: forget gate
     *        - [4PL + 3L .. 4PL + 4L-1]: output gate
     *    ~ internal weights (from other cells in the same layer):
     *        - [4(P+1)L + 0   .. 4(P+1)L + LL-1]:  net input
     *        - [4(P+1)L + LL  .. 4(P+1)L + 2LL-1]: input gate
     *        - [4(P+1)L + 2LL .. 4(P+1)L + 3LL-1]: forget gate
     *        - [4(P+1)L + 3LL .. 4(P+1)L + 4LL-1]: output gate
     *    ~ peephole weights (from cell state to all gates in the same cell):
     *        - [4(P+1+L)L + 0   .. 4(P+1+L)L + L-1]:  input gate
     *        - [4(P+1+L)L + LL  .. 4(P+1+L)L + 2L-1]: forget gate
     *        - [4(P+1+L)L + 2LL .. 4(P+1+L)L + 3L-1]: output gate
     *
     * @param TDevice The computation device (Cpu or Gpu)
     *********************************************************************************************/
    template <typename TDevice>
    class LstmLayer : public TrainableLayer<TDevice>
    {
        typedef typename TDevice::real_vector real_vector;
	typedef typename TDevice::int_vector  int_vector;
	typedef typename TDevice::bool_vector bool_vector;
	typedef typename TDevice::pattype_vector pattype_vector;
	
        struct weight_matrices_t {
            helpers::Matrix<TDevice> niInput;
            helpers::Matrix<TDevice> igInput;
            helpers::Matrix<TDevice> fgInput;
            helpers::Matrix<TDevice> ogInput;
            helpers::Matrix<TDevice> niInternal;
            helpers::Matrix<TDevice> igInternal;
            helpers::Matrix<TDevice> fgInternal;
            helpers::Matrix<TDevice> ogInternal;
        };

        struct timestep_matrices_t {
            helpers::Matrix<TDevice> tmpOutputs;
            helpers::Matrix<TDevice> tmpOutputErrors;
            helpers::Matrix<TDevice> niActs;
            helpers::Matrix<TDevice> igActs;
            helpers::Matrix<TDevice> fgActs;
            helpers::Matrix<TDevice> ogActs;
            helpers::Matrix<TDevice> niDeltas;
            helpers::Matrix<TDevice> igDeltas;
            helpers::Matrix<TDevice> fgDeltas;
            helpers::Matrix<TDevice> ogDeltas;

	    // For Clock LSTM
	    helpers::Matrix<TDevice> niH2HWrap;  // wrapper for h2input gate matrix 
            helpers::Matrix<TDevice> igH2HWrap;  // ... 
            helpers::Matrix<TDevice> fgH2HWrap;  // ...
            helpers::Matrix<TDevice> ogH2HWrap;
            int                      skipCRPos;  // offset to find the skip vector of current step
	    int                      h2hIdx;     // the index to retrieve the matrix of current step
        };

        struct forward_backward_info_t {
            real_vector tmpOutputs;
            real_vector tmpOutputErrors;
            real_vector cellStates;
            real_vector cellStateErrors;
            real_vector niActs;
            real_vector igActs;
            real_vector fgActs;
            real_vector ogActs;
            real_vector niDeltas;
            real_vector igDeltas;
            real_vector fgDeltas;
            real_vector ogDeltas;

            helpers::Matrix<TDevice> niActsMatrix;
            helpers::Matrix<TDevice> igActsMatrix;
            helpers::Matrix<TDevice> fgActsMatrix;
            helpers::Matrix<TDevice> ogActsMatrix;
            helpers::Matrix<TDevice> niDeltasMatrix;
            helpers::Matrix<TDevice> igDeltasMatrix;
            helpers::Matrix<TDevice> fgDeltasMatrix;
            helpers::Matrix<TDevice> ogDeltasMatrix;

            weight_matrices_t                weightMatrices;
            weight_matrices_t                weightUpdateMatrices;
            std::vector<timestep_matrices_t> timestepMatrices;

	    bool_vector skipCR;   // the vector to specify the skipping
	    
        };

    private:
        const bool m_isBidirectional;
        
        real_t *_rawNiBiasWeights;
        real_t *_rawIgBiasWeights;
        real_t *_rawFgBiasWeights;
        real_t *_rawOgBiasWeights;
        real_t *_rawIgPeepholeWeights;
        real_t *_rawFgPeepholeWeights;
        real_t *_rawOgPeepholeWeights;

        forward_backward_info_t m_fw;
        forward_backward_info_t m_bw;

        helpers::Matrix<TDevice> m_precLayerOutputsMatrix;

	// For CLLSTM
	bool                     m_clockRNN;        // whether use clock LSTM
	std::string              m_crStepStr;       //
	Cpu::int_vector          m_crStep;          // a vector of [start1,end1,...,startN,endN]
	int_vector               m_crStepDevice;    //
	real_vector              m_h2hClockRNN;     // for hidden to hidden link
	int                      m_numH2Hmat;       // number of possible Clock updating schedule

	
    public:
        /**
         * Constructs the Layer
         *
         * @param layerChild     The layer child of the JSON configuration for this layer
         * @param weightsSection The weights section of the JSON configuration
         * @param precedingLayer The layer preceding this one
         * @param bidirectional  Wether the layer should be bidirectional or unidirectional
         */
        LstmLayer(
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
        virtual ~LstmLayer();

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
         * Returns the cell states
         *
         * @return The cell states
         */
        const real_vector& cellStates() const;

        /**
         * Returns the cell state errors
         *
         * @return The cell state errors
         */
        const real_vector& cellStateErrors() const;

        /**
         * Returns the net input activations
         *
         * @return The net input activations
         */
        const real_vector& netInputActs() const;

        /**
         * Returns the net input activation deltas
         *
         * @return The net input activation deltas
         */
        const real_vector& netInputDeltas() const;

        /**
         * Returns the input gate activations
         *
         * @return The input gate activations
         */
        const real_vector& inputGateActs() const;

        /**
         * Returns the input gate deltas
         *
         * @return The input gate deltas
         */
        const real_vector& inputGateDeltas() const;

        /**
         * Returns the forget gate activations
         *
         * @return The forget gate activations
         */
        const real_vector& forgetGateActs() const;

        /**
         * Returns the forget gate deltas
         *
         * @return The forget gate deltas
         */
        const real_vector& forgetGateDeltas() const;

        /**
         * Returns the output gate activations
         *
         * @return The output gate activations
         */
        const real_vector& outputGateActs() const;

        /**
         * Returns the output gate deltas
         *
         * @return The output gate deltas
         */
        const real_vector& outputGateDeltas() const;

        /**
         * @see Layer::loadSequences
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


	/*
	* NN forward per frame
	*/
	virtual void computeForwardPass(const int timeStep, const int nnState);

	virtual void prepareStepGeneration(const int timeStep);

	/**
         * @see Layer::exportLayer()
         */
	virtual void exportLayer(const helpers::JsonValue &layersArray, 
				 const helpers::JsonAllocator &allocator) const;


	/*
	 * 
	 */
	virtual int hiddenStateSize();
	
	// retreve the hidden state
	virtual void retrieveHiddenState(const int timeStep, real_vector& readBuffer);

	// set the hidden state
	virtual void setHiddenState(const int timeStep, real_vector& writeBuffer);

	
    };

} // namespace layers


#endif // LAYERS_LSTMLAYER_HPP
