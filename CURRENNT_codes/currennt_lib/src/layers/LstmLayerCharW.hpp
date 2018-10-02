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

#ifndef LAYERS_LSTMLAYERCHARW_HPP
#define LAYERS_LSTMLAYERCHARW_HPP

#include "TrainableLayer.hpp"
#include "../helpers/Matrix.hpp"

#include <boost/shared_ptr.hpp>

namespace layers {
    
    /*********************************************************************************
     *
    *********************************************************************************/
    template <typename TDevice>
    class CharW
    {
	typedef typename TDevice::real_vector real_vector;
	typedef typename TDevice::int_vector  int_vector;
	typedef typename Cpu::real_vector cpu_real_vector;
	
    public:
	real_vector      *m_weightPtr;        // pointer to the start of weight Wcc
	real_vector      *m_outputsLstm;      // pointer to the output data of LSTM unit
	real_vector      *m_outErrorsLstm;    // pointer to the output errors of LSTM unit
	real_vector      *m_outputsAll;       // pointer to the [LSTM, CharW] output 
	real_vector      *m_outErrorsAll;     // pointer to the [LSTM, CharW] errors buffer

	real_vector m_rhoVec;            // rho[mixture, position, time]
	                                 // sum_k rho[k, u, t] = phi[u, t]	
	real_vector m_phiVec;            // sum of m_rhoVec
	
	real_vector m_feaVec;            // output after transformation
	real_vector m_chaPara;           // parameter of each mixture [alpha, beta, kappa](input)
	real_vector m_strWord;           // string of the sentence
	real_vector m_strWordRev;
	real_vector m_feaVecError;      
	real_vector m_chaParaError;
	real_vector m_tau_t;             // a small vector to store the intermediate results in BP
	real_vector m_txtBank;

	real_vector m_tmpOne;            // for simplicity, prepare a vector of [1,1,1,1...1]
	real_vector m_rhoErrorBuf;       // used in error propatation [3K, U]
	
	const int m_mixNum;              // mixture number
	const int m_chaDim;              // the size (dimension) of the inpput char data 
	const int m_lstmDim;             // the dimension of the lstm
	const int m_paraNum;             // the number of parameter in each mixture

	int       m_chaMaxLength;        // the maximum length of the string 
	                                 //   (for allocating memory)
	int       m_chaCurLength;        // the length of the currennt string  (<= m_chaMaxLength)
	int       m_seqMaxLength;        // the maximum length of utterance (in frames)
	int       m_seqCurLength;        // the length of the currennt utterance 
	int       m_paraSequence;        // the same as preceding layer
	
	CharW(const int mixNum,  const int chaDim, 
	      const int lstmDim, const int paraNum,
	      const int chaMaxLength, 
	      const int seqMaxLength,
	      const int paraSequence,
	      real_vector *weight,      
	      real_vector *output,         
	      real_vector *outputErrors,
	      real_vector *outputAll,   
	      real_vector *outErrorsAll,
	      const bool blstm);
	
	virtual ~CharW();
	
	// Read in the target string
	virtual void loadSequence(const data_sets::DataSetFraction &fraction);
	
	virtual void computeForwardPass();
	
	virtual void computeBackwardPass();
	
	virtual real_vector& chaInput();

	virtual real_vector& chaOutput();
	
	virtual real_vector& chaInputError();

	virtual real_vector& chaOutputError();
	
	int inSize() const;
	
	int outSize() const;
	
    };

    
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
     *    ~ Add for the CharW, (from output for Lstm to CharW)
     *        
     * @param TDevice The computation device (Cpu or Gpu)
     *********************************************************************************************/
    template <typename TDevice>
    class LstmLayerCharW : public TrainableLayer<TDevice>
    {
        typedef typename TDevice::real_vector real_vector;
	typedef typename Cpu::real_vector cpu_real_vector;
        struct weight_matrices_t {
            helpers::Matrix<TDevice> niInput;
            helpers::Matrix<TDevice> igInput;
            helpers::Matrix<TDevice> fgInput;
            helpers::Matrix<TDevice> ogInput;
            helpers::Matrix<TDevice> niInternal;
            helpers::Matrix<TDevice> igInternal;
            helpers::Matrix<TDevice> fgInternal;
            helpers::Matrix<TDevice> ogInternal;
	    
	    // Matrix for LSTM -> CharW
	    helpers::Matrix<TDevice> lstm2CharW;
	    
	    // Matrices for the CharW -> LSTM (next step)
	    helpers::Matrix<TDevice> niInternalC;
            helpers::Matrix<TDevice> igInternalC;
            helpers::Matrix<TDevice> fgInternalC;
            helpers::Matrix<TDevice> ogInternalC;
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

	    // tmpOutputs from the CharW
	    helpers::Matrix<TDevice> tmpOutputsC;
            helpers::Matrix<TDevice> tmpOutputErrorsC;
	    
	    helpers::Matrix<TDevice> tmpCharPara;
	    helpers::Matrix<TDevice> tmpCharParaError;

	    
	    real_t *tmpRho;
	    int     rhoOffset;

	    real_t *tmpOutputsCPtr;
	    real_t *tmpOutputErrorsCPtr;
	    int     outputsCOffset;

	    real_t *tmpCharParaPtr;
	    real_t *tmpCharParaErrorPtr;
	    int     charParaOffset;
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
	    
	    // tmpOutputs from the CharW
            real_vector tmpOutputsC;
            real_vector tmpOutputErrorsC;

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
	
	// Bias for LSTM -> CharW
	real_t *_rawLSTM2CharWBias;
	
        forward_backward_info_t m_fw;
        forward_backward_info_t m_bw;

        helpers::Matrix<TDevice> m_precLayerOutputsMatrix;
	

	boost::shared_ptr<CharW<TDevice> > m_charW;

	const int m_lstmDim;
	const int m_chaDim;
	const int m_mixNum;
	int       m_lstmWeightNum;
	int       m_inWNumPerBlock;      // to store the number of internal weights

	// Note: The output of CharW should be concatenated with
	//       LSTM output.  outputsAll = [outputFromLSTM, outputFromCharW].
	//       
	real_vector m_outputsAll;
	real_vector m_outErrorsAll;

    public:
        /**
         * Constructs the Layer
         *
         * @param layerChild     The layer child of the JSON configuration for this layer
         * @param weightsSection The weights section of the JSON configuration
         * @param precedingLayer The layer preceding this one
         * @param bidirectional  Wether the layer should be bidirectional or unidirectional
         */
        LstmLayerCharW(
            const helpers::JsonValue &layerChild, 
            const helpers::JsonValue &weightsSection,
            Layer<TDevice>           &precedingLayer,
	    const int                 chaDim,
	    const int                 maxTxtLength,
            bool                      blstm = false
            );

        /**
         * Destructs the Layer
         */
        virtual ~LstmLayerCharW();

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
        virtual void loadSequences(const data_sets::DataSetFraction &fraction);

        /**
         * @see Layer::computeForwardPass()
         */
        virtual void computeForwardPass();

         /**
         * @see Layer::computeBackwardPass()
         */
        virtual void computeBackwardPass();


	virtual int size() const;

	virtual int lstmSize() const;

	// Note: this is overload for LstmLayerCharW
	virtual real_vector& outputs();

	virtual real_vector& outputErrors();
	
	virtual real_vector& outputsLstm();

	virtual real_vector& outputErrorsLstm();
	

	/**
	 * exportLayer
	 */
	virtual void exportLayer(const helpers::JsonValue &layersArray, 
				 const helpers::JsonAllocator &allocator) const;
	
	/**
	 * exportWeight
	 */
	virtual void exportWeights(const helpers::JsonValue &weightsObject, 
				   const helpers::JsonAllocator &allocator) const;
	

    };

} // namespace layers


#endif // LAYERS_LSTMLAYERCHARW_HPP
