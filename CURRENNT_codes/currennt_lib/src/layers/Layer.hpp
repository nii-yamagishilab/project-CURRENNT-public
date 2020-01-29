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

#ifndef LAYERS_LAYER_HPP
#define LAYERS_LAYER_HPP

#include "../Types.hpp"
#include "../data_sets/DataSetFraction.hpp"
#include "../helpers/JsonClassesForward.hpp"
#include "../helpers/vecPoolManager.hpp"

#include <string>


namespace layers {

    /********************************************************************//**
     * Represents a layer in the neural network
     *
     * @param TDevice The computation device (Cpu or Gpu)
     ***********************************************************************/
    template <typename TDevice>
    class Layer
    {
	// declaration of the vector type
        typedef typename TDevice::real_vector    real_vector;
        typedef typename TDevice::pattype_vector pattype_vector;
	typedef typename Cpu::real_vector        cpu_real_vector;
	typedef typename Cpu::pattype_vector     cpu_pattype_vector;
	
    private:

	/**
	 * Fixed attributes of one layer 
	 */
	
	// layer name
        const std::string m_name;

	// layer size
        const int         m_size;
	
	// layer index
	const int         m_layerID;

	// number of utterances in one parallel processing
        const int         m_parallelSequences;

	// maximum utterance length in whole database
        const int         m_maxSeqLength;

	// time resolution (relative to time resolution of 1)
	const int         m_timeResolution;

	
	/**
	 *   Atributes of one layer that can be changed 
	 */
	
	// maximum utterance length in this training "batch"
	int               m_curMaxSeqLength;

	// minimum utterance length in this training "batch"
        int               m_curMinSeqLength;

	// number of utterances in this training "batch"
        int               m_curNumSeqs;

	// a flag used by derived layer type
	int               m_layerMode;        

	// a string to indicate layer sub-type
	std::string       m_layerFlag;

	// index of current training epoch
	int               m_currTrainingEpoch;

	// index of currennt training "batch"
	int               m_currTrainingFrac;

	// flag to indicate the training / generation
	bool              m_flagTrainingMode;

	// flag to indicate whether memory save mode is used
	bool              m_flagSaveOutputMemory;

	// flag to indicate for word-embedding layer (obsolete)
	bool              m_InputWeUpdate; 
	
		
	/**
	 * Data buffer
	 */

	// output of this layer
        real_vector       m_outputs;

	// training error w.r.t output of this layer
        real_vector       m_outputErrors;

	// flag to indicate start/middle/end of this utterance
        pattype_vector    m_patTypes;
	
	// cpu data buffer to copy training error
	cpu_real_vector   m_outputErrorsCopy; 


	/**
	 * pointers 
	 */
	
	// pointer to preceding layer (in *.jsn)
	Layer<TDevice>   *m_precedingLayer;

	// pointer to following layer (in *.jsn)
	Layer<TDevice>   *m_followingLayer;

      
    protected:

	// return the output buffer
        real_vector& _outputs();

	// set m_InputWeUpdate (obsolete)
	bool         _setInputWeUpdate(const bool& flag);
	
    public:
	
        /**
         * Constructs the Layer
         *
         * @param layerChild        The layer child of the JSON configuration for this layer
         * @param parallelSequences The maximum number of sequences computed in parallel
         * @param maxSeqLength      The maximum length of a sequence
         * @param createOutputs     If false, then the outputs vector will be left empty
         */
        Layer(const helpers::JsonValue &layerChild,
	      int   parallelSequences,
	      int   maxSeqLength,
	      bool  flagTrainingMode,
	      int   layerID,
	      Layer<TDevice> *precedingLayer,
	      bool  createOutputs = true
	      );
	
        /**
         * Destructs the Layer
         */
        virtual ~Layer();

        /**
         * Returns the name of the layer
         *
         * @return The name of the layer
         */
        const std::string& name() const;

        /**
         * Returns the number of blocks in the layer
         * 
         * @return The number of blocks in the layer
         */
        virtual int size() const;

        /**
         * Returns the maximum number of sequences that can be computed in parallel
         *
         * @return The maximum number of sequences that can be computed in parallel
         */
        int parallelSequences() const;

        /**
         * Returns the maximum length of a sequence
         *
         * @return The maximum length of a sequence
         */
        int maxSeqLength() const;

        /**
         * Returns the maximum length of the currently loaded sequences
         *
         * @return The maximum length of the currently loaded sequences
         */
        int curMaxSeqLength() const;

        /**
         * Returns the minimum length of the currently loaded sequences
         *
         * @return The minimum length of the currently loaded sequences
         */
        int curMinSeqLength() const;

        /**
         * Returns the number sequences in the current data set fraction
         *
         * @return The number sequences in the current data set fraction
         */
        int curNumSeqs() const;

        /**
         * Calculates the output errors of the layer
         *
         * @return The output error
         */
        virtual real_vector& outputErrors();

	cpu_real_vector& outputErrorsCpu();   

        /**
         * Returns the pattern types vector
         * 
         * @return The pattern types vector
         */
        const pattype_vector& patTypes() const;

        /**
         * Returns a string describing the layer type
         *
         * @return A string describing the layer type
         */
        virtual const std::string& type() const =0;

        /**
         * Returns the outputs of the layer
         *
         * @return The outputs
         */
        virtual real_vector& outputs();

	void copyOutputs(real_vector& dataBuffer);

        /**
         * Loads sequences from a data set
         *
         * @param fraction The fraction of the data set to load
         */
        virtual void loadSequences(const data_sets::DataSetFraction &fraction,
				   const int nnState);

        /**
         * Computes the forward pass
         */
        virtual void computeForwardPass(const int nnState) =0;

        /**
         * Computes the backward pass, including the weight updates
         */
        virtual void computeBackwardPass(const int nnState) =0;
	
        /**
         * Stores the description of the layer in a JSON object
         *
         * @param layersArray The array of layers in the document
         * @param allocator   The allocator to use
         */
        virtual void exportLayer(const helpers::JsonValue &layersArray, 
				 const helpers::JsonAllocator &allocator) const;

	/**
	 * Methods to link target layer (overridden by derived layer classes)
	 * 
	 */
	virtual void linkTargetLayer(Layer<TDevice> &targetLayer);
	
	virtual void linkFollowingLayer(Layer<TDevice> &targetLayer);

	virtual int returnTargetLayerID();
	
	/**
	 * Set and read the m_currTrainingEpoch
	 */
	virtual void setCurrTrainingEpoch(const int curTrainingEpoch);
	
	virtual int& getCurrTrainingEpoch();
	
	virtual void setCurrTrainingFrac(const int curTrainingFrac);
	
	virtual int& getCurrTrainingFrac();

	/*
	 *  Provide additional information
	 */
	virtual const std::string& layerAddInfor(const int opt) const;
	
	/*
	 * Add 1221, prepareStepGeneration: 
	 *  prepare the layer for generating the timeStep-th frame
	 *  
	 */
	virtual void prepareStepGeneration(const int timeStep);
	
	virtual void computeForwardPass(const int timeStep, const int nnState)=0;

	virtual void computeBackwardPass(const int timeStep, const int nnState)=0;

	virtual real_vector& feedbackOutputs(const bool flagTrain);

	virtual void cleanGradidents();


	/*
	 * Layer state
	 */
	// return the number of elements in internal state
	virtual int hiddenStateSize();
	
	// retreve the hidden state
	virtual void retrieveHiddenState(const int timeStep, real_vector& readBuffer);

	// set the hidden state
	virtual void setHiddenState(const int timeStep, real_vector& writeBuffer);

	
	/*
	 * To optimize the memory usage in compuuteForwardPass(const int timeStep)
	 */
	bool flagTrainingMode() const;
	
	const int& getResolution();

	virtual const std::string& getLayerFlag();

	const int& getLayerID();

	void setLayerMode(const int mode);

	int getLayerMode();
	
	/**
         * Returns the preceding layer
         *
         * @return The preceding layer
         */
        Layer<TDevice>& precedingLayer();
        Layer<TDevice>& followingLayer();
	
        /**
         * Returns the preceding layer
         *
         * @return The preceding layer
         */
        const Layer<TDevice>& precedingLayer() const;
        const Layer<TDevice>& followingLayer() const;

	/*
	 * Return the ID of layers that this layer depends on
	 */
	virtual std::vector<int> dependLayerIDs();
	
	/*
	 * Returns the error calculated in hidden layers
	 */
	virtual real_t intermediateError();

	/** 
	 * Memory management
	 */
	// functions for auto-regressive generation
	virtual void reduceOutputBuffer();

	virtual int  outputBufPtrBias(const int timeStepTimesParallel,
				      const int nnState);

	// functions for layer-by-layer memory save generation
	virtual void clearAllBuffers();

	virtual void resizeAllBuffers(const int timeLength);
	
	void clearOutputBuffer();

	void resizeOutputBuffer(const int bufferSize);
	
	void setSaveMemoryFlag(const bool newFlag);
	
	bool getSaveMemoryFlag() const;
	
	virtual void logAllBuffers(helpers::vecPoolManager<TDevice> &vecPoolMng,
				   bool flag_add);
	
	virtual void swapAllBuffers(helpers::vecPoolManager<TDevice> &vecPoolMng,
				    bool flag_get);

	/**
	 * For word-embedding layers (obsolete)
	 */
	virtual void reInitWeight() = 0;
	bool&        inputWeUpdate();
	
    };

} // namespace layers


#endif // LAYERS_LAYER_HPP
