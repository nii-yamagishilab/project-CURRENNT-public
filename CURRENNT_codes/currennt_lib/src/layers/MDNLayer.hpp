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
#ifndef LAYERS_MDNOUTPUTLAYER_HPP
#define LAYERS_MDNOUTPUTLAYER_HPP


#include "PostOutputLayer.hpp"
#include <boost/shared_ptr.hpp>
#include "MDNUnit.hpp"

#define MDN_TYPE_SIGMOID 0
#define MDN_TYPE_SOFTMAX -1

namespace layers {
        
    /********************************************************
     MDNLayer: 
        1. forward computation: transform the output of NN
           into statistical distribution
        2. calculate the likelihood of training data
        3. back-propagation
        4. sampling (predict) output

                    |-  MDNUnit1
        MDNLayer  --|-  MDNUnit2
            ^       |-  MDNUnit3
            |
     PostOutputLayer
    ********************************************************/    
    // MDN layer definition
    template <typename TDevice>
    class MDNLayer : public PostOutputLayer<TDevice>
    {

	typedef typename TDevice::real_vector real_vector;
	typedef typename TDevice::int_vector int_vector;
	typedef typename Cpu::real_vector cpu_real_vector;

    protected:
	cpu_real_vector m_mdnVec;        // the vector of mdnunit flag
	cpu_real_vector m_mdnConfigVec;  // vector of the mdn configuration
	real_vector     m_mdnParaVec;    // vector of parameters of all MDNUnits
	
	
	// the vector of MDNUnit for computation
	std::vector<boost::shared_ptr<MDNUnit<TDevice> > > m_mdnUnits;  
	int m_mdnParaDim;               // the size of MDN parameters
	                                //  is equal to the size of NN output layer
	                                //  (the layer before PostOutputLayer)
	                                // this->size() is the dimension of target features

	bool m_tieVarMDNUnit;           // whether the variance should be tied
	                                // across dimension for each MDNUnit mixture
	                                // model?
	
	real_vector      m_secondOutput;    // for feedback connection	
	Cpu::int_vector  m_secondOutputOpt; // string to control the secondOutput
	int              m_secondOutputDim; 
	
	// parameter for generation
	real_t         m_genPara;

	// Trainable part
	// In case the MDNUnits are trainable
	// note: the weight space can be accessed by all the MDNUnits
	real_vector    m_sharedWeights;       // weight space shared by all the MDNUnits
	real_vector    m_sharedWeightUpdates; // shared space for udpate
	bool           m_trainable;           // whether this layer is trainable
	int            m_trainableNum;        // number of parameters to be trained


	// for probablistic model
	std::string     m_probBiasDir;        //
	int             m_probBiasDim;        //
	cpu_real_vector m_probBiasVec;

	// for uvSigmoid softmax
	std::string    m_uvSigmoidStr;
	std::string    m_quanMergeStr;
	int_vector     m_quanMergeVal;

	// for sigmoid function
	std::string    m_conValSigStr;
	
	// one-sided smoothing
	int            m_oneSidedSmoothing;

	// mdn config path
	std::string    m_mdnConfigPath;      //

	bool           m_flagNewAROption;
	
    public:
	MDNLayer(
		 const helpers::JsonValue &layerChild,
		 const helpers::JsonValue &weightsSection,
		 Layer<TDevice> &precedingLayer,
		 int maxSeqLength,
		 int layerID
		 );
	
	virtual ~MDNLayer();
	
	virtual const std::string& type() const;
	
	virtual real_t calculateError();
	
	virtual void   computeForwardPass(const int nnState);

	virtual void   computeForwardPass(const int timeStep, const int nnState);
	
	virtual void   computeBackwardPass(const int nnState);

	virtual void   computeBackwardPass(const int timeStep, const int nnState);
	
	virtual real_vector& mdnParaVec();
	
	virtual int          mdnParaDim();
	
	virtual void initPreOutput();
	
	virtual cpu_real_vector getMdnConfigVec();

	virtual bool flagTrainable() const;
	
	void getOutput(const real_t para);

	void getOutput(const int timeStep, const real_t para);

	void exportConfig(const helpers::JsonValue &weightsObject, 
			  const helpers::JsonAllocator &allocator) const;

	void exportWeights(const helpers::JsonValue &weightsObject, 
			   const helpers::JsonAllocator &allocator) const;
	

	/* Trainable Part */
	virtual void reReadWeight(const helpers::JsonValue &weightsSection, const int readCtrFlag);
	
	virtual void reInitWeight();

	virtual real_vector& weights();

	const virtual real_vector& weights() const;

        virtual real_vector& weightUpdates();
	
	/**
	 * Set and read the m_currTrainingEpoch
	 */
	virtual void setCurrTrainingEpoch(const int curTrainingEpoch);
	
	virtual int& getCurrTrainingEpoch();
	
	virtual const std::string& layerAddInfor(const int opt) const;

	virtual void loadSequences(const data_sets::DataSetFraction &fraction, const int nnState);

	/* *
	 * Functions to support the feedback data
	 */
	virtual void retrieveFeedBackData();

	virtual void retrieveFeedBackData(real_vector& randNum, const int method);

	virtual void retrieveFeedBackData(const int timeStep, const int method=0);

	virtual real_t retrieveProb(const int timeStep, const int state);
	
	virtual void setFeedBackData(const int timeStep, const int state);
	
	virtual real_vector& feedbackOutputs(const bool flagTrain);

	// export
	virtual void exportLayer(const helpers::JsonValue &layersArray, 
				 const helpers::JsonAllocator &allocator) const;
	
    };

} // namespace layers

#endif //
