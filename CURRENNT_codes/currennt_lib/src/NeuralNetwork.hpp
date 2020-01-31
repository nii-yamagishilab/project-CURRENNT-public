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

#ifndef NEURALNETWORK_HPP
#define NEURALNETWORK_HPP

#include "layers/InputLayer.hpp"
#include "layers/TrainableLayer.hpp"
#include "layers/PostOutputLayer.hpp"
#include "layers/MDNLayer.hpp"
#include "layers/SkipLayer.hpp"
#include "layers/vqLayer.hpp"

#include "data_sets/DataSet.hpp"

#include "helpers/JsonClassesForward.hpp"
#include "helpers/layerDependency.hpp"
#include "helpers/vecPoolManager.hpp"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <boost/shared_ptr.hpp>

#include <vector>
#include <memory>


/************************************************************************//**
 * Represents the neural network
 *
 * @param TDevice The computation device
 ***************************************************************************/


template <typename TDevice>
class NeuralNetwork
{
    typedef typename TDevice::real_vector real_vector;
    typedef typename Cpu::real_vector cpu_real_vector;

private:
    // vector of layer pointer
    std::vector<boost::shared_ptr<layers::Layer<TDevice> > > m_layers;

    // vector of skip layers pointers
    std::vector<layers::Layer<TDevice>*> m_skipAddLayers;

    // vector of feedback layer index
    std::vector<int> m_feedBackLayers;

    // vector of vae or vqvae layer index
    std::vector<int> m_vaeLayers;

    // vector of normflow layer index
    std::vector<int> m_normflowLayers;

    // vector of distilling layer index
    std::vector<int> m_distillingLayers;

    // vector of signalgen layer index 
    std::vector<int> m_signalGenLayerId;

    // vector of featureTransform (feattrans featsse) layer index
    std::vector<int> m_featTransNetRange;

    // vector of feedback_hidden layer index
    std::vector<int> m_feedBackHiddenLayers;

    // vector of feedback_hidden layer time resolutions
    std::vector<int> m_feedBackHiddenLayersTimeResos;

    // vector of InterMetric layers
    std::vector<int> m_interMetricLayers;

    // vector of special feedback layers
    std::vector<int> m_specialFeedbackLayers;
    
    // vector of temporary layer index used in AR dependency building
    std::vector<int> m_tmpLayerIdx;

    int m_firstFeedBackLayer;
    int m_middlePostOutputLayer;
    int m_featMatchLayer;
    int m_vaeLayer;
    
    int m_vaeNetworkType;
    int m_trainingEpoch;
    int m_trainingFrac;
    int m_trainingState;

    int m_waveNetMemSaveFlag;
    int m_totalNumLayers;

    int m_wavNetCoreFirstIdx;
    int m_dftLayerIdx;

    int m_interWeaveIdx;

    // network dependency/topology mangager
    helpers::networkDepMng m_networkMng;

    // network memory manager (only for memory-save mode)
    helpers::vecPoolManager<TDevice> m_vecPoolMng;
    
    // initialize parameters
    void __InitializeParameters();
    
    // initialize the layer indices by readong jsonDoc
    void __InitializeNetworkLayerIdx(const helpers::JsonDocument &jsonDoc);

    // create the network structure
    void __CreateNetworkLayers(const helpers::JsonDocument &jsonDoc,
			       int parallelSequences,  int maxSeqLength, int inputSizeOverride);
    
    // check the created layers
    void __CheckNetworkLayers();
    
    // link the layers 
    void __LinkNetworkLayers();

    // create the dependency map
    void __CreateDependency();

    // other optional initializations
    void __MiscInitialization(const helpers::JsonDocument &jsonDoc,
			      int parallelSequences,  int maxSeqLength);
    
    // Training forward parts:
    //  layer by layer (normal mode)
    void __computeForward_LayerByLayer(const int curMaxSeqLength, const real_t uttCnt);
    
    //  layer by layer (for AR model with teacher forcing training)
    void __computeForward_TeacherForce_LayerByLayer(const int curMaxSeqLength,
						    const real_t uttCnt);
    
    //  step by step (for AR model with schedule sampling)
    void __computeForward_ScheduleSamp_LayerByLayer(const int curMaxSeqLength,
						    const real_t uttCnt);
    
    //  step by step (for RNN network with feedback among hidden layers)
    void __computeForward_StepByStep(const int curMaxSeqLength, const real_t uttCnt);

    // Training backward parts:
    //  layer by layer (normal mode)
    //  note: the first three modes above use this mode for backward propagation.
    //         Although Schedule sampling should have used backward_stepbystep 
    void __computeBackward_LayerByLayer(const int curMaxSeqLength, const real_t uttCnt);
    //  step by step (for RNN with feedback among hidden layers)
    void __computeBackward_StepByStep(const int curMaxSeqLength, const real_t uttCnt);

    
    // Generation parts:
    //  layer by layer (normal mode)
    void __computeGenPass_LayerByLayer(const data_sets::DataSetFraction &fraction,
				       const int curMaxSeqLength, const real_t generationOpt);
    
    //  normalization flow (for autoregressive flow, NOT inverse autoregressive flow)
    void __computeGenPass_NormFlow(const data_sets::DataSetFraction &fraction,
				   const int curMaxSeqLength, const real_t generationOpt);
    //  layer by layer with memory release/allocation per layer (for NSF)
    void __computeGenPass_LayerByLayer_mem(const data_sets::DataSetFraction &fraction,
					   const int curMaxSeqLength, const real_t generationOpt);

    //  layer by layer for VAE network
    void __computeGenPass_VAE(const data_sets::DataSetFraction &fraction,
			      const int curMaxSeqLength, const real_t generationOpt);

    void __computeGenPass_VAEwithAR(const data_sets::DataSetFraction &fraction,
				    const int curMaxSeqLength, const real_t generationOpt);

    void __computeGenPass_VAEwithMA(const data_sets::DataSetFraction &fraction,
				    const int curMaxSeqLength, const real_t generationOpt);

    // for auto-encoder network
    void __computeGenPass_AE(const data_sets::DataSetFraction &fraction,
			     const int curMaxSeqLength, const real_t generationOpt);

    //  step by step (for all types of AR model)
    void __computeGenPass_StepByStep_AR(const data_sets::DataSetFraction &fraction,
					const int curMaxSeqLength, const real_t generationOpt);
    
    //  step by step (for all types of RNN with feedback in hidden layers)
    void __computeGenPass_StepByStep_RNN_FBH(const data_sets::DataSetFraction &fraction,
					     const int curMaxSeqLength,
					     const real_t generationOpt);

    //  special mode for NSF with FBH
    void __computeGenPass_special_NSF_FBH(const data_sets::DataSetFraction &fraction,
					  const int curMaxSeqLength,
					  const real_t generationOpt);

    // Simple methods
    bool __stopBackPropagation(const int layerID, const int runningMode);
    
public:
    /**
     * Creates the neural network from the process configuration
     *
     * @param jsonDoc           The JSON document containing the network configuration
     * @param parallelSequences The maximum number of sequences in parallel
     * @param maxSeqLength      The maximum length of a sequence
     */
    NeuralNetwork(const helpers::JsonDocument &jsonDoc,
		  int parallelSequences, 
		  int maxSeqLength,
		  int inputSizeOverride=-1,
		  int outputSizeOverride=-1);

    /**
     * Destructs the neural network
     */
    ~NeuralNetwork();

    /**
     * Returns the layers
     *
     * @return The layers
     */
    const std::vector<boost::shared_ptr<layers::Layer<TDevice> > >& layers() const;

    /**
     * Returns the input layer
     *
     * @return The input layer
     */
    layers::InputLayer<TDevice>& inputLayer();

    /**
     * Returns the output layer
     *
     * @return The output layer
     */
    layers::Layer<TDevice>& outputLayer(const int layerID=-1);

    layers::SkipLayer<TDevice>* outGateLayer(const int layerID);
    
    layers::MDNLayer<TDevice>* outMDNLayer(const int layerID=-1);
    
    layers::vqLayer<TDevice>* outvqLayer(const int layerID);
    
    /**
     * Returns the post output layer
     *
     * @return The post output layer
     */
    layers::PostOutputLayer<TDevice>& postOutputLayer();

    /**
     * Loads sequences to the device
     *
     * @param fraction The data set fraction containing the sequences
     */
    void loadSequences(const data_sets::DataSetFraction &fraction);

    bool flagDataValid();

    void restoreTarget(const data_sets::DataSetFraction &fraction);
    
    /**
     * Computes the forward pass
     */
    void computeForwardPass(const int curMaxSeqLength, const real_t uttCnt);

    /**
     * Computes the forward pass
     */    
    void computeForwardPassGen(const data_sets::DataSetFraction &fraction,
			       const int curMaxSeqLength, const real_t generationOpt);
    
    /**
     * Computes the backward pass, including the weight updates
     *
     * The forward pass must be computed first!
     */
    void computeBackwardPass(const int curMaxSeqLength, const real_t uttCnt);

    /**
     * Calculates the error at the output layer
     *
     * The forward pass must be computed first!
     *
     * @return The computed error
     */
    real_t calculateError(const bool flagGenerateMainError) const;

    /**
     * Stores the description of the layers in a JSON tree
     *
     * @param jsonDoc The JSON document
     */
    void exportLayers(const helpers::JsonDocument& jsonDoc) const;

    /**
     * Stores the weights in a JSON tree
     *
     * @param jsonDoc The JSON document
     */
    void exportWeights(const helpers::JsonDocument& jsonDoc) const;

    /**
     * Returns the outputs of the processed fraction
     *
     * ...[1][2][3] contains the activation of the 4th output neuron at the 3nd timestep
     * of the 2nd parallel sequence.
     *
     * @return Outputs of the processed fraction
     */
    std::vector<std::vector<std::vector<real_t> > > getOutputs(const real_t  mdnoutput   = -4.0);

    std::vector<real_t> getOutputNew(const real_t  mdnoutput   = -4.0);
    
    /**
     * Read in the weight from trained_network.jsn or .autosave
     * 
     */
    void importWeights(const helpers::JsonDocument &jsonDoc, const std::string &ctrStr);

    
    /** 
     * Support for word-embedding updating (obsolete)
     */
    bool initWeUpdate(const std::string weBankPath, const unsigned weDim, 
		      const unsigned weIDDim, const unsigned maxLength);
    
    bool initWeNoiseOpt(const int weNoiseStartDim, const int weNoiseEndDim,
			const real_t weNoiseDev);
    
    bool flagInputWeUpdate() const;

    bool saveWe(const std::string weFile) const;

    
    /**
     * Funcs for network training
     */
    // load RMSE output mask 
    bool initMseWeight(const std::string mseWeightPath);

    // load weight mask 
    bool initWeightMask(const std::string weightMaskPath, const int weightMaskOpt);

    // mask the network weights
    void maskWeight();
    
    // re-initialize the network weights
    void reInitWeight();

    // specificla initialization for the layer before MDN
    void initOutputForMDN(const helpers::JsonDocument &jsonDoc);
    
    // Read the mean and variance to the output layer
    void readMVForOutput(const data_sets::DataSetMV &datamv);
    
    // Get the MDN config
    Cpu::real_vector getMdnConfigVec();
    
    // Add 170515 update the current state
    void updateNNState(const int trainingEpoch, const int fracNum, const bool backpropagation);
    
    void updateNNStateForGeneration();
    
    
    /**
     * Functions for GAN training 
     */
    void cleanGradientsForDiscriminator();

    bool flagNetworkForGAN() const;

    /**
     * Functions for network structure 
     */    
    // Whether this layer is AR dependent
    bool flagARdependency(const int layerID);
    bool flagARdependencyEntry(const int layerID);

    // Whether this layer is AR dependent with a particular layer on the path
    bool flagARdependencyWithLayer(const int layerID, const int checkID);
    bool flagARdependencyWithLayerEntry(const int layerID, const int checkID);

    // Whether this layer can be optimized for MA WaveNet models
    bool flagLayerCanbeOptimizedMA(const int layerID);

    
    /**
     * Misc
     */
    bool externalOutputMV(Cpu::real_vector& mean, Cpu::real_vector& var);

    int  outputDimension(const real_t mdnoutput);

    void printLayerDependecy();

    void printNetworkSummary();
    
    int  layerSize(const int layerID);

    bool isMDNLayer(const int layerID);
    
    // print the weight matrix in binary format
    void printWeightMatrix(const std::string weightPath, const int opt);
    
    // notify the current training epoch to each layer
    void notifyCurrentEpoch(const int trainingEpoch);

    // notify the current training epoch to each layer
    void notifyCurrentFrac(const int fracNum);

    // save network
    void saveNetwork(const std::string &filename,
		     const real_t nnlr, const real_t welr);

    // check network is classification network?
    bool isClassificationNet();

    // decide de-normalization vectors for MDN network
    void setDenormalizationMV(cpu_real_vector &outputMeans,
			      cpu_real_vector &outputStdevs);
    
};


#endif // NEURALNETWORK_HPP
