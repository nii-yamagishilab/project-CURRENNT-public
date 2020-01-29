/******************************************************************************
 * This file is heavily modified by
 * Xin WANG
 * National Institute of Informatics, Japan
 * 2016 - 2019
 * 
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
 *
 * You should have received a copy of the GNU General Public License
 * along with CURRENNT.  If not, see <http://www.gnu.org/licenses/>.
 *****************************************************************************/

#include "NeuralNetwork.hpp"
#include "Configuration.hpp"
#include "LayerFactory.hpp"
#include "layers/Layer.hpp"
#include "layers/InputLayer.hpp"
#include "layers/PostOutputLayer.hpp"
#include "layers/SkipLayer.hpp"
#include "layers/NormFlowLayer.hpp"
#include "layers/vaeMiddleLayer.hpp"
#include "layers/InterMetric.hpp"

#include "helpers/JsonClasses.hpp"
#include "helpers/dataProcess.hpp"
#include "helpers/beamSearcher.hpp"
#include "MacroDefine.hpp"

#include "helpers/misFuncs.hpp"
#include "helpers/dotPlot.hpp"

#include <vector>
#include <set>
#include <stdexcept>
#include <algorithm>
#include <cassert>
#include <fstream>

#include <boost/foreach.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#include <boost/random/mersenne_twister.hpp>

#define VAENETWORKTYPE_0 0 // not VAE
#define VAENETWORKTYPE_1 1 // Encoder with feedback, decoder with feedback
#define VAENETWORKTYPE_2 2 // Encoder with feedback, decoder without feedback
#define VAENETWORKTYPE_3 3 // Encoder without feedback, decoder with feedback
#define VAENETWORKTYPE_4 4 // Encoder without feedback, decoder without feedback

#define NETWORKRUNNING_MODE_NORMAL 0      // normal
#define NETWORKRUNNING_MODE_STOPLR0 1     // stop if any layer has LR = 0
#define NETWORKRUNNING_MODE_STOPTEACHER 2 // stop bp in teacher network
#define NETWORKRUNNING_MODE_DISTEACHER 3  // propagate the gradients through the teacher

#define NETWORK_WAVENET_SAVE_NO 0         // not save memory in WaveNet
#define NETWORK_WAVENET_SAVE_AR 1         // save memory for AR WaveNet
#define NETWORK_WAVENET_SAVE_MA 2         // save memory for Non-AR WaveModel


namespace internal {
namespace {    
    bool invalidMiddleMDN(const std::string layerType){
	// check whether the next layer is skip layer
	return (layerType !="skipini" && layerType!="skipadd" && layerType!="skipcat" &&
		layerType !="skipweightadd" && layerType !="operator");
    }

    bool skipLayerTypes(const std::string layerType){
	return (layerType == "skipadd"           ||
		layerType == "skipini"           ||
		layerType == "skipcat"           ||
		layerType == "skipweightadd"     ||
		layerType == "weighted_merge"    ||
		layerType == "skippara_logistic" ||
		layerType == "skippara_relu"     || 
		layerType == "skippara_tanh"     ||
		layerType == "skippara_identity" ||
		layerType == "normflow"          ||
		layerType == "structTrans"       ||
		layerType == "distilling"        ||
		layerType == "feattrans"         ||
		layerType == "random_shuffle");
    }

    bool skipNonParaLayerTypes(const std::string layerType){
	return (layerType == "skipadd"           ||
		layerType == "skipini"           ||
		layerType == "skipcat"           ||
		layerType == "skipweightadd"     ||
		layerType == "weighted_merge"    ||
		layerType == "normflow"          ||
		layerType == "structTrans"       ||
		layerType == "distilling"        ||
		layerType == "feattrans"         ||
		layerType == "random_shuffle");
    }

    bool skipParaLayerTypes(const std::string layerType){
	return (layerType == "skippara_logistic" ||
		layerType == "skippara_relu"     || 
		layerType == "skippara_tanh"     ||
		layerType == "skippara_identity");
    }
    
    bool layersAllowTimeResolutionChange(const std::string layerType){
	return (layerType == std::string("operator")       ||
		layerType == std::string("externalloader") ||
		layerType == std::string("feedback")       ||
		layerType == std::string("vqlayer")        ||
		layerType == std::string("signalgen"));
    }

    int get_tmp_maxSeqLength(const int orignal_maxSeqLength, const int layerResolution,
			     const int waveNetMemSaveFlag,   const int layerID){
	if (waveNetMemSaveFlag == NETWORK_WAVENET_SAVE_MA && layerResolution == 1)
	    return 10;
	else
	    return orignal_maxSeqLength;
    }

    // whether a layer's memory can be optimized for an AR model
    bool flagLayerCanBeOptimizedAR(const int layerID,
				   const int totalNumLayers,
				   std::vector<int> &feedBackLayers,
				   std::vector<int> &vaeLayers)
    {
	if (vaeLayers.size() == 0){
	    // normal WaveNetMode
	    if (feedBackLayers.size() < 1)
		throw std::runtime_error("Impossible error: WaveNet has no feedback loop");
	    
	    if (layerID < (totalNumLayers - 2) && layerID > feedBackLayers[0])
		return true;
	    else
		return false;
	}else{
	    // WaveNet with VAE
	    // Assume the last feedback is used for WaveNet, while others are for VAE encoder
	    
	    if (layerID < (totalNumLayers - 2) && layerID > feedBackLayers.back())
		return true;
	    else
		return false;
	}
    }
    
    // whether a layer's memory can be optimized for a moving average model (non-AR model)
    bool flagLayerCanBeOptimizedMA(const int layerID,
				   const int sourceExcitationLayerID,
				   const int totalNumLayers,
				   const int layerResolution)
    {
	// A layer's memory should be NOT released in Non-AR WaveModel when
	//  1. this layer resolution is not at the waveform level
	//  
	//  3. this layer will be the actual output layer
	
	if (layerResolution > 1 || layerID >= totalNumLayers - 2)
	    return false;
	else
	    return true;
    }
    
}
}

/* ----- Definition for NeuralNetwork  ----- */
/* ------------------------------------------*/
template <typename TDevice>
void NeuralNetwork<TDevice>::__InitializeParameters()
{
    // all types of layer pointer/counter	
    m_signalGenLayerId.clear();
    m_normflowLayers.clear();
    m_feedBackLayers.clear();
    m_vaeLayers.clear();
    m_distillingLayers.clear();
    m_featTransNetRange.clear();
    m_feedBackHiddenLayers.clear();
    m_feedBackHiddenLayersTimeResos.clear();
    m_interMetricLayers.clear();
    m_specialFeedbackLayers.clear();
    
    
    m_firstFeedBackLayer    = -1;     // Idx of the first feedback layer
    m_middlePostOutputLayer = -1;     // Idx of the PostOutputLayer inside the network
    m_featMatchLayer        = -1;     // Idx of the featMatching layer (for GAN)
    m_vaeLayer              = -1;     // Idx of the VAE interface layer
    
    m_vaeNetworkType = VAENETWORKTYPE_0;
	
    m_trainingEpoch         = -1;     // initialize the training epoch counter
    m_trainingFrac          = -1;     // initialize the training data counter
    m_trainingState         = -1;     // initialize the training state of the network
    
    m_wavNetCoreFirstIdx    = -1;     // Idx of the first waveNet core module (for waveNet)
    m_dftLayerIdx           = -1;     // index of DFT error layer
    m_interWeaveIdx         = -1;
    
    m_waveNetMemSaveFlag    = NETWORK_WAVENET_SAVE_NO;
	
    // get a total number of layers
    m_totalNumLayers = 0;

}

template <typename TDevice>
void NeuralNetwork<TDevice>::__InitializeNetworkLayerIdx(const helpers::JsonDocument &jsonDoc)
{    
    try{
	
	const Configuration &config = Configuration::instance();
	
	// check the layers and weight sections
        if (!jsonDoc->HasMember("layers"))
            throw std::runtime_error("Missing section 'layers'");
        rapidjson::Value &layersSection  = (*jsonDoc)["layers"];
        if (!layersSection.IsArray())
            throw std::runtime_error("Section 'layers' is not an array");
        helpers::JsonValue weightsSection;
        if (jsonDoc->HasMember("weights")) {
            if (!(*jsonDoc)["weights"].IsObject())
                throw std::runtime_error("Section 'weights' is not an object");
            weightsSection = helpers::JsonValue(&(*jsonDoc)["weights"]);
        }
	
	// get a total number of layers
	m_totalNumLayers = 0;
	for (rapidjson::Value::ValueIterator layerChild = layersSection.Begin(); 
	     layerChild != layersSection.End(); ++layerChild)
	    m_totalNumLayers++;
	
	/* -------- processing loop -------- */	
	// preloop to determine type o fnetwork
	int counter = 0;
        for (rapidjson::Value::ValueIterator layerChild = layersSection.Begin(); 
	     layerChild != layersSection.End();
	     ++layerChild, counter++){
	    
            if (!layerChild->IsObject())
                throw std::runtime_error("A layer in the 'layers' array is not an object");
            if (!layerChild->HasMember("type"))
                throw std::runtime_error("Missing value 'type' in layer description");
            std::string layerType = (*layerChild)["type"].GetString();

	    // logging information
	    if (layerType == "feedback"){
		// feedback layer
		m_feedBackLayers.push_back(counter);
		if (m_firstFeedBackLayer < 0) m_firstFeedBackLayer = counter;
		
	    }else if (layerType == "middleoutput"){
		// for GAN
		m_middlePostOutputLayer = counter;
		
	    }else if (layerType == "featmatch"){
		// for GAN feature matching layer
		m_featMatchLayer = counter;
		
	    }else if (layerType == "vae" || layerType == "vqlayer"){
		// for vae
		m_vaeLayers.push_back(counter);
		m_vaeLayer       = counter;
		
	    }else if (layerType == "wavnetc" && m_wavNetCoreFirstIdx < 0){
		// for the wavenet component
		m_wavNetCoreFirstIdx   = counter;

		// if this is a wavenet and the generation stage requires
		// loop over time, turn on the mem-save mode
		if (!config.trainingMode() && m_firstFeedBackLayer > 0 &&
		    config.waveNetMemSaveFlag())
		    m_waveNetMemSaveFlag = NETWORK_WAVENET_SAVE_AR;
		
	    }else if (layerType == "normflow"){
		// normalizing flow: transformation layers
		m_normflowLayers.push_back(counter);
		
	    }else if (layerType == "signalgen"){
		// signal generation layers
		m_signalGenLayerId.push_back(counter);
		
	    }else if (layerType == "distilling"){
		// distilling layers (not maintained)
		m_distillingLayers.push_back(counter);
		if (!config.trainingMode() && config.waveNetMemSaveFlag() && m_wavNetCoreFirstIdx)
		    m_waveNetMemSaveFlag = NETWORK_WAVENET_SAVE_MA;
		
	    }else if (layerType == "dft"){
		// DFT training errors layers
		m_dftLayerIdx = counter;
		if (!config.trainingMode() && config.waveNetMemSaveFlag())
		    m_waveNetMemSaveFlag = NETWORK_WAVENET_SAVE_MA;
		
	    }else if (layerType == "feattrans" || layerType == "featsse"){
		// spot the start of feature extraction network
		m_featTransNetRange.push_back(counter);
		
	    }else if (layerType == "feedback_hidden"){
		// feedback layers that takes the output of hidden layers
		m_feedBackHiddenLayers.push_back(counter);
		
	    }else if (layerType.compare(0, 5, "inter") == 0){
		// intermetric layers
		m_interMetricLayers.push_back(counter);
		
	    }else if (layerType == "mutual_weave"){
		// inter weave layer index
		m_interWeaveIdx = counter;
	    }else if (layerType == "simple_feedback"){
		// 
		m_specialFeedbackLayers.push_back(counter);
	    }else{
		// do nothing
	    }   
	}

	if (m_waveNetMemSaveFlag == NETWORK_WAVENET_SAVE_MA)
	    printf("\nGenerating in GPU memory-save mode\n");
	
	
    }catch (const std::exception &e) {
        throw std::runtime_error(std::string("Invalid network file: ") + e.what());
    }
}


template <typename TDevice>
void NeuralNetwork<TDevice>::__CreateNetworkLayers(const helpers::JsonDocument &jsonDoc,
						   int parallelSequences, 
						   int maxSeqLength,
						   int inputSizeOverride)
{
    try {
	
	/* ----- initialization ----- */
	const Configuration &config = Configuration::instance();
		
	int counter          = 0;
	int tmp_maxSeqLength = maxSeqLength;
	
	// check the layers and weight sections
        if (!jsonDoc->HasMember("layers"))
            throw std::runtime_error("Missing section 'layers'");
        rapidjson::Value &layersSection  = (*jsonDoc)["layers"];
        if (!layersSection.IsArray())
            throw std::runtime_error("Section 'layers' is not an array");
        helpers::JsonValue weightsSection;
        if (jsonDoc->HasMember("weights")) {
            if (!(*jsonDoc)["weights"].IsObject())
                throw std::runtime_error("Section 'weights' is not an object");
            weightsSection = helpers::JsonValue(&(*jsonDoc)["weights"]);
        }

	/* ----- creating layers ------ */
        for (rapidjson::Value::ValueIterator layerChild = layersSection.Begin(); 
	     layerChild != layersSection.End();
	     ++layerChild, counter++){
	    
            printf("\nLayer (%d)", counter);
	    
            std::string layerName = (*layerChild)["name"].GetString();
	    printf(" [ %s ] ", layerName.c_str());
            std::string layerType = (*layerChild)["type"].GetString();
	    printf(" %s ", layerType.c_str());
	    
            // update the input layer size
	    // I don't know why the original CURRENNT does this
	    // this part is transparent to the user
            if (inputSizeOverride > 0 && layerType == "input"){
		// for word embedding, input layer size can be changed here
		if (config.weUpdate() && config.trainingMode())
		    inputSizeOverride += (config.weDim() - 1);
		// overwrite the layer size
		(*layerChild)["size"].SetInt(inputSizeOverride);
            }

	    // create a layer
            try {
		
            	layers::Layer<TDevice> *layer;
		
		if(!internal::skipLayerTypes(layerType)){
		    // Normal layers
		    if (m_layers.empty())
			layer = LayerFactory<TDevice>::createLayer(
				layerType, &*layerChild, weightsSection,
				parallelSequences, tmp_maxSeqLength, counter,
				NULL);
		    else
			layer = LayerFactory<TDevice>::createLayer(
				layerType, &*layerChild, weightsSection,
				parallelSequences, tmp_maxSeqLength, counter,
				m_layers.back().get());		
		}else{
		    // skip layers
		    
		    if (m_layers.empty())
			throw std::runtime_error("Skip layers cannot be the first layer");

		    // vector to store previous layers that can be linked to this skip layer
		    std::vector<layers::Layer<TDevice>*> SkipLayers;
		    
		    // for skipadd layer:
		    //   no need to check whether the last skiplayer is directly 
		    //   connected to current skiplayer
		    //   in that case, F(x) + x = 2*x, the gradients will be multiplied by 2
		    // for skippara layer:
		    //   need to check, because H(x)*T(x)+x(1-T(x)) = x if H(x)=x
		    //   check it in SkipParaLayer.cu

		    // Check and preparation
		    if (m_skipAddLayers.size() == 0){
			// if this is the first skip layer
			if (internal::skipParaLayerTypes(layerType)){
			    // skippara requires previous skip layers
			    printf("Error: no preceding skipini/skipadd layer is found");
			    printf("to give x for this layer, which does H(x)*T(x)+x*(1-T(X))\n");
			    printf("Please add a skipini layer after the layer that generates x.");
			    throw std::runtime_error("Error: please modify network.jsn");
			}
		    }else{
			if (layerType == "skipini"){
			    // do nothing
			}else if (internal::skipParaLayerTypes(layerType)){
			    // skippara (highway block) only takes one skiplayer as input source
			    SkipLayers.push_back(m_skipAddLayers.back());
			}else{
			    // skipadd and skipcat can take multiple skiplayers
			    BOOST_FOREACH (layers::Layer<TDevice>* skiplayer, m_skipAddLayers){
				SkipLayers.push_back(skiplayer);
			    }
			}
		    }

		    // Also put the previous layer inside the buffer
		    // Note: previous layer may have been inserted in the above code block
		    //  but it does not matter
		    SkipLayers.push_back(m_layers.back().get());

		    // Add this skip layer based on previous skip layers
		    if (internal::skipParaLayerTypes(layerType)){
			// Skip parametric layer (for highway network)
			layer = LayerFactory<TDevice>::createSkipParaLayer(
				  layerType,     &*layerChild,
				  weightsSection, parallelSequences, 
				  tmp_maxSeqLength,   counter,
				  SkipLayers);
		    }else{
			// Skip non-parametric layer (skip-connection/normalizing flow)
			layer = LayerFactory<TDevice>::createSkipNonParaLayer(
				  layerType,     &*layerChild,
				  weightsSection, parallelSequences, 
				  tmp_maxSeqLength,   counter,
				  SkipLayers);
		    }
		    // Add the new skipadd layer to the buffer of the network
		    m_skipAddLayers.push_back(layer);
		    
		}
			
		
		// post processing for waveNet / NSF or other waveform models
		// This part should be modified so that other networks using WaveNetCore/DFT layer
		// will not go into the memory-save mode
		if (m_wavNetCoreFirstIdx > 0 || m_dftLayerIdx > 0){
		    
		    // for wavenet, link the wavenet block and allocate memory
		    if (counter == m_wavNetCoreFirstIdx){
			layer->linkTargetLayer(*layer);
			for (size_t i = 0; i < counter; ++i) {
			    if (m_layers[i]->getLayerFlag() ==
				std::string("wavenetConditionInputLayer")){
				if (m_layers[i]->type() == std::string("wavenetc"))
				    throw std::runtime_error("External input cannot wavenetc");
				layer->linkTargetLayer(*(m_layers[i].get()));
				break;
			    }
			}
		    }else if (layer->type() == std::string("wavnetc") &&
			      counter > m_wavNetCoreFirstIdx){
			layer->linkTargetLayer(*(m_layers[m_wavNetCoreFirstIdx].get()));
		    }else{
			// nothing
		    }
		   
		    // for wavenet / NSF, reduce the memory in generation
		    if (m_waveNetMemSaveFlag == NETWORK_WAVENET_SAVE_AR){
			// Save the memory for AR WaveNet
			// only save the memory for layers between the feedback and output layer
			if (internal::flagLayerCanBeOptimizedAR(counter, m_totalNumLayers,
								m_feedBackLayers,
								this->m_vaeLayers)){
			    // don't save memory if we want to see its output at every time step
			    if (counter != Configuration::instance().outputFromWhichLayer())
				layer->reduceOutputBuffer();
			}
		    }else if (m_waveNetMemSaveFlag == NETWORK_WAVENET_SAVE_MA){
			// Save the memory for Non-AR WaveModel
			if (internal::flagLayerCanBeOptimizedMA(
				counter, m_signalGenLayerId[0], m_totalNumLayers,
				layer->getResolution()))
			{
			    layer->clearAllBuffers();
			}
			
		    }else{
			// nothing 
		    }
		}

		// Add the new layer to the layer stack of the network
                m_layers.push_back(boost::shared_ptr<layers::Layer<TDevice> >(layer));
	       		
            }
            catch (const std::exception &e) {
                throw std::runtime_error(std::string("Could not create layer: ") + e.what());
            }
	    
        } // Processing loop done
	
	// Let each layer know the following layer
	for (size_t i = 0; i < m_totalNumLayers-1; ++i) 
	    m_layers[i]->linkFollowingLayer(*(m_layers[i+1].get()));
	
    }catch (const std::exception &e) {
        throw std::runtime_error(std::string("Invalid network file: ") + e.what());
    }
}


template <typename TDevice>
void NeuralNetwork<TDevice>::__CheckNetworkLayers()
{
    try{
	/* ----- post-processing check ----- */
	if (m_totalNumLayers != m_layers.size())
	    throw std::runtime_error("Error in network creation: failed \n");
	
        if (m_totalNumLayers < 3)
            throw std::runtime_error("Error in network.jsn: there must be a hidden layer\n");
	
        // check Input
        if (!dynamic_cast<layers::InputLayer<TDevice>*>(m_layers.front().get()))
            throw std::runtime_error("The first layer is not an input layer");
        for (size_t i = 1; i < m_totalNumLayers; ++i) {
            if (dynamic_cast<layers::InputLayer<TDevice>*>(m_layers[i].get()))
                throw std::runtime_error("Multiple input layers defined");
        }

        // check if only the last layer is a post output layer
        if (!dynamic_cast<layers::PostOutputLayer<TDevice>*>(m_layers.back().get()))
            throw std::runtime_error("The last layer is not a post output layer");

	// check the post output layer
	{
	    layers::PostOutputLayer<TDevice>* lastPOLayer;
	    lastPOLayer = dynamic_cast<layers::PostOutputLayer<TDevice>*>(m_layers.back().get());
	    
	    layers::PostOutputLayer<TDevice>* midPOLayer;
	    for (size_t i = 0; i < m_totalNumLayers-1; ++i) {
		midPOLayer = dynamic_cast<layers::PostOutputLayer<TDevice>*>(m_layers[i].get());

		
		if (midPOLayer && midPOLayer->type() == "middleoutput"){
		    // tell the last postoutput layer about the existence of middleoutput
		    lastPOLayer->linkMiddleOutptu(midPOLayer);
		    midPOLayer->setPostLayerType(NN_POSTOUTPUTLAYER_MIDDLEOUTPUT);
		    if (internal::invalidMiddleMDN(m_layers[i+1]->type()))
			throw std::runtime_error("No skipini/add/cat layer after middleoutput");
		    
		}else if (midPOLayer && midPOLayer->type() == "featmatch" && flagNetworkForGAN()){
		    midPOLayer->setPostLayerType(NN_POSTOUTPUTLAYER_FEATMATCH);
		    if (internal::invalidMiddleMDN(m_layers[i+1]->type()))
			throw std::runtime_error("No skipini/add/cat layer after featmatch");
		    
		}else if (midPOLayer && midPOLayer->type() == "mdn" && flagNetworkForGAN()){
		    midPOLayer->setPostLayerType(NN_POSTOUTPUTLAYER_NOTLASTMDN);
		    if (internal::invalidMiddleMDN(m_layers[i+1]->type()))
			throw std::runtime_error("No skipini/add/cat layer after MDN(GAN)");
		    
		}else if (midPOLayer && midPOLayer->type() == "vae"){
		    midPOLayer->setPostLayerType(NN_POSTOUTPUTLAYER_VAEKL);
		    if (flagNetworkForGAN())
			throw std::runtime_error("GAN + VAE is not implemented");
		    // vae layer is defined as PostOutputLayer, not TrainableLayer.
		    // Some layers don't define the code to propagate gradients
		    // to a non-trainable layer. Adding a skipini will receive
		    // the gradients for vae layer
		    if (internal::invalidMiddleMDN(m_layers[i+1]->type()))
			throw std::runtime_error("Please use skipini/add/cat layer after VAE");
		    
		}else if (midPOLayer){
		    throw std::runtime_error("Multiple post output layers defined");

		}else{
		    // no other postoutput layer
		}		    
	    }
	}
	
        // check if two layers have the same name
        for (size_t i = 0; i < m_totalNumLayers; ++i) {
            for (size_t j = 0; j < m_totalNumLayers; ++j) {
                if (i != j && m_layers[i]->name() == m_layers[j]->name())
                    throw std::runtime_error(
			std::string("Error in network.jsn: different layers have the name '") + 
			m_layers[i]->name() + "'");
            }
        }

	// Check VAE type
	if (m_vaeLayer > 0){
	    bool tmpFlag1 = false;
	    bool tmpFlag2 = false;
	    if (!m_feedBackLayers.empty()){
		for (size_t i = 0; i < m_feedBackLayers.size(); i++){
		    if (m_feedBackLayers[i] < m_vaeLayer) tmpFlag1 = true;
		    if (m_feedBackLayers[i] > m_vaeLayer) tmpFlag2 = true;
		}
		if (tmpFlag1 && tmpFlag2) m_vaeNetworkType = VAENETWORKTYPE_1;
		else if (tmpFlag1 && !tmpFlag2) m_vaeNetworkType = VAENETWORKTYPE_2;
		else if (!tmpFlag1 && tmpFlag2) m_vaeNetworkType = VAENETWORKTYPE_3;
		else throw std::runtime_error("Impossible error");

		// Further check VAENETWORKTYPE_2
		// although no feedback layer in decoder, there may be skip-connection
		//  which depends on feedback layer
		if (m_vaeNetworkType == VAENETWORKTYPE_2 &&
		    this->flagARdependencyEntry(m_totalNumLayers - 1)){
		    m_vaeNetworkType == VAENETWORKTYPE_1;
		}   
	    }else{
		m_vaeNetworkType = VAENETWORKTYPE_4;
	    }
	    printf("\nVAE network type: ");
	    switch (m_vaeNetworkType){
	    case VAENETWORKTYPE_1:
		printf("encoder with feedback, decoder with feedback");
		break;
	    case VAENETWORKTYPE_2:
		printf("encoder with feedback, decoder without feedback");
		break;		
	    case VAENETWORKTYPE_3:
		printf("encoder without feedback, decoder with feedback");
		break;
	    case VAENETWORKTYPE_4:
		printf("encoder without feedback, decoder without feedback");
		break;
	    default:
		throw std::runtime_error("Impossible error: unknown VAE network type");
		break;
	    }
	    printf("\n");	    
	}else{
	    //printf("None-VAE network");
	    m_vaeNetworkType = VAENETWORKTYPE_0;
	}
	
	// Check the time resolution
	for (size_t i = 1; i < m_totalNumLayers; ++i){
	    if (m_layers[i]->getResolution() != m_layers[i-1]->getResolution()){
		// Only allow Operator, externalLoader,
		//   feedback (without reading previous output)
		if (!internal::layersAllowTimeResolutionChange(m_layers[i]->type())){
		    printf("Time resolution of %s conflicts with previous layer\n",
			   m_layers[i]->name().c_str());
		    throw std::runtime_error("Please check time resolution configuration\n");
		}
	    }
	}

	// check if feature extraction network exists
	if (!m_featTransNetRange.empty()){
	    if (m_featTransNetRange.size() != 2)
		throw std::runtime_error("Error in network.jsn: one feattrans needs one featsse");
	    printf("\nFeat extraction subnet from layer (%d) to (%d)\n",
		   m_featTransNetRange[0], m_featTransNetRange[1]);
	}	
    }
    catch (const std::exception &e) {
        throw std::runtime_error(std::string("Invalid network file: ") + e.what());
    }
}

template <typename TDevice>
void NeuralNetwork<TDevice>::__LinkNetworkLayers()
{
    try{
	// Link the target layer with the feedback layer
	if (!m_feedBackLayers.empty()){
	    
	    if (flagNetworkForGAN())
		throw std::runtime_error("GAN + Feedback is not implemented");

	    for (size_t i = 0; i<m_feedBackLayers.size(); i++){
		m_layers[m_feedBackLayers[i]]->linkTargetLayer(*(m_layers.back().get()));
	    }
	    // check the bi-directional RNN after the feedback layer
	    for (size_t i = m_firstFeedBackLayer; i < m_totalNumLayers-1; i++){
		if (m_layers[i]->type()==std::string("brnn") ||
		    m_layers[i]->type()==std::string("blstm")){

		    // check for none-VAE network
		    if (this->flagARdependencyWithLayerEntry(m_totalNumLayers-1, i)){
			printf("Layer %s cannot be a bi-directional RNN\n",
			       m_layers[i]->name().c_str());
			throw std::runtime_error("Error in network.jsn");
		    }
			
		    /*for (size_t j = 0; j<feedBacklayerId.size(); j++){
			if (feedBacklayerId[j] > m_vaeLayer && feedBacklayerId[j] < i){
			    printf("Layer %s is invalid as a bi-directional rnn",
				   m_layers[i]->name().c_str());
			    throw std::runtime_error("Error in network.jsn");
			}
		    }*/
		}else if (m_layers[i]->type()==std::string("featmatch") ||
			  m_layers[i]->type()==std::string("middleoutput")){
		    throw std::runtime_error(
			 std::string("Error in network.jsn.") +
			 std::string("Feedback was not implemented for GAN"));
		}
	    }
	}

	/*// Link the wavnet copmonents 
	if (m_wavNetCoreFirstIdx > 0){

	    // Step1: set the flag in initial WaveNet-core and link following WaveNet-core
	    //     to initial WaveNet-core
	    for (size_t i = 0; i < m_totalNumLayers; ++i) {
		if (m_layers[i]->type() == std::string("wavnetc")){
		    m_layers[i]->linkTargetLayer(*(m_layers[m_wavNetCoreFirstIdx].get()));
		}
	    }
	    // Step2: link the linguistic context to the initial Wavenet core
	    for (size_t i = 0; i < m_totalNumLayers; ++i) {
		if (m_layers[i]->getLayerFlag() == std::string("wavenetConditionInputLayer")){
		    if (m_layers[i]->type() == std::string("wavenetc"))
			throw std::runtime_error("External input cannot be from wavenetc");
		    m_layers[m_wavNetCoreFirstIdx]->linkTargetLayer(*(m_layers[i].get()));
		    break;
		}
	    }
	}*/

	
	// Link the target layer for signalgen
	if (!m_signalGenLayerId.empty()){
	    for (size_t i = 0; i < m_signalGenLayerId.size(); i++)
		m_layers[m_signalGenLayerId[i]]->linkTargetLayer(*(m_layers.back().get()));
	}

	// Link the target layer for distilling interfaces
	if (!m_distillingLayers.empty()){
	    for (size_t i = 0; i < m_distillingLayers.size(); i++)
		m_layers[m_distillingLayers[i]]->linkTargetLayer(*(m_layers.back().get()));
	}

	// Link the target layer for dft postoutput layer
	if (m_dftLayerIdx > 0){
	    for (size_t i = 0; i < m_totalNumLayers; i++)
		m_layers[m_dftLayerIdx]->linkTargetLayer(*(m_layers[i].get()));
	}

	// Link special feedback layers
	if (!m_specialFeedbackLayers.empty()){
	    // link the target layers
	    for (size_t i = 0; i < m_specialFeedbackLayers.size(); i++)
		for (size_t j = 0; j < m_totalNumLayers; j++)
		    m_layers[m_specialFeedbackLayers[i]]->linkTargetLayer(*(m_layers[j].get()));
	}

	
	// Link the target layer with inter_weave layer
	if (m_interWeaveIdx > 0){
	    m_layers[m_interWeaveIdx]->linkTargetLayer(*(m_layers.back().get()));
	}
	
	// Link the source layer for sse_muti or sse_cos layers
	if (m_layers[m_totalNumLayers-1]->type() == "sse_multi" ||
	    m_layers[m_totalNumLayers-1]->type() == "sse_cos"){
	    for (size_t i = 0; i < m_totalNumLayers-1; i++)
		m_layers[m_totalNumLayers-1]->linkTargetLayer(*(m_layers[i].get()));
	}
	
	// Link the feedback hidden layers
	if (m_feedBackHiddenLayers.size()){
	    // link the target layers
	    for (size_t i = 0; i < m_feedBackHiddenLayers.size(); i++)
		for (size_t j = 0; j < m_totalNumLayers; j++)
		    m_layers[m_feedBackHiddenLayers[i]]->linkTargetLayer(*(m_layers[j].get()));
	    
	    // get the time resolutions
	    int counter = 0;
	    int timeResolution = -1;
	    for (size_t i = 0; i < m_feedBackHiddenLayers.size(); i++){

		// decide the layer range of the feedback block
		int feedback_layer_idx = m_feedBackHiddenLayers[i];
		int target_layer_idx = m_layers[m_feedBackHiddenLayers[i]]->returnTargetLayerID();
		if (target_layer_idx < feedback_layer_idx ||
		    target_layer_idx > m_totalNumLayers)
		    throw std::runtime_error("feedback_hidden target layer not linked");

		timeResolution = -1;
		counter = 0;
		BOOST_FOREACH (boost::shared_ptr<layers::Layer<TDevice> > &layer, m_layers){
		    if (counter >= feedback_layer_idx && counter <= target_layer_idx)
			if (timeResolution < 0)
			    timeResolution = layer->getResolution();
			else if (timeResolution != layer->getResolution())
			    throw std::runtime_error("Error: multi-resolutions in feedback block");
		    counter++;
		}
		if (timeResolution < 0)
		    throw std::runtime_error("Error: time resos negative");
		m_feedBackHiddenLayersTimeResos.push_back(timeResolution);
	    }	
	}
	
	
	
	// Check for normalization flow
	if (!m_normflowLayers.empty()){
	    if ((!m_feedBackLayers.empty()) || (m_vaeNetworkType != VAENETWORKTYPE_0) ||
		flagNetworkForGAN())
		throw std::runtime_error("Feedback/GAN/VAE cannot be used with normflow");
	    if (m_normflowLayers.size() == 1)
		throw std::runtime_error("There is only one normflow layer, which is invalid");

	    // Link the target layer
	    // Currently, only the final output layer can be used a target layer
	    for (size_t i = 0; i < m_normflowLayers.size(); i++)
		m_layers[m_normflowLayers[i]]->linkTargetLayer(*(m_layers.back().get()));

	    // Further check
	    for (size_t i = 0; i < m_normflowLayers.size()-1; i++){
		int layerIdx1 = m_normflowLayers[i];
		int layerIdx2 = m_normflowLayers[i+1];

		// Whether two normalizing layers are neighors
		if ((layerIdx1 + 1) == layerIdx2){
		    printf("No transformation layer between %d & %d", layerIdx1, layerIdx2);
		    throw std::runtime_error("Error in network.jsn");
		}

		// Whether bi-directional RNN is between two normalizing layers
		for (size_t j = layerIdx1+1; j < layerIdx2; j++){
		    if (m_layers[j]->type() == "blstm" || m_layers[j]->type() == "brnn"){
			printf("Layer %s cannot be a bi-directional RNN\n",
			       m_layers[j]->name().c_str());
			throw std::runtime_error("Error in network.jsn");
		    }	
		}

		// Another chech should be added: whether the convoluion layer is causal
	    }

	    // Notify the last normlayer that he/she is the last normlayer
	    int lastIdx = m_normflowLayers.back();
	    layers::NormFlowLayer<TDevice>* tmpPtr =
		dynamic_cast<layers::NormFlowLayer<TDevice>*>(m_layers[lastIdx].get());
	    if (tmpPtr)
		tmpPtr->noticeLastFlow();
	    else
		throw std::runtime_error("Impossible error");
	}
	
    }catch (const std::exception &e) {
        throw std::runtime_error(std::string("Invalid network file: ") + e.what());
    }
}

template <typename TDevice>
void NeuralNetwork<TDevice>::__CreateDependency()
{
    try{
	// collect the dependency between layers
	this->m_networkMng.build(m_totalNumLayers);
	BOOST_FOREACH (boost::shared_ptr<layers::Layer<TDevice> > &layer, m_layers){
	    this->m_networkMng.add_layerDep(layer->getLayerID(), layer->dependLayerIDs());
	}
	// print the dependency into .dot graphic language
	this->printLayerDependecy();
	
    }catch (const std::exception &e) {
        throw std::runtime_error(std::string("Invalid network file: ") + e.what());
    }
}



template <typename TDevice>
NeuralNetwork<TDevice>::NeuralNetwork(const helpers::JsonDocument &jsonDoc,
				      int parallelSequences, 
				      int maxSeqLength,
				      int inputSizeOverride,
				      int outputSizeOverride)
    : m_networkMng ()
    , m_vecPoolMng (maxSeqLength, parallelSequences)
{
    // Initialzie a few parameters
    this->__InitializeParameters();
    
    // Initialize the network layer indices
    this->__InitializeNetworkLayerIdx(jsonDoc);
    
    // Create the layers in a network
    this->__CreateNetworkLayers(jsonDoc, parallelSequences, maxSeqLength, inputSizeOverride);

    // prelimary check
    this->__CheckNetworkLayers();

    // link layers inside the network
    this->__LinkNetworkLayers();

    // create the dependency graph
    this->__CreateDependency();

}

template <typename TDevice>
NeuralNetwork<TDevice>::~NeuralNetwork()
{
}


template <typename TDevice>
void NeuralNetwork<TDevice>::printLayerDependecy()
{

    // Write the network structure to dot
    // Further optimization is required ???
    std::string dotFile(Configuration::instance().networkGraphFile());
    
    if (dotFile.size()){
	
	std::ofstream ofs(dotFile.c_str(), std::ofstream::out);
	if (!ofs.good()){
	    std::cout << "Fail to open " << dotFile << std::endl;
	    return;
	}
	
	dotPlot::printDotHead(ofs);
	BOOST_FOREACH (helpers::layerDep layerDep_tmp, this->m_networkMng.get_layerDeps()){
	    BOOST_FOREACH (int towhich, layerDep_tmp.get_towhich()){
		dotPlot::printDotNode(ofs, layerDep_tmp.get_layerID(),
				      m_layers[layerDep_tmp.get_layerID()]->name(),
				      m_layers[layerDep_tmp.get_layerID()]->type(),
				      towhich,
				      m_layers[towhich]->name(),
				      m_layers[towhich]->type());
	    }
	}
	dotPlot::printDotEnd(ofs);
    
	ofs.close();
	std::cout << "Print network graph file to " << dotFile << std::endl;
	
    }else{
	return;
    }
}

template <typename TDevice>
bool NeuralNetwork<TDevice>::flagNetworkForGAN() const{
    // Whether this network is for GAN
    // output: 
    //    true: this network is for training GAN
    //   false: a normal NN
    return (m_middlePostOutputLayer > 0);
}


template <typename TDevice>
const std::vector<boost::shared_ptr<layers::Layer<TDevice> > >& NeuralNetwork<TDevice>::layers() const
{
    return m_layers;
}

template <typename TDevice>
layers::InputLayer<TDevice>& NeuralNetwork<TDevice>::inputLayer()
{
    return static_cast<layers::InputLayer<TDevice>&>(*m_layers.front());
}

template <typename TDevice>
layers::Layer<TDevice>& NeuralNetwork<TDevice>::outputLayer(const int layerID)
{
    // default case, the output layer
    int tmpLayerID = layerID;
    if (tmpLayerID < 0)
	tmpLayerID = m_totalNumLayers-2;
    
    // check
    if (tmpLayerID > (m_totalNumLayers-1))
	throw std::runtime_error(std::string("Invalid output_tap ID (out of range)"));

    /* Modify 04-08 to tap in the output of arbitary layer */
    /*template <typename TDevice>
      layers::TrainableLayer<TDevice>& NeuralNetwork<TDevice>::outputLayer()
      {
      return static_cast<layers::TrainableLayer<TDevice>&>(*m_layers[m_totalNumLayers-2]);
      }
    */

    return (*m_layers[tmpLayerID]);
}

template <typename TDevice>
layers::SkipLayer<TDevice>* NeuralNetwork<TDevice>::outGateLayer(const int layerID)
{
    // default case, the output
    int tmpLayerID = layerID;
    
    // check
    if (tmpLayerID > (m_totalNumLayers-2) || tmpLayerID < 0)
	throw std::runtime_error(std::string("Invalid gate_output_tap ID (out of range)"));
    
    return dynamic_cast<layers::SkipLayer<TDevice>*>(m_layers[tmpLayerID].get());
}

template <typename TDevice>
layers::MDNLayer<TDevice>* NeuralNetwork<TDevice>::outMDNLayer(const int layerID)
{
    if (layerID < 0){
	return dynamic_cast<layers::MDNLayer<TDevice>*>(m_layers[m_totalNumLayers-1].get());
    }else{
	return dynamic_cast<layers::MDNLayer<TDevice>*>(m_layers[layerID].get());
    }
}

template <typename TDevice>
layers::vqLayer<TDevice>* NeuralNetwork<TDevice>::outvqLayer(const int layerID)
{
    if (layerID > m_totalNumLayers-1 || layerID < 0)
	throw std::runtime_error(std::string("Invalid vqLayer ID (out of range)"));
    return dynamic_cast<layers::vqLayer<TDevice>*>(m_layers[layerID].get());
}



template <typename TDevice>
bool NeuralNetwork<TDevice>::flagARdependencyEntry(const int layerID)
{
    m_tmpLayerIdx.clear();
    return this->flagARdependency(layerID);
}

template <typename TDevice>
bool NeuralNetwork<TDevice>::flagARdependency(const int layerID)
{
    // Check the AR dependency Recursively
    // terminal condition:
    //     input layer: false
    //     vaeLayer:    false
    //     feedback:    true
    // recursive:
    //     this is the postoutput layer/trainable layer/Skip connect/layer
    //     search all the paths
    //
    size_t tmpID;
    if (std::find(m_tmpLayerIdx.begin(), m_tmpLayerIdx.end(), layerID) != m_tmpLayerIdx.end())
	return false;
    else
	m_tmpLayerIdx.push_back(layerID);
    
    if (layerID < 0 || layerID >= m_totalNumLayers){
	printf("Invalid input layerID %d", layerID);
	return false;
    }else if (layerID == 0){
	// input layer
	return false;
    }else if (layerID == m_vaeLayer){
	// vae layer
	return false;
    }else if (std::find(m_feedBackLayers.begin(), m_feedBackLayers.end(),
			layerID) != m_feedBackLayers.end()){
	// feedback layer (before VAE layer)
	return true;
    }else{
	
	layers::SkipLayer<TDevice>* tmpPtr1 =
	    dynamic_cast<layers::SkipLayer<TDevice>*>(m_layers[layerID].get());
        layers::TrainableLayer<TDevice> *tmpPtr3 = 
	    dynamic_cast<layers::TrainableLayer<TDevice>*>(m_layers[layerID].get());
	layers::PostOutputLayer<TDevice> *tmpPtr4 = 
	    dynamic_cast<layers::PostOutputLayer<TDevice>*>(m_layers[layerID].get());
	
	if (tmpPtr1){
	    BOOST_FOREACH (layers::Layer<TDevice> *tmpL, tmpPtr1->PreLayers()){
		if (this->flagARdependency(tmpL->getLayerID()))
		    return true;
	    }
	}else if (tmpPtr3){
	    if (this->flagARdependency(tmpPtr3->precedingLayer().getLayerID()))
		return true;
	}else if (tmpPtr4){
	    if (this->flagARdependency(tmpPtr4->precedingLayer().getLayerID()))
		return true;
	}else{
	    printf("Invalid layer type %d: neither trainable or postoutput. ", layerID);
	    throw std::runtime_error("Imposible error network.jsn");
	}
	return false;
    }
}

template <typename TDevice>
bool NeuralNetwork<TDevice>::flagARdependencyWithLayerEntry(const int layerID, const int checkID)
{
    m_tmpLayerIdx.clear();
    return this->flagARdependencyWithLayer(layerID, checkID);
}

template <typename TDevice>
bool NeuralNetwork<TDevice>::flagARdependencyWithLayer(const int layerID, const int checkID)
{
    // The function is mainly for identify the layer with checkID in all the possible paths
    // 
    size_t tmpID;
    if (layerID == checkID)
	return this->flagARdependencyEntry(checkID);
    
    
    if (std::find(m_tmpLayerIdx.begin(), m_tmpLayerIdx.end(), layerID) != m_tmpLayerIdx.end())
	return false;
    else
	m_tmpLayerIdx.push_back(layerID);


    
    if (layerID < 0 || layerID >= m_totalNumLayers){
	printf("Invalid input layerID %d", layerID);
	return false;
    }else if (layerID == 0){
	// input layer
	return false;
    }else if (layerID == m_vaeLayer){
	// vae layer
	return false;
    }else if (std::find(m_feedBackLayers.begin(), m_feedBackLayers.end(),
			layerID) != m_feedBackLayers.end()){
	// feedback layer (before VAE layer)
	return false;
    }else{
	
	layers::SkipLayer<TDevice>* tmpPtr1 =
	    dynamic_cast<layers::SkipLayer<TDevice>*>(m_layers[layerID].get());
        layers::TrainableLayer<TDevice> *tmpPtr3 = 
	    dynamic_cast<layers::TrainableLayer<TDevice>*>(m_layers[layerID].get());
	layers::PostOutputLayer<TDevice> *tmpPtr4 = 
	    dynamic_cast<layers::PostOutputLayer<TDevice>*>(m_layers[layerID].get());
	
	if (tmpPtr1){
	    BOOST_FOREACH (layers::Layer<TDevice> *tmpL, tmpPtr1->PreLayers()){
		if (this->flagARdependencyWithLayer(tmpL->getLayerID(), checkID))
		    return true;
	    }
	}else if (tmpPtr3){
	    if (this->flagARdependencyWithLayer(tmpPtr3->precedingLayer().getLayerID(), checkID))
		return true;
	}else if (tmpPtr4){
	    if (this->flagARdependencyWithLayer(tmpPtr4->precedingLayer().getLayerID(), checkID))
		return true;
	}else{
	    printf("Invalid layer type %d: neither trainable or postoutput. ", layerID);
	    throw std::runtime_error("Imposible error network.jsn");
	}
	return false;
    }
}

template <typename TDevice>
bool NeuralNetwork<TDevice>::flagLayerCanbeOptimizedMA(const int layerID)
{
    //  m_signalGenLayerID[0] is the layer ID of the first source excitation layer,
    //   which is assumed to be the last layer of condition module
    //  m_totalNumLayers - 2 is the layer ID of the actual output layer
    if (layerID < 0 || layerID >= m_totalNumLayers || layerID >= m_layers.size())
	throw std::runtime_error("Error: flagLayerCanbeOptimizedMA input layerID invalid");
    
    return internal::flagLayerCanBeOptimizedMA(layerID, m_signalGenLayerId[0], m_totalNumLayers,
					       m_layers[layerID]->getResolution());
}


template <typename TDevice>
layers::PostOutputLayer<TDevice>& NeuralNetwork<TDevice>::postOutputLayer()
{
    return static_cast<layers::PostOutputLayer<TDevice>&>(*m_layers.back());
}

template <typename TDevice>
void NeuralNetwork<TDevice>::loadSequences(const data_sets::DataSetFraction &fraction)
{
    BOOST_FOREACH (boost::shared_ptr<layers::Layer<TDevice> > &layer, m_layers)
    {
        layer->loadSequences(fraction, m_trainingState);
    }
}


template <typename TDevice>
bool NeuralNetwork<TDevice>::flagDataValid()
{
    
    // Check the validity of the data (loaded by the target layer)
    //  specific functions should be defined in helpers/dataProcess.hpp 
    if (Configuration::instance().dataCheckerIdx() == DATACHECKER_WAVEFORM_SILENCE){
	layers::PostOutputLayer<TDevice>* postoutLayer =
	    dynamic_cast<layers::PostOutputLayer<TDevice>*>(m_layers.back().get());
	return postoutLayer->checkOutputValid(DATACHECKER_WAVEFORM_SILENCE);
    }else{
	return true;
    }
}
    


template <typename TDevice>
void NeuralNetwork<TDevice>::restoreTarget(const data_sets::DataSetFraction &fraction)
{
    const Configuration &config = Configuration::instance();

    if (config.scheduleSampOpt() == NN_FEEDBACK_SC_SOFT ||
	config.scheduleSampOpt() == NN_FEEDBACK_SC_MAXONEHOT ||
	config.scheduleSampOpt() == NN_FEEDBACK_SC_RADONEHOT){
        m_layers[m_totalNumLayers-1]->loadSequences(fraction, m_trainingState);
    }
}


template <typename TDevice>
bool NeuralNetwork<TDevice>::__stopBackPropagation(const int layerID, const int runningMode)
{
    if (runningMode == NETWORKRUNNING_MODE_STOPLR0){
	layers::TrainableLayer<TDevice> *trainableLayer = 
	    dynamic_cast<layers::TrainableLayer<TDevice>*>(m_layers[layerID].get());
	if (trainableLayer && misFuncs::closeToZero(trainableLayer->learningRate()))
	    return true;
    }
    return false;
}

template <typename TDevice>
void NeuralNetwork<TDevice>::__computeForward_LayerByLayer(const int curMaxSeqLength,
							   const real_t uttCnt)
{
    
    if (m_featTransNetRange.size()){
	
	// if feat extraction network exists
	int counter = 0;
	BOOST_FOREACH (boost::shared_ptr<layers::Layer<TDevice> > &layer, m_layers){
	    layer->computeForwardPass(m_trainingState);
	    if (counter == m_featTransNetRange[0])
		break;
	    counter++;
	}

	// copy natural data
	m_layers[m_featTransNetRange[0]]->copyOutputs(
		m_layers[m_featTransNetRange[1]]->outputs());

	counter = 0;
	// propagte the natural data
	BOOST_FOREACH (boost::shared_ptr<layers::Layer<TDevice> > &layer, m_layers){
	    if (counter > m_featTransNetRange[0])
		layer->computeForwardPass(m_trainingState);
	    counter++;
	}

	counter = 0;
	// propagte the synthetic data
	BOOST_FOREACH (boost::shared_ptr<layers::Layer<TDevice> > &layer, m_layers){
	    if (counter >= m_featTransNetRange[0])
		layer->computeForwardPass(m_trainingState);
	    counter++;
	}	    
	    
    }else{
	
	// No feedback, normal forward computation.
	// Note that, normalizing layers are ordered so that it can be treated as a
	//  normal network during training.
	BOOST_FOREACH (boost::shared_ptr<layers::Layer<TDevice> > &layer, m_layers)	    
	    layer->computeForwardPass(m_trainingState);
	
	// For GAN with feature matching, do additional propagation
	if (m_trainingState == NN_STATE_GAN_GEN_FEATMAT && flagNetworkForGAN() && 
	    m_featMatchLayer > 0){
	    m_trainingState = NN_STATE_GAN_GEN; // return to the normal state
	    for (int i = m_middlePostOutputLayer; i < m_totalNumLayers; i++)
		m_layers[i]->computeForwardPass(m_trainingState);
	}
	
    }
}

template <typename TDevice>
void NeuralNetwork<TDevice>::__computeForward_TeacherForce_LayerByLayer(const int curMaxSeqLength,
									const real_t uttCnt)
{

    /* --- initialization */
    const Configuration &config = Configuration::instance();    
	
    // prepare random numbers
    static boost::mt19937 *gen = NULL;
    if (!gen) {
	gen = new boost::mt19937;
	gen->seed(config.randomSeed()+98); // any random number
    }
    boost::random::uniform_real_distribution<real_t> dist(0, 1);

    int scheduleSampOpt = config.scheduleSampOpt();
    int scheduleSampPara= config.scheduleSampPara();
    int dropoutBeforeVAE= config.dropoutbeforeVAE();
    int layerCnt = 0;
    int methodCode = 0;
    real_t threshold = 0;
    
    // Load the natural feedback data from the target layer
    this->postOutputLayer().retrieveFeedBackData();
    
    
    switch (scheduleSampOpt){
    case NN_FEEDBACK_GROUND_TRUTH:
	{
	    // Case0: use ground truth directly, without dropout
	    layerCnt = 0;
	    BOOST_FOREACH (boost::shared_ptr<layers::Layer<TDevice> > &layer, m_layers){
		layer->computeForwardPass(m_trainingState);
		
		// Special case:
		// for distilling network, reload the target after student network
		// calculates the output, i.e., use student's output as feedback data
		if ((!m_distillingLayers.empty()) && layerCnt == m_distillingLayers.back())
		    this->postOutputLayer().retrieveFeedBackData();
		
		layerCnt++;
	    }
	    break;
	}
    case NN_FEEDBACK_DROPOUT_1N:
    case NN_FEEDBACK_DROPOUT_ZERO:
	{
	    // Case 1 & 2: schedule dropout, set data to 1/N (case 1) or 0 (case 2)
	    threshold = ((real_t)scheduleSampPara)/100;

	    // Prepare the random vector
	    Cpu::real_vector randNum;
	    randNum.reserve(curMaxSeqLength);
	    for (size_t i = 0; i < curMaxSeqLength; ++i)
		if (dist(*gen) > threshold)
		    randNum.push_back(0);  // dropout
		else
		    randNum.push_back(1);  // not dropout

	    // Propagate
	    if (m_vaeLayer > 0){
		// If VAE exists, decide whether to use dropout for encoder
		
		if (dropoutBeforeVAE == 1){
		    // Drop out the feedback data for encoder
		    typename TDevice::real_vector temp = randNum;
		    this->postOutputLayer().retrieveFeedBackData(temp, scheduleSampOpt);
		}
		
		int cnt = 0;
		BOOST_FOREACH (boost::shared_ptr<layers::Layer<TDevice> > &layer, m_layers){
		    layer->computeForwardPass(m_trainingState);
		    if (cnt == m_vaeLayer && dropoutBeforeVAE == 0){
			// Dropout after encoder
			typename TDevice::real_vector temp = randNum;
			this->postOutputLayer().retrieveFeedBackData(temp, scheduleSampOpt);
		    }
		    cnt++;
		}
		
	    }else{
		// Normal cases, dropout before feedback
		// Drop out the feedback data randomly
		typename TDevice::real_vector temp = randNum;
		this->postOutputLayer().retrieveFeedBackData(temp, scheduleSampOpt);

		// ComputeForward
		BOOST_FOREACH (boost::shared_ptr<layers::Layer<TDevice> > &layer, m_layers)
		    layer->computeForwardPass(m_trainingState);
	    }
	    break;
	}
    default:
	{
	    throw std::runtime_error("Impossible error: unknown teacher forceing mode");
	}
    }
    // Done
}



template <typename TDevice>
void NeuralNetwork<TDevice>::__computeForward_ScheduleSamp_LayerByLayer(const int curMaxSeqLength,
									const real_t uttCnt)
{
    /* Schedule sampling is not test after this function is separated from computeForwardPass */

    const Configuration &config = Configuration::instance();
	
    // prepare random numbers
    static boost::mt19937 *gen = NULL;
    if (!gen) {
	gen = new boost::mt19937;
	gen->seed(config.randomSeed()+98); // any random number
    }
    boost::random::uniform_real_distribution<real_t> dist(0, 1);

    int scheduleSampOpt = config.scheduleSampOpt();
    int scheduleSampPara= config.scheduleSampPara();
    int dropoutBeforeVAE= config.dropoutbeforeVAE();
    int layerCnt = 0;
    int methodCode = 0;
    
    // Load the natural feedback data from the target layer
    this->postOutputLayer().retrieveFeedBackData();

    // Propagate
    switch (scheduleSampOpt){
	
    case NN_FEEDBACK_SC_SOFT:
    case NN_FEEDBACK_SC_MAXONEHOT:
    case NN_FEEDBACK_SC_RADONEHOT:
	{
	    
	    // use soft vector as feedback (case 3) or one-hot (case 4),
	    // or random output as feedback
		
	    real_t sampThreshold;
	    methodCode = scheduleSampOpt;
		
	    // Forward computation for layers before the Feedback layer
	    int cnt = 0;
	    BOOST_FOREACH (boost::shared_ptr<layers::Layer<TDevice> > &layer, m_layers){
		if (cnt == m_firstFeedBackLayer) break; 
		layer->computeForwardPass(m_trainingState);
		cnt++;
	    }
		
	    // Determine the threshold 
	    if (scheduleSampPara > 0){
		// randomly use the generated sample
		sampThreshold =
		    (1.0 / (1.0 + exp((uttCnt - NN_FEEDBACK_SCHEDULE_SIG) * 1.0 /
				      scheduleSampPara)));
		// sampThreshold = 1.0 - ((real_t)uttCnt/scheduleSampPara);
		//sampThreshold = pow(scheduleSampPara/100.0, uttCnt);
		sampThreshold = ((sampThreshold  < NN_FEEDBACK_SCHEDULE_MIN) ?
				 NN_FEEDBACK_SCHEDULE_MIN : sampThreshold);
	    }else{
		sampThreshold = (-1.0 * (real_t)scheduleSampPara / 100.0);
	    }

	    // printf("%f %f\n", uttCnt, sampThreshold);
	    // Forward computation for layer above feedback using schedule sampling
	    for (int timeStep = 0; timeStep < curMaxSeqLength; timeStep++){

		cnt = 0;
		BOOST_FOREACH (boost::shared_ptr<layers::Layer<TDevice> > &layer, m_layers){
		    if (cnt >= m_firstFeedBackLayer){
			// For Rnn and LSTM, prepareStepGeneration is necessary
			// to prepare the data matrix per frame
			layer->prepareStepGeneration(timeStep); 
			layer->computeForwardPass(timeStep, m_trainingState);    
		    }
		    cnt++;
		}
		    
		// 
		if (dist(*gen) > sampThreshold){
		    //printf("\n %d HIT", timeStep);
		    layers::MDNLayer<TDevice> *olm;
		    olm = outMDNLayer();
		    if (olm != NULL){
			olm->getOutput(timeStep, 0.0001); 
			olm->retrieveFeedBackData(timeStep, methodCode);
			/******** Fatal Error *******/
			// After getOutput, the targets will be overwritten by generated data.
			// But the target will be used by calculateError and computeBackWard.
			// Thus, targets of the natural data should be re-written
			// This is now implemented as this->restoreTarget(frac)	    
		    }    
		}
	    }
	    break;
	}
    default:
	{
	    throw std::runtime_error("Impossible error: unknown schedule sampling mode");
	}
    }
    // Done
}


template <typename TDevice>
void NeuralNetwork<TDevice>::__computeForward_StepByStep(const int curMaxSeqLength,
							 const real_t uttCnt)
{
    /* current implementation only supports on feedback hidden layer*/
    if (m_feedBackHiddenLayers.size() != 1)
	throw std::runtime_error("No support for network with multiple feedback_hidden layers");

    int feedback_layer_idx = m_feedBackHiddenLayers[0];
    int feedback_target_layer_idx = m_layers[m_feedBackHiddenLayers[0]]->returnTargetLayerID();
    int timeResolution = m_feedBackHiddenLayersTimeResos[0];
    
    if (feedback_target_layer_idx < feedback_layer_idx ||
	feedback_target_layer_idx > m_totalNumLayers)
	throw std::runtime_error("feedback_hidden target layer not linked");

    int counter = 0;
    // before feedback layer, simple layer-by-layer propagation
    BOOST_FOREACH (boost::shared_ptr<layers::Layer<TDevice> > &layer, m_layers){
	if (counter == feedback_layer_idx)
	    break;
	layer->computeForwardPass(m_trainingState);
	counter++;
    }
	
    // inside a feedback block, step by step, layer by layer
    for (int timeStep = 0;
	 timeStep < misFuncs::getResoLength(curMaxSeqLength, timeResolution, 1);
	 timeStep++){
	
	counter = 0;

	BOOST_FOREACH (boost::shared_ptr<layers::Layer<TDevice> > &layer, m_layers){
	    if (counter > feedback_target_layer_idx)
		break;
	    if (counter >= feedback_layer_idx && counter <= feedback_target_layer_idx){
		layer->prepareStepGeneration(timeStep);
		layer->computeForwardPass(timeStep, m_trainingState);
	    }
	    counter++;
	}
    }

    // after feedback layer range
    counter = 0;
    BOOST_FOREACH (boost::shared_ptr<layers::Layer<TDevice> > &layer, m_layers){
	if (counter > feedback_target_layer_idx)
	    layer->computeForwardPass(m_trainingState);
	counter++;
    }

    return;
}

template <typename TDevice>
void NeuralNetwork<TDevice>::computeForwardPass(const int curMaxSeqLength,
						const real_t uttCnt)
{
   
    const Configuration &config = Configuration::instance();

    if (m_feedBackLayers.size() == 0 && m_feedBackHiddenLayers.size() == 0){
	// Propagate layer by layer
	this->__computeForward_LayerByLayer(curMaxSeqLength, uttCnt);
	
    }else if (m_feedBackLayers.size() > 0 && m_feedBackHiddenLayers.size() == 0) {
	// Two modes for AR model
	
	int scheduleSampOpt = config.scheduleSampOpt();

	switch (scheduleSampOpt){
	    
	case NN_FEEDBACK_GROUND_TRUTH:
	case NN_FEEDBACK_DROPOUT_1N:
	case NN_FEEDBACK_DROPOUT_ZERO:
	    {
		// Get feedback data (teacher forcing)
		// Then propagate layer by layer
		this->__computeForward_TeacherForce_LayerByLayer(curMaxSeqLength, uttCnt);
		break;
	    }
	case NN_FEEDBACK_SC_SOFT:
	case NN_FEEDBACK_SC_MAXONEHOT:
	case NN_FEEDBACK_SC_RADONEHOT:
	    {
		// Schedule sampling mode
		this->__computeForward_ScheduleSamp_LayerByLayer(curMaxSeqLength, uttCnt);
		break;
	    }
	default:
	    {
		throw std::runtime_error("Unknown AR model training mode");
	    }
	}
	
    }else if (m_feedBackLayers.size() == 0 && m_feedBackHiddenLayers.size() > 0) {
	// Model with feedback hidden layers
	this->__computeForward_StepByStep(curMaxSeqLength, uttCnt);

    }else{
	throw std::runtime_error("Unsupported network with feedback_hidden and feedback layers");
    }
}


template <typename TDevice>
void NeuralNetwork<TDevice>::__computeGenPass_LayerByLayer_mem(
						const data_sets::DataSetFraction &fraction,
						const int curMaxSeqLength, 
						const real_t generationOpt)
{
    int layerID = 0;

    /**
     * Step1. get the required GPU resources and allocate memory
     */
    if (m_vecPoolMng.empty()){
	// get network dependency map
	helpers::networkDepMng tmp_networkMng2 = this->m_networkMng;
	
	BOOST_FOREACH (boost::shared_ptr<layers::Layer<TDevice> > &layer, m_layers){
	    
	    // only count layers that can be optimized
	    if (this->flagLayerCanbeOptimizedMA(layerID)){
		if (!(tmp_networkMng2.get_layerDep(layerID).empty_towhich())){
		    layer->logAllBuffers(m_vecPoolMng, true);
		}
		
		// release the memory of layers that send input data to this layer
		BOOST_FOREACH (int fromwhich,
			       tmp_networkMng2.get_layerDep(layerID).get_fromwhich()){
		    
		    if (this->flagLayerCanbeOptimizedMA(fromwhich)){
			
			tmp_networkMng2.get_layerDep(fromwhich).del_towhich(layerID);
			
			if (tmp_networkMng2.get_layerDep(fromwhich).empty_towhich())
			    this->m_layers[fromwhich]->logAllBuffers(m_vecPoolMng, false);
		    }
		}
		// Delete the dependency fromwhich
		tmp_networkMng2.get_layerDep(layerID).nul_fromwhich();
	    }	    
	    layerID++;
	}
	m_vecPoolMng.createVecPool();
    }


    /**
     * Step2. layer by layer generation with memory swap
     */
    
    // get network dependency map
    helpers::networkDepMng tmp_networkMng = this->m_networkMng;

    // reset the layer counter
    layerID = 0;
    
    // layer by layer generation mode for Non-AR WaveModel
    BOOST_FOREACH (boost::shared_ptr<layers::Layer<TDevice> > &layer, m_layers){

	// print out the index every 10 layers
	if (layerID % 10 == 0) std::cout << layerID << " " << std::flush;
	
	if (!(this->flagLayerCanbeOptimizedMA(layerID))){	
	    // This layer can not be optimized in memory, do normal propagation
	    layer->loadSequences(fraction, m_trainingState);
	    layer->computeForwardPass(m_trainingState);
	    
	}else{
	    // If this layer can be optimized, do the memory-save operation
	    
	    // check each layer in fromwhich layers
	    BOOST_FOREACH (int fromwhich,
			   tmp_networkMng.get_layerDep(layerID).get_fromwhich()){
		if (fromwhich > 0 && this->m_layers[fromwhich]->outputs().size() == 0)
		    throw std::runtime_error("Error from which");
	    }

	    // allocate memory and do propagation
	    if (!(tmp_networkMng.get_layerDep(layerID).empty_towhich())){
		// layer->resizeAllBuffers(curMaxSeqLength);
		layer->swapAllBuffers(m_vecPoolMng, true);
		layer->loadSequences(fraction, m_trainingState);
		layer->computeForwardPass(m_trainingState);
	    }
	    
	    // release the memory of layers that send input data to this layer
	    BOOST_FOREACH (int fromwhich,
			   tmp_networkMng.get_layerDep(layerID).get_fromwhich()){
		    
		if (this->flagLayerCanbeOptimizedMA(fromwhich)){
		    // If the 'input' layer can be optimized
		    // Delete dependency temporarily
		    tmp_networkMng.get_layerDep(fromwhich).del_towhich(layerID);
		    
		    // If the 'input' layer is no longer needed by any layer, release the mem
		    if (tmp_networkMng.get_layerDep(fromwhich).empty_towhich()){
			//this->m_layers[fromwhich]->clearAllBuffers();
			this->m_layers[fromwhich]->swapAllBuffers(m_vecPoolMng, false);
		    }
		}
	    }
	    
	    // Delete the dependency fromwhich
	    tmp_networkMng.get_layerDep(layerID).nul_fromwhich();	    
	}
	layerID++;
    }
    return;
}

template <typename TDevice>
void NeuralNetwork<TDevice>::__computeGenPass_LayerByLayer(
					const data_sets::DataSetFraction &fraction,
					const int curMaxSeqLength, 
					const real_t generationOpt)
{
    // load sequences
    this->loadSequences(fraction);
    
    // Normal forward computation
    this->__computeForward_LayerByLayer(curMaxSeqLength, -1);

    layers::MDNLayer<TDevice> *olm;
    
    // if MDN is available, infer the output, or copy the MDN parameter vector
    olm = outMDNLayer();
    if (olm != NULL) olm->getOutput(generationOpt);
}

template <typename TDevice>
void NeuralNetwork<TDevice>::__computeGenPass_NormFlow(
					const data_sets::DataSetFraction &fraction,
					const int curMaxSeqLength, 
					const real_t generationOpt)
{
    this->loadSequences(fraction);
    
    // computation within the normal layers
    int cnt = 0;
    BOOST_FOREACH (boost::shared_ptr<layers::Layer<TDevice> > &layer, m_layers){
	// 
	if (cnt < m_normflowLayers[0] || cnt > m_normflowLayers[m_normflowLayers.size()-1])
	    layer->computeForwardPass(m_trainingState);
	cnt++;
    }

    layers::MDNLayer<TDevice> *olm;
    // generate normalzied output from MDN
    olm = outMDNLayer();
    if (olm != NULL) olm->getOutput(generationOpt);

    // de-transform the output from MDN by the normalizing flows
    // 1. load the output from MDN to the last normflow
    layers::NormFlowLayer<TDevice>* tmpPtr =
	dynamic_cast<layers::NormFlowLayer<TDevice>*>(m_layers[m_normflowLayers.back()].get());
    if (tmpPtr) tmpPtr->loadNormedOutput();
	
    // 2. de-transformation using the normflow layers
    for (size_t index = m_normflowLayers.size()-1; index > 0 ; index--){
	int layerIdx1 = m_normflowLayers[index];
	int layerIdx2 = m_normflowLayers[index-1];
	    
	// de-transformation between layerIdx1 and layerIdx2
	for (int timeStep = 0, cnt = 0; timeStep < curMaxSeqLength; timeStep ++, cnt = 0){
	    BOOST_FOREACH (boost::shared_ptr<layers::Layer<TDevice> > &layer, m_layers){
		if (cnt > layerIdx2 && cnt <= layerIdx1){
		    if (timeStep % (layer->getResolution()) == 0){
			// prepare the matrix (for rnn, lstm)
			layer->prepareStepGeneration(timeStep/layer->getResolution());
			
			// compute for 1 frame			
			layer->computeForwardPass(timeStep/layer->getResolution(),
						  m_trainingState);
		    }
		}
		cnt++;
	    }
	}	    
	// prepare for the next flow
	tmpPtr = dynamic_cast<layers::NormFlowLayer<TDevice>*>(m_layers[layerIdx2].get());
	if (tmpPtr) tmpPtr->loadNormedOutput();
    }
    // 3. copy the de-transformed output to the output of network
    m_layers[m_normflowLayers[0]]->computeForwardPass(0, m_trainingState);
}

template <typename TDevice>
void NeuralNetwork<TDevice>::__computeGenPass_VAE(const data_sets::DataSetFraction &fraction,
						  const int curMaxSeqLength, 
						  const real_t generationOpt)
{
    this->loadSequences(fraction);	
    // Inference for VAE
	
    // For a VAE network, the feedback layer in the encoder should take the
    // golden target features as input if we just want to extract latent
    // variables from the network. In this case, config.vaeEncoderOutputlayer() is
    // used to specify the layer to generate the latent variables
    const Configuration &config = Configuration::instance();
	
    if (config.vaeEncoderOutputLayer() >= m_totalNumLayers)
	throw std::runtime_error("vaeEncoderOutputLayer is larger than network depth");
    if (m_vaeLayer < 0)
	throw std::runtime_error("vaeEncoderOutputLayer() is used while network is not VAE");
	
    // Feedback the natural output data provided by data.nc
    this->postOutputLayer().retrieveFeedBackData();

    if (config.dropoutbeforeVAE() == 1){
	// Prepare the random seed
	static boost::mt19937 *gen = NULL;
	if (!gen) {
	    gen = new boost::mt19937;
	    gen->seed(config.randomSeed()+98); // any random number
	}
	boost::random::uniform_real_distribution<real_t> dist(0, 1);
	
	real_t threshold = ((real_t)config.scheduleSampPara())/100;
	
	// Prepare the random vector
	Cpu::real_vector randNum;
	randNum.reserve(curMaxSeqLength);
	for (size_t i = 0; i < curMaxSeqLength; ++i){
	    if (dist(*gen) > threshold){
		randNum.push_back(0);
	    }else{
		randNum.push_back(1);
	    }
	}

	// dropout 
	typename TDevice::real_vector temp = randNum;
	this->postOutputLayer().retrieveFeedBackData(temp, config.scheduleSampOpt());
    }
    
    // Assume no dropout here
    // propagate until the vae layer
    int cnt = 0 ;
    BOOST_FOREACH (boost::shared_ptr<layers::Layer<TDevice> > &layer, m_layers){
	if (cnt > m_vaeLayer) break; 
	layer->computeForwardPass(m_trainingState);
	cnt++;
    }
    //if (config.vaeEncoderOutputLayer() == m_totalNumLayers - 1){
    //  olm = outMDNLayer();
    //  if (olm != NULL) olm->getOutput(generationOpt);
    //}
    return;
}

template <typename TDevice>
void NeuralNetwork<TDevice>::__computeGenPass_StepByStep_AR(
					const data_sets::DataSetFraction &fraction,
					const int curMaxSeqLength, 
					const real_t generationOpt)
{
    this->loadSequences(fraction);
    const Configuration &config = Configuration::instance();
    // Feedback exists, and not for latent code inference
    layers::MDNLayer<TDevice> *olm;
    
    {	
	// Prepare the random seed
	static boost::mt19937 *gen = NULL;
	if (!gen) {
	    gen = new boost::mt19937;
	    gen->seed(config.randomSeed()+98); // any random number
	}
	boost::random::uniform_real_distribution<real_t> dist(0, 1);

	
	int scheduleSampOpt = config.scheduleSampOpt();
	int scheduleSampPara= config.scheduleSampPara();
	printf("SSAMPOpt: %d, SSAMPPara: %d\n", scheduleSampOpt, scheduleSampPara);
	
	real_t sampThreshold = 0.0;
	int    methodCode    = 0;
	int    cnt           = 0;

	if (this->m_vaeNetworkType != VAENETWORKTYPE_0){
	    // Special network of VAE
	    
	    switch (this->m_vaeNetworkType){
	    case VAENETWORKTYPE_1:
	    case VAENETWORKTYPE_3:
		// propagate for layers without AR dependency

		cnt = 0;
		BOOST_FOREACH (boost::shared_ptr<layers::Layer<TDevice> > &layer, m_layers){
		    if (cnt == m_firstFeedBackLayer) break;
		    layer->computeForwardPass(m_trainingState);
		    cnt++;
		}
		break;
	    case VAENETWORKTYPE_2:
		// encoder and condition network has no AR dependency,
		// just propagate through all layers
		// (assume latent code has been loaded in vae layer)
		this->computeForwardPass(curMaxSeqLength, -1);
		olm = outMDNLayer();
		if (olm != NULL) olm->getOutput(generationOpt);
		return;
		break;
	    case VAENETWORKTYPE_4:
	    default:
		throw std::runtime_error("Impossible error");
		break;
	    }
	    
	}else{
	    // For a normal network with feedback layers
	    
	    // layers without time dependency
	    // can be simultaneously calculated with all frames
	    // Forward computation for layers before Feedback
	    cnt = 0;
	    BOOST_FOREACH (boost::shared_ptr<layers::Layer<TDevice> > &layer, m_layers){
		if (cnt == m_firstFeedBackLayer) break; 
		layer->computeForwardPass(m_trainingState);
		cnt++;
	    }
	}
	    
	
	// Parameter for genreation
	switch (scheduleSampOpt){

	// Case 0: use probability vector for feedback
	//         1. native training approach
	//         2. schedule sampling (soft-feedback training)
	case NN_FEEDBACK_GROUND_TRUTH:
	case NN_FEEDBACK_SC_SOFT:
	    // always uses the soft vector (default option)
	    sampThreshold  = 1;
	    methodCode     = NN_FEEDBACK_GROUND_TRUTH;
	    break;

	// Case 1: use one hot vector
	case NN_FEEDBACK_SC_MAXONEHOT:
	    if (scheduleSampPara > 0){
		sampThreshold = 1;
		methodCode = NN_FEEDBACK_GROUND_TRUTH;
	    }else{
		sampThreshold = (-1.0 * (real_t)scheduleSampPara / 100.0);
		methodCode = NN_FEEDBACK_SC_MAXONEHOT;
	    }
	    // use the one-hot best
	    break;
	    
	// Case 2: dropout
	case NN_FEEDBACK_DROPOUT_1N:
	    methodCode = NN_FEEDBACK_DROPOUT_1N;
	    sampThreshold = ((real_t)scheduleSampPara)/100;
	    break;					    
	case NN_FEEDBACK_DROPOUT_ZERO:
	    methodCode = NN_FEEDBACK_DROPOUT_ZERO;
	    sampThreshold = ((real_t)scheduleSampPara)/100;
	    break;
	    
	// Case 3: beam search
	case NN_FEEDBACK_BEAMSEARCH:
	    methodCode = NN_FEEDBACK_SC_MAXONEHOT;
	    //beamSize   = (int)scheduleSampPara;
	    if (config.vaeEncoderOutputLayer() >= 0)
		throw std::runtime_error("vaeEncoderOutputLayer is implemented for beamsearch");
	    break;
	}


	//Generation
	// Normal generation (Greedy)
	if (scheduleSampOpt != NN_FEEDBACK_BEAMSEARCH){
	    
	    int feedBackFrom = m_firstFeedBackLayer;
	    
	    for (int timeStep = 0, cnt = 0; timeStep < curMaxSeqLength; timeStep ++, cnt = 0){
		
		BOOST_FOREACH (boost::shared_ptr<layers::Layer<TDevice> > &layer, m_layers){
		    if (cnt >= feedBackFrom){

			if (timeStep % (layer->getResolution()) == 0){
			    // prepare the matrix (for rnn, lstm)
			    layer->prepareStepGeneration(timeStep/layer->getResolution());
			    // compute for 1 frame			
			    layer->computeForwardPass(timeStep/layer->getResolution(),
						      m_trainingState);
			}
		    }
		    cnt++;
		}
		
		// Generate the output from MDN
		if (timeStep % (this->postOutputLayer().getResolution()) == 0){
		    int tmpOutputLayerReso = this->postOutputLayer().getResolution();
		    
		    olm = outMDNLayer();
		    if (olm != NULL) olm->getOutput(timeStep/olm->getResolution(), generationOpt);
		
		    // Feedback the data
		    if (dist(*gen) < sampThreshold){
			// default case: feedback prob vec
			this->postOutputLayer().retrieveFeedBackData(timeStep/tmpOutputLayerReso,
								     NN_FEEDBACK_GROUND_TRUTH);
		    }else{
			// special method: use one-hot or dropout
			this->postOutputLayer().retrieveFeedBackData(timeStep/tmpOutputLayerReso,
								     methodCode);
			// just display the timeStep a few times
			if (timeStep % 500 == 0)
			    std::cout << timeStep << " " << std::flush;
			//printf("%d ", timeStep);
		    }
		}
	    }
	    
	// Beam search generation
	}else{
	    
	    int stateNum;       // number of states per time step
	    int layerCnt;       // counter of the hidden layers
	    int beamSize   = (int)scheduleSampPara; // size of beam
	    
	    /* ----- pre-execution check  ----- */
	    if (beamSize < 0)
		throw std::runtime_error("beam size cannot be < 1");
	    if (m_firstFeedBackLayer < 0)
		throw std::runtime_error("No need to use beam size for network without feedback");
	    olm = outMDNLayer();
	    if (olm == NULL)
		throw std::runtime_error("Beam size is used for non-MDN output layer");
	    stateNum = olm->mdnParaDim();
	    if (beamSize >= stateNum)
		throw std::runtime_error("Beam size is larger than the number of state");

	    /* ----- initialization ----- */
	    // count the number of hidden elements in the network
	    std::vector<int> netStateSize;
	    int hidEleNum = 0;	// number of hidden elements in the network
	    
	    layerCnt  = 0;
	    BOOST_FOREACH (boost::shared_ptr<layers::Layer<TDevice> > &layer, m_layers){
		if (layerCnt >= m_firstFeedBackLayer){
		    netStateSize.push_back(layer->hiddenStateSize());
		    hidEleNum += layer->hiddenStateSize();
		}
		layerCnt++;
	    }
	    // allocate memory spaces for searching 
	    beamsearch::searchState<TDevice>  bmState(netStateSize, curMaxSeqLength, stateNum);
	    beamsearch::searchEngine<TDevice> bmEngine(beamSize);
	    for (int i = 0; i < beamSize + beamSize * stateNum; i++)
		bmEngine.addState(bmState);
	    bmEngine.setValidBeamSize(1);
	    std::vector<beamsearch::sortUnit> preSortVec(stateNum);
	    // allocate memory spaces for hidden features of network
	    Cpu::real_vector netStateTmpTmp(hidEleNum, 0.0);
	    typename TDevice::real_vector netStateTmp = netStateTmpTmp;


	    
	    /* ----- Search loop ----- */
	    for (int timeStep = 0; timeStep < curMaxSeqLength; timeStep++){

		// Count the extended number of states
		int stateCnt = 0;
		
		// loop over beam
		for (int searchPT = 0; searchPT < bmEngine.getValidBeamSize(); searchPT ++){

		    // get the state to be extended
		    beamsearch::searchState<TDevice>& bmState2 = bmEngine.retrieveState(searchPT);
		    
		    // prepare states from bmState2
		    bmState.liteCopy(bmState2);
		    
		    // set the network state
		    // 1. set the feedback data
		    if (timeStep > 0)
			this->postOutputLayer().setFeedBackData(timeStep-1, bmState2.getStateID());
		    
		    // 2. set the hidde layers and compute
		    layerCnt = 0;
		    BOOST_FOREACH (boost::shared_ptr<layers::Layer<TDevice> > &layer, m_layers){
			if (layerCnt >= m_firstFeedBackLayer){
			    int layerID = layerCnt - m_firstFeedBackLayer;
			    layer->prepareStepGeneration(timeStep);
			    if (timeStep > 0){
				netStateTmp = bmState2.getNetState(layerID);
				layer->setHiddenState(timeStep-1, netStateTmp);
			    }
			    layer->computeForwardPass(timeStep, m_trainingState);

			    // store the state of network in new states 
			    // this should be in step3. but this is more efficient
			    layer->retrieveHiddenState(timeStep, netStateTmp);
			    netStateTmpTmp = netStateTmp;
			    bmState.setNetState(layerID, netStateTmpTmp);
			}
			layerCnt++;
		    }
		    // 3. pre-select the states to be explored
		    for (int newStateID = 0; newStateID < stateNum; newStateID++){
			preSortVec[newStateID].prob = olm->retrieveProb(timeStep, newStateID);
			preSortVec[newStateID].idx  = newStateID;
		    }
		    std::sort(preSortVec.begin(), preSortVec.end(), beamsearch::compareFunc);
		    
		    // 4. add new search state
		    //  probability before this step
		    for (int i = 0; i < bmEngine.getBeamSize(); i++){
			
			bmState.setStateID(preSortVec[i].idx);
			bmState.setStateTrace(timeStep, preSortVec[i].idx);
			bmState.setTimeStep(timeStep);
			if (preSortVec[i].prob < 1e-15f)
			    continue; // trim the zero probability path
			else
			    bmState.setProb(bmState2.getProb() + std::log(preSortVec[i].prob));
			bmState.setProbTrace(timeStep, preSortVec[i].prob);
			bmEngine.setState(bmEngine.getBeamSize() + stateCnt, bmState);
			bmEngine.setSortUnit(stateCnt, bmState);
			stateCnt++;
		    }
		}	
		bmEngine.sortSet(stateCnt);
		bmEngine.printBeam();
	    }
	    
	    // Finish the beam search, finally, generate
	    bmEngine.printBeam();
	    bmState.fullCopy(bmEngine.retrieveState(0));
	    for (int timeStep = 0; timeStep < curMaxSeqLength; timeStep++){
		if (timeStep > 0)
		    this->postOutputLayer().setFeedBackData(timeStep-1,
							    bmState.getStateID(timeStep-1));
		
		layerCnt = 0;
		BOOST_FOREACH (boost::shared_ptr<layers::Layer<TDevice> > &layer, m_layers){
		    if (layerCnt >= m_firstFeedBackLayer){
			layer->prepareStepGeneration(timeStep);
			layer->computeForwardPass(timeStep, m_trainingState);
		    }
		    layerCnt++;
		}
		
		olm = outMDNLayer();
		if (olm != NULL) olm->getOutput(timeStep, generationOpt);
	    }
	    
	} // Beam search generation
	
	return;
    }
}
template <typename TDevice>
void NeuralNetwork<TDevice>::__computeGenPass_VAEwithMA(
				  const data_sets::DataSetFraction &fraction,
				  const int curMaxSeqLength, 
				  const real_t generationOpt)
{
    this->postOutputLayer().loadSequences(fraction, m_trainingState);
    this->postOutputLayer().retrieveFeedBackData();
    this->__computeGenPass_LayerByLayer_mem(fraction, curMaxSeqLength, generationOpt);
}

template <typename TDevice>
void NeuralNetwork<TDevice>::__computeGenPass_AE(
				  const data_sets::DataSetFraction &fraction,
				  const int curMaxSeqLength, 
				  const real_t generationOpt)
{
    this->postOutputLayer().loadSequences(fraction, m_trainingState);
    this->postOutputLayer().retrieveFeedBackData();
    this->__computeGenPass_LayerByLayer(fraction, curMaxSeqLength, generationOpt);
}

template <typename TDevice>
void NeuralNetwork<TDevice>::__computeGenPass_VAEwithAR(
				  const data_sets::DataSetFraction &fraction,
				  const int curMaxSeqLength, 
				  const real_t generationOpt)
{
    
    // Note: this method is exclusively used by VAE + AR model where the AR model only
    //      has one feedback link

    this->loadSequences(fraction);
    const Configuration &config = Configuration::instance();
    
    layers::MDNLayer<TDevice> *olm;
    
    {	
	// Prepare the random seed
	static boost::mt19937 *gen = NULL;
	if (!gen) {
	    gen = new boost::mt19937;
	    gen->seed(config.randomSeed()+98); // any random number
	}
	boost::random::uniform_real_distribution<real_t> dist(0, 1);

	
	int scheduleSampOpt = config.scheduleSampOpt();
	int scheduleSampPara= config.scheduleSampPara();
	printf("SSAMPOpt: %d, SSAMPPara: %d\n", scheduleSampOpt, scheduleSampPara);
	
	real_t sampThreshold = 0.0;
	int    methodCode    = 0;
	int    cnt           = 0;

	// assume the input to the encoder is stored in the targetLayer->output()
	this->postOutputLayer().retrieveFeedBackData();
	
	// assume the last feedback layer is the feedback link in VAE+AR model
	// do forward propagation until the Feedback look of AR model
	cnt = 0;
	BOOST_FOREACH (boost::shared_ptr<layers::Layer<TDevice> > &layer, m_layers){
	    if (cnt == m_feedBackLayers.back()) break; 
	    layer->computeForwardPass(m_trainingState);
	    cnt++;
	}
	    
	
	// Parameter for genreation
	switch (scheduleSampOpt){

	// Case 0: use probability vector for feedback
	//         1. native training approach
	//         2. schedule sampling (soft-feedback training)
	case NN_FEEDBACK_GROUND_TRUTH:
	case NN_FEEDBACK_SC_SOFT:
	    // always uses the soft vector (default option)
	    sampThreshold  = 1;
	    methodCode     = NN_FEEDBACK_GROUND_TRUTH;
	    break;

	// Case 1: use one hot vector
	case NN_FEEDBACK_SC_MAXONEHOT:
	    if (scheduleSampPara > 0){
		sampThreshold = 1;
		methodCode = NN_FEEDBACK_GROUND_TRUTH;
	    }else{
		sampThreshold = (-1.0 * (real_t)scheduleSampPara / 100.0);
		methodCode = NN_FEEDBACK_SC_MAXONEHOT;
	    }
	    // use the one-hot best
	    break;
	    
	// Case 2: dropout
	case NN_FEEDBACK_DROPOUT_1N:
	    methodCode = NN_FEEDBACK_DROPOUT_1N;
	    sampThreshold = ((real_t)scheduleSampPara)/100;
	    break;					    
	case NN_FEEDBACK_DROPOUT_ZERO:
	    methodCode = NN_FEEDBACK_DROPOUT_ZERO;
	    sampThreshold = ((real_t)scheduleSampPara)/100;
	    break;
	    
	// Case 3: beam search
	case NN_FEEDBACK_BEAMSEARCH:
	    methodCode = NN_FEEDBACK_SC_MAXONEHOT;
	    break;
	}


	//Generation
	// Normal generation (Greedy)
	if (scheduleSampOpt != NN_FEEDBACK_BEAMSEARCH){
	    
	    int feedBackFrom = m_feedBackLayers.back();
	    
	    for (int timeStep = 0, cnt = 0; timeStep < curMaxSeqLength; timeStep ++, cnt = 0){
		
		BOOST_FOREACH (boost::shared_ptr<layers::Layer<TDevice> > &layer, m_layers){
		    if (cnt >= feedBackFrom){

			if (timeStep % (layer->getResolution()) == 0){
			    // prepare the matrix (for rnn, lstm)
			    layer->prepareStepGeneration(timeStep/layer->getResolution());
			    // compute for 1 frame			
			    layer->computeForwardPass(timeStep/layer->getResolution(),
						      m_trainingState);
			}
		    }
		    cnt++;
		}
		
		// Generate the output from MDN
		if (timeStep % (this->postOutputLayer().getResolution()) == 0){
		    int tmpOutputLayerReso = this->postOutputLayer().getResolution();
		    
		    olm = outMDNLayer();
		    if (olm != NULL) olm->getOutput(timeStep/olm->getResolution(), generationOpt);
		
		    // Feedback the data
		    if (dist(*gen) < sampThreshold){
			// default case: feedback prob vec
			this->postOutputLayer().retrieveFeedBackData(timeStep/tmpOutputLayerReso,
								     NN_FEEDBACK_GROUND_TRUTH);
		    }else{
			// special method: use one-hot or dropout
			this->postOutputLayer().retrieveFeedBackData(timeStep/tmpOutputLayerReso,
								     methodCode);
			// just display the timeStep a few times
			if (timeStep % 500 == 0)
			    std::cout << timeStep << " " << std::flush;
		    }
		}
	    }
	    
	}else{
	    throw std::runtime_error("Beam search disabled for VAE+AR model");
	}
	return;
    }
}


template <typename TDevice>
void NeuralNetwork<TDevice>::__computeGenPass_StepByStep_RNN_FBH(
					const data_sets::DataSetFraction &fraction,
					const int curMaxSeqLength, 
					const real_t generationOpt)
{
    this->loadSequences(fraction);
    
    if (m_waveNetMemSaveFlag == NETWORK_WAVENET_SAVE_MA)
	throw std::runtime_error("To be implemented for combination of NSF with RNN_FBH");
    
    this->__computeForward_StepByStep(curMaxSeqLength, -1);
    layers::MDNLayer<TDevice> *olm;
    
    // if MDN is available, infer the output, or copy the MDN parameter vector
    olm = outMDNLayer();
    if (olm != NULL) olm->getOutput(generationOpt);

}

template <typename TDevice>
void NeuralNetwork<TDevice>::__computeGenPass_special_NSF_FBH(
					const data_sets::DataSetFraction &fraction,
					const int curMaxSeqLength, const real_t generationOpt)
{
    
    /* current implementation only supports on feedback hidden layer*/
    if (m_feedBackHiddenLayers.size() != 1)
	throw std::runtime_error("No support for network with multiple feedback_hidden layers");

    int feedback_layer_idx = m_feedBackHiddenLayers[0];
    int feedback_target_layer_idx = m_layers[m_feedBackHiddenLayers[0]]->returnTargetLayerID();
    int timeResolution = m_feedBackHiddenLayersTimeResos[0];
    
    if (feedback_target_layer_idx < feedback_layer_idx ||
	feedback_target_layer_idx > m_totalNumLayers)
	throw std::runtime_error("feedback_hidden target layer not linked");
    if (timeResolution <= 1)
	throw std::runtime_error("feedback_hidden should be in condition module");

    // Condition module, use the method of computeforward_stepbystep
    int counter = 0;
    // before feedback layer, simple layer-by-layer propagation
    BOOST_FOREACH (boost::shared_ptr<layers::Layer<TDevice> > &layer, m_layers){
	if (counter == feedback_layer_idx)
	    break;
	layer->loadSequences(fraction, m_trainingState);
	layer->computeForwardPass(m_trainingState);
	counter++;
    }
	
    // inside a feedback block, step by step, layer by layer
    for (int timeStep = 0;
	 timeStep < misFuncs::getResoLength(curMaxSeqLength, timeResolution, 1);
	 timeStep++){
	
	counter = 0;

	BOOST_FOREACH (boost::shared_ptr<layers::Layer<TDevice> > &layer, m_layers){
	    if (counter > feedback_target_layer_idx)
		break;
	    if (counter >= feedback_layer_idx && counter <= feedback_target_layer_idx){
		layer->loadSequences(fraction, m_trainingState);
		layer->prepareStepGeneration(timeStep);
		layer->computeForwardPass(timeStep, m_trainingState);
	    }
	    counter++;
	}
    }

    // LayerByLayer_mem
    // Make a copy of the network layer dependency
    helpers::networkDepMng tmp_networkMng = this->m_networkMng;
	
    int layerID = 0;
    BOOST_FOREACH (boost::shared_ptr<layers::Layer<TDevice> > &layer, m_layers){

	if (layerID <= feedback_target_layer_idx){
	    // nothing
	}else{
		
	    // print out the index every 10 layers
	    if (layerID % 10 == 0) std::cout << layerID << " " << std::flush;

	    
	    if (!(this->flagLayerCanbeOptimizedMA(layerID))){	
		// This layer can not be optimized in memory, do normal propagation
		layer->loadSequences(fraction, m_trainingState);
		layer->computeForwardPass(m_trainingState);    
	    }else{

		// If this layer can be optimized, do the memory-save operation
		
		// check each layer in fromwhich layers
		BOOST_FOREACH (int fromwhich,
			       tmp_networkMng.get_layerDep(layerID).get_fromwhich()){
		    if (fromwhich > 0 && this->m_layers[fromwhich]->outputs().size() == 0)
			throw std::runtime_error("Error from which");
		}

		if (tmp_networkMng.get_layerDep(layerID).empty_towhich()){
		    // no other layer needs the output of this layer, skip propagation
		}else{
		    layer->resizeAllBuffers(curMaxSeqLength);
		    layer->loadSequences(fraction, m_trainingState);
		    layer->computeForwardPass(m_trainingState);
		}	
		// release the memory of layers that send input data to this layer
		BOOST_FOREACH (int fromwhich,
			       tmp_networkMng.get_layerDep(layerID).get_fromwhich()){
		    
		    if (this->flagLayerCanbeOptimizedMA(fromwhich)){
			// If the 'input' layer can be optimized
			// Delete dependency temporarily
			tmp_networkMng.get_layerDep(fromwhich).del_towhich(layerID);
			// If the 'input' layer is no longer needed by any layer, release the mem
			if (tmp_networkMng.get_layerDep(fromwhich).empty_towhich())
			    this->m_layers[fromwhich]->clearAllBuffers();
		    }
		}
		// Delete the dependency fromwhich
		tmp_networkMng.get_layerDep(layerID).nul_fromwhich();	    
	    }
	}
	layerID++;
    }
    return;
}


template <typename TDevice>
void NeuralNetwork<TDevice>::computeForwardPassGen(const data_sets::DataSetFraction &fraction,
						   const int curMaxSeqLength, 
						   const real_t generationOpt)
{

    const Configuration &config = Configuration::instance();

    if (m_waveNetMemSaveFlag != NETWORK_WAVENET_SAVE_NO){
	// special networks with memmory save modes on (WaveNet/NSF ...)

	if (!m_normflowLayers.empty())
	    throw std::runtime_error("Error computeForwardPassGen: unsupported normflow waveMem");

	// VAE with other models
	if (m_vaeLayer >= 0){
	    if (m_waveNetMemSaveFlag == NETWORK_WAVENET_SAVE_AR){
		// VAE + AR (WaveNet model)
		this->__computeGenPass_VAEwithAR(fraction, curMaxSeqLength, generationOpt);
	    }else{
		// VAR + MA (NSF model)
		this->__computeGenPass_VAEwithMA(fraction, curMaxSeqLength, generationOpt);
	    }
	    return;
	}

	// WaveNet
	if (m_waveNetMemSaveFlag == NETWORK_WAVENET_SAVE_AR &&
	    m_feedBackHiddenLayers.empty() && (!m_feedBackLayers.empty())){
	    this->__computeGenPass_StepByStep_AR(fraction, curMaxSeqLength, generationOpt);
	    return;
	}
	
	// NSF
	if (m_waveNetMemSaveFlag == NETWORK_WAVENET_SAVE_MA){

	    // normal NSF without any feedback feedback_hidden
	    if (m_feedBackHiddenLayers.empty() && m_feedBackLayers.empty()){
		this->__computeGenPass_LayerByLayer_mem(fraction, curMaxSeqLength, generationOpt);
		return;
	    }
	    // NSF with feedback hidden link (maybe in acoustic model part)
	    if ((!m_feedBackHiddenLayers.empty()) && m_feedBackLayers.empty()){
		this->__computeGenPass_special_NSF_FBH(fraction, curMaxSeqLength, generationOpt);
		return;
	    }
	    // NSF with feedback link (maybe for NSF + VAE / AE)
	    if (m_feedBackHiddenLayers.empty() && (!m_feedBackLayers.empty())){
		this->__computeGenPass_VAEwithMA(fraction, curMaxSeqLength, generationOpt);
	    }
	}


	
    }else{
	// common models

	if(m_feedBackHiddenLayers.empty() && m_feedBackLayers.empty() && m_normflowLayers.empty()){
	    // for normal network
	    this->__computeGenPass_LayerByLayer(fraction, curMaxSeqLength, generationOpt);
	    return;
	}else if((!m_normflowLayers.empty()) &&
		 m_feedBackHiddenLayers.empty() && m_feedBackLayers.empty()){
	    // for normal network with normalizing flow
	    this->__computeGenPass_NormFlow(fraction, curMaxSeqLength, generationOpt);
	    return;
	}else if ((!m_feedBackLayers.empty()) &&
		  m_feedBackHiddenLayers.empty() && m_normflowLayers.empty()){

	    if (config.ARmodelSpecialGenMode() == 1){
		// use AR model as encoding -> decoding model (the data to be encoded/decoded
		// are saved in target->outputs())
		this->__computeGenPass_AE(fraction, curMaxSeqLength, generationOpt);
		return;
	    }else{
		// for models with feedback links (from target layer to hidden layers)
		if (m_vaeLayer >= 0 && config.vaeEncoderOutputLayer()>0)
		    // for VAE network
		    this->__computeGenPass_VAE(fraction, curMaxSeqLength, generationOpt);
		else
		    // for AR network
		    this->__computeGenPass_StepByStep_AR(fraction, curMaxSeqLength, generationOpt);
		return;
	    }
	    
	}else if ((!m_feedBackHiddenLayers.empty()) &&
		  m_feedBackLayers.empty() && m_normflowLayers.empty()){
	    // for RNN with feedback among hidden layers
	    this->__computeGenPass_StepByStep_RNN_FBH(fraction, curMaxSeqLength, generationOpt);
	    return;
	}else{
	    throw std::runtime_error("Error computeForwardPassGen: unsupported model");
	}
    }
    
    // Done
}


template <typename TDevice>
void NeuralNetwork<TDevice>::__computeBackward_LayerByLayer(const int curMaxSeqLength,
							    const real_t uttCnt)
{
    size_t layerCnt = m_totalNumLayers;
    
    BOOST_REVERSE_FOREACH (boost::shared_ptr<layers::Layer<TDevice> > &layer, m_layers) {

	layerCnt--;
	
	// runningMode
	if (Configuration::instance().runningMode() == NETWORKRUNNING_MODE_STOPLR0){

	    // Stop the backpropagation when the layer's learning rate is specified as 0
	    layers::TrainableLayer<TDevice> *trainableLayer = 
		dynamic_cast<layers::TrainableLayer<TDevice>*>(layer.get());
	    if (trainableLayer && misFuncs::closeToZero(trainableLayer->learningRate()))
		break;

	    // Or, stop if it is a mdn output layer in acoustic model
	    // Note, this is specific for GAN
	    /* To be revised for non-gan network */
	    layers::PostOutputLayer<TDevice> *mdnlayer = 
		dynamic_cast<layers::PostOutputLayer<TDevice>*>(layer.get());
	    if (mdnlayer && mdnlayer->postLayerType() == NN_POSTOUTPUTLAYER_NOTLASTMDN)
		break;
	    
	}else if (Configuration::instance().runningMode() == NETWORKRUNNING_MODE_STOPTEACHER &&
		  !m_distillingLayers.empty()){
	    
	    // for distilling network, skip backpropagation in the teacher network
	    if (layerCnt < m_totalNumLayers - 2 && layerCnt > m_distillingLayers.back())
		continue;

	    // for distilling network, skip backpropgation in the common conditioning network
	    layers::TrainableLayer<TDevice> *trainableLayer = 
		dynamic_cast<layers::TrainableLayer<TDevice>*>(layer.get());
	    if (trainableLayer && misFuncs::closeToZero(trainableLayer->learningRate()))
		continue;

	}else if (Configuration::instance().runningMode() == NETWORKRUNNING_MODE_DISTEACHER &&
		  !m_distillingLayers.empty()){

	    // not using the Gradients calculated by maximum likelihood in postoutput layer
	    if (layerCnt ==  m_totalNumLayers - 1)
		continue;
	    // propagate through all the layers
	}
	
        layer->computeBackwardPass(m_trainingState);
	
	// For debugging
	//std::cout << "output errors " << layer->name() << std::endl;
	//thrust::copy(layer->outputErrors().begin(), layer->outputErrors().end(), 
	// std::ostream_iterator<real_t>(std::cout, ";"));
	//std::cout << std::endl;
    }
}

template <typename TDevice>
void NeuralNetwork<TDevice>::__computeBackward_StepByStep(const int curMaxSeqLength,
							  const real_t uttCnt)
{
    
    int feedback_layer_idx = m_feedBackHiddenLayers[0];
    int feedback_target_layer_idx = m_layers[m_feedBackHiddenLayers[0]]->returnTargetLayerID();
    int timeResolution = m_feedBackHiddenLayersTimeResos[0];

    const Configuration &config = Configuration::instance();
    
    if (feedback_target_layer_idx < feedback_layer_idx ||
	feedback_target_layer_idx > m_totalNumLayers)
	throw std::runtime_error("feedback_hidden target layer not linked");

    // After the feedback block, propagate backwards layer by layer
    int counter = m_totalNumLayers-1;
    BOOST_REVERSE_FOREACH (boost::shared_ptr<layers::Layer<TDevice> > &layer, m_layers){
	if (counter == feedback_target_layer_idx)
	    break;

	if (this->__stopBackPropagation(counter, config.runningMode()))
	    return;
	
	layer->computeBackwardPass(m_trainingState);
	counter--;
    }

    // Within the feedback block, step by step, layer by layer
    for (int timeStep = misFuncs::getResoLength(curMaxSeqLength, timeResolution, 1) - 1;
	 timeStep >= 0;
	 timeStep--){
	
	counter = m_totalNumLayers-1;

	BOOST_REVERSE_FOREACH (boost::shared_ptr<layers::Layer<TDevice> > &layer, m_layers){
	    if (counter < feedback_layer_idx)
		break;
	    if (counter >= feedback_layer_idx && counter <= feedback_target_layer_idx)
		layer->computeBackwardPass(timeStep, m_trainingState);
	    counter--;
	}
    }

    // Before the feedback block
    counter = m_totalNumLayers-1;
    BOOST_REVERSE_FOREACH (boost::shared_ptr<layers::Layer<TDevice> > &layer, m_layers){
	if (counter < feedback_layer_idx){
	    if (this->__stopBackPropagation(counter, config.runningMode()))
		return;
	    layer->computeBackwardPass(m_trainingState);
	}
	counter--;
    }
    
    return;
}




template <typename TDevice>
void NeuralNetwork<TDevice>::computeBackwardPass(const int curMaxSeqLength, const real_t uttCnt)
{
    
    if (m_feedBackHiddenLayers.size() > 0) {
	// Model with feedback hidden layers
	this->__computeBackward_StepByStep(curMaxSeqLength, uttCnt);
    }else{
	// other modes
	this->__computeBackward_LayerByLayer(curMaxSeqLength, uttCnt);
    }
}

template <typename TDevice>
void NeuralNetwork<TDevice>::cleanGradientsForDiscriminator()
{
    // For GAN
    if (flagNetworkForGAN() && m_trainingState == NN_STATE_GAN_GEN){
	// clean the discrminator gradients when only generator is trained
	int cnt = 0;
	BOOST_FOREACH (boost::shared_ptr<layers::Layer<TDevice> > &layer, m_layers) {
	    if (cnt > m_middlePostOutputLayer)
		layer->cleanGradidents();
	    cnt++;
	}
    }

    // For general usage
    
    
}


template <typename TDevice>
real_t NeuralNetwork<TDevice>::calculateError(const bool flagGenerateMainError) const
{
    if (flagNetworkForGAN()){
	// GAN 
	
	if (flagGenerateMainError){
	    // return MSE error (or likelihood)
	    return static_cast<layers::PostOutputLayer<TDevice>&>(
			*m_layers[m_middlePostOutputLayer]).calculateError();
	}else{
	    // return the discriminator classification rate
	    return static_cast<layers::PostOutputLayer<TDevice>&>(
			*m_layers.back()).calculateError();
	}
    }else if (m_vaeLayer > 0){
	// VAE

	if (flagGenerateMainError){
	    // return the MSE (or likelihood) on target features
	    return static_cast<layers::PostOutputLayer<TDevice>&>(
			*m_layers.back()).calculateError();
	}else if (m_layers[m_vaeLayer]->type() == "vae"){
	    // return the KL divergence
	    return static_cast<layers::VaeMiddleLayer<TDevice>&>(
			*m_layers[m_vaeLayer]).vaeKLD();
	}else if (m_layers[m_vaeLayer]->type() == "vqlayer"){
	    // return the MSE between inferred latent data and codes
	    real_t tmpError = 0.0;
	    BOOST_FOREACH (int vaeLayerID, m_vaeLayers){
		tmpError += static_cast<layers::vqLayer<TDevice>&>(
				*m_layers[vaeLayerID]).codeError();
	    }
	    return tmpError;
	}else{
	    return 0;
	}
    }else if (!m_normflowLayers.empty()){
	// Normalizing flow
	if (flagGenerateMainError){
	    real_t negLog = static_cast<layers::PostOutputLayer<TDevice>&>(
			*m_layers.back()).calculateError();

	    for (int i = 1; i < m_normflowLayers.size(); i++){
		negLog  -= static_cast<layers::NormFlowLayer<TDevice>&>(
				*m_layers[m_normflowLayers[i]]).likelihoodJac();
	    }
	    return negLog;	    
	}else{
	    real_t negLog = 0.0;
	    // This computation is the same as the above above, but without error from MDN
	    for (int i = 1; i < m_normflowLayers.size(); i++){
		negLog  -= static_cast<layers::NormFlowLayer<TDevice>&>(
				*m_layers[m_normflowLayers[i]]).likelihoodJac();
	    }
	    return negLog;	    
	}
    }else if (!m_distillingLayers.empty()){
	
	// Distilling network
	if(flagGenerateMainError){
	    return static_cast<layers::PostOutputLayer<TDevice>&>(
			*m_layers.back()).calculateError();
	}else{
	    real_t tmpError = 0.0;
	    BOOST_FOREACH (int distillingLayerID, m_distillingLayers){
		tmpError += m_layers[distillingLayerID]->intermediateError();
	    }	    
	    return tmpError;
	}
	
    }else if (!m_interMetricLayers.empty()){
	
	// With intermediate metrics
	if(flagGenerateMainError){
	    return static_cast<layers::PostOutputLayer<TDevice>&>(
			*m_layers.back()).calculateError();
	}else{
	    real_t tmpError = 0.0;
	    BOOST_FOREACH (int layerID, m_interMetricLayers){
		tmpError += static_cast<layers::InterMetricLayer<TDevice>&>(
					*m_layers[layerID]).calculateError();
	    }	    
	    return tmpError;
	}
	
    }else{
	// Normal network
	if(flagGenerateMainError)
	    return static_cast<layers::PostOutputLayer<TDevice>&>(
			*m_layers.back()).calculateError();
	else
	    return 0;
    }
}

template <typename TDevice>
void NeuralNetwork<TDevice>::exportLayers(const helpers::JsonDocument& jsonDoc) const
{
    if (!jsonDoc->IsObject())
        throw std::runtime_error("JSON document root must be an object");

    // create the layers array
    rapidjson::Value layersArray(rapidjson::kArrayType);

    // create the layer objects
    for (size_t i = 0; i < m_totalNumLayers; ++i)
        m_layers[i]->exportLayer(&layersArray, &jsonDoc->GetAllocator());

    // if the section already exists, we delete it first
    if (jsonDoc->HasMember("layers"))
        jsonDoc->RemoveMember("layers");

    // add the section to the JSON document
    jsonDoc->AddMember("layers", layersArray, jsonDoc->GetAllocator());
}

template <typename TDevice>
void NeuralNetwork<TDevice>::exportWeights(const helpers::JsonDocument& jsonDoc) const
{
    if (!jsonDoc->IsObject())
        throw std::runtime_error("JSON document root must be an object");

    // create the weights object
    rapidjson::Value weightsObject(rapidjson::kObjectType);

    // create the weight objects
    BOOST_FOREACH (const boost::shared_ptr<layers::Layer<TDevice> > &layer, m_layers) {
    	layers::TrainableLayer<TDevice> *trainableLayer = 
	    dynamic_cast<layers::TrainableLayer<TDevice>*>(layer.get());
        if (trainableLayer){
            trainableLayer->exportWeights(&weightsObject, &jsonDoc->GetAllocator());
	}else{
	    // Modify 0507 Wang: for mdn PostProcess Layer
	    layers::MDNLayer<TDevice> *mdnlayer = 
		dynamic_cast<layers::MDNLayer<TDevice>*>(layer.get());
	    if (mdnlayer)
		mdnlayer->exportConfig(&weightsObject, &jsonDoc->GetAllocator());
	}
    }

    // if the section already exists, we delete it first
    if (jsonDoc->HasMember("weights"))
        jsonDoc->RemoveMember("weights");

    // add the section to the JSON document
    jsonDoc->AddMember("weights", weightsObject, jsonDoc->GetAllocator());
}

template <typename TDevice>
std::vector<std::vector<std::vector<real_t> > > NeuralNetwork<TDevice>::getOutputs(
	const real_t mdnoutput)
{
    /* All the features should have been generated and saved, by calling computeForwardPassGen.
       This function only retrieves the desired features from the specified layer, and save them
       into CPU vectors.
     */
    
    /* -----  Get configuration ----- */
    const Configuration &config = Configuration::instance();

    // specify the layer for generating output features (may not be the last layer)
    const int  layerID        = config.outputFromWhichLayer();
    
    // whether to output highway gate control signals from highway layers?
    const bool flagGateOutput = config.outputFromGateLayer();

    // whether to output VQ index from vqLayers
    const int  flagVQIndex    = config.vaeGetVQIndex();

    // if we only want to save a few frames from the output sequences
    const int  saveFrameNum   = config.outputFrameNum();
    
    if (saveFrameNum > 0)
	printf("\n\tOnly save the first %d frames from the output sequence", saveFrameNum);
    
    /* ----- Variables and initialization ----- */
    // buffer to store output features
    std::vector<std::vector<std::vector<real_t> > > outputs;

    // pointer to highway layers (SkipParaLayer)
    layers::SkipLayer<TDevice> *olg;
    // pointer to MDN layer
    layers::MDNLayer<TDevice>  *olm;
    // pointer to VQVAE layer
    layers::vqLayer<TDevice>   *olq;

    // Generation methods
    unsigned char genMethod;
    enum genMethod {ERROR = 0,VQINDEX,GATEOUTPUT, MDNSAMPLING, MDNPARAMETER, MDNEMGEN, NORMAL};

    // a temporary layer ID variable 
    int tempLayerID;

    
    /* Old CURRENNT generation methods
      specify old, olm, tempLayerId
       -3.0 is chosen for convience.
       
       < -3.0: no MDN generation
       > -3.0 && < -1.5: generating EM-style
       > -1.5 && < 0.0: generate MDN parameters (mdnoutput = -1.0)
       > 0.0 : generate samples from MDN with the variance = variance * mdnoutput 
    if (mdnoutput >= -3.0 && getGateOutput){
	genMethod = ERROR;
	throw std::runtime_error("MDN output and gate output can not be generated together");

    }else if (mdnoutput < -3.0 && getGateOutput){
	olg = outGateLayer(layerID);
	olm = NULL;
	tempLayerId = layerID;
	if (olg == NULL)
	    throw std::runtime_error("Gate output tap ID invalid\n");
	genMethod = GATEOUTPUT;

    }else if (mdnoutput >= -3.0 && !getGateOutput){
	olg = NULL;
	olm = outMDNLayer();
	if (olm == NULL)
	    throw std::runtime_error("No MDN layer in the current network");
	//olm->getOutput(mdnoutput); // Move to computeForward(curMaxSeqLength, generationOpt)
	tempLayerId = m_totalNumLayers-1;
	genMethod = (mdnoutput < 0.0) ? ((mdnoutput < -1.5) ? MDNEMGEN:MDNPARAMETER):MDNSAMPLING;
	
    }else{
	olg = NULL;
	olm = NULL;
	tempLayerId = layerID;
	genMethod = NORMAL;
    }*/

    /* Since we move the olm->getOutput(mdnoutput) to computeForwardPassGen, mdnoutput is not 
       necessay here
    */


    /* ----- Determine the output layer ----- */    
    if (layerID < 0){
	// If layerID is not specified, generate from the last output/postoutput layer
	olg = NULL;
	olq = NULL;
	
	olm = outMDNLayer(-1);
	if (olm == NULL)
	    tempLayerID = this->m_totalNumLayers-2; // layer be postouput
	else
	    tempLayerID = this->m_totalNumLayers-1; // MDN postoutput layer
	
    }else{
	
	// If layerID is specified, generate from that layer
	if (flagGateOutput){
	    // generate from Highway gate
	    olg = outGateLayer(layerID);
	    olm = NULL;
	    olq = NULL;
	    if (olg == NULL) throw std::runtime_error("Gate output tap ID invalid\n");
	}else if (flagVQIndex){
	    // generate from vqLayer
	    olg = NULL;
	    olm = NULL;
	    olq = outvqLayer(layerID);
	    if (olq == NULL) throw std::runtime_error("vqLayer ID invalid\n");	    
	}else{
	    // generate from specified layerID
	    olg = NULL;
	    olm = outMDNLayer(layerID);
	    olq = NULL;
	}
	tempLayerID = layerID;
    }

    
    /* ----- Determine the generation method ----- */
    // Both output layer type and generation options are taken into consideration
    if (olg == NULL && olq == NULL){
	if (olm == NULL)
	    // output from the layer output
	    genMethod = NORMAL;
	else
	    // output from the MDN layer
	    genMethod = (mdnoutput<0.0) ? ((mdnoutput < -1.5) ? MDNEMGEN:MDNPARAMETER):MDNSAMPLING;
    }else if (olq == NULL){
	// output from the highway gate
	genMethod = GATEOUTPUT;
    }else{
	// output from vqLayer by index
	genMethod = VQINDEX;
    }

    /* ----- Retrieve the output from the output layer ------ */
    layers::Layer<TDevice> &ol      = outputLayer(tempLayerID);
    Cpu::pattype_vector tmpPatTypes = ol.patTypes();
    
    for (int patIdx = 0; patIdx < (int)ol.patTypes().size(); ++patIdx) {

	// If we only save a few frames
	if (saveFrameNum > 0 && patIdx == saveFrameNum)
	    break;
	    
	switch (tmpPatTypes[patIdx]) {
	case PATTYPE_FIRST:
	    outputs.resize(outputs.size() + 1);
	    
	case PATTYPE_NORMAL:
	case PATTYPE_LAST: {
	    switch (genMethod){
	    case MDNEMGEN:
	    case MDNSAMPLING:
	    case NORMAL:
		{
		    // Save output features from output layers
		    Cpu::real_vector pattern(ol.outputs().begin() + patIdx * ol.size(), 
					     ol.outputs().begin() + (patIdx+1) * ol.size());
		    int psIdx = patIdx % ol.parallelSequences();
		    outputs[psIdx].push_back(std::vector<real_t>(pattern.begin(), pattern.end()));
		    break;
		}
	    case MDNPARAMETER:
		{
		    // Save MDN parameters from MDN layers
		    Cpu::real_vector pattern(
				olm->mdnParaVec().begin()+patIdx*olm->mdnParaDim(), 
				olm->mdnParaVec().begin()+(patIdx+1)*olm->mdnParaDim());
		    int psIdx = patIdx % ol.parallelSequences();
		    outputs[psIdx].push_back(std::vector<real_t>(pattern.begin(), pattern.end()));
		    break;
		}
	    case GATEOUTPUT:
		{
		    // Save Highway gate output from Highway layers
		    Cpu::real_vector pattern(olg->outputFromGate().begin() + patIdx * ol.size(),
					     olg->outputFromGate().begin()+(patIdx+1) * ol.size());
		    int psIdx = patIdx % ol.parallelSequences();
		    outputs[psIdx].push_back(std::vector<real_t>(pattern.begin(), pattern.end()));
		    break;
		}
	    case VQINDEX:
		{
		    // Save VQVAE code index (maybe from multiple codebooks) from the vqlayer
		    Cpu::real_vector pattern(
			olq->codeIdx().begin() + patIdx     * olq->codeBookNum(),
			olq->codeIdx().begin() + (patIdx+1) * olq->codeBookNum());
		    
		    int psIdx = patIdx % ol.parallelSequences();
		    outputs[psIdx].push_back(std::vector<real_t>(pattern.begin(), pattern.end()));
		    break;
		}
	    default:
		break;   
	    }

	}
	default:
	    break;
	}
    }

    return outputs;
}


/* Add 16-02-22 Wang: for WE updating */
// Initialization for using external WE bank
// (read in the word embeddings and save them in a matrix)
template <typename TDevice>
bool NeuralNetwork<TDevice>::initWeUpdate(const std::string weBankPath, const unsigned weDim, 
					  const unsigned weIDDim, const unsigned maxLength)
{
    // check if only the first layer is an input layer
    layers::InputLayer<TDevice>* inputLayer = 
	dynamic_cast<layers::InputLayer<TDevice>*>(m_layers.front().get());
    if (!inputLayer)
	throw std::runtime_error("The first layer is not an input layer");
    else if (!inputLayer->readWeBank(weBankPath, weDim, weIDDim, maxLength)){
	throw std::runtime_error("Fail to initialize for we updating");
    }
}

template <typename TDevice>
bool NeuralNetwork<TDevice>::initWeNoiseOpt(const int weNoiseStartDim, const int weNoiseEndDim,
					    const real_t weNoiseDev)
{
    // check if only the first layer is an input layer
    layers::InputLayer<TDevice>* inputLayer = 
	dynamic_cast<layers::InputLayer<TDevice>*>(m_layers.front().get());
    if (!inputLayer)
	throw std::runtime_error("The first layer is not an input layer");
    else if (!inputLayer->initWeNoiseOpt(weNoiseStartDim, weNoiseEndDim, weNoiseDev)){
	throw std::runtime_error("Fail to initialize for we updating");
    }
    return true;
}



// check whether the input layer uses external we bank
template <typename TDevice>
bool NeuralNetwork<TDevice>::flagInputWeUpdate() const
{
    layers::InputLayer<TDevice>* inputLayer = 
	dynamic_cast<layers::InputLayer<TDevice>*>(m_layers.front().get());
    if (!inputLayer){
	throw std::runtime_error("The first layer is not an input layer");
	return false;
    }
    else
	return inputLayer->flagInputWeUpdate();
}

// save the updated we bank in the input layer
template <typename TDevice>
bool NeuralNetwork<TDevice>::saveWe(const std::string weFile) const
{
    layers::InputLayer<TDevice>* inputLayer = 
	dynamic_cast<layers::InputLayer<TDevice>*>(m_layers.front().get());
    if (!inputLayer){
	throw std::runtime_error("The first layer is not an input layer");
	return false;
    }
    else
	return inputLayer->saveWe(weFile);
}

/* Add 0401 Wang: for MSE weight initialization*/
template <typename TDevice>
bool NeuralNetwork<TDevice>::initMseWeight(const std::string mseWeightPath)
{
    
    layers::PostOutputLayer<TDevice>* outputLayer = 
	dynamic_cast<layers::PostOutputLayer<TDevice>*>(m_layers.back().get());
    if (!outputLayer){
	throw std::runtime_error("The output layer is not a postoutput layer");
	return false;
    }
    else
	return outputLayer->readMseWeight(mseWeightPath);
   
}

/* Add 0413 Wang: for weight mask */
template <typename TDevice>
bool NeuralNetwork<TDevice>::initWeightMask(const std::string weightMaskPath,
					    const int         weightMaskOpt)
{
    std::ifstream ifs(weightMaskPath.c_str(), std::ifstream::binary | std::ifstream::in);
    if (!ifs.good())
	throw std::runtime_error(std::string("Fail to open") + weightMaskPath);
    
    // get the number of we data
    std::streampos numEleS, numEleE;
    long int numEle;
    numEleS = ifs.tellg();
    ifs.seekg(0, std::ios::end);
    numEleE = ifs.tellg();
    numEle  = (numEleE-numEleS)/sizeof(real_t);
    ifs.seekg(0, std::ios::beg);

    real_t tempVal;
    std::vector<real_t> tempVec;
    for (unsigned int i = 0; i<numEle; i++){
	ifs.read ((char *)&tempVal, sizeof(real_t));
	tempVec.push_back(tempVal);
    }
    
    printf("Initialize weight mask: %d mask elements in total, ", (int)numEle);
    printf("under the mode %d", weightMaskOpt);
    
    int pos = 0;
    if (weightMaskOpt > 0){
	printf("\n\tRead mask for embedded vectors ");
	layers::InputLayer<TDevice>* inputLayer = 
	    dynamic_cast<layers::InputLayer<TDevice>*>((m_layers[0]).get());
	pos = inputLayer->readWeMask(tempVec.begin());
	printf("(%d elements)", pos);
    }

    if (weightMaskOpt == 0 || weightMaskOpt==2){
	printf("\n\tRead mask for NN weights (");
	BOOST_FOREACH (boost::shared_ptr<layers::Layer<TDevice> > &layer, m_layers){
	    layers::TrainableLayer<TDevice>* weightLayer = 
		dynamic_cast<layers::TrainableLayer<TDevice>*>(layer.get());
	    if (weightLayer){
		if (weightLayer->weightNum()+pos > numEle){
		    throw std::runtime_error(std::string("Weight mask input is not long enough"));
		}else{
		    weightLayer->readWeightMask(tempVec.begin()+pos, 
						tempVec.begin()+pos+weightLayer->weightNum());
		    pos = pos+weightLayer->weightNum();
		}
		printf("%d ", weightLayer->weightNum());
	    }
	}
	printf("elements)");
    }
    printf("\n");
}

template <typename TDevice>
void NeuralNetwork<TDevice>::maskWeight()
{
    // mask the embedded vectors (if applicable)
    layers::InputLayer<TDevice>* inputLayer = 
	dynamic_cast<layers::InputLayer<TDevice>*>((m_layers[0]).get());
    inputLayer->maskWe();

    // mask the weight (always do, as the default mask value is 1.0)
    BOOST_FOREACH (boost::shared_ptr<layers::Layer<TDevice> > &layer, m_layers){
	layers::TrainableLayer<TDevice>* weightLayer = 
	    dynamic_cast<layers::TrainableLayer<TDevice>*>(layer.get());
	if (weightLayer){
	    weightLayer->maskWeight();
	}
    }
}

template <typename TDevice>
void NeuralNetwork<TDevice>::notifyCurrentEpoch(const int trainingEpoch)
{
    BOOST_FOREACH (boost::shared_ptr<layers::Layer<TDevice> > &layer, m_layers){
	layer->setCurrTrainingEpoch(trainingEpoch);
    }
    m_trainingEpoch = trainingEpoch;
}

template <typename TDevice>
void NeuralNetwork<TDevice>::notifyCurrentFrac(const int fracNum)
{
    BOOST_FOREACH (boost::shared_ptr<layers::Layer<TDevice> > &layer, m_layers){
	layer->setCurrTrainingFrac(fracNum);
    }
    m_trainingFrac = fracNum;
}

template <typename TDevice>
void NeuralNetwork<TDevice>::updateNNState(const int trainingEpoch, const int fracNum,
					   const bool backpropagation)
{

    if (flagNetworkForGAN()){
	// Rule:
	//  temp == 1: train discriminator using natural data
	//  temp == 2: train discriminator using generated data
	//  temp == 0:
	//         if featMatch is used, train the generator using feature matching
	//         else the normal way
	
	int temp = ((fracNum + 1) % 3);
	if (temp == 1)
	    m_trainingState = NN_STATE_GAN_DIS_NATDATA;
	else if (temp == 2)
	    m_trainingState = NN_STATE_GAN_DIS_GENDATA;
	else if (temp == 0)
	    m_trainingState = (m_featMatchLayer > 0)?(NN_STATE_GAN_GEN_FEATMAT):(NN_STATE_GAN_GEN);
	else
	    throw std::runtime_error("Undefined nnstate");
    }else{
	// Normal network
	if (backpropagation)
	    m_trainingState = NN_STATE_GAN_NOGAN_TRAIN;
	else
	    m_trainingState = NN_STATE_GAN_NOGAN;
    }
}

template <typename TDevice>
void NeuralNetwork<TDevice>::updateNNStateForGeneration()
{
    m_trainingState = NN_STATE_GENERATION_STAGE;
}


template <typename TDevice>
void NeuralNetwork<TDevice>::reInitWeight()
{
    printf("Reinitialize the weight\n");
    BOOST_FOREACH (boost::shared_ptr<layers::Layer<TDevice> > &layer, m_layers){
	layer->reInitWeight();
    }
}

template <typename TDevice>
void NeuralNetwork<TDevice>::initOutputForMDN(const helpers::JsonDocument &jsonDoc,
					      const data_sets::DataSetMV &datamv)
{
    if (jsonDoc->HasMember("weights")) {
	printf("\nEscapre MDN initialization\n");
	return;
    }
    BOOST_FOREACH (boost::shared_ptr<layers::Layer<TDevice> > &layer,
		   m_layers){
	layers::MDNLayer<TDevice>* mdnLayer = 
	    dynamic_cast<layers::MDNLayer<TDevice>*>(layer.get());
	if (mdnLayer){
	    mdnLayer->initPreOutput(datamv.outputM(), datamv.outputV());
	    printf("\nRe-initialize the layer before MDN \t");
	    if (datamv.outputM().size()<1)
		printf("using global zero mean and uni variance");
	    else
		printf("using data mean and variance");
	}else{

	    layers::NormFlowLayer<TDevice>* nmLayer = 
		dynamic_cast<layers::NormFlowLayer<TDevice>*>(layer.get());
	    if (nmLayer){
		nmLayer->initPreOutput();
	    }
	}
    }
}

template <typename TDevice>
void NeuralNetwork<TDevice>::readMVForOutput(const data_sets::DataSetMV &datamv)
{
    BOOST_FOREACH (boost::shared_ptr<layers::Layer<TDevice> > &layer,
		   m_layers){
	layers::PostOutputLayer<TDevice>* outputLayer = 
	    dynamic_cast<layers::PostOutputLayer<TDevice>*>(layer.get());
	if (outputLayer){
	    outputLayer->readMV(datamv.outputM(), datamv.outputV());
	    printf("Read mean and variance into output layer \t");
	}
    }
}


/* importWeights
 * import weights from pre-trained model
 */
template <typename TDevice>
void NeuralNetwork<TDevice>::importWeights(const helpers::JsonDocument &jsonDoc, 
					   const std::string &ctrStr)
{
    try{
	// Step1. read in the control vector, a sequence of 1 0
	Cpu::int_vector tempctrStr;
	tempctrStr.resize(m_totalNumLayers, 1);

	if (ctrStr.size() > 0){
	    // if --trainedModelCtr is given
	    if (ctrStr.size()!= m_totalNumLayers){
		printf("\n\tLength of trainedModelCtr is unequal to the number of layers in ");
		printf("the network to be trained\n");
		throw std::runtime_error("Please check trainedModelCtr");
	    }
	    for (int i=0; i<ctrStr.size(); i++)
		tempctrStr[i] = ctrStr[i]-'0';
	    
	}else{
	    // --trainedModelCtr is not given
	    // just assume every layer will be load from jsonDic
	    
	}
		
	// Step2. read in the weights from the pre-trained network
	helpers::JsonValue weightsSection;
        if (jsonDoc->HasMember("weights")) {
            if (!(*jsonDoc)["weights"].IsObject())
                throw std::runtime_error("Section 'weights' is not an object");
            weightsSection = helpers::JsonValue(&(*jsonDoc)["weights"]);
        }else{
	    throw std::runtime_error("No weight section found");
	}

	// Step3. for each layer in the network to be traiend
	//        load the weights from the pre-trained network
	//        if 
	//         1. control option is 1
	//         2. the name matches (checked handles in Layer->reReadWeight())
	
	int cnt=0;
	BOOST_FOREACH (boost::shared_ptr<layers::Layer<TDevice> > &layer, m_layers)
	{
	    layers::TrainableLayer<TDevice>* Layer = 
		dynamic_cast<layers::TrainableLayer<TDevice>*>(layer.get());
	    
	    // Read in the parameter for a hidden layer
	    if (Layer && tempctrStr[cnt] > 0){
		printf("\n\t(%d) ", cnt);
		Layer->reReadWeight(weightsSection, Layer->size(), tempctrStr[cnt]);
				
	    // Read in the parameter for MDN layer with trainable link
	    }else if(tempctrStr[cnt] > 0){
		layers::MDNLayer<TDevice>* mdnlayer = 
		    dynamic_cast<layers::MDNLayer<TDevice>*>(layer.get());
		if (mdnlayer && mdnlayer->flagTrainable()){
		    printf("\n\t(%d) ", cnt);
		    mdnlayer->reReadWeight(weightsSection, tempctrStr[cnt]);
		}
		
	    // This layer is skipped
	    }else if(Layer){
		printf("\n\t(%d) not read weight for layer %s", cnt, Layer->name().c_str());
	    }else{
		// other cases
	    }
	    cnt++;
	}
	printf("\tdone\n\n");
	
    }catch (const std::exception &e){
	printf("\n\t%s\n", e.what());
	printf("\n\tError in reading weight from a pre-trained network:");
	printf("\n\tPlease check the configuration of trainedModel and trainedModelCtr");
	printf("\n\t1. trainedModel points to the correct pre-trained network.jsn ?");
	printf("\n\t2. trainedModelCtr, a sequence of 0/1, has the same length as");
	printf("the number of layers in network.jsn to be trained?");
	printf("\n\t3. the layer to be initialized has the same name as the pre-trained layer?\n");
	throw std::runtime_error(std::string("Fail to read network weight")+e.what());
    }
}


template <typename TDevice>
Cpu::real_vector NeuralNetwork<TDevice>::getMdnConfigVec()
{
    Cpu::real_vector temp;
    BOOST_FOREACH (boost::shared_ptr<layers::Layer<TDevice> > &layer, m_layers){
	layers::MDNLayer<TDevice>* mdnLayer = 
	    dynamic_cast<layers::MDNLayer<TDevice>*>(layer.get());
	if (mdnLayer)
	    temp = mdnLayer->getMdnConfigVec();
    }    
    return temp;
}

// PrintWeightMatrix
// print the weight of a network to a binary data
// use ReadCURRENNTWeight(filename,format,swap) matlab function to read the data
template <typename TDevice>
void NeuralNetwork<TDevice>::printWeightMatrix(const std::string weightPath, const int opt)
{
    std::fstream ifs(weightPath.c_str(),
		      std::ifstream::binary | std::ifstream::out);
    if (!ifs.good()){
	throw std::runtime_error(std::string("Fail to open output weight path: "+weightPath));
    }

    // format of the output binary weight
    std::vector<int> weightSize;
    weightSize.clear();
    BOOST_FOREACH (boost::shared_ptr<layers::Layer<TDevice> > &layer, m_layers){
	layers::TrainableLayer<TDevice>* Layer = 
	    dynamic_cast<layers::TrainableLayer<TDevice>*>(layer.get());
	
	if (Layer){
	    weightSize.push_back(Layer->weights().size());
	    weightSize.push_back(Layer->size());
	    weightSize.push_back(Layer->precedingLayer().size());
	    weightSize.push_back(Layer->inputWeightsPerBlock());
	    weightSize.push_back(Layer->internalWeightsPerBlock());
	    if (opt==1){
		if (Layer->type()=="feedforward_tanh")
		    weightSize.push_back(0);
		else if (Layer->type()=="feedforward_logistic")
		    weightSize.push_back(1);
		else if (Layer->type()=="feedforward_identity")
		    weightSize.push_back(2);
		else if (Layer->type()=="feedforward_relu")
		    weightSize.push_back(3);		
		else if (Layer->type()=="lstm")
		    weightSize.push_back(4);		
		else if (Layer->type()=="blstm")
		    weightSize.push_back(5);
		else
		    printf("other weight type not implemented\n");
	    }
	}else{
	    layers::MDNLayer<TDevice>* mdnlayer = 
		dynamic_cast<layers::MDNLayer<TDevice>*>(layer.get());
	    if (mdnlayer && mdnlayer -> flagTrainable()){
		weightSize.push_back(mdnlayer->weights().size());
		weightSize.push_back(mdnlayer->weights().size());
		weightSize.push_back(0);  // previous size = 0
		weightSize.push_back(1);  // internal block = 1
		weightSize.push_back(0);  // internal weight = 0
	    }
	}
    }

    printf("Writing network to binary format: \n");
    // macro information
    // Number of layers
    // weight size, layer size, preceding layer size, inputWeightsPerBlock, internalWeightsPerBlock
    real_t tmpPtr;
    tmpPtr = (real_t)weightSize.size()/((opt==1)?6:5);
    ifs.write((char *)&tmpPtr, sizeof(real_t));
    for (int i = 0 ; i<weightSize.size(); i++){
	tmpPtr = (real_t)weightSize[i];
	ifs.write((char *)&tmpPtr, sizeof(real_t));
    }

    // weights
    int cnt = 0;
    real_t *tmpPtr2;
    Cpu::real_vector weightVec;
    BOOST_FOREACH (boost::shared_ptr<layers::Layer<TDevice> > &layer, m_layers){
	layers::TrainableLayer<TDevice>* Layer = 
	    dynamic_cast<layers::TrainableLayer<TDevice>*>(layer.get());
	if (Layer){
	    weightVec = Layer->weights();
	    tmpPtr2 = weightVec.data();
	    if (weightVec.size()>0 && tmpPtr2)
		ifs.write((char *)tmpPtr2, sizeof(real_t)*Layer->weights().size());	
	    printf("Layer (%2d) %s with %lu weights\n", cnt, Layer->type().c_str(), weightVec.size());
	}else{
	    layers::MDNLayer<TDevice>* mdnlayer = 
		dynamic_cast<layers::MDNLayer<TDevice>*>(layer.get());
	    if (mdnlayer && mdnlayer -> flagTrainable()){
		weightVec = mdnlayer->weights();
		tmpPtr2 = weightVec.data();
		if (weightVec.size()>0 && tmpPtr2){
		    ifs.write((char *)tmpPtr2, sizeof(real_t)*mdnlayer->weights().size());
		}else{
		    throw std::runtime_error("Fail to output weight. Void pointer");
		}
		printf("Layer (%2d) MDN with %lu weights\n", cnt, weightVec.size());
	    }
	}
	cnt++;
    }
    ifs.close();
    printf("Writing done\n");
}

template <typename TDevice>
int NeuralNetwork<TDevice>::layerSize(const int layerID)
{
    if (layerID < 0)
	return m_layers.back()->size();
    else if (layerID > (m_totalNumLayers-1))
	throw std::runtime_error(std::string("Invalid layer ID. In NN.layerSize"));
    else
	return m_layers[layerID]->size();
}

template <typename TDevice>
bool NeuralNetwork<TDevice>::isMDNLayer(const int layerID)
{
    if (layerID < 0)
	return ((dynamic_cast<layers::MDNLayer<TDevice>*>(m_layers.back().get())) != NULL);
    else if (layerID > (m_totalNumLayers-1))
	throw std::runtime_error(std::string("Invalid layer ID. In NN.isMDNLayer"));
    else
	return ((dynamic_cast<layers::MDNLayer<TDevice>*>(m_layers[layerID].get())) != NULL);
}

template <typename TDevice>
bool NeuralNetwork<TDevice>::externalOutputMV(Cpu::real_vector& mean, Cpu::real_vector& var)
{
    layers::PostOutputLayer<TDevice> *postoutputLayer = 
	dynamic_cast<layers::PostOutputLayer<TDevice>*>(m_layers.back().get());

    int mvLength =  postoutputLayer->externalOutputMV().size()/2;
    if (mvLength > 0){
	mean.resize(mvLength);
	var.resize(mvLength);
	thrust::copy(postoutputLayer->externalOutputMV().begin(),
		     postoutputLayer->externalOutputMV().begin() + mvLength,
		     mean.begin());
	thrust::copy(postoutputLayer->externalOutputMV().begin() + mvLength,
		     postoutputLayer->externalOutputMV().end(),
		     var.begin());
	return true;
    }else{
	return false;
    }
}

template <typename TDevice>
int NeuralNetwork<TDevice>::outputPatternSize(const real_t mdnoutput)
{
    // Get configuration
    const Configuration &config = Configuration::instance();
    
    const int  layerID        = config.outputFromWhichLayer();
    const bool flagGateOutput = config.outputFromGateLayer();
    const int  flagVQIndex    = config.vaeGetVQIndex();
    
    std::vector<std::vector<std::vector<real_t> > > outputs;
    layers::SkipLayer<TDevice> *olg;
    layers::MDNLayer<TDevice>  *olm;
    layers::vqLayer<TDevice>   *olq;
    
    unsigned char genMethod;
    int tempLayerID;
    
    enum genMethod {ERROR = 0,VQINDEX,GATEOUTPUT, MDNSAMPLING, MDNPARAMETER, MDNEMGEN, NORMAL};


    // Determine the output layer
    if (layerID < 0){
	// If layerID is not specified, generate from the last output/postoutput layer
	olg = NULL;
	olq = NULL;
	
	olm = outMDNLayer(-1);
	if (olm == NULL)
	    tempLayerID = this->m_totalNumLayers-2; // postouput MDN
	else
	    tempLayerID = this->m_totalNumLayers-1; // output
	
    }else{
	// If layerID is specified, generate from that layer
	if (flagGateOutput){
	    // generate from Highway gate
	    olg = outGateLayer(layerID);
	    olm = NULL;
	    olq = NULL;
	    if (olg == NULL) throw std::runtime_error("Gate output tap ID invalid\n");
	}else if (flagVQIndex){
	    // generate from vqLayer
	    olg = NULL;
	    olm = NULL;
	    olq = outvqLayer(layerID);
	    if (olq == NULL) throw std::runtime_error("vqLayer ID invalid\n");	    
	}else{
	    // generate from specified layerID
	    olg = NULL;
	    olm = outMDNLayer(layerID);
	    olq = NULL;
	}
	tempLayerID = layerID;
    }

    // Determine the generation method
    if (olg == NULL && olq == NULL){
	if (olm == NULL)
	    // output from the layer output
	    genMethod = NORMAL;
	else
	    // output from the MDN layer
	    genMethod = (mdnoutput<0.0) ? ((mdnoutput < -1.5) ? MDNEMGEN:MDNPARAMETER):MDNSAMPLING;
    }else if (olq == NULL){
	// output from the highway gate
	genMethod = GATEOUTPUT;
    }else{
	// output from vqLayer by index
	genMethod = VQINDEX;
    }

    // retrieve the output
    layers::Layer<TDevice> &ol      = outputLayer(tempLayerID);
    switch (genMethod){
	    case MDNEMGEN:
	    case MDNSAMPLING:
	    case NORMAL:
		{
		    return ol.size();
		    break;
		}
	    case MDNPARAMETER:
		{
		    return olm->mdnParaDim();
		    break;
		}
	    case GATEOUTPUT:
		{
		    return ol.size();
		    break;
		}
	    case VQINDEX:
		{
		    return olq->codeBookNum();
		    break;
		}
	    default:
		return -1;
		break;   
    }
    
}

// explicit template instantiations
template class NeuralNetwork<Cpu>;
template class NeuralNetwork<Gpu>;
