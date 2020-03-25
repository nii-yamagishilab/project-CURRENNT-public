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

#ifdef _MSC_VER
#   pragma warning (disable: 4244)
#endif

#include "Layer.hpp"
#include "../helpers/misFuncs.hpp"
#include "../helpers/JsonClasses.hpp"

#include <sstream>
#include <stdexcept>


namespace layers {

    int get_maxSeqLength(const helpers::JsonValue &layerChild,
			 const int expected_maxLen)
    {
	int tmp_reso = (layerChild->HasMember("resolution") ? 
			(*layerChild)["resolution"].GetInt() : 1);
	int tmp_maxLen = misFuncs::getResoLength(expected_maxLen, tmp_reso, 1);

	if (layerChild->HasMember("singleTimeStep") &&
	    (*layerChild)["singleTimeStep"].GetInt()){
	    tmp_maxLen = 1;
	}
	return tmp_maxLen;
    }
    
    template <typename TDevice>
    typename Layer<TDevice>::real_vector& Layer<TDevice>::_outputs()
    {
        return m_outputs;
    }
    
    /* Add 16-02-22 Wang: for WE updating */
    template <typename TDevice>
    bool& Layer<TDevice>::inputWeUpdate()
    {
	return m_InputWeUpdate;
    }
    
    template <typename TDevice>
    bool Layer<TDevice>::_setInputWeUpdate(const bool& flag)
    {
	m_InputWeUpdate = flag;
	return true;
    }
    

    template <typename TDevice>
    Layer<TDevice>::Layer(const helpers::JsonValue &layerChild,
			  int  parallelSequences,
			  int  maxSeqLength,
			  bool flagTrainingMode,
			  int  layerID,
			  Layer<TDevice> *precedingLayer,
			  bool createOutputs)
        : m_name             (layerChild->HasMember("name") ? 
			      (*layerChild)["name"].GetString()  : "")
        , m_size             (layerChild->HasMember("size") ? 
			      (*layerChild)["size"].GetInt()     : 0)
	, m_timeResolution   (layerChild->HasMember("resolution") ? 
			      (*layerChild)["resolution"].GetInt() : 1)
	, m_layerID          (layerID)
        , m_parallelSequences(parallelSequences)
        , m_maxSeqLength     (get_maxSeqLength(layerChild, maxSeqLength))
        , m_curMaxSeqLength  (0)
        , m_curMinSeqLength  (0)
        , m_curNumSeqs       (0)
	, m_InputWeUpdate        (false)
	, m_flagTrainingMode     (true)
	, m_flagSaveOutputMemory (false)
	, m_precedingLayer   (precedingLayer)
	, m_followingLayer   (NULL)
	, m_layerMode        (-99)
	, m_singleTimeStep   (layerChild->HasMember("singleTimeStep") ? 
			      (*layerChild)["singleTimeStep"].GetInt() : false)
    {
        // check if the name and size values exist
        if (!layerChild->HasMember("name"))
            throw std::runtime_error("Missing value 'name' in network.jsn");
	
        if (m_name.empty())
            throw std::runtime_error("Empty layer name in network.jsn");
	
        if (!layerChild->HasMember("size"))
            throw std::runtime_error("Missing 'size' in layer");

	if (m_timeResolution > 1){
	    printf("\n\tLayer resolution %d\n", m_timeResolution);
	}else if (m_timeResolution < 1){
	    throw std::runtime_error("resolution cannot be less than 1");
	}

	if (m_singleTimeStep){
	    printf("\n\tLayer with single time step\n");
	    if (m_timeResolution > 1)
		printf("\n\tTime resolution will be ignored\n");
	}
	
        // allocate memory for output
        if (createOutputs)
            m_outputs = Cpu::real_vector(m_parallelSequences * m_maxSeqLength * m_size);
	
	// allocate memory for time mark
        m_patTypes = Cpu::pattype_vector(m_parallelSequences * m_maxSeqLength);
	
        // allocate memory for gradients buffer
	if (flagTrainingMode)
	    m_outputErrors  = Cpu::real_vector(this->_outputs().size(), (real_t)0);    
	else
	    m_outputErrors.clear();
	m_outputErrorsCopy.clear();
	
	// initialize the training epoch counter
	m_currTrainingEpoch = -1;

	// set the flag
	m_flagTrainingMode  = (flagTrainingMode ? true : false);

	m_layerFlag = (layerChild->HasMember("layerFlag") ? 
		       (*layerChild)["layerFlag"].GetString() : "");
	
    }

    template <typename TDevice>
    Layer<TDevice>::~Layer()
    {
    }
    
    template <typename TDevice>
    const std::string& Layer<TDevice>::name() const
    {
        return m_name;
    }

    template <typename TDevice>
    int Layer<TDevice>::size() const
    {
        return m_size;
    }

    template <typename TDevice>
    const std::string& Layer<TDevice>::getLayerFlag()
    {
	return m_layerFlag;
    }

    template <typename TDevice>
    void Layer<TDevice>::setLayerMode(const int mode)
    {
	m_layerMode = mode;
    }

    template <typename TDevice>
    int Layer<TDevice>::getLayerMode()
    {
	return m_layerMode;
    }
    
    template <typename TDevice>
    int Layer<TDevice>::parallelSequences() const
    {
        return m_parallelSequences;
    }

    template <typename TDevice>
    int Layer<TDevice>::maxSeqLength() const
    {
        return m_maxSeqLength;
    }

    template <typename TDevice>
    int Layer<TDevice>::curMaxSeqLength() const
    {
        return m_curMaxSeqLength;
    }

    template <typename TDevice>
    int Layer<TDevice>::curMinSeqLength() const
    {
        return m_curMinSeqLength;
    }

    template <typename TDevice>
    int Layer<TDevice>::curNumSeqs() const
    {
        return m_curNumSeqs;
    }

    template <typename TDevice>
    const int& Layer<TDevice>::getResolution()
    {
	return m_timeResolution;
    }

    template <typename TDevice>
    const int& Layer<TDevice>::getLayerID()
    {
	return m_layerID;
    }

    template <typename TDevice>
    const typename Layer<TDevice>::pattype_vector& Layer<TDevice>::patTypes() const
    {
        return m_patTypes;
    }

    template <typename TDevice>
    typename Layer<TDevice>::real_vector& Layer<TDevice>::outputs()
    {
        return m_outputs;
    }

    template <typename TDevice>
    typename Layer<TDevice>::real_vector& Layer<TDevice>::outputErrors()
    {
        return m_outputErrors;
    }

    template <typename TDevice>
    Cpu::real_vector& Layer<TDevice>::outputErrorsCpu()
    {
	if (m_outputErrorsCopy.size() != m_outputErrors.size())
	    m_outputErrorsCopy = m_outputErrors;
	thrust::copy(m_outputErrors.begin(), m_outputErrors.end(),
		     m_outputErrorsCopy.begin());
        return m_outputErrorsCopy;
    }
    
    template <typename TDevice>
    void Layer<TDevice>::loadSequences(const data_sets::DataSetFraction &fraction,
				       const int nnState)
    {
	if (m_singleTimeStep){
	    // signle time 
	    m_curMaxSeqLength = m_parallelSequences;
	    m_curMinSeqLength = m_parallelSequences;
	    m_curNumSeqs = fraction.numSequences();
	    thrust::copy(fraction.patTypes().begin(),
			 fraction.patTypes().begin() + m_parallelSequences,
			 m_patTypes.begin());
	    
	}else{
	    // normal cases
	    m_curMaxSeqLength = misFuncs::getResoLength(fraction.maxSeqLength(),
							m_timeResolution, 1);
	    
	    m_curMinSeqLength = misFuncs::getResoLength(fraction.minSeqLength(),
							m_timeResolution, 1);
	
	    m_curNumSeqs      = fraction.numSequences();
	    
	    if (m_timeResolution == 1){
		m_patTypes    = fraction.patTypes();
	    
	    }else{

		thrust::fill(m_patTypes.begin(), m_patTypes.end(), PATTYPE_NONE);
		int buffPos   = fraction.patTypesLowTimesResPos(m_timeResolution);
		int buffLen   = fraction.patTypesLowTimesResLen(m_timeResolution);
		if (buffPos < 0 || buffLen < 0){
		    printf("\n %s resolution not in --resolutions \n", this->name().c_str());
		    printf("\n please check --resolutions\n");
		    throw std::runtime_error("Resolution error");
		}
		if (buffPos > fraction.patTypesLowTimeRes().size() ||
		    (buffLen + buffPos) > fraction.patTypesLowTimeRes().size()){
		    printf("\n%s resolution not in --resolutions", this->name().c_str());
		    printf("\n please check --resolutions\n");		
		    throw std::runtime_error("Resolution error");
		}
		//m_patTypes.resize(buffLen, PATTYPE_NONE);
		//thrust::fill(m_patTypes.begin(), m_patTypes.end(), PATTYPE_NONE);
		thrust::copy(fraction.patTypesLowTimeRes().begin() + buffPos,
			     fraction.patTypesLowTimeRes().begin() + buffPos + buffLen,
			     m_patTypes.begin());
	    	
	    }
	}
	

    }
    
    template <typename TDevice>
    void Layer<TDevice>::exportLayer(const helpers::JsonValue &layersArray, 
				     const helpers::JsonAllocator &allocator) const
    {
        if (!layersArray->IsArray())
            throw std::runtime_error("The JSON value is not an array");

        // create and fill the layer object
        rapidjson::Value layerObject(rapidjson::kObjectType);
        layerObject.AddMember("name", name().c_str(), allocator);
        layerObject.AddMember("type", type().c_str(), allocator);
        layerObject.AddMember("size", size(),         allocator);

	if (m_timeResolution > 1)
	    layerObject.AddMember("resolution", m_timeResolution, allocator);
	if (m_layerFlag.size() > 0)
	    layerObject.AddMember("layerFlag", m_layerFlag.c_str(), allocator);
	if (m_singleTimeStep)
	    layerObject.AddMember("singleTimeStep", 1, allocator);
	
        // add the layer object to the layers array
        layersArray->PushBack(layerObject, allocator);
    }
    
    template <typename TDevice>
    void Layer<TDevice>::setCurrTrainingEpoch(const int curTrainingEpoch)
    {
	m_currTrainingEpoch = curTrainingEpoch;
    }
    
    template <typename TDevice>
    int& Layer<TDevice>::getCurrTrainingEpoch()
    {
	return m_currTrainingEpoch;
    }

    template <typename TDevice>
    void Layer<TDevice>::setCurrTrainingFrac(const int curTrainingFrac)
    {
	m_currTrainingFrac = curTrainingFrac;
    }
    
    template <typename TDevice>
    int& Layer<TDevice>::getCurrTrainingFrac()
    {
	return m_currTrainingFrac;
    }

    template <typename TDevice>
    void Layer<TDevice>::linkTargetLayer(Layer<TDevice> &targetLayer)
    {
	// this function will be override by layers requiring targetLayer
    }
    
    template <typename TDevice>
    int Layer<TDevice>::returnTargetLayerID()
    {
	return -1;
    }

    
    template <typename TDevice>
    void Layer<TDevice>::linkFollowingLayer(Layer<TDevice> &targetLayer)
    {
	// This function is only called by NeuralNetwork
	m_followingLayer = &targetLayer;
    }
    

    template <typename TDevice>
    const std::string& Layer<TDevice>::layerAddInfor(const int opt) const
    {
	// used together with feedbackOutputs
        static std::string s;
        if (s == "" && opt==1){
	    std::ostringstream Convert;
	    Convert << this->size() << "_";  
	    s = Convert.str();
	}
	//else{
	//    s = "";
	//}
	
        return s;
    }
    
    template <typename TDevice>
    void Layer<TDevice>::prepareStepGeneration(const int timeStep)
    {
	// do nothing
    }

    template <typename TDevice>
    typename Layer<TDevice>::real_vector& Layer<TDevice>::feedbackOutputs(
						const bool flagTrain)
    {
        return m_outputs;
    }

    template <typename TDevice>
    void Layer<TDevice>::cleanGradidents()
    {
	//thrust::fill(m_outputErrors.begin(), m_outputErrors.end(), 0.0);
    }

    template <typename TDevice>
    int Layer<TDevice>::hiddenStateSize()
    {
	return 0;
    }

    template <typename TDevice>
    void Layer<TDevice>::retrieveHiddenState(const int timeStep, real_vector& readBuffer)
    {	
    }
    
    template <typename TDevice>
    void Layer<TDevice>::setHiddenState(const int timeStep, real_vector& writeBuffer)
    {	
    }
    
    template <typename TDevice>
    bool Layer<TDevice>::flagTrainingMode() const
    {
	return m_flagTrainingMode;
    }

    template <typename TDevice>
    void Layer<TDevice>::clearOutputBuffer()
    {
	m_outputs.clear();
	m_outputs.shrink_to_fit();
	real_vector tmp(0);
	tmp.swap(m_outputs);
    }
    
    template <typename TDevice>
    void Layer<TDevice>::resizeOutputBuffer(const int bufferSize)
    {
	if (m_outputs.size())
	    this->clearOutputBuffer();
	m_outputs = Cpu::real_vector(bufferSize);
	// Can't use m_outputs.resize() because:
	// Not really. The extent of Thrust's CUDA support for pure C++ code is
	// the bare minimum to allow device_vector to be constructed and
	// destroyed for POD types. That's why thrust::fill has a pure C++ 
	// implementation even for the CUDA backend -- so it can be called by 
	// device_vector's constructor.
	// https://groups.google.com/forum/#!topic/thrust-users/abVI3htMrkw
    }

    template <typename TDevice>
    void Layer<TDevice>::reduceOutputBuffer()
    {
	// default: do nothing
    }

    template <typename TDevice>
    real_t Layer<TDevice>::intermediateError()
    {
	// default: return 0
	return 0.0;
    }
    
    template <typename TDevice>
    int Layer<TDevice>::outputBufPtrBias(const int timeStepTimesParallel,
					 const int nnState)
    {
	// don't shift
	return 0;
    }
    
    template <typename TDevice>
    void Layer<TDevice>::setSaveMemoryFlag(const bool newFlag)
    {
	m_flagSaveOutputMemory = newFlag;
    }

    template <typename TDevice>
    bool Layer<TDevice>::getSaveMemoryFlag() const
    {
	return m_flagSaveOutputMemory;
    }

    template <typename TDevice>
    Layer<TDevice>& Layer<TDevice>::precedingLayer()
    {
	if (m_precedingLayer == NULL)
	    throw std::runtime_error("PrecedingLayer Ptr is NULL");
	
	return *m_precedingLayer;
	
    }

    template <typename TDevice>
    const Layer<TDevice>& Layer<TDevice>::precedingLayer() const
    {
	if (m_precedingLayer == NULL)
	    throw std::runtime_error("PrecedingLayer Ptr is NULL");
	
	return *m_precedingLayer;
	
    }

    template <typename TDevice>
    Layer<TDevice>& Layer<TDevice>::followingLayer()
    {
	if (m_followingLayer == NULL)
	    throw std::runtime_error("FollowingLayer Ptr is NULL");
	return *m_followingLayer;	
    }

    template <typename TDevice>
    const Layer<TDevice>& Layer<TDevice>::followingLayer() const
    {
	if (m_followingLayer == NULL)
	    throw std::runtime_error("FollowingLayer Ptr is NULL");
	return *m_followingLayer;
    }


    template <typename TDevice>
    std::vector<int> Layer<TDevice>::dependLayerIDs()
    {
	// by default, only the previous layer
	std::vector<int> depend_layerIDs;
	if (m_precedingLayer != NULL)
	    depend_layerIDs.push_back(m_precedingLayer->getLayerID());
	else
	    depend_layerIDs.clear();
	return depend_layerIDs;
    }

    template <typename TDevice>
    bool Layer<TDevice>::getSingleTimeStepFlag() const
    {
	if (m_singleTimeStep)
	    return true;
	else
	    return false;
    }
    
    template <typename TDevice>
    void Layer<TDevice>::clearAllBuffers()
    {
	this->clearOutputBuffer();
    }

    template <typename TDevice>
    void Layer<TDevice>::resizeAllBuffers(const int timeLength)
    {
	this->resizeOutputBuffer(timeLength * this->parallelSequences() * this->size());
    }

    template <typename TDevice>
    void Layer<TDevice>::copyOutputs(real_vector& dataBuffer)
    {
	if (dataBuffer.size() == this->outputs().size())
	    thrust::copy(dataBuffer.begin(),
			 dataBuffer.end(),
			 this->outputs().begin());
    }

    template <typename TDevice>
    void Layer<TDevice>::logAllBuffers(helpers::vecPoolManager<TDevice> &vecPoolMng,
				       bool flag_add)
    {
	vecPoolMng.addOrRemoveNewVec(this->size(), flag_add);
    }
    
    template <typename TDevice>    
    void Layer<TDevice>::swapAllBuffers(helpers::vecPoolManager<TDevice> &vecPoolMng,
					bool flag_get)
    {
	vecPoolMng.getSwapVector(m_outputs,
				 this->getLayerID(), this->size(), flag_get);
    }


    
    // explicit template instantiations
    template class Layer<Cpu>;
    template class Layer<Gpu>;

} // namespace layers
