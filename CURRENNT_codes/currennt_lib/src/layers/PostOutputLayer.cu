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

#include "PostOutputLayer.hpp"
#include "../helpers/getRawPointer.cuh"
#include "../helpers/misFuncs.hpp"
#include "../helpers/dataProcess.hpp"
#include "../MacroDefine.hpp"
#include "../Configuration.hpp"
#include "../helpers/JsonClasses.hpp"

#include <boost/lexical_cast.hpp>


#include <stdexcept>

#include <fstream>

namespace internal{
namespace {


    // Drop output the feedback data
    struct feedBackDropOut
    {
	real_t *dropProb;
	real_t *buffer;
	int bufDim;
	int parall;
	
	// from 1 to timesteps
        __host__ __device__ void operator() (const thrust::tuple<real_t&, const int&> &values) const
        {
	    const int outputIdx = values.get<1>() / bufDim;
	    const int dimIdx    = values.get<1>() % bufDim;
	    if ((dropProb && dropProb[(outputIdx/parall)] < 0.5) ||
		dropProb == NULL)
		buffer[outputIdx * bufDim + dimIdx] = 0.0;
        }
    };

    struct loadExternalData
    {
	int  featureDim;
	int  paralNum;
	int  maxFeatureLength;
	int  resolution;
	int  indexLoadM;
	
	const real_t *sourceData;
	const real_t *frameIndex;
	const real_t *contextMV;
	const char   *patTypes;
	
	__host__ __device__ void operator() (const thrust::tuple<real_t&, int> &t) const
	{
	    int dimIdx  = t.get<1>() % featureDim;
	    int timeIdx = t.get<1>() / featureDim;
	    int reTimeIdx = timeIdx  / paralNum;
	    int paralIdx  = timeIdx  % paralNum;

	    if (patTypes[timeIdx] == PATTYPE_NONE)
		t.get<0>() = 0.0;
	    else{
		int frameIdx = frameIndex[reTimeIdx * resolution * paralNum + paralIdx];
		if (indexLoadM == NN_INDEXLOADMETHOD_1)
		    frameIdx = frameIdx / resolution;
		
		int featIdx  = frameIdx * paralNum + paralIdx;
		if (frameIdx >= maxFeatureLength){
		    t.get<0>() = 0.0;
		}else if(contextMV){
		    t.get<0>() = ((sourceData[featIdx * featureDim + dimIdx] - contextMV[dimIdx])/
				  ((contextMV[dimIdx + featureDim]<1e-5f) ?
				   (1.0): (contextMV[dimIdx + featureDim])));
		}else{
		    t.get<0>() = sourceData[featIdx * featureDim + dimIdx];
		}
		
	    }
	}
    };

}
}

namespace layers {

    template <typename TDevice>
    typename PostOutputLayer<TDevice>::real_vector& PostOutputLayer<TDevice>::_targets()
    {
        return this->outputs();
    }

    template <typename TDevice>
    typename PostOutputLayer<TDevice>::real_vector& PostOutputLayer<TDevice>::_actualOutputs()
    {
        return this->precedingLayer().outputs();
    }

    template <typename TDevice>
    typename PostOutputLayer<TDevice>::real_vector& PostOutputLayer<TDevice>::_outputErrors()
    {
        return this->precedingLayer().outputErrors();
    }
    
    /* Add 16-04-01 return the vector for mseWeight  */
    template <typename TDevice>
    typename PostOutputLayer<TDevice>::real_vector& PostOutputLayer<TDevice>::_mseWeight()
    {
	return m_outputMseWeights;
    }
    template <typename TDevice>
    typename PostOutputLayer<TDevice>::cpu_real_vector& PostOutputLayer<TDevice>::_mseWeightCPU()
    {
	return m_outputMseWeightsCPU;
    }
    
    template <typename TDevice>
    typename PostOutputLayer<TDevice>::real_vector& PostOutputLayer<TDevice>::_mvVector()
    {
	return m_targetDataMV;
    }

    template <typename TDevice>
    typename PostOutputLayer<TDevice>::real_vector& PostOutputLayer<TDevice>::_dataBuffer()
    {
	return m_dataBuffer;
    }

    template <typename TDevice>
    bool PostOutputLayer<TDevice>::_flagSeqWeights()
    {
	return m_flagSeqErrorWeights > 0;
    }

    template <typename TDevice>
    typename PostOutputLayer<TDevice>::real_vector& PostOutputLayer<TDevice>::_seqWeights()
    {
	return m_seqErrorWeights;
    }
    
    template <typename TDevice>
    typename PostOutputLayer<TDevice>::cpu_real_vector& PostOutputLayer<TDevice>::_seqWeightsCPU()
    {
	return m_seqErrorWeightsCPU;
    }
    
    template <typename TDevice>
    PostOutputLayer<TDevice>::PostOutputLayer(
        const helpers::JsonValue &layerChild, 
        Layer<TDevice> &precedingLayer,
        int  requiredSize,
	int  maxSeqLength,
	int  layerID,
        bool createOutputs)
        : Layer<TDevice>  (layerChild, precedingLayer.parallelSequences(), 
			   maxSeqLength,
			   Configuration::instance().trainingMode(),
			   layerID,
			   &precedingLayer,
			   createOutputs)
	, m_precedingMiddleOutLayer (NULL)
	, m_postoutputFlag(NN_POSTOUTPUTLAYER_LAST)
	, m_useExternalOutput(layerChild->HasMember("useExternalOutput") ? 
			      (*layerChild)["useExternalOutput"].GetInt() : 0)
    {
	
	/* ---------- Check ----------- */
	// Modify 0506. For MDN, requireSize = -1, no need to check here
	// if (this->size() != requiredSize)
        if (requiredSize > 0 && this->size() != requiredSize){
	    printf("Layer size mis-match: %d vs %d",
		   this->size(), requiredSize);
	    throw std::runtime_error("Error in network.jsn");
	}

	
	/* ---------- Prepare buffer and variables ----------- */
	/* Add 0401 wang */
	// assign the vector to output weights for RMSE
	// m_outputMseWeights = TDevice::real_vector(this->size(), (real_t)1.0);
	m_flagMseWeight = false;

	// Add 170411
	// Prepare the feedback data buffer
	m_feedBackOutput = this->outputs();

	// Buffer for reading external data
	m_dataBuffer.clear();

	
	/* ---------- Load options ----------- */
	// m_indexLoadMethod (same as that in ExternalLoader.cu):
	//   default:  index [1, ... N], index n -> n-th data frame
	//             assume the number of data frame is N
	//   method 1: index [1, ... N], index n -> n/timeResolution -th data frame
	//             assume the number of data frame is N/timeResolution + N%timeResolution>0
	// 
	m_indexLoadMethod   = ((layerChild->HasMember("indexLoadMethod")) ?
			       ((*layerChild)["indexLoadMethod"].GetInt()) : 0);


	if (Configuration::instance().dataTargetPath().size()){
	    m_externalDataMVStr = Configuration::instance().dataTargetPath();
	}else{
	    m_externalDataMVStr = ((layerChild->HasMember("externalDataMV")) ?
				   ((*layerChild)["externalDataMV"].GetString()) : "");
	}

	m_flagSeqErrorWeights= Configuration::instance().seqWeightsFlag();

	
	
	/* ---------- Data Load ----------- */
	// Load external Data MV vector
	cpu_real_vector tmp;
	m_externalDataMV.clear();
	if (m_externalDataMVStr.size()){
	    if (misFuncs::ReadRealData(m_externalDataMVStr, tmp) != this->size() * 2)
		throw std::runtime_error("externalDataMV dimension unmatched with layer size");
	    m_externalDataMV = tmp;
	    printf("\n\tRead MV from %s", m_externalDataMVStr.c_str());
	}else{
	    printf("\n\tSkip reading MV");
	}
	
	// Prepare buffer for target data MV
	Cpu::real_vector tmpVec(this->size()*2, 0.0);
	thrust::fill(tmpVec.begin(), tmpVec.begin() + this->size(), 0.0);
	thrust::fill(tmpVec.begin() + this->size(), tmpVec.end(), 1.0);
	m_targetDataMV = tmpVec;


	/* ---------- Print information ----------- */
	if (m_flagSeqErrorWeights){
	    printf("sequence weigt is turned on. Auxillary data are load as sequence weights");
	    m_seqErrorWeights = Cpu::real_vector(this->parallelSequences() * maxSeqLength, 0.0);
	    m_seqErrorWeightsCPU = m_seqErrorWeights;
	}else{
	    m_seqErrorWeights.clear();
	    m_seqErrorWeightsCPU.clear();
	}
	
    }

    template <typename TDevice>
    PostOutputLayer<TDevice>::~PostOutputLayer()
    {
    }

    template <typename TDevice>
    void PostOutputLayer<TDevice>::loadSequences(const data_sets::DataSetFraction &fraction,
						 const int nnState)
    {

	/* ------------ Load target data --------------- */	    
	if (m_precedingMiddleOutLayer == NULL){
	    
	    // For normal postoutput layer:
	    // 1. load target data
	    // 2. if necessary, load the sequence weight
	    
	    Layer<TDevice>::loadSequences(fraction, nnState);
	    
	    if (m_useExternalOutput == 0){

		// The normal case, loading from data.nc
		
		if (fraction.outputPatternSize() != this->size()) {
		    printf("Patter size mis match %d (layer) vs %d (data)",
			   this->size(), fraction.outputPatternSize());
		    throw std::runtime_error("Error in network.jsn and data.nc");
		}
		
		if (!this->_outputs().empty()){
		    thrust::copy(fraction.outputs().begin(), fraction.outputs().end(), 
				 this->_outputs().begin());
		}
		
	    }else if (fraction.externalOutputSize() < 1){
		
		// The specical case, no need do to load target data during generation
		// do nothing
		
	    }else{
		
		// The specical case, loading data directly from files
		
		if (fraction.externalOutputSize() != this->size()){
		    printf("Patter size mis match %d (layer) vs %d (data)",
			   this->size(), fraction.externalOutputSize());
		    throw std::runtime_error("Error in network.jsn and external data");
		}
		
		if (!this->_outputs().empty()){
		    
		    m_dataBuffer.resize(fraction.outputs().size() +
					fraction.exOutputData().size(), 0.0);
		    thrust::copy(fraction.outputs().begin(),
				 fraction.outputs().end(),
				 m_dataBuffer.begin());
		    thrust::copy(fraction.exOutputData().begin(),
				 fraction.exOutputData().end(),
				 m_dataBuffer.begin() + fraction.outputs().size());
	
		    internal::loadExternalData fn1;
		    fn1.featureDim = fraction.externalOutputSize();
		    fn1.paralNum   = this->parallelSequences();
		    fn1.maxFeatureLength = fraction.maxExOutputLength();
		    fn1.sourceData = (helpers::getRawPointer(m_dataBuffer) +
				      fraction.outputs().size());
		    fn1.frameIndex = helpers::getRawPointer(m_dataBuffer);
		    fn1.patTypes   = helpers::getRawPointer(this->patTypes());
		    fn1.contextMV  = ((m_externalDataMV.size() ==this->size() * 2)?
				      helpers::getRawPointer(m_externalDataMV) : NULL);
		    fn1.resolution = this->getResolution();
		    fn1.indexLoadM = this->m_indexLoadMethod;
		    
		    int n = this->curMaxSeqLength() * this->parallelSequences() * this->size();
		    thrust::for_each(
			thrust::make_zip_iterator(
				thrust::make_tuple(this->_outputs().begin(),
						   thrust::counting_iterator<int>(0))),
			thrust::make_zip_iterator(
				thrust::make_tuple(this->_outputs().begin()           + n,
						   thrust::counting_iterator<int>(0) + n)),
			fn1);
		    
		}
		    
	    }

	    // Flag: this is not a network for GAN
	    m_ganState = NN_SIGMOID_GAN_DEFAULT;

	    // Loading sequence weight if necessary
	    if (m_flagSeqErrorWeights){
		// Currently, auxRealData is loaded as the sequence weight
		if (fraction.auxRealData().size() == 0)
		    throw std::runtime_error("PostoutputLayer requires sequence weights (auxData)");
		m_seqErrorWeights = fraction.auxRealData();
		m_seqErrorWeightsCPU = fraction.auxRealData();
	    }
	    
	}else{

	    // This is a postoutput layer of the GAN generator

	    Layer<TDevice>::loadSequences(fraction, nnState);
	    
	    // The normal case, loading output from previous layer
	    if (this->outputs().size() != m_precedingMiddleOutLayer->secondOutputs().size()) 
		throw std::runtime_error("Postoutput size is unequal to middlepost secondOutputs");
	    
	    if (!this->outputs().empty())
        	thrust::copy(m_precedingMiddleOutLayer->secondOutputs().begin(),
			     m_precedingMiddleOutLayer->secondOutputs().end(), 
			     this->outputs().begin());

	    // This part is dirty
	    if (nnState == NN_STATE_GAN_DIS_NATDATA){
		m_ganState = NN_SIGMOID_GAN_ONE_SIDED_FLAG;
	    }else{
		m_ganState = NN_SIGMOID_GAN_DEFAULT;
	    }
	    
	}
    }
    template <typename TDevice>
    int PostOutputLayer<TDevice>::ganState()
    {
	// m_ganState is only used for one-sided label smoothing
	return m_ganState;
    }

    template <typename TDevice>
    void PostOutputLayer<TDevice>::computeBackwardPass(const int nnState)
    {
	if (m_precedingMiddleOutLayer && nnState == NN_STATE_GAN_GEN){
	    // Generator only part
	    thrust::fill(this->outputErrors().begin(), this->outputErrors().end(), (real_t)(-1.0));
	    thrust::transform(this->_outputErrors().begin(), this->_outputErrors().end(),
			      this->outputErrors().begin(),  this->_outputErrors().begin(),
			      thrust::plus<real_t>());
	}
    }
    

    /* Add 0401 wang for weighted MSE*/
    // return flag
    template <typename TDevice>
    bool PostOutputLayer<TDevice>::flagMseWeight()
    {
	return this->m_flagMseWeight;
    }
    
    /* Add 1012 read the mean and variance vector */
    template <typename TDevice>
    bool PostOutputLayer<TDevice>::readMV(const PostOutputLayer<TDevice>::cpu_real_vector &mVec, 
					  const PostOutputLayer<TDevice>::cpu_real_vector &vVec)
    {
	if (mVec.size() != this->size() || vVec.size() != this->size()){
	    throw std::runtime_error(std::string("Unmatched number of dimension of mean and var"));
	}
	Cpu::real_vector tmpVec(this->size()*2, 0.0);
	thrust::copy(mVec.begin(), mVec.end(), tmpVec.begin());
	thrust::copy(vVec.begin(), vVec.end(), tmpVec.begin()+this->size());
	m_targetDataMV = tmpVec;
	return true;
    }

    
    // initialize the weight for Mse calculation
    template <typename TDevice>
    bool PostOutputLayer<TDevice>::readMseWeight(const std::string mseWeightPath)
    {
	// Dust 01
	Cpu::real_vector tempVec2;
	misFuncs::ReadRealData(mseWeightPath, tempVec2);
	m_outputMseWeights    = tempVec2;
	m_outputMseWeightsCPU = tempVec2;
	std::cout << "Read #dim " << tempVec2.size() << " mse vector" << std::endl;
	m_flagMseWeight       = true;
	return true;
    }
    
    template <typename TDevice>
    void PostOutputLayer<TDevice>::reInitWeight()
    {
	// nothing to be done here
    }
    
    template <typename TDevice>
    Layer<TDevice>& PostOutputLayer<TDevice>::precedingLayer()
    {
	return Layer<TDevice>::precedingLayer();
    }
    
    template <typename TDevice>
    const Layer<TDevice>& PostOutputLayer<TDevice>::precedingLayer() const
    {
	return Layer<TDevice>::precedingLayer();
    }
    
    template <typename TDevice>
    bool PostOutputLayer<TDevice>::flagTrainable() const
    {
	return false;
    }

    // Functions to retrieve the feedback data
    template <typename TDevice>
    void PostOutputLayer<TDevice>::retrieveFeedBackData()
    {
	// directly copy the targets to the feedbackOutputs
	thrust::copy(this->outputs().begin(), this->outputs().end(),
		     m_feedBackOutput.begin());
    }
    
    template <typename TDevice>
    void PostOutputLayer<TDevice>::retrieveFeedBackData(real_vector& randNum, const int method)
    {
	// only used for dropout feedback (zero and 1/N backoff)
	internal::feedBackDropOut fn;
	fn.dropProb = helpers::getRawPointer(randNum);
	fn.buffer   = helpers::getRawPointer(m_feedBackOutput);
	fn.bufDim   = this->size();
	fn.parall   = this->precedingLayer().parallelSequences();

	int n = this->precedingLayer().curMaxSeqLength();
	n = n * this->precedingLayer().parallelSequences() * this->size();
	thrust::for_each(
	  thrust::make_zip_iterator(
		thrust::make_tuple(m_feedBackOutput.begin(), 
				   thrust::counting_iterator<int>(0))),
	  thrust::make_zip_iterator(
		thrust::make_tuple(m_feedBackOutput.begin() + n, 
				   thrust::counting_iterator<int>(0) + n)),
	  fn);
    }
    
    template <typename TDevice>
    void PostOutputLayer<TDevice>::retrieveFeedBackData(const int timeStep, const int method)
    {
	// only the generated output will be feedback in this function
	// In default, only one utterance is processed (parallel = 1)
	// Step1: copy 
	int startTime = timeStep * this->size();
	int endTime   = (timeStep + 1) * this->size();

	thrust::copy(this->precedingLayer().outputs().begin() + startTime,
		     this->precedingLayer().outputs().begin() + endTime,
		     m_feedBackOutput.begin()+ startTime);
	
	// Feed back Dropout
	if (method == NN_FEEDBACK_DROPOUT_1N || method == NN_FEEDBACK_DROPOUT_ZERO){
	    internal::feedBackDropOut fn;
	    fn.dropProb = NULL;
	    fn.buffer   = helpers::getRawPointer(m_feedBackOutput);
	    fn.bufDim   = this->size();
	    fn.parall   = this->precedingLayer().parallelSequences();
	    
	    thrust::for_each(
	      thrust::make_zip_iterator(
		thrust::make_tuple(m_feedBackOutput.begin() + startTime, 
				   thrust::counting_iterator<int>(0) + startTime)),
	      thrust::make_zip_iterator(
		thrust::make_tuple(m_feedBackOutput.begin() + endTime, 
				   thrust::counting_iterator<int>(0) + endTime)),
	      fn);
	}
    }

    template <typename TDevice>
    void PostOutputLayer<TDevice>::setFeedBackData(const int timeStep, const int state)
    {
	// This is not used at all;
	throw std::runtime_error("setFeedBackData should be not used for non-MDN layer");
	//return m_feedBackOutput;
    }
    
    template <typename TDevice>
    typename PostOutputLayer<TDevice>::real_vector& PostOutputLayer<TDevice>::feedbackOutputs(
		const bool flagTrain)
    {
	return m_feedBackOutput;
	/*if (flagTrain){
	    return this->outputs();
	}else{
	    return this->precedingLayer().outputs();
	}*/
    }

    template <typename TDevice>
    real_t PostOutputLayer<TDevice>::retrieveProb(const int timeStep, const int state)
    {
	return 0.0;
    }
    
    template <typename TDevice>
    typename PostOutputLayer<TDevice>::real_vector& PostOutputLayer<TDevice>::secondOutputs()
    {
	// This is not used at all;
	throw std::runtime_error("This function should not be used");
	//return m_feedBackOutput;
    }

    template <typename TDevice>
    void PostOutputLayer<TDevice>::linkMiddleOutptu(
		PostOutputLayer<TDevice> *precedingMiddleOutLayer)
    {
	m_precedingMiddleOutLayer = precedingMiddleOutLayer;
    }
	    
    template <typename TDevice>
    void PostOutputLayer<TDevice>::exportLayer(const helpers::JsonValue &layersArray, 
					      const helpers::JsonAllocator &allocator) const
    {
        Layer<TDevice>::exportLayer(layersArray, allocator);
	if (m_useExternalOutput)
	    (*layersArray)[layersArray->Size() - 1].AddMember("useExternalOutput",
							      m_useExternalOutput, allocator);
	if (m_externalDataMVStr.size())
	    (*layersArray)[layersArray->Size() - 1].AddMember("externalDataMV",
							      m_externalDataMVStr.c_str(),
							      allocator);

	if (m_indexLoadMethod > 0)
	    (*layersArray)[layersArray->Size() - 1].AddMember("indexLoadMethod",
							      m_indexLoadMethod,
							      allocator);
    }
    
    template <typename TDevice>
    void PostOutputLayer<TDevice>::setPostLayerType(const int flag)
    {
	m_postoutputFlag = flag;
    }

    template <typename TDevice>
    const int& PostOutputLayer<TDevice>::_postLayerType()
    {
	return m_postoutputFlag;
    }

    template <typename TDevice>
    const int& PostOutputLayer<TDevice>::postLayerType()
    {
	return m_postoutputFlag;
    }    

    template <typename TDevice>
    typename PostOutputLayer<TDevice>::real_vector& PostOutputLayer<TDevice>::externalOutputMV()
    {
	return m_externalDataMV;
    }

    template <typename TDevice>
    bool PostOutputLayer<TDevice>::checkOutputValid(const int option)
    {
	return checkWaveformValidity(this->outputs(), this->curMaxSeqLength());
    }
    
    
    // explicit template instantiations
    template class PostOutputLayer<Cpu>;
    template class PostOutputLayer<Gpu>;

} // namespace layers


/*
	// Dust 01
	std::ifstream ifs(mseWeightPath.c_str(), std::ifstream::binary | std::ifstream::in);
	if (!ifs.good()){
	    throw std::runtime_error(std::string("Fail to open ")+mseWeightPath);
	}
	m_flagMseWeight         = true;

	// get the number of we data
	std::streampos numEleS, numEleE;
	long int numEle;
	numEleS = ifs.tellg();
	ifs.seekg(0, std::ios::end);
	numEleE = ifs.tellg();
	numEle  = (numEleE-numEleS)/sizeof(real_t);
	ifs.seekg(0, std::ios::beg);
	
	if (numEle != this->size()){
	    printf("MSE weight vector length incompatible: %d %d", (int)numEle, (int)this->size());
	    throw std::runtime_error("Error in MSE weight configuration");
	}
	
	// read in the data
	real_t tempVal;
	std::vector<real_t> tempVec;
	for (unsigned int i = 0; i<numEle; i++){
	    ifs.read ((char *)&tempVal, sizeof(real_t));
	    tempVec.push_back(tempVal);
	    }
	
	Cpu::real_vector tempVec2(numEle, 1.0);
	thrust::copy(tempVec.begin(), tempVec.end(), tempVec2.begin());
	ifs.close();
*/
