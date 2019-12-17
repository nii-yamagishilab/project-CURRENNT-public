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

#include "TrainableLayer.hpp"
#include "../helpers/getRawPointer.cuh"
#include "../helpers/JsonClasses.hpp"
#include "../Configuration.hpp"

#include <stdexcept>

#include <boost/random/normal_distribution.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#include <boost/random/mersenne_twister.hpp>

#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>
#include <thrust/iterator/counting_iterator.h>

#include <cmath>

#define NNTRAINABLELAYER_BIAS_INI_0      0.0
#define NNTRAINABLELAYER_BIAS_INI_2      2.0

#define NNTRAINABLELAYER_SPECIAL_INI_0 0  // default, just initialize every weight randomly
#define NNTRAINABLELAYER_SPECIAL_INI_1 1  // special 1: assume feedforward_identity
                                          //            w = 0, b = 2.0 
#define NNTRAINABLELAYER_SPECIAL_INI_2 2  // special 1: assume feedforward_identity
                                          //            w = 0, b = 0.0

#define NNTRAINABLELAYER_SPECIAL_INI_3 3  // special 3: assume feedforward_identity
                                          //            w = 1/N, b = 0.0

namespace internal {
namespace {
    struct CopyWeight
    {
	real_t *sourceW;
	real_t *targetW;
	int sNRow;
	int tNRow;
	int mode;

	__host__ __device__ void operator() (const int idx) const
	{
	    int tarIdx = (mode==2) ? ( (idx / sNRow) * tNRow + idx % sNRow) : (idx);
	    *(targetW + tarIdx) = *(sourceW + idx);
	}
    };
}
}

namespace layers {

    template <typename TDevice>
    typename TrainableLayer<TDevice>::real_vector& TrainableLayer<TDevice>::_weightUpdates()
    {
        return m_weightUpdates;
    }
         
    template <typename TDevice>
    const unsigned& TrainableLayer<TDevice>::optOpt() const
    {
        return m_optOpt;
    }
    
    template <typename TDevice>
    TrainableLayer<TDevice>::TrainableLayer(const helpers::JsonValue &layerChild, 
					    const helpers::JsonValue &weightsSection, 
                                            int inputWeightsPerBlock, 
					    int internalWeightsPerBlock, 
					    Layer<TDevice> &precedingLayer,
					    int maxSeqLength,
					    int layerID)
        : Layer<TDevice>           (layerChild,
				    precedingLayer.parallelSequences(), 
				    maxSeqLength,
				    Configuration::instance().trainingMode(),
				    layerID,
				    &precedingLayer)
        , m_inputWeightsPerBlock   (inputWeightsPerBlock)
        , m_internalWeightsPerBlock(internalWeightsPerBlock)
        , m_bias          (layerChild->HasMember("bias") ? 
			   static_cast<real_t>((*layerChild)["bias"].GetDouble()) : 0)
        , m_learningRate  (layerChild->HasMember("learningRate") ? 
			   static_cast<real_t>((*layerChild)["learningRate"].GetDouble()) : -1)
	, m_weightNum (-1)
	, m_optOpt    (0)
	, m_specialInitialization (NNTRAINABLELAYER_SPECIAL_INI_0)
    {
        // std::cout << "Creating layer " << this->name() << std::endl;
        // check if the bias value exists
        if (!layerChild->HasMember("bias"))
            throw std::runtime_error(std::string("Missing value 'bias' in layer '") + 
				     this->name() + "'");

	const Configuration &config = Configuration::instance();
	
	//m_optOpt = config.optimizerOption();
	
	// extract the weights if they are given in the network file
        Cpu::real_vector weights;

	if (m_learningRate > -0.5)
	    printf("\n\tlearning rate = %f", m_learningRate);

	if (layerChild->HasMember("specialInitialization"))
	    m_specialInitialization = (*layerChild)["specialInitialization"].GetInt();
	
        if (weightsSection.isValid() && weightsSection->HasMember(this->name().c_str())) {
	    // Load weights from input network.jsn or epoch***.autosave
	    printf("\n\tTrainable layer: re-read weight");
            if (!weightsSection->HasMember(this->name().c_str()))
                throw std::runtime_error(std::string("Missing weights section for layer '") + 
					 this->name() + "'");
            const rapidjson::Value &weightsChild = (*weightsSection)[this->name().c_str()];
            if (!weightsChild.IsObject())
                throw std::runtime_error(std::string("Weights section for layer '") + 
					 this->name() + "' is not an object");

            if (!weightsChild.HasMember("input") || !weightsChild["input"].IsArray())
                throw std::runtime_error(std::string("Missing array 'weights/") + 
					 this->name() + "/input'");
            if (!weightsChild.HasMember("bias") || !weightsChild["bias"].IsArray())
                throw std::runtime_error(std::string("Missing array 'weights/") + 
					 this->name() + "/bias'");
            if (!weightsChild.HasMember("internal") || !weightsChild["internal"].IsArray())
                throw std::runtime_error(std::string("Missing array 'weights/") + 
					 this->name() + "/internal'");
        
            const rapidjson::Value &inputWeightsChild    = weightsChild["input"];
            const rapidjson::Value &biasWeightsChild     = weightsChild["bias"];
            const rapidjson::Value &internalWeightsChild = weightsChild["internal"];

            if (inputWeightsChild.Size() != (this->size() * inputWeightsPerBlock *
					     this->precedingLayer().size())){
		if (inputWeightsPerBlock == 0){
		    printf("\n\tWARNING: the network file has no input weight for layer %s",
			   this->name().c_str());
		    printf("\n\tIgnore this warning if this layer is SkipAdd or SkipIni");
		}else{
		    throw std::runtime_error(std::string("Invalid number of input weight:")
					     + this->name());
		}
	    }
            if (biasWeightsChild.Size() != this->size() * inputWeightsPerBlock){
		if (inputWeightsPerBlock == 0){
		    printf("\n\tWARNING: the network file has no input weight for layer %s. ",
			   this->name().c_str());
		    printf("\n\tIgnore this warning if this layer is SkipAdd or SkipIni.");
		}else{
		    throw std::runtime_error(std::string("Invalid number of bias weight:") 
					     + this->name());
		}
	    }
            if (internalWeightsChild.Size() != this->size() * internalWeightsPerBlock)
                throw std::runtime_error(std::string("Invalid number of internal weight:'")
					 + this->name());
	    
	    
	    if ((inputWeightsPerBlock + internalWeightsPerBlock) == 0){
		// skip reading the weight
		// for compatability with old version of CURRENNT, where skpiini has weights
		weights.clear();
		
	    }else{

		weights.reserve(inputWeightsChild.Size() +  biasWeightsChild.Size()  + 
				internalWeightsChild.Size());

		for (rapidjson::Value::ConstValueIterator it = inputWeightsChild.Begin(); 
		     it != inputWeightsChild.End(); ++it)
		    weights.push_back(static_cast<real_t>(it->GetDouble()));
		for (rapidjson::Value::ConstValueIterator it = biasWeightsChild.Begin(); 
		     it != biasWeightsChild.End(); ++it)
		    weights.push_back(static_cast<real_t>(it->GetDouble()));
		for (rapidjson::Value::ConstValueIterator it = internalWeightsChild.Begin(); 
		     it != internalWeightsChild.End(); ++it)
		    weights.push_back(static_cast<real_t>(it->GetDouble()));
	    }
	    
        }else {
	    
	    // create random weights if no weights are given in the network file
            weights.resize(this->size() * 
			   (inputWeightsPerBlock * (this->precedingLayer().size() + 1) + 
			    internalWeightsPerBlock));
	    
	    printf("\n\tTrainable layer: ");
	    
	    if (m_specialInitialization == NNTRAINABLELAYER_SPECIAL_INI_0){
		// normal initialization
		printf("initialize weight randomly");
		static boost::mt19937 *gen = NULL;
		if (!gen) {
		    gen = new boost::mt19937;
		    gen->seed(config.randomSeed());
		}
            
		if (config.weightsDistributionType()==Configuration::DISTRIBUTION_UNIFORM) {
		    real_t range = (config.weightsDistributionUniformMax() - 
				    config.weightsDistributionUniformMin());
		    boost::random::uniform_real_distribution<real_t> dist(0, range);
		    for (size_t i = 0; i < weights.size(); ++i)
			weights[i] = dist(*gen) + config.weightsDistributionUniformMin();
		    
		}else if (config.weightsDistributionType()==
			  Configuration::DISTRIBUTION_UNINORMALIZED){
		    // Add 02-29 Wang, for uniform distribution with normalzied min-max , 
		    // Xavier Glorot, Understanding the dif ... 2010
		    // Here, we only make a approximation by assuming 
		    //    n_i+1 = n_i, x~[-sqrt(3)/sqrt(n), sqrt(3)/sqrt(n)]
		    real_t range = 2*std::sqrt(3.0/(real_t)this->size());
		    boost::random::uniform_real_distribution<real_t> dist(0, range);
		    for (size_t i = 0; i < weights.size(); ++i)
			weights[i] = dist(*gen) - range/2.0;
		    
		}else {
		    boost::random::normal_distribution<real_t> dist(
		     config.weightsDistributionNormalMean(),
		     config.weightsDistributionNormalSigma());
		    for (size_t i = 0; i < weights.size(); ++i)
			weights[i] = dist(*gen);
		}
		
	    }else if (m_specialInitialization == NNTRAINABLELAYER_SPECIAL_INI_1 ||
		      m_specialInitialization == NNTRAINABLELAYER_SPECIAL_INI_2 ||
		      m_specialInitialization == NNTRAINABLELAYER_SPECIAL_INI_3){

		// special initialization strategy
		if (inputWeightsPerBlock == 1 && internalWeightsPerBlock == 0){

		    
		    if (m_specialInitialization == NNTRAINABLELAYER_SPECIAL_INI_1){
			// w = 0, b=2.0
			printf("w = 0.0, bias = %f", NNTRAINABLELAYER_BIAS_INI_2);
			
			thrust::fill(weights.begin(), weights.end(),
				     NNTRAINABLELAYER_BIAS_INI_2);
			    
			thrust::fill(weights.begin(),
				     weights.begin() +
				     this->size() * this->precedingLayer().size(),
				     0.0);
			
		    }else if (m_specialInitialization == NNTRAINABLELAYER_SPECIAL_INI_2){
			// w = 0, b = 0.0
			printf("w = 0.0, bias = %f", NNTRAINABLELAYER_BIAS_INI_0);
			thrust::fill(weights.begin(), weights.end(), 0.0);
			
		    }else{
			// w = 1/layer_size, b = 0.0
			printf("w = %f, bias = 0",
			       1.0/(this->size() * this->precedingLayer().size()));
			thrust::fill(weights.begin(), weights.end(), 0.0);
			thrust::fill(weights.begin(),
				     weights.begin() +
				     this->size() * this->precedingLayer().size(),
				     1.0/(this->size() * this->precedingLayer().size()));
		    }
		    
		}else{
		    throw std::runtime_error("specialInitialization 1/2 only for feedforward");
		}
	    }else{
		throw std::runtime_error("Unknown specialInitialization in network.jsn");
	    }
        }

        m_weights       = weights;
        m_weightUpdates = weights;
	
	// Add 04013 Wang: for weight Mask
	for (size_t i = 0; i < weights.size(); ++i)
	    weights[i] = 1.0;
	m_weightMask    = weights;          // make it the same length as weights matrix 
	m_weightNum     = weights.size(); 
	m_weightMaskFlag= false;

    }

   
    template <typename TDevice>
    TrainableLayer<TDevice>::~TrainableLayer()
    {
    }

    
    template <typename TDevice>
    Layer<TDevice>& TrainableLayer<TDevice>::precedingLayer()
    {
        return Layer<TDevice>::precedingLayer(); //m_precedingLayer;
    }

    template <typename TDevice>
    const Layer<TDevice>& TrainableLayer<TDevice>::precedingLayer() const
    {
        return Layer<TDevice>::precedingLayer(); //m_precedingLayer;
    }
    
    template <typename TDevice>
    real_t TrainableLayer<TDevice>::bias() const
    {
        return m_bias;
    }

    template <typename TDevice>
    real_t TrainableLayer<TDevice>::learningRate() const
    {
        return m_learningRate;
    }

    /*  template <typename TDevice>
    typename TrainableLayer<TDevice>::real_vector& TrainableLayer<TDevice>::outputErrors()
    {
        return m_outputErrors;
    }*/

    template <typename TDevice>
    typename TrainableLayer<TDevice>::real_vector& TrainableLayer<TDevice>::weights()
    {
        return m_weights;
    }

    template <typename TDevice>
    const typename TrainableLayer<TDevice>::real_vector& TrainableLayer<TDevice>::weights() const
    {
        return m_weights;
    }
    
    template <typename TDevice>
    const typename TrainableLayer<TDevice>::real_vector& TrainableLayer<TDevice>::weightMask() const
    {
	return m_weightMask;
    }

    template <typename TDevice>
    const typename TrainableLayer<TDevice>::real_vector& TrainableLayer<TDevice>::weightUpdates() const
    {
        return m_weightUpdates;
    }

    // Add 0413 weight matrix
    template <typename TDevice>
    const int& TrainableLayer<TDevice>::weightNum() const
    {
	return m_weightNum;
    }
    template <typename TDevice>
    void TrainableLayer<TDevice>::readWeightMask(std::vector<real_t>::iterator b,
						 std::vector<real_t>::iterator e)
    {
	Cpu::real_vector tempVec;
	bool tempflag;
	tempflag = false;
	tempVec.resize(m_weightNum);
	std::vector<real_t>::iterator t = b;
	for (int i = 0; t != e; ++t, i++){
	    if (*t <0 || *t >1){
		tempflag = true;
		tempVec[i] = 1;
	    }else
		tempVec[i] = *t;
	}
	if (tempflag){
	    throw std::runtime_error("DataMask is out of range [0, 1]");
	}
	    
	// copy the mask data into m_weightMask
	m_weightMask = tempVec;
	
	// set the flag
	m_weightMaskFlag = true;
    }
    
    template <typename TDevice>
    const bool& TrainableLayer<TDevice>::flagUseWeightMask() const
    {
	return m_weightMaskFlag;
    }

    template <typename TDevice>
    void TrainableLayer<TDevice>::maskWeight()
    {
	thrust::transform(weights().begin(),    weights().end(),
			  weightMask().begin(), weights().begin(),
			  thrust::multiplies<real_t>());
    }

    template <typename TDevice>
    void TrainableLayer<TDevice>::injectWeightNoise(real_t sigma) 
    {
        // generate vector of weight noise on the host
        // note: RNG is sequential, so we can't parallelize ...
        static boost::mt19937 *gen = NULL;
        if (!gen) {
            gen = new boost::mt19937;
            gen->seed(Configuration::instance().randomSeed());
        }
        boost::normal_distribution<real_t> dist(0.0f, sigma);
        Cpu::real_vector weightNoise(weights().size());
        for (int i = 0; i < weightNoise.size(); ++i) {
            weightNoise[i] = dist(*gen);
        }

        // copy weight noise to device
        real_vector weightNoiseD(weights().size());
        thrust::copy(weightNoise.begin(), weightNoise.end(), weightNoiseD.begin());

        // add weight noise to device vector of weights
        thrust::transform(weights().begin(), weights().end(),
			  weightNoiseD.begin(), weights().begin(), thrust::plus<real_t>());
    }

    template <typename TDevice>
    void TrainableLayer<TDevice>::exportWeights(const helpers::JsonValue &weightsObject,
						const helpers::JsonAllocator &allocator) const
    {
        if (!weightsObject->IsObject())
            throw std::runtime_error("The JSON value is not an object");

        // do nothing if we don't have any weights
        if (m_weights.empty())
            return;

        // create and fill the weight arrays
        rapidjson::Value inputWeightsArray(rapidjson::kArrayType);
        int inputWeightsCount = (this->size() * m_inputWeightsPerBlock *
				 this->precedingLayer().size());
        inputWeightsArray.Reserve(inputWeightsCount, allocator);
        for (int i = 0; i < inputWeightsCount; ++i)
            inputWeightsArray.PushBack(m_weights[i], allocator);

        rapidjson::Value biasWeightsArray(rapidjson::kArrayType);
        int biasWeightsCount = this->size() * m_inputWeightsPerBlock;
        biasWeightsArray.Reserve(biasWeightsCount, allocator);
        for (int i = 0; i < biasWeightsCount; ++i)
            biasWeightsArray.PushBack(m_weights[inputWeightsCount + i], allocator);

        rapidjson::Value internalWeightsArray(rapidjson::kArrayType);
        int internalWeightsCount = this->size() * m_internalWeightsPerBlock;
        internalWeightsArray.Reserve(internalWeightsCount, allocator);
        for (int i = 0; i < internalWeightsCount; ++i)
            internalWeightsArray.PushBack(m_weights[inputWeightsCount + biasWeightsCount + i],
					  allocator);

        // create and fill the weights subsection
        rapidjson::Value weightsSection(rapidjson::kObjectType);
        weightsSection.AddMember("input",    inputWeightsArray,    allocator);
        weightsSection.AddMember("bias",     biasWeightsArray,     allocator);
        weightsSection.AddMember("internal", internalWeightsArray, allocator);

        // add the weights section tot he weights object
        weightsObject->AddMember(this->name().c_str(), weightsSection, allocator);
    }

    template <typename TDevice>
    void TrainableLayer<TDevice>::exportLayer(const helpers::JsonValue &layersArray, 
					      const helpers::JsonAllocator &allocator) const
    {
        Layer<TDevice>::exportLayer(layersArray, allocator);
        (*layersArray)[layersArray->Size() - 1].AddMember("bias", m_bias, allocator);
	if (m_learningRate >= 0.0)
	    (*layersArray)[layersArray->Size() - 1].AddMember("learningRate", m_learningRate,
							      allocator);

    }
    
    // Add 0511: re-initialize the weight (used for learning_rate checking)
    template <typename TDevice>
    void TrainableLayer<TDevice>::reInitWeight()
    {
	// copied from the initializer of TrainableLayer
	
	Cpu::real_vector weights;
	weights.resize(this->size() *
		       (m_inputWeightsPerBlock * (this->precedingLayer().size() + 1) +
			m_internalWeightsPerBlock));
	const Configuration &config = Configuration::instance();
	static boost::mt19937 *gen = NULL;
	if (!gen) {
	    gen = new boost::mt19937;
	    gen->seed(config.randomSeed());
	}
        if (config.weightsDistributionType() == Configuration::DISTRIBUTION_UNIFORM) {
	    real_t range = config.weightsDistributionUniformMax()-
		config.weightsDistributionUniformMin();
	    boost::random::uniform_real_distribution<real_t> dist(0, range);
	    for (size_t i = 0; i < weights.size(); ++i)
		weights[i] = dist(*gen) + config.weightsDistributionUniformMin();
	}else if (config.weightsDistributionType() == Configuration::DISTRIBUTION_UNINORMALIZED){
	    // Add 02-29 Wang, for uniform distribution with normalzied min-max , 
	    // Xavier Glorot, Understanding the dif ... 2010
	    // Here, we only make a approximation by assuming 
	    //    n_i+1 = n_i, x~[-sqrt(3)/sqrt(n), sqrt(3)/sqrt(n)]
	    real_t range = 2*std::sqrt(3.0/(real_t)this->size());
	    boost::random::uniform_real_distribution<real_t> dist(0, range);
	    for (size_t i = 0; i < weights.size(); ++i)
		weights[i] = dist(*gen) - range/2.0;
	    
	}else {
	    boost::random::normal_distribution<real_t> dist(
		config.weightsDistributionNormalMean(), config.weightsDistributionNormalSigma());
	    for (size_t i = 0; i < weights.size(); ++i)
		weights[i] = dist(*gen);
	}
	m_weights       = weights;
        m_weightUpdates = weights;

    }
    
    // Add 0527: re-read the weight from weightsSection
    template <typename TDevice>
    void TrainableLayer<TDevice>::reReadWeight(const helpers::JsonValue &weightsSection,
					       const int layerSize, const int readCtrFlag)
    {

	Cpu::real_vector weights;
	if (weightsSection.isValid() && weightsSection->HasMember(this->name().c_str())){
	    printf("read weight for layer %s", this->name().c_str());
	    const rapidjson::Value &weightsChild = (*weightsSection)[this->name().c_str()];
            if (!weightsChild.IsObject())
                throw std::runtime_error(std::string("Weights section for layer '") + 
					 this->name() + "' is not an object");
            if (!weightsChild.HasMember("input") || !weightsChild["input"].IsArray())
                throw std::runtime_error(std::string("Missing array 'weights/") + 
					 this->name() + "/input'");
            if (!weightsChild.HasMember("bias") || !weightsChild["bias"].IsArray())
                throw std::runtime_error(std::string("Missing array 'weights/") + 
					 this->name() + "/bias'");
            if (!weightsChild.HasMember("internal") || !weightsChild["internal"].IsArray())
                throw std::runtime_error(std::string("Missing array 'weights/") + 
					 this->name() + "/internal'");
	    const rapidjson::Value &inputWeightsChild    = weightsChild["input"];
            const rapidjson::Value &biasWeightsChild     = weightsChild["bias"];
            const rapidjson::Value &internalWeightsChild = weightsChild["internal"];
	    
	    // three kinds of possibility to read the weights
	    if (readCtrFlag==1){
		// the number of parameter should match exactly
		if (inputWeightsChild.Size() != layerSize * 
		    m_inputWeightsPerBlock * this->precedingLayer().size())
		    throw std::runtime_error(std::string("Invalid number of input weights: '") 
					     + this->name() + "'");
		if (biasWeightsChild.Size() != layerSize * m_inputWeightsPerBlock)
		    throw std::runtime_error(std::string("Invalid number of bias weights: '") 
					     + this->name() + "'");
		if (internalWeightsChild.Size() != layerSize * m_internalWeightsPerBlock)
		    throw std::runtime_error(std::string("Invalid number of internal '") 
					     + this->name() + "'");

		weights.reserve(inputWeightsChild.Size() + 
				biasWeightsChild.Size() + 
				internalWeightsChild.Size());

		for (rapidjson::Value::ConstValueIterator it = inputWeightsChild.Begin(); 
		     it != inputWeightsChild.End(); 
		     ++it)
		    weights.push_back(static_cast<real_t>(it->GetDouble()));
		for (rapidjson::Value::ConstValueIterator it = biasWeightsChild.Begin(); 
		     it != biasWeightsChild.End(); 
		     ++it)
		    weights.push_back(static_cast<real_t>(it->GetDouble()));
		for (rapidjson::Value::ConstValueIterator it = internalWeightsChild.Begin(); 
		     it != internalWeightsChild.End(); 
		     ++it)
		    weights.push_back(static_cast<real_t>(it->GetDouble()));
		
	    }else if (readCtrFlag == 2 || readCtrFlag == 3){
		
		if (m_inputWeightsPerBlock != 1 || m_internalWeightsPerBlock != 0)
		    throw std::runtime_error(std::string("trainParameterCtr=2 not support LSTM"));
		
		int tempThisWeightSize = this->size() * 
		    (m_inputWeightsPerBlock * (this->precedingLayer().size() + 1) + 
		     m_internalWeightsPerBlock);
		
		int preTrainedNRow     = ((readCtrFlag == 2) ? 
					  (inputWeightsChild.Size() / this->size()) : 
					  (this->precedingLayer().size()));

		if (inputWeightsChild.Size() > tempThisWeightSize)
		    throw std::runtime_error(std::string("not support larger pre-trained layer"));
		
		// space for this layer, the remaining parameter are 0.0
		weights.resize(tempThisWeightSize, 0.0);
		
		// space for the pretrained matrix
		Cpu::real_vector preTrainedWeights;
		preTrainedWeights.reserve(inputWeightsChild.Size() + 
					  biasWeightsChild.Size() + 
					  internalWeightsChild.Size());
		
		
		for (rapidjson::Value::ConstValueIterator it = inputWeightsChild.Begin(); 
		     it != inputWeightsChild.End(); 
		     ++it)
		    preTrainedWeights.push_back(static_cast<real_t>(it->GetDouble()));
		for (rapidjson::Value::ConstValueIterator it = biasWeightsChild.Begin(); 
		     it != biasWeightsChild.End(); 
		     ++it)
		    preTrainedWeights.push_back(static_cast<real_t>(it->GetDouble()));
		
		// copy the weight part
		{{
			
			internal::CopyWeight fn;
			fn.sNRow = preTrainedNRow;
			fn.tNRow = this->precedingLayer().size();
			fn.mode  = readCtrFlag;
			fn.sourceW = helpers::getRawPointer(preTrainedWeights);
			fn.targetW = helpers::getRawPointer(weights);

			int n = inputWeightsChild.Size();
			//thrust::counting_iterator<int> first(0);
			//thrust::counting_iterator<int> last = first + n;
			//thrust::for_each(first, last, fn);
			
			for (int idx=0; idx < n; idx++){
			    int tarIdx = ((readCtrFlag==2) ? 
					  ((idx / fn.sNRow) * fn.tNRow + idx % fn.sNRow) : 
					  (idx));
			    weights[tarIdx] = preTrainedWeights[idx];
			}
			
		}}
		
		// copy the bias part
		thrust::copy(preTrainedWeights.begin() + inputWeightsChild.Size(),
			     preTrainedWeights.end(),
			     weights.begin() + this->size() * this->precedingLayer().size());
		
	    }else{
		throw std::runtime_error(std::string("trainedParameterCtr not string of 0/1/2/3"));
	    }
	    m_weights       = weights;
	    m_weightUpdates = weights;
	    
	}else{
	    printf("not read weight for layer %s", this->name().c_str());
	}
    }
    
    template <typename TDevice>
    int TrainableLayer<TDevice>::inputWeightsPerBlock()
    {
	return m_inputWeightsPerBlock;

    }
    template <typename TDevice>
    int TrainableLayer<TDevice>::internalWeightsPerBlock()
    {
	return m_internalWeightsPerBlock;
    }

    template <typename TDevice>
    void TrainableLayer<TDevice>::cleanGradidents()
    {
	// Fatal Error: clean gradients should clean the m_weightUpdates, not outputErrors
	thrust::fill(m_weightUpdates.begin(), m_weightUpdates.end(), 0.0);
	// thrust::fill(this->outputErrors().begin(), this->outputErrors().end(), 0.0);
    }


    // explicit template instantiations
    template class TrainableLayer<Cpu>;
    template class TrainableLayer<Gpu>;

} // namespace layers
