/******************************************************************************
 * This file is an addtional component of CURRENNT. 
 * Xin WANG
 * National Institute of Informatics, Japan
 * 2019
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


#include "InterWeaveLayer.hpp"

#include "../helpers/getRawPointer.cuh"
#include "../helpers/Matrix.hpp"
#include "../helpers/JsonClasses.hpp"
#include "../helpers/misFuncs.hpp"
#include "../activation_functions/Logistic.cuh"
#include "../activation_functions/Tanh.cuh"
#include "../MacroDefine.hpp"

#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/for_each.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/fill.h>
#include <thrust/random.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>

#include <boost/foreach.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>
#include <vector>
#include <stdexcept>

#define INTERWEAVELAYER_MODE_121323 0     //
#define INTERWEAVELAYER_MODE_RANDOMPICK 1
#define INTERWEAVELAYER_MODE_DEFAULT 1    //


namespace internal{
namespace {

    struct genNoise
    {
	float a, b;
	int   seed;
	
	__host__ __device__
	genNoise(float _a=-1.f, float _b=1.f, int _seed=123) : a(_a), b(_b), seed(_seed) {};

	__host__ __device__
	float operator()(const unsigned int n) const
	{
	    thrust::default_random_engine rng(seed);
	    thrust::uniform_real_distribution<float> dist(a, b);
	    rng.discard(n);
	    return dist(rng);
	}
    };

    
    struct InterweaveFeature_121323
    {
	int preLayerSize;
	int curLayerSize;

	real_t *preLayerOutput;   // previous layer's output
	real_t *stateBuffer;      // buffer to store the state sequence
	
	const char *patTypes;

	__host__ __device__ void operator() (const thrust::tuple<real_t&, int> &t) const
	{
	    
	    int outputIdx = t.get<1>();
	    int timeIdx   = outputIdx / curLayerSize;
	    int dimIdx    = outputIdx % curLayerSize;

	    int tmp_timeIdx = timeIdx / 3;
	    
	    // skip dummy frame (for parallel sentence processing)
	    if (patTypes != NULL &&
		(patTypes[tmp_timeIdx * 3 + 0] == PATTYPE_NONE ||
		 patTypes[tmp_timeIdx * 3 + 1] == PATTYPE_NONE ||
		 patTypes[tmp_timeIdx * 3 + 2] == PATTYPE_NONE)){
		t.get<0>() = 0.0;
		stateBuffer[timeIdx] = -1.0;
	    }else{
		
		if (dimIdx < preLayerSize){
		    if (timeIdx % 3 == 0 || timeIdx % 3 == 1)	
			t.get<0>() = preLayerOutput[tmp_timeIdx * 3 * preLayerSize + dimIdx];
		    else
			t.get<0>() = preLayerOutput[(tmp_timeIdx * 3 + 1) * preLayerSize + dimIdx];
		}else{
		    dimIdx = dimIdx - preLayerSize;
		    if (timeIdx % 3 == 0)	
			t.get<0>() = preLayerOutput[(tmp_timeIdx * 3 + 1) * preLayerSize + dimIdx];
		    else
			t.get<0>() = preLayerOutput[(tmp_timeIdx * 3 + 2) * preLayerSize + dimIdx];
		}

		// state buffer
		if (timeIdx % 3 == 0)
		    stateBuffer[timeIdx] = 1.0;   // the two vectors are from the same sequence
		else
		    stateBuffer[timeIdx] = 0.0;   // the two vectors are from different sequences
		
	    }
	}
    };

    struct InterweaveFeature_121323_grad
    {
	int folLayerSize;
	int curLayerSize;

	real_t *inputGrad;   // previous layer's output
	const char *patTypes;

	__host__ __device__ void operator() (const thrust::tuple<real_t&, int> &t) const
	{
	    
	    int outputIdx = t.get<1>();
	    int timeIdx   = outputIdx / curLayerSize;
	    int dimIdx    = outputIdx % curLayerSize;

	    int tmp_timeIdx = timeIdx / 3;
	    // skip dummy frame (for parallel sentence processing)
	    if (patTypes != NULL && patTypes[timeIdx] == PATTYPE_NONE){
		t.get<0>() = 0.0;
	    }else{
		if (timeIdx % 3 == 0){
		    t.get<0>() = inputGrad[tmp_timeIdx * 3 * folLayerSize + dimIdx] +
			inputGrad[(tmp_timeIdx * 3 + 1) * folLayerSize + dimIdx];
		}else if (timeIdx % 3 == 1){
		    t.get<0>() = inputGrad[tmp_timeIdx * 3 * folLayerSize + dimIdx + curLayerSize] +
			inputGrad[(tmp_timeIdx * 3 + 2) * folLayerSize + dimIdx];
		}else{
		    dimIdx = dimIdx + curLayerSize;
		    t.get<0>() = inputGrad[(tmp_timeIdx * 3 + 1) * folLayerSize + dimIdx] +
			inputGrad[(tmp_timeIdx * 3 + 2) * folLayerSize + dimIdx];
		}
		
	    }
	}
    };



    struct InterweaveFeature_random
    {
	int preLayerSize;
	int curLayerSize;

	real_t *preLayerOutput;   // previous layer's output
	real_t *stateBuffer;      // buffer to store the state sequence
	real_t *randomNumbers;
	const char *patTypes;

	__host__ __device__ void operator() (const thrust::tuple<real_t&, int> &t) const
	{
	    
	    int outputIdx = t.get<1>();
	    int timeIdx   = outputIdx / curLayerSize;
	    int dimIdx    = outputIdx % curLayerSize;

	    int tmp_timeIdx = timeIdx / 3;
	    
	    // skip dummy frame (for parallel sentence processing)
	    if (patTypes != NULL &&
		(patTypes[tmp_timeIdx * 3 + 0] == PATTYPE_NONE ||
		 patTypes[tmp_timeIdx * 3 + 1] == PATTYPE_NONE ||
		 patTypes[tmp_timeIdx * 3 + 2] == PATTYPE_NONE)){
		t.get<0>() = 0.0;
		stateBuffer[timeIdx] = -1.0;
	    }else{

		real_t randomNum1 = randomNumbers[timeIdx * 2 + 0];
		real_t randomNum2 = randomNumbers[timeIdx * 2 + 1];
		
		if (dimIdx < preLayerSize){
		    // pick the first chunk based on number 1
		    if (randomNum1 <= 0.333)	
			t.get<0>() = preLayerOutput[tmp_timeIdx * 3 * preLayerSize + dimIdx];
		    else if (randomNum1 <= 0.666)
			t.get<0>() = preLayerOutput[(tmp_timeIdx * 3 + 1) * preLayerSize + dimIdx];
		    else
			t.get<0>() = preLayerOutput[(tmp_timeIdx * 3 + 2) * preLayerSize + dimIdx];
		}else{

		    // pick the second chunk based on number 1 and 2
		    dimIdx = dimIdx - preLayerSize;

		    if (randomNum1 <= 0.333){
			if (randomNum2 < 0.75){
			    t.get<0>() = preLayerOutput[(tmp_timeIdx * 3 + 1) * preLayerSize + dimIdx];
			    stateBuffer[timeIdx] = 1.0;
			}else{
			    t.get<0>() = preLayerOutput[(tmp_timeIdx * 3 + 2) * preLayerSize + dimIdx];
			    stateBuffer[timeIdx] = 0.0;
			}
			    
		    }else if (randomNum1 <= 0.666){
			if (randomNum2 < 0.75){
			    t.get<0>() = preLayerOutput[(tmp_timeIdx * 3 + 0) * preLayerSize + dimIdx];
			    stateBuffer[timeIdx] = 1.0;
			}else{
			    t.get<0>() = preLayerOutput[(tmp_timeIdx * 3 + 2) * preLayerSize + dimIdx];
			    stateBuffer[timeIdx] = 0.0;
			}
		    }else{
			if (randomNum2 < 0.50){
			    t.get<0>() = preLayerOutput[(tmp_timeIdx * 3 + 0) * preLayerSize + dimIdx];
			}else{
			    t.get<0>() = preLayerOutput[(tmp_timeIdx * 3 + 1) * preLayerSize + dimIdx];
			}
			stateBuffer[timeIdx] = 0.0;
		    }
		    
		}

		
	    }
	}
    };


    struct InterweaveFeature_random_grad
    {
	int folLayerSize;
	int curLayerSize;

	real_t *randomState;
	real_t *inputGrad;     // previous layer's output

	real_t *outputGrad;
	const char *patTypes;

	__host__ __device__ void operator() (const thrust::tuple<real_t&, int> &t) const
	{
	    
	    int outputIdx   = t.get<1>();
	    int timeIdx     = outputIdx / curLayerSize;
	    int dimIdx      = outputIdx % curLayerSize;

	    // input data: 3 utterances are interweaved as one block
	    int blockIdx = timeIdx / 3;     // 
	    int blockStart = blockIdx * 3;  // 
	    
	    int blockPtr = 0;
	    
	    if ((timeIdx % 3 > 0) ||
		(patTypes != NULL &&
		 (patTypes[blockStart + 0] == PATTYPE_NONE) ||
		 (patTypes[blockStart + 1] == PATTYPE_NONE) ||
		 (patTypes[blockStart + 2] == PATTYPE_NONE))){
		return;
	    }

	    outputGrad[(blockStart + 0) * curLayerSize + dimIdx] = 0.0;
	    outputGrad[(blockStart + 1) * curLayerSize + dimIdx] = 0.0;
	    outputGrad[(blockStart + 2) * curLayerSize + dimIdx] = 0.0;

	    real_t randomNum1 = 0.0;
	    real_t randomNum2 = 0.0;

	    for (int index123 = 0; index123 < 3; index123++){
		
		blockPtr = blockStart + index123;
		
		randomNum1 = randomState[blockPtr * 2 + 0];
		randomNum2 = randomState[blockPtr * 2 + 1];
		
		if (randomNum1 <= 0.333){
		    if (randomNum2 < 0.75){
			outputGrad[(blockStart + 0) * curLayerSize + dimIdx] +=
			    inputGrad[blockPtr * folLayerSize + dimIdx];
			outputGrad[(blockStart + 1) * curLayerSize + dimIdx] +=
			    inputGrad[blockPtr * folLayerSize + dimIdx + curLayerSize];
		    }else{
			outputGrad[(blockStart + 0) * curLayerSize + dimIdx] +=
			    inputGrad[blockPtr * folLayerSize + dimIdx];
			outputGrad[(blockStart + 2) * curLayerSize + dimIdx] +=
			    inputGrad[blockPtr * folLayerSize + dimIdx + curLayerSize];
		    }
		}else if (randomNum1 <= 0.666){
		    if (randomNum2 < 0.75){
			outputGrad[(blockStart + 1) * curLayerSize + dimIdx] +=
			    inputGrad[blockPtr * folLayerSize + dimIdx];
			outputGrad[(blockStart + 0) * curLayerSize + dimIdx] +=
			    inputGrad[blockPtr * folLayerSize + dimIdx + curLayerSize];
		    }else{
			outputGrad[(blockStart + 1) * curLayerSize + dimIdx] +=
			    inputGrad[blockPtr * folLayerSize + dimIdx];
			outputGrad[(blockStart + 2) * curLayerSize + dimIdx] +=
			    inputGrad[blockPtr * folLayerSize + dimIdx + curLayerSize];
		    }
		}else{
		    if (randomNum2 < 0.75){
			outputGrad[(blockStart + 2) * curLayerSize + dimIdx] +=
			    inputGrad[blockPtr * folLayerSize + dimIdx];
			outputGrad[(blockStart + 0) * curLayerSize + dimIdx] +=
			    inputGrad[blockPtr * folLayerSize + dimIdx + curLayerSize];
		    }else{
			outputGrad[(blockStart + 2) * curLayerSize + dimIdx] +=
			    inputGrad[blockPtr * folLayerSize + dimIdx];
			outputGrad[(blockStart + 1) * curLayerSize + dimIdx] +=
			    inputGrad[blockPtr * folLayerSize + dimIdx + curLayerSize];
		    }
		}	    
	    }				
	    
	}
    };

    
    
}
}

namespace layers{

    // Construct the layer
    template <typename TDevice>
    InterWeaveLayer<TDevice>::InterWeaveLayer(const helpers::JsonValue &layerChild,
					      const helpers::JsonValue &weightsSection,
					      Layer<TDevice>           &precedingLayer,
					      int                       maxSeqLength,
					      int                       layerID)
	: TrainableLayer<TDevice>(layerChild, weightsSection, 0, 0,
				  precedingLayer, maxSeqLength, layerID)
    {
	/* ------- load configuration ------- */
	m_mode = (layerChild->HasMember("mode") ? 
		  (*layerChild)["mode"].GetInt() : INTERWEAVELAYER_MODE_DEFAULT);
	
	m_targetLayerName = (layerChild->HasMember("targetLayer")?
			     (*layerChild)["targetLayer"].GetString(): "");
	
	// allocate memory
	m_stateRandom.resize(maxSeqLength * this->parallelSequences(), -1.0);
	m_randomState.resize(maxSeqLength * this->parallelSequences() * 2, 0.0);

	m_targetLayer = NULL;
	
	/* ------------- check ------------- */
	if (this->size() != this->precedingLayer().size() * 2)
	    throw std::runtime_error("Error: interweave layer size != precedingLayer size * 2");
	
	if (m_mode == INTERWEAVELAYER_MODE_121323 || m_mode == INTERWEAVELAYER_MODE_RANDOMPICK)
	    if (this->parallelSequences() != 3)
		throw std::runtime_error("Error: interweave layer mode 0 requires parallel = 3");
	
    }	

    // Destructor
    template <typename TDevice>
    InterWeaveLayer<TDevice>::~InterWeaveLayer()
    {
    }

    template <typename TDevice>
    const std::string& InterWeaveLayer<TDevice>::type() const
    {
        static std::string s;
        if (s.empty()) s = "mutual_weave";
        return s;
    }

    template <typename TDevice>
    void InterWeaveLayer<TDevice>::computeForwardPass(const int nnState)
    {
	int timeLength = this->curMaxSeqLength() * this->parallelSequences();

	if (m_mode == INTERWEAVELAYER_MODE_121323){
	    // Default mode of inter-weaving

	    // given [x1, x2, x3, x4, x5, x6]
	    // gives this->outputs() = [ [x1, x2], [x1, x3], [x2, x3]]
	    //       m_stateRandom   = [        1,        0,        0]
	    {
		internal::InterweaveFeature_121323 fn;
		fn.preLayerSize = this->precedingLayer().size();
		fn.curLayerSize = this->size();
		
		fn.preLayerOutput = helpers::getRawPointer(this->precedingLayer().outputs());
		fn.stateBuffer    = helpers::getRawPointer(this->m_stateRandom);

		fn.patTypes       = helpers::getRawPointer(this->patTypes());
		
		thrust::for_each(
		    thrust::make_zip_iterator(
			thrust::make_tuple(
				this->outputs().begin(), 
				thrust::counting_iterator<int>(0))),
		    thrust::make_zip_iterator(
			thrust::make_tuple(
				this->outputs().begin()           + timeLength * this->size(), 
				thrust::counting_iterator<int>(0) + timeLength * this->size())),
		    fn);    
	    }
	    thrust::copy(m_stateRandom.begin(), m_stateRandom.end(),
			 m_targetLayer->outputs().begin());
	    
	}else if (m_mode == INTERWEAVELAYER_MODE_RANDOMPICK){
	    // step1. generate random number sequence [[a1_1, a2_1], [a1_2, a2_2] ,...]
	    // 1. a1_n decides which one to pick among the three
	    // 2. a2_n decides which one to pick from the remaining two

	    {
		thrust::counting_iterator<unsigned int> index_sequence_begin(0);
		thrust::transform(index_sequence_begin,
				  index_sequence_begin + timeLength * 2,
				  m_randomState.begin(),
				  internal::genNoise(0, 1.0,
						     (int)(misFuncs::GetRandomNumber()*10000.0)));
	    }
	    
	    // step2. load data
	    {
	    {
		internal::InterweaveFeature_random fn;
		fn.preLayerSize = this->precedingLayer().size();
		fn.curLayerSize = this->size();
	       
		fn.preLayerOutput = helpers::getRawPointer(this->precedingLayer().outputs());
		fn.stateBuffer    = helpers::getRawPointer(this->m_stateRandom);
		fn.randomNumbers  = helpers::getRawPointer(this->m_randomState); 
		fn.patTypes       = helpers::getRawPointer(this->patTypes());
		
		thrust::for_each(
		    thrust::make_zip_iterator(
			thrust::make_tuple(
				this->outputs().begin(), 
				thrust::counting_iterator<int>(0))),
		    thrust::make_zip_iterator(
			thrust::make_tuple(
				this->outputs().begin()           + timeLength * this->size(), 
				thrust::counting_iterator<int>(0) + timeLength * this->size())),
		    fn);    
	    }
	    }
	    
	    // step3. prepare m_stateRandom -> target->outputs()
	    thrust::copy(m_stateRandom.begin(), m_stateRandom.end(),
			 m_targetLayer->outputs().begin());		

	}else{
	    throw std::runtime_error("Error: interweave layer, mode not implemented");
	}
	
    }

    template <typename TDevice>
    void InterWeaveLayer<TDevice>::computeForwardPass(const int timeStep, const int nnState)
    {
	throw std::runtime_error("Error: random shuffle not support online training/generation");
    }
    
    template <typename TDevice>
    void InterWeaveLayer<TDevice>::computeBackwardPass(const int nnState)
    {
	int timeLength = this->curMaxSeqLength() * this->parallelSequences();
	int numEle     = timeLength * this->precedingLayer().size();
	
	if (m_mode == INTERWEAVELAYER_MODE_121323){
	    {
		internal::InterweaveFeature_121323_grad fn;
		fn.folLayerSize = this->size();
		fn.curLayerSize = this->precedingLayer().size();
		fn.patTypes     = helpers::getRawPointer(this->patTypes());
		fn.inputGrad    = helpers::getRawPointer(this->outputErrors());
	    
		thrust::for_each(
		  thrust::make_zip_iterator(
		     thrust::make_tuple(this->precedingLayer().outputErrors().begin(),
					thrust::counting_iterator<int>(0))),
		  thrust::make_zip_iterator(
		     thrust::make_tuple(this->precedingLayer().outputErrors().begin() + numEle,
					thrust::counting_iterator<int>(0)             + numEle)),
		  fn);
	    }
	}else if (m_mode == INTERWEAVELAYER_MODE_RANDOMPICK){
	    {
		
		internal::InterweaveFeature_random_grad fn;
		fn.folLayerSize = this->size();
		fn.curLayerSize = this->precedingLayer().size();
		fn.patTypes     = helpers::getRawPointer(this->patTypes());
		fn.inputGrad    = helpers::getRawPointer(this->outputErrors());
		fn.outputGrad   = helpers::getRawPointer(this->precedingLayer().outputErrors());
		fn.randomState  = helpers::getRawPointer(this->m_randomState);
		
		thrust::for_each(
		  thrust::make_zip_iterator(
		     thrust::make_tuple(this->precedingLayer().outputErrors().begin(),
					thrust::counting_iterator<int>(0))),
		  thrust::make_zip_iterator(
		     thrust::make_tuple(this->precedingLayer().outputErrors().begin() + numEle,
					thrust::counting_iterator<int>(0)             + numEle)),
		  fn);
		
	    }
	}else{
	    throw std::runtime_error("Error: interweave layer, mode not implemented");
	}
	
    }

    template <typename TDevice>
    void InterWeaveLayer<TDevice>::computeBackwardPass(const int timeStep, const int nnState)
    {
	throw std::runtime_error("Error: random shuffle not support online training");
    }
    
    
    template <typename TDevice>
    void InterWeaveLayer<TDevice>::exportLayer(const helpers::JsonValue &layersArray,
					       const helpers::JsonAllocator &allocator) const
    {
	TrainableLayer<TDevice>::exportLayer(layersArray, allocator);
	(*layersArray)[layersArray->Size() - 1].AddMember("mode", m_mode, allocator);
	(*layersArray)[layersArray->Size() - 1].AddMember("targetLayer",
							  m_targetLayerName.c_str(),
							  allocator);
    }

    template <typename TDevice>
    void InterWeaveLayer<TDevice>::clearAllBuffers()
    {
    }

    template <typename TDevice>
    void InterWeaveLayer<TDevice>::resizeAllBuffers(const int timeLength)
    {
    }    


    template <typename TDevice>
    void InterWeaveLayer<TDevice>::linkTargetLayer(Layer<TDevice> &targetLayer)
    {
	if (targetLayer.name() == m_targetLayerName){
	    // link the target layer
	    m_targetLayer = &targetLayer;
	    if (m_targetLayer->size() != 1)
		throw std::runtime_error("Error: size of target layer of InterWeaveLayer != 1");
	    printf("\n\tMutualWeave link target layer");
	}else{
	    m_targetLayer = NULL;
	}
	
    }    

    
    template class InterWeaveLayer<Cpu>;
    template class InterWeaveLayer<Gpu>;
    
}
