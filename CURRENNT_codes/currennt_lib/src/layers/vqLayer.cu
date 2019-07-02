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

#include "../Configuration.hpp"
#include "vqLayer.hpp"

#include "SkipLayer.hpp" // have to call this function

#include "../helpers/getRawPointer.cuh"
#include "../helpers/Matrix.hpp"
#include "../helpers/min.cuh"
#include "../helpers/max.cuh"
#include "../helpers/safeExp.cuh"
#include "../helpers/JsonClasses.hpp"
#include "../helpers/misFuncs.hpp"
#include "../helpers/NumericLimits.cuh"

#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/for_each.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/fill.h>
#include <thrust/random.h>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#include <boost/foreach.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/algorithm/string.hpp>
#include <vector>

#define NN_VQLAYER_CODE_NOTUSE 0  // not use the code, not update it
#define NN_VQLAYER_CODE_USE    1  // use the code, but not train it
#define NN_VQLAYER_CODE_TRAIN  2  // use and train the code

#define NN_VQINPUT_CODE_VECTOR 0
#define NN_VQINPUT_CODE_PROB   1
#define NN_VQINPUT_CODE_INDEX  2

#define SKIP_MARKER helpers::NumericLimits<real_t>::max()

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

    struct computeDisMatrix
    {
	int featureDim;      // code dimension
	int codeBookSize;    // code book size
	int resoRatio;
	int paralNum;
	
	real_t *inputData;
	real_t *codeData;
	
	const char *patTypes;
	
	__host__ __device__ void operator() (const thrust::tuple<real_t&, int> &t) const
	{
	    int index = t.get<1>();
	    int timeIdx = index / codeBookSize; // which timeStep
	    int codeIdx = index % codeBookSize; // which code
	    int relTime = (timeIdx / paralNum) / resoRatio * paralNum + (timeIdx % paralNum);  
	    
	    real_t distance = 0.0;

	    if (patTypes[timeIdx] == PATTYPE_NONE)
		t.get<0>() = distance;
	    else{
		
		for (int i = 0; i < featureDim; i++){
		    distance += ((codeData[codeIdx * featureDim + i] -
				  inputData[relTime * featureDim + i]) *
				 (codeData[codeIdx * featureDim + i] -
				  inputData[relTime * featureDim + i]));
		}
		t.get<0>() = distance;
	    }   
	}
    };

    struct getBestIndex
    {
	int     howmanyBook;
	int     whichBook;
	int     codeBookSize;
	
	int    *indexBuffer;
	real_t *indexBufferF;
	real_t *disMatrix;
	
	const char *patTypes;

	// for 0 : T 
	__host__ __device__ void operator() (const thrust::tuple<int&, int> &t) const
	{
	    int timeIdx = t.get<1>();
	    int saveIdx = timeIdx * howmanyBook + whichBook;
	    if (patTypes[timeIdx] == PATTYPE_NONE)
		indexBuffer[saveIdx] = -1;
	    else{
		if (codeBookSize == 1)
		    indexBuffer[saveIdx] = -1;
		else{
		    real_t tempMin = disMatrix[timeIdx * codeBookSize];
		    real_t tempId  = 0; 
		    for (int i = 1; i < codeBookSize; i++){
			if (disMatrix[timeIdx * codeBookSize + i] < tempMin){
			    tempMin = disMatrix[timeIdx * codeBookSize + i];
			    tempId  = i;
			}
		    }
		    indexBuffer[saveIdx] = tempId;
		}
	    }
	    indexBufferF[saveIdx] = indexBuffer[saveIdx];
	}
    };


    struct LoadVq
    {
	int codeDim;
	int dimPtr;
	int featureDim;
	int codeBookSize;
	int howmanyBook;
	int whichBook;
	
	real_t *outputData;
	real_t *codeData;
	real_t *index;
	int *codeDimFlag;
	
	const char *patTypes;

	// for 0 : T * featureDim
	__host__ __device__ void operator() (const thrust::tuple<real_t&, int> &t) const
	{
	    int timeIdx = t.get<1>() / codeDim;
	    int featIdx = t.get<1>() % codeDim;
	    
	    if (patTypes[timeIdx] == PATTYPE_NONE ||
		codeDimFlag[dimPtr + featIdx] == NN_VQLAYER_CODE_NOTUSE){
		outputData[timeIdx * featureDim + dimPtr + featIdx] = 0.0;
	    }else{
		int tmpIndex = (int)index[timeIdx * howmanyBook + whichBook];
		if (tmpIndex >= 0){
		    outputData[timeIdx * featureDim + dimPtr + featIdx] =
			codeData[tmpIndex * codeDim + featIdx];
		}else{
		    outputData[timeIdx * featureDim + dimPtr + featIdx] = 0.0;
		}
	    }
	}
    };
    

    struct vqLoadAvarageCode
    {
	int codeDim;
	int dimPtr;
	
	int featureDim;

	int bestNcode;
	
	int howmanyBook;
	int whichBook;
	
	real_t *outputData;
	real_t *codeData;
	real_t *index;
	int    *codeDimFlag;
	
	const char *patTypes;

	// for 0 : T * featureDim
	__host__ __device__ void operator() (const thrust::tuple<real_t&, int> &t) const
	{
	    int timeIdx = t.get<1>() / codeDim;
	    int featIdx = t.get<1>() % codeDim;
	    
	    if (patTypes[timeIdx] == PATTYPE_NONE ||
		codeDimFlag[dimPtr + featIdx] == NN_VQLAYER_CODE_NOTUSE){
		outputData[timeIdx * featureDim + dimPtr + featIdx] = 0.0;
	    }else{
		int tmpIndex = 0;
		outputData[timeIdx * featureDim + dimPtr + featIdx] = 0.0;
		
		for (int idx = 0 ; idx < bestNcode; idx++){
		    tmpIndex = (int)index[(timeIdx * howmanyBook + whichBook) * bestNcode + idx];
		    outputData[timeIdx * featureDim + dimPtr + featIdx] +=
			codeData[tmpIndex * codeDim + featIdx] / (real_t)bestNcode;
		}
		
	    }
	}
    };
    
    
    struct WeightedSumVq
    {
	int codeDim;
	int dimPtr;
	int featureDim;
	int codeBookSize;
	int codeBookSizeA;
	int howmanyBook;
	int whichBookDim;
	
	real_t *outputData;
	real_t *codeData;
	real_t *weights;
	
	const char *patTypes;

	// for 0 : T * featureDim
	__host__ __device__ void operator() (const thrust::tuple<real_t&, int> &t) const
	{
	    int timeIdx = t.get<1>() / codeDim;
	    int featIdx = t.get<1>() % codeDim;
	    
	    if (patTypes[timeIdx] == PATTYPE_NONE){
		outputData[timeIdx * featureDim + dimPtr + featIdx] = 0.0;
	    }else{
		real_t tmpFeat = 0.0;
		for (int i=0; i < codeBookSize; i++){
		    tmpFeat += (weights[timeIdx * codeBookSizeA + whichBookDim + i] *
				codeData[i * codeDim + featIdx]);
		}
		outputData[timeIdx * featureDim + dimPtr + featIdx] = tmpFeat;
	    }
	}
    };


    
    struct CodeDiff
    {
	int howmanyBook;
	int whichBook;
	int resoRatio;
	int paralNum;
	
	int     featureDim;
	int     codeBookSize;	
	real_t *codeData;
	real_t *inputData;
	int    *index;
	
	const char *patTypes;

	// for 0 : T * featureDim
	__host__ __device__ real_t operator() (const thrust::tuple<const real_t&, int> &t) const
	{
	    int timeIdx = t.get<1>() / featureDim;
	    int relTime = (timeIdx / paralNum) / resoRatio * paralNum + (timeIdx % paralNum);
	    
	    int featIdx = t.get<1>() % featureDim;
	    int codeIdx = timeIdx * howmanyBook + whichBook;
	    if (patTypes[timeIdx] == PATTYPE_NONE)
		return 0.0;
	    else if (index[codeIdx] >= 0)
		return ((inputData[relTime * featureDim + featIdx] -
			 codeData[index[codeIdx] * featureDim + featIdx]) *
			(inputData[relTime * featureDim + featIdx] -
			 codeData[index[codeIdx] * featureDim + featIdx]));
	    else
		return 0.0;
	}
    };


    struct vqCodeDiffEMA
    {
	int     encoderLayerSize;
	int     thisLayerSize;
	int     dimPtr;
	
	real_t *outputData;
	real_t *inputData;
	
	const char *patTypes;

	// for 0 : T * featureDim
	__host__ __device__ real_t operator() (const thrust::tuple<const real_t&, int> &t) const
	{
	    
	    int timeIdx = t.get<1>() / encoderLayerSize;
	    int dimIdx  = t.get<1>() % encoderLayerSize;
	    
	    
	    if (patTypes[timeIdx] == PATTYPE_NONE){
		return 0.0;
	    }else{
		return ((inputData[timeIdx * encoderLayerSize + dimIdx] -
			 outputData[timeIdx * thisLayerSize + dimPtr + dimIdx]) *
			(inputData[timeIdx * encoderLayerSize + dimIdx] -
			 outputData[timeIdx * thisLayerSize + dimPtr + dimIdx]));
	    }
	}
    };


    
    struct CopyGradient
    {
	int preLayerDim;
	int thisLayerDim;
	int paralNum;
	int dimPtr;
	int resoRatio;
	int *codeDimFlag;
	
	real_t *gradientInput;
	const char *patTypes;
	__host__ __device__ void operator() (const thrust::tuple<real_t&, int> &t) const
	{
	    int timeIdx = t.get<1>() / preLayerDim;
	    int featIdx = t.get<1>() % preLayerDim;
	    int timeBlock = timeIdx / paralNum;
	    int blockIdx  = timeIdx % paralNum;
	    
	    if (patTypes[timeIdx] == PATTYPE_NONE ||
		codeDimFlag[dimPtr + featIdx] == NN_VQLAYER_CODE_NOTUSE ||
		codeDimFlag[dimPtr + featIdx] == NN_VQLAYER_CODE_USE){
		t.get<0>() = 0.0;
		return;
	    }else{
		t.get<0>() = 0.0;
		int gradIdx = 0;
		for (int i = 0; i < resoRatio; i++){
		    gradIdx = (timeBlock * resoRatio + i) * paralNum + blockIdx;
		    t.get<0>() += gradientInput[gradIdx * thisLayerDim + dimPtr + featIdx];
		}
	    }
	    
	}
    };


    struct CopyGradientBeta
    {
	real_t beta;
	int    inputLayerDim;
	int    thisLayerDim;
	int    dimPtr;
	
	int        *codeDimFlag;
	real_t     *vqOutput;
	const char *patTypes;
	
	__host__ __device__ void operator() (const thrust::tuple<real_t&, real_t&, int> &t) const
	{

	    int timeStep = t.get<2>() / inputLayerDim;
	    int dimIdx   = t.get<2>() % inputLayerDim;
	    

	    if (patTypes[timeStep] == PATTYPE_NONE ||
		codeDimFlag[dimPtr + dimIdx] == NN_VQLAYER_CODE_NOTUSE ||
		codeDimFlag[dimPtr + dimIdx] == NN_VQLAYER_CODE_USE)
	    {
		// do nothing
	    }else{
		t.get<0>() += beta * (t.get<1>() -
				      vqOutput[timeStep * thisLayerDim + dimPtr + dimIdx]);
	    }
	}
    };

    
    struct GradientForCodeBook
    {
	int codeDim;
	int codeBookSize;
	int howmanyBook;
	int whichBook;
	int resoRatio;
	int paralNum;
	int dimPtr;
	
	real_t  beta;

	int    *codeDimFlag;
	real_t *inputData;
	real_t *codeData;
	real_t *preLayerGrad;

	
	int *index;

	int timeLength;
	const char *patTypes;

	// for codeBookSize * featureDim
	__host__ __device__ void operator() (const thrust::tuple<real_t&, int> &t) const
	{

	    int codeIdx = t.get<1>() / codeDim;
	    int featIdx = t.get<1>() % codeDim;

	    int timeIdx = 0;
	    
	    real_t sumInput = 0.0;
	    real_t cnt = 0.0;
	    
	    for (int i = 0; i<timeLength; i++){
		if (patTypes[i] == PATTYPE_NONE)
		    break;
		if (index[i * howmanyBook + whichBook] == codeIdx){
		    cnt += 1.0;
		    
		    // Methods1/2: Calculate the moving average of the input latent codes
		    timeIdx = (i / paralNum) / resoRatio * paralNum + (i % paralNum);
		    sumInput += (inputData[timeIdx * codeDim + featIdx] - sumInput)/cnt;
		    
		    // Propagate to the previous layer
		    // There is no conflict in writing since for any time t, there is only
		    // one codeIdx that can reach this step
		    if (codeDimFlag[dimPtr + featIdx] == NN_VQLAYER_CODE_TRAIN)
			preLayerGrad[timeIdx * codeDim + featIdx] +=
			    beta * (inputData[timeIdx *codeDim+featIdx] - codeData[t.get<1>()]);
		}
	    }
	    // Method1: average the gradients over time
	    // t.get<0>() = codeData[t.get<1>()] - sumInput;
	    
	    // Method2: sum of the gradients over time
	    if (codeDimFlag[dimPtr + featIdx] == NN_VQLAYER_CODE_TRAIN)
		t.get<0>() = (codeData[t.get<1>()] - sumInput) * cnt;
	    else
		t.get<0>() = 0.0;
	}
    };

    struct GradientForCodeBookEMA
    {
	int codeDim;
	int codeBookSize;
	
	int howmanyBook;
	int whichBook;

	int dimPtr;

	int bestNcode;
	
	int    *codeDimFlag;
	real_t *inputData;
	real_t  lambdaPara;
	
	int *index;

	int timeLength;
	const char *patTypes;

	// for codeBookSize * featureDim
	__host__ __device__ void operator() (const thrust::tuple<real_t&, int> &t) const
	{

	    int codeIdx = t.get<1>() / codeDim;
	    int featIdx = t.get<1>() % codeDim;

	    int timeIdx = 0;
	    
	    real_t sumInput = 0.0;
	    real_t cnt = 0.0;
	    
	    for (int i = 0; i<timeLength; i++){
		
		if (patTypes[i] == PATTYPE_NONE)
		    break;

		for (int j = 0; j < bestNcode; j++){
		    if (index[(i * howmanyBook + whichBook) * bestNcode + j] == codeIdx){
			cnt += 1.0;
			timeIdx = i;
			sumInput += (inputData[timeIdx * codeDim + featIdx] - sumInput)/cnt;
		    
		    }
		}
	    }
	    
	    // Method2: sum of the gradients over time
	    if (codeDimFlag[dimPtr + featIdx] == NN_VQLAYER_CODE_TRAIN)
		t.get<0>() = t.get<0>() * lambdaPara + (1.0 - lambdaPara) * sumInput;
	    else
		t.get<0>() = 0.0;
	}
    };

    struct vqCalculateOffsetFn
    {
        int codeBooksize;
	
        const real_t *dismatrix;
        const char *patTypes;

        __host__ __device__ real_t operator() (const int &patIdx) const
        {
            if (patTypes[patIdx] == PATTYPE_NONE)
                return SKIP_MARKER;

            // search for the min and max output
            real_t max = helpers::NumericLimits<real_t>::min();
            real_t min = helpers::NumericLimits<real_t>::max();

            const real_t *offOutputs = &dismatrix[patIdx * codeBooksize];

            for (int i = 0; i < codeBooksize; ++i) {
                real_t x = offOutputs[i];
                min = helpers::min(min, x);
                max = helpers::max(max, x);
            }

            // calculate the offset
            real_t offset = (real_t)0.5 * (min + max);

            return offset;
        }
    };


    struct vqCalculateExpFn
    {
        int codeBooksize;

        const real_t *offsets;

        __host__ __device__ void operator() (const thrust::tuple<real_t&, int> &t) const
        {
            // unpack the tuple
            real_t output = t.get<0>();
            int outputIdx = t.get<1>();

            // calculate the pattern index
            int patIdx = outputIdx / codeBooksize;

            // check if we can stop the calculation
            real_t offset = offsets[patIdx];
            if (offset == SKIP_MARKER)
                return;

            // calculate the exponent
            real_t x = helpers::safeExp(output - offset);

            // store the result
            t.get<0>() = x;
        }
    };

    
    struct vqSumUpOutputsFn
    {
        int codeBooksize;

        const real_t *disMatrix;

        __host__ __device__ void operator() (const thrust::tuple<real_t&, int> &t) const
        {
            // unpack the tuple
            int patIdx = t.get<1>();

            // check if the pattern belongs to a sequence
            if (t.get<0>() == SKIP_MARKER)
                return;

            // sum up the outputs
            const real_t *offOutputs = &disMatrix[patIdx * codeBooksize];

            real_t sum = 0;
            for (int i = 0; i < codeBooksize; ++i)
                sum += offOutputs[i];

            // store the result
            t.get<0>() = sum;
        }
    };

    struct vqNormalizeOutputsFn
    {
        int codeBooksize;

        const real_t *normFacts;

        __host__ __device__ void operator() (const thrust::tuple<real_t&, int> &t) const
        {
            // unpack the tuple
            int outputIdx = t.get<1>();

            // calculate the pattern index
            int patIdx = outputIdx / codeBooksize;

            // check if we can stop the calculation
            real_t normFact = normFacts[patIdx];
            if (normFact == SKIP_MARKER)
                return;

            // calculate the normalized value
            real_t x = t.get<0>() / normFact;

            // store the result
            t.get<0>() = x;
        }
    };


    struct vqSamplingSoftmax
    {
        int codeBooksize;
	int codeNbest;
	int howmanyBook;
	int whichBook;
	
	real_t       *randomSeeds;
	real_t       *indexBufferF;
	int          *indexBuffer;
	
	const real_t *probBuff;    // softmax probability
	const char   *patTypes;

	
        __host__ __device__ void operator() (const thrust::tuple<real_t&, int> &t) const
        {
	    int timeIdx   = t.get<1>() / codeNbest;
	    int searchIdx = t.get<1>() % codeNbest;
		
	    if (patTypes[timeIdx] == PATTYPE_NONE)
		return;
	    
	    real_t probAccum = 0.0;
	    real_t randomNum = randomSeeds[t.get<1>()];

	    int    ptr = (timeIdx * howmanyBook + whichBook) * codeNbest + searchIdx;
	    
	    // 
	    for (int i = 0; i < codeBooksize; i++){
		probAccum += probBuff[timeIdx * codeBooksize + i];
		if (randomNum < probAccum){
		    indexBufferF[ptr] = (real_t)i;
		    indexBuffer[ptr]  = i;
		    break;
		}
	    }
        }
    };

    
    
}
}


namespace layers{

    // Construct the layer
    template <typename TDevice>
    vqLayer<TDevice>::vqLayer(const helpers::JsonValue &layerChild,
			      const helpers::JsonValue &weightsSection,
			      Layer<TDevice> &precedingLayer,
			      int             maxSeqLength,
			      int             layerID)
	: TrainableLayer<TDevice>(layerChild, weightsSection, 0,
				  (layerChild->HasMember("vqCodeBookSize") ? 
				   ((*layerChild)["vqCodeBookSize"].GetInt()) : 0),
				  precedingLayer,
				  maxSeqLength,
				  layerID)
	, m_vqCodeBookSizeAll ((layerChild->HasMember("vqCodeBookSize") ? 
				((*layerChild)["vqCodeBookSize"].GetInt()) : 0))
    {

	// load options
	this->__loadOpts(layerChild);
	
	// link to the output layers of encoder
	this->__setUp_encoder_inputs();

	// allocate memory
	this->__setUp_allocateMem();

	// show information
	this->__showOpts();
	
	// done
    }	

    // Destructor
    template <typename TDevice>
    vqLayer<TDevice>::~vqLayer()
    {
    }


    template <typename TDevice>
    void vqLayer<TDevice>::__loadOpts(const helpers::JsonValue &layerChild)
    {

	// Size of each codebook (string)
	m_vqCodeBookSizeStr = (layerChild->HasMember("vqEachCodeBookSize") ? 
			       ((*layerChild)["vqEachCodeBookSize"].GetString()) : "");

	// Outpu layers from encoders
	m_preLayerNamesStr  = (layerChild->HasMember("inputLayers") ? 
			       ((*layerChild)["inputLayers"].GetString()) : "");
	
	// Status of each codebook
	m_codeBookStatusStr = (layerChild->HasMember("codeBookStatus") ? 
			       ((*layerChild)["codeBookStatus"].GetString()) : "");

	// 1-best code or N-best code?
	m_bestNcode         = ((layerChild->HasMember("nbest") ? 
				((*layerChild)["nbest"].GetInt()) : 1));

	// weight for commitment loss
	m_betaPara     = (layerChild->HasMember("beta") ? 
			  ((*layerChild)["beta"].GetDouble()) : 0.25);

	// parameter for exponential moving average update of codes
	m_lambdaPara   = (layerChild->HasMember("lambda") ? 
			  ((*layerChild)["lambda"].GetDouble()) : 0.99);


	if ((!this->flagTrainingMode()) && m_bestNcode > 1){
	    printf("\t\nGeneration uses 1-best VQ code");
	    m_bestNcode = 1;
	}
	// done
	
    }

    template <typename TDevice>
    void vqLayer<TDevice>::__setUp_encoder_inputs()
    {
	// Parse the preLayer option
	
	if (m_preLayerNamesStr.size()){
	    
	    std::vector<std::string> tmpOpt;
	    misFuncs::ParseStrOpt(m_preLayerNamesStr, tmpOpt, ",");

	    // find the encoder's output layer
	    for (int cnt = 0 ; cnt < tmpOpt.size(); cnt++) {
		layers::Layer<TDevice>          *tmpLayerPtr = &this->precedingLayer();
		layers::TrainableLayer<TDevice> *tmpTLayerPtr;
		layers::SkipLayer<TDevice>      *tmpSLayerPtr;

		// Find the input layer for each code book
		while (tmpLayerPtr != NULL){
		    tmpTLayerPtr = dynamic_cast<layers::TrainableLayer<TDevice>*>(tmpLayerPtr);
		    
		    if (tmpLayerPtr->name() == tmpOpt[cnt]){
			m_preLayers.push_back(tmpTLayerPtr);

			// check if previous layers are skip
			tmpSLayerPtr = dynamic_cast<layers::SkipLayer<TDevice>*>(tmpLayerPtr);
			
			if (tmpSLayerPtr == NULL &&
			    tmpLayerPtr->name() != this->precedingLayer().name()){
			    printf("Please add skipini after %s", tmpLayerPtr->name().c_str());
			    throw std::runtime_error("Invalid network structure");
			}
			break;
		    }
		    
		    if (tmpTLayerPtr)
			tmpLayerPtr = &tmpTLayerPtr->precedingLayer();
		    else{
			// until the input layer
			printf("Cannot find previous layer %s for %s",
			       tmpOpt[cnt].c_str(), this->name().c_str());
			throw std::runtime_error("Please check network.jsn");
		    }
		}
	    }

	    // check
	    if (m_vqCodeBookSizeStr.size() == 0){
		throw std::runtime_error("vqEachCodeBookSize is not specified");
	    }else{
		misFuncs::ParseIntOpt(m_vqCodeBookSizeStr, m_vqCodeBookSizeVec);
		if (m_vqCodeBookSizeVec.size() != m_preLayers.size())
		    throw std::runtime_error("vqEachCodeBookSize & inputLayers unequal length");
		if (misFuncs::SumCpuIntVec(m_vqCodeBookSizeVec) != m_vqCodeBookSizeAll)
		    throw std::runtime_error("sum of vqEachCodeBookSize unqual vqCodeBookSize");
	    }
	    
	}else{
	    layers::TrainableLayer<TDevice> *tmpTLayerPtr =
		dynamic_cast<layers::TrainableLayer<TDevice>*>(&this->precedingLayer());
	    m_preLayers.push_back(tmpTLayerPtr);
	    m_vqCodeBookSizeVec.push_back(m_vqCodeBookSizeAll);
	    m_vqCodeBookSizeStr = "";
	}

	// 
	cpu_int_vector tmpBookStatus;
	if (m_codeBookStatusStr.size()){
	    misFuncs::ParseIntOpt(m_codeBookStatusStr, tmpBookStatus);
	    if (tmpBookStatus.size() != this->m_vqCodeBookSizeVec.size())
		throw std::runtime_error("codeBookStatus != vqEachCodeBookSize in length");
	    printf("\n\t code book status: %s", m_codeBookStatusStr.c_str());
	}else{
	    tmpBookStatus.clear();
	}
	cpu_int_vector tmpCodeDimStatus(this->size(), NN_VQLAYER_CODE_TRAIN);

	
	printf("\n\t Receive input from:");
	int tmpVecSum = 0;
	int tmpCodeBookCnt = 0;
	int validWeightNum = 0;
	int layerCnt = 0;
	BOOST_FOREACH (layers::TrainableLayer<TDevice> *layer, m_preLayers) {
	    printf("%s ", layer->name().c_str());

	    if (tmpVecSum > this->size() || (tmpVecSum + layer->size()) > this->size())
		throw std::runtime_error("vqLayer layer size != sum of input layer(s)");

	    validWeightNum += layer->size() * m_vqCodeBookSizeVec[layerCnt];
	    
	    // check the code status
	    if (tmpBookStatus.size() > 0){
		if (tmpBookStatus[tmpCodeBookCnt] == NN_VQLAYER_CODE_NOTUSE ||
		    tmpBookStatus[tmpCodeBookCnt] == NN_VQLAYER_CODE_USE    ||
		    tmpBookStatus[tmpCodeBookCnt] == NN_VQLAYER_CODE_TRAIN )
		    thrust::fill(tmpCodeDimStatus.begin() + tmpVecSum,
				 tmpCodeDimStatus.begin() + tmpVecSum + layer->size(),
				 tmpBookStatus[tmpCodeBookCnt]);
		else
		    throw std::runtime_error("Unknown config for codeBookStatus");
	    }

	    // accumulat the dimension of code from each book
	    tmpVecSum += layer->size();
	    tmpCodeBookCnt++;
	    
	    // resolution check
	    if (layer->getResolution() < this->getResolution()){
		printf("%s's resolution %d is smaller than vqLayer",
		       layer->name().c_str(), layer->getResolution());
		throw std::runtime_error("Please check resolution of vqLayer");
	    }
	    if (layer->getResolution() % this->getResolution() > 0){
		printf("%s's resolution %d cannot be divided by vqLayer's resolution",
		       layer->name().c_str(), layer->getResolution());
		throw std::runtime_error("Please check resolution of vqLayer");		
	    }

	    if (layer->getResolution() != this->getResolution() && m_bestNcode > 1)
		throw std::runtime_error("N-best vq cannot work for resolution changed vqLayer");
	    
	    layerCnt++;
	}

	m_codeDimStatus = tmpCodeDimStatus;
	
	// Initial check
	if (tmpVecSum != this->size())
	    throw std::runtime_error("vqLayer layer size != sum of input layer(s)");
	
	if (m_vqCodeBookSizeAll < 1)
	    throw std::runtime_error("vqLayer vqCodeBookSize is not an positive integer");
	
	// Set the gradients of dummy weights to zero
	real_vector tmpWeights = this->weights();
	this->reInitWeight();
	thrust::copy(tmpWeights.begin(), tmpWeights.begin() + validWeightNum,
		     this->weights().begin());
	thrust::fill(this->_weightUpdates().begin() + validWeightNum,
		     this->_weightUpdates().end(), 0.0);

    }

    template <typename TDevice>
    void vqLayer<TDevice>::__setUp_allocateMem()
    {
	// Initialize the distance matrix
	int maxSeqLengthParallel = this->outputs().size() / this->size();
	cpu_real_vector temp(maxSeqLengthParallel * m_vqCodeBookSizeAll, 0.0);
	m_disMatrix = temp;
	m_softmaxRand = temp;
	
	cpu_int_vector  temp2(maxSeqLengthParallel * m_vqCodeBookSizeVec.size() * m_bestNcode, 0);
	m_selectedIdx  = temp2;

	cpu_real_vector temp3(maxSeqLengthParallel * m_vqCodeBookSizeVec.size() * m_bestNcode, 0);
	m_selectedIdxF = temp3;

	cpu_real_vector temp4(maxSeqLengthParallel, 0);
	m_softmaxTmp   = temp4;	
	
    }

    template <typename TDevice>
    void vqLayer<TDevice>::__showOpts()
    {
	if (m_bestNcode >= 1)
	    printf("\n\tVQ Layer uses %d best codes", m_bestNcode);
	else
	    throw std::runtime_error("\n\tVQ Layer has no latent code");
    }
    
    // Load sequences
    template <typename TDevice>
    void vqLayer<TDevice>::loadSequences(const data_sets::DataSetFraction &fraction,
					 const int nnState)
    {
	// Load sequences through the conventional TrainableLayer interface
	TrainableLayer<TDevice>::loadSequences(fraction, nnState);

	// If vqcodes are provided externally
	if (Configuration::instance().vaeCodeInputDir().size()){
	
	    if (fraction.numSequences() > 1)
		throw std::runtime_error("Please turn off parallel mode");

	    // Load the raw data from IO
	    cpu_real_vector codeData;
	    std::vector<std::string> tmpVaeDirs;
	    std::string fileName;
	    int numEle = -1;
	    misFuncs::ParseStrOpt(Configuration::instance().vaeCodeInputDir(), tmpVaeDirs, ",");
	    if (tmpVaeDirs.size() == 1){
		// In case there is only one directory
		fileName = Configuration::instance().vaeCodeInputDir() + "/" +
		    fraction.seqInfo(0).seqTag + ".bin";
		numEle = misFuncs::ReadRealData(fileName, codeData);
	    }else if ((tmpVaeDirs.size() > 1) & (tmpVaeDirs.size() % 2 == 0)){
		// In case there are multiple directories (for multiple vqLayers)
		for (int cnt = 0; cnt < (tmpVaeDirs.size() / 2); cnt++){
		    if (tmpVaeDirs[2*cnt] == this->name()){
			fileName = tmpVaeDirs[2*cnt+1]+"/"+fraction.seqInfo(0).seqTag+".bin";
			numEle = misFuncs::ReadRealData(fileName, codeData);
			break;
		    }
		}
		if (fileName.size() < 1){
		    printf("Cannot find the code for layer %s", this->name().c_str());
		    throw std::runtime_error("Error");
		}
	    }else{
		throw std::runtime_error("vaeCodeInputDir cannot be parsed");
	    }
	    

	    // Load code data
	    if (numEle == this->curMaxSeqLength() * m_vqCodeBookSizeVec.size()){
		
		// Input is the code index (single dimension)
		
		thrust::copy(codeData.begin(), codeData.begin() + numEle,
			     this->m_selectedIdxF.begin());
		
		int uttLength = this->curMaxSeqLength() * this->parallelSequences();
		
		int cnt = 0;
		int codeBookPtr = 0;
		int dimPtr = 0;
		BOOST_FOREACH (layers::TrainableLayer<TDevice> *layer, m_preLayers){
		    // Load the code vector
		    {{
			internal::LoadVq fn3;
			fn3.codeDim      = layer->size();
			fn3.dimPtr       = dimPtr;
			fn3.featureDim   = this->size();
			fn3.codeBookSize = this->m_vqCodeBookSizeVec[cnt];
			fn3.codeData     = helpers::getRawPointer(this->weights()) + codeBookPtr;
			fn3.index        = helpers::getRawPointer(this->m_selectedIdxF);
			fn3.patTypes     = helpers::getRawPointer(this->patTypes());
			fn3.outputData   = helpers::getRawPointer(this->outputs());
			fn3.codeDimFlag  = helpers::getRawPointer(this->m_codeDimStatus);
			
			fn3.howmanyBook  = this->m_vqCodeBookSizeVec.size();
			fn3.whichBook    = cnt;
	    
			int n = uttLength * layer->size();
			thrust::for_each(
			       thrust::make_zip_iterator(
				  thrust::make_tuple(this->outputs().begin(),
						     thrust::counting_iterator<int>(0))),
			       thrust::make_zip_iterator(
				  thrust::make_tuple(this->outputs().begin()           + n,
						     thrust::counting_iterator<int>(0) + n)),
			       fn3);  
		     }}
	    
		    // Move the pointer to the next code book
		    codeBookPtr += layer->size() * this->m_vqCodeBookSizeVec[cnt];
		    dimPtr      += layer->size();
		    cnt += 1;
		}
		
	    }else if ((numEle % this->size() == 0        &&
		       numEle == this->curMaxSeqLength() * this->size()) ||
		      (numEle % m_vqCodeBookSizeAll == 0 &&
		       numEle == this->curMaxSeqLength() * m_vqCodeBookSizeAll)){
		
		// Load prob or code vector
		int tempLoadMethod = -1;
		
		if (this->size() == m_vqCodeBookSizeAll){
		    // Need to check the additional option vaePutVQIndex
		    if (Configuration::instance().vaePutVQIndex() == NN_VQINPUT_CODE_INDEX){
			printf("\nCode vector dimension is equal to the code book size.\n");
			printf("\nPlease set vaeInputCodeType = 1 (code prob) or 0 (code vector)");
			throw std::runtime_error("Error in configuration");
		    }else if (Configuration::instance().vaePutVQIndex() == NN_VQINPUT_CODE_PROB){
			tempLoadMethod = NN_VQINPUT_CODE_PROB;
		    }else if (Configuration::instance().vaePutVQIndex() == NN_VQINPUT_CODE_VECTOR){
			tempLoadMethod = NN_VQINPUT_CODE_VECTOR;
		    }else{
			printf("\nPlease set vaeInputCodeType = 1 (code prob) or 0 (code vector)");
			throw std::runtime_error("Error in configuration");			
		    }
		}else if (numEle == this->curMaxSeqLength() * this->size()){
		    tempLoadMethod = NN_VQINPUT_CODE_VECTOR;
		}else if (numEle == this->curMaxSeqLength() * m_vqCodeBookSizeAll){
		    tempLoadMethod = NN_VQINPUT_CODE_PROB;
		}else{
		    throw std::runtime_error("Impossible Error");
		}

		if (tempLoadMethod == NN_VQINPUT_CODE_VECTOR){
		    // Input is the float code data
		    thrust::copy(codeData.begin(), codeData.begin() + numEle,
				 this->outputs().begin());

		}else if (tempLoadMethod == NN_VQINPUT_CODE_PROB){
		    // Input is the probability of each code
		
		    // use m_disMatrix to store the code probability 
		    thrust::copy(codeData.begin(), codeData.begin() + numEle,
				 this->m_disMatrix.begin());

		    int uttLength = this->curMaxSeqLength() * this->parallelSequences();
		    
		    int cnt = 0;
		    int codeDimPtr  = 0;
		    int codeBookPtr = 0;
		    int dimPtr = 0;
		    BOOST_FOREACH (layers::TrainableLayer<TDevice> *layer, m_preLayers){
			// step3. merge the vector as the output
			{{
			internal::WeightedSumVq fn3;
			fn3.codeDim      = layer->size();
			fn3.codeBookSize = this->m_vqCodeBookSizeVec[cnt];
			fn3.codeBookSizeA= this->m_vqCodeBookSizeAll;
			fn3.codeData     = helpers::getRawPointer(this->weights()) + codeBookPtr;
			fn3.howmanyBook  = this->m_vqCodeBookSizeVec.size();
			fn3.whichBookDim = codeDimPtr;
	    
			fn3.dimPtr       = dimPtr;
			fn3.featureDim   = this->size();
			
			fn3.weights      = helpers::getRawPointer(this->m_disMatrix);
			
			fn3.patTypes     = helpers::getRawPointer(this->patTypes());
			fn3.outputData   = helpers::getRawPointer(this->outputs());
			
			int n = uttLength * layer->size();
			thrust::for_each(
			       thrust::make_zip_iterator(
				  thrust::make_tuple(this->outputs().begin(),
						     thrust::counting_iterator<int>(0))),
			       thrust::make_zip_iterator(
				  thrust::make_tuple(this->outputs().begin()           + n,
						     thrust::counting_iterator<int>(0) + n)),
			       fn3);  
			}}
			
			// Move the pointer to the next code book
			codeBookPtr += layer->size() * this->m_vqCodeBookSizeVec[cnt];
			dimPtr      += layer->size();
			codeDimPtr  += this->m_vqCodeBookSizeVec[cnt];
			cnt += 1;
		    }
		}else{
		    throw std::runtime_error("Impossible Error");
		}
		
	    }else{
		throw std::runtime_error("Input code, either index or raw data, is incompatible.");
	    }
	}
    }

    // NN forward
    template <typename TDevice>
    void vqLayer<TDevice>::computeForwardPass(const int nnState)
    {
	// If the code has been loaded, no need to propagate again
	if (Configuration::instance().vaeCodeInputDir().size())
	    return;

	// Length of the input data sequence
	int uttLength = this->curMaxSeqLength() * this->parallelSequences();
	
	m_codeError     = 0.0;
	int cnt         = 0;
	int codeBookPtr = 0;
	int dimPtr      = 0;

	// For each input layer from the encoder
	BOOST_FOREACH (layers::TrainableLayer<TDevice> *layer, m_preLayers){

	    // Step1. compute the distance matrix
	    {{
		internal::computeDisMatrix fn1;
		fn1.featureDim   = layer->size();
		fn1.codeBookSize = this->m_vqCodeBookSizeVec[cnt];
		fn1.inputData    = helpers::getRawPointer(layer->outputs());
		fn1.codeData     = helpers::getRawPointer(this->weights()) + codeBookPtr;
		fn1.patTypes     = helpers::getRawPointer(this->patTypes());
		fn1.resoRatio    = layer->getResolution() / this->getResolution();
		fn1.paralNum     = this->parallelSequences();
		
		int n = uttLength * this->m_vqCodeBookSizeVec[cnt];
		thrust::for_each(
                 thrust::make_zip_iterator(
		  thrust::make_tuple(this->m_disMatrix.begin(),
				     thrust::counting_iterator<int>(0))),
		 thrust::make_zip_iterator(
		  thrust::make_tuple(this->m_disMatrix.begin()         + n,
				     thrust::counting_iterator<int>(0) + n)),
		 fn1);
	    }}

	    // 
	    if (m_bestNcode == 1){
		
		// If only one-best from VQ code book
		{{
		    // step2. search for the best index
		    internal::getBestIndex fn2;
		    fn2.codeBookSize = this->m_vqCodeBookSizeVec[cnt];
		    fn2.disMatrix    = helpers::getRawPointer(this->m_disMatrix);
		    fn2.patTypes     = helpers::getRawPointer(this->patTypes());
		    fn2.indexBuffer  = helpers::getRawPointer(this->m_selectedIdx);
		    fn2.indexBufferF = helpers::getRawPointer(this->m_selectedIdxF);
		    fn2.howmanyBook  = this->m_vqCodeBookSizeVec.size();
		    fn2.whichBook    = cnt;
		
		    thrust::for_each(
                     thrust::make_zip_iterator(
			thrust::make_tuple(this->m_selectedIdx.begin(),
					   thrust::counting_iterator<int>(0))),
		     thrust::make_zip_iterator(
		        thrust::make_tuple(this->m_selectedIdx.begin() + uttLength,
					   thrust::counting_iterator<int>(0) + uttLength)),
		     fn2);	
		}}
		
		{{
		    // step3. optional, calculate the error
		    internal::CodeDiff fn4;
		    fn4.featureDim   = layer->size();
		    fn4.codeBookSize = this->m_vqCodeBookSizeVec[cnt];
		    fn4.codeData     = helpers::getRawPointer(this->weights()) + codeBookPtr;
		    fn4.index        = helpers::getRawPointer(this->m_selectedIdx);
		    fn4.patTypes     = helpers::getRawPointer(this->patTypes());
		    fn4.howmanyBook  = this->m_vqCodeBookSizeVec.size();
		    fn4.whichBook    = cnt;
		    fn4.resoRatio    = layer->getResolution() / this->getResolution();
		    fn4.inputData    = helpers::getRawPointer(layer->outputs());
		    fn4.paralNum     = this->parallelSequences();
	    
		    int n = uttLength * layer->size();
		    real_t tmpError = thrust::transform_reduce(
                      thrust::make_zip_iterator(
			thrust::make_tuple(this->outputs().begin(),
					   thrust::counting_iterator<int>(0))),
		      thrust::make_zip_iterator(
		        thrust::make_tuple(this->outputs().begin()           + n,
					   thrust::counting_iterator<int>(0) + n)),
		      fn4, (real_t)0.0, thrust::plus<real_t>());
		    if (m_codeDimStatus[dimPtr] != NN_VQLAYER_CODE_NOTUSE)
			m_codeError += (tmpError / uttLength);
		}}

	    
		// step4. use the best vector as the output
		{{
		    internal::LoadVq fn3;
		    fn3.codeDim      = layer->size();
		    fn3.dimPtr       = dimPtr;
		    fn3.featureDim   = this->size();
		    fn3.codeBookSize = this->m_vqCodeBookSizeVec[cnt];
		    fn3.codeData     = helpers::getRawPointer(this->weights()) + codeBookPtr;
		    fn3.index        = helpers::getRawPointer(this->m_selectedIdxF);
		    fn3.patTypes     = helpers::getRawPointer(this->patTypes());
		    fn3.outputData   = helpers::getRawPointer(this->outputs());
		    fn3.codeDimFlag  = helpers::getRawPointer(this->m_codeDimStatus);
	    
		    fn3.howmanyBook  = this->m_vqCodeBookSizeVec.size();
		    fn3.whichBook    = cnt;
	    
		    int n = uttLength * layer->size();
		    thrust::for_each(
                      thrust::make_zip_iterator(
			thrust::make_tuple(this->outputs().begin(),
					   thrust::counting_iterator<int>(0))),
		      thrust::make_zip_iterator(
		        thrust::make_tuple(this->outputs().begin()           + n,
					   thrust::counting_iterator<int>(0) + n)),
		      fn3);
		    
		}}
		
	    }else{
		
		// If bestNcode > 1
		
		// step2. get the categorical distribution based on m_disMatrix
		int codeNumTotal = uttLength * this->m_vqCodeBookSizeVec[cnt];
		{{
		    internal::vqCalculateOffsetFn fn;
		    fn.codeBooksize = this->m_vqCodeBookSizeVec[cnt];
		    fn.dismatrix    = helpers::getRawPointer(this->m_disMatrix);
		    fn.patTypes     = helpers::getRawPointer(this->patTypes());

		    thrust::transform(
                      thrust::counting_iterator<int>(0),
		      thrust::counting_iterator<int>(0) + uttLength,
		      m_softmaxTmp.begin(),
		      fn);
	       }}

		// calculate the exponent
		{{
		    internal::vqCalculateExpFn fn;
		    fn.codeBooksize = this->m_vqCodeBookSizeVec[cnt];
		    fn.offsets   = helpers::getRawPointer(m_softmaxTmp);
		    
		    thrust::for_each(
		      thrust::make_zip_iterator(
			thrust::make_tuple(this->m_disMatrix.begin(),
					   thrust::counting_iterator<int>(0))),
		      thrust::make_zip_iterator(
		        thrust::make_tuple(this->m_disMatrix.begin()         + codeNumTotal,
					   thrust::counting_iterator<int>(0) + codeNumTotal)),
		      fn);
		}}

		// sum up all outputs for each pattern
		{{
		    internal::vqSumUpOutputsFn fn;
		    fn.codeBooksize = this->m_vqCodeBookSizeVec[cnt];
		    fn.disMatrix    = helpers::getRawPointer(m_disMatrix);

		    thrust::for_each(
		      thrust::make_zip_iterator(
			thrust::make_tuple(m_softmaxTmp.begin(),
					   thrust::counting_iterator<int>(0))),
		      thrust::make_zip_iterator(
			thrust::make_tuple(m_softmaxTmp.begin()              + uttLength,
					   thrust::counting_iterator<int>(0) + uttLength)),
		      fn);
		}}

		// normalize the outputs
		{{
		    internal::vqNormalizeOutputsFn fn;
		    fn.codeBooksize = this->m_vqCodeBookSizeVec[cnt];
		    fn.normFacts    = helpers::getRawPointer(m_softmaxTmp);

		    thrust::for_each(
		      thrust::make_zip_iterator(
			thrust::make_tuple(this->m_disMatrix.begin(),
					   thrust::counting_iterator<int>(0))),
		      thrust::make_zip_iterator(
			thrust::make_tuple(this->m_disMatrix.begin()         + codeNumTotal,
					   thrust::counting_iterator<int>(0) + codeNumTotal)),
		      fn);
		}}

		// step3. sampling from the categorical distribution
		{{
		    // generate the random number
		    thrust::counting_iterator<unsigned int> index_sequence_begin(0);
		    thrust::transform(
			index_sequence_begin, index_sequence_begin + uttLength * m_bestNcode,
			m_softmaxRand.begin(),
			internal::genNoise(0.0, 1.0,
					   (int)(misFuncs::GetRandomNumber()*1000.0)));

		    // sampling
		    internal::vqSamplingSoftmax fn;
		    fn.codeBooksize = this->m_vqCodeBookSizeVec[cnt];
		    fn.codeNbest    = this->m_bestNcode;

		    fn.randomSeeds  = helpers::getRawPointer(m_softmaxRand);
		    fn.probBuff     = helpers::getRawPointer(m_disMatrix);

		    fn.howmanyBook  = this->m_vqCodeBookSizeVec.size();
		    fn.whichBook    = cnt;

		    fn.indexBuffer  = helpers::getRawPointer(this->m_selectedIdx);
		    fn.indexBufferF = helpers::getRawPointer(this->m_selectedIdxF);
		    
		    fn.patTypes     = helpers::getRawPointer(this->precedingLayer().patTypes());
		
		    thrust::for_each(
		     thrust::make_zip_iterator(
		      thrust::make_tuple(
				this->m_selectedIdxF.begin(),
				thrust::counting_iterator<int>(0))),
		     thrust::make_zip_iterator(
		      thrust::make_tuple(
				this->m_selectedIdxF.begin()      + uttLength * m_bestNcode,
				thrust::counting_iterator<int>(0) + uttLength * m_bestNcode)),
		     fn);
		}}
		
		// step4. load the average code
		{{
		    internal::vqLoadAvarageCode fn3;
		    fn3.codeDim      = layer->size();
		    fn3.dimPtr       = dimPtr;
		    
		    fn3.featureDim   = this->size();
		    fn3.bestNcode    = m_bestNcode;

		    fn3.howmanyBook  = this->m_vqCodeBookSizeVec.size();
		    fn3.whichBook    = cnt;

		    fn3.outputData   = helpers::getRawPointer(this->outputs());
		    fn3.codeData     = helpers::getRawPointer(this->weights()) + codeBookPtr;
		    fn3.index        = helpers::getRawPointer(this->m_selectedIdxF);
		    fn3.codeDimFlag  = helpers::getRawPointer(this->m_codeDimStatus);
		    
		    fn3.patTypes     = helpers::getRawPointer(this->patTypes());
		    
		    int n = uttLength * layer->size();
		    thrust::for_each(
                      thrust::make_zip_iterator(
			thrust::make_tuple(this->outputs().begin(),
					   thrust::counting_iterator<int>(0))),
		      thrust::make_zip_iterator(
		        thrust::make_tuple(this->outputs().begin()           + n,
					   thrust::counting_iterator<int>(0) + n)),
		      fn3);

		}}

		// step3. calculate the code differenaces
		{{
		    // do it later
		    internal::vqCodeDiffEMA fn4;
		    fn4.encoderLayerSize = layer->size();
		    fn4.thisLayerSize    = this->size();
		    
		    fn4.dimPtr  = dimPtr;

		    fn4.inputData  = helpers::getRawPointer(layer->outputs());
		    fn4.outputData = helpers::getRawPointer(this->outputs());

		    fn4.patTypes     = helpers::getRawPointer(this->patTypes());
		    
		    int n = uttLength * layer->size();
		    
		    real_t tmpError = thrust::transform_reduce(
                      thrust::make_zip_iterator(
			thrust::make_tuple(this->outputs().begin(),
					   thrust::counting_iterator<int>(0))),
		      thrust::make_zip_iterator(
		        thrust::make_tuple(this->outputs().begin()           + n,
					   thrust::counting_iterator<int>(0) + n)),
		      fn4, (real_t)0.0, thrust::plus<real_t>());
		    
		    if (m_codeDimStatus[dimPtr] != NN_VQLAYER_CODE_NOTUSE)
			m_codeError += (tmpError / uttLength);

		}}
		
	    }
	    
	    // Move the pointer to the next code book
	    codeBookPtr += layer->size() * this->m_vqCodeBookSizeVec[cnt];
	    dimPtr      += layer->size();
	    cnt += 1;
	}
	

    }

    // NN forward
    template <typename TDevice>
    void vqLayer<TDevice>::computeForwardPass(const int timeStep, const int nnState)
    {
	// To be implemented for inference stage
	if (this->precedingLayer().getSaveMemoryFlag())
	    throw std::runtime_error("The layer before vae is reduced in mem");

	if (Configuration::instance().vaeCodeInputDir().size() == 0)
	    throw std::runtime_error("Please provide latent data by using --vaeCodeInputDir");

	// the output (code) has been loaded from output. No need to compute again
	// or, it is not supported to generate code at each time step during generation
	
    }


    // NN backward
    template <typename TDevice>
    void vqLayer<TDevice>::computeBackwardPass(const int nnState)
    {
	int timeLength = this->parallelSequences() * this->curMaxSeqLength();

	int cnt = 0;
	int codeBookPtr = 0;
	int dimPtr = 0;
	int n = 0;
	
	layers::SkipLayer<TDevice> *tmpSkipLayerPtr;
	BOOST_FOREACH (layers::TrainableLayer<TDevice> *layer, m_preLayers){
	    
	    tmpSkipLayerPtr = dynamic_cast<layers::SkipLayer<TDevice>*>(layer);
	    // Gradients should be accumulated through skiplayer for layers not directly
	    // connected to VQLayer
	    {{
	      internal::CopyGradient fn0;
	      fn0.preLayerDim  = layer->size();
	      fn0.thisLayerDim = this->size();
	      fn0.paralNum     = this->parallelSequences();
	      fn0.dimPtr       = dimPtr;
	      fn0.gradientInput= helpers::getRawPointer(this->outputErrors());
	      fn0.resoRatio    = (layer->getResolution()/this->getResolution());
	      fn0.patTypes     = helpers::getRawPointer(layer->patTypes());
	      fn0.codeDimFlag  = helpers::getRawPointer(this->m_codeDimStatus);
	      
	      
	      n = layer->patTypes().size() * layer->size();
	      if (tmpSkipLayerPtr){
		  // Note: previous skipini will only one-input-one-output
		  //       it is save to set outputErrorsFromSkipLayer()[*] = 0 in
		  //       CopyGradient
		  thrust::for_each(
		    thrust::make_zip_iterator(
		       thrust::make_tuple(tmpSkipLayerPtr->outputErrorsFromSkipLayer().begin(),
					  thrust::counting_iterator<int>(0))),
		    thrust::make_zip_iterator(
		       thrust::make_tuple(tmpSkipLayerPtr->outputErrorsFromSkipLayer().begin() + n,
					  thrust::counting_iterator<int>(0) + n)),
		    fn0);
	      }else{
		  thrust::for_each(
			thrust::make_zip_iterator(
		          thrust::make_tuple(layer->outputErrors().begin(),
					     thrust::counting_iterator<int>(0))),
			thrust::make_zip_iterator(
		          thrust::make_tuple(layer->outputErrors().begin()     + n,
					     thrust::counting_iterator<int>(0) + n)),
			fn0);
	      }
	    }}

	    
	    // Update the codeBook
	    if (m_bestNcode == 1){
		{
		    internal::GradientForCodeBook fn1;
		    fn1.codeDim      = layer->size();
		    fn1.codeBookSize = this->m_vqCodeBookSizeVec[cnt];
		    fn1.howmanyBook  = this->m_vqCodeBookSizeVec.size();
		    fn1.whichBook    = cnt;
		    fn1.dimPtr       = dimPtr;
		    fn1.resoRatio    = (layer->getResolution()/this->getResolution());
		    fn1.paralNum     = this->parallelSequences();
	    
		    fn1.codeData     = helpers::getRawPointer(this->weights()) + codeBookPtr;
		    fn1.inputData    = helpers::getRawPointer(layer->outputs());
		    fn1.codeDimFlag  = helpers::getRawPointer(this->m_codeDimStatus);
		      
		    if (tmpSkipLayerPtr)
			fn1.preLayerGrad = helpers::getRawPointer(
						    tmpSkipLayerPtr->outputErrorsFromSkipLayer());
		    else
			fn1.preLayerGrad = helpers::getRawPointer(layer->outputErrors());
	    
		    fn1.beta         = m_betaPara;
		    fn1.index        = helpers::getRawPointer(this->m_selectedIdx);
	    
		    fn1.timeLength   = timeLength;
		    fn1.patTypes     = helpers::getRawPointer(this->patTypes());
	    
		    n = layer->size() * this->m_vqCodeBookSizeVec[cnt];
		    thrust::for_each(
                     thrust::make_zip_iterator(
		      thrust::make_tuple(this->_weightUpdates().begin() + codeBookPtr,
					 thrust::counting_iterator<int>(0))),
		     thrust::make_zip_iterator(
		      thrust::make_tuple(this->_weightUpdates().begin() + codeBookPtr   + n,
					 thrust::counting_iterator<int>(0) + n)),
		     fn1);
		}
		
	    }else{
		
		// backward for n-best codes case
		
		// step1. propagate the commitment loss \beta (encoder_output - ave_embedding)
		
		internal::CopyGradientBeta fn1;
		fn1.beta          = this->m_betaPara;
		fn1.inputLayerDim = layer->size();
		fn1.thisLayerDim  = this->size();
		fn1.dimPtr        = dimPtr;
		
		fn1.vqOutput      = helpers::getRawPointer(this->outputs());
		fn1.patTypes      = helpers::getRawPointer(layer->patTypes());
		fn1.codeDimFlag   = helpers::getRawPointer(this->m_codeDimStatus);
		
		n = layer->patTypes().size() * layer->size();
		
		if (tmpSkipLayerPtr){
		    
		  thrust::for_each(
		    thrust::make_zip_iterator(
		       thrust::make_tuple(tmpSkipLayerPtr->outputErrorsFromSkipLayer().begin(),
					  layer->outputs().begin(),
					  thrust::counting_iterator<int>(0))),
		    thrust::make_zip_iterator(
		       thrust::make_tuple(tmpSkipLayerPtr->outputErrorsFromSkipLayer().begin() + n,
					  layer->outputs().begin() + n,
					  thrust::counting_iterator<int>(0) + n)),
		    fn1);
		}else{
		  thrust::for_each(
			thrust::make_zip_iterator(
		          thrust::make_tuple(layer->outputErrors().begin(),
					     layer->outputs().begin(),
					     thrust::counting_iterator<int>(0))),
			thrust::make_zip_iterator(
		          thrust::make_tuple(layer->outputErrors().begin()     + n,
					     layer->outputs().begin()          + n,
					     thrust::counting_iterator<int>(0) + n)),
			fn1);
		}

		// step2. update the codebook using exponential moving average
		{
		    internal::GradientForCodeBookEMA fn1;
		    fn1.codeDim      = layer->size();
		    fn1.codeBookSize = this->m_vqCodeBookSizeVec[cnt];
		    
		    fn1.howmanyBook  = this->m_vqCodeBookSizeVec.size();
		    fn1.whichBook    = cnt;
		    
		    
		    fn1.dimPtr       = dimPtr;
		   
		    fn1.inputData    = helpers::getRawPointer(layer->outputs());
		    fn1.codeDimFlag  = helpers::getRawPointer(this->m_codeDimStatus);
		    fn1.index        = helpers::getRawPointer(this->m_selectedIdx);
		    fn1.timeLength   = timeLength;
		    fn1.patTypes     = helpers::getRawPointer(this->patTypes());

		    fn1.lambdaPara   = this->m_lambdaPara;
		    fn1.bestNcode    = this->m_bestNcode;
		    
		    n = layer->size() * this->m_vqCodeBookSizeVec[cnt];
		    
		    thrust::for_each(
                     thrust::make_zip_iterator(
		      thrust::make_tuple(this->weights().begin() + codeBookPtr,
					 thrust::counting_iterator<int>(0))),
		     thrust::make_zip_iterator(
		      thrust::make_tuple(this->weights().begin() + codeBookPtr   + n,
					 thrust::counting_iterator<int>(0) + n)),
		     fn1);
		}

		// clean the gradients for SGD on codes
		thrust::fill(this->_weightUpdates().begin(), this->_weightUpdates().end(), 0.0);
	    }

	    
	    // Move the pointer to the next code book
	    codeBookPtr += layer->size() * this->m_vqCodeBookSizeVec[cnt];
	    dimPtr      += layer->size();
	    cnt += 1;
	}
	
    }
	    
    template <typename TDevice>
    const std::string& vqLayer<TDevice>::type() const
    {
	static std::string s1("vqlayer");
        return s1;
    }

    template <typename TDevice>
    void vqLayer<TDevice>::exportLayer(const helpers::JsonValue &layersArray,
				       const helpers::JsonAllocator &allocator) const
    {
        TrainableLayer<TDevice>::exportLayer(layersArray, allocator);	
	(*layersArray)[layersArray->Size() - 1].AddMember("vqCodeBookSize",
							  m_vqCodeBookSizeAll, allocator);
	
	(*layersArray)[layersArray->Size() - 1].AddMember("beta",
							  m_betaPara, allocator);
	if (m_preLayerNamesStr.size())
	    (*layersArray)[layersArray->Size() - 1].AddMember("inputLayers",
							      m_preLayerNamesStr.c_str(),
							      allocator);
	if (m_vqCodeBookSizeStr.size())
	    (*layersArray)[layersArray->Size() - 1].AddMember("vqEachCodeBookSize",
							      m_vqCodeBookSizeStr.c_str(),
							      allocator);
	if (m_codeBookStatusStr.size())
	    (*layersArray)[layersArray->Size() - 1].AddMember("codeBookStatus",
							      m_codeBookStatusStr.c_str(),
							      allocator);
	if (m_bestNcode > 1){
	    (*layersArray)[layersArray->Size() - 1].AddMember("nbest", m_bestNcode, allocator);
	    (*layersArray)[layersArray->Size() - 1].AddMember("lambda", m_lambdaPara, allocator);
	}

    }

        // Add 0527: re-read the weight from weightsSection
    template <typename TDevice>
    void vqLayer<TDevice>::reReadWeight(const helpers::JsonValue &weightsSection,
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

	    if (inputWeightsChild.Size()!= 0)
		throw std::runtime_error("Error in reading vqLayer (input) weights");
	    if (biasWeightsChild.Size() != 0)
		throw std::runtime_error("Error in reading vqLayer (bias) weights");

	    // three kinds of possibility to read the weights
	    if (readCtrFlag==1){
		
		// the number of parameter should match exactly
		if (internalWeightsChild.Size() != this->size() * m_vqCodeBookSizeAll)
		    throw std::runtime_error("Error: vqLayer weights number doesn't match");

		weights.reserve(internalWeightsChild.Size());

		for (rapidjson::Value::ConstValueIterator it = internalWeightsChild.Begin(); 
		     it != internalWeightsChild.End(); 
		     ++it)
		    weights.push_back(static_cast<real_t>(it->GetDouble()));
		
	    }else if (readCtrFlag == 2 || readCtrFlag == 3){

		if (internalWeightsChild.Size() > this->size() * m_vqCodeBookSizeAll)
		    throw std::runtime_error("Error: vqLayer has more weights");

		
		// By defauly, copy the weight to the initial block
		weights.reserve(internalWeightsChild.Size());
		for (rapidjson::Value::ConstValueIterator it = internalWeightsChild.Begin(); 
		     it != internalWeightsChild.End(); 
		     ++it)
		    weights.push_back(static_cast<real_t>(it->GetDouble()));
		
	    }else{
		throw std::runtime_error(std::string("trainedParameterCtr not string of 0/1/2/3"));
	    }
	    
	    thrust::copy(weights.begin(), weights.end(), this->weights().begin());
	    
	}else{
	    printf("not read weight for layer %s", this->name().c_str());
	}
    }


    // NN backward
    template <typename TDevice>
    void vqLayer<TDevice>::computeBackwardPass(const int timeStep, const int nnState)
    {
	throw std::runtime_error("vqLayer computeBackwardPass(timeStep) not implemented");
    }
    
    template <typename TDevice>
    real_t vqLayer<TDevice>::codeError() const
    {
	return m_codeError;
    }

    template <typename TDevice>
    typename vqLayer<TDevice>::real_vector& vqLayer<TDevice>::codeIdx()
    {
	//thrust::copy(m_selectedIdx.begin(), m_selectedIdx.end(), m_selectedIdxF.begin());
	return m_selectedIdxF;
    }

    template <typename TDevice>
    int vqLayer<TDevice>::codeBookNum()
    {
	return (int)m_vqCodeBookSizeVec.size();
    }

    template class vqLayer<Cpu>;
    template class vqLayer<Gpu>;
    
}
