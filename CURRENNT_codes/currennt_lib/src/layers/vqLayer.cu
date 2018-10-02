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
#include "../helpers/JsonClasses.hpp"
#include "../helpers/misFuncs.hpp"

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

namespace internal{
namespace {
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
	m_vqCodeBookSizeStr = (layerChild->HasMember("vqEachCodeBookSize") ? 
			       ((*layerChild)["vqEachCodeBookSize"].GetString()) : "");
	m_preLayerNamesStr  = (layerChild->HasMember("inputLayers") ? 
			       ((*layerChild)["inputLayers"].GetString()) : "");
	m_codeBookStatusStr = (layerChild->HasMember("codeBookStatus") ? 
			       ((*layerChild)["codeBookStatus"].GetString()) : "");


	// Parse the preLayer option
	if (m_preLayerNamesStr.size()){
	    std::vector<std::string> tmpOpt;
	    misFuncs::ParseStrOpt(m_preLayerNamesStr, tmpOpt, ",");
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
			    tmpLayerPtr->name() != this->precedingLayer().name())
			{
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

	    layerCnt++;
	}

	m_codeDimStatus = tmpCodeDimStatus;
	
	// Initial check
	if (tmpVecSum != this->size())
	    throw std::runtime_error("vqLayer layer size != sum of input layer(s)");
	
	if (m_vqCodeBookSizeAll < 1)
	    throw std::runtime_error("vqLayer vqCodeBookSize is not an positive integer");

	// Initialize the distance matrix
	int maxSeqLengthParallel = this->parallelSequences() * maxSeqLength;
	cpu_real_vector temp(maxSeqLengthParallel * m_vqCodeBookSizeAll, 0.0);
	m_disMatrix = temp;

	cpu_int_vector temp2(maxSeqLengthParallel * m_vqCodeBookSizeVec.size(), 0);
	m_selectedIdx  = temp2;

	cpu_real_vector temp3(maxSeqLengthParallel * m_vqCodeBookSizeVec.size(), 0);
	m_selectedIdxF = temp3;

	m_betaPara    = (layerChild->HasMember("beta") ? 
			 ((*layerChild)["beta"].GetDouble()) : 0.25);

	// Set the gradients of dummy weights to zero
	real_vector tmpWeights = this->weights();
	this->reInitWeight();
	thrust::copy(tmpWeights.begin(), tmpWeights.begin() + validWeightNum,
		     this->weights().begin());
	thrust::fill(this->_weightUpdates().begin() + validWeightNum,
		     this->_weightUpdates().end(), 0.0);
    }	

    // Destructor
    template <typename TDevice>
    vqLayer<TDevice>::~vqLayer()
    {
    }
    
    // Load sequences
    template <typename TDevice>
    void vqLayer<TDevice>::loadSequences(const data_sets::DataSetFraction &fraction,
					 const int nnState)
    {
	TrainableLayer<TDevice>::loadSequences(fraction, nnState);
	
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

	int uttLength = this->curMaxSeqLength() * this->parallelSequences();

	m_codeError = 0.0;
	int cnt = 0;
	int codeBookPtr = 0;
	int dimPtr = 0;
	BOOST_FOREACH (layers::TrainableLayer<TDevice> *layer, m_preLayers){
	   	    
	    {{
	    // step1. compute the distance matrix
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


	    // step4. optional, calculate the error
	    {{
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

	    
	    // step3. use the best vector as the output
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
	      
	      
	      int n = layer->patTypes().size() * layer->size();
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
	    {{
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
	    
	    int n = layer->size() * this->m_vqCodeBookSizeVec[cnt];
	    thrust::for_each(
               thrust::make_zip_iterator(
		  thrust::make_tuple(this->_weightUpdates().begin() + codeBookPtr,
				     thrust::counting_iterator<int>(0))),
	       thrust::make_zip_iterator(
		  thrust::make_tuple(this->_weightUpdates().begin() + codeBookPtr   + n,
				     thrust::counting_iterator<int>(0) + n)),
	       fn1);

	    }}

	    
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
