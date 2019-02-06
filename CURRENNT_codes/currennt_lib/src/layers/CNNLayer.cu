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


#ifdef _MSC_VER
#   pragma warning (disable: 4244) // thrust/iterator/iterator_adaptor.h(121): warning C4244: '+=' : conversion from '__int64' to 'int', possible loss of data
#endif


#include "CNNLayer.hpp"

#include "../helpers/getRawPointer.cuh"
#include "../helpers/Matrix.hpp"
#include "../helpers/min.cuh"
#include "../helpers/max.cuh"
#include "../helpers/safeExp.cuh"
#include "../helpers/JsonClasses.hpp"
#include "../helpers/misFuncs.hpp"

#include "../activation_functions/Tanh.cuh"
#include "../activation_functions/Logistic.cuh"
#include "../activation_functions/Identity.cuh"
#include "../activation_functions/Relu.cuh"

#include "../Configuration.hpp"
#include "../MacroDefine.hpp"

#include <boost/foreach.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>

#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>

#include <fstream>
#include <cmath>
#include <vector>
#include <stdexcept>


#define LOCAL_CNN_WINFO_NUM 5

#define DEBUG_LOCAL_CNN 1

namespace internal{
namespace{

    typedef activation_functions::Tanh     cell_act_fn_t;

    //
    // dubstin.txt 20170421x01
    
    struct ConvolutionCore
    {

	real_t *dataBuffer;
	//real_t *targetBuff;
	real_t *biasWeight;
	
	int    *winSizeCum;
	int    *winHalfSize;
	int    *winTapInter;
	
	int     curLayerSize; 
	int     winTotalLength;
	
	const char *patTypes;
	int     paral;                
	int     maxSeqLength;         // max length of one utterance

	int     causal;
	int     outputTanh;
        __host__ __device__ void operator() (const thrust::tuple<real_t&, int> &t) const
        {
	    
            // unpack the tuple
            int outputIdx = t.get<1>();

            // calculate the pattern index
            int timeIdx = outputIdx / curLayerSize;   //
	    int dimIdx  = outputIdx % curLayerSize;   // which filter

	    if (patTypes[timeIdx] == PATTYPE_NONE)
		return;

	    int dimS  = winSizeCum[dimIdx];     //
	    int dimE  = winSizeCum[dimIdx+1];   // 
	    int winHS = winHalfSize[dimIdx];    // half window size
	    int inter = winTapInter[dimIdx];    // tap interval
	    
	    // location of the element to be added;
	    int dTmp     = dimS + winHS;
	    int tTmp     = timeIdx;
	    /* Fatal Error */
	    // How could I use
	    // int maxValue = 0; ???
	    real_t maxValue = 0;
	    
	    for (int shift = -1 * winHS; shift <= ((causal>0)?0:winHS); shift += 1){
		dTmp = (dimS + winHS) + shift;
		tTmp = timeIdx + shift * inter * paral;
		
		if (tTmp < 0                       || tTmp >= (maxSeqLength * paral) ||
		    patTypes[tTmp] == PATTYPE_NONE ||
		    dTmp < dimS                    || dTmp >= dimE)
		    continue;

		// accumulate the feature
		maxValue += dataBuffer[tTmp * winTotalLength + dTmp];
	    }

	    // add bias and pass through the activation function
	    if (outputTanh)
		t.get<0>() = cell_act_fn_t::fn(maxValue + biasWeight[dimIdx]);
	    else
		t.get<0>() = maxValue + biasWeight[dimIdx];
        }
    };


    struct ConvolutionCoreGra
    {

	real_t *dataBuffer;
	//real_t *GradBuffer;

	int    *winSizeCum;
	int    *winHalfSize;
	int    *winTapInter;
	
	int     curLayerSize; 
	int     winTotalLength;
	
	const char *patTypes;
	
	int     paral;                
	int     maxSeqLength;         // max length of one utterance

	int     causal;

        __host__ __device__ void operator() (const thrust::tuple<const real_t&, int> &t) const
        {
	    
            // unpack the tuple
            int outputIdx = t.get<1>();

            // calculate the pattern index
            int timeIdx = outputIdx / curLayerSize;   //
	    int dimIdx  = outputIdx % curLayerSize;   // which filter

	    if (patTypes[timeIdx] == PATTYPE_NONE)
		return;

	    int dimS  = winSizeCum[dimIdx];     //
	    int dimE  = winSizeCum[dimIdx+1];   // 
	    int winHS = winHalfSize[dimIdx];    // half window size
	    int inter = winTapInter[dimIdx];    // tap interval
	    
	    // location of the element to be added;
	    int dTmp  = dimS + winHS;
	    int tTmp  = timeIdx;
	    
	    for (int shift = -1 * winHS; shift <= ((causal>0)?0:winHS); shift += 1){
		dTmp = (dimS + winHS) + shift;
		tTmp = timeIdx + shift * inter * paral;
		
		if (tTmp < 0                       || tTmp >= (maxSeqLength * paral) ||
		    patTypes[tTmp] == PATTYPE_NONE ||
		    dTmp < dimS                    || dTmp >= dimE)
		    continue;
		
		// copy the gradient
		dataBuffer[tTmp * winTotalLength + dTmp] = t.get<0>();//GradBuffer[outputIdx];
	    }
        }
    };
        
    struct ComputeDeltaFn
    {
        __host__ __device__ void operator() (const thrust::tuple<real_t&, const real_t&> &t) const
        {
            real_t delta = cell_act_fn_t::deriv(t.get<1>()) * t.get<0>();
            t.get<0>() = delta;
        }
    };

    struct CNNFilterWeightCopy
    {
	real_t *weightBuffer;
	int    *weightCopyInfo;
	int    *filterIndexMap;
	
	int     filterNum;
	int     preLayerSize;
	int     biasPosition;
	int     biasPositionBuf;
	bool    reverse;
	
	__host__ __device__ void operator() (const thrust::tuple<real_t&, int> &t) const
	{
	    int weightIdx   = t.get<1>();
	    int filterIndex = filterIndexMap[weightIdx];

	    if (filterIndex >= filterNum)
		return; // impossible

	    if (weightIdx < biasPosition){
		// weight part
		int filterStartOriginal = weightCopyInfo[filterIndex * LOCAL_CNN_WINFO_NUM + 0];
		int filterStartBuffer   = weightCopyInfo[filterIndex * LOCAL_CNN_WINFO_NUM + 1];
		//int filterWidth         = weightCopyInfo[filterIndex * LOCAL_CNN_WINFO_NUM + 2];
		int filterHeight        = weightCopyInfo[filterIndex * LOCAL_CNN_WINFO_NUM + 3];
		
		int weightRelativeIdx   = weightIdx - filterStartOriginal;
		int dimIndex            = weightRelativeIdx % filterHeight;
		int colIndex            = weightRelativeIdx / filterHeight;
		
		int weightBufferIdx     = filterStartBuffer + colIndex * preLayerSize + dimIndex;

		if (reverse){
		    t.get<0>() = weightBuffer[weightBufferIdx];
		}else{
		    weightBuffer[weightBufferIdx] = t.get<0>();
		}
		
	    }else{
		// bias part
		int filterBiasBuffer    = weightCopyInfo[filterIndex * LOCAL_CNN_WINFO_NUM + 4];
		int biasBufferIdx       = biasPositionBuf + filterBiasBuffer;

		if (reverse){
		    t.get<0>() = weightBuffer[biasBufferIdx];
		}else{
		    weightBuffer[biasBufferIdx] = t.get<0>();
		}
	    }
	    
	}
    };

    
    struct CNNFilterDuplicate
    {
	real_t *weight;
	int    *wColIndex;
	int    *wRowIndex;
	int    *wFilterHeight;
	int    *wFilterShift;
	
	int     layerSize;
	int     matrixWNum;
	
	__host__ __device__ void operator() (const thrust::tuple<real_t&, int> &t) const
        {
	    int outputIdx = t.get<1>();

	    if (outputIdx < matrixWNum){
		// the weight part
		// calculate the pattern index
		int colIndex = outputIdx / layerSize;
		int rowIndex = outputIdx % layerSize;

		int colShift = wColIndex[colIndex];
		int rowShift = wRowIndex[colIndex];
		int filterH  = wFilterHeight[colIndex];
		
		if (rowIndex < rowShift || rowIndex >= (filterH + rowShift)){
		    // this dimension is void (set to zero)
		    t.get<0>() = 0.0;
		}else{
		    if (colShift == 0){
			// this is the block not need to be shifted
			// do nothing
		    }else{
			if ((colIndex - colShift) < 0)
			    return;			// impossible
			
			if ((rowIndex - rowShift) < 0)
			    return;			// impossible
			
			t.get<0>() = weight[(colIndex - colShift) * layerSize +
					    (rowIndex - rowShift)];
		    }
		}
	    }else{
		// the bias part
		int shift = wFilterShift[outputIdx - matrixWNum];
		if (shift == 0){
		    // the original bias, doing nothing
		}else{
		    t.get<0>() = weight[outputIdx - shift];
		}
	    }
	}
    };

    struct CNNFilterGradientMerge
    {
	real_t *weightGra;
	int    *wColIndex;
	int    *wRowIndex;
	int    *wFilterHeight;
	int    *wFilterShift;
	
	int    *wFilterWidth;
	int    *wFilterShiftNum;
	int    *wFilterShiftNumBias;
	
	int     layerSize;
	int     matrixWNum;
	
	__host__ __device__ void operator() (const thrust::tuple<real_t&, int> &t) const
        {
	    int outputIdx = t.get<1>();

	    if (outputIdx < matrixWNum){
		// the weight part
		// calculate the pattern index
		int colIndex = outputIdx / layerSize;
		int rowIndex = outputIdx % layerSize;

		int colShift = wColIndex[colIndex];
		int stride   = wRowIndex[colIndex];
		int filterH  = wFilterHeight[colIndex];
		int shiftNum = wFilterShiftNum[colIndex];
		int filterW  = wFilterWidth[colIndex];
		
		if (colShift == 0){
		    if (rowIndex >= filterH){
			// zero part
			t.get<0>() = 0;
		    }else{
			// point to the gradient buffer for the original feature
			for (int i = 1; i < shiftNum; i++)
			    t.get<0>() += weightGra[outputIdx + i * filterW * layerSize + i*stride];
		    }
		}
	    }else{
		// the bias part
		int shift = wFilterShiftNumBias[outputIdx - matrixWNum];
		if (shift == 0){
		    // the shifted part
		}else{
		    for (int i = 1; i < shift; i++)
			t.get<0>() += weightGra[outputIdx + i];
		}
	    }
	}
    };



    struct ConvolutionCoreMemSaveMode
    {

	real_t *dataBuffer;
	real_t *biasWeight;
       
	int     recFieldSize;     // recep field size
	int     curLayerSize;     // output feature dimension
	int     winTotalLength;   // dimension of the con buffer (3 * curLayerSize)

	int     timeStep;         // absolute time index
	int     parallel;
	int     outputTanh;
	const char *patTypes;

	// for parallel * curLayerSize
        __host__ __device__ void operator() (const thrust::tuple<real_t&, int> &t) const
        {
            // calculate the index
            int uttIdx  = t.get<1>() / curLayerSize;   // position in one parallel block
	    int dimIdx  = t.get<1>() % curLayerSize;   // dimension idx

	    if (patTypes[timeStep * parallel + uttIdx] == PATTYPE_NONE)
		return;

	    int timeIdxBuf1 = (timeStep % (recFieldSize+1)) * parallel + uttIdx;
	    int timeIdxBuf2 = ((timeStep+1) % (recFieldSize+1)) * parallel + uttIdx;
	    // (time+1) % (recFieldSize+1) = (time - recFieldSize) % (recFieldSize+1)
	    int dimIdxBuf1  = dimIdx * 3 + 1; // transformed by the curennt link of CNN
	    int dimIdxBuf2  = dimIdx * 3;     // transformed by the previous link of CNN
	    real_t summedOutput = (dataBuffer[timeIdxBuf1 * winTotalLength + dimIdxBuf1] +
				   dataBuffer[timeIdxBuf2 * winTotalLength + dimIdxBuf2]);
	    // add bias and pass through the activation function
	    if (outputTanh)
		t.get<0>() = cell_act_fn_t::fn(summedOutput + biasWeight[dimIdx]);
	    else
		t.get<0>() = summedOutput + biasWeight[dimIdx];
        }
    };

    
} // namespace 
} // namespace internal

namespace CNNTools{


    // return the number of shifted filters
    //  input:
    //   featureDim:  dimension of the input feature
    //   height:      height of the filter
    //   stride:      stride of the filter
    //  output = floor((FeatureDim - FilterHeight)/stride) + 1
    int getShiftSteps(const int featureDim, const int height, const int stride){
	// no zero padding
	// the same filter will be shifted along the dimension of input feature matrix
	//
	if (featureDim < height){
	    printf("Feature dimension %d cannot < filter height %d\n", featureDim, height);
	    throw std::runtime_error("Please check filter window_height");
	}
	if (stride < 1){
	    printf("Filter stride cannot < 1\n");
	    throw std::runtime_error("Please check filter window_stride");
	}
	return ((int)std::floor((float)(featureDim - height)/(float)stride) + 1);
    }
    
    // return the width of window = 2 * halfWidth + 1
    // input:
    //    opt: half-width of a window
    int getWinWidth(int opt){
	return (opt * 2 + 1);
    }

    // return the total width of windows
    // input:
    //    opt: vector of half-width of each window
    int getWinTotalLength(Cpu::int_vector &opt){
	int cnt =0;
	for (int i=0; i < opt.size(); i++){
	    cnt += getWinWidth(opt[i]);
	}
	return cnt;
    }

    // return the total width of windows
    // input:
    //    opt: vector of half-width of each window
    int getWinCausalLength(Cpu::int_vector &opt){
	int cnt =0;
	for (int i=0; i < opt.size(); i++){
	    cnt += (opt[i] + 1);
	}
	return cnt;
    }
    // parse the option for CNN configuration
    // input:
    //    cnnOpt: text string of configuration for original filter set
    //    layerSize: layer size (this layer)
    //    dupOpt: index of the shifted filter to the original filter set
    //            e.g., [1, 1, 1, 2, 2],
    //            the first/second/third shifted filters corresponding to 1st original filter
    //            the forth/fifth shifted filters correponsding to 2nd orignal filter
    // output:
    //    outOpt: vector of parased configuration for each shifted filter
    void ParseCNNOpt(const std::string cnnOpt, int layerSize, Cpu::int_vector &dupOpt,
		     Cpu::int_vector  &outOpt){

	misFuncs::ParseIntOpt(cnnOpt, outOpt);
	if (dupOpt.size()<1)
	    return;
		
	Cpu::int_vector tmp(layerSize,  0);
	for (int i = 0; i < layerSize; i++){
	    if (dupOpt[i] >= outOpt.size()){
		printf("Parsing %s", cnnOpt.c_str());
		throw std::runtime_error("Error in parsing");
	    }
	    // dupOpt[i] is the index of the shifted filter in the original filter set
	    tmp[i]  = outOpt[dupOpt[i]];
	}
	outOpt = tmp;
	  
    }

    // preparing index data for CNN operation
    // input:
    //    winWidthOpt:  input configuration of filter width (original filter)
    //    winHeightOpt: input configuration of filter height (original filter)
    //    thisLayerSize:
    //    preLayerSize:
    // output:
    //    winIndex:     index of each shifted filter in the original filter set
    //    winHeight:    height of the shifted filter
    //    winShift:     parameter for shifting of each shifted filter
    //    winShiftRev:  how many shifted filters is used for each original filter
    //                  this is used by the shifted filter with shift == 0
    //    winStride:    stride of each filter
    //    winWidthCum:  position of the filter in the filter stack
    //    wCopyInfo:    a buffer to store the informaiton about the filters
    int getWindowIndex(const std::string winWidthOpt,
		       const std::string winHeightOpt,
		       const std::string winStrideOpt, 
		       int thisLayerSize,
		       int preLayerSize,
		       Cpu::int_vector &winIndex,  Cpu::int_vector &winHeight,
		       Cpu::int_vector &winShift,  Cpu::int_vector &winShiftRev,
		       Cpu::int_vector &winStride, Cpu::int_vector &wCopyInfo,
		       Cpu::int_vector &winWidthCum){
	//
	winIndex.resize(thisLayerSize, 0);
	winHeight.resize(thisLayerSize, 0);
	winShift.resize(thisLayerSize, 0);
	winShiftRev.resize(thisLayerSize, 0);
	winStride.resize(thisLayerSize, 0);
	winWidthCum.resize(thisLayerSize + 1, 0);


	if (winHeightOpt.size() > 0){
	    
	    // if 2-D convolution
	    Cpu::int_vector width;
	    Cpu::int_vector height;
	    Cpu::int_vector stride;
		
	    misFuncs::ParseIntOpt(winHeightOpt, height);
	    misFuncs::ParseIntOpt(winWidthOpt,  width);
	    if (winStrideOpt.size() > 0)
		misFuncs::ParseIntOpt(winStrideOpt,  stride);
	    else
		stride.resize(width.size(), 1);
		
	    if (width.size() != height.size() || width.size() != stride.size())
		throw std::runtime_error("Unequal length of window_height, window_width");

	    // 
	    wCopyInfo.resize(width.size() * LOCAL_CNN_WINFO_NUM, 0);
	    
	    int cnt          = 0;  // counter of the shifted filter
	    int widthCum     = 0;
	    int wPosOriginal = 0;  // position of the pointer in the original weight matrix
	    int wPosBuffer   = 0;  // position of the pointer in the weight buffer
	    int wShiftSum    = 0;
	    int expectedSize = 0;

	    // check layer size
	    for (int i = 0; i < width.size(); i++){
		if (height[i] > preLayerSize)
		    throw std::runtime_error("Error filter height larger than input feature dim");
		expectedSize += getShiftSteps(preLayerSize,  height[i], stride[i]);
	    }
	    if (expectedSize != thisLayerSize){
		printf("\nCNN layer size should be %d. ", expectedSize);
		throw std::runtime_error("Please change configuration in network.jsn");
	    }
	    
	    for (int i = 0; i < width.size(); i++){

		// How many times a filter will be shifted over the input feature map
		int shiftSteps = getShiftSteps(preLayerSize,  height[i], stride[i]);
		
		// store the index of the shifted filter in the orignal filter set
		for (int j = 0; j < shiftSteps; j++){
		    if (cnt >= thisLayerSize)
			throw std::runtime_error("Impossible error");
		    
		    winIndex[cnt]    = i;         // index of the shifted filter in the orignal set
		    winShift[cnt]    = j;         // shifted position of this shifted filter
		    winHeight[cnt]   = height[i]; // height
		    winStride[cnt]   = stride[i]; // stride
		    winWidthCum[cnt] = widthCum;
		    if (j==0)
			winShiftRev[cnt] = shiftSteps; // how many shifted filters ?
		    cnt++;
		    widthCum += getWinWidth(width[i]);
		}

		// store the weight position in the weight buffer and weight matrix
		// start position in the weight matrix
		wCopyInfo[i * LOCAL_CNN_WINFO_NUM + 0] = wPosOriginal;
		wPosOriginal += getWinWidth(width[i]) * height[i];
		// start position in the weight buffer
		wCopyInfo[i * LOCAL_CNN_WINFO_NUM + 1] = wPosBuffer;
		wPosBuffer   += (getWinWidth(width[i]) * preLayerSize) * shiftSteps;
		// start position for the bias part
		wCopyInfo[i * LOCAL_CNN_WINFO_NUM + 4] = wShiftSum;
		wShiftSum    += shiftSteps; 
		// height and width
		wCopyInfo[i * LOCAL_CNN_WINFO_NUM + 2] = getWinWidth(width[i]);
		wCopyInfo[i * LOCAL_CNN_WINFO_NUM + 3] = height[i];
		
	    }
	    winWidthCum[thisLayerSize] = widthCum;
	    
	    if (cnt != thisLayerSize)
		throw std::runtime_error("Layer size != feature dimension");
	    
	    return width.size();
	    
	}else{
	    
	    // 1-D convolution
	    // For 1-D CNN,
	    // dimension of output feature vector (for one frame)= #. output channel
	    // dimension of output feature vector = #. filter
	    // 
	    wCopyInfo.resize(thisLayerSize, 0);
	   
	    Cpu::int_vector width;
	    misFuncs::ParseIntOpt(winWidthOpt,  width);
	    if (width.size() != thisLayerSize)
		throw std::runtime_error("Unequal length of window_width config and layer size");

	    int cum = 0;
	    for (int i = 0; i< thisLayerSize; i++){
		// index of each filter
		winIndex[i]  = i;
		winHeight[i] = preLayerSize;
		winStride[i] = 1;
		winShift[i]  = 0;
		winWidthCum[i] = cum;
		cum += getWinWidth(width[i]);
	    }
	    winWidthCum[thisLayerSize] = cum;
	    return thisLayerSize;
	}
    }

    
    // Return the number of CNN weight / thisLayerSize
    // Note, in TriainableLayers, the number of weight will be multiplied by this->size
    //       thus, returned value is (ceil(#/layerSize)) 
    int getCNNWeight(const std::string winWidthOpt,  const std::string winHeightOpt,
		     const std::string winStrideOpt, int thisLayerSize, int preLayerSize,
		     bool accurate, bool causal){
	
	Cpu::int_vector width;
	Cpu::int_vector height;
	Cpu::int_vector stride;
	
	if (winHeightOpt.size() > 0){
	    // 2-D convolution

	    misFuncs::ParseIntOpt(winHeightOpt, height);
	    misFuncs::ParseIntOpt(winWidthOpt,  width);

	    if (winStrideOpt.size() > 0)
		misFuncs::ParseIntOpt(winStrideOpt,  stride);
	    else
		stride.resize(width.size(), 1);
		
	    if (width.size() != height.size() || width.size() != stride.size()){
		printf("Unequal length of configuration string:\n");
		printf("there are %d window width parameters (window_width),", (int)width.size());
		printf(" %d window height parameters (window_height),", (int)height.size());
		printf(" %d window stride parameters (window_stride),", (int)stride.size());
		throw std::runtime_error("Please check window_height,window_width,window_stride");
	    }

	    int cnt = 0;
	    if (accurate){
		// Accurate number of filter number = \sum_i filter_width_i * filter_height_i
		for (int i = 0; i < width.size(); i++)
		    cnt += height[i] * (causal?(width[i]+1):getWinWidth(width[i]));
		cnt += width.size();
		return cnt;
	    }else{
		// Number of weights for memory allocation.
		// Always allocate the memory space for the non-causal part
		for (int i = 0; i < width.size(); i++)
		    cnt += height[i] * getWinWidth(width[i]);
		cnt += width.size();
		
		// divide thisLayerSize, this number will be multiplied by thisLayerSize
		// in TrainableLayer.cu
		return ((int)std::ceil((float)cnt/(float)thisLayerSize));
	    }
	    // # B1
	}else{

	    // default 1-D convolution
	    
	    misFuncs::ParseIntOpt(winWidthOpt, width);
	    if (width.size() > 0){
		// count the total number of filter widths
		// parameter = width * pre_layer_size + #filter
		if (accurate)
		    return ((causal?getWinCausalLength(width):getWinTotalLength(width)) *
			    preLayerSize + thisLayerSize);
		else	
		    return ((int)std::ceil(getWinTotalLength(width) *
					   (float)preLayerSize /
					   (float)thisLayerSize) + 1);
	    }else{
		return 0;
	    }
	}
    }

    // Return various data vectors to be used by 2-D convolution
    //  input:
    //    filterWidth:  half-width of each shifted filter
    //    filterIndex:  index of shifted filter in original filter set
    //    filterHeight: height of each each shifted filter
    //    shiftNum:     how many shifted filters for each original filter?
    //  output:
    //    ColoutIndex:  shifted position from the original filter (column)
    //    RowoutIndex:  shifted position from the original filter (row)
    //    filHeight:    height of this filter
    //    widthCol:     width of the filter
    //    widthShift:   how many shifted filters for each original filter?
    void fillInColIndex(Cpu::int_vector &filterWidth,  Cpu::int_vector &filterIndex,
			Cpu::int_vector &filterHeight, Cpu::int_vector &shiftNum,
			Cpu::int_vector &stride,       const int winTotalL,
			Cpu::int_vector &ColoutIndex,
			Cpu::int_vector &RowoutIndex,
			Cpu::int_vector &filHeight,
			Cpu::int_vector &widthCol,
			Cpu::int_vector &widthShift){

	ColoutIndex.resize(winTotalL, 0);
	RowoutIndex.resize(winTotalL, 0);
	filHeight.resize(winTotalL, 0);
	widthCol.resize(winTotalL, 0);
	widthShift.resize(winTotalL, 0);
	
	if (filterWidth.size() != filterIndex.size())
	    throw std::runtime_error("Error fillInColIndex, filterWidth filterIndex unequal size");
	
	int cnt          = 0;
	int indexTracker = -1;
	int colshiftTmp  = 0;
	int rowshiftTmp  = 0;
	//int winTotalL    = ColoutIndex.size();
	for (int i = 0; i < filterWidth.size(); i++){
	    if (filterIndex[i] != indexTracker){
		// come to the new filter
		colshiftTmp  = 0;
		rowshiftTmp  = 0;
		indexTracker = filterIndex[i];
	    }else{
		colshiftTmp += getWinWidth(filterWidth[i]); // shift distance = filter width
		rowshiftTmp += stride[i];                    // 
	    }
	    
	    for (int j = 0; j < getWinWidth(filterWidth[i]); j++){
		if (cnt > winTotalL){
		    printf("column number %d > m_winTotalL %d\n", cnt, (int)ColoutIndex.size());
		    throw std::runtime_error("Error in parsing fillInColIndex");
		}
		ColoutIndex[cnt] = colshiftTmp;
		RowoutIndex[cnt] = rowshiftTmp;
		filHeight[cnt]   = filterHeight[i];
		widthCol[cnt]    = getWinWidth(filterWidth[i]);
		if (colshiftTmp == 0)
		    widthShift[cnt]  = shiftNum[i];
		cnt++;
	    }
	}
	if (cnt != winTotalL)
	    throw std::runtime_error("Error in parsing fillInColIndex. Cnt != winTotalL");
    }


    void fillInFilterWeightMap(const std::string winWidthOpt,  const std::string winHeightOpt,
			       const int weightNum,
			       Cpu::int_vector &filterWeightMap){
	
	Cpu::int_vector width;
	Cpu::int_vector height;
	filterWeightMap.resize(weightNum, 0);
	
	misFuncs::ParseIntOpt(winHeightOpt, height);
	misFuncs::ParseIntOpt(winWidthOpt,  width);
	if (height.size() != width.size())
	    throw std::runtime_error("Error height width unequal length");

	int weightCnt = 0;
	for (int filterCnt = 0; filterCnt < width.size(); filterCnt++){
	    int filterWidth  = getWinWidth(width[filterCnt]);
	    int filterHeight = height[filterCnt];
	    int wNum = filterWidth * filterHeight;
	    for (int i = 0; i<wNum; i++){
		if (weightCnt > weightNum)
		    throw std::runtime_error("weight number larger than expected");
		filterWeightMap[weightCnt] = filterCnt;
		weightCnt++;
	    }
	}
	for (int filterCnt = 0; filterCnt < width.size(); filterCnt++){
	    filterWeightMap[weightCnt] = filterCnt;
	    weightCnt++;
	}
	if (weightCnt != weightNum)
	    throw std::runtime_error("weight number unequal to expected number");
    }
    
    void printCNNConfiguration(const int originalFilterNum, const int layerSize,
			       const int preLayerSize, 
			       Cpu::int_vector &winIndex,  Cpu::int_vector &winWidth,
			       Cpu::int_vector &winHeight, Cpu::int_vector &winStride){
	
	int channelNum = layerSize; // use the standard name
	
	if (layerSize != winIndex.size() || layerSize != winHeight.size() ||
	    layerSize != winStride.size()|| layerSize != winWidth.size()){
	    // impossible error
	    throw std::runtime_error("Unequal length of winIndex, winStride, winHeight");
	}
	printf("\n\tFilter details:");
	printf("\n\t--Filter ID-- | -- Width -- | -- Height -- |");
	printf("-- Stride -- | -- Weight -- | -- Output Channel -- \n");
	int cnt = 0;
	for (int cnt1 = 0; cnt1 < originalFilterNum; cnt1++){
	    printf("\t%10d    |", cnt1);
	    for (int cnt2 = 0; cnt2 < channelNum; cnt2++){
		if (winIndex[cnt2] == cnt1){
		    int chanlocal = getShiftSteps(preLayerSize, winHeight[cnt2], winStride[cnt2]);
		    printf("%10d   |", getWinWidth(winWidth[cnt2]));
		    printf("%11d   |", winHeight[cnt2]);
		    printf("%10d   |", winStride[cnt2]);
		    printf("%11d   |", winHeight[cnt2] * getWinWidth(winWidth[cnt2]) + 1);
		    printf("%10d   \n",  chanlocal);
		    cnt += chanlocal;
		    break;
		}
	    }
	}
	if (cnt != channelNum){
	    // impossible error
	    throw std::runtime_error("Unequal output channel and layersize");
	}
	printf("\tCNN total output channels: %d\n", channelNum);
    }
}

namespace layers {
   
    /*****************************************************************************************
     * CNN layer 
     *****************************************************************************************/
    template <typename TDevice>
    CNNLayer<TDevice>::CNNLayer(const helpers::JsonValue &layerChild, 
				const helpers::JsonValue &weightsSection,
				Layer<TDevice> &precedingLayer,
				int maxSeqLength,
				int layerID)
	: m_winWidth_Opt    ((layerChild->HasMember("window_width")) ? 
			     ((*layerChild)["window_width"].GetString()) : (""))
	, m_winInterval_Opt ((layerChild->HasMember("window_tap_interval")) ? 
			     ((*layerChild)["window_tap_interval"].GetString()) : (""))
	, m_winHeight_Opt   ((layerChild->HasMember("window_height")) ? 
			     ((*layerChild)["window_height"].GetString()) : (""))
	, m_winStride_Opt   ((layerChild->HasMember("window_stride")) ? 
			     ((*layerChild)["window_stride"].GetString()) : (""))
	, TrainableLayer<TDevice>  (layerChild, weightsSection,
				    0,
				    CNNTools::getCNNWeight(
					(layerChild->HasMember("window_width")) ? 
					((*layerChild)["window_width"].GetString()) : (""),
					(layerChild->HasMember("window_height")) ? 
					((*layerChild)["window_height"].GetString()) : (""),
					(layerChild->HasMember("window_stride")) ? 
					((*layerChild)["window_stride"].GetString()) : (""),
					(layerChild->HasMember("size")) ? 
					((*layerChild)["size"].GetInt()) : (0),
					precedingLayer.size(), false, false),
				    precedingLayer, maxSeqLength, layerID)
	, m_outputTanh(1)
    {
	
	// Check casual filter
	m_causalFlag = (layerChild->HasMember("causal") ? 
			static_cast<real_t>((*layerChild)["causal"].GetInt()) : 0);
	
	// whether use tanh as activation function
	m_outputTanh = (layerChild->HasMember("tanhoutput") ? 
			static_cast<real_t>((*layerChild)["tanhoutput"].GetInt()) : 1);
	
	if (m_winWidth_Opt.size() < 1)
	    throw std::runtime_error("Fail to find window_width in network.jsn");

	// get all kinds of index, pointers, and so on
	int filterNum = CNNTools::getWindowIndex(
		m_winWidth_Opt,  m_winHeight_Opt, m_winStride_Opt,
		this->size(),    precedingLayer.size(),
		m_winIndex_H,    m_winHeight_H,   m_winShiftIndex_H,  m_winShiftRevId_H,
		m_winStride_H,   m_wCopyInfo_H,   m_winWidth_Cum_H);
	m_winIndex_D      = m_winIndex_H;
	m_winHeight_D     = m_winHeight_H;
	m_winShiftIndex_D = m_winShiftIndex_H;
	m_winShiftRevId_D = m_winShiftRevId_H;
	m_winStride_D     = m_winStride_H;
	m_wCopyInfo_D     = m_wCopyInfo_H;
	m_winWidth_Cum_D  = m_winWidth_Cum_H;
	m_winNumOrignal   = filterNum;

	// Parse the width of filter window
	m_winWidth_H.clear();
	CNNTools::ParseCNNOpt(m_winWidth_Opt, this->size(), m_winIndex_H, m_winWidth_H);
	m_winWidth_D = m_winWidth_H;

	// total width of filter window
	m_winTotalL  = CNNTools::getWinTotalLength(m_winWidth_H);

	// number of weights (accurate)
	int weightNumAcc  = CNNTools::getCNNWeight(m_winWidth_Opt, m_winHeight_Opt, m_winStride_Opt,
						   this->size(), precedingLayer.size(), true,
						   m_causalFlag);
	
	// number of weights for memory allocation
	m_weightNum = CNNTools::getCNNWeight(m_winWidth_Opt, m_winHeight_Opt, m_winStride_Opt,
					     this->size(), precedingLayer.size(), true,
					     false);
	
	m_biasPos    = m_weightNum - filterNum;
	
	// number of weights (in buffer)
	m_weightBufferNum  = m_winTotalL * precedingLayer.size() + this->size();
	m_biasPosInBuffer  = m_winTotalL * precedingLayer.size();

	// m_numMatrixW = m_winTotalL * precedingLayer.size();
	
	if (m_winHeight_Opt.size()>0){
	    // 2-D convolution
	    // get the column index for filter shifting
	    CNNTools::fillInColIndex(
		m_winWidth_H,      m_winIndex_H,    m_winHeight_H,
		m_winShiftRevId_H, m_winStride_H,   m_winTotalL,
		m_winColIndex_H,   m_winRowIndex_H, m_winColHeight_H,
		m_winWidthCol_H,   m_winShiftNum_H);
	    m_winColIndex_D  = m_winColIndex_H;
	    m_winRowIndex_D  = m_winRowIndex_H;
	    m_winColHeight_D = m_winColHeight_H;
	    m_winWidthCol_D  = m_winWidthCol_H;
	    m_winShiftNum_D  = m_winShiftNum_H;

	    // fill in a map between individual filter weight and filter index
	    Cpu::int_vector tmp;
	    CNNTools::fillInFilterWeightMap(m_winWidth_Opt, m_winHeight_Opt, m_weightNum, tmp);
	    m_weightFilter_map = tmp;
	    m_1DCNNOnly = 0;
	}else{
	    // 1-D convolution
	    m_winColIndex_H.clear();
	    m_winRowIndex_H.clear();
	    m_winColHeight_H.clear();
	    m_winShiftNum_H.clear();
	    m_winWidthCol_H.clear();
	    m_winColIndex_D  = m_winColIndex_H;
	    m_winRowIndex_D  = m_winRowIndex_H;
	    m_winColHeight_D = m_winColHeight_H;
	    m_winWidthCol_D  = m_winWidthCol_H;
	    m_winShiftNum_D  = m_winShiftNum_H;
	    m_weightFilter_map.clear();
	    m_1DCNNOnly = 1;
	}
	
	// parse the tap interval
	m_winInterval_H.clear();
	if (m_winInterval_Opt.size())
	    CNNTools::ParseCNNOpt(m_winInterval_Opt, this->size(), m_winIndex_H, m_winInterval_H);
	else{
	    m_winInterval_H = m_winWidth_H;
	    thrust::fill(m_winInterval_H.begin(), m_winInterval_H.end(), 1);
	}
	m_winInterval_D = m_winInterval_H;
	
	if (m_winInterval_H.size() != m_winWidth_H.size() ||
	    m_winInterval_H.size() != this->size())
	    throw std::runtime_error("Incompatible layer size and window configuration in CNN");

	/*
	// Create index to the first weight cell of each window filter
	//Cpu::int_vector tmp(m_winWidth_H.size() + 1, 0);
	Cpu::int_vector tmp2(m_winWidth_H.size() + 1, 0);
	for (int i = 1; i < (m_winWidth_H.size()+1); i++){
	    // Index of the first weight element for this filter
	    //tmp[i] = tmp[i-1]+ CNNTools::getWinWidth(m_winWidth_H[i-1]) * precedingLayer.size();
	    // How many columns of filter weight before this filter
	    tmp2[i]= tmp2[i-1] + CNNTools::getWinWidth(m_winWidth_H[i-1]);
	}
	//m_weightIdx    = tmp;
	m_winWidth_Cum = tmp2;
	*/
	
	// allocate memory for convolution buffer (\sum_window_Length * Time)
	this->__allocateLocalMem();
	

	// allocate memmory for weight buffer
	m_weightBuffer.resize(precedingLayer.size() * m_winTotalL + this->size(), 0.0);
	
	// done
	printf("\n");
	if (m_winHeight_Opt.size())
	    printf("\tCNN 2-D convolution\n");
	else
	    printf("\tCNN 1-D convolution\n");

	// more memory space is allocated to store the weights
	// for example, causal filters use the same amount of memory as normal filters
	printf("\tCNN trainable weights: %d (weights in Network summary may be inaccurate)\n",
	       weightNumAcc);
	if (m_causalFlag > 0)
	    printf("\tCNN uses causal filter\n");
	if (m_outputTanh == 0)
	    printf("\tCNN without tanh output function\n");
	    
	if (Configuration::instance().verboseLevel() == OP_VERBOSE_LEVEL_3){
	    CNNTools::printCNNConfiguration(filterNum, this->size(), precedingLayer.size(),
					    m_winIndex_H,  m_winWidth_H,
					    m_winHeight_H, m_winStride_H);
	}else{
	    printf("\tCNN winwidth:    %s\n", m_winWidth_Opt.c_str());
	    printf("\tCNN winDilution: %s\n", m_winInterval_Opt.c_str());
	    printf("\tCNN winHeight:   %s\n", m_winHeight_Opt.c_str());
	    printf("\tCNN winStride:   %s\n", m_winStride_Opt.c_str());
	}
    }
    
    template <typename TDevice>
    CNNLayer<TDevice>::~CNNLayer()
    {
    }

    template <typename TDevice>
    void CNNLayer<TDevice>::__allocateLocalMem()
    {
	m_conBuffer.resize(this->outputs().size() / this->size() * m_winTotalL, 0);
    }

    template <typename TDevice>
    void CNNLayer<TDevice>::__clearLocalMem()
    {
	m_conBuffer.clear();
	m_conBuffer.shrink_to_fit();
    }
    
    template <typename TDevice>
    void CNNLayer<TDevice>::computeForwardPass(const int nnState)
    {
	if (this->getSaveMemoryFlag())
	    throw std::runtime_error("Memory save mode should be turned off");
	
	// Step0: 
	{{

	    if (m_winHeight_Opt.size()>0){

		// 2-D convolution
		
		// Copy weight from this->weights() to m_weightBuffer
		internal::CNNFilterWeightCopy fn1;
		fn1.weightBuffer   = helpers::getRawPointer(m_weightBuffer);
		fn1.weightCopyInfo = helpers::getRawPointer(m_wCopyInfo_D);
		fn1.filterIndexMap = helpers::getRawPointer(m_weightFilter_map);
		fn1.filterNum      = m_winNumOrignal;
		fn1.preLayerSize   = this->precedingLayer().size();
		fn1.biasPosition   = m_biasPos;
		fn1.biasPositionBuf= m_biasPosInBuffer;
		fn1.reverse        = false;
		
		int n = m_weightNum;
		thrust::for_each(
		  thrust::make_zip_iterator(
			thrust::make_tuple(this->weights().begin(),
					   thrust::counting_iterator<int>(0))),
		  thrust::make_zip_iterator(
			thrust::make_tuple(this->weights().begin() + n, 
					   thrust::counting_iterator<int>(0) + n)),
		  fn1);
		
		
		// duplicate weights inside m_weightBuffer
		internal::CNNFilterDuplicate fn;
		fn.weight       = helpers::getRawPointer(m_weightBuffer);
		fn.wColIndex    = helpers::getRawPointer(m_winColIndex_D);
		fn.wRowIndex    = helpers::getRawPointer(m_winRowIndex_D);
		fn.wFilterHeight= helpers::getRawPointer(m_winColHeight_D);
		fn.wFilterShift = helpers::getRawPointer(m_winShiftIndex_D);
		fn.layerSize    = this->precedingLayer().size();
		fn.matrixWNum   = m_biasPosInBuffer;
		
		n = this->m_winTotalL * this->precedingLayer().size() + this->size();
		thrust::for_each(
		  thrust::make_zip_iterator(
			thrust::make_tuple(this->m_weightBuffer.begin(),
					   thrust::counting_iterator<int>(0))),
		  thrust::make_zip_iterator(
			thrust::make_tuple(this->m_weightBuffer.begin() + n, 
					   thrust::counting_iterator<int>(0) + n)),
		  fn);
		
	    }else{

		// 1-D convolution
		// copy the weights from this->weights() to m_weightBuffer
		thrust::copy(this->weights().begin(),
			     this->weights().begin() + m_weightBufferNum,
			     m_weightBuffer.begin());
	    }
	}}

	// Step1: prepare the data buffer by matrix transformation
	{{
	    helpers::Matrix<TDevice> weightMatrix   (&this->m_weightBuffer,
						     this->precedingLayer().size(),
						     this->m_winTotalL);
	    
	    helpers::Matrix<TDevice> plOutputsMatrix(&this->precedingLayer().outputs(), 
						     this->precedingLayer().size(), 
						     this->curMaxSeqLength() * 
						     this->parallelSequences());

            helpers::Matrix<TDevice> outputsMatrix  (&this->m_conBuffer,                 
						     this->m_winTotalL,                  
						     this->curMaxSeqLength() * 
						     this->parallelSequences());

            outputsMatrix.assignProduct(weightMatrix, true, plOutputsMatrix, false);
	}}

	// Step2: sum the result
	{{
	    internal::ConvolutionCore fn;
	    	    
	    fn.dataBuffer       = helpers::getRawPointer(this->m_conBuffer);
	    //fn.targetBuff       = helpers::getRawPointer(this->outputs());
	    fn.biasWeight       = helpers::getRawPointer(this->m_weightBuffer) + m_biasPosInBuffer;
	    
	    fn.winSizeCum       = helpers::getRawPointer(m_winWidth_Cum_D);
	    fn.winHalfSize      = helpers::getRawPointer(m_winWidth_D);
	    fn.winTapInter      = helpers::getRawPointer(m_winInterval_D);
		
	    fn.curLayerSize     = this->size();
	    fn.winTotalLength   = this->m_winTotalL;

	    fn.patTypes         = helpers::getRawPointer(this->patTypes());
	    fn.paral            = this->precedingLayer().parallelSequences();
	    fn.maxSeqLength     = this->curMaxSeqLength();

	    fn.causal           = this->m_causalFlag;
	    fn.outputTanh       = this->m_outputTanh;
	    int n =this->precedingLayer().curMaxSeqLength();
	    n = n*this->precedingLayer().parallelSequences();
	    n = n*this->size();

	    thrust::for_each(
	     thrust::make_zip_iterator(
		thrust::make_tuple(this->outputs().begin(),
					   thrust::counting_iterator<int>(0))),
	     thrust::make_zip_iterator(
		thrust::make_tuple(this->outputs().begin()+n, 
					   thrust::counting_iterator<int>(0)+n)),
	     fn);

	}}
	
	// dustbin.txt 20170421x02
    }
    
    template <typename TDevice>
    void CNNLayer<TDevice>::computeForwardPass(const int timeStep, const int nnState)
    {
	
	// Step0: prepare at the begining of the sentence
	if (timeStep == 0){{
	    if (m_winHeight_Opt.size()>0){
		if (this->getSaveMemoryFlag())
		    throw std::runtime_error("memory-save mode for 2D conv is unavailable");

		// 2-D convolution		
		// Copy weight from this->weights() to m_weightBuffer
		internal::CNNFilterWeightCopy fn1;
		fn1.weightBuffer   = helpers::getRawPointer(m_weightBuffer);
		fn1.weightCopyInfo = helpers::getRawPointer(m_wCopyInfo_D);
		fn1.filterIndexMap = helpers::getRawPointer(m_weightFilter_map);
		fn1.filterNum      = m_winNumOrignal;
		fn1.preLayerSize   = this->precedingLayer().size();
		fn1.biasPosition   = m_biasPos;
		fn1.biasPositionBuf= m_biasPosInBuffer;
		fn1.reverse        = false;
		
		int n = m_weightNum;
		thrust::for_each(
		  thrust::make_zip_iterator(
			thrust::make_tuple(this->weights().begin(),
					   thrust::counting_iterator<int>(0))),
		  thrust::make_zip_iterator(
			thrust::make_tuple(this->weights().begin() + n, 
					   thrust::counting_iterator<int>(0) + n)),
		  fn1);
		
		// duplicate weights inside m_weightBuffer
		internal::CNNFilterDuplicate fn;
		fn.weight       = helpers::getRawPointer(m_weightBuffer);
		fn.wColIndex    = helpers::getRawPointer(m_winColIndex_D);
		fn.wRowIndex    = helpers::getRawPointer(m_winRowIndex_D);
		fn.wFilterHeight= helpers::getRawPointer(m_winColHeight_D);
		fn.wFilterShift = helpers::getRawPointer(m_winShiftIndex_D);
		fn.layerSize    = this->precedingLayer().size();
		fn.matrixWNum   = m_biasPosInBuffer;
		
		n = this->m_winTotalL * this->precedingLayer().size() + this->size();
		thrust::for_each(
		  thrust::make_zip_iterator(
			thrust::make_tuple(this->m_weightBuffer.begin(),
					   thrust::counting_iterator<int>(0))),
		  thrust::make_zip_iterator(
			thrust::make_tuple(this->m_weightBuffer.begin() + n, 
					   thrust::counting_iterator<int>(0) + n)),
		  fn);
		
	    }else{

		// 1-D convolution
		// copy the weights from this->weights() to m_weightBuffer
		thrust::copy(this->weights().begin(),
			     this->weights().begin() + m_weightBufferNum,
			     m_weightBuffer.begin());
	    }

	    // initialize the data buffer
	    thrust::fill(m_conBuffer.begin(), m_conBuffer.end(), 0.0);
	}}


	int st = timeStep * this->parallelSequences();
	int et = (timeStep + 1) * this->parallelSequences();
	int shiftIn = this->precedingLayer().outputBufPtrBias(st, nnState);

	// Step1-2: matrix transformation and data summation
	if (this->getSaveMemoryFlag()){
	    // memory save mode for wavenet
	    
	    // Step1. matrix transformation
	    // receptive filed size
	    int recField = m_winInterval_H[0];
	    // absolute address in the conv buffer
	    int bufAddr  = (timeStep % (recField+1)) * this->parallelSequences() * m_winTotalL;
	    // This transofmration will transform the input data 
	    helpers::Matrix<TDevice> weightMatrix   (&this->m_weightBuffer,
						     this->precedingLayer().size(),
						     this->m_winTotalL);
	    helpers::Matrix<TDevice> plOutputsMatrix(&this->precedingLayer().outputs(), 
						     this->precedingLayer().size(), 
						     this->parallelSequences(),
						     st * this->precedingLayer().size() - shiftIn);
            helpers::Matrix<TDevice> outputsMatrix  (&this->m_conBuffer,                 
						     this->m_winTotalL,                   
						     this->parallelSequences(),
						     bufAddr);
            outputsMatrix.assignProduct(weightMatrix, true, plOutputsMatrix, false);
	    
	    // Step2. data summation
	    internal::ConvolutionCoreMemSaveMode fn;
	    	    
	    fn.dataBuffer       = helpers::getRawPointer(this->m_conBuffer);
	    fn.biasWeight       = helpers::getRawPointer(this->m_weightBuffer) + m_biasPosInBuffer;

	    fn.recFieldSize     = recField;
	    fn.curLayerSize     = this->size();
	    fn.winTotalLength   = this->m_winTotalL;

	    fn.timeStep         = timeStep;
	    fn.outputTanh       = this->m_outputTanh;
	    fn.parallel         = this->precedingLayer().parallelSequences();
	    fn.patTypes         = helpers::getRawPointer(this->patTypes());

	    int numEle = this->parallelSequences() * this->size();
	    thrust::for_each(
	     thrust::make_zip_iterator(
		thrust::make_tuple(this->outputs().begin(),
				   thrust::counting_iterator<int>(0))),
	     thrust::make_zip_iterator(
		thrust::make_tuple(this->outputs().begin() + numEle,
				   thrust::counting_iterator<int>(0) + numEle)),
	     fn);
	    
	    
	}else{
	    
	    // normal mode
	    // Step1: prepare the data buffer by matrix transformation
	    {{
	    helpers::Matrix<TDevice> weightMatrix   (&this->m_weightBuffer,
						     this->precedingLayer().size(),
						     this->m_winTotalL);
	    
	    helpers::Matrix<TDevice> plOutputsMatrix(&this->precedingLayer().outputs(), 
						     this->precedingLayer().size(), 
						     this->parallelSequences(),
						     st * this->precedingLayer().size() - shiftIn);

            helpers::Matrix<TDevice> outputsMatrix  (&this->m_conBuffer,                 
						     this->m_winTotalL,                   
						     this->parallelSequences(),
						     st * this->m_winTotalL);

            outputsMatrix.assignProduct(weightMatrix, true, plOutputsMatrix, false);
	    }}

	    // Step2: sum the result
	    {{
	    internal::ConvolutionCore fn;
	    	    
	    fn.dataBuffer       = helpers::getRawPointer(this->m_conBuffer);
	    //fn.targetBuff       = helpers::getRawPointer(this->outputs());
	    fn.biasWeight       = helpers::getRawPointer(this->m_weightBuffer) + m_biasPosInBuffer;
	    
	    fn.winSizeCum       = helpers::getRawPointer(m_winWidth_Cum_D);
	    fn.winHalfSize      = helpers::getRawPointer(m_winWidth_D);
	    fn.winTapInter      = helpers::getRawPointer(m_winInterval_D);
		
	    fn.curLayerSize     = this->size();
	    fn.winTotalLength   = this->m_winTotalL;

	    fn.patTypes         = helpers::getRawPointer(this->patTypes());
	    fn.paral            = this->precedingLayer().parallelSequences();
	    fn.maxSeqLength     = this->curMaxSeqLength();

	    fn.causal           = this->m_causalFlag;
	    fn.outputTanh       = this->m_outputTanh;
	    int n =this->precedingLayer().curMaxSeqLength();
	    n = n*this->precedingLayer().parallelSequences();
	    n = n*this->size();

	    thrust::for_each(
	     thrust::make_zip_iterator(
		thrust::make_tuple(this->outputs().begin() + st * this->size(),
				   thrust::counting_iterator<int>(0)+ st * this->size())),
	     thrust::make_zip_iterator(
		thrust::make_tuple(this->outputs().begin() + et * this->size(), 
				   thrust::counting_iterator<int>(0)+ et * this->size())),
	     fn);

	    }}
	}
    }
    
    template <typename TDevice>
    void CNNLayer<TDevice>::computeBackwardPass(const int nnState)
    {
	if (this->getSaveMemoryFlag())
	    throw std::runtime_error("Memory save mode should be turned off");
	
	// Step1: Pass throught the nonlinear function
	if (m_outputTanh){{
            internal::ComputeDeltaFn fn;
            int n = this->curMaxSeqLength() * this->parallelSequences() * this->size();
            thrust::for_each(
               thrust::make_zip_iterator(
		  thrust::make_tuple(this->outputErrors().begin(),   this->outputs().begin())),
	       thrust::make_zip_iterator(
		  thrust::make_tuple(this->outputErrors().begin()+n, this->outputs().begin()+n)),
                fn);
	}}

	thrust::fill(this->precedingLayer().outputErrors().begin(),
		     this->precedingLayer().outputErrors().end(), 0.0);	

	thrust::fill(m_conBuffer.begin(),
		     m_conBuffer.begin() + this->m_winTotalL * this->curMaxSeqLength() * 
		     this->parallelSequences(), 0.0);
	
	// Step2: propagate the gradient
	{{
	    internal::ConvolutionCoreGra fn;
	    	    
	    fn.dataBuffer       = helpers::getRawPointer(this->m_conBuffer);
	    //fn.GradBuffer       = helpers::getRawPointer(this->outputErrors());

	    fn.winSizeCum       = helpers::getRawPointer(m_winWidth_Cum_D);
	    fn.winHalfSize      = helpers::getRawPointer(m_winWidth_D);
	    fn.winTapInter      = helpers::getRawPointer(m_winInterval_D);
	    
	    fn.curLayerSize     = this->size();
	    fn.winTotalLength   = this->m_winTotalL;

	    fn.patTypes         = helpers::getRawPointer(this->patTypes());
	    fn.paral            = this->precedingLayer().parallelSequences();
	    fn.maxSeqLength     = this->curMaxSeqLength();

	    fn.causal           = this->m_causalFlag;
	    
	    int n =this->precedingLayer().curMaxSeqLength();
	    n = n*this->precedingLayer().parallelSequences();
	    n = n*this->size();

	    thrust::for_each(
	     thrust::make_zip_iterator(
			thrust::make_tuple(this->outputErrors().begin(),
					   thrust::counting_iterator<int>(0))),
	     thrust::make_zip_iterator(
			thrust::make_tuple(this->outputErrors().begin()+n, 
					   thrust::counting_iterator<int>(0)+n)),
	     fn);

	}}

	// Step3: gradient to previous layer
	{{
	    helpers::Matrix<TDevice> weightMatrix   (&this->m_weightBuffer,
						     this->precedingLayer().size(),
						     this->m_winTotalL);

	    helpers::Matrix<TDevice> curErrorMatrix (&this->m_conBuffer,                 
						     this->m_winTotalL,                  
						     this->curMaxSeqLength() * 
						     this->parallelSequences());

            helpers::Matrix<TDevice> preErrorMatrix (&this->precedingLayer().outputErrors(),
						     this->precedingLayer().size(),
						     this->curMaxSeqLength() * 
						     this->parallelSequences());

            preErrorMatrix.assignProduct(weightMatrix, false, curErrorMatrix, false);
	}}

	// Step4: gradient to the weight
	{{
	    // use m_weightBuffer as the buffer for gradients
	    helpers::Matrix<TDevice> weightError   (&this->m_weightBuffer,
						     this->precedingLayer().size(),
						     this->m_winTotalL);

	    helpers::Matrix<TDevice> curErrorMatrix (&this->m_conBuffer,                 
						     this->m_winTotalL,                  
						     this->curMaxSeqLength() * 
						     this->parallelSequences());

            helpers::Matrix<TDevice> preOutputMatrix (&this->precedingLayer().outputs(),
						      this->precedingLayer().size(),
						      this->curMaxSeqLength() * 
						      this->parallelSequences());

            weightError.assignProduct(preOutputMatrix, false, curErrorMatrix, true);
	}}

	// Step5: gradient to the bias part
	{{
	    // Borrow the m_conBuffer as one vector [1, 1, 1, 1, 1]
	    thrust::fill(m_conBuffer.begin(),
			 m_conBuffer.begin() + this->curMaxSeqLength() * this->parallelSequences(),
			 1.0);
	    
	    helpers::Matrix<TDevice> biasError   (&this->m_weightBuffer, 1, this->size(),
						  m_biasPosInBuffer);

	    helpers::Matrix<TDevice> curErrorMatrix (&this->outputErrors(),                 
						     this->size(),                  
						     this->curMaxSeqLength() * 
						     this->parallelSequences());

            helpers::Matrix<TDevice> onesVec (&this->m_conBuffer, 1,
					      this->curMaxSeqLength() * this->parallelSequences());

            biasError.assignProduct(onesVec, false, curErrorMatrix, true);
	    
	}}

	// Step6: merge the gradient
	{{
	    
	    if (m_winHeight_Opt.size()>0){
		
		// 2-D convolution

		// merge the gradients in the buffer
		internal::CNNFilterGradientMerge fn;
		fn.weightGra    = helpers::getRawPointer(this->m_weightBuffer);
		fn.wColIndex    = helpers::getRawPointer(m_winColIndex_D);
		fn.wRowIndex    = helpers::getRawPointer(m_winRowIndex_D);
		fn.wFilterHeight= helpers::getRawPointer(m_winColHeight_D);
		fn.wFilterShift = helpers::getRawPointer(m_winShiftIndex_D);

		fn.wFilterWidth        = helpers::getRawPointer(m_winWidthCol_D);
		fn.wFilterShiftNum     = helpers::getRawPointer(m_winShiftNum_D);
		fn.wFilterShiftNumBias = helpers::getRawPointer(m_winShiftRevId_D);
		
		fn.layerSize    = this->precedingLayer().size();
		fn.matrixWNum   = this->m_winTotalL * this->precedingLayer().size();
		
		int n = this->m_winTotalL * this->precedingLayer().size() + this->size();
		thrust::for_each(
		  thrust::make_zip_iterator(
			thrust::make_tuple(this->m_weightBuffer.begin(),
					   thrust::counting_iterator<int>(0))),
		  thrust::make_zip_iterator(
			thrust::make_tuple(this->m_weightBuffer.begin() + n, 
					   thrust::counting_iterator<int>(0) + n)),
		  fn);

		// copy the gradients to the weightsUpdate buffer
		internal::CNNFilterWeightCopy fn1;
		fn1.weightBuffer   = helpers::getRawPointer(m_weightBuffer);
		fn1.weightCopyInfo = helpers::getRawPointer(m_wCopyInfo_D);
		fn1.filterIndexMap = helpers::getRawPointer(m_weightFilter_map);
		fn1.filterNum      = m_winNumOrignal;
		fn1.preLayerSize   = this->precedingLayer().size();
		fn1.biasPosition   = m_biasPos;
		fn1.biasPositionBuf= m_biasPosInBuffer;
		fn1.reverse        = true;
		
		n = m_weightNum;
		thrust::for_each(
		  thrust::make_zip_iterator(
			thrust::make_tuple(this->_weightUpdates().begin(),
					   thrust::counting_iterator<int>(0))),
		  thrust::make_zip_iterator(
			thrust::make_tuple(this->_weightUpdates().begin() + n, 
					   thrust::counting_iterator<int>(0) + n)),
		  fn1);
		
		
	    }else{

		// 1-D convolution
		thrust::copy(this->m_weightBuffer.begin(),
			     this->m_weightBuffer.begin() + m_weightBufferNum,
			     this->_weightUpdates().begin());
	    }

	}}
	
	// dustbin.txt 20170421x03
    }


    template <typename TDevice>
    void CNNLayer<TDevice>::computeBackwardPass(const int timeStep, const int nnState)
    {
	throw std::runtime_error("BackwardPas(timeStep, nnState) not implemented for CNN");
    }

    
    template <typename TDevice>
    void CNNLayer<TDevice>::loadSequences(const data_sets::DataSetFraction &fraction,
					  const int nnState)
    {
	// load the sequences for TrainableLayers
	TrainableLayer<TDevice>::loadSequences(fraction, nnState);
	
	// 
    }
    
    template <typename TDevice>
    const std::string& CNNLayer<TDevice>::type() const
    {
	static const std::string m("cnn");
	return m;
    }

    template <typename TDevice>
    void CNNLayer<TDevice>::exportLayer(
	const helpers::JsonValue     &layersArray, 
	const helpers::JsonAllocator &allocator) const
    {
        TrainableLayer<TDevice>::exportLayer(layersArray, allocator);
	(*layersArray)[layersArray->Size() - 1].AddMember("window_width",
							  m_winWidth_Opt.c_str(),
							  allocator);
        (*layersArray)[layersArray->Size() - 1].AddMember("window_tap_interval",
							  m_winInterval_Opt.c_str(),
							  allocator);
        (*layersArray)[layersArray->Size() - 1].AddMember("window_height",
							  m_winHeight_Opt.c_str(),
							  allocator);
        (*layersArray)[layersArray->Size() - 1].AddMember("window_stride",
							  m_winStride_Opt.c_str(),
							  allocator);
	(*layersArray)[layersArray->Size() - 1].AddMember("causal", m_causalFlag,
							  allocator);
	(*layersArray)[layersArray->Size() - 1].AddMember("tanhoutput", m_outputTanh,
							  allocator);	
    }


    template <typename TDevice>
    void CNNLayer<TDevice>::reduceOutputBuffer()
    {
	// check whether this is the CNN in wavenet
	// 1. causal filter
	// 2. 1-D CNN
	// 3. filter width is 3 (although 2 columns of weights are used by causal CNN)
	// 4. filter interval is the same
	if (m_causalFlag && m_1DCNNOnly && (m_winTotalL == (this->size() * 3))){
	    // check the filter interval
	    int recepField = m_winInterval_H[0];
	    for (int i = 0; i < m_winInterval_H.size(); i++)
		if (recepField != m_winInterval_H[i])
		    return;
	    
	    // save the intermediate buffer
	    m_conBuffer.resize(this->parallelSequences() * m_winTotalL * (recepField + 1), 0);
	    m_conBuffer.shrink_to_fit();
	    
	    // save the output buffer size
	    this->resizeOutputBuffer(this->parallelSequences() * this->size());
	    printf("\t[mem saved]");
	    this->setSaveMemoryFlag(true);
	}
    }

    template <typename TDevice>
    int CNNLayer<TDevice>::outputBufPtrBias(const int timeStepTimesParallel, const int nnState)
    {
	if (this->getSaveMemoryFlag()){
	    return timeStepTimesParallel * this->size();
	}else{
	    return 0;
	}
    }

    template <typename TDevice>
    void CNNLayer<TDevice>::clearAllBuffers()
    {
	this->clearOutputBuffer();
	this->__clearLocalMem();
    }

    template <typename TDevice>
    void CNNLayer<TDevice>::resizeAllBuffers(const int timeLength)
    {
	this->resizeOutputBuffer(timeLength * this->parallelSequences() * this->size());
	this->__allocateLocalMem();
    }

    template class CNNLayer<Gpu>;
    template class CNNLayer<Cpu>;
}


// # B1
/*
	    if (accurate){
		
	    }else{
		
		Cpu::int_vector width2(thisLayerSize,  0);
		int cnt = 0;
		for (int i = 0; i < width.size(); i++){
		    int shiftSteps = getShiftSteps(preLayerSize, height[i], stride[i]);
		    for (int j = 0; j < shiftSteps; j++){
			if (cnt >= thisLayerSize)
			    throw std::runtime_error("CNN size is smaller than expected");
			width2[cnt]  = width[i];
			cnt++;
		    }
		}
		
		if (cnt != thisLayerSize){
		    printf("CNN size %d is unequal to expected dimension %d", thisLayerSize, cnt);
		    throw std::runtime_error("Please check layer size or CNN configuration");
		}
		width = width2;
		return ((int)std::ceil(getWinTotalLength(width) *
					   (float)preLayerSize /
					   (float)thisLayerSize) + 1);
					   }*/
	    
