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

#include "OperationLayer.hpp"

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
#include <boost/foreach.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>
#include <vector>
#include <stdexcept>
#include <cmath>

#include "../Configuration.hpp"


// Options for feature extraction functions
// for sentence-level mode
// (extract one feature vector from an input sequence)
#define NN_OPE_LAST_SHOT_TURNOFF 0  // no last shot

#define NN_OPE_LAST_SHOT_MODE1 1  // use the last shot of sentence end -> first frame 
#define NN_OPE_LAST_SHOT_MODE2 2  // use the last shot of sentence end, repeat across frames
#define NN_OPE_LAST_SHOT_MODE5 5  // use the first/last shot of sentence -> first frame
#define NN_OPE_LAST_SHOT_MODE6 6  // use the first/last shot of sentence, repeat across frames

#define NN_OPE_LAST_SHOT_MODE9 9  // take the average over utterance, and repeat across frames

// for segments mode
// (extract one feature per segment, which requires auxilliary segment boundaries as input)
#define NN_OPE_LAST_SHOT_MODE3 3  // use the last shot of segments
#define NN_OPE_LAST_SHOT_MODE4 4  // use the last shot of segments, repeat across frames
#define NN_OPE_LAST_SHOT_MODE7 7  // use the first/last shot of segments
#define NN_OPE_LAST_SHOT_MODE8 8  // use the first/last shot of segments, repeat across frames

#define NN_OPE_LAST_SHOT_MODE10 10 // use the first shot of segments (causal dependency)

// for sine wave generator
// (note: this function has been moved to signalGenLayer)
#define NN_OPE_FREGEN_F0_LF0  1   // input is mel-logF0 = 1127 * log (1 + F0 / 700)
#define NN_OPE_FREGEN_F0_QF0  2   // input is qF0

#define PI_DEFINITION 3.141592654f

// for F02UV
#define NN_OPE_F02UV_THRESHOLD 10.0 // F0 > 10Hz, voiced

// CUDA functions
namespace internal{
namespace {

    // Generation uniform distribution noise
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

    // Element-wise operation function, which does
    // 1. scale the input datum
    // 2. concatenate the output vector with noise data, if provided
    struct fillOutputVec
    {
	int curLayerSize;
	int preLayerSize;
	int noiseDim;
	int noiseRepeat;

	bool flagForward;
	
	real_t *preOutput;
	long int preShift;
	
	real_t *noiseData;

	real_t *weights;
	real_t *shifts;
	real_t *stopGrad;
	real_t *binarize;
	
	const char *patTypes;
	
	__host__ __device__ void operator() (const thrust::tuple<real_t&, int> &t) const
	{
	    int outputIdx = t.get<1>();
	    int dimIdx    = outputIdx % curLayerSize;
	    int timeIdx   = outputIdx / curLayerSize;

	    real_t tmp_output;
	    // Assume one vector of one frame is formulated as [data, noise]
	    if (dimIdx < preLayerSize){

		/* Before stopGrad is added

		// for the data part
		if ((patTypes != NULL && patTypes[timeIdx] == PATTYPE_NONE)){
		    // none valid time splot
		    t.get<0>() = 0.0;
		    
		}else if (weights[dimIdx] < 0){
		    // Dirty code: whey setZero/scale is below 0, it is used to stop gradients
		    
		    //  forward: propagate as usual
		    //  backward: kill the gradients
		    if (flagForward)
			t.get<0>() = preOutput[timeIdx * preLayerSize + dimIdx - preShift]
			    * (-1.0 * weights[dimIdx]) + shifts[dimIdx];
		    else
			t.get<0>() = 0.0;
		    
		}else{
		    // normal node, where scale weight is used upon both forward and backward
		    // note: the shift will not affect gradient
		    if (flagForward)
			t.get<0>() = preOutput[timeIdx * preLayerSize + dimIdx - preShift]
			    * weights[dimIdx] + shifts[dimIdx];
		    else
			t.get<0>() = preOutput[timeIdx * preLayerSize + dimIdx - preShift]
			    * weights[dimIdx];
			    }*/
		// for the data part
		
		if ((patTypes != NULL && patTypes[timeIdx] == PATTYPE_NONE)){
		    // none valid time splot
		    t.get<0>() = 0.0;
		    
		}else if (flagForward){

		    // forward propagation
		    tmp_output = preOutput[timeIdx * preLayerSize + dimIdx - preShift]
			* weights[dimIdx] + shifts[dimIdx];
		    if (binarize[dimIdx] > 0.0){
			t.get<0>() = (tmp_output > 0.0?1.0:-1.0);
		    }else{
			t.get<0>() = tmp_output;
		    }
		}else{
		    // backward propagation

		    if (binarize[dimIdx] > 0.0 || stopGrad[dimIdx] > 0.0 )
			// if stop gradient mode is on, or this dimension is binarized
			t.get<0>() = 0.0;
		    else
			// normal mode
			t.get<0>() = preOutput[timeIdx * preLayerSize + dimIdx - preShift]
			    * weights[dimIdx];
		}
		
	    }else{
		// for the noise part
		
		// repeat the noise across time
		if (noiseRepeat == NN_OPERATOR_LAYER_NOISE_TIMEREPEAT) 
		    t.get<0>() = noiseData[(dimIdx - preLayerSize)];
		// repeat the noise across dimension
		else if (noiseRepeat == NN_OPERATOR_LAYER_NOISE_DIMREPEAT) 
		    t.get<0>() = noiseData[timeIdx * noiseDim];
		// normal case
		else             
		    t.get<0>() = noiseData[timeIdx * noiseDim + (dimIdx - preLayerSize)];
	    }	    
	}
    };

    // Inverted Dropout function
    // Note: this inverted dropout function is applied on the output of the previous layer
    // For dropout before the non-linear activation function, it is only implemented for
    // feedforward layers (FeedForwardLayer.cu)
    struct invertedDropout
    {
	int     curLayerSize;
	real_t  threshold;
	real_t *noise;
	const char *patTypes;
	
	__host__ __device__ void operator() (const thrust::tuple<real_t&, real_t&, int> &t) const
	{
	    int outputIdx = t.get<2>();
	    int timeIdx   = outputIdx / curLayerSize;
	    
	    if ((patTypes != NULL && patTypes[timeIdx] == PATTYPE_NONE)){
		// none valid time splot
		t.get<0>() = 0.0;	    
	    }else{
		// drop out
		if (noise[outputIdx] > threshold)
		    t.get<0>() = t.get<1>() / (1-threshold);
		else
		    t.get<0>() = 0.0;
	    }
	}
    };

    // Change time resolution
    // Note: the input sequence may not have the same length as output sequence due to the 
    //  difference of time resolution
    // For upsampling:   simply duplicate 
    // For downsampling: simply extract one datum
    struct timeResolutionChange
    {
	int         inputRes;   // time resolution of previous layer
	int         outputRes;  // time resolution of this layer
	int         layerSize;  // layer size
	int         parallel;   // parallel number
	
	real_t     *sourceData;
	const char *patTypes;

	// From 1 : T
	__host__ __device__ void operator() (const thrust::tuple<real_t&, int> &t) const
	{
	    int outputIdx = t.get<1>();
	    int dimIdx    = outputIdx % layerSize;  // dimension index
	    int timeIdx   = outputIdx / layerSize;  // time index (regardless of parallel)
	    int BlockIdx  = timeIdx / parallel;     // time index (considering parallel mode)
	    int BlockInIdx= timeIdx % parallel;     // index within a parallel block

	    int fraction  = 1;  // change ratio

	    if (patTypes[timeIdx] == PATTYPE_NONE){
		t.get<0>() = 0;
		return;
	    }		
	    if (outputRes >= inputRes){
		// down sampling
		fraction = outputRes / inputRes;
		t.get<0>() = sourceData[((BlockIdx * fraction) * parallel + BlockInIdx) *
					layerSize + dimIdx];
	    }else{
		// up sampling
		fraction = inputRes / outputRes;
		t.get<0>() = sourceData[((BlockIdx / fraction) * parallel + BlockInIdx) *
					layerSize + dimIdx];
	    }
	    
	}
    };

    // Change time resolution (for gradient propagation)
    // For upsampling: accumulate the gradients over the duplicated time steps
    // For downsampling: only propagation the gradient to one datum that has been being used
    struct timeResolutionChangeGrad
    {
	int         inputRes;   // time resolution of previous layer
	int         outputRes;  // time resolution of this layer
	int         layerSize;  // layer size
	int         parallel;   // parallel number
	
	real_t     *sourceData; // source Data is the gradients of this layer
	const char *patTypes;   // previous layer's patTypes

	// From 1 : T (of the previous layer)
	__host__ __device__ void operator() (const thrust::tuple<real_t&, int> &t) const
	{
	    int outputIdx = t.get<1>();
	    int dimIdx    = outputIdx % layerSize;  // dimension index
	    int timeIdx   = outputIdx / layerSize;  // time index (regardless of parallel)
	    int BlockIdx  = timeIdx / parallel;     // time index (considering parallel mode)
	    int BlockInIdx= timeIdx % parallel;     // index within a parallel block

	    int fraction  = 1;  // change ratio

	    if (patTypes[timeIdx] == PATTYPE_NONE){
		t.get<0>() = 0;
		return;
	    }		
	    
	    if (outputRes >= inputRes){
		// down sampling
		fraction = outputRes / inputRes;
		if (BlockIdx % fraction == 0){
		    t.get<0>() = sourceData[((BlockIdx/fraction) * parallel + BlockInIdx) *
					    layerSize + dimIdx];
		}else{
		    t.get<0>() = 0;
		}
	    }else{
		// up sampling
		fraction = inputRes / outputRes;
		t.get<0>() = 0;
		for (int i = 0; i<fraction; i++)
		    t.get<0>() += sourceData[((BlockIdx * fraction+i) * parallel + BlockInIdx) *
					     layerSize + dimIdx];
	    }
	    
	}
    };
    
    // #1

    // Divide the sequence into a sequence of blocks (block length = resolution)
    // Duplicate the last frame of each block to all the rest frames within that block
    struct outDuplicationOperation
    {
	int featureDim;
	int resolution;
	int maxTimeLength;
	int parall;
	
	real_t *dataMatrix;
	const char   *patTypes;
	
	__host__ __device__ void operator() (const thrust::tuple<const real_t&, int> &t) const
	{
	    int timeIdx  = t.get<1>() / featureDim;
	    int dimIdx   = t.get<1>() % featureDim;
	    int timeRel  = timeIdx    / parall;
	    int paraPos  = timeIdx    % parall;
	    
	    int blockIdx = timeRel    / resolution;

	    int copyIdx  = (blockIdx + 1) * resolution * parall - parall + paraPos;
	    if (((timeRel % resolution) == (resolution -1)) || patTypes[timeIdx] == PATTYPE_LAST)
		// either this is the last point of one block, or the end point of sentence
		return;
	    else{
		// if copyIdx is larger than sentence length, move back to end point of sentence
		while(patTypes[copyIdx]==PATTYPE_NONE){
		    copyIdx -= parall;
		}
		dataMatrix[t.get<1>()] = dataMatrix[copyIdx * featureDim + dimIdx];
	    }
	    //if (copyIdx < maxTimeLength)
	    //dataMatrix[t.get<1>()] = dataMatrix[copyIdx * featureDim + dimIdx];
	    //else
	    //dataMatrix[t.get<1>()] = dataMatrix[(maxTimeLength - parall + paraPos) * featureDim
	    //+ dimIdx];
	    
	}
    };

    // Gradient function for outDuplicationOperation
    struct outDuplicationGradOperation
    {
	int featureDim;
	int resolution;
	int maxTimeLength;
	int parall;
	
	real_t *dataMatrix;
	const char   *patTypes;
	
	__host__ __device__ void operator() (const thrust::tuple<const real_t&, int> &t) const
	{
	    int timeIdx  = t.get<1>() / featureDim;
	    int dimIdx   = t.get<1>() % featureDim;
	    int timeRel  = timeIdx    / parall;	    

	    if (patTypes[timeIdx] == PATTYPE_NONE){
		return;
	    }else if ((timeRel % resolution) == (resolution -1)){
		// accumulate the gradients
		int idx;
		for (int i = 1; i < resolution; i++){
		    idx = (timeIdx - i * parall) * featureDim + dimIdx;
		    dataMatrix[t.get<1>()] += dataMatrix[idx];
		    dataMatrix[idx] = 0;
		}
	    }else if (patTypes[timeIdx] == PATTYPE_LAST){
		int idx;
		for (int i = 1; i <= (timeRel % resolution); i++){
		    idx = (timeIdx - i * parall) * featureDim + dimIdx;
		    dataMatrix[t.get<1>()] += dataMatrix[idx];
		    dataMatrix[idx] = 0;
		}
	    }else{
		return;
	    }
	}
    };

    // LastShot mode: extract feature vector(s) for the whole sequence
    // Used by 
    // NN_OPE_LAST_SHOT_MODE1 1  // use the last shot of sentence end -> first frame 
    // NN_OPE_LAST_SHOT_MODE2 2  // use the last shot of sentence end, repeat across frames
    // NN_OPE_LAST_SHOT_MODE5 5  // use the first/last shot of sentence -> first frame
    // NN_OPE_LAST_SHOT_MODE6 6  // use the first/last shot of sentence, repeat across frames
    struct lastShotForward
    {
	int     featureDim;
	int     paralSeqNm;
	int     lastShotOp;
	
	int    *seqLengthD;
	real_t *sourceData;
	
	const char *patTypes;
	__host__ __device__ void operator() (const thrust::tuple<real_t&, int> &t) const
	{
	    int outputIdx = t.get<1>();
	    int dimIdx    = outputIdx % featureDim;
	    int timeIdx   = outputIdx / featureDim;

	    if (patTypes[timeIdx] == PATTYPE_NONE){
		t.get<0>() = 0.0;
		return;
	    }
	    
	    int paralBlk  = timeIdx / paralSeqNm;
	    int sentIdx   = timeIdx % paralSeqNm;
	    int seqLength = seqLengthD[sentIdx];

	    if (lastShotOp == NN_OPE_LAST_SHOT_MODE1 || lastShotOp == NN_OPE_LAST_SHOT_MODE2){
		
		// uni-directional
		if (lastShotOp == NN_OPE_LAST_SHOT_MODE1){
		    // only copy the last timestep to the first timestep
		    if (paralBlk == 0)
			t.get<0>() = sourceData[((seqLength-1) * paralSeqNm + sentIdx) * featureDim
						+ dimIdx];
		    else
			t.get<0>() = 0.0;
		}else{
		    // copy the last timestep to all timesteps
		    t.get<0>() = sourceData[((seqLength-1) * paralSeqNm + sentIdx) * featureDim
					    + dimIdx];
		}
	    }else if(lastShotOp == NN_OPE_LAST_SHOT_MODE5 || lastShotOp == NN_OPE_LAST_SHOT_MODE6){

		// bi-directional
		if (lastShotOp == NN_OPE_LAST_SHOT_MODE5){
		    
		    // always copy to the first block
		    if (paralBlk == 0){
			if (dimIdx < featureDim / 2){
			    // copy from the forward direction (the last frame)
			    t.get<0>() = sourceData[((seqLength-1) * paralSeqNm + sentIdx) *
						    featureDim + dimIdx];
			}else{
			    // copy from the backward direction (the first frame)
			    t.get<0>() = sourceData[sentIdx * featureDim + dimIdx];
			}
		    }else
			t.get<0>() = 0.0;
		}else{
		    // copy the last timestep to all timesteps
		    if (dimIdx < featureDim / 2){
			// copy from the forward direction (the last frame)
			t.get<0>() = sourceData[((seqLength-1) * paralSeqNm + sentIdx) *
						featureDim + dimIdx];
		    }else{
			// copy from the backward direction (the first frame)
			t.get<0>() = sourceData[sentIdx * featureDim + dimIdx];			
		    }
		}
	    }
	}
	
    };

    // LastShot mode: extract feature vector(s) for the segment
    // Note: the segBoundary data different for different SHOT_MODE
    
    // Used by MODE3 MODE4 MODE
    // NN_OPE_LAST_SHOT_MODE3 3  // use the last shot of segments
    // NN_OPE_LAST_SHOT_MODE4 4  // use the last shot of segments, repeat across frames
    // NN_OPE_LAST_SHOT_MODE7 7  // use the first/last shot of segments
    // NN_OPE_LAST_SHOT_MODE8 8  // use the first/last shot of segments, repeat across frames
    // NN_OPE_LAST_SHOT_MODE10   // use the first/last shot of segments, repeat across frames
    struct lastShotForwardSegBoundary
    {
	int     featureDim;
	int     paralSeqNm;
	int     lastShotOp;
	
	int    *segBoundary;
	real_t *sourceData;
	
	const char *patTypes;
	__host__ __device__ void operator() (const thrust::tuple<real_t&, int> &t) const
	{
	    int outputIdx = t.get<1>();
	    int dimIdx    = outputIdx % featureDim;
	    int timeIdx   = outputIdx / featureDim;

	    if (patTypes[timeIdx] == PATTYPE_NONE){
		t.get<0>() = 0.0;
		return;
	    }
	    
	    //int paralBlk  = timeIdx / paralSeqNm;
	    int sentIdx   = timeIdx % paralSeqNm;

	    int boundary  = -1;
	    if ((lastShotOp == NN_OPE_LAST_SHOT_MODE7 || lastShotOp == NN_OPE_LAST_SHOT_MODE8) &&
		(dimIdx >= featureDim / 2)){
		boundary = segBoundary[timeIdx * 2 + 1];
	    }else{
		boundary = segBoundary[timeIdx * 2];
	    }

	    if (boundary < 0){
		t.get<0>() = 0.0;
	    }else{
		t.get<0>() = sourceData[(boundary * paralSeqNm + sentIdx) * featureDim + dimIdx];
	    }
	}
    };

    // Grad for Mode3 and Mode7
    struct lastShotForwardSegBoundaryGradMode3
    {
	int     featureDim;
	int     paralSeqNm;
	int     lastShotOp;
	
	int    *segBoundary;
	real_t *targetData;
	
	const char *patTypes;
	__host__ __device__ void operator() (const thrust::tuple<real_t&, int> &t) const
	{
	    int outputIdx = t.get<1>();
	    int dimIdx    = outputIdx % featureDim;
	    int timeIdx   = outputIdx / featureDim;

	    if (patTypes[timeIdx] == PATTYPE_NONE){
		t.get<0>() = 0.0;
		return;
	    }
	    
	    //int paralBlk  = timeIdx / paralSeqNm;
	    int sentIdx   = timeIdx % paralSeqNm;
	    int boundary  = -1;
	    
	    if ((lastShotOp == NN_OPE_LAST_SHOT_MODE7) && (dimIdx >= featureDim /2)){
		boundary  = segBoundary[2 * timeIdx + 1];
	    }else{
		boundary  = segBoundary[2 * timeIdx];
	    }
	    
	    if (boundary < 0){
		// not boundary
	    }else{
		targetData[(boundary * paralSeqNm + sentIdx) * featureDim + dimIdx] = t.get<0>();
	    }
	}
    };

    // Grad for Mode4 and Mode8 Mode10
    // becaue they need to to accumulate the gradients from the duplicated time steps
    struct lastShotForwardSegBoundaryGradMode4
    {
	int     featureDim;
	int     paralSeqNm;
	int     lastShotOp;
	
	int    *segBoundary;
	real_t *sourceData;
	
	const char *patTypes;
	__host__ __device__ void operator() (const thrust::tuple<real_t&, int> &t) const
	{
	    int outputIdx = t.get<1>();
	    int dimIdx    = outputIdx % featureDim;
	    int timeIdx   = outputIdx / featureDim;

	    if (patTypes[timeIdx] == PATTYPE_NONE){
		t.get<0>() = 0.0;
		return;
	    }
	    
	    int paralBlk  = timeIdx / paralSeqNm;
	    int sentIdx   = timeIdx % paralSeqNm;
	    int boundary  = -1;
	    
	    if ((lastShotOp == NN_OPE_LAST_SHOT_MODE8) && (dimIdx >= featureDim /2)){
		boundary  = segBoundary[2 * timeIdx + 1];
	    }else{
		boundary  = segBoundary[2 * timeIdx];
	    }

	    t.get<0>() = 0.0;
	    // Decide whether this is boundary place or end of sentence
	    if (paralBlk == boundary){

		if ((lastShotOp == NN_OPE_LAST_SHOT_MODE8) && (dimIdx >= featureDim /2)){
		    // accumulate in the backward direction until the segmental boundary
		    for (int i = 0; ;i++){
			// if this is the begining of the sentence or another segment 
			if (patTypes[timeIdx + i * paralSeqNm] == PATTYPE_NONE ||
			    segBoundary[(timeIdx + i * paralSeqNm) * 2 + 1] != boundary){
			    return;
			}
			t.get<0>() += sourceData[(timeIdx + i * paralSeqNm)*featureDim+dimIdx];
		    }
		    
		}else if (lastShotOp == NN_OPE_LAST_SHOT_MODE4){
		    
		    // accumulate in the backward direction until the segmental boundary 
		    for (int i = 0; ;i++){
			// if this is the begining of the sentence or another segment 
			if ((paralBlk - i) < 0 ||
			    segBoundary[(timeIdx - i * paralSeqNm) * 2] != boundary){
			    return;
			}
		    
			t.get<0>() += sourceData[((paralBlk-i)*paralSeqNm+sentIdx)*featureDim
						 + dimIdx];
		    }
		}else if (lastShotOp == NN_OPE_LAST_SHOT_MODE10){
		    
		    // accumulate in the forward direction until the segmental boundary 
		    for (int i = 0; ;i++){

			if (patTypes[timeIdx + i * paralSeqNm] == PATTYPE_NONE ||
			    segBoundary[(timeIdx + i * paralSeqNm) * 2] != boundary){
			    return;
			}
			t.get<0>() += sourceData[(timeIdx + i * paralSeqNm)*featureDim+dimIdx];
		    }
		    
		}
	    }
	}
    };

    // Generate vectors for calculating average over one utterance (as matrix multiplication)
    struct changeOneVec{
	real_t sentL;
	int    parallelNum;
	__host__ __device__ void operator() (const thrust::tuple<real_t&, int> &t) const
	{
	    if (t.get<1>() % parallelNum == (parallelNum - 1)){
		// the element to be changed
		//       A B C D A B C D
		// 0 0 0 1 0 0 0 1 0 0 0          sum As, shift 3
		t.get<0>() = (real_t)1.0/(sentL);
	    }
	}	
    };

    // Duplicate the result to every frame of the utterance
    struct duplicateSentVec
    {
	int     featureDim;
	int     paralSeqNm;
	
	real_t *sourceData;
	
	const char *patTypes;
	__host__ __device__ void operator() (const thrust::tuple<real_t&, int> &t) const
	{
	    int outputIdx = t.get<1>();
	    int dimIdx    = outputIdx % featureDim;
	    int timeIdx   = outputIdx / featureDim;

	    if (patTypes[timeIdx] == PATTYPE_NONE){
		t.get<0>() = 0.0;
		return;
	    }
	    
	    int paralBlk  = timeIdx / paralSeqNm;
	    int sentIdx   = timeIdx % paralSeqNm;

	    if (paralBlk == 0){
		// the first block should have stored the data
		return;
	    }else{
		// source data will be the data array it self
		// copy from the first block
		t.get<0>() = sourceData[sentIdx * featureDim + dimIdx];
	    }
	}
	
    };

    
    // Convert F0 (normaized) into U/V
    //  normed_F0 * std + mean > threshold
    //  normed_F0 > (threshod - mean) / std
    struct normF0ToUV
    {
	real_t F0mean;
	real_t F0std;
	
	__host__ __device__ void operator() (const thrust::tuple<real_t&, real_t&> &t) const
	{
	    if (t.get<0>() > ((NN_OPE_F02UV_THRESHOLD - F0mean) / F0std))
		t.get<1>() = 1.0;
	    else
		t.get<1>() = 0.0;
	}
	
    };

    // Sine signal generation (moved to FeedForwardLayer.cu)
    struct sinWaveGenerator_accum
    {
	int inputRes;
	int outputRes;
	int thislayerSize;
	int prelayerSize;
	
	int freqDim;
	int freqOpt;
	int seqLength;
	
	real_t freqSR;
	real_t qf0Max;
	real_t qf0Min;
	real_t qf0Lev;
	real_t f0M;
	real_t f0S;
	
	int inputShiftTime;
	int outputShiftTime;
	
	int parallel;
	real_t *sourceData;
	real_t *targetData;
	const char *patTypes;
	
	__host__ __device__ void operator() (const thrust::tuple<real_t&, int> &t) const
	{

	    int timeReso  = inputRes / outputRes;

	    //int trueTime  = 0;
	    //int paraBlok  = 0;

	    real_t initPhase = 2.0 * PI_DEFINITION * ((real_t)(t.get<1>())/thislayerSize);
	    real_t spPhase = initPhase;
	    for (int timeIdx = 0; timeIdx < seqLength; timeIdx++){

		if (patTypes[timeIdx] == PATTYPE_NONE)
		    continue;
		
		//trueTime = timeIdx / parallel;
		//paraBlok = timeIdx % parallel;

		// read and denormalize the raw data
		real_t freq = sourceData[(timeIdx/timeReso - inputShiftTime) * prelayerSize
					 + freqDim] * f0S + f0M;
		
		if (freq > 1.0){
		    
		    if (freqOpt == NN_OPE_FREGEN_F0_LF0){
			// LF0 -> F0
			freq = (exp(freq / 1127.0) -1.0 ) * 700.0;
		    }else if (freqOpt == NN_OPE_FREGEN_F0_QF0){
			// QF0 -> LF0
			freq = (freq - 1.0) / (qf0Lev-2.0) * (qf0Max - qf0Min) + qf0Min;
			// LF0 -> F0
			freq = (exp(freq / 1127.0) -1.0 ) * 700.0;
		    }
		    // accumulate the phase
		    spPhase += 2.0 * PI_DEFINITION * freq / freqSR;
		    if (spPhase > 2.0 * PI_DEFINITION)
			spPhase = spPhase - 2.0 * PI_DEFINITION;
		    else if (spPhase < -2.0 * PI_DEFINITION)
			spPhase = spPhase + 2.0 * PI_DEFINITION;
		    
		    targetData[timeIdx * thislayerSize + t.get<1>()] += sin(spPhase);
		    
		}else{
		    spPhase = initPhase;
		}
		
		
	    }
	    
	}
	
    };

    // Generate the positional code
    struct positionalCode
    {
	int featureDim;
	int parallel;
	
	__host__ __device__ void operator() (const thrust::tuple<real_t&, int> &t) const
	{
	    real_t timeIdx  = (real_t)(t.get<1>() / featureDim / parallel);
	    int dimIdx   = t.get<1>() % featureDim;

	    if (dimIdx % 2 == 0)
		t.get<0>() = sin(timeIdx / powf(10000, dimIdx / 2 * 2.0 / featureDim));
	    else
		t.get<0>() = cos(timeIdx / powf(10000, dimIdx / 2 * 2.0 / featureDim));
	    
	}
    };


    // Duplicate the result to every frame of the utterance
    struct dimensionExpand
    {
	int     previousDim;
	int     expandedDim;
	int     expandRatio;
	
	real_t *sourceData;
	
	const char *patTypes;
	__host__ __device__ void operator() (const thrust::tuple<real_t&, int> &t) const
	{
	    int outputIdx   = t.get<1>();
	    int inputDimIdx = outputIdx % expandedDim / expandRatio;
	    int timeIdx     = outputIdx / expandedDim;

	    if (patTypes[timeIdx] == PATTYPE_NONE){
		t.get<0>() = 0.0;
		return;
	    }
	    
	    t.get<0>() = sourceData[timeIdx * previousDim + inputDimIdx];   
	}
    };

    // Duplicate the result to every frame of the utterance
    struct dimensionExpandGrad
    {
	int     previousDim;
	int     expandedDim;
	int     expandRatio;
	
	real_t *inputGrad;
	
	const char *patTypes;
	
	__host__ __device__ void operator() (const thrust::tuple<real_t&, int> &t) const
	{
	    int gradIndex      = t.get<1>();
	    int previousDimIdx = gradIndex % previousDim;
	    int timeIdx        = gradIndex / previousDim;

	    t.get<0>() = 0.0;
	    
	    if (patTypes[timeIdx] == PATTYPE_NONE)
		return;
	    else
		for (int cnt = 0; cnt < expandRatio; cnt++)
		    t.get<0>() += inputGrad[timeIdx * expandedDim +
					    previousDimIdx * expandRatio + cnt];
	}
    };

    
}
}


namespace layers{

    void printLastShotMessage(const int lastshotMode){
	printf("\tlast shot mode: ");
	switch (lastshotMode){
	case NN_OPE_LAST_SHOT_MODE1:
	    printf("move utterance end to initial frame\n");
	    break;
	case NN_OPE_LAST_SHOT_MODE2:
	    printf("copy utterance end to each frame\n");
	    break;
	case NN_OPE_LAST_SHOT_MODE5:
	    printf("concatenate 1/2 of utterance end and start (from bi-rnn) to inital frame\n");
	    break;
	case NN_OPE_LAST_SHOT_MODE6:
	    printf("concatenate 1/2 of utterance end and start (from bi-rnn) to each frame\n");
	    break;
	case NN_OPE_LAST_SHOT_MODE9:
	    printf("copy average over utterance to each frame\n");
	    break;
	case NN_OPE_LAST_SHOT_MODE3:
	    printf("copy last frame of segments to inital frame of segment\n");
	    break;
	case NN_OPE_LAST_SHOT_MODE4:
	    printf("copy last frame of segments to each frame of segment\n");
	    break;
	case NN_OPE_LAST_SHOT_MODE7:
	    printf("concatenate first/last frame of segments to inital frame of segment\n");
	    break;
	case NN_OPE_LAST_SHOT_MODE8:
	    printf("concatenate first/last frame of segments to each frame of segment\n");
	    break;
	case NN_OPE_LAST_SHOT_MODE10:
	    printf("copy first frame of segment to each frame in segment (causal dependency)\n");
	    break;
	default:
	    printf("Unknown lastshot mode\n");
	    break;
	}
    }
	
    
    template <typename TDevice>
    OperationLayer<TDevice>::OperationLayer(const helpers::JsonValue &layerChild,
					    const helpers::JsonValue &weightsSection,
					    Layer<TDevice>           &precedingLayer,
					    int                       maxSeqLength,
					    int                       layerID)
	: TrainableLayer<TDevice>(layerChild, weightsSection, 0, 0,
				  precedingLayer, maxSeqLength, layerID)
	, m_noiseMag    (1.0)
	, m_noiseSize   (0)
	, m_noiseRepeat (0)
	, m_outDupRate  (0)
	, m_lastShot    (NN_OPE_LAST_SHOT_TURNOFF)
	, m_segLevel    (-1)
	, m_changeTimeRes (0)
    {

	/* ------ Modes support by OperationLayer -----  

	   lastShot:       lastShot mode
	   dropOut:        dropout mode
	   changeTimeRes:  change the time resolution
	   normal:         normal mode that supports noise injection, 
                           feature scaling
	   signalGen:      signal generator, sin wave
	 */

	
	/* ------ Configuration for noise generation ------ */
	// Note: in operation layer, noise is concatenated with input features
	//       for noises added to the input features, use skipIni layer
	m_noiseMag    = (layerChild->HasMember("noiseRatio") ? 
			 static_cast<real_t>((*layerChild)["noiseRatio"].GetDouble()) : 1.0);
	m_noiseSize   = (layerChild->HasMember("noiseDim") ? 
			 static_cast<real_t>((*layerChild)["noiseDim"].GetInt()) : 0);
	m_noiseRepeat = (layerChild->HasMember("noiseRepeat") ? 
			 static_cast<real_t>((*layerChild)["noiseRepeat"].GetInt()) : 0);
	
	// check the layer size
	if (m_noiseSize > 0){
	    if (this->size() != (this->precedingLayer().size() + m_noiseSize))
		throw std::runtime_error("Error, noiseDim + preLayerSize = operator layerSize");
	    
	    // Buffer for noise
	    m_noiseInput.resize(m_noiseSize * (this->outputs().size() / this->size()), 0.0);
	}

	/* ------ Configuration for weighting the input features ------ */
	// Configuration for the weight of input vector
	// Note: setZero can be negative, which is useby the dirty code in fillOutputVec for a
	//       special stop-gradient mode
	// Now: the stop-gradient flag is moved to m_stopGradStr

	// For compatiblity: setZero is used as a special mode that combines weight and stop
	// grad

	m_setZeroStr = (layerChild->HasMember("setZero")?
			(*layerChild)["setZero"].GetString(): "");
	m_setBiasStr = (layerChild->HasMember("shift")?
			(*layerChild)["shift"].GetString() : "");
	m_setScaleStr = (layerChild->HasMember("scale")?
			 (*layerChild)["scale"].GetString() : "");
	m_stopGradStr = (layerChild->HasMember("stopGrad")?
			 (*layerChild)["stopGrad"].GetString() : "");
	m_setBinarizeStr = (layerChild->HasMember("binarize")?
			    (*layerChild)["binarize"].GetString() : "");

	if (m_setBinarizeStr.size()){
	    m_setBinarizeVec_H.clear();
	    misFuncs::ParseFloatOpt(m_setBinarizeStr, m_setBinarizeVec_H);
	}else{
	    m_setBinarizeVec_H.resize(this->precedingLayer().size(), 0.0);
	}
	m_setBinarizeVec_D = m_setBinarizeVec_H;
	
	if (m_setBiasStr.size()){
	    m_setBiasVec_H.clear();
	    misFuncs::ParseFloatOpt(m_setBiasStr, m_setBiasVec_H);
	}else{
	    m_setBiasVec_H.resize(this->precedingLayer().size(), 0.0);
	}
	m_setBiasVec_D = m_setBiasVec_H;

	if (m_setScaleStr.size()){
	    m_setScaleVec_H.clear();
	    misFuncs::ParseFloatOpt(m_setScaleStr, m_setScaleVec_H);
	}else{
	    m_setScaleVec_H.resize(this->precedingLayer().size(), 1.0);
	}
	m_setScaleVec_D = m_setScaleVec_H;
	
	if (m_stopGradStr.size()){
	    m_stopGradVec_H.clear();
	    misFuncs::ParseFloatOpt(m_stopGradStr, m_stopGradVec_H);
	}else{
	    m_stopGradVec_H.resize(this->precedingLayer().size(), 0.0);
	}
	m_stopGradVec_D = m_stopGradVec_H;
	    
	if (this->precedingLayer().size() != m_setBiasVec_D.size())
	    throw std::runtime_error(
		std::string("Error in operator ") + this->name() +
		std::string(", shifts vector length unequal to previous layer size"));
	if (this->precedingLayer().size() != m_setScaleVec_D.size())
	    throw std::runtime_error(
		std::string("Error in operator ") + this->name() +
		std::string(", scales vector length unequal to previous layer size"));
	if (this->precedingLayer().size() != m_stopGradVec_D.size())
	    throw std::runtime_error(
		std::string("Error in operator ") + this->name() +
		std::string(", stopGrad vector length unequal to previous layer size"));

	
	if (m_setZeroStr.size()){
	    m_setZeroVec_H.clear();
	    misFuncs::ParseFloatOpt(m_setZeroStr, m_setZeroVec_H);
	    m_setZeroVec_D = m_setZeroVec_H;

	    if (this->precedingLayer().size() != m_setZeroVec_D.size())
		throw std::runtime_error(
			std::string("Error in operator ") + this->name() +
			std::string(", setZero vector length unequal to previous layer size"));
	
	    // for compatiblity, convert setZero to scales and stopGrad
	    for (int i = 0; i < m_setZeroVec_H.size(); i++)
		if (m_setZeroVec_H[i] < 0.0){
		    m_stopGradVec_H[i] = 1.0; // stop grad
		    m_setScaleVec_H[i] = -1.0 * m_setZeroVec_H[i]; // get the correct weight
		}else{
		    m_stopGradVec_H[i] = 0.0; // not stop grad
		    m_setScaleVec_H[i] = m_setZeroVec_H[i]; 
		}
	    m_setScaleVec_D = m_setScaleVec_H;
	    m_stopGradVec_D = m_stopGradVec_H;
	    
	}else{
	    m_setZeroVec_D.resize(this->precedingLayer().size(), 1.0);
	}


	/* ------ Configuration for F0 -> u/v conversion ----- */
	m_F02UV = (layerChild->HasMember("F02UV")?
			 (*layerChild)["F02UV"].GetInt() : 0);
	if (m_F02UV){
	    m_F0DataMean = (layerChild->HasMember("frequencyF0Mean")?
			    (*layerChild)["frequencyF0Mean"].GetDouble() : 0.0);
	    m_F0DataStd  = (layerChild->HasMember("frequencyF0Std")?
			    (*layerChild)["frequencyF0Std"].GetDouble() : 1.0);
	    const Configuration &config = Configuration::instance();
	    if (config.f0dataMean_signalgen() > -1)
		m_F0DataMean = config.f0dataMean_signalgen();
	    if (config.f0dataStd_signalgen() > -1)
		m_F0DataStd  = config.f0dataStd_signalgen();

	    if (this->size() != 1)
		throw std::runtime_error("Error: U/V extraction only receives 1d input");
	}
	
	
	/* ------ Configuration of the output duplication ------ */
	if (layerChild->HasMember("outputDownSampling")){
	    printf("\toutputDownSampling flag has been changed to outputDuplicating\n");
	    throw std::runtime_error("Error: old configuration name in OperationLayer");
	}
	m_outDupRate   = (layerChild->HasMember("outputDuplicating") ? 
			   static_cast<int>((*layerChild)["outputDuplicating"].GetInt()) : 0);

	
	/* ------ Configuration of last shot mode ------ */
	//
	m_lastShot      = (layerChild->HasMember("lastShot")?
			   static_cast<int>((*layerChild)["lastShot"].GetInt()) :
			   NN_OPE_LAST_SHOT_TURNOFF);
	
	// Configuration of the extraction of the last time steps (only used
	// for m_lastShot = 3, 4, 7, 8, 10)
	m_segLevel      = (layerChild->HasMember("segLevel")?
			   static_cast<int>((*layerChild)["segLevel"].GetInt()) : -1);
	
	if (m_lastShot != NN_OPE_LAST_SHOT_TURNOFF){
	    // Prepare the segment boundary
	    
	    // only use the utterance end boundary	    
	    if (m_lastShot == NN_OPE_LAST_SHOT_MODE1 || m_lastShot == NN_OPE_LAST_SHOT_MODE2 ||
		m_lastShot == NN_OPE_LAST_SHOT_MODE5 || m_lastShot == NN_OPE_LAST_SHOT_MODE6 ||
		m_lastShot == NN_OPE_LAST_SHOT_MODE9){
		
		// buffer to store the length of input sequences
		m_seqLengthBuffH.resize(this->parallelSequences(), 0);
		m_seqLengthBuffD = m_seqLengthBuffH;


		// buffer to store the coefficiets (1.0) for gathering gradients
		cpu_real_vector tmp(this->parallelSequences() * this->maxSeqLength() +
				    this->parallelSequences() - 1, 0.0);

		if (tmp.size()){
		    if (m_lastShot == NN_OPE_LAST_SHOT_MODE2 ||
			m_lastShot == NN_OPE_LAST_SHOT_MODE6){
			//       A B C D A B C D
			// 0 0 0 1 0 0 0 1 0 0 0         
			for (int i = this->parallelSequences(); i<tmp.size();
			     i+=this->parallelSequences())
			    tmp[i-1] = 1.0;
		    }else{
			//       A B C D A B C D
			// 0 0 0 1 0 0 0 0 0 0 0         
			tmp[this->parallelSequences()-1] = 1.0;
		    }
		}
		// check for bi-directional last shot mode
		if (m_lastShot == NN_OPE_LAST_SHOT_MODE5 ||
		    m_lastShot == NN_OPE_LAST_SHOT_MODE6){
		    if (this->size() % 2 > 0){
			printf("Layer size %d should be an even number ", this->size());
			throw std::runtime_error("for bi-directional last shot");
		    }
		}
		m_oneVec  = tmp;
		
	    // use the segmental boundary
	    }else if (m_lastShot == NN_OPE_LAST_SHOT_MODE3 ||
		      m_lastShot == NN_OPE_LAST_SHOT_MODE4 ||
		      m_lastShot == NN_OPE_LAST_SHOT_MODE7 ||
		      m_lastShot == NN_OPE_LAST_SHOT_MODE8 ||
		      m_lastShot == NN_OPE_LAST_SHOT_MODE10){
		
		if (m_segLevel < 0)
		    throw std::runtime_error("segLevel is not configured for Operationlayer");

		// boundary buffer is allocated
		// boundary will be loaded in loadSequences
		m_segBoundaryH.resize(this->maxSeqLength() * this->parallelSequences() * 2, 0);
		m_segBoundaryD = m_segBoundaryH;
		
	    }else{
		throw std::runtime_error("Unknown lastShot option number");
	    }
	}

	/* ------- Configuration of the time resolution change */
	m_changeTimeRes   = (layerChild->HasMember("changeResolution") ? 
			     static_cast<int>((*layerChild)["changeResolution"].GetInt()) : 0);
	if (m_changeTimeRes && this->size() != this->precedingLayer().size())
	    throw std::runtime_error("Layer size unequal for time resolution change");
	if (this->getResolution() > this->precedingLayer().getResolution()){
	    // down sampling
	    if ((this->getResolution() % this->precedingLayer().getResolution()) != 0)
		throw std::runtime_error("Fractional resolution change is not supported");
	}else{
	    // up sampling
	    if ((this->precedingLayer().getResolution() % this->getResolution())!= 0)
		throw std::runtime_error("Fractional resolution change is not supported");
	}

	/* ------- Configuration of dropout */
	m_dropoutRate     = (layerChild->HasMember("dropOut") ? 
			     static_cast<real_t>((*layerChild)["dropOut"].GetDouble()) : -1.0);
	if (m_dropoutRate > 0){
	    if (this->size() != this->precedingLayer().size())
		throw std::runtime_error("Layer size unequal for dropout");
	    m_noiseInput.resize(this->precedingLayer().outputs().size(), 0.0);
	    // in the future, memory of this->outputs() may be released
	}

	/* ------- Configuration of signal generator */
	// signal generator is kept in OperationLayer for compatibility
	// it has been moved to SignalGenLayer
	
	m_freqDim = (layerChild->HasMember("frequencyDim")?
		     static_cast<real_t>((*layerChild)["frequencyDim"].GetDouble()) : -1);
	
	if (m_freqDim >= 0){
	    
	    m_freqOpt = (layerChild->HasMember("frequencyOpt")?
			 static_cast<real_t>((*layerChild)["frequencyOpt"].GetDouble()) : -1);
	    m_freqSR  = (layerChild->HasMember("frequencySR")?
			 static_cast<real_t>((*layerChild)["frequencySR"].GetDouble()) : -1);
	    
	    m_freqQF0min = (layerChild->HasMember("frequencyQF0min")?
			    static_cast<real_t>((*layerChild)["frequencyQF0min"].GetDouble()):-1);
	    m_freqQF0max = (layerChild->HasMember("frequencyQF0max")?
			    static_cast<real_t>((*layerChild)["frequencyQF0max"].GetDouble()):-1);
	    m_freqQF0Lev = (layerChild->HasMember("frequencyQF0Lev")?
			    static_cast<real_t>((*layerChild)["frequencyQF0Lev"].GetDouble()):-1);

	    m_freqDataM = (layerChild->HasMember("frequencyF0Mean")?
			    static_cast<real_t>((*layerChild)["frequencyF0Mean"].GetDouble()):0);
	    m_freqDataS = (layerChild->HasMember("frequencyF0Std")?
			    static_cast<real_t>((*layerChild)["frequencyF0Std"].GetDouble()):1);
	    
	    if (m_freqDim >= this->precedingLayer().size())
		throw std::runtime_error("frequencyDim is larger than previous layer size");
	    if (m_freqDim >= 0 && m_freqSR < 1)
		throw std::runtime_error("frequencySR is not specified");
	    
	    m_noiseMag = (layerChild->HasMember("frequencyNoiseMag") ? 
			  static_cast<real_t>((*layerChild)["frequencyNoiseMag"].GetDouble()):1.0);
	    
	    m_noiseInput = this->outputs();
	}

	/* ------ positional encoding -------- */
	m_positional_code_mode = (layerChild->HasMember("positional_code")?
		     static_cast<int>((*layerChild)["positional_code"].GetDouble()) : -1);
	if (m_positional_code_mode >= 0)
	    printf("\tpositional code mode %d enabled", m_positional_code_mode);
	    

	/* ------ reverse gradient ----------- */
	m_reverse_grad = (layerChild->HasMember("reverse_grad")?
		     static_cast<real_t>((*layerChild)["reverse_grad"].GetDouble()) : -1);
	if (m_reverse_grad >= 0){
	    if (this->size() != this->precedingLayer().size())
		throw std::runtime_error("reverse_grad layer size not equal to previous layer");
	    else
		printf("\treverse grad");
	}

	/* ------ expand dimension ----------- */
	m_dimExpand = (layerChild->HasMember("dim_expand")?
		     static_cast<int>((*layerChild)["dim_expand"].GetDouble()) : 0);
	if (m_dimExpand){
	    if (this->size() <= this->precedingLayer().size()){
		throw std::runtime_error("dim_expand layer should be wider than previous layer");
	    }else{
		m_dimExpand = (int)std::ceil(((float)this->size())/this->precedingLayer().size());
		printf("\tdimension expand: %d", m_dimExpand);
		if (m_dimExpand < 1)
		    throw std::runtime_error("dim_expand layer should be wider than previous layer");
	    }
	}

	
	
	/* ------ print the information ------ */

	// check the layer size
	if (m_noiseSize > 0){
	    if (this->size() != (this->precedingLayer().size() + m_noiseSize))
		throw std::runtime_error("Error, noiseDim + preLayerSize = operator layerSize");
	}else if (m_freqDim >= 0 || m_dimExpand){
	    // free to choose the layer size
	}else{
	    if (this->size() != this->precedingLayer().size())
		throw std::runtime_error("Error, preLayerSize = operator layerSize");
	}
	
	
	printf("\tOperator layer: \n");
	if (m_noiseSize > 0)
	    printf("\tinject noise: dim %d, u[-%f, %f]\n", m_noiseSize, m_noiseMag, m_noiseMag);

	if (m_setZeroStr.size())
	    printf("\tinput/output configuration: %s\n", m_setZeroStr.c_str());
	
	if (m_noiseRepeat){
	    if (m_noiseRepeat == NN_OPERATOR_LAYER_NOISE_TIMEREPEAT)
		printf("\trepeat the same noise across frames\n");
	    else if (m_noiseRepeat == NN_OPERATOR_LAYER_NOISE_DIMREPEAT)
		printf("\trepeat the same noise across dimension\n");
	    else
		printf("\tunknown noise repeat option\n");
	}

	if (m_outDupRate > 1)
	    printf("\toutput duplication at the rate of %d\n", m_outDupRate);
	

	if (m_freqDim >= 0){
	    if (m_noiseSize > 0     || m_outDupRate > 1    ||
		m_setZeroStr.size() || m_lastShot != NN_OPE_LAST_SHOT_TURNOFF      ||
		m_changeTimeRes > 0 || m_dropoutRate > 0)
		throw std::runtime_error("freqDim can't be used with other operations");
	    printf("\t Since wave generation is on\n");
	}

	if (m_dropoutRate > 0){
	    if (m_noiseSize > 0     || m_outDupRate > 1    ||
		m_setZeroStr.size() || m_lastShot != NN_OPE_LAST_SHOT_TURNOFF      ||
		m_changeTimeRes > 0 || m_freqDim >= 0)
		throw std::runtime_error("dropout can't be used with other operations");
	    printf("\t (Inverted) DropOut rate: %f\n", m_dropoutRate);
	}

	
	if (m_changeTimeRes){
	    if (m_noiseSize > 0     || m_outDupRate > 1    ||
		m_setZeroStr.size() || m_lastShot != NN_OPE_LAST_SHOT_TURNOFF      ||
		m_dropoutRate > 0   || m_freqDim >= 0)
		throw std::runtime_error("time resolution can't be used with other operations");
	    printf("\tTurn on time resolution change across layers: from %d to %d",
		   this->precedingLayer().getResolution(), this->getResolution());
	}
	
	if (m_lastShot != NN_OPE_LAST_SHOT_TURNOFF){
	    printLastShotMessage(m_lastShot);
	    if (m_noiseSize > 0     || m_outDupRate > 1    ||
		m_setZeroStr.size() || m_changeTimeRes > 0 ||
		m_dropoutRate > 0   || m_freqDim >= 0)
		throw std::runtime_error("lastShot mode can't be used with other operations");
	    if (this->size() != this->precedingLayer().size())
		throw std::runtime_error("Layer size is unequal to previous one");
	}

	if (m_F02UV)
	    printf("\tF02UV using F0 mean and std: %f %f\n", m_F0DataMean, m_F0DataStd);
	
	
	if (this->precedingLayer().getSaveMemoryFlag() &&
	    (m_lastShot != NN_OPE_LAST_SHOT_TURNOFF || m_changeTimeRes > 0 || m_outDupRate > 1))
	    throw std::runtime_error("Operator doesn't support memory reduce");  
	printf("\n");
    }

    template <typename TDevice>
    OperationLayer<TDevice>::~OperationLayer()
    {
    }

    template <typename TDevice>
    void OperationLayer<TDevice>::exportLayer(const helpers::JsonValue     &layersArray, 
					      const helpers::JsonAllocator &allocator) const
    {
        TrainableLayer<TDevice>::exportLayer(layersArray, allocator);
	if (m_setZeroStr.size())
	    (*layersArray)[layersArray->Size() - 1].AddMember("setZero",  m_setZeroStr.c_str(),
							      allocator);
	if (m_setBiasStr.size())
	    (*layersArray)[layersArray->Size() - 1].AddMember("shift",    m_setBiasStr.c_str(),
							      allocator);
	if (m_setScaleStr.size())
	    (*layersArray)[layersArray->Size() - 1].AddMember("scale",    m_setScaleStr.c_str(),
							      allocator);
	if (m_stopGradStr.size())
	    (*layersArray)[layersArray->Size() - 1].AddMember("stopGrad",  m_stopGradStr.c_str(),
							      allocator);
	if (m_setBinarizeStr.size())
	    (*layersArray)[layersArray->Size() - 1].AddMember("binarize",  m_setBinarizeStr.c_str(),
							      allocator);
	if (m_noiseSize > 0){
	    (*layersArray)[layersArray->Size() - 1].AddMember("noiseRatio", m_noiseMag,
							      allocator);
	    (*layersArray)[layersArray->Size() - 1].AddMember("noiseDim",   m_noiseSize,
							      allocator);
	    (*layersArray)[layersArray->Size() - 1].AddMember("noiseRepeat",m_noiseRepeat,
							      allocator);
	}

	if (m_F02UV){
	    (*layersArray)[layersArray->Size() - 1].AddMember("F02UV", m_F02UV,
							      allocator);
	    (*layersArray)[layersArray->Size() - 1].AddMember("frequencyF0Mean", m_F0DataMean,
							      allocator);
	    (*layersArray)[layersArray->Size() - 1].AddMember("frequencyF0Std", m_F0DataStd,
							      allocator);
	}
	    
	if (m_positional_code_mode >= 0){
	    (*layersArray)[layersArray->Size() - 1].AddMember("positional_code",
							      m_positional_code_mode,
							      allocator);
	}
	
	if (m_reverse_grad >= 0){
	    (*layersArray)[layersArray->Size() - 1].AddMember("reverse_grad", m_reverse_grad,
							      allocator);
	}

	if (m_freqDim >= 0){
	    
	    (*layersArray)[layersArray->Size() - 1].AddMember("frequencyDim", m_freqDim,
							      allocator);
	    (*layersArray)[layersArray->Size() - 1].AddMember("frequencyOpt", m_freqOpt,
							      allocator);
	    (*layersArray)[layersArray->Size() - 1].AddMember("frequencySR",  m_freqSR,
							      allocator);
	    (*layersArray)[layersArray->Size() - 1].AddMember("frequencyQF0min", m_freqQF0min,
							      allocator);
	    (*layersArray)[layersArray->Size() - 1].AddMember("frequencyQF0max", m_freqQF0max,
							      allocator);
	    (*layersArray)[layersArray->Size() - 1].AddMember("frequencyQF0Lev", m_freqQF0Lev,
							      allocator);
	    (*layersArray)[layersArray->Size() - 1].AddMember("frequencyF0Mean", m_freqDataM,
							      allocator);
	    (*layersArray)[layersArray->Size() - 1].AddMember("frequencyF0Std",  m_freqDataS,
							      allocator);
	    (*layersArray)[layersArray->Size() - 1].AddMember("frequencyNoiseMag", m_noiseMag,
							      allocator);
	}
	
	if (m_dropoutRate > 0){
	    (*layersArray)[layersArray->Size() - 1].AddMember("dropOut",m_dropoutRate,
							      allocator);
	}
	
	if (m_outDupRate > 1)
	    (*layersArray)[layersArray->Size() - 1].AddMember("outputDuplicating", m_outDupRate,
							      allocator);
	if (m_lastShot != NN_OPE_LAST_SHOT_TURNOFF){
	    (*layersArray)[layersArray->Size() - 1].AddMember("lastShot", m_lastShot,
							      allocator);
	    (*layersArray)[layersArray->Size() - 1].AddMember("segLevel", m_segLevel,
							      allocator);
	}

	if (m_dimExpand)
	    (*layersArray)[layersArray->Size() - 1].AddMember("dim_expand", m_dimExpand, allocator);
	
	if (m_changeTimeRes > 0)
	    (*layersArray)[layersArray->Size() - 1].AddMember("changeResolution",
							      m_changeTimeRes,
							      allocator);
    }

    template <typename TDevice>
    void OperationLayer<TDevice>::loadSequences(const data_sets::DataSetFraction &fraction,
						const int nnState)
    {
	TrainableLayer<TDevice>::loadSequences(fraction, nnState);

	if (m_lastShot == NN_OPE_LAST_SHOT_MODE1 || m_lastShot == NN_OPE_LAST_SHOT_MODE2 ||
	    m_lastShot == NN_OPE_LAST_SHOT_MODE5 || m_lastShot == NN_OPE_LAST_SHOT_MODE6 ||
	    m_lastShot == NN_OPE_LAST_SHOT_MODE9){
	    
	    // load the sequence length 
	    for (int i = 0; i<fraction.numSequences(); i++)
		m_seqLengthBuffH[i] = misFuncs::getResoLength(fraction.seqInfo(i).length,
							      this->getResolution(), 1);
	    m_seqLengthBuffD = m_seqLengthBuffH;
	    
	}else if (m_lastShot == NN_OPE_LAST_SHOT_MODE3 || m_lastShot == NN_OPE_LAST_SHOT_MODE4 ||
		  m_lastShot == NN_OPE_LAST_SHOT_MODE7 || m_lastShot == NN_OPE_LAST_SHOT_MODE8 ||
		  m_lastShot == NN_OPE_LAST_SHOT_MODE10){
	    
	    // load the segments length
	    if (m_segLevel > CHAR_BIT)
		throw std::runtime_error("Operationlayer: segLevel is larger than expected");
	    
	    if (fraction.auxPattypeData().size() == 0)
		throw std::runtime_error("Operationlayer: requires boundary (auxData)");

	    
	    int pos;
	    
	    // boundary in forward / backward direction
	    int boundary, boundaryR;
	    
	    // bit operator, 2^m_segLevel
	    char bitOp = (0b01 << m_segLevel);

	    // 
	    for (int i = 0; i < fraction.numSequences(); i++){
		
		// the last segment boundry is the end of utterance
		boundary  = (fraction.seqInfo(i).length - 1);
		boundaryR = 0;

		// decide the lingusitic segment boundary 
		for (int time = fraction.seqInfo(i).length - 1; time>=0; time--){

		    // Abosolute position in the data sequence
		    pos = time * this->parallelSequences() + i;  

		    // -------
		    // Forward direction
		    //                 N N+1      abosolute time (for each single sentence)
		    // ... N  N  N  N  N N+K      MODE4/MODE8
		    // ... N -1 -1 -1 -1 N+K      MODE3/MODE7
		    if (m_lastShot == NN_OPE_LAST_SHOT_MODE3 ||
			m_lastShot == NN_OPE_LAST_SHOT_MODE7){
			// if not repeat 
			if (fraction.auxPattypeData()[pos] & bitOp){
			    m_segBoundaryH[2 * pos] = boundary;  // 
			}else{
			    m_segBoundaryH[2 * pos] = -1;       // other frames
			}
			
		    }else if (m_lastShot == NN_OPE_LAST_SHOT_MODE4 ||
			      m_lastShot == NN_OPE_LAST_SHOT_MODE8){
			// repeat across the segment
			m_segBoundaryH[2 * pos] = boundary;
			
		    }else if (m_lastShot == NN_OPE_LAST_SHOT_MODE10){
			m_segBoundaryH[2 * pos+1] = boundary;
			
		    }else{
			throw std::runtime_error("Unknown lastShot mode");
		    }

		    // Update the boundary
		    if (fraction.auxPattypeData()[pos] & bitOp){
			boundary = time - 1;
		    }

		    // -------		    
		    // Backward direction
		    pos = (fraction.seqInfo(i).length-1-time) * this->parallelSequences() + i;

		    // Update the boundary
		    if (fraction.auxPattypeData()[pos] & bitOp){
			boundaryR = (fraction.seqInfo(i).length-1-time);
		    }

		    //     M             M+K      abosolute time (for each single sentence)
		    // ... M  M  M  M  M M+K      MODE4/MODE8
		    // ...-1 -1 -1 -1  M M+K      MODE3/MODE7
  		    if (m_lastShot == NN_OPE_LAST_SHOT_MODE3 ||
			m_lastShot == NN_OPE_LAST_SHOT_MODE7){
			// if not repeat 
			if (time == 0||
			    (fraction.auxPattypeData()[pos + this->parallelSequences()] & bitOp)){
			    m_segBoundaryH[2 * pos + 1] = boundaryR; // 
			}else{
			    m_segBoundaryH[2 * pos + 1] = -1;        // other frames
			}
			
		    }else if (m_lastShot == NN_OPE_LAST_SHOT_MODE4 ||
			      m_lastShot == NN_OPE_LAST_SHOT_MODE8){
			// if repeat across the segments
			m_segBoundaryH[2 * pos + 1] = boundaryR;
			
		    }else if (m_lastShot == NN_OPE_LAST_SHOT_MODE10){
			
			m_segBoundaryH[2 * pos] = boundaryR;
		    }else{
			throw std::runtime_error("Unknown lastShot mode");
		    }
		    
		}
		
	    }
	    m_segBoundaryD = m_segBoundaryH;
	    
	}
	
    }
    
    template <typename TDevice>
    const std::string& OperationLayer<TDevice>::type() const
    {
        static std::string s;
        if (s.empty()) s = "operator";
        return s;
    }

    template <typename TDevice>
    void OperationLayer<TDevice>::computeForwardPass(const int nnState)
    {
	int timeLength = this->curMaxSeqLength() * this->parallelSequences();

	if (m_lastShot == NN_OPE_LAST_SHOT_MODE1 || m_lastShot == NN_OPE_LAST_SHOT_MODE2 ||
	    m_lastShot == NN_OPE_LAST_SHOT_MODE5 || m_lastShot == NN_OPE_LAST_SHOT_MODE6){
	    // use last shot mode
	    internal::lastShotForward fn1;
	    fn1.featureDim = this->size();
	    fn1.paralSeqNm = this->parallelSequences();
	    fn1.lastShotOp = this->m_lastShot;
	    fn1.seqLengthD = helpers::getRawPointer(m_seqLengthBuffD);
	    fn1.sourceData = helpers::getRawPointer(this->precedingLayer().outputs());
	    fn1.patTypes   = helpers::getRawPointer(this->patTypes());
	    
	    int n = timeLength * this->size();
	    thrust::for_each(
               thrust::make_zip_iterator(
		  thrust::make_tuple(this->outputs().begin(),
				     thrust::counting_iterator<int>(0))),
	       thrust::make_zip_iterator(
		  thrust::make_tuple(this->outputs().begin()           + n,
				     thrust::counting_iterator<int>(0) + n)),
	       fn1);
	    
	}else if (m_lastShot == NN_OPE_LAST_SHOT_MODE9){
	    // Use the average mode
	    // The average \sum_t=1^T x_t / T is conducted as matrix multiplication
	    //   average = [1/T, 1/T .... 1/T] * [x_1, x_2, ..., x_T]^{\top} 
	    
	    // for each utterance in the parallel block
	    // 1. adjust the vector [1/T, ..., 1/T]
	    // 2. matrix multiplication
	    // 3. duplicate the average to each frame
	    // Because each utterance has different T, the first two steps
	    //  must be conducted for each utterance separately

	    // for each utterance in the parallel model
	    for (int i = 0; i<this->parallelSequences(); i++){

		if (this->m_seqLengthBuffH[i] < 1){
		    // skip over dummy sentences
		    continue;
		}

		// step1. adjust the vector [1/T, ..., 1/T]
		int tmpUttL = this->parallelSequences() * this->m_seqLengthBuffH[i];
		
		{{
		    thrust::fill(m_oneVec.begin(), m_oneVec.end(), 0.0);
		    
		    internal::changeOneVec fn1;
		    fn1.sentL       = (real_t)this->m_seqLengthBuffH[i];
		    fn1.parallelNum = this->parallelSequences();

		    thrust::for_each(
                        thrust::make_zip_iterator(
			    thrust::make_tuple(m_oneVec.begin(),
					       thrust::counting_iterator<int>(0))),
			thrust::make_zip_iterator(
			    thrust::make_tuple(m_oneVec.begin()                  + tmpUttL,
					       thrust::counting_iterator<int>(0) + tmpUttL)),
			fn1);
		}}

		// step2. 
		// get the average, and save to the first frame of each utterance
		helpers::Matrix<TDevice> onevec (&this->m_oneVec,
						 tmpUttL, 1,
						 (this->parallelSequences() - 1 - i));
		helpers::Matrix<TDevice> source (&this->precedingLayer().outputs(),
						 this->size(), tmpUttL);
		helpers::Matrix<TDevice> output (&this->outputs(), 
						  this->size(), 1,
						  i * this->size());
		output.assignProduct(source, false, onevec, false);
	    }

	    // step3. duplicate the average to all the frames
	    {{
		internal::duplicateSentVec fn2;
		fn2.featureDim = this->size();
		fn2.paralSeqNm = this->parallelSequences();
		fn2.sourceData = helpers::getRawPointer(this->outputs());
		fn2.patTypes   = helpers::getRawPointer(this->patTypes());
	    
		int n = timeLength * this->size();
		thrust::for_each(
                 thrust::make_zip_iterator(
		  thrust::make_tuple(this->outputs().begin(),
				     thrust::counting_iterator<int>(0))),
		 thrust::make_zip_iterator(
		  thrust::make_tuple(this->outputs().begin()           + n,
				     thrust::counting_iterator<int>(0) + n)),
		 fn2);
	    }}
	    

	}else if (m_lastShot == NN_OPE_LAST_SHOT_MODE3 || m_lastShot == NN_OPE_LAST_SHOT_MODE4 ||
		  m_lastShot == NN_OPE_LAST_SHOT_MODE7 || m_lastShot == NN_OPE_LAST_SHOT_MODE8 ||
		  m_lastShot == NN_OPE_LAST_SHOT_MODE10){
	    
	    // use last shot mode based on segmental boundary
	    internal::lastShotForwardSegBoundary fn1;
	    fn1.featureDim  = this->size();
	    fn1.paralSeqNm  = this->parallelSequences();
	    fn1.lastShotOp  = this->m_lastShot;
	    fn1.segBoundary = helpers::getRawPointer(m_segBoundaryD);
	    fn1.sourceData  = helpers::getRawPointer(this->precedingLayer().outputs());
	    fn1.patTypes    = helpers::getRawPointer(this->patTypes());
	    
	    int n = timeLength * this->size();
	    thrust::for_each(
               thrust::make_zip_iterator(
		  thrust::make_tuple(this->outputs().begin(),
				     thrust::counting_iterator<int>(0))),
	       thrust::make_zip_iterator(
		  thrust::make_tuple(this->outputs().begin()           + n,
				     thrust::counting_iterator<int>(0) + n)),
	       fn1);
	    	    
	}else if (m_dropoutRate > 0){
	    // inverted dropout data
	    int n = timeLength * this->size();	    
	    if (nnState == NN_STATE_GAN_NOGAN_TRAIN){
		// Training stage
		
		// Generate the noise
		thrust::counting_iterator<unsigned int> index_sequence_begin(0);
		thrust::transform(
			index_sequence_begin,
			index_sequence_begin + timeLength * this->size(),
			m_noiseInput.begin(),
			internal::genNoise(0.0, 1.0, (int)(misFuncs::GetRandomNumber()*10000.0)));

		// Dropout
		internal::invertedDropout fn1;
		fn1.curLayerSize = this->size();
		fn1.threshold    = m_dropoutRate;
		
		fn1.noise        = helpers::getRawPointer(m_noiseInput);
		fn1.patTypes     = helpers::getRawPointer(this->patTypes());
		//fn1.sourceData   = helpers::getRawPointer(this->precedingLayer().outputs());
	    
		thrust::for_each(
                 thrust::make_zip_iterator(
		  thrust::make_tuple(this->outputs().begin(),
				     this->precedingLayer().outputs().begin(),
				     thrust::counting_iterator<int>(0))),
		 thrust::make_zip_iterator(
		  thrust::make_tuple(this->outputs().begin()                  + n,
				     this->precedingLayer().outputs().begin() + n,
				     thrust::counting_iterator<int>(0)        + n)),
		 fn1);
		
	    }else{
		// Generation: directly copy
		thrust::copy(this->precedingLayer().outputs().begin(),
			     this->precedingLayer().outputs().begin() + n,
			     this->outputs().begin());
	    }
	    
	    
	}else if (m_changeTimeRes){
	    
	    internal::timeResolutionChange fn1;
	    fn1.inputRes  = this->precedingLayer().getResolution();
	    fn1.outputRes = this->getResolution();
	    fn1.layerSize = this->size();
	    fn1.parallel  = this->parallelSequences();
	    fn1.sourceData  = helpers::getRawPointer(this->precedingLayer().outputs());
	    fn1.patTypes    = helpers::getRawPointer(this->patTypes());
	    
	    int n = timeLength * this->size();
	    thrust::for_each(
               thrust::make_zip_iterator(
		  thrust::make_tuple(this->outputs().begin(),
				     thrust::counting_iterator<int>(0))),
	       thrust::make_zip_iterator(
		  thrust::make_tuple(this->outputs().begin()           + n,
				     thrust::counting_iterator<int>(0) + n)),
	       fn1);
	    
	}else if (m_freqDim >= 0){


	    // generate noise for unvoiced segments
	    thrust::counting_iterator<unsigned int> index_sequence_begin(0);
	    thrust::transform(
			      index_sequence_begin,
			      index_sequence_begin + timeLength * this->size(),
			      m_noiseInput.begin(),
			      internal::genNoise(-1.0 * m_noiseMag, 1.0 * m_noiseMag,
						 (int)(misFuncs::GetRandomNumber()*10000.0)));
	    thrust::copy(m_noiseInput.begin(),
			 m_noiseInput.begin() + timeLength * this->size(),
			 this->outputs().begin());
	    
	    
	    internal::sinWaveGenerator_accum fn1;
	    fn1.inputRes  = this->precedingLayer().getResolution();
	    fn1.outputRes = this->getResolution();
	    fn1.thislayerSize = this->size();
	    fn1.prelayerSize  = this->precedingLayer().size();

	    fn1.inputShiftTime = 0;
	    fn1.outputShiftTime = 0;
	    
	    fn1.freqDim       = (int)this->m_freqDim;
	    fn1.freqOpt       = (int)this->m_freqOpt;
	    fn1.freqSR        = (real_t)this->m_freqSR;
	    fn1.qf0Max        = this->m_freqQF0max;
	    fn1.qf0Min        = this->m_freqQF0min;
	    fn1.qf0Lev        = this->m_freqQF0Lev;
	    fn1.f0M           = this->m_freqDataM;
	    fn1.f0S           = this->m_freqDataS;
	    
	    fn1.parallel    = this->parallelSequences();
	    fn1.targetData  = helpers::getRawPointer(this->outputs());
	    fn1.sourceData  = helpers::getRawPointer(this->precedingLayer().outputs());
	    fn1.patTypes    = helpers::getRawPointer(this->patTypes());
	    fn1.seqLength   = timeLength;


	    thrust::for_each(
               thrust::make_zip_iterator(
		  thrust::make_tuple(this->outputs().begin(),
				     thrust::counting_iterator<int>(0))),
	       thrust::make_zip_iterator(
		  thrust::make_tuple(this->outputs().begin()           + this->size(),
				     thrust::counting_iterator<int>(0) + this->size())),
	       fn1);
	       
	    
	}else if (m_F02UV){
	    // convert the input F0 into U/V
	    thrust::fill(this->outputErrors().begin(), this->outputErrors().end(), 0.0);
	    {
		internal::normF0ToUV fn1;
		fn1.F0mean = m_F0DataMean;
		fn1.F0std  = m_F0DataStd;
		thrust::for_each(
		  thrust::make_zip_iterator(
			thrust::make_tuple(this->precedingLayer().outputs().begin(),
					   this->outputs().begin())),
		  thrust::make_zip_iterator(
			thrust::make_tuple(this->precedingLayer().outputs().begin() + timeLength,
					   this->outputs().begin()                  + timeLength)),
		  fn1);
	    }
	    
	}else if (m_positional_code_mode >= 0){
	    // generate the positional code
	    {
	    	internal::positionalCode fn1;
		fn1.featureDim = this->size();
		fn1.parallel   = this->parallelSequences();
		
		thrust::for_each(
		  thrust::make_zip_iterator(
		    thrust::make_tuple(this->outputs().begin(),
				       thrust::counting_iterator<int>(0))),
		  thrust::make_zip_iterator(
		    thrust::make_tuple(this->outputs().begin()     + this->size() * timeLength,
				 thrust::counting_iterator<int>(0) + this->size() * timeLength)),
		  fn1);
	    }
	}else if (m_reverse_grad >= 0){
	    
	    // nothing for forward propagation during gradients reverse mode
	    this->outputs() = this->precedingLayer().outputs();
	    
	}else if (m_dimExpand){

	    // expand the dimensions
	    {
	    	internal::dimensionExpand fn1;
		fn1.previousDim = this->precedingLayer().size();
		fn1.expandedDim = this->size();
		fn1.expandRatio = m_dimExpand;
		fn1.patTypes    = helpers::getRawPointer(this->patTypes());
		fn1.sourceData  = helpers::getRawPointer(this->precedingLayer().outputs());
		
		thrust::for_each(
		  thrust::make_zip_iterator(
		    thrust::make_tuple(this->outputs().begin(),
				       thrust::counting_iterator<int>(0))),
		  thrust::make_zip_iterator(
		    thrust::make_tuple(this->outputs().begin()     + this->size() * timeLength,
				 thrust::counting_iterator<int>(0) + this->size() * timeLength)),
		  fn1);
	    }
	    
	}else{
	    // normal mode
	    if (m_noiseSize > 0){
		// generate the noise for all frames
		thrust::counting_iterator<unsigned int> index_sequence_begin(0);
		thrust::transform(
			index_sequence_begin,
			index_sequence_begin + timeLength * m_noiseSize,
			m_noiseInput.begin(),
			internal::genNoise(-1.0 * m_noiseMag, m_noiseMag,
					   (int)(misFuncs::GetRandomNumber()*10000.0)));

	    }
	
	    {
	    internal::fillOutputVec fn;
	    fn.curLayerSize = this->size();
	    fn.preLayerSize = this->precedingLayer().size();
	    fn.noiseDim     = m_noiseSize;
	    fn.noiseRepeat  = m_noiseRepeat;
	    fn.flagForward  = true;
	    fn.preShift  = 0;
	    
	    fn.weights   = helpers::getRawPointer(m_setScaleVec_D);
	    fn.shifts    = helpers::getRawPointer(m_setBiasVec_D);
	    fn.stopGrad  = helpers::getRawPointer(m_stopGradVec_D);
	    fn.binarize  = helpers::getRawPointer(m_setBinarizeVec_D);
	    
	    fn.preOutput = helpers::getRawPointer(this->precedingLayer().outputs());
	    fn.noiseData = helpers::getRawPointer(this->m_noiseInput);
	    fn.patTypes  = helpers::getRawPointer(this->patTypes());

	    int n = timeLength * this->size();
	    thrust::for_each(
               thrust::make_zip_iterator(
		  thrust::make_tuple(this->outputs().begin(),
				     thrust::counting_iterator<int>(0))),
	       thrust::make_zip_iterator(
		  thrust::make_tuple(this->outputs().begin()           + n,
				     thrust::counting_iterator<int>(0) + n)),
	       fn);
	    
	    }
	    
	    if (m_outDupRate > 1){
		internal::outDuplicationOperation fn1;
		fn1.featureDim = this->size();
		fn1.resolution = m_outDupRate;
		fn1.maxTimeLength = timeLength;
		fn1.dataMatrix = helpers::getRawPointer(this->outputs());
		fn1.parall     = this->parallelSequences();
		fn1.patTypes  = helpers::getRawPointer(this->patTypes());
		
		int n = timeLength * this->size();
		thrust::for_each(
		    thrust::make_zip_iterator(
			thrust::make_tuple(this->outputs().begin(),
					   thrust::counting_iterator<int>(0))),
		    thrust::make_zip_iterator(
			thrust::make_tuple(this->outputs().begin()           + n,
					   thrust::counting_iterator<int>(0) + n)),
		    fn1);
	    }
	}

    }

    template <typename TDevice>
    void OperationLayer<TDevice>::computeForwardPass(const int timeStep, const int nnState)
    {
	int timeLength = this->curMaxSeqLength() * this->parallelSequences();

	int effTimeStart = timeStep * this->parallelSequences();
	int effTimeEnd   = (timeStep+1) * this->parallelSequences();
	
	// Pointer to the output of previous layer (input buffer)
	int shiftIn  = this->precedingLayer().outputBufPtrBias(timeStep*this->parallelSequences(),
							       nnState);
	// Pointer to the output of this layer
	int shiftOut = this->outputBufPtrBias(timeStep * this->parallelSequences(), nnState);

	
	if (m_lastShot == NN_OPE_LAST_SHOT_MODE1 || m_lastShot == NN_OPE_LAST_SHOT_MODE2 ||
	    m_lastShot == NN_OPE_LAST_SHOT_MODE5 || m_lastShot == NN_OPE_LAST_SHOT_MODE6 ||
	    m_lastShot == NN_OPE_LAST_SHOT_MODE9){

	    // 2017-12-14 Should not use LastShot mode for inference
	    //
	    
	    // Tricky code
	    //
	    //  This part should be handled. LastShot mode is now only used before the VAE
	    //  layer. In the generation time, VAE with m_vaeUseageOpt==2 will not transform
	    //  the output from the lastShot layers. 

	    //  The code just set up the boundary of the segments
	    //
	    
	    // Although operator with last shot should not be used after a feedback layer
	    // (because it generates the output at the end of segment and uses it at the begining
	    //  of a segment), the boundary information can be generated
	    /*
	    if (timeStep == 0){
		thrust::fill(this->precedingLayer().outputs().begin(),
			     this->precedingLayer().outputs().begin()+timeLength * this->size(),
			     1.0);
		internal::lastShotForward fn1;
		fn1.featureDim = this->size();
		fn1.paralSeqNm = this->parallelSequences();
		fn1.lastShotOp = this->m_lastShot;
		fn1.seqLengthD = helpers::getRawPointer(m_seqLengthBuffD);
		fn1.sourceData = helpers::getRawPointer(this->precedingLayer().outputs());
		fn1.patTypes   = helpers::getRawPointer(this->patTypes());
	    
		int n = timeLength * this->size();
		thrust::for_each(
		 thrust::make_zip_iterator(
		  thrust::make_tuple(this->outputs().begin(),
				     thrust::counting_iterator<int>(0))),
		 thrust::make_zip_iterator(
		  thrust::make_tuple(this->outputs().begin()           + n,
				     thrust::counting_iterator<int>(0) + n)),
		 fn1);
		 }*/
	    
	}else if (m_lastShot == NN_OPE_LAST_SHOT_MODE3 || m_lastShot == NN_OPE_LAST_SHOT_MODE4 ||
		  m_lastShot == NN_OPE_LAST_SHOT_MODE7 || m_lastShot == NN_OPE_LAST_SHOT_MODE8){
	    /*
	      This part should be handled. LastShot mode is now only used before the VAE
	      layer. In the generation time, VAE with m_vaeUseageOpt==2 will not transform
	      the output from the lastShot layers. 

	      The code just set up the boundary of the segments
	     */
	    
	    // Last shot mode can not be used here
	    // 
	    if (timeStep == 0){
		thrust::fill(this->precedingLayer().outputs().begin(),
			     this->precedingLayer().outputs().begin()+timeLength * this->size(),
			     1.0);
		internal::lastShotForwardSegBoundary fn1;
		fn1.featureDim  = this->size();
		fn1.paralSeqNm  = this->parallelSequences();
		fn1.lastShotOp  = this->m_lastShot;
		fn1.segBoundary = helpers::getRawPointer(m_segBoundaryD);
		fn1.sourceData  = helpers::getRawPointer(this->precedingLayer().outputs());
		fn1.patTypes    = helpers::getRawPointer(this->patTypes());
	    
		int n = timeLength * this->size();
		thrust::for_each(
                 thrust::make_zip_iterator(
		  thrust::make_tuple(this->outputs().begin(),
				     thrust::counting_iterator<int>(0))),
		 thrust::make_zip_iterator(
		  thrust::make_tuple(this->outputs().begin()           + n,
				     thrust::counting_iterator<int>(0) + n)),
		 fn1);
	    }
	    
	}else if (m_lastShot == NN_OPE_LAST_SHOT_MODE10){
	    // MODE10 satifies the causal dependency
	    int st = timeStep * this->size();
	    int et = timeStep * this->size() + this->size();
	    
	    // use last shot mode based on segmental boundary
	    internal::lastShotForwardSegBoundary fn1;
	    fn1.featureDim  = this->size();
	    fn1.paralSeqNm  = this->parallelSequences();
	    fn1.lastShotOp  = this->m_lastShot;
	    fn1.segBoundary = helpers::getRawPointer(m_segBoundaryD);
	    fn1.sourceData  = helpers::getRawPointer(this->precedingLayer().outputs());
	    fn1.patTypes    = helpers::getRawPointer(this->patTypes());
	    
	    thrust::for_each(
               thrust::make_zip_iterator(
		  thrust::make_tuple(this->outputs().begin()           + st,
				     thrust::counting_iterator<int>(0) + st)),
	       thrust::make_zip_iterator(
		  thrust::make_tuple(this->outputs().begin()           + et,
				     thrust::counting_iterator<int>(0) + et)),
	       fn1);
	    
	    // Done
	    
	}else if (m_dropoutRate > 0){
	    // inverted dropout data
	    int st = effTimeStart * this->size();
	    int et = effTimeEnd   * this->size();
	    
	    if (nnState == NN_STATE_GAN_NOGAN_TRAIN){
		// Training stage
		
		// Generate the noise at the first step
		if (timeStep == 0){
		    thrust::counting_iterator<unsigned int> index_sequence_begin(0);
		    thrust::transform(
			index_sequence_begin,
			index_sequence_begin + timeLength * this->size(),
			m_noiseInput.begin(),
			internal::genNoise(0.0, 1.0, (int)(misFuncs::GetRandomNumber()*10000.0)));
		}

		// Dropout
		internal::invertedDropout fn1;
		fn1.curLayerSize = this->size();
		fn1.threshold    = m_dropoutRate;
		
		fn1.noise        = helpers::getRawPointer(m_noiseInput);
		fn1.patTypes     = helpers::getRawPointer(this->patTypes());
		//fn1.sourceData   = helpers::getRawPointer(this->precedingLayer().outputs());
	    
		thrust::for_each(
                 thrust::make_zip_iterator(
		  thrust::make_tuple(this->outputs().begin()                   + (st - shiftOut),
				     this->precedingLayer().outputs().begin()  + (st - shiftIn),
				     thrust::counting_iterator<int>(0)         + st)),
		 thrust::make_zip_iterator(
		  thrust::make_tuple(this->outputs().begin()                   + (et - shiftOut),
				     this->precedingLayer().outputs().begin()  + (et - shiftIn),
				     thrust::counting_iterator<int>(0)         + et)),
		 fn1);
		
	    }else{
		// Generation: directly copy
		thrust::copy(this->precedingLayer().outputs().begin() + (st - shiftIn),
			     this->precedingLayer().outputs().begin() + (et - shiftIn),
			     this->outputs().begin()                  + (st - shiftOut));
	    }
	    
	}else if (m_changeTimeRes){

	    /* Input timeStep is the default time resolution of the network	      
	       Assume that timeResolution >= 1
	     */
	    
	    // time resolution has been considered in NeuralNetwork.cpp
	    //int st = (timeStep / this->getResolution()) * this->size();
	    //int et = (timeStep / this->getResolution()) * this->size() + this->size();
	    int st = timeStep * this->size();
	    int et = timeStep * this->size() + this->size() * this->parallelSequences();
	    
	    internal::timeResolutionChange fn1;
	    fn1.inputRes    = this->precedingLayer().getResolution();
	    fn1.outputRes   = this->getResolution();
	    fn1.layerSize   = this->size();
	    fn1.parallel    = this->parallelSequences();
	    fn1.sourceData  = helpers::getRawPointer(this->precedingLayer().outputs());
	    fn1.patTypes    = helpers::getRawPointer(this->patTypes());
	    
	    thrust::for_each(
               thrust::make_zip_iterator(
		  thrust::make_tuple(this->outputs().begin()           + st,
				     thrust::counting_iterator<int>(0) + st)),
	       thrust::make_zip_iterator(
		  thrust::make_tuple(this->outputs().begin()           + et,
				     thrust::counting_iterator<int>(0) + et)),
	       fn1);
	}else if (m_freqDim > 0){

	    if (timeStep == 0){
		// generate noise for unvoiced segments
		thrust::counting_iterator<unsigned int> index_sequence_begin(0);
		thrust::transform(
				  index_sequence_begin,
				  index_sequence_begin + timeLength * this->size(),
				  m_noiseInput.begin(),
				  internal::genNoise(-1.0, 1.0,
						     (int)(misFuncs::GetRandomNumber()*10000.0)));
	    }
	    
	    int st = effTimeStart * this->size();
	    int et = effTimeEnd   * this->size();
	    
	    thrust::copy(m_noiseInput.begin() + st,
			 m_noiseInput.begin() + et,
			 this->outputs().begin() + st - shiftOut);
	    
	    // to be finished
	    
	}else if (m_F02UV){
	    {
		internal::normF0ToUV fn1;
		fn1.F0mean = m_F0DataMean;
		fn1.F0std  = m_F0DataStd;
		thrust::for_each(
		  thrust::make_zip_iterator(
			thrust::make_tuple(this->precedingLayer().outputs().begin() + effTimeStart,
					   this->outputs().begin() + effTimeStart)),
		  thrust::make_zip_iterator(
			thrust::make_tuple(this->precedingLayer().outputs().begin() + effTimeEnd,
					   this->outputs().begin()                  + effTimeEnd)),
		  fn1);
	    }
	}else if (m_positional_code_mode >= 0){
	    if (timeStep == 0)
		this->computeForwardPass(nnState);
	    
	}else if (m_reverse_grad >=0 ){
	    thrust::copy(this->precedingLayer().outputs().begin() + effTimeStart,
			 this->precedingLayer().outputs().begin() + effTimeEnd,
			 this->outputs().begin() + effTimeStart);
	    
	}else if (m_dimExpand){

	    throw std::runtime_error("Error: dimExpanded not implemented for online mode");
	    
	}else{
	
	    if (m_noiseSize > 0 && timeStep == 0){
		// generate the noise for all frames at the 1st timeStep
		thrust::counting_iterator<unsigned int> index_sequence_begin(0);
		thrust::transform(
				  index_sequence_begin,
				  index_sequence_begin + timeLength * m_noiseSize,
				  m_noiseInput.begin(),
				  internal::genNoise(-1.0 * m_noiseMag, m_noiseMag,
						     (int)(misFuncs::GetRandomNumber()*10000.0)));

	    }
	    {
	    internal::fillOutputVec fn;
	    fn.curLayerSize = this->size();
	    fn.preLayerSize = this->precedingLayer().size();
	    fn.noiseDim     = m_noiseSize;
	    fn.noiseRepeat  = m_noiseRepeat;
	    fn.flagForward  = true;

	    fn.weights   = helpers::getRawPointer(m_setScaleVec_D);
	    fn.shifts    = helpers::getRawPointer(m_setBiasVec_D);
	    fn.stopGrad  = helpers::getRawPointer(m_stopGradVec_D);
	    fn.binarize  = helpers::getRawPointer(m_setBinarizeVec_D);
	    
	    fn.preOutput = helpers::getRawPointer(this->precedingLayer().outputs());
	    fn.preShift  = shiftIn;
	    fn.noiseData = helpers::getRawPointer(this->m_noiseInput);
	    fn.patTypes  = helpers::getRawPointer(this->patTypes());

	    int st = effTimeStart * this->size();
	    int et = effTimeEnd   * this->size();
	    
	    if (timeStep == 0 && this->precedingLayer().type()=="vae"){
		// for VAE layer, we need to load the noise for all frames
		st = 0;
		et = timeLength * this->size();
	    }
	    thrust::for_each(
               thrust::make_zip_iterator(
		 thrust::make_tuple(this->outputs().begin()           + (st-shiftOut),
				    thrust::counting_iterator<int>(0) + st)),
	       thrust::make_zip_iterator(
		 thrust::make_tuple(this->outputs().begin()           + (et-shiftOut),
				     thrust::counting_iterator<int>(0) + et)),
	       fn);
	    }

	    if (m_outDupRate > 1){
		internal::outDuplicationOperation fn1;
		fn1.featureDim = this->size();
		fn1.resolution = m_outDupRate;
		fn1.maxTimeLength = timeLength;
		fn1.dataMatrix = helpers::getRawPointer(this->outputs());
		fn1.parall     = this->parallelSequences();
		fn1.patTypes  = helpers::getRawPointer(this->patTypes());
	    
		int st = timeStep * this->size();
		int et = timeStep * this->size() + this->size();
		thrust::for_each(
		 thrust::make_zip_iterator(
		  thrust::make_tuple(this->outputs().begin()           + st,
				     thrust::counting_iterator<int>(0) + st)),
		 thrust::make_zip_iterator(
		  thrust::make_tuple(this->outputs().begin()           + et,
				     thrust::counting_iterator<int>(0) + et)),
		 fn1);
	    }
	}

    }

    template <typename TDevice>
    void OperationLayer<TDevice>::computeBackwardPass(const int nnState)
    {	
	int timeLength = this->curMaxSeqLength() * this->parallelSequences();

	if (m_lastShot == NN_OPE_LAST_SHOT_MODE1 || m_lastShot == NN_OPE_LAST_SHOT_MODE2 ||
	    m_lastShot == NN_OPE_LAST_SHOT_MODE5 || m_lastShot == NN_OPE_LAST_SHOT_MODE6){

	    // initialize the target gradients buffer
	    thrust::fill(this->precedingLayer().outputErrors().begin(),
			 this->precedingLayer().outputErrors().end(), 0.0);
	    
	    // collect the gradients for each utterance in this parallel block
	    
	    for (int i = 0; i<this->parallelSequences(); i++){
		if (this->m_seqLengthBuffH[i] < 1)
		    continue;
		
		timeLength = this->parallelSequences() * this->m_seqLengthBuffH[i];
		// A trick for parallel training mode: circular move over one vector
		//       A B C D A B C D
		// 0 0 0 1 0 0 0 1 0 0 0          sum As, shift 3
		//   0 0 0 1 0 0 0 1 0 0 0        sum Bs, shift 2
		//     0 0 0 1 0 0 0 1 0 0 0      sum Cs, shift 1
		//       0 0 0 1 0 0 0 1 0 0 0    sum Ds, shift 0
		helpers::Matrix<TDevice> onevec  (&this->m_oneVec, timeLength, 1,
						  (this->parallelSequences() - 1 - i));
		
		helpers::Matrix<TDevice> source  (&this->outputErrors(), this->size(),
						  timeLength);
		
		helpers::Matrix<TDevice> output  (&this->precedingLayer().outputErrors(), 
						  this->size(), 1,
						  (timeLength - this->parallelSequences() + i) *
						  this->size());
		// sum the gradients for a_k
		output.assignProduct(source, false, onevec, false);

		// By default, gathering the gradients and store then to the last frame
		// For bi-directional last-shot (assume size/2 forward, size/2 backward):
		//   1. copy:: move the gradients from last to first frame
		//   2. fill:: set the dimension to zero at the end of the buffer          
		if (m_lastShot == NN_OPE_LAST_SHOT_MODE5 || m_lastShot == NN_OPE_LAST_SHOT_MODE6){
		    long int tempPtr1 = ((timeLength-this->parallelSequences()+i) * this->size() +
					 this->size() / 2);
		    long int tempPtr2 = tempPtr1 + this->size() / 2;
		    thrust::copy(this->precedingLayer().outputErrors().begin() + tempPtr1,
				 this->precedingLayer().outputErrors().begin() + tempPtr2,
				 this->precedingLayer().outputErrors().begin() + this->size()/2);
		    thrust::fill(this->precedingLayer().outputErrors().begin() + tempPtr1,
				 this->precedingLayer().outputErrors().begin() + tempPtr2,
				 0.0);
		}
	    }
	    
	}else if (m_lastShot == NN_OPE_LAST_SHOT_MODE9){

	    // output_ave = \sum_t=1^T input_t
	    // next_layer_input_t = output_ave, for t \in[1,T]
	    // Therefore, \partial E / \partial input_t
	    //    = \sum_m=1^T \partial E / \partial next_layer_input_m *
	    //         \partial next_layer_input_m / \partial input_t
	    //    = \sum_m=1^T \partial E / \partial next_layer_input_m *
	    //         \partial output_ave / \partial input_t
	    //    = 1/T * \sum_m=1^T \partial E / \partial next_layer_input_m
	    //
	    // Back-propagation uses the same procedure as forward propagation
	    
	    // For each utterance in the parallel mode
	    for (int i = 0; i<this->parallelSequences(); i++){
		if (this->m_seqLengthBuffH[i] < 1){
		    // skip over dummy sentences
		    continue;
		}
		
		// step1. prepare the [1/T, ..., 1/T]
		int tmpUttL = this->parallelSequences() * this->m_seqLengthBuffH[i];

		{{
		    thrust::fill(m_oneVec.begin(), m_oneVec.end(), 0.0);
		    
		    internal::changeOneVec fn1;
		    fn1.sentL       = (real_t)this->m_seqLengthBuffH[i];
		    fn1.parallelNum = this->parallelSequences();

		    thrust::for_each(
                        thrust::make_zip_iterator(
			    thrust::make_tuple(m_oneVec.begin(),
					       thrust::counting_iterator<int>(0))),
			thrust::make_zip_iterator(
			    thrust::make_tuple(m_oneVec.begin()                  + tmpUttL,
					       thrust::counting_iterator<int>(0) + tmpUttL)),
			fn1);
		}}


		// step2. 
		// get the average over gradients, and save to the first frame
		helpers::Matrix<TDevice> onevec (&this->m_oneVec,
						 tmpUttL, 1,
						 (this->parallelSequences() - 1 - i));
		
		helpers::Matrix<TDevice> source (&this->outputErrors(),
						 this->size(), tmpUttL);
		
		helpers::Matrix<TDevice> output (&this->precedingLayer().outputErrors(), 
						  this->size(), 1,
						  i * this->size());
		// sum the gradients for a_k
		output.assignProduct(source, false, onevec, false);
	    }

	    // step3. 
	    // duplicate the output to all the frames
	    {{
		internal::duplicateSentVec fn2;
		fn2.featureDim = this->size();
		fn2.paralSeqNm = this->parallelSequences();
		fn2.sourceData = helpers::getRawPointer(this->precedingLayer().outputErrors());
		fn2.patTypes   = helpers::getRawPointer(this->patTypes());
	    
		int n = timeLength * this->size();
		thrust::for_each(
                 thrust::make_zip_iterator(
		  thrust::make_tuple(this->precedingLayer().outputErrors().begin(),
				     thrust::counting_iterator<int>(0))),
		 thrust::make_zip_iterator(
		  thrust::make_tuple(this->precedingLayer().outputErrors().begin() + n,
				     thrust::counting_iterator<int>(0)             + n)),
		 fn2);
	    }}


	}else if (m_lastShot == NN_OPE_LAST_SHOT_MODE3 || m_lastShot == NN_OPE_LAST_SHOT_MODE4 ||
		  m_lastShot == NN_OPE_LAST_SHOT_MODE7 || m_lastShot == NN_OPE_LAST_SHOT_MODE8 ||
		  m_lastShot == NN_OPE_LAST_SHOT_MODE10){
	    
	    thrust::fill(this->precedingLayer().outputErrors().begin(),
			 this->precedingLayer().outputErrors().end(), 0.0);
	    
	    if (m_lastShot == NN_OPE_LAST_SHOT_MODE3 || m_lastShot == NN_OPE_LAST_SHOT_MODE7){
		// use last shot mode based on segmental boundary
		internal::lastShotForwardSegBoundaryGradMode3 fn1;
		fn1.featureDim  = this->size();
		fn1.paralSeqNm  = this->parallelSequences();
		fn1.lastShotOp  = this->m_lastShot;
		fn1.segBoundary = helpers::getRawPointer(m_segBoundaryD);
		fn1.targetData  = helpers::getRawPointer(this->precedingLayer().outputErrors());
		fn1.patTypes    = helpers::getRawPointer(this->patTypes());
	    
		int n = timeLength * this->size();
		thrust::for_each(
                 thrust::make_zip_iterator(
		  thrust::make_tuple(this->outputErrors().begin(),
				     thrust::counting_iterator<int>(0))),
		 thrust::make_zip_iterator(
		  thrust::make_tuple(this->outputErrors().begin() + n,
				     thrust::counting_iterator<int>(0) + n)),
		 fn1);
		
	    }else{
		// sum the gradients for the one use segmental boundary
		internal::lastShotForwardSegBoundaryGradMode4 fn1;
		fn1.featureDim  = this->size();
		fn1.paralSeqNm  = this->parallelSequences();
		fn1.lastShotOp  = this->m_lastShot;
		fn1.segBoundary = helpers::getRawPointer(m_segBoundaryD);
		fn1.sourceData  = helpers::getRawPointer(this->outputErrors());
		fn1.patTypes    = helpers::getRawPointer(this->patTypes());
	    
		int n = timeLength * this->size();
		thrust::for_each(
                 thrust::make_zip_iterator(
		  thrust::make_tuple(this->precedingLayer().outputErrors().begin(),
				     thrust::counting_iterator<int>(0))),
		 thrust::make_zip_iterator(
		  thrust::make_tuple(this->precedingLayer().outputErrors().begin() + n,
				     thrust::counting_iterator<int>(0) + n)),
		 fn1);
	    }
	}else if (m_dropoutRate > 0){
	    
	    // inverted dropout data
	    int n = timeLength * this->size();
	    
	    if (nnState == NN_STATE_GAN_NOGAN_TRAIN){
		// Training stage
		
		internal::invertedDropout fn1;
		fn1.curLayerSize = this->size();
		fn1.threshold    = m_dropoutRate;
		
		fn1.noise        = helpers::getRawPointer(m_noiseInput);
		fn1.patTypes     = helpers::getRawPointer(this->patTypes());
		//fn1.sourceData   = helpers::getRawPointer(this->outputErrors());
		
		thrust::for_each(
                 thrust::make_zip_iterator(
			thrust::make_tuple(this->precedingLayer().outputErrors().begin(),
					   this->outputErrors().begin(),
					   thrust::counting_iterator<int>(0))),
		 thrust::make_zip_iterator(
			thrust::make_tuple(this->precedingLayer().outputErrors().begin() + n,
					   this->outputErrors().begin()                  + n,
					   thrust::counting_iterator<int>(0)             + n)),
		 fn1);
		
	    }
	    
	}else if (m_changeTimeRes){

	    internal::timeResolutionChangeGrad fn1;
	    fn1.inputRes  = this->precedingLayer().getResolution();
	    fn1.outputRes = this->getResolution();
	    fn1.layerSize = this->size();
	    fn1.parallel  = this->parallelSequences();
	    fn1.sourceData  = helpers::getRawPointer(this->outputErrors());
	    fn1.patTypes    = helpers::getRawPointer(this->precedingLayer().patTypes());
	    
	    int n = (this->precedingLayer().curMaxSeqLength()    *
		     this->precedingLayer().parallelSequences()  * this->precedingLayer().size());
	    
	    thrust::for_each(
              thrust::make_zip_iterator(
		thrust::make_tuple(this->precedingLayer().outputErrors().begin(),
				   thrust::counting_iterator<int>(0))),
	      thrust::make_zip_iterator(
		thrust::make_tuple(this->precedingLayer().outputErrors().begin() + n,
				     thrust::counting_iterator<int>(0) + n)),
	       fn1);
	    
	    
	}else if (m_freqDim >= 0){
	    // do nothing
	    
	}else if (m_positional_code_mode >= 0){
	    // do nothing
	    
	}else if (m_reverse_grad >=0){
	    
	    thrust::transform(this->outputErrors().begin(),
			      this->outputErrors().end(), 
			      thrust::make_constant_iterator(-1.0 * m_reverse_grad),
			      this->precedingLayer().outputErrors().begin(),
			      thrust::multiplies<real_t>());
	    
	}else if (m_dimExpand){

	    // expand the dimensions
	    {
	    	internal::dimensionExpandGrad fn1;
		fn1.previousDim = this->precedingLayer().size();
		fn1.expandedDim = this->size();
		fn1.expandRatio = m_dimExpand;
		fn1.patTypes    = helpers::getRawPointer(this->patTypes());
		fn1.inputGrad   = helpers::getRawPointer(this->outputErrors());
		
		thrust::for_each(
		  thrust::make_zip_iterator(
		    thrust::make_tuple(this->precedingLayer().outputErrors().begin(),
				       thrust::counting_iterator<int>(0))),
		  thrust::make_zip_iterator(
		    thrust::make_tuple(this->precedingLayer().outputErrors().begin()
				       + this->precedingLayer().size() * timeLength,
				 thrust::counting_iterator<int>(0)
				       + this->precedingLayer().size() * timeLength)),
		  fn1);
	    }

	    
	}else if (m_F02UV){
	    thrust::fill(this->precedingLayer().outputErrors().begin(),
			 this->precedingLayer().outputErrors().end(), 0.0);
	}else{
	    
	    if (m_outDupRate > 1){
		internal::outDuplicationGradOperation fn1;
		fn1.featureDim = this->size();
		fn1.resolution = m_outDupRate;
		fn1.maxTimeLength = timeLength;
		fn1.dataMatrix = helpers::getRawPointer(this->outputErrors());	
		fn1.parall     = this->parallelSequences();
		fn1.patTypes   = helpers::getRawPointer(this->patTypes());
		
		int n = timeLength * this->size();
		thrust::for_each(
                 thrust::make_zip_iterator(
		  thrust::make_tuple(this->outputErrors().begin(),
				     thrust::counting_iterator<int>(0))),
		 thrust::make_zip_iterator(
		  thrust::make_tuple(this->outputErrors().begin()      + n,
				     thrust::counting_iterator<int>(0) + n)),
		 fn1);
	    }
	
	    {
		// Although it is tricky to use the same function as in computeForwardPass,
		// it works by constraining the curLayerSize and preLayerSize
	    internal::fillOutputVec fn;
	    fn.curLayerSize = this->precedingLayer().size();
	    fn.preLayerSize = this->size();
	    fn.noiseDim     = m_noiseSize;
	    fn.flagForward  = false;
	    fn.preShift  = 0;
	    
	    fn.weights   = helpers::getRawPointer(m_setScaleVec_D);
	    fn.shifts    = helpers::getRawPointer(m_setBiasVec_D);
	    fn.stopGrad  = helpers::getRawPointer(m_stopGradVec_D);
	    fn.binarize  = helpers::getRawPointer(m_setBinarizeVec_D);
	    
	    fn.preOutput = helpers::getRawPointer(this->outputErrors());
	    fn.noiseData = helpers::getRawPointer(this->m_noiseInput);
	    fn.patTypes  = helpers::getRawPointer(this->patTypes());
	    
	    int n = timeLength * this->precedingLayer().size();
	    thrust::for_each(
               thrust::make_zip_iterator(
		thrust::make_tuple(this->precedingLayer().outputErrors().begin(),
				     thrust::counting_iterator<int>(0))),
	       thrust::make_zip_iterator(
		thrust::make_tuple(this->precedingLayer().outputErrors().begin() + n,
				     thrust::counting_iterator<int>(0)           + n)),
	       fn);
	    }
	}

    }


    template <typename TDevice>
    void OperationLayer<TDevice>::computeBackwardPass(const int timeStep, const int nnState)
    {
	
	int timeLength = this->curMaxSeqLength() * this->parallelSequences();
	int effTimeStart = timeStep       * this->parallelSequences();
	int effTimeEnd   = (timeStep + 1) * this->parallelSequences();

	if (this->getSaveMemoryFlag())
	    throw std::runtime_error("Memory save mode should be turned off");

	
	if (m_lastShot == NN_OPE_LAST_SHOT_MODE1 || m_lastShot == NN_OPE_LAST_SHOT_MODE2 ||
	    m_lastShot == NN_OPE_LAST_SHOT_MODE5 || m_lastShot == NN_OPE_LAST_SHOT_MODE6 ||
	    m_lastShot == NN_OPE_LAST_SHOT_MODE9 ||
	    m_lastShot == NN_OPE_LAST_SHOT_MODE3 || m_lastShot == NN_OPE_LAST_SHOT_MODE4 ||
	    m_lastShot == NN_OPE_LAST_SHOT_MODE7 || m_lastShot == NN_OPE_LAST_SHOT_MODE8){
	    
	    // these modes cannot support online training
	    
	}else if (m_lastShot == NN_OPE_LAST_SHOT_MODE10){
	    // to be implemented
	    
	}else if (m_dropoutRate > 0){
	    
	    // inverted dropout data
	    
	    if (nnState == NN_STATE_GAN_NOGAN_TRAIN){
		// Training stage
		
		internal::invertedDropout fn1;
		fn1.curLayerSize = this->size();
		fn1.threshold    = m_dropoutRate;
		
		fn1.noise        = helpers::getRawPointer(m_noiseInput);
		fn1.patTypes     = helpers::getRawPointer(this->patTypes());
		//fn1.sourceData   = helpers::getRawPointer(this->outputErrors());
		
		thrust::for_each(
                 thrust::make_zip_iterator(
		  thrust::make_tuple(
		    this->precedingLayer().outputErrors().begin() + this->size() * effTimeStart,
		    this->outputErrors().begin()                  + this->size() * effTimeStart,
		    thrust::counting_iterator<int>(0)             + this->size() * effTimeStart)),
		 thrust::make_zip_iterator(
		  thrust::make_tuple(
		    this->precedingLayer().outputErrors().begin() + this->size() * effTimeEnd,
		    this->outputErrors().begin()                  + this->size() * effTimeEnd,
		    thrust::counting_iterator<int>(0)             + this->size() * effTimeEnd)),
		 fn1);
	    }
	    
	}else if (m_changeTimeRes){

	    internal::timeResolutionChangeGrad fn1;
	    fn1.inputRes  = this->precedingLayer().getResolution();
	    fn1.outputRes = this->getResolution();
	    fn1.layerSize = this->size();
	    fn1.parallel  = this->parallelSequences();
	    fn1.sourceData  = helpers::getRawPointer(this->outputErrors());
	    fn1.patTypes    = helpers::getRawPointer(this->precedingLayer().patTypes());
	    	    
	    thrust::for_each(
              thrust::make_zip_iterator(
		thrust::make_tuple(
		 this->precedingLayer().outputErrors().begin() + this->size() * effTimeStart,
		 thrust::counting_iterator<int>(0)             + this->size() * effTimeStart)),
	      thrust::make_zip_iterator(
		thrust::make_tuple(
		 this->precedingLayer().outputErrors().begin() + this->size() * effTimeEnd,
		 thrust::counting_iterator<int>(0)             + this->size() * effTimeEnd)),
	      fn1);
	    
	}else if (m_freqDim >= 0){
	    // do nothing
	    
	}else if (m_positional_code_mode >= 0){

	}else if (m_reverse_grad >= 0){
	    
	    thrust::transform(this->outputErrors().begin() + effTimeStart,
			      this->outputErrors().begin() + effTimeEnd, 
			      thrust::make_constant_iterator(-1.0 * m_reverse_grad),
			      this->precedingLayer().outputErrors().begin() + effTimeStart,
			      thrust::multiplies<real_t>());

	}else if (m_dimExpand){
	    
	    // expand the dimensions
	    {
	    	internal::dimensionExpandGrad fn1;
		fn1.previousDim = this->precedingLayer().size();
		fn1.expandedDim = this->size();
		fn1.expandRatio = m_dimExpand;
		fn1.patTypes    = helpers::getRawPointer(this->patTypes());
		fn1.inputGrad   = helpers::getRawPointer(this->outputErrors());

		int st = effTimeStart * this->precedingLayer().size();
		int et = effTimeEnd   * this->precedingLayer().size();
		
		thrust::for_each(
		  thrust::make_zip_iterator(
		    thrust::make_tuple(this->precedingLayer().outputErrors().begin() + st,
				       thrust::counting_iterator<int>(0)             + st)),
		  thrust::make_zip_iterator(
		    thrust::make_tuple(this->precedingLayer().outputErrors().begin() + et,
				       thrust::counting_iterator<int>(0)             + et)),
		  fn1);
	    }

	    
	}else{
	    
	    if (m_outDupRate > 1){
		internal::outDuplicationGradOperation fn1;
		fn1.featureDim    = this->size();
		fn1.resolution    = m_outDupRate;
		fn1.maxTimeLength = timeLength;
		fn1.dataMatrix    = helpers::getRawPointer(this->outputErrors());	
		fn1.parall        = this->parallelSequences();
		fn1.patTypes      = helpers::getRawPointer(this->patTypes());
		
		thrust::for_each(
                 thrust::make_zip_iterator(
		    thrust::make_tuple(
		       this->outputErrors().begin()      + this->size() * effTimeStart,
		       thrust::counting_iterator<int>(0) + this->size() * effTimeStart)),
		 thrust::make_zip_iterator(
		  thrust::make_tuple(
		       this->outputErrors().begin()      + this->size() * effTimeEnd,
		       thrust::counting_iterator<int>(0) + this->size() * effTimeEnd)),
		 fn1);
	    }
	
	    {
		
		internal::fillOutputVec fn;
		fn.curLayerSize = this->precedingLayer().size();
		fn.preLayerSize = this->size();
		fn.noiseDim     = m_noiseSize;
		fn.flagForward  = false;
		fn.preShift  = 0;
	    
		fn.weights   = helpers::getRawPointer(m_setScaleVec_D);
		fn.shifts    = helpers::getRawPointer(m_setBiasVec_D);
		fn.stopGrad  = helpers::getRawPointer(m_stopGradVec_D);
		fn.binarize  = helpers::getRawPointer(m_setBinarizeVec_D);
		
		fn.preOutput = helpers::getRawPointer(this->outputErrors());
		fn.noiseData = helpers::getRawPointer(this->m_noiseInput);
		fn.patTypes  = helpers::getRawPointer(this->patTypes());
		
		int pl_size = this->precedingLayer().size();
		thrust::for_each(
                 thrust::make_zip_iterator(
		   thrust::make_tuple(
		     this->precedingLayer().outputErrors().begin() + pl_size * effTimeStart,
		     thrust::counting_iterator<int>(0)             + pl_size * effTimeStart)),
		 thrust::make_zip_iterator(
		   thrust::make_tuple(
		     this->precedingLayer().outputErrors().begin() + pl_size * effTimeEnd,
		     thrust::counting_iterator<int>(0)             + pl_size * effTimeEnd)),
	       fn);
	    }
	}

    }


    
    template <typename TDevice>
    void OperationLayer<TDevice>::reduceOutputBuffer()
    {
	if (m_lastShot != NN_OPE_LAST_SHOT_TURNOFF ||
	    m_changeTimeRes > 0 || m_outDupRate > 1)
	    throw std::runtime_error("This operator layer doesn't support memory reduce mode");
	
	if (this->precedingLayer().type()=="vae")
	    throw std::runtime_error("This operator layer doesn't support memory reduce mode");  
	
	this->resizeOutputBuffer(this->parallelSequences() * this->size());
	
	this->setSaveMemoryFlag(true);
	printf("\t[mem saved]");
    }
    
    template <typename TDevice>
    int OperationLayer<TDevice>::outputBufPtrBias(const int timeStepTimesParallel,
						  const int nnState)
    {
	if (this->getSaveMemoryFlag()){
	    return timeStepTimesParallel * this->size();
	}else{
	    return 0;
	}
    }	

    template <typename TDevice>
    void OperationLayer<TDevice>::clearAllBuffers()
    {
	this->clearOutputBuffer();
    }

    template <typename TDevice>
    void OperationLayer<TDevice>::resizeAllBuffers(const int timeLength)
    {
	this->resizeOutputBuffer(timeLength * this->size());
    }


    
    template class OperationLayer<Cpu>;
    template class OperationLayer<Gpu>;
    
}



/*
    #1
    struct outDuplicationMatrix
    {
	int size;
	int resolution;
	int parall;
	__host__ __device__ void operator() (const thrust::tuple<real_t&, int> &t) const
	{
	    int rowidx   = t.get<1>() % size;
	    int colidx   = t.get<1>() / size;

	    int rowidx2  = rowidx  / parall;
	    int colidx2  = colidx  / parall;
	    
	    //int paralPos = rowidx  % parall;
	    int blockidx = rowidx2 / resolution;
	    
	    // the matrix is column major
	    // [ 0 0 0 0 0 0 0 0
	    //   ...
	    //   1 1 ... 1 0 0 0  -> resolution-th row
	    //   |--------|
	    //   resolution columns
	    // move right and repeat this pattern

	    // each block
	    if (((rowidx2 % resolution) == (resolution - 1)) &&
		(colidx2 >= (blockidx * resolution)) &&
		(colidx2 <  ((blockidx + 1) * resolution)) &&
		(rowidx % parall) == (colidx % parall))
		t.get<0>() = 1.0/((real_t)resolution);

	    // last row
	    else if (rowidx2 == (size / parall - 1) &&
		     (colidx2 >= (blockidx * resolution)) &&
		     (rowidx % parall) == (colidx % parall))
		t.get<0>() = 1.0/((real_t)resolution);
	    
	}
    };
    
*/
