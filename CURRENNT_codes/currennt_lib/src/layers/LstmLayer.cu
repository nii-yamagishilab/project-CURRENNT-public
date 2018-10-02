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
#   pragma warning (disable: 4244) // thrust/iterator/iterator_adaptor.h(121): warning C4244: '+=' : conversion from '__int64' to 'int', possible loss of data
#endif

#include "LstmLayer.hpp"
#include "../helpers/limitedError.cuh"
#include "../helpers/getRawPointer.cuh"
#include "../helpers/Matrix.hpp"
#include "../helpers/JsonClasses.hpp"
#include "../activation_functions/Logistic.cuh"
#include "../activation_functions/Tanh.cuh"
#include "../Configuration.hpp"

#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/for_each.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>

#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>

#include <climits> // to define CHAR_BIT (need support for CHAR_BIT unequal to 8)

#define DEBUG_CLOCKLSTM 0

namespace internal {
namespace {

    typedef activation_functions::Logistic gate_act_fn_t;
    typedef activation_functions::Tanh     cell_input_act_fn_t;
    typedef activation_functions::Tanh     cell_output_act_fn_t;


    struct ComputeBlockOutputFn
    {
        int    effLayerSize;
        int    prevOutputDistance;
        real_t bias;

        const char   *patTypes;

        const real_t *niBiasWeights;
        const real_t *igBiasWeights;
        const real_t *fgBiasWeights;
        const real_t *ogBiasWeights;

        const real_t *igPeepWeights;
        const real_t *fgPeepWeights;
        const real_t *ogPeepWeights;

	const bool   *skipCRNN;     // whether this step should be skipped
	
        real_t *cellStates;
        real_t *niActs;
        real_t *igActs;
        real_t *fgActs;
        real_t *ogActs;

        __host__ __device__ real_t operator() (const int &outputIdx,
					       const thrust::tuple<bool, bool> &t) const
        {
            // unpack the tuple
            bool firstCall    = t.get<0>();
            bool checkPatType = t.get<1>();

            // check if we can skip the whole calculation because the pattern is a dummy
            // in that case, we set the all values of that pattern to zero
            if (checkPatType) {
                int patIdx = outputIdx / effLayerSize;
                if (patTypes[patIdx] == PATTYPE_NONE) {
                    if (prevOutputDistance > 0)
                        cellStates[outputIdx] = 0;
                    return 0;
                }
            }

            // calculate indices
            int blockIdx = outputIdx % effLayerSize;

	    if (skipCRNN != NULL && skipCRNN[blockIdx] && !firstCall){
		// CLLSTM, Skip update
		// cell state copy from previous step
		cellStates[outputIdx] = cellStates[outputIdx + prevOutputDistance];

		
		/*
		niActs[outputIdx] = 0.0;
		igActs[outputIdx] = 0.0;
		ogActs[outputIdx] = 0.0;
		fgActs[outputIdx] = 1.0; // This is 1.0 because of the gradient on cellstate
		                         // In ComputeBlockErrorFn, fgActs[outputIdx-prev] is used
		                         // If that LSTM unit is skipped, cellStateErr should
		                         // be directly propagated
					 */

		niActs[outputIdx] = 0.0; // zero 
		igActs[outputIdx] = 0.0; // zero
		// use previous ogActs (after the activation function)
		ogActs[outputIdx] = ogActs[outputIdx + prevOutputDistance];
		// the same reason above
		fgActs[outputIdx] = 1.0;

		
		// hidden output 'copies' from previous state (computed again)
		return (cell_output_act_fn_t::fn(cellStates[outputIdx + prevOutputDistance]) *
			ogActs[outputIdx + prevOutputDistance]);
	    }else{
		// Normal LSTM computation
		
		// load the niag activations
		real_t niAct = niActs[outputIdx];
		real_t igAct = igActs[outputIdx];
		real_t fgAct = fgActs[outputIdx];
		real_t ogAct = ogActs[outputIdx];
		
		// add bias activations
		niAct += bias * niBiasWeights[blockIdx];
		igAct += bias * igBiasWeights[blockIdx];
		fgAct += bias * fgBiasWeights[blockIdx];
		ogAct += bias * ogBiasWeights[blockIdx];
		
		// add activation from peephole weights
		if (!firstCall) {
		    real_t prevCellState = cellStates[outputIdx + prevOutputDistance];
		    
		    igAct += prevCellState * igPeepWeights[blockIdx];
		    fgAct += prevCellState * fgPeepWeights[blockIdx];
		}
		
		// apply the activation functions
		niAct = cell_input_act_fn_t::fn(niAct);
		igAct = gate_act_fn_t      ::fn(igAct);
		fgAct = gate_act_fn_t      ::fn(fgAct);
		
		// store the niag activations
		niActs[outputIdx] = niAct;
		igActs[outputIdx] = igAct;
		fgActs[outputIdx] = fgAct;
		
		// calculate the cell state and store the result
		real_t cellState = niAct * igAct;
		
		if (!firstCall)
		    cellState += cellStates[outputIdx + prevOutputDistance] * fgAct;

		cellStates[outputIdx] = cellState;

		// calculate the output gate activation and store the result
		ogAct += cellState * ogPeepWeights[blockIdx];
		ogAct = gate_act_fn_t::fn(ogAct);
		ogActs[outputIdx] = ogAct;
		
		// calculate the block output
		real_t output = cell_output_act_fn_t::fn(cellState) * ogAct;

		return output;
	    }
        }
    };

    struct ResortOutputsFn
    {
        int layerSize;
        int effLayerSize;

        const real_t *fwOutputs;
        const real_t *bwOutputs;

        __host__ __device__ real_t operator() (const int &outputIdx) const
        {
            // calculate indices
            int patIdx = outputIdx / layerSize;
            int valIdx = outputIdx % layerSize;
            int offset = patIdx * effLayerSize + valIdx;

            // store the value
            if (valIdx < effLayerSize)
                return fwOutputs[offset];
            else
                return bwOutputs[offset - effLayerSize];
        }
    };

    struct ResortOutputErrorsFn
    {
        int layerSize;
        int effLayerSize;

        real_t *fwOutputErrors;
        real_t *bwOutputErrors;

        __host__ __device__ void operator() (const thrust::tuple<const real_t&, int> &t) const
        {
            // unpack the tuple
            real_t outputErr = t.get<0>();
            int    outputIdx = t.get<1>();

            // calculate indices
            int patIdx = outputIdx / layerSize;
            int valIdx = outputIdx % layerSize;
            int offset = patIdx * effLayerSize + valIdx;

            // store the value
            if (valIdx < effLayerSize)
                fwOutputErrors[offset] = outputErr;
            else
                bwOutputErrors[offset - effLayerSize] = outputErr;
        }
    };

    struct ComputeBlockErrorsFn
    {
        int effLayerSize;
        int prevOutputDistance;

        const char *patTypes;

        const real_t *igPeepWeights;
        const real_t *fgPeepWeights;
        const real_t *ogPeepWeights;

        const real_t *cellStates;
        const real_t *niActs;
        const real_t *igActs;
        const real_t *fgActs;
        const real_t *ogActs;
	const bool   *skipCRNN;
	
        real_t *cellStateErrors;
        real_t *niDeltas;
        real_t *igDeltas;
        real_t *fgDeltas;
        real_t *ogDeltas;

        __host__ __device__ void operator() (
		const thrust::tuple<const real_t&, int, bool, bool, bool> &t) const
        {
            // unpack the tuple
            real_t outputErr    = t.get<0>();
            int    outputIdx    = t.get<1>();
            bool   firstCall    = t.get<2>();
            bool   lastCall     = t.get<3>();
            bool   checkPatType = t.get<4>();

            // check if we can skip the whole calculation because the pattern is a dummy
            // in that case, we set all values of that pattern to zero
            if (checkPatType) {
                int patIdx = outputIdx / effLayerSize;
                if (patTypes[patIdx] == PATTYPE_NONE) {
                    niDeltas       [outputIdx] = 0;
                    igDeltas       [outputIdx] = 0;
                    fgDeltas       [outputIdx] = 0;
                    ogDeltas       [outputIdx] = 0;
                    cellStateErrors[outputIdx] = 0;
                    return;
                }
            }

            // calculate indices
            int blockIdx = outputIdx % effLayerSize;

	    if (skipCRNN != NULL && skipCRNN[blockIdx]){

		/*
		// store the niag deltas and the cell state error
		// ig, fg, set to zero, so that a normal LSTM unit precedings a skipped unit can
		//  calculate the cellStateError correctly (see cellStateErr below)
		// og set to zero, ni set to outputErr. Thus, only one outputErr will be
		//  accumulated to the previous unit (during hidden-to-hidden multiplication)
		niDeltas       [outputIdx] = outputErr;
		igDeltas       [outputIdx] = 0.0;
		fgDeltas       [outputIdx] = 0.0;
		ogDeltas       [outputIdx] = 0.0;
		*/

		fgDeltas       [outputIdx] = 0.0;
		igDeltas       [outputIdx] = 0.0;
		niDeltas       [outputIdx] = 0.0; 
		ogDeltas       [outputIdx] = 0.0; 

		real_t ogAct     = ogActs      [outputIdx]; // output after activation function
		real_t cellState = cellStates  [outputIdx]; // cellState 
		
		real_t ogDelta   = (gate_act_fn_t::deriv(ogAct) *
				    cell_output_act_fn_t::fn(cellState) * outputErr) + ogDeltas[outputIdx];
		
		real_t cellStateErr = ogAct *
		    cell_output_act_fn_t::deriv(cell_output_act_fn_t::fn(cellState)) * outputErr;

		if (!firstCall){
		    //  add to the cellErrors from previous (next) step
		    real_t nextFgAct        = fgActs         [outputIdx - prevOutputDistance];
		    real_t nextCellStateErr = cellStateErrors[outputIdx - prevOutputDistance];
		    real_t nextIgDelta      = igDeltas       [outputIdx - prevOutputDistance];
		    real_t nextFgDelta      = fgDeltas       [outputIdx - prevOutputDistance];
		    
		    real_t igPeepWeight = igPeepWeights[blockIdx];
		    real_t fgPeepWeight = fgPeepWeights[blockIdx];
		
		    cellStateErr += (nextFgAct    * nextCellStateErr +
				     igPeepWeight * nextIgDelta      +
				     fgPeepWeight * nextFgDelta);
		    
		}
		if (!lastCall){
		    // accumulate the graidients of outgate to the previous time step
		    ogDeltas[outputIdx + prevOutputDistance] = ogDelta;
		}
		cellStateErrors[outputIdx] = cellStateErr;
		// //cellStateErrors[outputIdx] = cellStateErrors[outputIdx - prevOutputDistance];
		//else{
		//cellStateErrors[outputIdx] = 0.0;
		//}
		
	    }else{

		// load the niag activations, the cell state and the output error
		real_t niAct     = niActs      [outputIdx]; // output after activation function
		real_t igAct     = igActs      [outputIdx]; // output after activation function
		real_t ogAct     = ogActs      [outputIdx]; // output after activation function
		real_t cellState = cellStates  [outputIdx]; // cellState 
		
		// calculate the output gate delta (before activation)
		// NOTE: interestly, possibly due to the compiler and CUDA,
		//       the order of the variables below slightly influences the result
		//       I use this order so that it is consistent with the original code
		real_t ogDelta   = ((gate_act_fn_t::deriv(ogAct) *
				    cell_output_act_fn_t::fn(cellState) * outputErr)
				    + ogDeltas[outputIdx]);

		// calculate the cell state error
		real_t ogPeepWeight = ogPeepWeights[blockIdx];
		real_t cellStateErr = ogAct *
		    cell_output_act_fn_t::deriv(cell_output_act_fn_t::fn(cellState)) * outputErr  +
		    ogPeepWeight * ogDelta;
	    
		if (!firstCall) {
		    //  add to the cellErrors from previous (next) step
		    real_t nextFgAct        = fgActs         [outputIdx - prevOutputDistance];
		    real_t nextCellStateErr = cellStateErrors[outputIdx - prevOutputDistance];
		    real_t nextIgDelta      = igDeltas       [outputIdx - prevOutputDistance];
		    real_t nextFgDelta      = fgDeltas       [outputIdx - prevOutputDistance];
		    
		    real_t igPeepWeight = igPeepWeights[blockIdx];
		    real_t fgPeepWeight = fgPeepWeights[blockIdx];
		
		    cellStateErr += (nextFgAct    * nextCellStateErr +
				     igPeepWeight * nextIgDelta      +
				     fgPeepWeight * nextFgDelta);
		}
	    
		// calculate the net input delta
		real_t niDelta = igAct * cell_input_act_fn_t::deriv(niAct) * cellStateErr;

		// calculate the forget gate delta
		real_t fgDelta = 0;

		if (!lastCall) {
		    real_t fgAct         = fgActs    [outputIdx];
		    real_t prevCellState = cellStates[outputIdx + prevOutputDistance];

		    fgDelta = gate_act_fn_t::deriv(fgAct) * prevCellState * cellStateErr;
		}

		// calculate the input gate delta
		real_t igDelta = gate_act_fn_t::deriv(igAct) * niAct * cellStateErr;

		// store the niag deltas and the cell state error
		niDeltas       [outputIdx] = helpers::limitedError(niDelta);
		igDeltas       [outputIdx] = helpers::limitedError(igDelta);
		fgDeltas       [outputIdx] = helpers::limitedError(fgDelta);
		ogDeltas       [outputIdx] = helpers::limitedError(ogDelta);
		cellStateErrors[outputIdx] = cellStateErr;
	    }
        }
    };

    struct ComputeWeightUpdateFn
    {
        int    layerSize;
        int    effLayerSize;
        int    precLayerSize;
        int    timestepDistance;
        int    parallelSequences;
        int    patternsCount;
        int    biasWeightsOffset;
        int    internalWeightsOffset;
        int    peepholeWeightsOffset;
        real_t bias;

        const real_t *plOutputs;
        const real_t *fwOutputs;   
        const real_t *bwOutputs;   
        const real_t *fwCellStates;
        const real_t *bwCellStates;
        const real_t *fwNiDeltas;  
        const real_t *bwNiDeltas;  
        const real_t *fwIgDeltas;  
        const real_t *bwIgDeltas;  
        const real_t *fwFgDeltas;  
        const real_t *bwFgDeltas;  
        const real_t *fwOgDeltas;  
        const real_t *bwOgDeltas;  

        __host__ __device__ real_t operator() (const int &weightIdx) const
        {
            // determine the weight type
            // 
            // weightType = 0bXXYY with XX = {input, bias, internal, peephole}
            //                     and  YY = {NI, IG, FG, OG}
            //
            // weightType = 0b0000 ( 0): NI input weight
            //              0b0001 ( 1): IG input weight
            //              0b0010 ( 2): FG input weight
            //              0b0011 ( 3): OG input weight
            //              0b0100 ( 4): NI bias weight
            //              0b0101 ( 5): IG bias weight
            //              0b0110 ( 6): FG bias weight
            //              0b0111 ( 7): OG bias weight
            //              0b1000 ( 8): NI internal weight
            //              0b1001 ( 9): IG internal weight
            //              0b1010 (10): FG internal weight
            //              0b1011 (11): OG internal weight
            //              0b1100 (12): not used
            //              0b1101 (13): IG peephole weight
            //              0b1110 (14): FG peephole weight
            //              0b1111 (15): OG peephole weight
            int inwc = layerSize * precLayerSize;
            int biwc = layerSize;
            int itwc = layerSize * effLayerSize;
            int pewc = layerSize;

            int weightType = (int)(weightIdx >= 0                     + 1 * inwc) +
                             (int)(weightIdx >= 0                     + 2 * inwc) +
                             (int)(weightIdx >= 0                     + 3 * inwc) +
                             (int)(weightIdx >= 0                     + 4 * inwc) + 
                             (int)(weightIdx >= biasWeightsOffset     + 1 * biwc) +
                             (int)(weightIdx >= biasWeightsOffset     + 2 * biwc) +
                             (int)(weightIdx >= biasWeightsOffset     + 3 * biwc) +
                             (int)(weightIdx >= biasWeightsOffset     + 4 * biwc) +
                             (int)(weightIdx >= internalWeightsOffset + 1 * itwc) +
                             (int)(weightIdx >= internalWeightsOffset + 2 * itwc) +
                             (int)(weightIdx >= internalWeightsOffset + 3 * itwc) +
                             (int)(weightIdx >= internalWeightsOffset + 4 * itwc) * 2 +
                             (int)(weightIdx >= peepholeWeightsOffset + 1 * pewc) +
                             (int)(weightIdx >= peepholeWeightsOffset + 2 * pewc);

            int weightTypeX = weightType & 0xC;
            int weightTypeY = weightType & 0x3;

            // calculate indices, offsets and increments 
            const real_t *offOutputs;
            int           tgtBlockIdx;
            int           offOutputsInc;
            bool          skipFirstPattern = false;
            bool          skipLastPattern  = false;
            bool          isBwStateWeight;

            switch (weightTypeX) {
            // input weight
            case 0x0: 
                {{
                    // calculate indices
                    int inputWeightIdx = weightIdx;

		    // dimension corresponds to the input data
                    int plBlockIdx     = inputWeightIdx % precLayerSize;

		    // dimension corresponds to the output of the input gate
		    // Note: there are four gates, blockIdx is the relative dimension
		    //       within the gate.
		    // 
		    //       we can also calculate
		    //       inputWeightIdx = (inputWeightIdx - weightTypeY * biasWeightsOffset/4)
		    //       plBlockIdx     = inputWeightIdx % precLayerSize
		    //       blockIdx       = inputWeightIdx / precLayerSize
		    //
		    //       or
		    //       blockIdx  = inputWeightIdx/preclayerSize - weightTypeY * layerSize
                    int blockIdx       = ((inputWeightIdx - weightTypeY * (biasWeightsOffset/4)) /
					  precLayerSize);
		    
                    // check if we calculate backward state weights and adjust the block index
                    isBwStateWeight = (blockIdx >= effLayerSize);
                    if (isBwStateWeight)
                        blockIdx -= effLayerSize;
                    
                    // set values for the loop below
                    tgtBlockIdx   = blockIdx;
                    offOutputs    = &plOutputs[plBlockIdx];
                    offOutputsInc = precLayerSize;
                }}
                break;

            // bias weight
            case 0x4: 
                {{
                    // calculate indices
                    int biasWeightIdx = weightIdx - biasWeightsOffset;
                    int blockIdx      = biasWeightIdx - weightTypeY * layerSize;

                    // check if we calculate backward state weights and adjust the block index
                    isBwStateWeight = (blockIdx >= effLayerSize);
                    if (isBwStateWeight)
                        blockIdx -= effLayerSize;

                    // set values for the loop below
                    tgtBlockIdx   = blockIdx;
                    offOutputs    = NULL;
                    offOutputsInc = 0;
                }}
                break;

            // internal weight
            case 0x8: 
                {{
                    // calculate indices
                    int internalWeightIdx = weightIdx - internalWeightsOffset;
                    int srcBlockIdx       = internalWeightIdx % effLayerSize;
                    int blockIdx          = (internalWeightIdx / effLayerSize -
					     weightTypeY * layerSize);

                    // check if we calculate backward state weights and adjust the block index
                    isBwStateWeight = (blockIdx >= effLayerSize);
                    if (isBwStateWeight)
                        blockIdx -= effLayerSize;

                    // set values for the loop below
                    tgtBlockIdx   = blockIdx;
                    offOutputs    = (isBwStateWeight ?
				     &bwOutputs[srcBlockIdx] : &fwOutputs[srcBlockIdx]);
                    offOutputsInc = effLayerSize;

                    if (isBwStateWeight) {
                        offOutputs += timestepDistance;
                        skipLastPattern = true;
                    }
                    else {
                        offOutputs -= timestepDistance;
                        skipFirstPattern = true;
                    }
                }}
                break;

            // peephole weight
            default: 
                {{
                    // calculate indices
                    int peepholeWeightIdx = weightIdx - peepholeWeightsOffset;
                    int blockIdx          = peepholeWeightIdx - (weightTypeY-1) * layerSize;
                    
                    // check if we calculate backward state weights and adjust the block index
                    isBwStateWeight = (blockIdx >= effLayerSize);
                    if (isBwStateWeight)
                        blockIdx -= effLayerSize;

                    // select the appropriate cell states and adjust the block index
                    const real_t *cellStates = (isBwStateWeight ? bwCellStates : fwCellStates);
                    
                    // set the timeshift
                    int timeShift;
                    if (weightTypeY == 0x3) {
                        timeShift = 0;
                    }
                    else {
                        if (isBwStateWeight) {
                            timeShift       = timestepDistance;
                            skipLastPattern = true;
                        }
                        else {
                            timeShift        = -timestepDistance;
                            skipFirstPattern = true;
                        }
                    }

                    // set values for the loop below
                    tgtBlockIdx   = blockIdx;
                    offOutputs    = &cellStates[blockIdx + timeShift];
                    offOutputsInc = effLayerSize;
                }}
                break;
            }

            // determine the start of the delta values
            const real_t *niagDeltasLut[] = {
                fwNiDeltas,
                fwIgDeltas,
                fwFgDeltas,
                fwOgDeltas,
                bwNiDeltas,
                bwIgDeltas,
                bwFgDeltas,
                bwOgDeltas
            };

            // calculate the weight update over all patterns            
            const real_t *offDeltas =
		&niagDeltasLut[weightTypeY + (isBwStateWeight ? 4 : 0)][tgtBlockIdx];

            if (skipFirstPattern) {
                offOutputs += parallelSequences * offOutputsInc;
                offDeltas  += parallelSequences * effLayerSize;
            }

            int numPatterns = patternsCount;
            if (skipFirstPattern || skipLastPattern)
                numPatterns -= parallelSequences;

            real_t wu = 0;
            for (int i = 0; i < numPatterns; ++i) {
                wu += (offOutputs ? *offOutputs : bias) * *offDeltas;
                    
                offOutputs += offOutputsInc;
                offDeltas  += effLayerSize;
            }

            return wu;
        }
    };

    struct SetH2HMatrixLSTM{
	// Create the H2Hmatrix for each time step
	// The created matrix is a lower-triangle (block) matrix
	int    *bandConfig;
	int     bandNum;
	real_t *sourceW;
	real_t *targetW;
	int     featDim;
	int     matrixSize;
	
	__host__ __device__ void operator() (const int idx) const{
	    int bandIdx     = idx / matrixSize + 1;
	    int sourcePos   = idx % matrixSize; 
	    int rows        = sourcePos % featDim;
	    int cols        = sourcePos / featDim;
	    
	    int colStart    = 0;
	    int colEnd      = 0;
	    int tmp         = 0b01;
	    bool flag       = false;
	    
	    // given a point in one of the H2H matrix
	    // for the matrix bands that are non-zero in this H2H matrix
	    //     check if this point locates in the band (and in the lower-triangle area)
	    //           if yes, this point is non-zero, copy the value for it
	    //           if not
	    //                diagonal point ? 1.0 : 0.0
	    for (int band = 0; band<bandNum; band++){
		if (bandIdx  & (tmp << band)){ // bit operation
		    colStart = (band > 0)?(bandConfig[2*band-1]):(colStart);
		    colEnd   = bandConfig[2*band+1];
		    if (cols >= colStart && cols < colEnd && rows >= colStart){
			*(targetW + idx) = *(sourceW + sourcePos);
			flag = true;
			break;
		    }
		}
	    }
	    if (flag == false){
		*(targetW + idx) = (rows==cols) ? (1.0):(0.0);
	    }
	}
    };

    struct CreateSkipVecForContextDepedentCLLSTM{	
	int         bandNum;
	int         featDim;
	bool       *skipVec;
	bool       *skipVec2;
	const char *clockTime;
	const int  *bandConfig;
	
	__host__ __device__ void operator() (const thrust::tuple<const int&, bool> &t) const{
	    int idx      = t.get<0>();
	    int dimIdx   = idx % featDim;
	    int timeIdx  = idx / featDim;
	    
	    int DimStart    = 0;
	    int DimEnd      = 0;
	    int tmp         = 0b01;
	    
	    *(skipVec+idx)  = true;
	    if (skipVec2 != NULL) *(skipVec2+idx)  = true;
	    
	    for (int band = 0; band<bandNum; band++){
		DimStart = (band > 0)?(bandConfig[2*band-1]):(DimStart);
		DimEnd   = bandConfig[2*band+1];
		if (dimIdx >= DimStart && dimIdx < DimEnd){
		    // find the band for this dimension
		    // check whether this band will be updated at this time step
		    // if yes, turn the skipFlag to false, i.e. update the hidden feature
		    if (clockTime[timeIdx] & (tmp << band)){ 
		    	*(skipVec+idx)  = false;
			if (skipVec2 != NULL) *(skipVec2+idx)  = false;
		    }
		    break;
		}
	    }
	}
    };

    struct CleanUnitDeltasClockLSTM{
	const bool   *skipCRNN;     // whether this step should be skipped
        real_t       *unitDeltas;   // 
	__host__ __device__ void operator() (const thrust::tuple<const int&, bool> &t) const {
	    int idx = t.get<0>();
	    if (skipCRNN[idx])
		*(unitDeltas+idx) = 0.0;
	}
    };

} // anonymous namespace
} // namespace internal


namespace layers {

    // Parse the option of ClockRNN
    // input: options, layer size
    // change: m_crStep
    // return: number of possible Hidden2Hidden Matrix
    int ReadClockLSTMOptions(const std::string options, Cpu::int_vector &m_crStep, const int size)
    {
	// read in the option
	std::vector<std::string> tempArgs;
	boost::split(tempArgs, options, boost::is_any_of("_"));
	if ((tempArgs.size() % 2) != 0){
	    printf("ClockRNN option should be TimeReso1_Dim1_TimeReso2_Dim2");
	    throw std::runtime_error("Error in RNNLayer");
	}
	m_crStep.resize(tempArgs.size(),-1);
	for (int i=0; i < tempArgs.size(); i++){
	    m_crStep[i] = boost::lexical_cast<int>(tempArgs[i]);
	}
	if (m_crStep[tempArgs.size()-1]!=size){
	    printf("ClockRNN options has unequal layer size: %d VS %d\n.Please check network.jsn", 
		   m_crStep[tempArgs.size()-1], size);
	    throw std::runtime_error("Error in RNNLayer");
	}
	return std::pow(2, tempArgs.size()/2)-1;
    }


        // Parse m_crStep
    // input:  m_crStep (from the function above) and time step
    // change: tmpSkipFlagCR (which dimension should be skipped in forward propagation)
    // return: which Hidden2Hidden matrix should be used ?
    int DimSkipFlagCRLSTM(Cpu::bool_vector &tmpSkipFlagCR, Cpu::int_vector &m_crStep, 
			  int timestep, int parallelSent)
    {
	int timeResolution;
	int featDim = tmpSkipFlagCR.size() / parallelSent;
	for (int idx = 0; idx < tmpSkipFlagCR.size(); idx++){
	    int dim = idx % featDim;
	    for (int block = 0; block < m_crStep.size()/2; block++){
		if (dim < m_crStep[block*2+1]){
		    timeResolution = m_crStep[block*2];
		    if ((timestep % timeResolution)!=0)
			tmpSkipFlagCR[dim] = true;	
		    break;
		}
	    }
	}
	int tmpNumber = 0b1;
	int matrixIdx = 0;
	for (int block = 0; block < m_crStep.size()/2; block++){
	    timeResolution = m_crStep[block*2];
	    if ((timestep % timeResolution)==0)
		matrixIdx += (tmpNumber << block);
	}
	return matrixIdx;
    }

    
    template <typename TDevice>
    LstmLayer<TDevice>::LstmLayer(const helpers::JsonValue &layerChild, 
                                  const helpers::JsonValue &weightsSection,
                                  Layer<TDevice> &precedingLayer,
				  int maxSeqLength,
				  int layerID,
                                  bool bidirectional)
        : TrainableLayer<TDevice>(
		layerChild, weightsSection, 4,
		(bidirectional ? 2 : 4) * helpers::safeJsonGetInt(layerChild, "size") + 3,
		precedingLayer, maxSeqLength, layerID)
        , m_isBidirectional      (bidirectional)
    {
        if (m_isBidirectional && this->size() % 2 != 0)
            throw std::runtime_error("Cannot create a bidirectional layer with an odd layer size");

	Configuration config     = Configuration::instance();
	
        int ls                   = this->size();
        int pls                  = this->precedingLayer().size();
	int els                  = this->size() / (m_isBidirectional ? 2 : 1);

	// ----- configuration for Clock LSTM
	m_crStepStr = ((layerChild->HasMember("clock"))?((*layerChild)["clock"].GetString()):(""));
	
	if (m_crStepStr.size()>0){
	    // Clock LSTM enabled
	    m_clockRNN   = true;

	    // Parse the option, get the number of possible updating schedule
	    m_numH2Hmat  = ReadClockLSTMOptions(m_crStepStr, m_crStep, els);
	    
	    // initialize the ClockRNN Hidden2Hidden matrices
	    m_h2hClockRNN.resize(els * els * (m_isBidirectional ? 2: 1) * m_numH2Hmat * 4, 0.0);
	    
	    // copy the configuration to device
	    m_crStepDevice = m_crStep;

	    // for Clock defined by auxiliary data

	    if (config.auxillaryDataDir().size() > 0){
		// check the number of block and the dimension of the auxiliary data
		if ((m_crStep.size()/2) > config.auxillaryDataDim() * CHAR_BIT){
		    printf("Auxillary data bit width is smaller than number of CLLSTM block");
		    throw std::runtime_error("Please check auxillary data");
		}
		// check parallel sentence
		if (this->parallelSequences() > 1){
		    throw std::runtime_error("ParallelSeq must be 1 for context-dependent CLLSTM");
		}
	    }
	    printf("[Clock enabled:%s]", m_crStepStr.c_str());
	}else{
	    m_clockRNN = false;
	    m_crStep.clear();
	    m_h2hClockRNN.clear();
	    m_crStepDevice.clear();
	}

	// ----- configuration of pointers
        _rawNiBiasWeights     = helpers::getRawPointer(this->weights()) + 4 * ls * pls + 0 * ls;
        _rawIgBiasWeights     = helpers::getRawPointer(this->weights()) + 4 * ls * pls + 1 * ls;
        _rawFgBiasWeights     = helpers::getRawPointer(this->weights()) + 4 * ls * pls + 2 * ls;
        _rawOgBiasWeights     = helpers::getRawPointer(this->weights()) + 4 * ls * pls + 3 * ls;
	
	// Manipulate the initial value of the bias for forget
	// One option to initialize the forget gate
	if (!(weightsSection.isValid() && weightsSection->HasMember(this->name().c_str())) &&
	    config.lstmForgetIni() > 0.0001){
	    Cpu::real_vector tmpBias(ls, config.lstmForgetIni());
	    thrust::copy(this->weights().begin()+ 4 * ls * pls + 2 * ls,
			 this->weights().begin()+ 4 * ls * pls + 3 * ls,
			 tmpBias.begin());
	    for (int i = 0; i<ls; i++)
		tmpBias[i] += config.lstmForgetIni();
	    
	    thrust::copy(tmpBias.begin(), tmpBias.end(),
			 this->weights().begin()+ 4 * ls * pls + 2 * ls);
	    printf("[ForgetBias %f]", config.lstmForgetIni());
	}
	

        _rawIgPeepholeWeights = (helpers::getRawPointer(this->weights()) + 4 * ls * pls + 4 * ls +
				 4 * ls * ls / (m_isBidirectional ? 2 : 1) + 0 * ls);
        _rawFgPeepholeWeights = (helpers::getRawPointer(this->weights()) + 4 * ls * pls + 4 * ls +
				 4 * ls * ls / (m_isBidirectional ? 2 : 1) + 1 * ls);
        _rawOgPeepholeWeights = (helpers::getRawPointer(this->weights()) + 4 * ls * pls + 4 * ls +
				 4 * ls * ls / (m_isBidirectional ? 2 : 1) + 2 * ls);
	
        // create the forward and backward info structs
        forward_backward_info_t* fwbwArr[] = { &m_fw, &m_bw };
        for (int fwbwArrIdx = 0; fwbwArrIdx < (m_isBidirectional ? 2 : 1); ++fwbwArrIdx) {
            forward_backward_info_t *fwbw = fwbwArr[fwbwArrIdx];

            // calculate sizes
            int pls = this->precedingLayer().size();
            int ls  = this->size();
            int els = this->size() / (m_isBidirectional ? 2 : 1);

            // cell states, niags, deltas, ...
            Cpu::real_vector tmp(this->outputs().size() / (m_isBidirectional ? 2 : 1), 0);

	    // for the CLLSTM
	    if (m_clockRNN){
		Cpu::bool_vector tmp2(this->outputs().size()/(m_isBidirectional ? 2 : 1), false);
		fwbw->skipCR      = tmp2;
	    }
	    
            if (m_isBidirectional) {
                fwbw->tmpOutputs      = tmp;
                if (this->flagTrainingMode())
		    fwbw->tmpOutputErrors = tmp;
            }else {
                fwbw->tmpOutputs     .swap(this->_outputs());
		if (this->flagTrainingMode())
		    fwbw->tmpOutputErrors.swap(this->outputErrors());
            }
	    
            fwbw->cellStates      = tmp;
            fwbw->niActs          = tmp;
            fwbw->igActs          = tmp;
            fwbw->fgActs          = tmp;
            fwbw->ogActs          = tmp;
	    if (this->flagTrainingMode()){
		fwbw->cellStateErrors = tmp;
		fwbw->niDeltas        = tmp;
		fwbw->igDeltas        = tmp;
		fwbw->fgDeltas        = tmp;
		fwbw->ogDeltas        = tmp;
	    }
            
            // weight matrices
            weight_matrices_t* wmArr [] = { &fwbw->weightMatrices, &fwbw->weightUpdateMatrices };
            real_vector*       wtsArr[] = { &this->weights(),      &this->_weightUpdates() };
            for (int wmArrIdx = 0; wmArrIdx < 2; ++wmArrIdx) {
                weight_matrices_t *wm  = wmArr [wmArrIdx];
                real_vector       *wts = wtsArr[wmArrIdx];

                int numInputWeights      = ls * pls;
                int numInternalWeights   = ls * els;
                int inputWeightsStart    = (((fwbwArrIdx == 1) ? (numInputWeights    / 2) : 0));
                int internalWeightsStart = (((fwbwArrIdx == 1) ? (numInternalWeights / 2) : 0) +
					    4 * (ls * (pls + 1)));

                wm->niInput = helpers::Matrix<TDevice>(wts, pls, els,
						       inputWeightsStart + 0 * numInputWeights);
                wm->igInput = helpers::Matrix<TDevice>(wts, pls, els,
						       inputWeightsStart + 1 * numInputWeights);
                wm->fgInput = helpers::Matrix<TDevice>(wts, pls, els,
						       inputWeightsStart + 2 * numInputWeights);
                wm->ogInput = helpers::Matrix<TDevice>(wts, pls, els,
						       inputWeightsStart + 3 * numInputWeights);

                wm->niInternal = helpers::Matrix<TDevice>(
					wts, els, els,
					internalWeightsStart + 0 * numInternalWeights);
                wm->igInternal = helpers::Matrix<TDevice>(
					wts, els, els,
					internalWeightsStart + 1 * numInternalWeights);
                wm->fgInternal = helpers::Matrix<TDevice>(
					wts, els, els,
					internalWeightsStart + 2 * numInternalWeights);
                wm->ogInternal = helpers::Matrix<TDevice>(
					wts, els, els,
					internalWeightsStart + 3 * numInternalWeights);
            }

            // matrices for each timestep
            for (int timestep = 0; timestep < this->maxSeqLength(); ++timestep) {
                int rows   = this->size() / (m_isBidirectional ? 2 : 1);
                int cols   = this->parallelSequences();
                int offset = timestep * rows * cols;
		int paraN  = this->parallelSequences();
		
                timestep_matrices_t tm;
                tm.tmpOutputs      = helpers::Matrix<TDevice>(&fwbw->tmpOutputs,
							      rows, cols, offset);
                tm.niActs          = helpers::Matrix<TDevice>(&fwbw->niActs,
							      rows, cols, offset);
                tm.igActs          = helpers::Matrix<TDevice>(&fwbw->igActs,
							      rows, cols, offset);
                tm.fgActs          = helpers::Matrix<TDevice>(&fwbw->fgActs,
							      rows, cols, offset);
                tm.ogActs          = helpers::Matrix<TDevice>(&fwbw->ogActs,
							      rows, cols, offset);
		if (this->flagTrainingMode()){
		    tm.tmpOutputErrors = helpers::Matrix<TDevice>(&fwbw->tmpOutputErrors,
								  rows, cols, offset);
		    tm.niDeltas        = helpers::Matrix<TDevice>(&fwbw->niDeltas,
								  rows, cols, offset);
		    tm.igDeltas        = helpers::Matrix<TDevice>(&fwbw->igDeltas,
								  rows, cols, offset);
		    tm.fgDeltas        = helpers::Matrix<TDevice>(&fwbw->fgDeltas,
								  rows, cols, offset);
		    tm.ogDeltas        = helpers::Matrix<TDevice>(&fwbw->ogDeltas,
								  rows, cols, offset);
		}
		
		// clock configuration
		if (m_clockRNN){
		    // 
		    // tmpFlagCR:    a skip flag for each dimenison in a parallel block
		    Cpu::bool_vector tmpFlagCR(rows * paraN, false);
		    
		    // h2hMatrixIdx: which hidden2hidden matrix should be used
		    //               h2hMatrixIdx \in [0  2^band_number-1]
		    int h2hMatrixIdx = DimSkipFlagCRLSTM(tmpFlagCR, m_crStep, timestep, paraN)-1;
		    
		    if (h2hMatrixIdx<0){
			printf("Zero input at time %d", timestep);
			throw std::runtime_error("Error in timeresolution configuration");
		    }
		    
		    tm.skipCRPos = timestep * rows * paraN;

		    if (DEBUG_CLOCKLSTM){
			printf("%d:\n", timestep);
			for(int i = 0; i < tmpFlagCR.size(); i++){
			    printf("%d ", tmpFlagCR[i]);
			}
			printf("\n");
		    }
		    
		    thrust::copy(tmpFlagCR.begin(), tmpFlagCR.end(), 
				 fwbw->skipCR.begin() + tm.skipCRPos);
		    
		    tm.h2hIdx    = h2hMatrixIdx;

		    // matrices for forward and backward direction are separated
		    h2hMatrixIdx = (h2hMatrixIdx * els * els +
				    ((fwbwArrIdx == 1) ? (m_h2hClockRNN.size()/2) : 0));

		    tm.niH2HWrap = helpers::Matrix<TDevice>(&m_h2hClockRNN, els, els,
							    h2hMatrixIdx);
		    h2hMatrixIdx = h2hMatrixIdx + els * els * m_numH2Hmat;
		    tm.igH2HWrap = helpers::Matrix<TDevice>(&m_h2hClockRNN, els, els,
							    h2hMatrixIdx);
		    h2hMatrixIdx = h2hMatrixIdx + els * els * m_numH2Hmat;
		    tm.fgH2HWrap = helpers::Matrix<TDevice>(&m_h2hClockRNN, els, els,
							    h2hMatrixIdx);
		    h2hMatrixIdx = h2hMatrixIdx + els * els * m_numH2Hmat;
		    tm.ogH2HWrap = helpers::Matrix<TDevice>(&m_h2hClockRNN, els, els,
							    h2hMatrixIdx);
		}
		
                fwbw->timestepMatrices.push_back(tm);
            }
        }

        if (!m_isBidirectional) {
            m_fw.tmpOutputs     .swap(this->_outputs());
	    if (this->flagTrainingMode())
		m_fw.tmpOutputErrors.swap(this->outputErrors());
        }
    }

    template <typename TDevice>
    LstmLayer<TDevice>::~LstmLayer()
    {
    }

    template <typename TDevice>
    const std::string& LstmLayer<TDevice>::type() const
    {
        static const std::string su("lstm");
        static const std::string sb("blstm");
        return (m_isBidirectional ? sb : su);
    }

    template <typename TDevice>
    bool LstmLayer<TDevice>::isBidirectional() const
    {
        return m_isBidirectional;
    }

    template <typename TDevice>
    const typename TDevice::real_vector& LstmLayer<TDevice>::cellStates() const
    {
        if (m_isBidirectional)
            throw std::runtime_error("Not implemented");
        else
            return m_fw.cellStates;
    }

    template <typename TDevice>
    const typename TDevice::real_vector& LstmLayer<TDevice>::cellStateErrors() const
    {
        if (m_isBidirectional)
            throw std::runtime_error("Not implemented");
        else
            return m_fw.cellStateErrors;
    }

    template <typename TDevice>
    const typename TDevice::real_vector& LstmLayer<TDevice>::netInputActs() const
    {
        if (m_isBidirectional)
            throw std::runtime_error("Not implemented");
        else
            return m_fw.niActs;
    }

    template <typename TDevice>
    const typename TDevice::real_vector& LstmLayer<TDevice>::netInputDeltas() const
    {
        if (m_isBidirectional)
            throw std::runtime_error("Not implemented");
        else
            return m_fw.niDeltas;
    }

    template <typename TDevice>
    const typename TDevice::real_vector& LstmLayer<TDevice>::inputGateActs() const
    {
        if (m_isBidirectional)
            throw std::runtime_error("Not implemented");
        else
            return m_fw.igActs;
    }

    template <typename TDevice>
    const typename TDevice::real_vector& LstmLayer<TDevice>::inputGateDeltas() const
    {
        if (m_isBidirectional)
            throw std::runtime_error("Not implemented");
        else
            return m_fw.igDeltas;
    }

    template <typename TDevice>
    const typename TDevice::real_vector& LstmLayer<TDevice>::forgetGateActs() const
    {
        if (m_isBidirectional)
            throw std::runtime_error("Not implemented");
        else
            return m_fw.fgActs;
    }

    template <typename TDevice>
    const typename TDevice::real_vector& LstmLayer<TDevice>::forgetGateDeltas() const
    {
        if (m_isBidirectional)
            throw std::runtime_error("Not implemented");
        else
            return m_fw.fgDeltas;
    }

    template <typename TDevice>
    const typename TDevice::real_vector& LstmLayer<TDevice>::outputGateActs() const
    {
        if (m_isBidirectional)
            throw std::runtime_error("Not implemented");
        else
            return m_fw.ogActs;
    }

    template <typename TDevice>
    const typename TDevice::real_vector& LstmLayer<TDevice>::outputGateDeltas() const
    {
        if (m_isBidirectional)
            throw std::runtime_error("Not implemented");
        else
            return m_fw.ogDeltas;
    }

    template <typename TDevice>
    void LstmLayer<TDevice>::loadSequences(const data_sets::DataSetFraction &fraction,
					   const int nnState)
    {
        TrainableLayer<TDevice>::loadSequences(fraction, nnState);

	if (this->precedingLayer().getSaveMemoryFlag()){
	    // This step is fact useless. The matrix will be defined in prepareStepGeneration
	    m_precLayerOutputsMatrix = helpers::Matrix<TDevice>(
		&this->precedingLayer().outputs(), this->precedingLayer().size(), 
		this->parallelSequences());
	}else{
	    m_precLayerOutputsMatrix = helpers::Matrix<TDevice>(
		&this->precedingLayer().outputs(), this->precedingLayer().size(), 
		this->curMaxSeqLength() * this->parallelSequences());
	}

        // ---- prepare the matrix for LSTM
	// update the niag matrices
        forward_backward_info_t* fwbwArr[] = { &m_fw, &m_bw };
        for (int fwbwArrIdx = 0; fwbwArrIdx < (m_isBidirectional ? 2 : 1); ++fwbwArrIdx) {
            forward_backward_info_t *fwbw = fwbwArr[fwbwArrIdx];

            int rows = this->size() / (m_isBidirectional ? 2 : 1);
            int cols = this->curMaxSeqLength() * this->parallelSequences();

            fwbw->niActsMatrix = helpers::Matrix<TDevice>(&fwbw->niActs, rows, cols);
            fwbw->igActsMatrix = helpers::Matrix<TDevice>(&fwbw->igActs, rows, cols);
            fwbw->fgActsMatrix = helpers::Matrix<TDevice>(&fwbw->fgActs, rows, cols);
            fwbw->ogActsMatrix = helpers::Matrix<TDevice>(&fwbw->ogActs, rows, cols);

	    if (this->flagTrainingMode()){
		fwbw->niDeltasMatrix = helpers::Matrix<TDevice>(&fwbw->niDeltas, rows, cols);
		fwbw->igDeltasMatrix = helpers::Matrix<TDevice>(&fwbw->igDeltas, rows, cols);
		fwbw->fgDeltasMatrix = helpers::Matrix<TDevice>(&fwbw->fgDeltas, rows, cols);
		fwbw->ogDeltasMatrix = helpers::Matrix<TDevice>(&fwbw->ogDeltas, rows, cols);
		
		// set ogDeltas to zero
		thrust::fill(fwbw->ogDeltas.begin(), fwbw->ogDeltas.end(), 0.0);
	    }
        }

	
	// ---- prepare the matrix for CLLSTM
	if (m_clockRNN){
	    
	    int rows    = this->size() / (m_isBidirectional ? 2 : 1);
            int cols    = this->curMaxSeqLength() * this->parallelSequences();
	    int ls      = this->size();
	    int pls     = this->precedingLayer().size();
	    int h2hsize = rows * rows;
	    
	    // Read in the context-dependent time clock
	    if (fraction.auxDataDim()>0){
		Cpu::pattype_vector clockTime = fraction.auxPattypeData();
		if (clockTime.size() != this->curMaxSeqLength())
		    throw std::runtime_error("Error unequal length of clockTime size");
		if (this->parallelSequences()>1){
		    printf("Please use parallel_sequences = 1\n");
		    throw std::runtime_error("Not implemented: clockRNN for parallel training");
		}
		int h2hMatrixIdx = 0;
		for (int t = 0; t < this->curMaxSeqLength(); t++){		    
		    // assign h2hIdx
		    h2hMatrixIdx = (int)clockTime[t] - 1;
		    
		    m_fw.timestepMatrices[t].h2hIdx = h2hMatrixIdx;
		    if(m_isBidirectional) {m_bw.timestepMatrices[t].h2hIdx = h2hMatrixIdx;}
		    
		    // assign h2hWarp
		    int offset   = h2hMatrixIdx * rows * rows;
		    m_fw.timestepMatrices[t].niH2HWrap = 
			helpers::Matrix<TDevice>(&m_h2hClockRNN, rows, rows, offset);
		    
		    offset       = offset + rows * rows * m_numH2Hmat;
		    m_fw.timestepMatrices[t].igH2HWrap =
			helpers::Matrix<TDevice>(&m_h2hClockRNN, rows, rows, offset);
		    
		    offset       = offset + rows * rows * m_numH2Hmat;
		    m_fw.timestepMatrices[t].fgH2HWrap =
			helpers::Matrix<TDevice>(&m_h2hClockRNN, rows, rows, offset);
		    
		    offset       = offset + rows * rows * m_numH2Hmat;
		    m_fw.timestepMatrices[t].ogH2HWrap =
			helpers::Matrix<TDevice>(&m_h2hClockRNN, rows, rows, offset);

		    // assign the skipCRNN vector and skipCRPos
		    m_fw.timestepMatrices[t].skipCRPos = t * rows;
		    
		    if(m_isBidirectional) {
			m_bw.timestepMatrices[t].skipCRPos = m_fw.timestepMatrices[t].skipCRPos;
			offset       = h2hMatrixIdx * rows * rows + (m_h2hClockRNN.size()/2);
			m_bw.timestepMatrices[t].niH2HWrap = 
			    helpers::Matrix<TDevice>(&m_h2hClockRNN, rows, rows, offset);
		    
			offset       = offset + rows * rows * m_numH2Hmat;
			m_bw.timestepMatrices[t].igH2HWrap =
			    helpers::Matrix<TDevice>(&m_h2hClockRNN, rows, rows, offset);
			
			offset       = offset + rows * rows * m_numH2Hmat;
			m_bw.timestepMatrices[t].fgH2HWrap =
			    helpers::Matrix<TDevice>(&m_h2hClockRNN, rows, rows, offset);
			
			offset       = offset + rows * rows * m_numH2Hmat;
			m_bw.timestepMatrices[t].ogH2HWrap =
			    helpers::Matrix<TDevice>(&m_h2hClockRNN, rows, rows, offset);
		    }
		}

		// Create the skipCR for every time step based on the boundary information
		//   and band information
		pattype_vector clockTimeDevice = clockTime;
		internal::CreateSkipVecForContextDepedentCLLSTM fn;
		fn.bandNum    = m_crStepDevice.size()/2;
		fn.featDim    = rows;
		fn.bandConfig = helpers::getRawPointer(m_crStepDevice);
		fn.skipVec    = helpers::getRawPointer(m_fw.skipCR);
		fn.skipVec2   = (m_isBidirectional ? (helpers::getRawPointer(m_bw.skipCR)):NULL);
		fn.clockTime  = helpers::getRawPointer(clockTimeDevice);
		thrust::for_each(
		   thrust::make_zip_iterator(
			thrust::make_tuple(thrust::counting_iterator<int>(0),
					   m_fw.skipCR.begin())),
		   thrust::make_zip_iterator(
			thrust::make_tuple(thrust::counting_iterator<int>(0) + m_fw.skipCR.size(),
					   m_fw.skipCR.end())),
		   fn);
	    }



	    // duplicate the matrix for each time step
	    int internalWeightsStart = 4 * (ls * (pls + 1));
	    int srcOff = 0;
	    int tarOff = 0;
	    internal::SetH2HMatrixLSTM fn;
	    for (int i = 0; i < 4; i++){
		if (m_isBidirectional){
		    fn.bandConfig = helpers::getRawPointer(m_crStepDevice);
		    fn.bandNum    = m_crStepDevice.size()/2;
		    fn.featDim    = rows;
		    fn.matrixSize = h2hsize;

		    srcOff        = internalWeightsStart + i * h2hsize * 2;
		    tarOff        = i * h2hsize * m_numH2Hmat;
		    fn.sourceW    = helpers::getRawPointer(this->weights()) + srcOff;
		    fn.targetW    = helpers::getRawPointer(m_h2hClockRNN)   + tarOff;
		    thrust::for_each(thrust::counting_iterator<int>(0),
				     thrust::counting_iterator<int>(0)  + h2hsize * m_numH2Hmat,
				     fn);
		    
		    srcOff        = internalWeightsStart + i * h2hsize * 2 + h2hsize;
		    tarOff        = i * h2hsize * m_numH2Hmat + 4 * h2hsize * m_numH2Hmat;
		    fn.sourceW    = helpers::getRawPointer(this->weights()) + srcOff;
		    fn.targetW    = helpers::getRawPointer(m_h2hClockRNN)   + tarOff;
		    thrust::for_each(thrust::counting_iterator<int>(0),
				     thrust::counting_iterator<int>(0)  + h2hsize * m_numH2Hmat,
				     fn);
		
		}else{
		    fn.bandConfig = helpers::getRawPointer(m_crStepDevice);
		    fn.bandNum    = m_crStepDevice.size()/2;
		    fn.featDim    = rows;
		    fn.matrixSize = h2hsize;

		    srcOff        = internalWeightsStart + i * h2hsize;
		    tarOff        = i * h2hsize * m_numH2Hmat;
		    fn.sourceW    = helpers::getRawPointer(this->weights()) + srcOff;
		    fn.targetW    = helpers::getRawPointer(m_h2hClockRNN)   + tarOff;
		    thrust::for_each(thrust::counting_iterator<int>(0),
				     thrust::counting_iterator<int>(0)  + h2hsize * m_numH2Hmat,
				     fn);
		}
	    }
	
	    // For debug
	    // Show all the H2H matrices
	    if (DEBUG_CLOCKLSTM){
		
		Cpu::bool_vector tmpFlagCR = m_fw.skipCR;
		for (int timestep=0; timestep < this->curMaxSeqLength(); timestep++){
		    printf("%d:\n", timestep);
		    for (int i = 0; i < rows; i++){
			printf("%d ", tmpFlagCR[i+timestep*rows]);
		    }
		    printf("\n");
		}

		Cpu::real_vector h2hMatrix_debug = m_h2hClockRNN;
		int biasPos_debug = 0;
		for (int i = 0; i < m_numH2Hmat; i++){
		    printf("Forward: Matrix %d\n", i);
		    for (int x_row = 0; x_row < rows; x_row++){
			for (int y_col = 0; y_col < rows; y_col++){
			    printf("%f ", h2hMatrix_debug[biasPos_debug + x_row + y_col*rows]);
			}
			printf("\n");
		    }
		    biasPos_debug  += h2hsize;
		}
		biasPos_debug  = biasPos_debug * 4;
		if (m_isBidirectional){
		    for (int i = 0; i < m_numH2Hmat; i++){
			printf("BackWard: Matrix %d\n", i);
			for (int x_row = 0; x_row < rows; x_row++){
			    for (int y_col = 0; y_col < rows; y_col++){
				printf("%f ", h2hMatrix_debug[biasPos_debug+x_row+y_col*rows]);
			    }
			    printf("\n");
			}
			biasPos_debug  += h2hsize;
		    }
		}
		// Show matrix index
		printf("Time-MatrixIdx\n");
		for (int t = 0; t < this->curMaxSeqLength(); t++){
		    printf("%5d-%3d ", t, m_fw.timestepMatrices[t].h2hIdx);
		    if (t % 10 == 9) printf("\n");
		}
		if (fraction.auxDataDim()>0){
		    Cpu::bool_vector tempskipCR    = m_fw.skipCR;
		    for (int t = 0; t < this->curMaxSeqLength(); t++){
			printf("%d:\n", t);
			for (int d = 0; d < rows; d++){
			    printf("%d ", tempskipCR[t*rows + d]);
			}
			printf("\n");
		    }
		}
	    }
	} // if (m_clockRNN)
	
	// finish
    }

    template <typename TDevice>
    void LstmLayer<TDevice>::prepareStepGeneration(const int timeStep)
    {
	m_precLayerOutputsMatrix = helpers::Matrix<TDevice>(
		&this->precedingLayer().outputs(),
		this->precedingLayer().size(), this->parallelSequences(),
		(timeStep * this->parallelSequences() * this->precedingLayer().size() - 
		 this->precedingLayer().outputBufPtrBias(timeStep*this->parallelSequences(), 0)));
    }


    template <typename TDevice>
    void LstmLayer<TDevice>::computeForwardPass(const int nnState)
    {	
        // for unidirectional LSTM, we can write the outputs directly in the layer output vector
        if (!m_isBidirectional) {
            m_fw.tmpOutputs.swap(this->_outputs());
        }

        // sum up the activations from the preceding layer
        {{
            // forward states
            m_fw.niActsMatrix.assignProduct(m_fw.weightMatrices.niInput,
					    true, m_precLayerOutputsMatrix, false);
            m_fw.igActsMatrix.assignProduct(m_fw.weightMatrices.igInput,
					    true, m_precLayerOutputsMatrix, false);
            m_fw.fgActsMatrix.assignProduct(m_fw.weightMatrices.fgInput,
					    true, m_precLayerOutputsMatrix, false);
            m_fw.ogActsMatrix.assignProduct(m_fw.weightMatrices.ogInput,
					    true, m_precLayerOutputsMatrix, false);

            // backward states
            if (m_isBidirectional) {
                m_bw.niActsMatrix.assignProduct(m_bw.weightMatrices.niInput,
						true, m_precLayerOutputsMatrix, false);
                m_bw.igActsMatrix.assignProduct(m_bw.weightMatrices.igInput,
						true, m_precLayerOutputsMatrix, false);
                m_bw.fgActsMatrix.assignProduct(m_bw.weightMatrices.fgInput,
						true, m_precLayerOutputsMatrix, false);
                m_bw.ogActsMatrix.assignProduct(m_bw.weightMatrices.ogInput,
						true, m_precLayerOutputsMatrix, false);
            }
        }}

        // compute the block outputs
        {{
            int els = this->size() / (m_isBidirectional ? 2 : 1);
            int n   = this->parallelSequences() * els;

            // forward states
            internal::ComputeBlockOutputFn fn;
            fn.effLayerSize       = els;
            fn.prevOutputDistance = -n;
            fn.bias               = this->bias();
            fn.patTypes           = helpers::getRawPointer(this->patTypes());
            fn.niBiasWeights      = _rawNiBiasWeights;
            fn.igBiasWeights      = _rawIgBiasWeights;
            fn.fgBiasWeights      = _rawFgBiasWeights;
            fn.ogBiasWeights      = _rawOgBiasWeights;
            fn.igPeepWeights      = _rawIgPeepholeWeights;
            fn.fgPeepWeights      = _rawFgPeepholeWeights;
            fn.ogPeepWeights      = _rawOgPeepholeWeights;
            fn.cellStates         = helpers::getRawPointer(m_fw.cellStates);
            fn.niActs             = helpers::getRawPointer(m_fw.niActs);
            fn.igActs             = helpers::getRawPointer(m_fw.igActs);
            fn.fgActs             = helpers::getRawPointer(m_fw.fgActs);
            fn.ogActs             = helpers::getRawPointer(m_fw.ogActs);

            for (int timestep = 0; timestep < this->curMaxSeqLength(); ++timestep) {
                // collect outputs from previous timestep
                if (timestep != 0) {
		    if (m_clockRNN){
			m_fw.timestepMatrices[timestep].niActs.addProduct(
			  m_fw.timestepMatrices[timestep].niH2HWrap, true,
			  m_fw.timestepMatrices[timestep-1].tmpOutputs, false);
			m_fw.timestepMatrices[timestep].igActs.addProduct(
			  m_fw.timestepMatrices[timestep].igH2HWrap, true,
			  m_fw.timestepMatrices[timestep-1].tmpOutputs, false);
			m_fw.timestepMatrices[timestep].fgActs.addProduct(
			  m_fw.timestepMatrices[timestep].fgH2HWrap, true,
			  m_fw.timestepMatrices[timestep-1].tmpOutputs, false);
			m_fw.timestepMatrices[timestep].ogActs.addProduct(
			  m_fw.timestepMatrices[timestep].ogH2HWrap, true,
			  m_fw.timestepMatrices[timestep-1].tmpOutputs, false);
		    }else{
			m_fw.timestepMatrices[timestep].niActs.addProduct(
			  m_fw.weightMatrices.niInternal, true,
			  m_fw.timestepMatrices[timestep-1].tmpOutputs, false);
			m_fw.timestepMatrices[timestep].igActs.addProduct(
			  m_fw.weightMatrices.igInternal, true,
			  m_fw.timestepMatrices[timestep-1].tmpOutputs, false);
			m_fw.timestepMatrices[timestep].fgActs.addProduct(
			  m_fw.weightMatrices.fgInternal, true,
			  m_fw.timestepMatrices[timestep-1].tmpOutputs, false);
			m_fw.timestepMatrices[timestep].ogActs.addProduct(
			  m_fw.weightMatrices.ogInternal, true,
			  m_fw.timestepMatrices[timestep-1].tmpOutputs, false);
		    }
                }

		// for ClockRNN
		if (m_clockRNN)
		    fn.skipCRNN  = (helpers::getRawPointer(m_fw.skipCR) + 
				    m_fw.timestepMatrices[timestep].skipCRPos);
		else
		    fn.skipCRNN  = NULL;

                // compute outputs
                thrust::transform(
                    thrust::counting_iterator<int>(n*timestep),
                    thrust::counting_iterator<int>(n*timestep) + n,
                    thrust::make_zip_iterator(
		      thrust::make_tuple(
			thrust::constant_iterator<bool>(!timestep),
			thrust::constant_iterator<bool>(timestep >= this->curMinSeqLength()))),
                    m_fw.tmpOutputs.begin() + n*timestep,
                    fn);
            }

            // backward states
            if (m_isBidirectional) {
                fn.prevOutputDistance = +n;
                fn.niBiasWeights     += els;
                fn.igBiasWeights     += els;
                fn.fgBiasWeights     += els;
                fn.ogBiasWeights     += els;
                fn.igPeepWeights     += els;
                fn.fgPeepWeights     += els;
                fn.ogPeepWeights     += els;
                fn.cellStates         = helpers::getRawPointer(m_bw.cellStates);
                fn.niActs             = helpers::getRawPointer(m_bw.niActs);
                fn.igActs             = helpers::getRawPointer(m_bw.igActs);
                fn.fgActs             = helpers::getRawPointer(m_bw.fgActs);
                fn.ogActs             = helpers::getRawPointer(m_bw.ogActs);

                for (int timestep = this->curMaxSeqLength()-1; timestep >= 0; --timestep) {
                    // collect outputs from previous timestep
                    if (timestep != this->curMaxSeqLength()-1) {
			if (m_clockRNN){
			    m_bw.timestepMatrices[timestep].niActs.addProduct(
				m_bw.timestepMatrices[timestep].niH2HWrap,    true,
				m_bw.timestepMatrices[timestep+1].tmpOutputs, false);
			    m_bw.timestepMatrices[timestep].igActs.addProduct(
				m_bw.timestepMatrices[timestep].igH2HWrap,  true,
				m_bw.timestepMatrices[timestep+1].tmpOutputs, false);
			    m_bw.timestepMatrices[timestep].fgActs.addProduct(
				m_bw.timestepMatrices[timestep].fgH2HWrap,  true,
				m_bw.timestepMatrices[timestep+1].tmpOutputs, false);
			    m_bw.timestepMatrices[timestep].ogActs.addProduct(
				m_bw.timestepMatrices[timestep].ogH2HWrap,  true,
				m_bw.timestepMatrices[timestep+1].tmpOutputs, false);
			}else{
			    m_bw.timestepMatrices[timestep].niActs.addProduct(
				m_bw.weightMatrices.niInternal,               true,
				m_bw.timestepMatrices[timestep+1].tmpOutputs, false);
			    m_bw.timestepMatrices[timestep].igActs.addProduct(
				m_bw.weightMatrices.igInternal,               true,
				m_bw.timestepMatrices[timestep+1].tmpOutputs, false);
			    m_bw.timestepMatrices[timestep].fgActs.addProduct(
				m_bw.weightMatrices.fgInternal,               true,
				m_bw.timestepMatrices[timestep+1].tmpOutputs, false);
			    m_bw.timestepMatrices[timestep].ogActs.addProduct(
				m_bw.weightMatrices.ogInternal,               true,
				m_bw.timestepMatrices[timestep+1].tmpOutputs, false);			    
			}
                    }

		    // for ClockRNN
		    if (m_clockRNN)
			fn.skipCRNN  = (helpers::getRawPointer(m_bw.skipCR) + 
					m_bw.timestepMatrices[timestep].skipCRPos);
		    else
			fn.skipCRNN  = NULL;

                    // compute outputs
                    thrust::transform(
                        thrust::counting_iterator<int>(n*timestep),
                        thrust::counting_iterator<int>(n*timestep) + n,
                        thrust::make_zip_iterator(
			  thrust::make_tuple(
			   thrust::constant_iterator<bool>(timestep == this->curMaxSeqLength()-1), 
			   thrust::constant_iterator<bool>(timestep >= this->curMinSeqLength()))),
                        m_bw.tmpOutputs.begin() + n*timestep,
                        fn);
                }
            }
        }}

        // resort outputs
        if (m_isBidirectional) {
            internal::ResortOutputsFn fn;
            fn.layerSize    = this->size();
            fn.effLayerSize = this->size() / 2;
            fn.fwOutputs    = helpers::getRawPointer(m_fw.tmpOutputs);
            fn.bwOutputs    = helpers::getRawPointer(m_bw.tmpOutputs);

            thrust::transform(
                thrust::counting_iterator<int>(0),
                (thrust::counting_iterator<int>(0) +
		 this->curMaxSeqLength() * this->parallelSequences() * this->size()),
                this->_outputs().begin(),
                fn
                );
        }else {
            this->_outputs().swap(m_fw.tmpOutputs);
        }
    }

    template <typename TDevice>
    void LstmLayer<TDevice>::computeForwardPass(const int timeStep, const int nnState)
    {

	if (m_isBidirectional)
	    return;
	
	// for unidirectional LSTM, we can write the outputs directly in the layer output vector
        if (!m_isBidirectional) {
            m_fw.tmpOutputs.swap(this->_outputs());
        }

        // sum up the activations from the preceding layer for one time step
        {{
	    // forward states
	    // m_preLayerOutputsMatrix is assigned to one frame by prePareStepGeneration(timeStep)
	    m_fw.timestepMatrices[timeStep].niActs.assignProduct(
			m_fw.weightMatrices.niInput, true, m_precLayerOutputsMatrix, false);
	    m_fw.timestepMatrices[timeStep].igActs.assignProduct(
			m_fw.weightMatrices.igInput, true, m_precLayerOutputsMatrix, false);
	    m_fw.timestepMatrices[timeStep].fgActs.assignProduct(
			m_fw.weightMatrices.fgInput, true, m_precLayerOutputsMatrix, false);
	    m_fw.timestepMatrices[timeStep].ogActs.assignProduct(
			m_fw.weightMatrices.ogInput, true, m_precLayerOutputsMatrix, false);
        }}

        // compute the block outputs
        {{
            int els = this->size() / (m_isBidirectional ? 2 : 1);
            int n   = this->parallelSequences() * els;

            // forward states
            internal::ComputeBlockOutputFn fn;
            fn.effLayerSize       = els;
            fn.prevOutputDistance = -n;
            fn.bias               = this->bias();
            fn.patTypes           = helpers::getRawPointer(this->patTypes());
            fn.niBiasWeights      = _rawNiBiasWeights;
            fn.igBiasWeights      = _rawIgBiasWeights;
            fn.fgBiasWeights      = _rawFgBiasWeights;
            fn.ogBiasWeights      = _rawOgBiasWeights;
            fn.igPeepWeights      = _rawIgPeepholeWeights;
            fn.fgPeepWeights      = _rawFgPeepholeWeights;
            fn.ogPeepWeights      = _rawOgPeepholeWeights;
            fn.cellStates         = helpers::getRawPointer(m_fw.cellStates);
            fn.niActs             = helpers::getRawPointer(m_fw.niActs);
            fn.igActs             = helpers::getRawPointer(m_fw.igActs);
            fn.fgActs             = helpers::getRawPointer(m_fw.fgActs);
            fn.ogActs             = helpers::getRawPointer(m_fw.ogActs);

            if (timeStep != 0) {
		if (m_clockRNN){
		    m_fw.timestepMatrices[timeStep].niActs.addProduct(
			m_fw.timestepMatrices[timeStep].niH2HWrap, true, 
			m_fw.timestepMatrices[timeStep-1].tmpOutputs, false);
		    m_fw.timestepMatrices[timeStep].igActs.addProduct(
			m_fw.timestepMatrices[timeStep].igH2HWrap, true, 
			m_fw.timestepMatrices[timeStep-1].tmpOutputs, false);
		    m_fw.timestepMatrices[timeStep].fgActs.addProduct(
			m_fw.timestepMatrices[timeStep].fgH2HWrap, true, 
			m_fw.timestepMatrices[timeStep-1].tmpOutputs, false);
		    m_fw.timestepMatrices[timeStep].ogActs.addProduct(
			m_fw.timestepMatrices[timeStep].ogH2HWrap, true, 
			m_fw.timestepMatrices[timeStep-1].tmpOutputs, false);		    
		}else{
		    m_fw.timestepMatrices[timeStep].niActs.addProduct(
			m_fw.weightMatrices.niInternal, true, 
			m_fw.timestepMatrices[timeStep-1].tmpOutputs, false);
		    m_fw.timestepMatrices[timeStep].igActs.addProduct(
			m_fw.weightMatrices.igInternal, true, 
			m_fw.timestepMatrices[timeStep-1].tmpOutputs, false);
		    m_fw.timestepMatrices[timeStep].fgActs.addProduct(
			m_fw.weightMatrices.fgInternal, true, 
			m_fw.timestepMatrices[timeStep-1].tmpOutputs, false);
		    m_fw.timestepMatrices[timeStep].ogActs.addProduct(
			m_fw.weightMatrices.ogInternal, true, 
			m_fw.timestepMatrices[timeStep-1].tmpOutputs, false);		    
		}
	    }
	    // for ClockRNN
	    if (m_clockRNN)
		fn.skipCRNN  = (helpers::getRawPointer(m_fw.skipCR) + 
				m_fw.timestepMatrices[timeStep].skipCRPos);
	    else
		fn.skipCRNN  = NULL;

	    // compute outputs
	    thrust::transform(
		thrust::counting_iterator<int>(n * timeStep),
		thrust::counting_iterator<int>(n * timeStep) + n,
		thrust::make_zip_iterator(
		   thrust::make_tuple(
		      thrust::constant_iterator<bool>(!timeStep), 
		      thrust::constant_iterator<bool>(timeStep >= this->curMinSeqLength()))),
		m_fw.tmpOutputs.begin() + n*timeStep,
		fn);
            
	    // bi-directional network is not allowed
        }}

        // resort outputs
        if (m_isBidirectional) {
	    
        }else {
            this->_outputs().swap(m_fw.tmpOutputs);
        }

    }


    template <typename TDevice>
    void LstmLayer<TDevice>::computeBackwardPass(const int nnState)
    {
        // for unidirectional LSTM,
	// we can write the output errors directly in the layer output errors vector
        if (m_isBidirectional) {
            internal::ResortOutputErrorsFn fn;
            fn.layerSize      = this->size();
            fn.effLayerSize   = this->size() / 2;
            fn.fwOutputErrors = helpers::getRawPointer(m_fw.tmpOutputErrors);
            fn.bwOutputErrors = helpers::getRawPointer(m_bw.tmpOutputErrors);

            int n = this->curMaxSeqLength() * this->parallelSequences() * this->size();

            thrust::for_each(
                thrust::make_zip_iterator(thrust::make_tuple(this->outputErrors().begin(),
							     thrust::counting_iterator<int>(0))),
                thrust::make_zip_iterator(thrust::make_tuple(this->outputErrors().begin()+n,
							     thrust::counting_iterator<int>(0)+n)),
                fn);
        }
        else {
            m_fw.tmpOutputs     .swap(this->outputs());
            m_fw.tmpOutputErrors.swap(this->outputErrors());
        }

        // calculate the block errors
        {{
            int els = this->size() / (m_isBidirectional ? 2 : 1);
            int n   = this->parallelSequences() * els;

            // forward states
            internal::ComputeBlockErrorsFn fn;
            fn.effLayerSize       = els;
            fn.prevOutputDistance = -n;
            fn.patTypes           = helpers::getRawPointer(this->patTypes());
            fn.igPeepWeights      = _rawIgPeepholeWeights;
            fn.fgPeepWeights      = _rawFgPeepholeWeights;
            fn.ogPeepWeights      = _rawOgPeepholeWeights;
            fn.cellStates         = helpers::getRawPointer(m_fw.cellStates);
            fn.niActs             = helpers::getRawPointer(m_fw.niActs);
            fn.igActs             = helpers::getRawPointer(m_fw.igActs);
            fn.fgActs             = helpers::getRawPointer(m_fw.fgActs);
            fn.ogActs             = helpers::getRawPointer(m_fw.ogActs);
            fn.cellStateErrors    = helpers::getRawPointer(m_fw.cellStateErrors);
            fn.niDeltas           = helpers::getRawPointer(m_fw.niDeltas);
            fn.igDeltas           = helpers::getRawPointer(m_fw.igDeltas);
            fn.fgDeltas           = helpers::getRawPointer(m_fw.fgDeltas);
            fn.ogDeltas           = helpers::getRawPointer(m_fw.ogDeltas);

            for (int timestep = this->curMaxSeqLength()-1; timestep >= 0; --timestep) {
                // collect errors from previous timestep
                if (timestep != this->curMaxSeqLength()-1) {

		    if (m_clockRNN){
			m_fw.timestepMatrices[timestep].tmpOutputErrors.addProduct(
				m_fw.timestepMatrices[timestep+1].niH2HWrap, false,
				m_fw.timestepMatrices[timestep+1].niDeltas, false);
			m_fw.timestepMatrices[timestep].tmpOutputErrors.addProduct(
				m_fw.timestepMatrices[timestep+1].igH2HWrap, false,
				m_fw.timestepMatrices[timestep+1].igDeltas, false);
			m_fw.timestepMatrices[timestep].tmpOutputErrors.addProduct(
				m_fw.timestepMatrices[timestep+1].fgH2HWrap, false,
				m_fw.timestepMatrices[timestep+1].fgDeltas, false);
			m_fw.timestepMatrices[timestep].tmpOutputErrors.addProduct(
				m_fw.timestepMatrices[timestep+1].ogH2HWrap, false,
				m_fw.timestepMatrices[timestep+1].ogDeltas, false);
		    }else{
			m_fw.timestepMatrices[timestep].tmpOutputErrors.addProduct(
				m_fw.weightMatrices.niInternal, false,
				m_fw.timestepMatrices[timestep+1].niDeltas, false);
			m_fw.timestepMatrices[timestep].tmpOutputErrors.addProduct(
				m_fw.weightMatrices.igInternal, false,
				m_fw.timestepMatrices[timestep+1].igDeltas, false);
			m_fw.timestepMatrices[timestep].tmpOutputErrors.addProduct(
				m_fw.weightMatrices.fgInternal, false,
				m_fw.timestepMatrices[timestep+1].fgDeltas, false);
			m_fw.timestepMatrices[timestep].tmpOutputErrors.addProduct(
				m_fw.weightMatrices.ogInternal, false,
				m_fw.timestepMatrices[timestep+1].ogDeltas, false);
		    }
                }

		if (m_clockRNN){
		    fn.skipCRNN  = helpers::getRawPointer(m_fw.skipCR) + 
			m_fw.timestepMatrices[timestep].skipCRPos;
		}else{
		    fn.skipCRNN = NULL;
		}

                // compute errors
                thrust::for_each(
                    thrust::make_zip_iterator(
		      thrust::make_tuple(
			  m_fw.tmpOutputErrors.begin() + n*timestep,
			  thrust::counting_iterator<int>(n*timestep),
			  thrust::constant_iterator<bool>(timestep == this->curMaxSeqLength()-1),
			  thrust::constant_iterator<bool>(!timestep),
			  thrust::constant_iterator<bool>(timestep >= this->curMinSeqLength()))),
                    thrust::make_zip_iterator(
		      thrust::make_tuple(
			  m_fw.tmpOutputErrors.begin() + n*timestep + n,
			  thrust::counting_iterator<int>(n*timestep)+ n,
			  thrust::constant_iterator<bool>(timestep == this->curMaxSeqLength()-1)+n,
			  thrust::constant_iterator<bool>(!timestep)+ n,
			  thrust::constant_iterator<bool>(timestep >= this->curMinSeqLength())+n)),
                    fn);
            }

            // backward states
            if (m_isBidirectional) {
                fn.prevOutputDistance = +n;
                fn.igPeepWeights     += els;
                fn.fgPeepWeights     += els;
                fn.ogPeepWeights     += els;
                fn.cellStates         = helpers::getRawPointer(m_bw.cellStates);
                fn.niActs             = helpers::getRawPointer(m_bw.niActs);
                fn.igActs             = helpers::getRawPointer(m_bw.igActs);
                fn.fgActs             = helpers::getRawPointer(m_bw.fgActs);
                fn.ogActs             = helpers::getRawPointer(m_bw.ogActs);
                fn.cellStateErrors    = helpers::getRawPointer(m_bw.cellStateErrors);
                fn.niDeltas           = helpers::getRawPointer(m_bw.niDeltas);
                fn.igDeltas           = helpers::getRawPointer(m_bw.igDeltas);
                fn.fgDeltas           = helpers::getRawPointer(m_bw.fgDeltas);
                fn.ogDeltas           = helpers::getRawPointer(m_bw.ogDeltas);

                for (int timestep = 0; timestep < this->curMaxSeqLength(); ++timestep) {
                    // collect errors from previous timestep
                    if (timestep != 0) {

			if (m_clockRNN){
			    m_bw.timestepMatrices[timestep].tmpOutputErrors.addProduct(
				m_bw.timestepMatrices[timestep-1].niH2HWrap, false,
				m_bw.timestepMatrices[timestep-1].niDeltas, false);
			    m_bw.timestepMatrices[timestep].tmpOutputErrors.addProduct(
				m_bw.timestepMatrices[timestep-1].igH2HWrap, false,
				m_bw.timestepMatrices[timestep-1].igDeltas, false);
			    m_bw.timestepMatrices[timestep].tmpOutputErrors.addProduct(
				m_bw.timestepMatrices[timestep-1].fgH2HWrap, false,
				m_bw.timestepMatrices[timestep-1].fgDeltas, false);
			    m_bw.timestepMatrices[timestep].tmpOutputErrors.addProduct(
				m_bw.timestepMatrices[timestep-1].ogH2HWrap, false,
				m_bw.timestepMatrices[timestep-1].ogDeltas, false);

			}else{
			    m_bw.timestepMatrices[timestep].tmpOutputErrors.addProduct(
				m_bw.weightMatrices.niInternal, false,
				m_bw.timestepMatrices[timestep-1].niDeltas, false);
			    m_bw.timestepMatrices[timestep].tmpOutputErrors.addProduct(
				m_bw.weightMatrices.igInternal, false,
				m_bw.timestepMatrices[timestep-1].igDeltas, false);
			    m_bw.timestepMatrices[timestep].tmpOutputErrors.addProduct(
				m_bw.weightMatrices.fgInternal, false,
				m_bw.timestepMatrices[timestep-1].fgDeltas, false);
			    m_bw.timestepMatrices[timestep].tmpOutputErrors.addProduct(
				m_bw.weightMatrices.ogInternal, false,
				m_bw.timestepMatrices[timestep-1].ogDeltas, false);
			}
                    }
		    
		    if (m_clockRNN)
			fn.skipCRNN  = helpers::getRawPointer(m_bw.skipCR) + 
			    m_bw.timestepMatrices[timestep].skipCRPos;
		    else
			fn.skipCRNN = NULL;
		    

                    // compute errors
                    thrust::for_each(
                      thrust::make_zip_iterator(
			thrust::make_tuple(
			  m_bw.tmpOutputErrors.begin() + n*timestep,
			  thrust::counting_iterator<int>(n*timestep),
			  thrust::constant_iterator<bool>(!timestep),
			  thrust::constant_iterator<bool>(timestep == this->curMaxSeqLength()-1),
			  thrust::constant_iterator<bool>(timestep >= this->curMinSeqLength()))),
		      thrust::make_zip_iterator(
			thrust::make_tuple(
			  m_bw.tmpOutputErrors.begin() + n*timestep + n,
			  thrust::counting_iterator<int>(n*timestep)+ n,
			  thrust::constant_iterator<bool>(!timestep)+ n,
			  thrust::constant_iterator<bool>(timestep == this->curMaxSeqLength()-1)+n,
			  thrust::constant_iterator<bool>(timestep >= this->curMinSeqLength())+n)),
                        fn);
                }
            }
        }}

	// set the gradient for skipped unit to zero
	// The reason is that, there is only one buffer to store the gradient for the activation
	// of each gate. For skipped unit, gradient must be propagated to previous LSTM units,
	// but not to the previous layer and weight/bias
	if (m_clockRNN){
	    {{
		internal::CleanUnitDeltasClockLSTM fn;
		fn.skipCRNN    = helpers::getRawPointer(m_fw.skipCR);
		fn.unitDeltas  = helpers::getRawPointer(m_fw.niDeltas);
		thrust::for_each(
		   thrust::make_zip_iterator(
			thrust::make_tuple(
				thrust::counting_iterator<int>(0),
				m_fw.skipCR.begin())),
		   thrust::make_zip_iterator(
			thrust::make_tuple(
				thrust::counting_iterator<int>(0) + m_fw.skipCR.size(),
				m_fw.skipCR.end())),
		   fn);

		if (m_isBidirectional){
		    fn.skipCRNN    = helpers::getRawPointer(m_bw.skipCR);
		    fn.unitDeltas  = helpers::getRawPointer(m_bw.niDeltas);
		    thrust::for_each(
		      thrust::make_zip_iterator(
			thrust::make_tuple(
				thrust::counting_iterator<int>(0),
				m_bw.skipCR.begin())),
		      thrust::make_zip_iterator(
			thrust::make_tuple(
				thrust::counting_iterator<int>(0) + m_bw.skipCR.size(),
				m_bw.skipCR.end())),
		      fn);
		    
		}
	    }}
	}

	
        // back-propagate the error to the preceding layer
        {{
            Layer<TDevice> *pl =
		dynamic_cast<Layer<TDevice>*>(&this->precedingLayer());
	    
            if (pl) {
                helpers::Matrix<TDevice> plErrorsMatrix(
			&pl->outputErrors(), pl->size(),
			this->curMaxSeqLength() * this->parallelSequences());

                // forward states
                plErrorsMatrix.assignProduct(m_fw.weightMatrices.niInput, false,
					     m_fw.niDeltasMatrix,         false);
                plErrorsMatrix.addProduct   (m_fw.weightMatrices.igInput, false,
					     m_fw.igDeltasMatrix,         false);
                plErrorsMatrix.addProduct   (m_fw.weightMatrices.fgInput, false,
					     m_fw.fgDeltasMatrix,         false);
                plErrorsMatrix.addProduct   (m_fw.weightMatrices.ogInput, false,
					     m_fw.ogDeltasMatrix,         false);

                // backward states
                if (m_isBidirectional) {
                    plErrorsMatrix.addProduct(m_bw.weightMatrices.niInput, false,
					      m_bw.niDeltasMatrix,         false);
                    plErrorsMatrix.addProduct(m_bw.weightMatrices.igInput, false,
					      m_bw.igDeltasMatrix,         false);
                    plErrorsMatrix.addProduct(m_bw.weightMatrices.fgInput, false,
					      m_bw.fgDeltasMatrix,         false);
                    plErrorsMatrix.addProduct(m_bw.weightMatrices.ogInput, false,
					      m_bw.ogDeltasMatrix,         false);
                }
            }else{
		throw std::runtime_error("Impossible error: previous layer is not a layer");
	    }
	    
        }}

        // compute the weight updates
        {{
            internal::ComputeWeightUpdateFn fn;
            fn.layerSize             = this->size();
            fn.effLayerSize          = this->size() / (m_isBidirectional ? 2 : 1);
            fn.precLayerSize         = this->precedingLayer().size();
            fn.timestepDistance      = (this->parallelSequences() * this->size() /
					(m_isBidirectional ? 2 : 1));
            fn.parallelSequences     = this->parallelSequences();
            fn.patternsCount         = this->curMaxSeqLength() * this->parallelSequences();
	    
	    // offset to the weight section
            fn.biasWeightsOffset     =  this->size() * this->precedingLayer().size() * 4;
            fn.internalWeightsOffset =  fn.biasWeightsOffset     + this->size() * 4;
            fn.peepholeWeightsOffset = (fn.internalWeightsOffset +
					this->size() * fn.effLayerSize * 4);
            fn.bias                  = this->bias();

	    // 
            fn.plOutputs             = helpers::getRawPointer(this->precedingLayer().outputs());
            fn.fwOutputs             = helpers::getRawPointer(m_fw.tmpOutputs);
            fn.bwOutputs             = helpers::getRawPointer(m_bw.tmpOutputs);
            fn.fwCellStates          = helpers::getRawPointer(m_fw.cellStates);
            fn.bwCellStates          = helpers::getRawPointer(m_bw.cellStates);
            fn.fwNiDeltas            = helpers::getRawPointer(m_fw.niDeltas);
            fn.bwNiDeltas            = helpers::getRawPointer(m_bw.niDeltas);
            fn.fwIgDeltas            = helpers::getRawPointer(m_fw.igDeltas);
            fn.bwIgDeltas            = helpers::getRawPointer(m_bw.igDeltas);
            fn.fwFgDeltas            = helpers::getRawPointer(m_fw.fgDeltas);
            fn.bwFgDeltas            = helpers::getRawPointer(m_bw.fgDeltas);
            fn.fwOgDeltas            = helpers::getRawPointer(m_fw.ogDeltas);
            fn.bwOgDeltas            = helpers::getRawPointer(m_bw.ogDeltas);

	    // elementwise operation
            thrust::transform(
                thrust::counting_iterator<int>(0),
                thrust::counting_iterator<int>(0) + (int)this->weightUpdates().size(),
                this->_weightUpdates().begin(),
                fn);
        }}

        // re-swap the output errors and the tmp output errors of the forward pass
        if (!m_isBidirectional) {
            this->outputErrors().swap(m_fw.tmpOutputErrors);
            this->_outputs()    .swap(m_fw.tmpOutputs);
        }
    }
    
    template <typename TDevice>
    void LstmLayer<TDevice>::exportLayer(const helpers::JsonValue &layersArray, 
					 const helpers::JsonAllocator &allocator) const
    {
        TrainableLayer<TDevice>::exportLayer(layersArray, allocator);
        (*layersArray)[layersArray->Size() - 1].AddMember("clock", m_crStepStr.c_str(), allocator);
    }


    template <typename TDevice>
    int LstmLayer<TDevice>::hiddenStateSize()
    {
	// return the size of hidden state and cell state for one frame
	return this->size() * 2;
    }

    template <typename TDevice>
    void LstmLayer<TDevice>::retrieveHiddenState(const int timeStep, real_vector& readBuffer)
    {
	// cannot be used for bi-directional layer
	if (m_isBidirectional)
	    throw std::runtime_error("retrieveHiddenState not implemented for BLSTM");
	if (timeStep >= this->curMaxSeqLength())
	    throw std::runtime_error("retrieveHiddenState time larger than expected");
	// assume readbuffer has been allocated
	int rows   = this->size();
	int cols   = this->parallelSequences();
	int offset = timeStep * rows * cols;
	thrust::copy(this->_outputs().begin() + offset, this->_outputs().begin() + offset + rows,
		     readBuffer.begin());
	thrust::copy(m_fw.cellStates.begin() + offset, m_fw.cellStates.begin() + offset + rows,
		     readBuffer.begin() + rows);
       
    }
    
    template <typename TDevice>
    void LstmLayer<TDevice>::setHiddenState(const int timeStep, real_vector& writeBuffer)
    {
	// cannot be used for bi-directional layer
	if (m_isBidirectional)
	    throw std::runtime_error("setHiddenState not implemented for BLSTM");
	if (timeStep >= this->curMaxSeqLength())
	    throw std::runtime_error("setHiddenState time larger than expected");
	if (writeBuffer.size() < this->size()*2)
	    throw std::runtime_error("setHiddenState vector dimension is smaller");
	int rows   = this->size();
	int cols   = this->parallelSequences();
	int offset = timeStep * rows * cols;
	thrust::copy(writeBuffer.begin(), writeBuffer.begin()+rows, 
		     this->_outputs().begin() + offset);
	thrust::copy(writeBuffer.begin()+rows, writeBuffer.begin()+rows*2,
		     m_fw.cellStates.begin() + offset);

    }
    

    
    // explicit template instantiations
    template class LstmLayer<Cpu>;
    template class LstmLayer<Gpu>;

} // namespace layers
