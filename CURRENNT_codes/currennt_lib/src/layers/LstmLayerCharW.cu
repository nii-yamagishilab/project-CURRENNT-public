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

#include "LstmLayerCharW.hpp"
#include "../helpers/limitedError.cuh"
#include "../helpers/getRawPointer.cuh"
#include "../helpers/Matrix.hpp"
#include "../activation_functions/Logistic.cuh"
#include "../activation_functions/Tanh.cuh"
#include "../helpers/safeExp.cuh"
#include "../helpers/JsonClasses.hpp"

#include "../Configuration.hpp"

#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/for_each.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>

#include <fstream>
#include <cmath>

#define CPNUM 3  // parameter number of the character window (alpha, beta, kappa)

// #define DEBUG_LOCAL_LSTMCHARW 1     // I turn on DEBUG_LOCAL for debugging


namespace internal {
namespace {

    typedef activation_functions::Logistic gate_act_fn_t;
    typedef activation_functions::Tanh     cell_input_act_fn_t;
    typedef activation_functions::Tanh     cell_output_act_fn_t;
    
    template <typename TActFn>
    struct ComputeOutputFn
    {
        int    layerSize;
        real_t bias;
        const real_t *biasWeights;

        __host__ __device__ real_t operator() (real_t a, const int &outputIdx) const
        {
            // calculate indices
            int blockIdx = outputIdx % layerSize; 
            // add the bias
            a += bias * biasWeights[blockIdx];
            // apply the activation function
            real_t b = TActFn::fn(a);
            // store the activation
            return b;
        }
    };

    
    struct ComputeBlockOutputFn
    {
        int    effLayerSize;
        int    prevOutputDistance;
        real_t bias;

        const char *patTypes;

        const real_t *niBiasWeights;
        const real_t *igBiasWeights;
        const real_t *fgBiasWeights;
        const real_t *ogBiasWeights;

        const real_t *igPeepWeights;
        const real_t *fgPeepWeights;
        const real_t *ogPeepWeights;

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
    };

    struct ResortOutputsFn
    {

        int layerSize;
	int layerlstmSize;
        int effLayerSize;

        const real_t *fwOutputs;
        const real_t *bwOutputs;
	const real_t *charWOuts;

        __host__ __device__ real_t operator() (const int &outputIdx) const
        {
            // calculate indices
            int patIdx = outputIdx / layerSize;
            int valIdx = outputIdx % layerSize;
	    
	    int chavalIdx = (valIdx - layerlstmSize);


            // store the value
	    if (chavalIdx < 0){	
		int offsetLSTM = patIdx * effLayerSize + valIdx;
		if (valIdx < effLayerSize)
		    return fwOutputs[offsetLSTM];
		else
		    return bwOutputs[offsetLSTM - effLayerSize];
	    }
	    else{
		int offsetChaW = patIdx * (layerSize - layerlstmSize);
		return charWOuts[offsetChaW + chavalIdx];
	    }
        }
    };

    struct ResortOutputErrorsFn
    {
        int layerSize;
	int layerlstmSize;
        int effLayerSize;

        real_t *fwOutputErrors;
        real_t *bwOutputErrors;
	real_t *charWoutErrors;

        __host__ __device__ void operator() (const thrust::tuple<const real_t&, int> &t) const
        {
            // unpack the tuple
            real_t outputErr = t.get<0>();
            int    outputIdx = t.get<1>();

            // calculate indices
            int patIdx = outputIdx / layerSize; // time step
            int valIdx = outputIdx % layerSize; 
	    
	    int chavalIdx = (valIdx - layerlstmSize);


            // store the value
	    if (chavalIdx < 0){	 
		// errors from the LSTM part
		int offsetLSTM = patIdx * effLayerSize + valIdx;
		if (valIdx < effLayerSize)
		    fwOutputErrors[offsetLSTM] = outputErr;
		else
		    bwOutputErrors[offsetLSTM - effLayerSize] = outputErr;
	    }
	    else{
		// errors for the CharW
		int offsetChaW = patIdx * (layerSize - layerlstmSize);
		charWoutErrors[offsetChaW + chavalIdx] =  outputErr;
	    }


            // calculate indices
            /*patIdx = outputIdx / layerSize;
            int valIdx = outputIdx % layerSize;
            int offset = patIdx * effLayerSize + valIdx;

            // store the value
            if (valIdx < effLayerSize)
                fwOutputErrors[offset] = outputErr;
            else
	    bwOutputErrors[offset - effLayerSize] = outputErr;*/
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

        real_t *cellStateErrors;
        real_t *niDeltas;
        real_t *igDeltas;
        real_t *fgDeltas;
        real_t *ogDeltas;

        __host__ __device__ void operator() (const thrust::tuple<const real_t&, int, bool, 
					                               bool, bool> &t) const
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

            // load the niag activations, the cell state and the output error
            real_t niAct     = niActs      [outputIdx];
            real_t igAct     = igActs      [outputIdx];
            real_t ogAct     = ogActs      [outputIdx];
            real_t cellState = cellStates  [outputIdx];

            // calculate the output gate delta
            real_t ogDelta = gate_act_fn_t::deriv(ogAct) * 
		             cell_output_act_fn_t::fn(cellState) * outputErr;

            // calculate the cell state error
            real_t ogPeepWeight = ogPeepWeights[blockIdx];
            real_t cellStateErr = ogAct * 
		                  cell_output_act_fn_t::deriv(cell_output_act_fn_t::fn(cellState)) *
		                  outputErr + ogPeepWeight * ogDelta;

            if (!firstCall) {
                real_t nextFgAct        = fgActs         [outputIdx - prevOutputDistance];
                real_t nextCellStateErr = cellStateErrors[outputIdx - prevOutputDistance];
                real_t nextIgDelta      = igDeltas       [outputIdx - prevOutputDistance];
                real_t nextFgDelta      = fgDeltas       [outputIdx - prevOutputDistance];
                
                real_t igPeepWeight = igPeepWeights[blockIdx];
                real_t fgPeepWeight = fgPeepWeights[blockIdx];

                cellStateErr += nextFgAct * nextCellStateErr + 
                                igPeepWeight * nextIgDelta + fgPeepWeight * nextFgDelta;
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
                    int plBlockIdx     = inputWeightIdx % precLayerSize;
                    int blockIdx       = ((inputWeightIdx - weightTypeY * (biasWeightsOffset / 4))
					  / precLayerSize);

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
                    int srcBlockIdx     = internalWeightIdx % effLayerSize;
                    int blockIdx      = internalWeightIdx / effLayerSize - weightTypeY * layerSize;

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
    
    // Compute the update for the lstm2ChaW and Charw -> lstm weight
    struct ComputeWeightUpdateForCharW1
    {
	// the gradients
	real_t *fwNiDeltas;
	real_t *bwNiDeltas;
	real_t *fwIgDeltas;
	real_t *bwIgDeltas;
	real_t *fwFgDeltas;
	real_t *bwFgDeltas;
	real_t *fwOgDeltas;
	real_t *bwOgDeltas;
	real_t *charWOutput;
	bool blstm;
	int patternsCount;
	int chaOutSize;
	int lstmSize;

	__host__ __device__ real_t operator() (const int &cntIdx) const
	{
	    
	    // step1. which matrix? which direction? which dim?
	    int dimCIdx  = (cntIdx % chaOutSize);   // which dimension of the CharW output data
	    int tmp      = (cntIdx / chaOutSize);  
	    int dimBLIdx = (tmp / lstmSize);        // which block (input, forget, or ...)
	    int dimWIdx  = (tmp % lstmSize);        // which dimension of lstm

	    real_t *deltaPtr;
	    real_t *outputPtr;
	    int offsetDel   = blstm? (lstmSize/2) : lstmSize;
	    int offsetOut   = blstm? chaOutSize*2: chaOutSize;
	    
	    bool bw = (blstm && (dimWIdx > lstmSize/2));
	    outputPtr = charWOutput + dimCIdx + (bw?chaOutSize:0);
	    switch(dimBLIdx){
	    case 0:
		{ // cell state
		    if (bw){
			deltaPtr = bwNiDeltas;
		    }else{
			deltaPtr = fwNiDeltas;
		    }
		}
		break;

	    case 1:
		{ // the input block
		    if (bw){
			deltaPtr = bwIgDeltas;
		    }else{
			deltaPtr = fwIgDeltas;
		    }
		}
		break;
	    case 2:
		{ // forget
		    if (bw){
			deltaPtr = bwFgDeltas;
		    }else{
			deltaPtr = fwFgDeltas;
		    }

		}
		break;

	    case 3:
		{ // output
		    if (bw){
			deltaPtr = bwOgDeltas;
		    }else{
			deltaPtr = fwOgDeltas;
		    }
		}
		break;
	    }
	    
	    real_t wu = 0;
	    for (int i = 0; i < patternsCount; i++){
		wu += (*deltaPtr) * (*outputPtr);
		deltaPtr = deltaPtr + offsetDel;
		outputPtr= outputPtr+ offsetOut;
	    }
	    return wu;
	}
    };


    struct ComputeWeightUpdateForCharW2
    {
	// the gradients
	real_t *charWinputError;
	real_t *lstmOutputFw;
	real_t *lstmOutputBw;
	
	bool blstm;
	int patternsCount;
	int chaInSize;
	int lstmSize;

	__host__ __device__ real_t operator() (const int &cntIdx) const
	{
	    
	    int effLstmSize = blstm?(lstmSize/2):lstmSize;

	    // step1. which matrix? which direction? which dim?
	    int dimLIdx  = (cntIdx % effLstmSize);   // which dimension of LSTM output
	    int dimCIdx  = (cntIdx / effLstmSize);   // which dimension of CharInput

	    real_t *deltaPtr;
	    real_t *outputPtr;
	    int offsetOut   = effLstmSize;
	    int offsetDel   = blstm? chaInSize*2: chaInSize;
	    
	    bool bw =  (blstm && dimLIdx > (effLstmSize));
	    
	    outputPtr = (bw?lstmOutputBw:lstmOutputFw) + dimLIdx;
	    deltaPtr  = charWinputError + dimCIdx + (bw?chaInSize:0);
	    real_t wu = 0;
	    for (int i = 0; i < patternsCount; i++){
		wu += (*deltaPtr) * (*outputPtr);
		deltaPtr = deltaPtr + offsetDel;
		outputPtr= outputPtr+ offsetOut;
	    }
	    return wu;
	}
    };

    struct ComputeWeightUpdateForCharW3
    {
	// the gradients
	real_t *charWinputError;
	int patternsCount;
	int chaInSize;

	__host__ __device__ real_t operator() (const int &cntIdx) const
	{
	    
	    
	    real_t *deltaPtr  = charWinputError + cntIdx;
	    real_t wu = 0;
	    for (int i = 0; i < patternsCount; i++){
		wu += (*deltaPtr);
		deltaPtr = deltaPtr + chaInSize;
	    }
	    return wu;
	}
    };
    
    // Extract feature from input char string for one data frame
    struct CharWParaTrans
    {
	real_t *paraBuffCurr;   // parameter of the current step
	real_t *paraBuffLast;   // parameter of the previous step
	real_t *biasWeight;     //
	bool boundary;          // whether this is the boundary ?
	
	__host__ __device__ void operator() (const int &idx) const
        {
	   
	    int dimIdx = (idx % CPNUM);
	    switch (dimIdx)
		{
		case 0:
		    
		case 1:
		    // alpha f(wx+b)
		    (*(paraBuffCurr + idx)) = helpers::safeExp(*(paraBuffCurr+idx) + 
							       (*(biasWeight+idx)));
		    break;
		case 2:
		    // kappa
		    (*(paraBuffCurr + idx)) = helpers::safeExp(*(paraBuffCurr + idx) + 
								   (*(biasWeight + idx))) + 
			(boundary?0:(*(paraBuffLast + idx)));	   
		default:
		    return;
		}
	}
    };
    
    struct CharWPara2Rho
    {
	
	real_t *rhoPtr;         // rho(k,u,t)
	real_t *chaPara;        // char para [alpha, beta, keppa]
	int mixNum;             //
	__host__ __device__ void operator() (const int &dataIdx) const
	{
	    int mixIdx = (dataIdx % mixNum);
	    int chaIdx = (dataIdx / mixNum);
	    real_t *mixPtr = chaPara + mixIdx * CPNUM;
	    // Rho
	    (*(rhoPtr+(mixNum)*chaIdx + mixIdx)) = *(mixPtr)* 
		helpers::safeExp( -1 * (*(mixPtr + 1))      *
				  ((*(mixPtr + 2))-chaIdx)  *
				  ((*(mixPtr + 2))-chaIdx));
	}
    
    };

    struct CharWParaFeatureExtract
    {
	real_t *charMatrix;
	real_t *rhoTmpput;
	real_t *featureOut;
	int curStrLength;
	int charDim;
	int mixNM;
	
	__host__ __device__ void operator() (const int &dimIdx) const
	{
	    real_t tmp = 0.0;
	    
	    for (int j = 0; j<curStrLength; j++){
		for (int i = 0; i<mixNM; i++){
		    tmp += (*(charMatrix + dimIdx + j*charDim)) * (*(rhoTmpput + i + j*mixNM));
		}
	    }
	    *(featureOut + dimIdx) = tmp;
	}
    };
    
    struct CharWParaFeatureExtractBack
    {
	real_t *charMatrix;
	real_t *rhoTmpput;
	real_t *featureOut;
	int curStrLength;
	int charDim;
	int mixNM;
	
	__host__ __device__ void operator() (const int &dimIdx) const
	{
	    real_t tmp = 0.0;
	    
	    for (int j = 0; j<curStrLength; j++){
		for (int i = 0; i<mixNM; i++){
		    tmp += (*(charMatrix + dimIdx + j*charDim)) * 
			(*(rhoTmpput + i + (curStrLength-1-j)*mixNM));
		}
	    }
	    *(featureOut + dimIdx) = tmp;
	}
    };
    
    struct PrepareRhoErrorBuf
    {
	// Prepare the error buf to propagate the error from output of CharW to
	// the input of CharW
	// Assume [alpha, beta, kappa] parameter set
	// buffer for alpha_t_k = [\rho_k,t,1, ... \rho_k,t,chaLength]
	//            beta_t_k  = [-\beta_k,t * (kappa_k,t-1)^2 \rho_k,t,1, ...]
	//            kappa_t_k = [-2\beta_k,t * (kappa_k,t-1) \rho_k,t,1, ...]
	real_t *rhoPtr;
	real_t *buffPtr;
	real_t *chaWPara;
    	int mixNum;          // 
	int paraNum;         // default as 3
	__host__ __device__ void operator() (const int& idx) const
	{
	    int chaPos = (idx / (mixNum * paraNum));    // u: position in the string
	    int bufIdx = (idx % (mixNum * paraNum));    
	    int mixIdx = (bufIdx / paraNum);            // which mixture ?
	    int parIdx = (bufIdx % paraNum);            // alpha, beta or kappa?

	    real_t tmp;
	    switch (parIdx){
	    case 0:
		// alpha
		*(buffPtr + idx) = *(rhoPtr + chaPos * mixNum + mixIdx);
		break;
	    case 1:
		// beta
		 tmp       = ((*(chaWPara + mixIdx * paraNum + 2)) - chaPos);
		*(buffPtr + idx) = -1 * 
		                   (*(chaWPara + mixIdx * paraNum + 1)) * 
		                   tmp * tmp *
		                   (*(rhoPtr   + chaPos * mixNum  + mixIdx));
		break;
	    case 2:
		// kappa
		tmp       = ((*(chaWPara + mixIdx * paraNum + 2)) - chaPos);
		*(buffPtr + idx) = -2 * 
		                   (*(chaWPara + mixIdx * paraNum + 1)) * 
		                   tmp *
		                   (*(rhoPtr   + chaPos * mixNum  + mixIdx));
		break;
	    default:
		break;
	    }
	}
	
    };
    
    struct KappaGradient
    {
	real_t *chaParaPtr;
	real_t *chaParaPtrNext;
	real_t *chaParaEPtr;	
	real_t *chaParaEPtrNext;       //
	bool boundaryE;               // whether this is the boundary (the first step of BP)?
	bool boundaryB;               // whether this is the boudnary (the last step of BP)?
	int  mixNum;
	
	__host__ __device__ void operator() (const int& mixIdx) const
	{
	    real_t *kappaEPtr  = chaParaEPtr     + mixIdx*CPNUM + 2;
	    real_t *kappaEPPtr = chaParaEPtrNext + mixIdx*CPNUM + 2;
	    real_t *kappaPtr   = chaParaPtr      + mixIdx*CPNUM + 2;
	    
	    // handling the previous step (or next step? anyway)
	    if (!boundaryE){	
		*kappaEPtr  = (*kappaEPPtr) - (*kappaEPtr);
		
		real_t *kappaPPtr = chaParaPtrNext + mixIdx*CPNUM + 2;
		*kappaEPPtr = (*kappaEPPtr) * (*kappaPPtr - *kappaPtr);
	    }

	    // if this is the begining, no next step to hand current step, do it now
	    if (boundaryB){
		(*kappaEPtr) = (*kappaEPtr) * (*kappaPtr);
	    }
	}
    };
    
    struct CharWParaError
    {
	// This function propogate the feaVecErrors through CharW to charErrors 
	// (for each timestep)
	
	// Input:
	//    CharW -> rho
	//    CharW -> chapara
	//    CharW -> feaVecErrors
	//    CharW -> strWord
	// Output:
	//    CharW -> chaparaErrors

	real_t *rhoPtr;
	real_t *tau_t;
	real_t *chaParaPtr;
	real_t *chaParaPtrNext;

	real_t *chaParaEPtr;	
	real_t *chaParaEPtrNext; //
	bool boundaryE;               // whether this is the boundary (the first step of BP)?
	bool boundaryB;               // whether this is the boudnary (the last step of BP)?
	int  curStrLength;
	int  mixNum;

	__host__ __device__ void operator() (const int& mixIdx) const
	{
	    real_t *alphaEPtr = chaParaEPtr + mixIdx*CPNUM;
	    real_t *betaEPtr  = chaParaEPtr + mixIdx*CPNUM + 1;
	    real_t *kappaEPtr = chaParaEPtr + mixIdx*CPNUM + 2;
	    
	    real_t *kappaEPPtr = chaParaEPtrNext + mixIdx*CPNUM + 2;

	    real_t *betaPtr  = chaParaPtr + mixIdx*CPNUM + 1;
	    real_t *kappaPtr = chaParaPtr + mixIdx*CPNUM + 2;
	    
	    *alphaEPtr = 0;
	    *betaEPtr  = 0;
	    *kappaEPtr = (boundaryE ? 0 : (*kappaEPPtr));
	    
	    
	    for (int i = 0; i< curStrLength; i++){
		(*alphaEPtr) +=  *(tau_t+i)    * (*(rhoPtr + i*mixNum + mixIdx));
		(*betaEPtr)  -=  *(tau_t+i)    * (*(rhoPtr + i*mixNum + mixIdx)) * 
		    ((*kappaPtr)-i) * ((*kappaPtr)-i) * (*betaPtr);
		
		(*kappaEPtr) -= 2*(*(tau_t+i)) * (*(rhoPtr + i*mixNum + mixIdx)) * 
		((*kappaPtr)-i) * (*betaPtr);
	    }
	    
	    // handling the previous step (or next step? anyway)
	    if (!boundaryE){	
		real_t *kappaPPtr = chaParaPtrNext + mixIdx*CPNUM + 2;
		*kappaEPPtr = (*kappaEPPtr) * (*kappaPPtr - *kappaPtr);
	    }

	    // if this is the begining, no next step to hand current step, do it now
	    if (boundaryB){
		(*kappaEPtr) = (*kappaEPtr) * (*kappaPtr);
	    }
	}

    };

    struct PrepareErrorBuf
    {
	// This function prepares the error buf for the backpropagation during each step
	// \sum_i ^ {charDim} gradient(t,i)c(u,i)
	real_t *tau_t;
	real_t *strWord;
	real_t *gradients;
	int chaDim;
	__host__ __device__ void operator() (const int& charPos) const
	{
	    *(tau_t + charPos) = 0.0;
	    for (int i = 0 ; i < chaDim; i++){
		(*(tau_t + charPos)) += (*(strWord + charPos*chaDim + i)) * (*(gradients + i));
	    }
	}
    };
    struct PrepareErrorBufBack
    {
	// This function prepares the error buf for the backpropagation during each step
	// \sum_i ^ {charDim} gradient(t,i)c(u,i)
	real_t *tau_t;
	real_t *strWord;
	real_t *gradients;
	int chaDim;
	int chaCurLength;

	__host__ __device__ void operator() (const int& charPos) const
	{
	    *(tau_t + charPos) = 0.0;
	    for (int i = 0 ; i < chaDim; i++){
		(*(tau_t + charPos)) += (*(strWord + (chaCurLength - 1 - charPos)*chaDim + i)) 
		    * (*(gradients + i));
	    }
	}
    };

    struct ReadInTxtData
    {
	real_t *txtBank;
	int *txtIdx;
	real_t *targetF;
	real_t *targetB;
	int chaDim;
	int curChaLength;
	__host__ __device__ void operator() (const int& idx) const
	{
	    int dim = (idx % chaDim);  // which dimension
	    int pos = (idx / chaDim);  // which character
	    int cha = *(txtIdx + pos);
	    *(targetF + idx) = *(txtBank + cha * chaDim + dim);
	    *(targetB + (curChaLength - 1 - pos)*chaDim + dim) = *(txtBank + cha * chaDim + dim);
	}
    };

    // read the txt data
    // Note: this function is similar to the readWeBank in InputLayer.cpp
    //       Actually, there is no need to declare this function publicly
    // The data is small, so ignore the overhead to copy the data 
    Cpu::real_vector readTxtBank(const unsigned dim)
    {
	const Configuration &config    = Configuration::instance();
	const std::string    txtBankPath(config.chaBankPath());
	Cpu::real_vector txtBank;
	
	if (txtBankPath.size()==0){
	    // no file to read
	    txtBank.clear();
	    return txtBank;
	}

	if (dim < 1)
	    throw std::runtime_error(std::string("Dimention of txtBank below 1"));
	
	std::ifstream ifs(txtBankPath.c_str(), std::ifstream::binary | std::ifstream::in);
	if (!ifs.good()){
	    throw std::runtime_error(std::string("Fail to open ")+txtBankPath);
	}
	
	
	// get the number of we data
	std::streampos numEleS, numEleE;
	long int       numEle;
	numEleS = ifs.tellg();
	ifs.seekg(0, std::ios::end);

	numEleE = ifs.tellg();
	numEle  = (numEleE-numEleS)/sizeof(real_t);
	ifs.seekg(0, std::ios::beg);
	
	// read in the data
	txtBank.resize(numEle, 0.0);
	
	real_t tempVal;
	std::vector<real_t> tempVec;

	for (unsigned int i = 0; i< numEle; i++){
	    ifs.read ((char *)&tempVal, sizeof(real_t));
	    tempVec.push_back(tempVal);
	}
	thrust::copy(tempVec.begin(), tempVec.end(), txtBank.begin());
	std::cout << "Read " << numEle/dim << " vectors for LstmCharW" << std::endl;
	
	ifs.close();
	return txtBank;
    }
    

    // Get the ChaDim. 
    // Configuration of the ChaDim is confusing. We can
    // 1. specify chadim in config.cfg
    // 2. or, we can use the chadim in network.jsn
    // Let's assume the chaDim in network has higher priority
    int getChaDim(int chaDim, const helpers::JsonValue &layerChild){
	int tmp = helpers::safeJsonGetInt(layerChild, "txtChaDim");
	if (tmp > 0){
	    if (chaDim > 0 && chaDim != tmp){
		printf("WARNING: options txtChaDim is not equal to that in .jsn .autosave\n");
		printf("         chaDim %d will be used.\n", tmp);
	    }
	    return tmp;
	}else{
	    if (chaDim < 1){
		throw std::runtime_error("Please specifiy --txtChaDim.");
	    }
	    return chaDim;
	}
    }

    // Get 
    int tempSize(int m_chaDim, int m_lstmDim){
	return std::ceil((real_t)m_chaDim/m_lstmDim);
    }
    
} // anonymous namespace
} // namespace internal






namespace layers {

    /***********************************************************************************
     * CharW unit
     * 
     ***********************************************************************************/

    template <typename TDevice>
    CharW<TDevice>::CharW(const int mixNum,  const int chaDim, 
			  const int lstmDim, const int paraNum,
			  const int chaMaxLength, 
			  const int seqMaxLength,
			  const int paraSequence,
			  real_vector *weight,      
			  real_vector *output,         
			  real_vector *outputErrors,
			  real_vector *outputAll,   
			  real_vector *outErrorsAll,
			  const bool blstm)
	: m_mixNum        (mixNum)
	, m_chaDim        (chaDim)
	, m_lstmDim       (lstmDim)
	, m_paraNum       (paraNum)
	, m_chaMaxLength  (chaMaxLength)
	, m_seqMaxLength  (seqMaxLength)
	, m_paraSequence  (paraSequence)
	, m_weightPtr     (weight)
	, m_outputsLstm   (output)
	, m_outErrorsLstm (outputErrors)
	, m_outputsAll    (outputAll)
	, m_outErrorsAll  (outErrorsAll)
    {
	// initialize the rhoVector rho[k, u, t]
	// two rho matrices for forward and backward in BLSTM
	Cpu::real_vector tmp;
	tmp.resize((blstm ? 2: 1) *m_mixNum * m_chaMaxLength * m_seqMaxLength, 0);
	m_rhoVec = tmp;
	
	tmp.resize((blstm ? 2: 1) *m_chaMaxLength * m_seqMaxLength, 0);
	m_phiVec = tmp;
	
	// initialize the buf for error propatation
	// [3k, U]^T,  
	// [rho_1_1_t, \beta_t^1 * (\kappa_t^1-1)^2 * \rho_1_1_t, ...
	//  rho_1_2_t, \beta_t^1 * (\kappa_t^1-2)^2 * \rho_1_1_2, ...]
	// 
	tmp.resize(m_chaMaxLength * CPNUM * m_mixNum);
	m_rhoErrorBuf = tmp;

	// initialize for simplicity
	m_tmpOne.resize(m_mixNum, 1.0);

	// initialize input to CharW
	// bi-directional:
	//     [alpha_t1_1_f, beta_t1_1_f, kappa_t1_1_f, 
	//      alpha_t1_2_f, beta_t1_2_f ...
	//      ....
	//	alpha_t1_1_b, beta_t1_1_b, kappa_t1_1_b,
	//      alpha_t1_2_b, beta_t1_2_b ...]
	// uni-directional:
	//     similar without *_b
	tmp.resize((m_seqMaxLength * m_paraNum * m_mixNum * (blstm ? 2 : 1)), 0);
	m_chaPara = tmp;      
	m_chaParaError = tmp;

	// initialize the output of CharW
	tmp.resize((m_seqMaxLength * m_chaDim * (blstm ? 2 : 1)), 0);
	m_feaVec   = tmp;
	m_feaVecError = tmp;

	// initialize the error buff
	// to store the \sum_i=1^{D} feaVecError(t,i)c(u,i)
	tmp.resize(m_chaMaxLength, 0);
	m_tau_t = tmp;


	
    }
    

    
    template <typename TDevice>
    int CharW<TDevice>::inSize() const
    {
        return m_paraNum * m_mixNum;
    }

    template <typename TDevice>
    int CharW<TDevice>::outSize() const
    {
        return m_chaDim;
    }

    template <typename TDevice>
    CharW<TDevice>::~CharW()
    {

    }

    template <typename TDevice>
    void CharW<TDevice>::computeForwardPass()
    {
    }

    template <typename TDevice>
    void CharW<TDevice>::computeBackwardPass()
    {

    }
    
    template <typename TDevice>
    void CharW<TDevice>::loadSequence(const data_sets::DataSetFraction &fraction)
    {
	// Load the maximum length of data
	m_seqCurLength = fraction.maxSeqLength();
	
	// read in the string of word
	m_chaCurLength = 0;//fraction.maxTxtLength();
	int_vector tmpIdx;
	//tmpIdx = //fraction.txtData();
	//if (m_chaCurLength != fraction.txtData().size())
	//    throw std::runtime_error("Please set parallel sequence = 1");
	
	m_strWord.resize(m_chaCurLength * m_chaDim, 0.0);
	m_strWordRev.resize(m_chaCurLength * m_chaDim, 0.0);
	
	{{
	    internal::ReadInTxtData fn;
	    fn.txtBank = helpers::getRawPointer(m_txtBank);
	    fn.txtIdx  = helpers::getRawPointer(tmpIdx);
	    fn.targetF = helpers::getRawPointer(m_strWord);
	    fn.targetB = helpers::getRawPointer(m_strWordRev);
	    fn.chaDim  = m_chaDim;
	    fn.curChaLength = m_chaCurLength;
	    thrust::for_each(thrust::counting_iterator<int>(0),
			     thrust::counting_iterator<int>(0) + m_chaCurLength * m_chaDim,
			     fn);
	}}
	

	// ---------- DEBUG ----------- //
        #ifdef DEBUG_LOCAL_LSTMCHARW	
	printf("\n");
	Cpu::int_vector tmpVec(m_strWord);
	for (int i =0 ;i < m_strWord.size(); i++){
	    printf("%d ", tmpVec[i]);
	    if (i % m_chaDim == (m_chaDim - 1))
		printf("\n");
	}
	tmpVec = m_strWordRev;
	printf("\n");
	for (int i =0 ;i < m_strWord.size(); i++){
	    printf("%d ", tmpVec[i]);
	    if (i % m_chaDim == (m_chaDim - 1))
		printf("\n");
	}
	tmpVec.clear();
        #endif
	
    }

    template <typename TDevice>
    CharW<TDevice>::real_vector& CharW<TDevice>::chaInput()
    {
	return m_chaPara;
    }
    template <typename TDevice>
    CharW<TDevice>::real_vector& CharW<TDevice>::chaOutput()
    {
	return m_feaVec;
    }
    template <typename TDevice>
    CharW<TDevice>::real_vector& CharW<TDevice>::chaInputError()
    {
	return m_chaParaError;
    }
    template <typename TDevice>
    CharW<TDevice>::real_vector& CharW<TDevice>::chaOutputError()
    {
	return m_feaVecError;
    }


    /***********************************************************************************
     * LSTMCharW layer
     *
     * LSTM -> CharW
     ***********************************************************************************/
    template <typename TDevice>
    LstmLayerCharW<TDevice>::LstmLayerCharW(const helpers::JsonValue &layerChild, 
					    const helpers::JsonValue &weightsSection,
					    Layer<TDevice> &precedingLayer,
					    const int chaDim, 
					    const int maxTxtLength, 
					    bool blstm)
        : m_lstmDim              (helpers::safeJsonGetInt(layerChild, "size"))
	, m_mixNum               (helpers::safeJsonGetInt(layerChild, "mixNum"))
	, m_chaDim               (internal::getChaDim(chaDim, layerChild))
	, m_isBidirectional      (blstm)
	, m_inWNumPerBlock       (
		(blstm?2:4)* helpers::safeJsonGetInt(layerChild, "size")+3 + 
		(CPNUM * helpers::safeJsonGetInt(layerChild, "mixNum")) +   
		(blstm?2:1)*internal::tempSize(chaDim,helpers::safeJsonGetInt(layerChild,"size"))+ 
		(chaDim * 4))
	, TrainableLayer<TDevice>(
		layerChild, weightsSection, 4, 
		(blstm?2:4)* helpers::safeJsonGetInt(layerChild, "size")+3 + 
		(CPNUM * helpers::safeJsonGetInt(layerChild, "mixNum")) +   
		(blstm?2:1)*internal::tempSize(chaDim,helpers::safeJsonGetInt(layerChild,"size"))+ 
		(chaDim * 4),
		precedingLayer, precedingLayer.maxSeqLength(), -1)
    {

	// Initialize CharW
	// ------------------------------------
	// Note, the equation to get the number of weights:
	//   m_lstmDim * (arg_3 * (precedingLayer_size() + 1) + arg_4)
	// 
	// In the original LSTM case:
	//   m_lstmDim * arg_3 * precedingLayer_size() : m_lstm * 4 * preceding_size
	//   m_lstmDim * arg_3                         : m_lstm * 4
	//   m_lstmDim * arg_4                         : m_lstm * (blstm?2:4) * m_lstm +
	//                                               m_lstm * 3
	// Now:
	//   m_lstmDim * arg_3 * precedingLayer_size() : m_lstm * 4 * preceding_size
	//   m_lstmDim * arg_3                         : m_lstm * 4
	//   m_lstmDim * arg_4                         : m_lstm * (blstm?2:4) * m_lstm +
	//                                               m_lstm * 3
	//                                               m_lstm * (CPNUM * mixNum)
	//                                               m_lstm * blstm?2:1
	//                                               m_lstm * chaDim * 4
	//                               
	// Arg_3, arg_4 are the 3rd and 4th arguments for TrainalableLayer
	// 
	// Compared with the standard lstm
	//   bi-directional case: 
	//       LSTM -> CharW (and bias)
	//       2 * CPNUM * mixtureNM * lstmDim / 2 + 2 * CPNUM * mixtureNM
	//       CharW -> LSTM (next step)
	//       4 * 2 * |CharW->outputDim()| * lstmDim / 2
	//       
	//   uni-directional case:
	//       LSTM -> CharW (and bias)
	//       1 * CPNUM * mixtureNM * lstmDim + CPNUM * mixtureNM
	//       CharW -> LSTM (next step)
	//       4 * 1 * |CharW->outputDim()| * lstmDim 
	// I wasted some memory units to store CharW bias 
	// (it should be layerSize*(blstm?2*1*mixNum*CPUM/layerSize)) 

	int chaMaxLength = maxTxtLength;

	// initial check
        if (m_isBidirectional && this->lstmSize() % 2 != 0)
            throw std::runtime_error("Cannot create a bidirectional layer with odd layer size");
	

	// resize output and outputErrors (dimension becomes lstmSize + charSize)
	int tmpSize = this->outputsLstm().size()/ m_lstmDim * (m_lstmDim + m_chaDim * (blstm?2:1));
	m_outputsAll  .resize(tmpSize, 0);
	m_outErrorsAll.resize(tmpSize, 0);

	// initialize the Char unit (skip the LSTM weights and bias)
	int seqMaxLength  = this->outputsLstm().size() / m_lstmDim;
	m_charW = boost::shared_ptr<CharW<TDevice> > (
			new CharW<TDevice>(m_mixNum, m_chaDim, m_lstmDim,CPNUM,
					   chaMaxLength, seqMaxLength, 
					   this->parallelSequences(),
					   (&this->weights()),
					   (&this->outputsLstm()),
					   (&this->outputErrorsLstm()),
					   (&m_outputsAll),
					   (&m_outErrorsAll),
					   blstm));

	// Read in the txt Bank information
	m_charW -> m_txtBank = internal::readTxtBank(chaDim);

	// Initiale Lstm
	// ------------------------------------
        // set raw pointers
        int ls  = this->lstmSize();
        int pls = this->precedingLayer().size();
	

	// point to the weight section
	// Note: because weights of CharW are attrached after the LSTM weight,
	//       they are transparent to the LSTM operation
        _rawNiBiasWeights     = helpers::getRawPointer(this->weights()) + 4 * ls * pls + 0 * ls;
        _rawIgBiasWeights     = helpers::getRawPointer(this->weights()) + 4 * ls * pls + 1 * ls;
        _rawFgBiasWeights     = helpers::getRawPointer(this->weights()) + 4 * ls * pls + 2 * ls;
        _rawOgBiasWeights     = helpers::getRawPointer(this->weights()) + 4 * ls * pls + 3 * ls;
        _rawIgPeepholeWeights = helpers::getRawPointer(this->weights()) + 4 * ls * pls + 4 * ls +
	    4 * ls * ls / (m_isBidirectional ? 2 : 1) + 0 * ls;
        _rawFgPeepholeWeights = helpers::getRawPointer(this->weights()) + 4 * ls * pls + 4 * ls +
	    4 * ls * ls / (m_isBidirectional ? 2 : 1) + 1 * ls;
        _rawOgPeepholeWeights = helpers::getRawPointer(this->weights()) + 4 * ls * pls + 4 * ls +
	    4 * ls * ls / (m_isBidirectional ? 2 : 1) + 2 * ls;

	// set the bias to weight(LSTM -> CharW)
	int ptrCharW      = m_lstmDim * ((precedingLayer.size()+1) * 4 + 
					 (blstm ? 2 : 4) * m_lstmDim   + 
					 3);
	m_lstmWeightNum   = ptrCharW;
	
	_rawLSTM2CharWBias= (helpers::getRawPointer(this->weights()) + 
			     ptrCharW + 
			     (m_charW->inSize() + 4 * m_charW->outSize() ) * ls);


        // create the forward and backward info structs
        forward_backward_info_t* fwbwArr[] = { &m_fw, &m_bw };
        for (int fwbwArrIdx = 0; fwbwArrIdx < (m_isBidirectional ? 2 : 1); ++fwbwArrIdx) {
            forward_backward_info_t *fwbw = fwbwArr[fwbwArrIdx];

            // calculate sizes
            int pls = this->precedingLayer().size();
            int ls  = this->lstmSize();
            int els = this->lstmSize() / (m_isBidirectional ? 2 : 1);

            // cell states, niags, deltas, ...
            Cpu::real_vector tmp(this->outputsLstm().size() / (m_isBidirectional ? 2 : 1), 0);
	    fwbw->tmpOutputs      = tmp;
	    fwbw->tmpOutputErrors = tmp;
	    
	    // Note: different from the Lstm case, we can't just simply swap
	    // Let's waste some memory space here
	    /*if (m_isBidirectional) {
            }
            else {
                fwbw->tmpOutputs     .swap(this->outputsLstm());
                fwbw->tmpOutputErrors.swap(this->outputErrorsLstm());
	    }
	    */

            fwbw->cellStates      = tmp;
            fwbw->cellStateErrors = tmp;
            fwbw->niActs          = tmp;
            fwbw->igActs          = tmp;
            fwbw->fgActs          = tmp;
            fwbw->ogActs          = tmp;
            fwbw->niDeltas        = tmp;
            fwbw->igDeltas        = tmp;
            fwbw->fgDeltas        = tmp;
            fwbw->ogDeltas        = tmp;
	    
	    // CharW (output buffer)
	    // ERROR: this is not pointer !!! fwbw->tmpOutputsC != m_charW->chaOutput()
	    // fwbw->tmpOutputsC      = m_charW->chaOutput();
	    // fwbw->tmpOutputErrorsC = m_charW->chaOutputError();
	    fwbw->tmpOutputsC.clear();
	    fwbw->tmpOutputErrorsC.clear();

            // weight matrices (and weight update)
            weight_matrices_t* wmArr [] = { &fwbw->weightMatrices, &fwbw->weightUpdateMatrices };
            real_vector*       wtsArr[] = { &this->weights(),      &this->_weightUpdates() };
            for (int wmArrIdx = 0; wmArrIdx < 2; ++wmArrIdx) {
                weight_matrices_t *wm  = wmArr [wmArrIdx];
                real_vector       *wts = wtsArr[wmArrIdx];

                int nmInW     = ls * pls; // number of input weights
                int nmInterW  = ls * els; // number of internal weights
                int inWStart  = ((fwbwArrIdx == 1) ? (nmInW / 2) : 0); // input weight start
                int intWSTart = (((fwbwArrIdx == 1) ? (nmInterW / 2) : 0) 
				 + 4 * (ls * (pls + 1)));              // internal weight start

                wm->niInput = helpers::Matrix<TDevice>(wts, pls, els, inWStart + 0 * nmInW);
                wm->igInput = helpers::Matrix<TDevice>(wts, pls, els, inWStart + 1 * nmInW);
                wm->fgInput = helpers::Matrix<TDevice>(wts, pls, els, inWStart + 2 * nmInW);
                wm->ogInput = helpers::Matrix<TDevice>(wts, pls, els, inWStart + 3 * nmInW);

                wm->niInternal = helpers::Matrix<TDevice>(wts, els, els, intWSTart+ 0*nmInterW);
                wm->igInternal = helpers::Matrix<TDevice>(wts, els, els, intWSTart+ 1*nmInterW);
                wm->fgInternal = helpers::Matrix<TDevice>(wts, els, els, intWSTart+ 2*nmInterW);
                wm->ogInternal = helpers::Matrix<TDevice>(wts, els, els, intWSTart+ 3*nmInterW);
		
		
		// LSTM -> CharW
		int nBiasL2C = ((fwbwArrIdx==1)? (els * m_charW->inSize()):0);
		wm->lstm2CharW = helpers::Matrix<TDevice>(wts, els, m_charW->inSize(),
							  ptrCharW + nBiasL2C);
		// CharW -> LSTM (to 4 gates)
		nBiasL2C  = ls * m_charW->inSize();  // skip the Lstm->CharW
		nBiasL2C += ((fwbwArrIdx==1)? (m_charW->outSize() * els):0);
		int nCharW2LSTM = ls * m_charW->outSize();
		wm->niInternalC = helpers::Matrix<TDevice>(wts, m_charW->outSize(), els,
							   ptrCharW + nBiasL2C + 0*nCharW2LSTM);
		wm->igInternalC = helpers::Matrix<TDevice>(wts, m_charW->outSize(), els,
							   ptrCharW + nBiasL2C + 1*nCharW2LSTM);
		wm->fgInternalC = helpers::Matrix<TDevice>(wts, m_charW->outSize(), els,
							   ptrCharW + nBiasL2C + 2*nCharW2LSTM);
		wm->ogInternalC = helpers::Matrix<TDevice>(wts, m_charW->outSize(), els,
							   ptrCharW + nBiasL2C + 3*nCharW2LSTM);
		
            }

            // matrices for each timestep
            for (int timestep = 0; timestep < this->maxSeqLength(); ++timestep) {
                int rows   = this->lstmSize() / (m_isBidirectional ? 2 : 1);
                int cols   = this->parallelSequences();
                int offset = timestep * rows * cols;

                timestep_matrices_t tm;
                tm.tmpOutputs  = helpers::Matrix<TDevice>(&fwbw->tmpOutputs, rows, cols, offset);
                
                tm.niActs      = helpers::Matrix<TDevice>(&fwbw->niActs,     rows, cols, offset);
                tm.igActs      = helpers::Matrix<TDevice>(&fwbw->igActs,     rows, cols, offset);
                tm.fgActs      = helpers::Matrix<TDevice>(&fwbw->fgActs,     rows, cols, offset);
                tm.ogActs      = helpers::Matrix<TDevice>(&fwbw->ogActs,     rows, cols, offset);
                tm.niDeltas    = helpers::Matrix<TDevice>(&fwbw->niDeltas,   rows, cols, offset);
                tm.igDeltas    = helpers::Matrix<TDevice>(&fwbw->igDeltas,   rows, cols, offset);
                tm.fgDeltas    = helpers::Matrix<TDevice>(&fwbw->fgDeltas,   rows, cols, offset);
                tm.ogDeltas    = helpers::Matrix<TDevice>(&fwbw->ogDeltas,   rows, cols, offset);
		tm.tmpOutputErrors = helpers::Matrix<TDevice>(&fwbw->tmpOutputErrors, 
							      rows, cols, offset);

		// link the input for CharW
		rows   = m_charW->inSize();
		// weave the forward and backward buffers in one time step
		offset = rows * cols * (timestep * (blstm ? 2:1) + ((fwbwArrIdx==1) ? 1:0));
		
		tm.tmpCharPara         = helpers::Matrix<TDevice>(&(m_charW->chaInput()), 
								  rows, cols, offset);
		tm.tmpCharParaPtr      = helpers::getRawPointer(m_charW->chaInput()) + offset;

		tm.tmpCharParaError    = helpers::Matrix<TDevice>(&(m_charW->chaInputError()), 
								  rows, cols, offset);
		tm.tmpCharParaErrorPtr = helpers::getRawPointer(m_charW->chaInputError())+offset;

		tm.charParaOffset      = offset; 

		// link the output for CharW
		rows   = m_charW->outSize();
		offset = rows * cols * (timestep * (blstm ? 2:1) + ((fwbwArrIdx==1) ? 1:0));

		tm.tmpOutputsC         = helpers::Matrix<TDevice>(&m_charW->chaOutput(), 
								  rows, cols,offset);
		tm.tmpOutputsCPtr      = helpers::getRawPointer(m_charW->chaOutput()) + offset;

		tm.tmpOutputErrorsC    = helpers::Matrix<TDevice>(&m_charW->chaOutputError(), 
								  rows, cols, offset);
		tm.tmpOutputErrorsCPtr = helpers::getRawPointer(m_charW->chaOutputError())+ offset;

		tm.outputsCOffset      = offset;

		// link the rho matrix for 
		rows         = m_charW->m_mixNum * m_charW->m_chaMaxLength;
		offset       = rows * (timestep * (blstm ? 2:1) + ((fwbwArrIdx==1) ? 1:0));
		tm.tmpRho    = helpers::getRawPointer(m_charW->m_rhoVec) + offset;
		tm.rhoOffset = offset;

		fwbw->timestepMatrices.push_back(tm);
            }
        }

        //if (!m_isBidirectional) {
        //    m_fw.tmpOutputs     .swap(this->outputsLstm());
        //    m_fw.tmpOutputErrors.swap(this->outputErrorsLstm());
        //}
    }

    template <typename TDevice>
    LstmLayerCharW<TDevice>::~LstmLayerCharW()
    {
    }

    template <typename TDevice>
    const std::string& LstmLayerCharW<TDevice>::type() const
    {
        static const std::string su("lstmw");
        static const std::string sb("blstmw");
        return (m_isBidirectional ? sb : su);
    }

    template <typename TDevice>
    int LstmLayerCharW<TDevice>::size() const
    {
        return m_lstmDim + m_chaDim*(m_isBidirectional?2:1);
    }

    template <typename TDevice>
    int LstmLayerCharW<TDevice>::lstmSize() const
    {
        return m_lstmDim;
    }

    template <typename TDevice>
    bool LstmLayerCharW<TDevice>::isBidirectional() const
    {
        return m_isBidirectional;
    }

    template <typename TDevice>
    const typename TDevice::real_vector& LstmLayerCharW<TDevice>::cellStates() const
    {
        if (m_isBidirectional)
            throw std::runtime_error("Not implemented");
        else
            return m_fw.cellStates;
    }

    template <typename TDevice>
    const typename TDevice::real_vector& LstmLayerCharW<TDevice>::cellStateErrors() const
    {
        if (m_isBidirectional)
            throw std::runtime_error("Not implemented");
        else
            return m_fw.cellStateErrors;
    }

    template <typename TDevice>
    const typename TDevice::real_vector& LstmLayerCharW<TDevice>::netInputActs() const
    {
        if (m_isBidirectional)
            throw std::runtime_error("Not implemented");
        else
            return m_fw.niActs;
    }

    template <typename TDevice>
    const typename TDevice::real_vector& LstmLayerCharW<TDevice>::netInputDeltas() const
    {
        if (m_isBidirectional)
            throw std::runtime_error("Not implemented");
        else
            return m_fw.niDeltas;
    }

    template <typename TDevice>
    const typename TDevice::real_vector& LstmLayerCharW<TDevice>::inputGateActs() const
    {
        if (m_isBidirectional)
            throw std::runtime_error("Not implemented");
        else
            return m_fw.igActs;
    }

    template <typename TDevice>
    const typename TDevice::real_vector& LstmLayerCharW<TDevice>::inputGateDeltas() const
    {
        if (m_isBidirectional)
            throw std::runtime_error("Not implemented");
        else
            return m_fw.igDeltas;
    }

    template <typename TDevice>
    const typename TDevice::real_vector& LstmLayerCharW<TDevice>::forgetGateActs() const
    {
        if (m_isBidirectional)
            throw std::runtime_error("Not implemented");
        else
            return m_fw.fgActs;
    }

    template <typename TDevice>
    const typename TDevice::real_vector& LstmLayerCharW<TDevice>::forgetGateDeltas() const
    {
        if (m_isBidirectional)
            throw std::runtime_error("Not implemented");
        else
            return m_fw.fgDeltas;
    }

    template <typename TDevice>
    const typename TDevice::real_vector& LstmLayerCharW<TDevice>::outputGateActs() const
    {
        if (m_isBidirectional)
            throw std::runtime_error("Not implemented");
        else
            return m_fw.ogActs;
    }

    template <typename TDevice>
    const typename TDevice::real_vector& LstmLayerCharW<TDevice>::outputGateDeltas() const
    {
        if (m_isBidirectional)
            throw std::runtime_error("Not implemented");
        else
            return m_fw.ogDeltas;
    }

    template <typename TDevice>
    void LstmLayerCharW<TDevice>::loadSequences(const data_sets::DataSetFraction &fraction)
    {
        TrainableLayer<TDevice>::loadSequences(fraction, -1);
	
	if (m_charW)
	    m_charW->loadSequence(fraction);

        m_precLayerOutputsMatrix = helpers::Matrix<TDevice>(
				&this->precedingLayer().outputs(), 
				this->precedingLayer().size(), 
				this->curMaxSeqLength() * this->parallelSequences());

        // update the niag matrices
        forward_backward_info_t* fwbwArr[] = { &m_fw, &m_bw };
        for (int fwbwArrIdx = 0; fwbwArrIdx < (m_isBidirectional ? 2 : 1); ++fwbwArrIdx) {
            forward_backward_info_t *fwbw = fwbwArr[fwbwArrIdx];

            int rows = this->lstmSize() / (m_isBidirectional ? 2 : 1);
            int cols = this->curMaxSeqLength() * this->parallelSequences();
	    
	    // form the matrix for forward propagation previousding -> input 
            fwbw->niActsMatrix = helpers::Matrix<TDevice>(&fwbw->niActs, rows, cols);
            fwbw->igActsMatrix = helpers::Matrix<TDevice>(&fwbw->igActs, rows, cols);
            fwbw->fgActsMatrix = helpers::Matrix<TDevice>(&fwbw->fgActs, rows, cols);
            fwbw->ogActsMatrix = helpers::Matrix<TDevice>(&fwbw->ogActs, rows, cols);

	    // form the matrix for backward propagation previousding -> input 
            fwbw->niDeltasMatrix = helpers::Matrix<TDevice>(&fwbw->niDeltas, rows, cols);
            fwbw->igDeltasMatrix = helpers::Matrix<TDevice>(&fwbw->igDeltas, rows, cols);
            fwbw->fgDeltasMatrix = helpers::Matrix<TDevice>(&fwbw->fgDeltas, rows, cols);
            fwbw->ogDeltasMatrix = helpers::Matrix<TDevice>(&fwbw->ogDeltas, rows, cols);
        }
    }

    template <typename TDevice>
    void LstmLayerCharW<TDevice>::computeForwardPass()
    {
	// No need to swap
        //if (!m_isBidirectional) {
        //    m_fw.tmpOutputs.swap(this->outputsLstm());
        //}
	
        // sum up the activations from the preceding layer
        {{
            // forward states
            m_fw.niActsMatrix.assignProduct(m_fw.weightMatrices.niInput, true, 
					    m_precLayerOutputsMatrix, false);
            m_fw.igActsMatrix.assignProduct(m_fw.weightMatrices.igInput, true, 
					    m_precLayerOutputsMatrix, false);
            m_fw.fgActsMatrix.assignProduct(m_fw.weightMatrices.fgInput, true, 
					    m_precLayerOutputsMatrix, false);
            m_fw.ogActsMatrix.assignProduct(m_fw.weightMatrices.ogInput, true, 
					    m_precLayerOutputsMatrix, false);

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
            int els = this->lstmSize() / (m_isBidirectional ? 2 : 1);
            int n   = this->parallelSequences() * els;
	    int m   = this->parallelSequences() * els * m_charW->inSize();
	    
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
		    
		    // assume CharW is ready, add the CharW to LSTM 
                    m_fw.timestepMatrices[timestep].niActs.addProduct(
                                m_fw.weightMatrices.niInternalC, true, 
				m_fw.timestepMatrices[timestep-1].tmpOutputsC, false);
                    m_fw.timestepMatrices[timestep].igActs.addProduct(
                                m_fw.weightMatrices.igInternalC, true, 
				m_fw.timestepMatrices[timestep-1].tmpOutputsC, false);
                    m_fw.timestepMatrices[timestep].fgActs.addProduct(
                                m_fw.weightMatrices.fgInternalC, true, 
				m_fw.timestepMatrices[timestep-1].tmpOutputsC, false);
                    m_fw.timestepMatrices[timestep].ogActs.addProduct(
                                m_fw.weightMatrices.ogInternalC, true, 
				m_fw.timestepMatrices[timestep-1].tmpOutputsC, false);
		    
				
                }
		
		// For n = 0:this->parallelSequences() * els
                // compute outputs (LSTM cell)
                thrust::transform(
                    thrust::counting_iterator<int>(n*timestep),
                    thrust::counting_iterator<int>(n*timestep) + n,
                    thrust::make_zip_iterator(
		       thrust::make_tuple(
			  thrust::constant_iterator<bool>(!timestep), 
			  thrust::constant_iterator<bool>(timestep >= this->curMinSeqLength())
		       )
		    ),
		    m_fw.tmpOutputs.begin() + n*timestep, 
		    fn);
		
		
		
		// get CharW parameter alpha, beta, keppa
		{{
		    m_fw.timestepMatrices[timestep].tmpCharPara.assignProduct(
                                m_fw.weightMatrices.lstm2CharW, true, 
				m_fw.timestepMatrices[timestep].tmpOutputs, false);
		    
		    internal::CharWParaTrans fn2;  
		    if (timestep == 0){
			fn2.paraBuffLast  = NULL;
			fn2.boundary      = true;
		    }else{
			fn2.paraBuffLast  = m_fw.timestepMatrices[timestep-1].tmpCharParaPtr;
			fn2.boundary      = false;
		    }
		    fn2.paraBuffCurr      = m_fw.timestepMatrices[timestep].tmpCharParaPtr;
		    fn2.biasWeight        = _rawLSTM2CharWBias;
		    thrust::for_each(thrust::counting_iterator<int>(0),
				     thrust::counting_iterator<int>(0) + m_mixNum*CPNUM,
				     fn2);
		}}
				
		{{
		    // get rho [k, u] for timestep t
		    internal::CharWPara2Rho fn2;
		    fn2.rhoPtr  = m_fw.timestepMatrices[timestep].tmpRho;
		    fn2.chaPara = m_fw.timestepMatrices[timestep].tmpCharParaPtr;
		    fn2.mixNum  = m_mixNum;
		    thrust::for_each(thrust::counting_iterator<int>(0),
				     thrust::counting_iterator<int>(0)+
				     m_mixNum*m_charW->m_chaCurLength,
				     fn2);
		    // get phi[u] for timestep t
		    helpers::Matrix<TDevice> rhoMat(
			&m_charW->m_rhoVec, m_mixNum, m_charW->m_chaCurLength,
			m_fw.timestepMatrices[timestep].rhoOffset);
		    helpers::Matrix<TDevice> one   (
			&m_charW->m_tmpOne, m_mixNum, 1, 0);
		    helpers::Matrix<TDevice> phiMat(
			&m_charW->m_phiVec,1, m_charW->m_chaCurLength,
			m_fw.timestepMatrices[timestep].rhoOffset/m_mixNum);
		    phiMat.assignProduct(one, true, rhoMat, false);
			
		    // extract output from ChaW \sum_u phi(u)Cu
		    helpers::Matrix<TDevice> strMat(&m_charW->m_strWord, m_chaDim, 
						    m_charW->m_chaCurLength, 0);
		    m_fw.timestepMatrices[timestep].tmpOutputsC.assignProduct(strMat,false, 
									      phiMat, true);
			

		    // -------- DEBUG ------- //
                    #ifdef DEBUG_LOCAL_LSTMCHARW
		    printf("\nTime %d\nKappa:", timestep);
		    int offset1  = m_charW->m_mixNum * m_charW->m_chaMaxLength *
			    (timestep * (m_isBidirectional?2:1));
		    int offset2  = m_charW->inSize() * (timestep * (m_isBidirectional?2:1));
		    cpu_real_vector t_rho = m_charW->m_rhoVec;
		    cpu_real_vector t_cha = m_charW->chaInput();
		    cpu_real_vector t_phi = m_charW->m_phiVec;
		    for (int t = 0; t < m_mixNum; t++){
			printf("%f ", t_cha[offset2 + t * CPNUM+2]);
		    }
		    printf("\nPhi\n");
		    for (int t = 0; t < m_charW->m_chaCurLength; t++){
			printf("%3.3f ", 
			       t_phi[m_fw.timestepMatrices[timestep].rhoOffset/m_mixNum + t]);
		    }
		    printf("\nRho\n");
		    for (int t = 0; t < m_mixNum * m_charW->m_chaCurLength; t++){
			int dataIdx = t;
			int mixIdx = (dataIdx % fn2.mixNum);
			int chaIdx = (dataIdx / fn2.mixNum);
			real_t alpha = t_cha[offset2 + mixIdx * CPNUM];
			real_t beta  = t_cha[offset2 + mixIdx * CPNUM+1];
			real_t kappa = t_cha[offset2 + mixIdx * CPNUM+2];
			real_t tmp = alpha * std::exp(-1*beta*(kappa-chaIdx)*(kappa-chaIdx));
			printf("%3.3f-%3.3f ", tmp, t_rho[offset1 + chaIdx * fn2.mixNum + mixIdx]);
			if (mixIdx == (fn2.mixNum-1))
			    printf("\n");
		    }
		    printf("\n"); 
                    #endif

		}}
		
		{{
			// -------- DEBUG ------- //
                        #ifdef DEBUG_LOCAL_LSTMCHARW
			internal::CharWParaFeatureExtract fn2;
			fn2.charMatrix   = helpers::getRawPointer(m_charW->m_strWord);
			fn2.rhoTmpput    = m_fw.timestepMatrices[timestep].tmpRho;
			fn2.featureOut   = m_fw.timestepMatrices[timestep].tmpOutputsCPtr;
			fn2.curStrLength = m_charW->m_chaCurLength;
			fn2.charDim      = m_charW->m_chaDim;
			fn2.mixNM        = m_charW->m_mixNum;
			/*thrust::for_each(
					 thrust::counting_iterator<int>(0),
					 thrust::counting_iterator<int>(0) + m_charW->m_chaDim,
					 fn2
					 );*/	
			cpu_real_vector t_charMatrix =  m_charW->m_strWord;
			int offset1  = m_charW->m_mixNum * m_charW->m_chaMaxLength *
			    (timestep * (m_isBidirectional?2:1));
			cpu_real_vector t_rho        =  m_charW->m_rhoVec;
			int offset2 =  m_charW->outSize() * timestep * (m_isBidirectional?2:1);
			cpu_real_vector t_result     = m_charW->chaOutput();
			for (int t = 0; t< m_charW->m_chaDim; t++)
			{
			    real_t tmp = 0.0;
			    for (int j = 0; j<fn2.curStrLength; j++)
				for (int i = 0; i<fn2.mixNM; i++){
				    tmp += t_charMatrix[t + j * fn2.charDim] * 
					t_rho[offset1 + i + j*fn2.mixNM];
				}
			    printf("%3.3f-%3.3f ", tmp, t_result[offset2+t]);
			}
			printf("\n");
                        #endif		    

		}}

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
                        m_bw.timestepMatrices[timestep].niActs.addProduct(
				m_bw.weightMatrices.niInternal, true, 
				m_bw.timestepMatrices[timestep+1].tmpOutputs, false);
                        m_bw.timestepMatrices[timestep].igActs.addProduct(
				m_bw.weightMatrices.igInternal, true, 
				m_bw.timestepMatrices[timestep+1].tmpOutputs, false);
                        m_bw.timestepMatrices[timestep].fgActs.addProduct(
                                m_bw.weightMatrices.fgInternal, true, 
				m_bw.timestepMatrices[timestep+1].tmpOutputs, false);
                        m_bw.timestepMatrices[timestep].ogActs.addProduct(
				m_bw.weightMatrices.ogInternal, true, 
				m_bw.timestepMatrices[timestep+1].tmpOutputs, false);
			
			m_bw.timestepMatrices[timestep].niActs.addProduct(
				m_bw.weightMatrices.niInternalC, true,
				m_bw.timestepMatrices[timestep+1].tmpOutputsC, false);
			m_bw.timestepMatrices[timestep].igActs.addProduct(
				m_bw.weightMatrices.igInternalC, true,
				m_bw.timestepMatrices[timestep+1].tmpOutputsC, false);
			m_bw.timestepMatrices[timestep].fgActs.addProduct(
				m_bw.weightMatrices.fgInternalC, true,
				m_bw.timestepMatrices[timestep+1].tmpOutputsC, false);
			m_bw.timestepMatrices[timestep].ogActs.addProduct(
				m_bw.weightMatrices.ogInternalC, true,
				m_bw.timestepMatrices[timestep+1].tmpOutputsC, false);
			
                    }

                    // compute outputs
                    thrust::transform(
                        thrust::counting_iterator<int>(n*timestep),
                        thrust::counting_iterator<int>(n*timestep) + n,
                        thrust::make_zip_iterator(thrust::make_tuple(
			    thrust::constant_iterator<bool>(timestep == this->curMaxSeqLength()-1), 
			    thrust::constant_iterator<bool>(timestep >= this->curMinSeqLength())
			)), 
			m_bw.tmpOutputs.begin() + n*timestep, 
			fn);
		    
		    
		    // get CharW parameter alpha, beta, keppa
		    {{
		        m_bw.timestepMatrices[timestep].tmpCharPara.assignProduct(
                                m_bw.weightMatrices.lstm2CharW, true, 
				m_bw.timestepMatrices[timestep].tmpOutputs, false);
			
			internal::CharWParaTrans fn2;  
			if (timestep == (this->curMaxSeqLength()-1)){
			    fn2.paraBuffLast  = NULL;
			    fn2.boundary      = true;
			}else{
			    fn2.paraBuffLast  = m_bw.timestepMatrices[timestep+1].tmpCharParaPtr;
			    fn2.boundary      = false;
			}
			fn2.paraBuffCurr      = m_bw.timestepMatrices[timestep].tmpCharParaPtr;
			fn2.biasWeight        = _rawLSTM2CharWBias + m_charW->inSize();
			thrust::for_each(thrust::counting_iterator<int>(0),
					 thrust::counting_iterator<int>(0) + m_mixNum*CPNUM,
					 fn2);
		     }}
		    
		    
		    {{
			// get rho [k, u] for timestep t
			internal::CharWPara2Rho fn2;
			fn2.rhoPtr  = m_bw.timestepMatrices[timestep].tmpRho;
			fn2.chaPara = m_bw.timestepMatrices[timestep].tmpCharParaPtr;
			fn2.mixNum  = m_mixNum;
			thrust::for_each(thrust::counting_iterator<int>(0),
					thrust::counting_iterator<int>(0)+
					 m_mixNum*m_charW->m_chaCurLength,
					fn2);

			// get phi[u] for timestep t
			helpers::Matrix<TDevice> rhoMat(
			    &m_charW->m_rhoVec, m_mixNum, m_charW->m_chaCurLength,
			    m_bw.timestepMatrices[timestep].rhoOffset);
			helpers::Matrix<TDevice> one   (
			    &m_charW->m_tmpOne, m_mixNum, 1, 0);
			helpers::Matrix<TDevice> phiMat(
			    &m_charW->m_phiVec,1, m_charW->m_chaCurLength,
			    m_bw.timestepMatrices[timestep].rhoOffset/m_mixNum);
			phiMat.assignProduct(one, true, rhoMat, false);
		    
			// -------- DEBUG ------- //
                        #ifdef DEBUG_LOCAL_LSTMCHARW
			printf("\nTime %d\n", timestep);
			int offset1  = m_charW->m_mixNum * m_charW->m_chaMaxLength *
			    (timestep * (m_isBidirectional?2:1) + 1);
			int offset2  = m_charW->inSize() * 
			    (timestep * (m_isBidirectional?2:1) + 1);
			cpu_real_vector t_rho = m_charW->m_rhoVec;
			cpu_real_vector t_cha = m_charW->chaInput();
			cpu_real_vector t_phi = m_charW->m_phiVec;
			printf("\nPhi\n");
			for (int t = 0; t < m_charW->m_chaCurLength; t++){
			    printf("%3.3f ", 
				   t_phi[m_bw.timestepMatrices[timestep].rhoOffset/m_mixNum+t]);
			}
			printf("\nRho\n");
			for (int t = 0; t < m_mixNum * m_charW->m_chaCurLength; t++){
			    int dataIdx = t;
			    int mixIdx = (dataIdx % fn2.mixNum);
			    int chaIdx = (dataIdx / fn2.mixNum);
			    real_t alpha = t_cha[offset2 + mixIdx * CPNUM];
			    real_t beta  = t_cha[offset2 + mixIdx * CPNUM+1];
			    real_t kappa = t_cha[offset2 + mixIdx * CPNUM+2];

			    real_t tmp = alpha * std::exp(-1*beta*(kappa-chaIdx)*(kappa-chaIdx));
			    
			    printf("%3.3f-%3.3f ", tmp, 
				   t_rho[offset1 + chaIdx * fn2.mixNum + mixIdx]);
			    if (mixIdx == (fn2.mixNum-1))
				printf("\n");
			    
			}
			printf("\n");
                        #endif

			// extract output from ChaW
			helpers::Matrix<TDevice> strMat(&m_charW->m_strWordRev, m_chaDim, 
							m_charW->m_chaCurLength, 0);
			m_bw.timestepMatrices[timestep].tmpOutputsC.assignProduct(strMat, false, 
										  phiMat, true);


		    }}
		    
		    // extract output from ChaW
		    {{

			// -------- DEBUG ------- //
                        #ifdef DEBUG_LOCAL_LSTMCHARW	
			internal::CharWParaFeatureExtractBack fn2;
			fn2.charMatrix   = helpers::getRawPointer(m_charW->m_strWord);
			fn2.rhoTmpput    = m_bw.timestepMatrices[timestep].tmpRho;
			fn2.featureOut   = m_bw.timestepMatrices[timestep].tmpOutputsCPtr;
			fn2.curStrLength = m_charW->m_chaCurLength;
			fn2.charDim      = m_charW->m_chaDim;
			fn2.mixNM        = m_charW->m_mixNum;

			/*thrust::for_each(
					 thrust::counting_iterator<int>(0),
					 thrust::counting_iterator<int>(0) + m_charW->m_chaDim,
					 fn2);*/
			
			cpu_real_vector t_charMatrix =  m_charW->m_strWord;
			int offset1  = m_charW->m_mixNum * m_charW->m_chaMaxLength *
			    (timestep * (m_isBidirectional?2:1) + 1);
			cpu_real_vector t_rho        =  m_charW->m_rhoVec;
			int offset2 = m_charW->outSize()*(timestep * (m_isBidirectional?2:1)+1);

			cpu_real_vector t_result     = m_charW->chaOutput();
			for (int t = 0; t< m_charW->m_chaDim; t++)
			{
			    real_t tmp = 0.0;
			    for (int j = 0; j<fn2.curStrLength; j++)
				for (int i = 0; i<fn2.mixNM; i++){
				    tmp += 
					t_charMatrix[t + (fn2.curStrLength-1-j) * fn2.charDim] * 
					t_rho[offset1 + i + j*fn2.mixNM];
				}
			    printf("%3.3f-%3.3f ", tmp, t_result[offset2+t]);
			}
			printf("\n");
                        #endif		    
		    }}

		    
                }
            }
        }}	

        // resort outputs
        if (m_isBidirectional) {
            internal::ResortOutputsFn fn;
	    fn.layerSize    = this->size();
            fn.layerlstmSize= this->lstmSize();
            fn.effLayerSize = this->lstmSize() / 2;
            fn.fwOutputs    = helpers::getRawPointer(m_fw.tmpOutputs);
            fn.bwOutputs    = helpers::getRawPointer(m_bw.tmpOutputs);
	    fn.charWOuts    = helpers::getRawPointer(m_charW->m_feaVec);
	    
	    //cpu_real_vector tmp1(m_charW->m_feaVec);
	    //cpu_real_vector tmp2(m_fw.tmpOutputsC);
	    
            thrust::transform(
                thrust::counting_iterator<int>(0),
                thrust::counting_iterator<int>(0) + 
		this->curMaxSeqLength() * this->parallelSequences() * this->size(),
                this->outputs().begin(),
                fn
                );
        }
        else {
            internal::ResortOutputsFn fn;
	    fn.layerSize    = this->size();
            fn.layerlstmSize= this->lstmSize();
            fn.effLayerSize = this->lstmSize();
            fn.fwOutputs    = helpers::getRawPointer(m_fw.tmpOutputs);
            fn.bwOutputs    = NULL;
	    fn.charWOuts    = helpers::getRawPointer(m_charW->m_feaVec);
            thrust::transform(
                thrust::counting_iterator<int>(0),
                thrust::counting_iterator<int>(0) + 
		this->curMaxSeqLength() * this->parallelSequences() * this->size(),
                this->outputs().begin(),
                fn
                );
        }
    }

    template <typename TDevice>
    void LstmLayerCharW<TDevice>::computeBackwardPass()
    {
        {{
	    // for unidirectional LSTM, we can write the output errors directly in the 
	    // layer output errors vector
            internal::ResortOutputErrorsFn fn;
	    fn.layerSize     = this->size();
	    fn.layerlstmSize = this->lstmSize();
	    fn.effLayerSize  = this->lstmSize() / (m_isBidirectional?2:1);
	    fn.fwOutputErrors = helpers::getRawPointer(m_fw.tmpOutputErrors);
	    fn.bwOutputErrors = (m_isBidirectional?
				 helpers::getRawPointer(m_bw.tmpOutputErrors) : NULL);
	    fn.charWoutErrors = helpers::getRawPointer(m_charW->chaOutputError());
	    
	    int n = this->curMaxSeqLength() * this->parallelSequences() * this->size();
	    thrust::for_each(
	       thrust::make_zip_iterator(thrust::make_tuple(this->outputErrors().begin(), 
							    thrust::counting_iterator<int>(0))),
	       thrust::make_zip_iterator(thrust::make_tuple(this->outputErrors().begin()+n, 
							    thrust::counting_iterator<int>(0)+n)),
	       fn);
	
	}}

        // calculate the block errors
        {{
            int els = this->lstmSize() / (m_isBidirectional ? 2 : 1);
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
		
		// -------- DEBUG ------- //
                //#ifdef DEBUG_LOCAL_LSTMCHARW	
		//printf("%d\n", timestep);
		//#endif
                if (timestep != this->curMaxSeqLength()-1) {
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

		    {{
			// propagate the errors through the CharW (back through CharW -> hidden)
			// step1. multiply the matrix, and accumulate to CharW->m_feaVecErrors
			// Note: CharW is attached after the LSTM layer, thus the graidents for 
			//       CharW also comes from niDeltas
			
			m_fw.timestepMatrices[timestep].tmpOutputErrorsC.addProduct(
			     m_fw.weightMatrices.niInternalC, false, 
			     m_fw.timestepMatrices[timestep+1].niDeltas, false);
			m_fw.timestepMatrices[timestep].tmpOutputErrorsC.addProduct(
			     m_fw.weightMatrices.igInternalC, false, 
			     m_fw.timestepMatrices[timestep+1].igDeltas, false);
                        m_fw.timestepMatrices[timestep].tmpOutputErrorsC.addProduct(
			     m_fw.weightMatrices.fgInternalC, false, 
			     m_fw.timestepMatrices[timestep+1].fgDeltas, false);
			m_fw.timestepMatrices[timestep].tmpOutputErrorsC.addProduct(
			     m_fw.weightMatrices.ogInternalC, false, 
			     m_fw.timestepMatrices[timestep+1].ogDeltas, false);
			
		    }}
		    
                }
		

		
		{{
		    // propagate the errors through the CharW (back through the CharW -> output)
		    // step1.compute errors in CharW from CharW->feaVecErrors to CharW->CharErrors
		    
		    bool boundary = (timestep == this->curMaxSeqLength()-1);
		    {{
			// prepare the error buf \tau_t_u = \sum_i^{chaDim} \delta_ti C_ui
			helpers::Matrix<TDevice> tauMat(&m_charW->m_tau_t, 
							m_charW->m_chaCurLength, 1, 0);
			helpers::Matrix<TDevice> strMat(&m_charW->m_strWord,
							m_chaDim, m_charW->m_chaCurLength, 0);
			tauMat.assignProduct(strMat, true, 
					     m_fw.timestepMatrices[timestep].tmpOutputErrorsC,
					     false);

			// ---- DEBUG ----- //
			#ifdef DEBUG_LOCAL_LSTMCHARW	
			cpu_real_vector tau1 = m_charW->m_tau_t;

			internal::PrepareErrorBuf fn3;
			fn3.gradients       = m_fw.timestepMatrices[timestep].tmpOutputErrorsCPtr;
			fn3.strWord         = helpers::getRawPointer(m_charW->m_strWord);
			fn3.tau_t           = helpers::getRawPointer(m_charW->m_tau_t);
			fn3.chaDim          = m_charW->m_chaDim;
			thrust::for_each(thrust::counting_iterator<int>(0),
					 thrust::counting_iterator<int>(0)+m_charW->m_chaCurLength,
					 fn3);
			cpu_real_vector tau2 = m_charW->m_tau_t;
			printf("\n Tau_t");
			for (int i = 0; i<m_charW->m_chaCurLength; i++)
			    printf("%3.3f-%3.3f ", tau2[i], tau2[i]);
			printf("\n");
			#endif
		    
			// prepare the Rho Error Buffer
			internal::PrepareRhoErrorBuf fn4;
			fn4.rhoPtr          = m_fw.timestepMatrices[timestep].tmpRho;
			fn4.buffPtr         = helpers::getRawPointer(m_charW->m_rhoErrorBuf);
			fn4.chaWPara        = m_fw.timestepMatrices[timestep].tmpCharParaPtr;
			fn4.mixNum          = m_mixNum;
			fn4.paraNum         = CPNUM;
			thrust::for_each(thrust::counting_iterator<int>(0),
					 thrust::counting_iterator<int>(0) + 
					 CPNUM * m_mixNum * m_charW->m_chaCurLength,
					 fn4);
			
			helpers::Matrix<TDevice> errBuf(&m_charW->m_rhoErrorBuf, 
							m_mixNum * CPNUM, m_charW->m_chaCurLength, 
							0);
			
			// get the gradients
			m_fw.timestepMatrices[timestep].tmpCharParaError.assignProduct(
	 			       errBuf, false, tauMat, false);
			
			// adjust the kappa gradient
			internal::KappaGradient fn5;
			fn5.chaParaPtr      = m_fw.timestepMatrices[timestep].tmpCharParaPtr;
			fn5.chaParaPtrNext  = boundary ? 
			    (NULL):(m_fw.timestepMatrices[timestep+1].tmpCharParaPtr);
			fn5.chaParaEPtr     = m_fw.timestepMatrices[timestep].tmpCharParaErrorPtr;
			fn5.chaParaEPtrNext = boundary ? 
			    (NULL):(m_fw.timestepMatrices[timestep+1].tmpCharParaErrorPtr);
			fn5.boundaryE        = boundary;
			fn5.boundaryB        = timestep == 0;
			fn5.mixNum           = m_mixNum;
			thrust::for_each(thrust::counting_iterator<int>(0),
					 thrust::counting_iterator<int>(0) + m_mixNum,
					 fn5);

			
			// --------- DEBUG ----------- //
                        #ifdef DEBUG_LOCAL_LSTMCHARW			
			// back-propagation through the CharW
		        internal::CharWParaError fn2;
			fn2.rhoPtr          = m_fw.timestepMatrices[timestep].tmpRho;
			fn2.tau_t           = helpers::getRawPointer(m_charW->m_tau_t);
			
			fn2.chaParaPtr      = m_fw.timestepMatrices[timestep].tmpCharParaPtr;
			fn2.chaParaPtrNext  = boundary ? 
			    (NULL):(m_fw.timestepMatrices[timestep+1].tmpCharParaPtr);
			fn2.chaParaEPtr     = m_fw.timestepMatrices[timestep].tmpCharParaErrorPtr;
			fn2.chaParaEPtrNext = boundary ? 
			    (NULL):(m_fw.timestepMatrices[timestep+1].tmpCharParaErrorPtr);
			
			fn2.boundaryE        = boundary;
			fn2.boundaryB        = timestep == 0;
			fn2.curStrLength     = m_charW->m_chaCurLength;
			fn2.mixNum           = m_mixNum;
			//thrust::for_each(thrust::counting_iterator<int>(0),
			//thrust::counting_iterator<int>(0) + m_mixNum,
			//fn2);
			

			int biasRho   =  m_fw.timestepMatrices[timestep].tmpRho- 
			    m_fw.timestepMatrices[0].tmpRho;
			cpu_real_vector tmpRho = m_charW->m_rhoVec;
			int biasPara  = m_fw.timestepMatrices[timestep].tmpCharParaPtr -
			    m_fw.timestepMatrices[0].tmpCharParaPtr;
			cpu_real_vector tmpPara = m_charW->m_chaPara;
			cpu_real_vector tmpParaE= m_charW->m_chaParaError;
			cpu_real_vector tmpTau = m_charW->m_tau_t;
			int biasPara2  = boundary?(0):
			    m_fw.timestepMatrices[timestep+1].tmpCharParaPtr -
			    m_fw.timestepMatrices[0].tmpCharParaPtr;
			
			for (int mixIdx = 0 ; mixIdx < m_mixNum; mixIdx ++){
			    real_t alphaEPtr = tmpParaE[biasPara + mixIdx*CPNUM];
			    real_t betaEPtr  = tmpParaE[biasPara + mixIdx*CPNUM+1];
			    real_t kappaEPtr = tmpParaE[biasPara + mixIdx*CPNUM+2];
			    real_t kappaEPPtr= tmpParaE[biasPara2 + mixIdx*CPNUM+2];
			    real_t betaPtr  = tmpPara[biasPara + mixIdx*CPNUM+1];
			    real_t kappaPtr = tmpPara[biasPara + mixIdx*CPNUM+2];
	    
			    alphaEPtr = 0;
			    betaEPtr  = 0;
			    kappaEPtr = (boundary ? 0 : (kappaEPPtr));
	    
	    
			    for (int i = 0; i< fn2.curStrLength; i++){
				(alphaEPtr) +=  (tmpTau[i])    * (tmpRho[i*fn2.mixNum + mixIdx]);
				(betaEPtr)  -=  (tmpTau[i])    * (tmpRho[i*fn2.mixNum + mixIdx]) * 
				    ((kappaPtr)-i) * ((kappaPtr)-i) * (betaPtr);
				
				(kappaEPtr) -= 2*(tmpTau[i]) * (tmpRho[i*fn2.mixNum + mixIdx]) * 
				    ((kappaPtr)-i) * (betaPtr);
			    }
			    
			}
                        #endif
			
		    }}
		    
		    // step2. add CharErrors to the tmpOutputErrors of LSTM  (pass through the W)
		    m_fw.timestepMatrices[timestep].tmpOutputErrors.addProduct(
			m_fw.weightMatrices.lstm2CharW, false, 
			m_fw.timestepMatrices[timestep].tmpCharParaError, false);
		    

		}} 
		
		
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
			   m_fw.tmpOutputErrors.begin() + n*timestep+n, 
			   thrust::counting_iterator<int>(n*timestep)+n, 
			   thrust::constant_iterator<bool>(timestep == this->curMaxSeqLength()-1)+n,
			   thrust::constant_iterator<bool>(!timestep)+n, 
			   thrust::constant_iterator<bool>(timestep >= this->curMinSeqLength())+n)),
                    fn
                    );
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
			{{
			// propagate the errors through the CharW (back through CharW -> hidden)
			// step1. multiply the matrix, and accumulate to CharW->m_feaVecErrors
				
			    m_bw.timestepMatrices[timestep].tmpOutputErrorsC.addProduct(
				m_bw.weightMatrices.niInternalC, false, 
				m_bw.timestepMatrices[timestep-1].niDeltas, false);
			    m_bw.timestepMatrices[timestep].tmpOutputErrorsC.addProduct(
				m_bw.weightMatrices.igInternalC, false, 
				m_bw.timestepMatrices[timestep-1].igDeltas, false);
			    m_bw.timestepMatrices[timestep].tmpOutputErrorsC.addProduct(
				m_bw.weightMatrices.fgInternalC, false, 
				m_bw.timestepMatrices[timestep-1].fgDeltas, false);
			    m_bw.timestepMatrices[timestep].tmpOutputErrorsC.addProduct(
				m_bw.weightMatrices.ogInternalC, false, 
				m_bw.timestepMatrices[timestep-1].ogDeltas, false);
				
			}}
                    }
		    
		    {{
			// propagate the errors through the CharW (back through the CharW -> output)
			// step1. compute errors in CharW (from CharW->feaVecErrors to 
			// CharW->CharErrors)
			bool boundary = (timestep == 0);
			{{
				
			   // prepare the error buf \tau_t_u = \sum_i^{chaDim} \delta_ti C_ui
			   helpers::Matrix<TDevice> tauMat(&m_charW->m_tau_t, 
							   m_charW->m_chaCurLength, 1, 0);
			   helpers::Matrix<TDevice> strMat(&m_charW->m_strWordRev,
							   m_chaDim, m_charW->m_chaCurLength, 0);
			   tauMat.assignProduct(strMat, true, 
						m_bw.timestepMatrices[timestep].tmpOutputErrorsC,
						false);
			   
			   // prepare the Rho Error Buffer
			   internal::PrepareRhoErrorBuf fn4;
			   fn4.rhoPtr          = m_bw.timestepMatrices[timestep].tmpRho;
			   fn4.buffPtr         = helpers::getRawPointer(m_charW->m_rhoErrorBuf);
			   fn4.chaWPara        = m_bw.timestepMatrices[timestep].tmpCharParaPtr;
			   fn4.mixNum          = m_mixNum;
			   fn4.paraNum         = CPNUM;
			   thrust::for_each(thrust::counting_iterator<int>(0),
					    thrust::counting_iterator<int>(0) + 
					    CPNUM * m_mixNum * m_charW->m_chaCurLength,
					    fn4);
			   
			   helpers::Matrix<TDevice> errBuf(&m_charW->m_rhoErrorBuf, 
							   m_mixNum*CPNUM, 
							   m_charW->m_chaCurLength, 
							   0);
			
			   // get the gradients
			   m_bw.timestepMatrices[timestep].tmpCharParaError.assignProduct(
					  errBuf, false, tauMat, false);
			
			   // adjust the kappa gradient
			   internal::KappaGradient fn5;
			   fn5.chaParaPtr     = m_bw.timestepMatrices[timestep].tmpCharParaPtr;
			   fn5.chaParaPtrNext = boundary ? 
			       (NULL):(m_bw.timestepMatrices[timestep-1].tmpCharParaPtr);
			   fn5.chaParaEPtr    =m_bw.timestepMatrices[timestep].tmpCharParaErrorPtr;
			   fn5.chaParaEPtrNext= boundary ? 
			       (NULL):(m_bw.timestepMatrices[timestep-1].tmpCharParaErrorPtr);
			   fn5.boundaryE      = boundary;
			   fn5.boundaryB      = (timestep == this->curMaxSeqLength()-1);
			   fn5.mixNum         = m_mixNum;
			   thrust::for_each(thrust::counting_iterator<int>(0),
					    thrust::counting_iterator<int>(0) + m_mixNum,
					    fn5);


			    /*
			    // prepare the error buf
			    internal::PrepareErrorBufBack fn3;
			    fn3.gradients  = m_bw.timestepMatrices[timestep].tmpOutputErrorsCPtr;
			    fn3.strWord    = helpers::getRawPointer(m_charW->m_strWord);
			    fn3.tau_t      = helpers::getRawPointer(m_charW->m_tau_t);
			    fn3.chaDim     = m_charW->m_chaDim;
			    fn3.chaCurLength= m_charW->m_chaCurLength;
			    thrust::for_each(
				    thrust::counting_iterator<int>(0),
				    thrust::counting_iterator<int>(0)+m_charW->m_chaCurLength,
				    fn3);
			    
			    // back-propagation through the CharW
			    internal::CharWParaError fn2;
			    fn2.rhoPtr        = m_bw.timestepMatrices[timestep].tmpRho;
			    fn2.tau_t         = helpers::getRawPointer(m_charW->m_tau_t);
			
			    fn2.chaParaPtr    = m_bw.timestepMatrices[timestep].tmpCharParaPtr;
			    fn2.chaParaPtrNext= boundary ? 
				(NULL):(m_bw.timestepMatrices[timestep-1].tmpCharParaPtr);
			    fn2.chaParaEPtr   = m_bw.timestepMatrices[timestep].tmpCharParaErrorPtr;
			    fn2.chaParaEPtrNext = boundary ? 
				(NULL):(m_bw.timestepMatrices[timestep-1].tmpCharParaErrorPtr);
			
			    fn2.boundaryE      = boundary;
			    fn2.boundaryB      = (timestep == this->curMaxSeqLength()-1);
			    fn2.curStrLength   = m_charW->m_chaCurLength;
			    fn2.mixNum         = m_mixNum;
			    thrust::for_each(thrust::counting_iterator<int>(0),
					     thrust::counting_iterator<int>(0) + m_mixNum,
					     fn2);
			    */
			}}
			
			// step2. add CharErrors to the tmpOutputErrors of LSTM  
			// (pass through the W)	
			m_bw.timestepMatrices[timestep].tmpOutputErrors.addProduct(
				m_bw.weightMatrices.lstm2CharW, false, 
				m_bw.timestepMatrices[timestep].tmpCharParaError, false);
			
		    }}
		    
		    
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
			   m_bw.tmpOutputErrors.begin() + n*timestep+n, 
			   thrust::counting_iterator<int>(n*timestep)+n, 
			   thrust::constant_iterator<bool>(!timestep)+n, 
			   thrust::constant_iterator<bool>(timestep == this->curMaxSeqLength()-1)+n,
			   thrust::constant_iterator<bool>(timestep >= this->curMinSeqLength())+n)),
		      fn);
		    
                }
            }
        }}

        // back-propagate the error to the preceding layer
        {{
            TrainableLayer<TDevice> *pl = dynamic_cast<TrainableLayer<TDevice>*>(&this->precedingLayer());
            if (pl) {
                helpers::Matrix<TDevice> plErrorsMatrix(&pl->outputErrors(), pl->size(), this->curMaxSeqLength() * this->parallelSequences());

                // forward states
                plErrorsMatrix.assignProduct(m_fw.weightMatrices.niInput, false, m_fw.niDeltasMatrix, false);
                plErrorsMatrix.addProduct   (m_fw.weightMatrices.igInput, false, m_fw.igDeltasMatrix, false);
                plErrorsMatrix.addProduct   (m_fw.weightMatrices.fgInput, false, m_fw.fgDeltasMatrix, false);
                plErrorsMatrix.addProduct   (m_fw.weightMatrices.ogInput, false, m_fw.ogDeltasMatrix, false);

                // backward states
                if (m_isBidirectional) {
                    plErrorsMatrix.addProduct(m_bw.weightMatrices.niInput, false, m_bw.niDeltasMatrix, false);
                    plErrorsMatrix.addProduct(m_bw.weightMatrices.igInput, false, m_bw.igDeltasMatrix, false);
                    plErrorsMatrix.addProduct(m_bw.weightMatrices.fgInput, false, m_bw.fgDeltasMatrix, false);
                    plErrorsMatrix.addProduct(m_bw.weightMatrices.ogInput, false, m_bw.ogDeltasMatrix, false);
                }
            }
        }}

        // compute the weight updates
        {{
            internal::ComputeWeightUpdateFn fn;
            fn.layerSize             = this->lstmSize();
            fn.effLayerSize          = this->lstmSize() / (m_isBidirectional ? 2 : 1);
            fn.precLayerSize         = this->precedingLayer().size();
            fn.timestepDistance      = this->parallelSequences() * this->lstmSize() / (m_isBidirectional ? 2 : 1);
            fn.parallelSequences     = this->parallelSequences();
            fn.patternsCount         = this->curMaxSeqLength() * this->parallelSequences();
            fn.biasWeightsOffset     = this->lstmSize() * this->precedingLayer().size() * 4;
            fn.internalWeightsOffset = fn.biasWeightsOffset + this->lstmSize() * 4;
            fn.peepholeWeightsOffset = fn.internalWeightsOffset + this->lstmSize() * fn.effLayerSize * 4;
            fn.bias                  = this->bias();
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

            thrust::transform(
                thrust::counting_iterator<int>(0),
                thrust::counting_iterator<int>(0) + (int)m_lstmWeightNum,
                this->_weightUpdates().begin(),
                fn
                );
        }}
	
	
	{{
	     
	    internal::ComputeWeightUpdateForCharW1 fn;
            fn.fwNiDeltas            = helpers::getRawPointer(m_fw.niDeltas);
            fn.bwNiDeltas            = helpers::getRawPointer(m_bw.niDeltas);
            fn.fwIgDeltas            = helpers::getRawPointer(m_fw.igDeltas);
            fn.bwIgDeltas            = helpers::getRawPointer(m_bw.igDeltas);
            fn.fwFgDeltas            = helpers::getRawPointer(m_fw.fgDeltas);
            fn.bwFgDeltas            = helpers::getRawPointer(m_bw.fgDeltas);
            fn.fwOgDeltas            = helpers::getRawPointer(m_fw.ogDeltas);
            fn.bwOgDeltas            = helpers::getRawPointer(m_bw.ogDeltas);
	    fn.charWOutput           = helpers::getRawPointer(m_charW->chaOutput());
	    fn.blstm                 = m_isBidirectional;
	    fn.patternsCount         = this->curMaxSeqLength() * this->parallelSequences();
	    fn.chaOutSize            = m_charW->outSize();
	    fn.lstmSize              = this->lstmSize();

	    // update weights of the four gates
	     thrust::transform(
		thrust::counting_iterator<int>(0),
		thrust::counting_iterator<int>(0) + (int) 4*(m_charW->outSize()*this->lstmSize()),
		this->_weightUpdates().begin()+m_lstmWeightNum+m_charW->inSize()*this->lstmSize(),
		fn);
	    {{
    	    // update weights of the lstm->charW	    
	    internal::ComputeWeightUpdateForCharW2 fn;
	    fn.charWinputError       = helpers::getRawPointer(m_charW->chaInputError());
	    fn.lstmOutputFw          = helpers::getRawPointer(m_fw.tmpOutputs);
	    fn.lstmOutputBw          = helpers::getRawPointer(m_bw.tmpOutputs);
	    fn.patternsCount         = this->curMaxSeqLength() * this->parallelSequences();
	    fn.blstm                 = m_isBidirectional;
	    fn.lstmSize              = this->lstmSize();
	    fn.chaInSize             = m_charW->inSize();
	    
	    
	    thrust::transform(
		thrust::counting_iterator<int>(0),
		thrust::counting_iterator<int>(0) + (int) (m_charW->inSize()*this->lstmSize()),
		this->_weightUpdates().begin()+m_lstmWeightNum,
		fn);
	    }}
	    {{
	    // update bias of the lstm->charW	    
	    internal::ComputeWeightUpdateForCharW3 fn;
	    fn.charWinputError       = helpers::getRawPointer(m_charW->chaInputError());
	    fn.patternsCount         = this->curMaxSeqLength() * this->parallelSequences();
	    fn.chaInSize             = m_charW->inSize();
	    
	    int tmpOffset = (m_charW->inSize() + 4 * m_charW->outSize() ) * this->lstmSize();
	    thrust::transform(
		thrust::counting_iterator<int>(0),
		thrust::counting_iterator<int>(0) + (int) (m_charW->inSize()),
		this->_weightUpdates().begin() + m_lstmWeightNum + tmpOffset,
		fn);
	    
	    }}
	}}
	
        // re-swap the output errors and the tmp output errors of the forward pass
	// ???
	
        /*if (!m_isBidirectional) {
            this->outputErrorsLstm().swap(m_fw.tmpOutputErrors);
            this->outputsLstm()    .swap(m_fw.tmpOutputs);
	    }*/
	
    }

    template <typename TDevice>
    typename LstmLayerCharW<TDevice>::real_vector& LstmLayerCharW<TDevice>::outputs()
    {
        return m_outputsAll;
    }

    template <typename TDevice>
    typename LstmLayerCharW<TDevice>::real_vector& LstmLayerCharW<TDevice>::outputErrors()
    {
        return m_outErrorsAll;
    }

    template <typename TDevice>
    typename LstmLayerCharW<TDevice>::real_vector& LstmLayerCharW<TDevice>::outputsLstm()
    {
        return TrainableLayer<TDevice>::outputs();
    }

    template <typename TDevice>
    typename LstmLayerCharW<TDevice>::real_vector& LstmLayerCharW<TDevice>::outputErrorsLstm()
    {
        return TrainableLayer<TDevice>::outputErrors();
    }
    
    /* *
     * Change to export the configuration of LstmCharW
     */
    template <typename TDevice>
    void LstmLayerCharW<TDevice>::exportLayer(const helpers::JsonValue &layersArray, 
					      const helpers::JsonAllocator &allocator) const
    {
        if (!layersArray->IsArray())
            throw std::runtime_error("The JSON value is not an array");

        // create and fill the layer object
        rapidjson::Value layerObject(rapidjson::kObjectType);
        layerObject.AddMember("name",      this->name().c_str(), allocator);
        layerObject.AddMember("type",      type().c_str(), allocator);
        layerObject.AddMember("size",      lstmSize(),     allocator); // lstmSize(), not size()
	layerObject.AddMember("bias",      this->bias(),         allocator);
	layerObject.AddMember("mixNum",    m_mixNum,       allocator); // add mixNum
	layerObject.AddMember("txtChaDim", m_chaDim,       allocator); // add txtChaDim
        // add the layer object to the layers array
        layersArray->PushBack(layerObject, allocator);
    }
    
    /**
     * Change to export the weight of LstmCharW
     * Note: the layout of the weight array should be transparent to TrainableLayer,
             The internal part contains all the weights in LstmCharW. 
     */
    template <typename TDevice>
    void LstmLayerCharW<TDevice>::exportWeights(const helpers::JsonValue &weightsObject, 
						const helpers::JsonAllocator &allocator) const
    {
	// 1. calculate the interal weight of LstmCharW
	int internalLstm      = (m_isBidirectional?2:4) * this->lstmSize() + 3;
	int internalLstmCharW = CPNUM * m_mixNum +             // LSTM -> CharW 
				(m_isBidirectional?2:1) +       
				m_chaDim * 4;                  // CharW-> LSTM (to gates)
	int inputWeightsPerBlock = 4;                          //
	
        if (!weightsObject->IsObject())
            throw std::runtime_error("The JSON value is not an object");

        // do nothing if we don't have any weights
        if (this->weights().empty())
            return;

        // create and fill the weight arrays
        rapidjson::Value inputWeightsArray(rapidjson::kArrayType);
        int inputWeightsCount=this->lstmSize()*inputWeightsPerBlock*this->precedingLayer().size();
        inputWeightsArray.Reserve(inputWeightsCount, allocator);
        for (int i = 0; i < inputWeightsCount; ++i)
            inputWeightsArray.PushBack(this->weights()[i], allocator);

        rapidjson::Value biasWeightsArray(rapidjson::kArrayType);
        int bWeightsCount = this->lstmSize() * inputWeightsPerBlock;
        biasWeightsArray.Reserve(bWeightsCount, allocator);
        for (int i = 0; i < bWeightsCount; ++i)
            biasWeightsArray.PushBack(this->weights()[inputWeightsCount + i], allocator);

        rapidjson::Value inWeightsArray(rapidjson::kArrayType);
        int inWeightsCount = this->lstmSize() * (internalLstm + internalLstmCharW);
        inWeightsArray.Reserve(inWeightsCount, allocator);
        for (int i = 0; i < inWeightsCount; ++i)
            inWeightsArray.PushBack(this->weights()[inputWeightsCount+bWeightsCount+i], allocator);

        // create and fill the weights subsection
        rapidjson::Value weightsSection(rapidjson::kObjectType);
        weightsSection.AddMember("input",    inputWeightsArray,    allocator);
        weightsSection.AddMember("bias",     biasWeightsArray,     allocator);
        weightsSection.AddMember("internal", inWeightsArray,       allocator);

        // add the weights section tot he weights object
        weightsObject->AddMember(this->name().c_str(), weightsSection, allocator);
    }
    
    
    
    // explicit template instantiations
    template class LstmLayerCharW<Cpu>;
    template class LstmLayerCharW<Gpu>;
    template class CharW<Cpu>;
    template class CharW<Gpu>;

} // namespace layers
