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


#include "StrucTransformLayer.hpp"
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
#include <boost/foreach.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/algorithm/string.hpp>
#include <vector>


#define STRUCTTRANS_AX_B      1        // Ax + B
#define STRUCTTRANS_AX_B_WITH_MU_STD 3 // Ax + B with mean and std into consideration
#define STRUCTTRANS_AX_B_WITH_MU_STD_SHIFT 4 // type 3, with 1 time-step shifted
#define STRUCTTRANS_DEFAULT   1

#define STRUCTTRANS_AX_B_WITH_MU_STD_STD_FLOOR 0.0000001

#define STRUCTTRANS_DIREC_NOTREVERSE 0
#define STRUCTTRANS_DIREC_REVERSE 1

namespace internal{
namespace{

    // Conduct transformation a*x+b
    struct trans_ax_b
    {
	real_t *abData;	  // data buffer to store a*x + b 
	int     dataDim;  // dimension of the data

	int     abShiftT; // shift to use a and b (default to 0). Prepared for AR model
	
	const char *patTypes; //

	__host__ __device__ void operator() (const thrust::tuple<real_t&, int> &t) const
	{

	    // index of the current data cell
	    int outputIdx = t.get<1>();
	    // frame index
	    int timeIdx   = outputIdx / dataDim;
	    // dimension index
	    int dimIdx    = outputIdx % dataDim;

	    // skip dummy frame (for parallel sentence processing)
	    if (patTypes != NULL && patTypes[timeIdx] == PATTYPE_NONE)
		return;
	    
	    // In each frame of abData, [B_1st_dim, B_2nd_dim ..., A_1st_dim, A_2nd_dim ...]
	    // (timeIdx-abShiftT) * 2 * dataDim -> points to the frame
	    //                        + dataDim -> points to the memory address of A
	    //                        + dataIdx -> points to the memory address of A[dimIdx]
	    
	    // (timeIdx-abShiftT) * 2 * dataDim -> points to the memory address of B
	    //                        + dataIdx -> points to the memory address of B[dimIdx]
	    
	    t.get<0>() = (t.get<0>() * abData[(timeIdx-abShiftT) * 2 * dataDim + dataDim + dimIdx] +
			  abData[(timeIdx-abShiftT) * 2 * dataDim + dimIdx]);

	}
    };


    struct trans_ax_b_grad
    {
	real_t *abData;     // data buffer for [A, B]
	real_t *abError;    // gradient buffer for [A, B]
	real_t *xData;      // data buffer for x
	real_t *xError;     // gradient buffer for x
	
	bool abFromSkipLayer; // if [A,B] are sent from a skip layer
	bool xFromSkipLayer;  // if x is given by a skip layer
	
	int dataDim;	      // data dimension
	const char *patTypes;

	__host__ __device__ void operator() (const thrust::tuple<real_t&, int> &t) const
	{
	    // t.get<0>(): outputErrors
	    
	    int outputIdx = t.get<1>();
	    int timeIdx   = outputIdx / dataDim;
	    int dimIdx    = outputIdx % dataDim;

	    // skip dummy frame (for parallel sentence processing)
	    if (patTypes != NULL && patTypes[timeIdx] == PATTYPE_NONE){
		// if previous layers are skiplayers,
		//  no need to do anything because gradients buffers have been initialized with 0
		// if previous layers are not skiplayers,
		//  just set the gradients to 0
		if (!abFromSkipLayer){
		    abError[timeIdx * dataDim * 2 + dimIdx] = 0;
		    abError[timeIdx * dataDim * 2 + dimIdx + dataDim] = 0;
		}
		if (!xFromSkipLayer)
		    xError[outputIdx] = 0;
	    }else{

		// \p E/ \p a = \p E / \p y * x
		// \p E/ \p b = \p E / \p y 
		if (abFromSkipLayer){
		    // if [A, B] are the output of a skip layer, accumulate the gradients
		    abError[timeIdx*dataDim*2+dimIdx+dataDim] += t.get<0>() * xData[outputIdx];
		    abError[timeIdx*dataDim*2+dimIdx] += t.get<0>();
		}else{
		    // if [A, B] are from a conventional layer, update the gradients
		    abError[timeIdx*dataDim*2+dimIdx+dataDim] = t.get<0>() * xData[outputIdx];
		    abError[timeIdx*dataDim*2+dimIdx] = t.get<0>();
		}

		// \p E/ \p x = \p E / \p y * a
		if (xFromSkipLayer){
		    xError[outputIdx] += t.get<0>() * abData[timeIdx*dataDim*2 + dimIdx +dataDim];
		}else{
		    xError[outputIdx] = t.get<0>() * abData[timeIdx*dataDim*2 + dimIdx +dataDim];
		}
	    }
	}
    };



    struct trans_ax_b_with_mean_std
    {
	real_t *abData;
	real_t *preOut;
	
	int dataDim;
	int preDim;
	
	const char *patTypes;

	__host__ __device__ void operator() (const thrust::tuple<real_t&, int> &t) const
	{
	    
	    int outputIdx = t.get<1>();
	    int timeIdx   = outputIdx / dataDim;
	    int dimIdx    = outputIdx % dataDim;
	    int featDim   = dataDim / 3;
	    
	    // skip dummy frame (for parallel sentence processing)
	    if (patTypes != NULL && patTypes[timeIdx] == PATTYPE_NONE)
		return;

	    /*
	    if (preDim == dataDim){
		// Previous transform layer has mean and std
		if (dimIdx < featDim){
		    // output dimension
		    t.get<0>() = preOut[t.get<1>()] * abData[timeIdx*2*featDim+featDim+dimIdx] +
			abData[timeIdx * 2 * featDim + dimIdx];
		}else if (dimIdx < featDim * 2){
		    // mean part
		    dimIdx = dimIdx - featDim;
		    t.get<0>() = preOut[t.get<1>()] * abData[timeIdx*2*featDim+featDim+dimIdx] +
			abData[timeIdx * 2 * featDim + dimIdx];
		}else{
		    // std part 
		    dimIdx = dimIdx - featDim * 2;
		    t.get<0>() = preOut[t.get<1>()] +
			0.5 * log(abData[timeIdx*2*featDim+featDim+dimIdx] *
				  abData[timeIdx*2*featDim+featDim+dimIdx] +
				  STRUCTTRANS_AX_B_WITH_MU_STD_STD_FLOOR);
		}
	    }else{
		// Previous transform layer has no mean and std
		if (dimIdx < featDim){
		    // output dimension
		    t.get<0>() = (preOut[timeIdx * preDim + dimIdx] *
				  abData[timeIdx * 2 * featDim + featDim + dimIdx] +
				  abData[timeIdx * 2 * featDim + dimIdx]);
		}else if (dimIdx < featDim * 2){
		    // mean part set to 0 * b + a
		    dimIdx = dimIdx - featDim;
		    t.get<0>() = abData[timeIdx * 2 * featDim + dimIdx];
		}else{
		    // std part 1 * 0.5 * log(b^2) (exp(\sigma) = exp(\sigma)*|b|)
		    dimIdx = dimIdx - featDim * 2;
		    t.get<0>() = 0.5 * log(abData[timeIdx * 2 * featDim + featDim + dimIdx]*
					   abData[timeIdx * 2 * featDim + featDim + dimIdx] +
					   STRUCTTRANS_AX_B_WITH_MU_STD_STD_FLOOR);
		}
		}*/

	    
	    if (preDim == dataDim){
		// Previous transform layer has mean and std
		if (dimIdx < featDim){
		    // output dimension
		    t.get<0>() = preOut[t.get<1>()] * exp(abData[timeIdx*2*featDim+featDim+dimIdx])+
			abData[timeIdx * 2 * featDim + dimIdx];
		}else if (dimIdx < featDim * 2){
		    // mean part
		    dimIdx = dimIdx - featDim;
		    t.get<0>() = preOut[t.get<1>()] * exp(abData[timeIdx*2*featDim+featDim+dimIdx])+
			abData[timeIdx * 2 * featDim + dimIdx];
		}else{
		    // std part 
		    dimIdx = dimIdx - featDim * 2;
		    t.get<0>() = preOut[t.get<1>()] + abData[timeIdx*2*featDim+featDim+dimIdx];
		}
		
	    }else{
		// Previous transform layer has no mean and std
		if (dimIdx < featDim){
		    // output dimension
		    t.get<0>() = (preOut[timeIdx * preDim + dimIdx] *
				  exp(abData[timeIdx * 2 * featDim + featDim + dimIdx]) +
				  abData[timeIdx * 2 * featDim + dimIdx]);
		}else if (dimIdx < featDim * 2){
		    // mean part set to 0 * b + a
		    dimIdx = dimIdx - featDim;
		    t.get<0>() = abData[timeIdx * 2 * featDim + dimIdx];
		}else{
		    
		    dimIdx = dimIdx - featDim * 2;
		    t.get<0>() = abData[timeIdx * 2 * featDim + featDim + dimIdx];
		}
	    }
	    
	    

	}
    };

    struct trans_ax_b_with_mean_std_shifted
    {
	real_t *abData;
	real_t *preOut;
	
	int dataDim;
	int preDim;
	int parallel;
	
	const char *patTypes;

	__host__ __device__ void operator() (const thrust::tuple<real_t&, int> &t) const
	{
	    
	    int outputIdx = t.get<1>();
	    int timeIdx   = outputIdx / dataDim;
	    int dimIdx    = outputIdx % dataDim;
	    int featDim   = dataDim / 3;
	    
	    // skip dummy frame (for parallel sentence processing)
	    if (patTypes != NULL && patTypes[timeIdx] == PATTYPE_NONE)
		return;
	    
	    if (timeIdx < parallel){
		// the first time block

		if (preDim == dataDim){
		    t.get<0>() = preOut[t.get<1>()];
		}else{
		    
		    if (dimIdx < featDim){
			// output dimension
			t.get<0>() = preOut[timeIdx * preDim + dimIdx];
		    }else if (dimIdx < featDim * 2){
			dimIdx = dimIdx - featDim;
			t.get<0>() = 0;
		    }else{
			dimIdx = dimIdx - featDim * 2;
			t.get<0>() = 1.0;
		    }
		}
	    }else{
	    
		if (preDim == dataDim){
		    // Previous transform layer has mean and std
		    if (dimIdx < featDim){
			// output dimension
			t.get<0>() = (preOut[t.get<1>()] *
				      exp(abData[(timeIdx - parallel)*2*featDim+featDim+dimIdx])+
				      abData[(timeIdx - parallel) * 2 * featDim + dimIdx]);
		    }else if (dimIdx < featDim * 2){
			// mean part
			dimIdx = dimIdx - featDim;
			t.get<0>() = (preOut[t.get<1>()] *
				      exp(abData[(timeIdx - parallel)*2*featDim+featDim+dimIdx])+
				      abData[(timeIdx - parallel) * 2 * featDim + dimIdx]);
		    }else{
			// std part 
			dimIdx = dimIdx - featDim * 2;
			t.get<0>() = (preOut[t.get<1>()] +
				      abData[(timeIdx - parallel) * 2 * featDim + featDim+dimIdx]);
		    }
		    
		}else{
		    // Previous transform layer has no mean and std
		    if (dimIdx < featDim){
			// output dimension
			t.get<0>() = (preOut[timeIdx * preDim + dimIdx] *
				      exp(abData[(timeIdx-parallel)*2*featDim+featDim+dimIdx]) +
				      abData[(timeIdx-parallel) * 2 * featDim + dimIdx]);
		    }else if (dimIdx < featDim * 2){
			// mean part set to 0 * b + a
			dimIdx = dimIdx - featDim;
			t.get<0>() = abData[(timeIdx-parallel) * 2 * featDim + dimIdx];
		    }else{
			dimIdx = dimIdx - featDim * 2;
			t.get<0>() = abData[(timeIdx-parallel) * 2 * featDim + featDim + dimIdx];
		    }
		}
	    
	    }
		
	}
    };

    struct trans_ax_b_with_mean_std_grad
    {
	
	real_t *abData;
	real_t *abError;
	
	real_t *preData;
	real_t *preError;

	real_t *thisError;
	
	int dataDim;
	int preDim;
	int featDim;
	
	bool abFromSkipLayer;
	bool preFromSkipLayer;
	
	const char *patTypes;

	__host__ __device__ void operator() (const thrust::tuple<real_t&, int> &t) const
	{
	    
	    int outputIdx = t.get<1>();
	    int timeIdx   = outputIdx / featDim;
	    int dimIdx    = outputIdx % featDim;

	    real_t grad_a = 0.0;
	    real_t grad_b = 0.0;
	    real_t grad_z = 0.0;
	    real_t grad_u = 0.0;
	    real_t grad_s = 0.0;

	    real_t scaleVal = 0.0;
	    
	    // skip dummy frame (for parallel sentence processing)
	    if (patTypes != NULL && patTypes[timeIdx] == PATTYPE_NONE){
		if (!abFromSkipLayer){
		    abError[timeIdx * featDim * 2 + dimIdx] = 0;
		    abError[timeIdx * featDim * 2 + dimIdx + featDim] = 0;
		}
		if (!preFromSkipLayer){
		    if (dataDim == preDim){
			preError[timeIdx * preDim + dimIdx] = 0; // x data
			preError[timeIdx * preDim + dimIdx + featDim] = 0; // mean
			preError[timeIdx * preDim + dimIdx + featDim + featDim] = 0; // std
		    }else{
			preError[timeIdx * preDim + dimIdx] = 0;
		    }
		}
		
	    }else{

		/*
		// gradient to a
		// \E/\a = \E/\z + \E/\u
		grad_a = (thisError[timeIdx * dataDim + dimIdx] +
			  thisError[timeIdx * dataDim + dimIdx + featDim]);

		// \E/\z = \E/\z * b
		grad_z = (thisError[timeIdx * dataDim + dimIdx] *
			  abData[timeIdx * featDim * 2 + dimIdx + featDim]);
		
		if (dataDim == preDim){
		    
		    // previous transform layer has mean and std
		    // \E/\b = \E / \z * z + \E / \s * 1/s + \E / \u * u
		    grad_b = (thisError[timeIdx * dataDim + dimIdx] *
			      preData[timeIdx * preDim + dimIdx]    +
			      thisError[timeIdx * dataDim + dimIdx + featDim] *
			      preData[timeIdx * preDim + dimIdx + featDim]    +
			      thisError[timeIdx * dataDim + dimIdx + 2 * featDim] *
			      preData[timeIdx * preDim + dimIdx + 2 * featDim] /
			      (preData[timeIdx * preDim + dimIdx + 2 * featDim] *
			       preData[timeIdx * preDim + dimIdx + 2 * featDim] +
			       STRUCTTRANS_AX_B_WITH_MU_STD_STD_FLOOR));

		    // \E / \s = \E / \s 
		    grad_s = thisError[timeIdx * dataDim + dimIdx + featDim * 2];

		    // \E / \u = \E / \u * b
		    grad_u = (thisError[timeIdx * dataDim + dimIdx + featDim] *
			      abData[timeIdx * featDim * 2 + dimIdx + featDim]);

		    if (preFromSkipLayer){
			preError[timeIdx * preDim + dimIdx]               += grad_z;
			preError[timeIdx * preDim + dimIdx + featDim]     += grad_u;
			preError[timeIdx * preDim + dimIdx + featDim * 2] += grad_s;
		    }else{
			preError[timeIdx * preDim + dimIdx]                = grad_z;
			preError[timeIdx * preDim + dimIdx + featDim]      = grad_u;
			preError[timeIdx * preDim + dimIdx + featDim * 2]  = grad_s;
		    }
		    
		}else{
		    
		    // previous transform layer has mean and std
		    // \E/\b = \E / \z * z + \E / \s * 1 + \E / \u * 0
		    grad_b = (thisError[timeIdx * dataDim + dimIdx] *
			      preData[timeIdx * preDim + dimIdx]    +
			      thisError[timeIdx * dataDim + dimIdx + 2 * featDim]);

		    if (preFromSkipLayer){
			preError[timeIdx * preDim + dimIdx] += grad_z;
		    }else{
			preError[timeIdx * preDim + dimIdx] = grad_z;
		    }
		}
		
		if (abFromSkipLayer){
		    abError[timeIdx * featDim * 2 + dimIdx + featDim] += grad_b;
		    abError[timeIdx * featDim * 2 + dimIdx]           += grad_a;
		}else{
		    abError[timeIdx * featDim * 2 + dimIdx + featDim] = grad_b;
		    abError[timeIdx * featDim * 2 + dimIdx]           = grad_a;
		}
		*/

		scaleVal = exp(abData[timeIdx * featDim * 2 + dimIdx + featDim]);
		
		// gradient to a
		// \E/\a = \E/\z + \E/\u
		grad_a = (thisError[timeIdx * dataDim + dimIdx] +
			  thisError[timeIdx * dataDim + dimIdx + featDim]);

		// \E/\z = \E/\z * b
		grad_z = thisError[timeIdx * dataDim + dimIdx] * scaleVal;
		
		if (dataDim == preDim){
		    
		    // previous transform layer has mean and std
		    // \E/\b = \E / \z * z + \E / \s * 1/s + \E / \u * u
		    grad_b = ((thisError[timeIdx * dataDim + dimIdx] *
			       preData[timeIdx * preDim + dimIdx] *
			       scaleVal) +
			      (thisError[timeIdx * dataDim + dimIdx + featDim] *
			       preData[timeIdx * preDim + dimIdx + featDim] *
			       scaleVal) +
			      thisError[timeIdx * dataDim + dimIdx + 2 * featDim]);

		    // \E / \s = \E / \s 
		    grad_s = thisError[timeIdx * dataDim + dimIdx + featDim * 2];

		    // \E / \u = \E / \u * b
		    grad_u = thisError[timeIdx * dataDim + dimIdx + featDim] * scaleVal;

		    if (preFromSkipLayer){
			preError[timeIdx * preDim + dimIdx]               += grad_z;
			preError[timeIdx * preDim + dimIdx + featDim]     += grad_u;
			preError[timeIdx * preDim + dimIdx + featDim * 2] += grad_s;
		    }else{
			preError[timeIdx * preDim + dimIdx]                = grad_z;
			preError[timeIdx * preDim + dimIdx + featDim]      = grad_u;
			preError[timeIdx * preDim + dimIdx + featDim * 2]  = grad_s;
		    }
		    
		}else{
		    
		    // previous transform layer has no mean and std
		    // \E/\b = \E / \z * z + \E / \s * 1 + \E / \u * 0
		    grad_b = ((thisError[timeIdx * dataDim + dimIdx] *
			       preData[timeIdx * preDim + dimIdx] *
			       scaleVal) +
			      thisError[timeIdx * dataDim + dimIdx + 2 * featDim]);

		    if (preFromSkipLayer){
			preError[timeIdx * preDim + dimIdx] += grad_z;
		    }else{
			preError[timeIdx * preDim + dimIdx] = grad_z;
		    }
		}
		
		if (abFromSkipLayer){
		    abError[timeIdx * featDim * 2 + dimIdx + featDim] += grad_b;
		    abError[timeIdx * featDim * 2 + dimIdx]           += grad_a;
		}else{
		    abError[timeIdx * featDim * 2 + dimIdx + featDim] = grad_b;
		    abError[timeIdx * featDim * 2 + dimIdx]           = grad_a;
		}
	    }
	}
    };

    struct trans_ax_b_with_mean_std_grad_shift
    {
	
	real_t *abData;
	real_t *abError;
	
	real_t *preData;
	real_t *preError;

	real_t *thisError;
	
	int dataDim;
	int preDim;
	int featDim;
	int parallel;
	
	bool abFromSkipLayer;
	bool preFromSkipLayer;
	
	const char *patTypes;

	__host__ __device__ void operator() (const thrust::tuple<real_t&, int> &t) const
	{
	    
	    int outputIdx = t.get<1>();
	    int timeIdx   = outputIdx / featDim;
	    int dimIdx    = outputIdx % featDim;

	    real_t grad_a = 0.0;
	    real_t grad_b = 0.0;
	    real_t grad_z = 0.0;
	    real_t grad_u = 0.0;
	    real_t grad_s = 0.0;

	    real_t scaleVal = 0.0;
	    
	    // skip dummy frame (for parallel sentence processing)
	    if (patTypes != NULL && patTypes[timeIdx] == PATTYPE_NONE){
		
		if (!abFromSkipLayer){
		    if (timeIdx > parallel){
			abError[(timeIdx - parallel) * featDim * 2 + dimIdx] = 0;
			abError[(timeIdx - parallel) * featDim * 2 + dimIdx + featDim] = 0;
		    }
		}
		if (!preFromSkipLayer){
		    if (dataDim == preDim){
			preError[timeIdx * preDim + dimIdx] = 0; // x data
			preError[timeIdx * preDim + dimIdx + featDim] = 0; // mean
			preError[timeIdx * preDim + dimIdx + featDim + featDim] = 0; // std
		    }else{
			preError[timeIdx * preDim + dimIdx] = 0;
		    }
		}
		
	    }else if (timeIdx < parallel){

		grad_z = thisError[timeIdx * dataDim + dimIdx];
		preError[timeIdx * preDim + dimIdx]               += grad_z;
		
	    }else{
		
		scaleVal = exp(abData[(timeIdx-parallel) * featDim * 2 + dimIdx + featDim]);
		
		// gradient to a
		// \E/\a = \E/\z + \E/\u
		grad_a = (thisError[timeIdx * dataDim + dimIdx] +
			  thisError[timeIdx * dataDim + dimIdx + featDim]);

		// \E/\z = \E/\z * b
		grad_z = thisError[timeIdx * dataDim + dimIdx] * scaleVal;
		
		if (dataDim == preDim){
		    
		    // previous transform layer has mean and std
		    // \E/\b = \E / \z * z + \E / \s * 1/s + \E / \u * u
		    grad_b = ((thisError[timeIdx * dataDim + dimIdx] *
			       preData[timeIdx * preDim + dimIdx] *
			       scaleVal) +
			      (thisError[timeIdx * dataDim + dimIdx + featDim] *
			       preData[timeIdx * preDim + dimIdx + featDim] *
			       scaleVal) +
			      thisError[timeIdx * dataDim + dimIdx + 2 * featDim]);

		    // \E / \s = \E / \s 
		    grad_s = thisError[timeIdx * dataDim + dimIdx + featDim * 2];

		    // \E / \u = \E / \u * b
		    grad_u = thisError[timeIdx * dataDim + dimIdx + featDim] * scaleVal;

		    if (preFromSkipLayer){
			preError[timeIdx * preDim + dimIdx]               += grad_z;
			preError[timeIdx * preDim + dimIdx + featDim]     += grad_u;
			preError[timeIdx * preDim + dimIdx + featDim * 2] += grad_s;
		    }else{
			preError[timeIdx * preDim + dimIdx]                = grad_z;
			preError[timeIdx * preDim + dimIdx + featDim]      = grad_u;
			preError[timeIdx * preDim + dimIdx + featDim * 2]  = grad_s;
		    }
		    
		}else{
		    
		    // previous transform layer has no mean and std
		    // \E/\b = \E / \z * z + \E / \s * 1 + \E / \u * 0
		    grad_b = ((thisError[timeIdx * dataDim + dimIdx] *
			       preData[timeIdx * preDim + dimIdx] *
			       scaleVal) +
			      thisError[timeIdx * dataDim + dimIdx + 2 * featDim]);

		    if (preFromSkipLayer){
			preError[timeIdx * preDim + dimIdx] += grad_z;
		    }else{
			preError[timeIdx * preDim + dimIdx] = grad_z;
		    }
		}
		
		if (abFromSkipLayer){
		    abError[(timeIdx - parallel) * featDim * 2 + dimIdx + featDim] += grad_b;
		    abError[(timeIdx - parallel) * featDim * 2 + dimIdx]           += grad_a;
		}else{
		    abError[(timeIdx - parallel) * featDim * 2 + dimIdx + featDim] = grad_b;
		    abError[(timeIdx - parallel) * featDim * 2 + dimIdx]           = grad_a;
		}
	    }
	}
    };

    // 
    struct TimeReverse
    {
	int    *seqLength;
	int     featDim;
	int     parallel;
	real_t *inputBuf;
	const char *patTypes;   // 

	__host__ __device__ void  operator() (
		const thrust::tuple<real_t&, const int&> &t) const
	{
	    int timeIdx  = t.get<1>() / featDim;
	    int dimIdx   = t.get<1>() % featDim;
	    int blockIdx = timeIdx / parallel;
	    int paralIdx = timeIdx % parallel;
	    
	    int thisSeqLength = seqLength[paralIdx];

	    if (patTypes[timeIdx] == PATTYPE_NONE){
		return;
	    }else{
		t.get<0>() = inputBuf[((thisSeqLength - 1 - blockIdx) * parallel + paralIdx) *
				      featDim + dimIdx];
	    }
	    
	}
    };

    
}
}


namespace layers{

    // Construct the layer
    template <typename TDevice>
    StructTransLayer<TDevice>::StructTransLayer(const helpers::JsonValue &layerChild,
						const helpers::JsonValue &weightsSection,
						std::vector<Layer<TDevice>*> &precedingLayers,
						int maxSeqLength,
						int layerID)
	// use preLayers[0] as fake preceding layers
	: SkipLayer<TDevice>(layerChild, weightsSection, precedingLayers,
			     maxSeqLength, layerID, false)
    {
	// Initial check
	if (precedingLayers.size() < 1)
	    throw std::runtime_error("Error no precedinglayers in skipadd/skipini");

	/* --------- Configuration -------- */
	// Type of transformation
	//   STRUCTTRANS_AX_B:
	//       Ax + B, common and simple transformation
	//   STRUCTTRANS_AX_B_WITH_MU_STD:
	//       Ax + B with mean and std, transformation the random variable and also the mean
	//       and std of the p(x) (Gaussian)
	//   STRUCTTRANS_AX_B_WITH_MU_STD_SHIFT:
	//       STRUCTTRANS_AX_B_WITH_MU_STD and make it causal
	// Note: I used STRUCTTRANS_AX_B_WITH_MU_STD and STRUCTTRANS_AX_B_WITH_MU_STD_SHIFT for
	//     parallel WaveNet and ClariNet, although not successful
	m_structTransType = (layerChild->HasMember("transType") ? 
			     ((*layerChild)["transType"].GetInt()) : STRUCTTRANS_DEFAULT);

	// Input layers should be specified through preSkipLayer
	m_previousSkipStr = (layerChild->HasMember("preSkipLayer") ? 
			     ((*layerChild)["preSkipLayer"].GetString()) : "");

	// Time reverse the signal after transformation
	m_structReverse   = (layerChild->HasMember("reverse") ? 
			     ((*layerChild)["reverse"].GetInt()) : STRUCTTRANS_DIREC_NOTREVERSE);

	/* --------- Data buffer and initialization -------- */
	// Buffer to store the length of sequences (used to reverse signal)
	m_seqLengthCpu = Cpu::int_vector(this->parallelSequences(), 0);
	m_seqLength = m_seqLengthCpu;


	// Get the name of input layers
	if (m_previousSkipStr.size()){    
	    // previous layers are specified by preSkipLayer
	    std::vector<std::string> tmpOpt;
	    misFuncs::ParseStrOpt(m_previousSkipStr, tmpOpt, ",");
	    for (int cnt = 0 ; cnt < tmpOpt.size(); cnt++) {
		BOOST_FOREACH (Layer<TDevice> *layer, precedingLayers) {
		    if (layer->name() == tmpOpt[cnt]){
			this->PreLayers().push_back(layer);
			break;
		    }
		}
	    }
	}else{
	    // default cause, use only the previous 1 skip and previous normal output layer
	    if (precedingLayers.size()<2)
		this->PreLayers().assign(precedingLayers.begin(), precedingLayers.end());
	    else
		this->PreLayers().assign(precedingLayers.end()-2, precedingLayers.end());
	}

	/* --------- Check ------------ */
	if (this->PreLayers().size() != 2)
	    throw std::runtime_error("Error: structTransform layer only needs two preSkipLayers");

	if (m_structTransType == STRUCTTRANS_AX_B){
	    // For y= Ax+B, preLayer[0] should contain A and B
	    //              preLayer[1] should contain x
	    //              preLayer[0]->size() should be 2 * preLayer[1]->size()
	    if (this->PreLayers()[0]->size() != this->size() * 2 ||
		this->PreLayers()[1]->size() != this->size())
		throw std::runtime_error("Error: Structtransform type 1, layer size mismatch");
	    printf("\n\tStruct transform Ax + B, where A, B from %s, x from %s\n",
		   this->PreLayers()[0]->name().c_str(), this->PreLayers()[1]->name().c_str());
	    
	}else if (m_structTransType == STRUCTTRANS_AX_B_WITH_MU_STD ||
		  m_structTransType == STRUCTTRANS_AX_B_WITH_MU_STD_SHIFT ){
	    // For y = Ax + B and mean std, preLayer[0] should contain A and B
	    //                              preLayer[1] should contain x, x_mu, x_std, or x only
	    //                              this layer should contain y, y_mu, y_std
	    if (this->size()/3 != this->PreLayers()[0]->size()/2){
		printf("\nThis layer size: %d, 1st preSkipLayr size should be %d",
		       this->size(), this->size()/3*2);
		throw std::runtime_error("Error: Structtransform type 3, pre-layer size mismatch");
	    }
	    if (this->size()/3 != this->PreLayers()[1]->size() &&
		this->size() != this->PreLayers()[1]->size()){
		printf("\nThis layer size: %d, 2nd preSkipLayr size should be %d or %d",
		       this->size(), this->size(), this->size()/3);
		throw std::runtime_error("Error: Structtransform type 3, pre-layer size mismatch");
	    }
	    printf("\n\tCaution: the next layer should be a skpicat or distilling layer\n");
	}else{
	    throw std::runtime_error("Error: Struct transform type not implemented");
	}
	
	if (m_structReverse == STRUCTTRANS_DIREC_REVERSE){
	    printf("\n\tStructform plus time reverse");
	}
	
	this->__allocateLocalMem();
    }	

    // Destructor
    template <typename TDevice>
    StructTransLayer<TDevice>::~StructTransLayer()
    {
    }

    template <typename TDevice>
    void StructTransLayer<TDevice>::__allocateLocalMem()
    {
	if (m_structReverse == STRUCTTRANS_DIREC_REVERSE){
	    m_tempOutput = this->outputs();
	}else{
	    m_tempOutput.clear();
	}
    }
    
    template <typename TDevice>
    void StructTransLayer<TDevice>::__clearLocalMem()
    {
	if (m_structReverse == STRUCTTRANS_DIREC_REVERSE)
	    m_tempOutput.clear(); m_tempOutput.shrink_to_fit();
	
    }
    
    
    // NN forward
    template <typename TDevice>
    void StructTransLayer<TDevice>::computeForwardPass(const int nnState)
    {
	if (this->getSaveMemoryFlag())
	    throw std::runtime_error("Memory save mode should be turned off");
	
	// Initialization for backward pass
	//   Because gradients will be accumulated from multiple layers,
	//   we clean the gradients buffer before accumulation
	if (this->flagTrainingMode()){
	    thrust::fill(this->outputErrors().begin(), 
			 (this->outputErrors().begin() + 
			  this->curMaxSeqLength() * this->parallelSequences() * this->size()),
			 0.0);

	    thrust::fill(this->outputErrorsFromSkipLayer().begin(),
			 (this->outputErrorsFromSkipLayer().begin() + 
			  this->curMaxSeqLength() * this->parallelSequences() * this->size()),
			 0.0);
	}

	int timeLength = this->curMaxSeqLength() * this->parallelSequences();
	
	if (this->m_structTransType == STRUCTTRANS_AX_B){
	    // Type 1 transformation
	    thrust::copy(this->PreLayers()[1]->outputs().begin(),
			 this->PreLayers()[1]->outputs().end(),
			 this->outputs().begin());
	    
	    internal::trans_ax_b fn1;
	    fn1.abData = helpers::getRawPointer(this->PreLayers()[0]->outputs());
	    fn1.dataDim = this->size();
	    fn1.abShiftT = 0;
	    fn1.patTypes = helpers::getRawPointer(this->patTypes());

	    thrust::for_each(
	      thrust::make_zip_iterator(
		thrust::make_tuple(this->outputs().begin(), 
				   thrust::counting_iterator<int>(0))),
	      thrust::make_zip_iterator(
		thrust::make_tuple(this->outputs().begin()           + timeLength * this->size(), 
				   thrust::counting_iterator<int>(0) + timeLength * this->size())),
	      fn1);
	    
	}else if (this->m_structTransType == STRUCTTRANS_AX_B_WITH_MU_STD){

	    
	    internal::trans_ax_b_with_mean_std fn1;
	    fn1.abData = helpers::getRawPointer(this->PreLayers()[0]->outputs());
	    fn1.preOut = helpers::getRawPointer(this->PreLayers()[1]->outputs());
	    
	    fn1.preDim  = this->PreLayers()[1]->size();
	    fn1.dataDim = this->size();

	    fn1.patTypes = helpers::getRawPointer(this->patTypes());

	    thrust::for_each(
	         thrust::make_zip_iterator(
		  thrust::make_tuple(
			this->outputs().begin(),
			thrust::counting_iterator<int>(0))),
		 thrust::make_zip_iterator(
		  thrust::make_tuple(
			this->outputs().begin()                 +timeLength * this->size(),
			thrust::counting_iterator<int>(0)       +timeLength * this->size())),
		 fn1);
	
	    
	}else if (this->m_structTransType == STRUCTTRANS_AX_B_WITH_MU_STD_SHIFT){

	    internal::trans_ax_b_with_mean_std_shifted fn1;
	    fn1.abData   = helpers::getRawPointer(this->PreLayers()[0]->outputs());
	    fn1.preOut   = helpers::getRawPointer(this->PreLayers()[1]->outputs());	
	    fn1.parallel = this->parallelSequences();
	    fn1.preDim   = this->PreLayers()[1]->size();
	    fn1.dataDim  = this->size();

	    fn1.patTypes = helpers::getRawPointer(this->patTypes());

	    thrust::for_each(
	         thrust::make_zip_iterator(
		  thrust::make_tuple(
			this->outputs().begin(),
			thrust::counting_iterator<int>(0))),
		 thrust::make_zip_iterator(
		  thrust::make_tuple(
			this->outputs().begin()                 +timeLength * this->size(),
			thrust::counting_iterator<int>(0)       +timeLength * this->size())),
		 fn1);
	
	    
	}else{
	    throw std::runtime_error("Impossible Error");
	}

	// time reverse
	if (this->m_structReverse == STRUCTTRANS_DIREC_REVERSE){
	    {{
		internal::TimeReverse fn;

		fn.seqLength = helpers::getRawPointer(m_seqLength);
		fn.featDim   = this->size();
		fn.parallel  = this->parallelSequences();
		fn.inputBuf  = helpers::getRawPointer(this->outputs());
		fn.patTypes  = helpers::getRawPointer(this->patTypes());
		
		thrust::for_each(
		  thrust::make_zip_iterator(
		    thrust::make_tuple(
			m_tempOutput.begin(),
			thrust::counting_iterator<int>(0))),
		  thrust::make_zip_iterator(
		    thrust::make_tuple(
			m_tempOutput.begin()              + timeLength * this->size(),
			thrust::counting_iterator<int>(0) + timeLength * this->size())),
		  fn);
	    }}
	    m_tempOutput.swap(this->outputs());
	}
	    
    }

    // NN forward
    template <typename TDevice>
    void StructTransLayer<TDevice>::computeForwardPass(const int timeStep, const int nnState)
    {

	// absolute time
	int effTimeS = timeStep     * this->parallelSequences();
	int effTimeE = (timeStep+1) * this->parallelSequences();
	
	// shift of the pointer to the data 
	int shiftIn0 = this->PreLayers()[0]->outputBufPtrBias(timeStep * this->parallelSequences(),
							      nnState);
	int shiftIn1 = this->PreLayers()[1]->outputBufPtrBias(timeStep * this->parallelSequences(),
							      nnState);
	int shiftOut = this->outputBufPtrBias(timeStep * this->parallelSequences(), nnState);

	// set gradients to zero in necessary
	if (timeStep == 0 && this->flagTrainingMode()){
	    thrust::fill(this->outputErrors().begin(), 
			 (this->outputErrors().begin() + 
			  this->curMaxSeqLength() * this->parallelSequences() * this->size()),
			 0.0);

	    thrust::fill(this->outputErrorsFromSkipLayer().begin(),
			 (this->outputErrorsFromSkipLayer().begin() + 
			  this->curMaxSeqLength() * this->parallelSequences() * this->size()),
			 0.0);
	}

	if (this->m_structTransType == STRUCTTRANS_AX_B){
	    // Type 1 transformation
	    thrust::copy(this->PreLayers()[1]->outputs().begin() +
			 effTimeS * this->PreLayers()[1]->size() - shiftIn1,
			 this->PreLayers()[1]->outputs().begin() +
			 effTimeE * this->PreLayers()[1]->size() - shiftIn1,
			 this->outputs().begin() + effTimeS * this->size() - shiftOut);
	    
	    internal::trans_ax_b fn1;
	    fn1.abData = helpers::getRawPointer(this->PreLayers()[0]->outputs()) +
		effTimeS * this->PreLayers()[0]->size() - shiftIn0;
	    fn1.dataDim = this->size();
	    fn1.abShiftT = 0;
	    fn1.patTypes = helpers::getRawPointer(this->patTypes());

	    thrust::for_each(
	     thrust::make_zip_iterator(
	      thrust::make_tuple(
		this->outputs().begin() + effTimeS * this->size() - shiftOut, 
		thrust::counting_iterator<int>(0) + effTimeS * this->size() - shiftOut)),
	     thrust::make_zip_iterator(
	      thrust::make_tuple(
	        this->outputs().begin() + effTimeE * this->size() - shiftOut, 
		thrust::counting_iterator<int>(0) + effTimeE * this->size() - shiftOut)),
	      fn1);
	    
	}else if (this->m_structTransType == STRUCTTRANS_AX_B_WITH_MU_STD){
	    throw std::runtime_error("Impossible Error");
	    
	}else if (this->m_structTransType == STRUCTTRANS_AX_B_WITH_MU_STD_SHIFT){
	    throw std::runtime_error("Impossible Error");
	    
	}else{
	    throw std::runtime_error("Impossible Error");
	}

		// time reverse
	if (this->m_structReverse == STRUCTTRANS_DIREC_REVERSE){
	    printf("\nError: structTransLayer computeForward(timeStep) cannot support reverse");
	    throw std::runtime_error("Error: structTransLayer computeForward(timeStep)");
	}
	
    }

    // NN forward
    template <typename TDevice>
    void StructTransLayer<TDevice>::computeBackwardPass(const int timeStep, const int nnState)
    {
	// Not implemented yet
	throw std::runtime_error("Not implemented structTransLayer computeBackward(timeStep)");	
    }

    // NN backward
    template <typename TDevice>
    void StructTransLayer<TDevice>::computeBackwardPass(const int nnState)
    {
	
	if (this->getSaveMemoryFlag())
	    throw std::runtime_error("Memory save mode should be turned off");
	
	// Both physical and virtual layers need to handle the gradients

	int timeLength = this->precedingLayer().curMaxSeqLength();
	timeLength = timeLength * this->precedingLayer().parallelSequences();

	// Step1. accumulate gradients from this->outputErrorsFromSkipLayer()
	//     this step is the same as other skipadd or skipini or skipcat layers
	thrust::transform(this->outputErrorsFromSkipLayer().begin(),
			  (this->outputErrorsFromSkipLayer().begin() + 
			   this->curMaxSeqLength() * this->parallelSequences() * this->size()),
			  this->outputErrors().begin(),
			  this->outputErrors().begin(),
			  thrust::plus<real_t>());


	// Step2. reverse the signal if necessary
	if (this->m_structReverse == STRUCTTRANS_DIREC_REVERSE){
	    // reverse back the output signals
	    m_tempOutput.swap(this->outputs());
	    
	    // reverse the gradients
	    {{
		internal::TimeReverse fn;
		fn.seqLength = helpers::getRawPointer(m_seqLength);
		fn.featDim   = this->size();
		fn.parallel  = this->parallelSequences();
		fn.inputBuf  = helpers::getRawPointer(this->outputErrors());
		fn.patTypes  = helpers::getRawPointer(this->patTypes());
		
		thrust::for_each(
		  thrust::make_zip_iterator(
		    thrust::make_tuple(
			m_tempOutput.begin(),
			thrust::counting_iterator<int>(0))),
		  thrust::make_zip_iterator(
		    thrust::make_tuple(
			m_tempOutput.begin()              + timeLength * this->size(),
			thrust::counting_iterator<int>(0) + timeLength * this->size())),
		  fn);
	    }}
	    thrust::copy(m_tempOutput.begin(), m_tempOutput.end(), this->outputErrors().begin());
	    
	}
	
	
       	// Step3. propagate the gradients
	if (this->m_structTransType == STRUCTTRANS_AX_B){

	    // In case the previous layers are skiplayers, the gradients
	    //  should be put into outputErrorsFromSkipLayer(), not outputErrors() directly
	    SkipLayer<TDevice>* tempLayerAb =
		dynamic_cast<SkipLayer<TDevice>*>(this->PreLayers()[0]);
	    SkipLayer<TDevice>* tempLayerX =
		dynamic_cast<SkipLayer<TDevice>*>(this->PreLayers()[1]);
	    
	    internal::trans_ax_b_grad fn1;
	    fn1.abData = helpers::getRawPointer(this->PreLayers()[0]->outputs());
	    fn1.xData  = helpers::getRawPointer(this->PreLayers()[1]->outputs());
	    if (tempLayerAb){
		fn1.abFromSkipLayer = true;
		fn1.abError = helpers::getRawPointer(tempLayerAb->outputErrorsFromSkipLayer());
	    }else{
		fn1.abFromSkipLayer = false;
		fn1.abError = helpers::getRawPointer(this->PreLayers()[0]->outputErrors());
	    }
	    if (tempLayerX){
		fn1.xFromSkipLayer = true;
		fn1.xError = helpers::getRawPointer(tempLayerX->outputErrorsFromSkipLayer());
	    }else{
		fn1.xFromSkipLayer = false;
		fn1.xError = helpers::getRawPointer(this->PreLayers()[1]->outputErrors());
	    }
	    fn1.dataDim = this->size();
	    fn1.patTypes = helpers::getRawPointer(this->patTypes());
	    
	    thrust::for_each(
	      thrust::make_zip_iterator(
		thrust::make_tuple(this->outputErrors().begin(), 
				   thrust::counting_iterator<int>(0))),
	      thrust::make_zip_iterator(
		thrust::make_tuple(this->outputErrors().begin()      + timeLength * this->size(), 
				   thrust::counting_iterator<int>(0) + timeLength * this->size())),
	      fn1);

	    //printf("%s", this->name().c_str());
	    
	}else if (this->m_structTransType == STRUCTTRANS_AX_B_WITH_MU_STD){

	    SkipLayer<TDevice>* tempLayerAb =
		dynamic_cast<SkipLayer<TDevice>*>(this->PreLayers()[0]);
	    SkipLayer<TDevice>* tempLayerPre =
		dynamic_cast<SkipLayer<TDevice>*>(this->PreLayers()[1]);
	    
	    internal::trans_ax_b_with_mean_std_grad fn1;
	    fn1.abData   = helpers::getRawPointer(this->PreLayers()[0]->outputs());
	    fn1.preData  = helpers::getRawPointer(this->PreLayers()[1]->outputs());
	    fn1.thisError= helpers::getRawPointer(this->outputErrors());
	    
	    if (tempLayerAb){
		fn1.abFromSkipLayer = true;
		fn1.abError = helpers::getRawPointer(tempLayerAb->outputErrorsFromSkipLayer());
	    }else{
		fn1.abFromSkipLayer = false;
		fn1.abError = helpers::getRawPointer(this->PreLayers()[0]->outputErrors());
	    }
	    
	    // In fact, tempLayerPre must be a skipLayer
	    if (tempLayerPre){
		fn1.preFromSkipLayer = true;
		fn1.preError = helpers::getRawPointer(tempLayerPre->outputErrorsFromSkipLayer());
	    }else{
		fn1.preFromSkipLayer = false;
		fn1.preError = helpers::getRawPointer(this->PreLayers()[1]->outputErrors());
	    }
	    
	    fn1.dataDim = this->size();
	    fn1.featDim = this->size()/3;
	    fn1.preDim  = this->PreLayers()[1]->size();
	    fn1.patTypes = helpers::getRawPointer(this->patTypes());

	    thrust::for_each(
	      thrust::make_zip_iterator(
		thrust::make_tuple(
			this->outputErrors().begin(), 
			thrust::counting_iterator<int>(0))),
	      thrust::make_zip_iterator(
		thrust::make_tuple(
			this->outputErrors().begin()      + timeLength * this->size()/3, 
			thrust::counting_iterator<int>(0) + timeLength * this->size()/3)),
	      fn1);

	    //printf("%s", this->name().c_str());
	

	}else if (this->m_structTransType == STRUCTTRANS_AX_B_WITH_MU_STD_SHIFT){

	    SkipLayer<TDevice>* tempLayerAb =
		dynamic_cast<SkipLayer<TDevice>*>(this->PreLayers()[0]);
	    SkipLayer<TDevice>* tempLayerPre =
		dynamic_cast<SkipLayer<TDevice>*>(this->PreLayers()[1]);
	    
	    internal::trans_ax_b_with_mean_std_grad_shift fn1;
	    fn1.abData   = helpers::getRawPointer(this->PreLayers()[0]->outputs());
	    fn1.preData  = helpers::getRawPointer(this->PreLayers()[1]->outputs());
	    fn1.thisError= helpers::getRawPointer(this->outputErrors());
	    fn1.parallel = this->parallelSequences();
	    if (tempLayerAb){
		fn1.abFromSkipLayer = true;
		fn1.abError = helpers::getRawPointer(tempLayerAb->outputErrorsFromSkipLayer());
	    }else{
		fn1.abFromSkipLayer = false;
		fn1.abError = helpers::getRawPointer(this->PreLayers()[0]->outputErrors());
	    }
	    
	    // In fact, tempLayerPre must be a skipLayer
	    if (tempLayerPre){
		fn1.preFromSkipLayer = true;
		fn1.preError = helpers::getRawPointer(tempLayerPre->outputErrorsFromSkipLayer());
	    }else{
		fn1.preFromSkipLayer = false;
		fn1.preError = helpers::getRawPointer(this->PreLayers()[1]->outputErrors());
	    }
	    
	    fn1.dataDim = this->size();
	    fn1.featDim = this->size()/3;
	    fn1.preDim  = this->PreLayers()[1]->size();
	    fn1.patTypes = helpers::getRawPointer(this->patTypes());

	    thrust::for_each(
	      thrust::make_zip_iterator(
		thrust::make_tuple(
			this->outputErrors().begin(), 
			thrust::counting_iterator<int>(0))),
	      thrust::make_zip_iterator(
		thrust::make_tuple(
			this->outputErrors().begin()      + timeLength * this->size()/3, 
			thrust::counting_iterator<int>(0) + timeLength * this->size()/3)),
	      fn1);

	    //printf("%s", this->name().c_str());
	

	}else{
	    throw std::runtime_error("Impossible Error");	    
	}
	
    }
	
    
    template <typename TDevice>
    const std::string& StructTransLayer<TDevice>::type() const
    {
	static std::string s1("structTrans");
        return s1;
    }

    template <typename TDevice>
    void StructTransLayer<TDevice>::loadSequences(const data_sets::DataSetFraction &fraction,
						  const int nnState)
    {
	SkipLayer<TDevice>::loadSequences(fraction, nnState);
	
	// load the sequence length in fraction
	if (fraction.numSequences() > this->parallelSequences())
	    throw std::runtime_error("Impossible error: too many sequences in one fraction");

	thrust::fill(m_seqLengthCpu.begin(), m_seqLengthCpu.end(), 0.0);
	for (int i = 0; i<fraction.numSequences(); i++)
	    m_seqLengthCpu[i] = fraction.seqInfo(i).length;
	m_seqLength = m_seqLengthCpu;    
    }

    template <typename TDevice>
    void StructTransLayer<TDevice>::exportLayer(const helpers::JsonValue &layersArray,
					 const helpers::JsonAllocator &allocator) const
    {
	SkipLayer<TDevice>::exportLayer(layersArray, allocator);
	(*layersArray)[layersArray->Size() - 1].AddMember("transType",
							  m_structTransType, allocator);
	(*layersArray)[layersArray->Size() - 1].AddMember("preSkipLayer",
							  m_previousSkipStr.c_str(), allocator);
	if (m_structReverse == STRUCTTRANS_DIREC_REVERSE)
	    (*layersArray)[layersArray->Size() - 1].AddMember("reverse",
							      m_structReverse, allocator);
	    
    }
    
    template <typename TDevice>
    void StructTransLayer<TDevice>::reduceOutputBuffer()
    {
	this->resizeOutputBuffer(this->parallelSequences() * this->size());
	this->setSaveMemoryFlag(true);
	printf("\t[mem saved]");
    }

    template <typename TDevice>
    int StructTransLayer<TDevice>::outputBufPtrBias(const int timeStepTimesParallel,
						    const int nnState)
    {
	if (this->getSaveMemoryFlag()){
	    return timeStepTimesParallel * this->size();
	}else{
	    return 0;
	}
    }
    

    template <typename TDevice>
    void StructTransLayer<TDevice>::clearAllBuffers()
    {
	this->clearOutputBuffer();
	this->__clearLocalMem();
    }

    template <typename TDevice>
    void StructTransLayer<TDevice>::resizeAllBuffers(const int timeLength)
    {
	this->resizeOutputBuffer(timeLength * this->parallelSequences() * this->size());
	this->__allocateLocalMem();
    }    

    
    template class StructTransLayer<Cpu>;
    template class StructTransLayer<Gpu>;
    
}
