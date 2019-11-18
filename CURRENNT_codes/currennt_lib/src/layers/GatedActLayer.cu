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

#include "GatedActLayer.hpp"

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
#include <fstream>

#include "../activation_functions/Tanh.cuh"
#include "../activation_functions/Logistic.cuh"
#include "../activation_functions/Identity.cuh"
#include "../activation_functions/Relu.cuh"

#include "../Configuration.hpp"


namespace internal{
namespace {
    
    struct tanhSigMerge
    {
	
	int         shiftBuf;
	int         outputSize;
	real_t     *coreBuf;
	const char *patTypes;
	
	__host__ __device__ void operator() (const thrust::tuple<real_t&, int> &t) const
	{
	    int dimIdx    = t.get<1>() % outputSize;
	    int timeIdx   = t.get<1>() / outputSize;

	    if (patTypes[timeIdx] == PATTYPE_NONE){
		t.get<0>() = 0.0;
	    }else{
		int idx = timeIdx * 2 * outputSize + dimIdx - shiftBuf;
		t.get<0>() = (activation_functions::Tanh::fn(coreBuf[idx]) *
			      activation_functions::Logistic::fn(coreBuf[idx + outputSize]));
	    }
	}
    };

    struct tanhSigMergeGradient
    {

	int         outputSize;
	real_t     *coreBuf;
	real_t     *errors;
	const char *patTypes;
	
	__host__ __device__ void operator() (const thrust::tuple<real_t&, int> &t) const
	{
	    int dimIdx    = t.get<1>() % outputSize;
	    int timeIdx   = t.get<1>() / outputSize;

	    if (patTypes[timeIdx] == PATTYPE_NONE){
		t.get<0>() = 0.0;
	    }else{
		
		if (dimIdx < (outputSize/2)){
		    int idx = timeIdx * outputSize + dimIdx;		    
		    real_t tmp = activation_functions::Tanh::fn(coreBuf[idx]);
		    t.get<0>() = (((real_t)1.0 - tmp * tmp) *
				  activation_functions::Logistic::fn(coreBuf[idx + outputSize/2]) *
				  errors[timeIdx * outputSize / 2 + dimIdx]);
		    
		}else{
		    int idx = timeIdx * outputSize + dimIdx;
		    real_t tmp = activation_functions::Logistic::fn(coreBuf[idx]);
		    t.get<0>() = (activation_functions::Tanh::fn(coreBuf[idx - outputSize/2]) *
				  (tmp * ((real_t)1.0 - tmp)) *
				  errors[timeIdx * outputSize /2 + dimIdx - outputSize / 2]);
		}
		
	    }
	}
    };

}
}


namespace layers{

 
    template <typename TDevice>
    GatedActLayer<TDevice>::GatedActLayer(const helpers::JsonValue &layerChild,
				    const helpers::JsonValue &weightsSection,
				    Layer<TDevice>           &precedingLayer,
				    int                       maxSeqLength,
				    int                       layerID)
	: TrainableLayer<TDevice>(layerChild, weightsSection, 0, 0,
				  precedingLayer, maxSeqLength, layerID)
    {
	if (this->size() != precedingLayer.size() / 2)
	    throw std::runtime_error("gatedActLayer size should be 1/2 * previous_layer_size");
	
	
    }

    template <typename TDevice>
    GatedActLayer<TDevice>::~GatedActLayer()
    {
    }


    
    template <typename TDevice>
    void GatedActLayer<TDevice>::exportLayer(const helpers::JsonValue     &layersArray, 
					      const helpers::JsonAllocator &allocator) const
    {
        TrainableLayer<TDevice>::exportLayer(layersArray, allocator);
    }


    
    template <typename TDevice>
    void GatedActLayer<TDevice>::loadSequences(const data_sets::DataSetFraction &fraction,
					    const int nnState)
    {
	TrainableLayer<TDevice>::loadSequences(fraction, nnState);
    }
    
    template <typename TDevice>
    const std::string& GatedActLayer<TDevice>::type() const
    {
        static std::string s;
        if (s.empty()) s = "gatedact";
        return s;
    }


    template <typename TDevice>
    void GatedActLayer<TDevice>::computeForwardPass(const int nnState)
    {
	if (this->getSaveMemoryFlag())
	    throw std::runtime_error("Memory save mode should be turned off");
	
	int timeLength = this->curMaxSeqLength() * this->parallelSequences();
	
	// Step3. transform as output tanh(x1) * sig(x2)
	{
	    internal::tanhSigMerge fn1;
	    fn1.outputSize = this->size();
	    fn1.coreBuf    = helpers::getRawPointer(this->precedingLayer().outputs());
	    fn1.patTypes   = helpers::getRawPointer(this->patTypes());
	    fn1.shiftBuf   = 0;
	    
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
    
    template <typename TDevice>
    void GatedActLayer<TDevice>::computeForwardPass(const int timeStep, const int nnState)
    {
	int effTimeStep = timeStep * this->parallelSequences();
	int shiftPre    = this->precedingLayer().outputBufPtrBias(effTimeStep, nnState);
	int shiftCur    = this->outputBufPtrBias(effTimeStep, nnState);


	// Step3. transform as output tanh(x1) * sig(x2)
	{
	    internal::tanhSigMerge fn1;
	    fn1.outputSize = this->size();
	    fn1.coreBuf    = helpers::getRawPointer(this->precedingLayer().outputs());
	    fn1.patTypes   = helpers::getRawPointer(this->patTypes());
	    
	    fn1.shiftBuf   = shiftPre;
	    
	    int st = effTimeStep * this->size();
	    int et = (effTimeStep + this->parallelSequences()) * this->size();
	    
	    thrust::for_each(
               thrust::make_zip_iterator(
		  thrust::make_tuple(this->outputs().begin() + st - shiftCur,
				     thrust::counting_iterator<int>(0) + st)),
	       thrust::make_zip_iterator(
		  thrust::make_tuple(this->outputs().begin() + et - shiftCur,
				     thrust::counting_iterator<int>(0) + et)),
	       fn1);
	}
    }

    template <typename TDevice>
    void GatedActLayer<TDevice>::computeBackwardPass(const int nnState)
    {
	if (this->getSaveMemoryFlag())
	    throw std::runtime_error("Memory save mode should be turned off");
	
	int timeLength = this->curMaxSeqLength() * this->parallelSequences();
	
	{
	    internal::tanhSigMergeGradient fn1;
	    fn1.outputSize = this->size() * 2;
	    fn1.coreBuf    = helpers::getRawPointer(this->precedingLayer().outputs());
	    fn1.errors     = helpers::getRawPointer(this->outputErrors());
	    fn1.patTypes   = helpers::getRawPointer(this->patTypes());
	    
	    int n = timeLength * this->size() * 2;
	    thrust::for_each(
               thrust::make_zip_iterator(
		thrust::make_tuple(this->precedingLayer().outputErrors().begin(),
				     thrust::counting_iterator<int>(0))),
	       thrust::make_zip_iterator(
		thrust::make_tuple(this->precedingLayer().outputErrors().begin() + n,
				     thrust::counting_iterator<int>(0) + n)),
	       fn1);
	}
	
    }

    template <typename TDevice>
    void GatedActLayer<TDevice>::computeBackwardPass(const int timeStep, const int nnState)
    {
	if (this->getSaveMemoryFlag())
	    throw std::runtime_error("Memory save mode should be turned off");
	
	// absolute time
	int effTimeS = timeStep     * this->parallelSequences();
	int effTimeE = (timeStep+1) * this->parallelSequences();

	{
	    internal::tanhSigMergeGradient fn1;
	    fn1.outputSize = this->size() * 2;
	    fn1.coreBuf    = helpers::getRawPointer(this->precedingLayer().outputs());
	    fn1.errors     = helpers::getRawPointer(this->outputErrors());
	    fn1.patTypes   = helpers::getRawPointer(this->patTypes());
	    
	    thrust::for_each(
               thrust::make_zip_iterator(
		thrust::make_tuple(
		  this->precedingLayer().outputErrors().begin() + effTimeS * this->size() * 2,
		  thrust::counting_iterator<int>(0)             + effTimeS * this->size() * 2)),
	       thrust::make_zip_iterator(
		thrust::make_tuple(
		  this->precedingLayer().outputErrors().begin() + effTimeE * this->size() * 2,
		  thrust::counting_iterator<int>(0)             + effTimeE * this->size() * 2)),
	       fn1);
	}
	
    }

    template <typename TDevice>
    void GatedActLayer<TDevice>::reduceOutputBuffer()
    {
	//Layer<TDevice>::reduceOutputBuffer();
	this->resizeOutputBuffer(this->parallelSequences() * this->size());
	this->setSaveMemoryFlag(true);
	printf("\t[mem saved]");
    }

    template <typename TDevice>
    int  GatedActLayer<TDevice>::outputBufPtrBias(const int timeStepTimesParallel, const int nnState)
    {
	if (this->getSaveMemoryFlag())
	    return timeStepTimesParallel * this->size();
	else
	    return 0;
    }
    

    template <typename TDevice>
    std::vector<int> GatedActLayer<TDevice>::dependLayerIDs()
    {
	std::vector<int> tmp;
	tmp.push_back(this->precedingLayer().getLayerID());
	return tmp;
    }

    template <typename TDevice>
    void GatedActLayer<TDevice>::clearAllBuffers()
    {
	this->clearOutputBuffer();
    }

    template <typename TDevice>
    void GatedActLayer<TDevice>::resizeAllBuffers(const int timeLength)
    {
	this->resizeOutputBuffer(timeLength * this->parallelSequences() * this->size());
    }
    
    template class GatedActLayer<Cpu>;
    template class GatedActLayer<Gpu>;
    
}
