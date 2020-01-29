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


#include "SkipForFeatTransform.hpp"
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



namespace layers{

    // Construct the layer
    template <typename TDevice>
    SkipForFeatTrans<TDevice>::SkipForFeatTrans(
					const helpers::JsonValue &layerChild,
					const helpers::JsonValue &weightsSection,
					std::vector<Layer<TDevice>*> &precedingLayers,
					int maxSeqLength,
					int layerID)
	// use preLayers[0] as fake preceding layers
	: SkipLayer<TDevice>(layerChild, weightsSection, precedingLayers,
			     maxSeqLength, layerID, false)
	, m_flagSkipInit    (false)
	, m_virtualLayer    (false)
    {
	// Initial check
	if (precedingLayers.size() < 1)
	    throw std::runtime_error("Error no precedinglayers in skipadd/skipini");
	
	// for SkipForFeatTransform only layer
	this->PreLayers().assign(precedingLayers.end()-1, precedingLayers.end());
	
	printf("\n\tReceive input from layer(s):");
	BOOST_FOREACH (Layer<TDevice> *layer, this->PreLayers()) {
	    printf(" %s,", layer->name().c_str());
	    if (layer->size() != this->size()){
		printf("Error: %s vs %s", layer->name().c_str(), this->name().c_str());
		throw std::runtime_error("Error unequal layer size");
	    }
	}
	printf("\n");

	
    }	

    // Destructor
    template <typename TDevice>
    SkipForFeatTrans<TDevice>::~SkipForFeatTrans()
    {
    }

 
    // NN forward
    template <typename TDevice>
    void SkipForFeatTrans<TDevice>::computeForwardPass(const int nnState)
    {
	if (this->getSaveMemoryFlag())
	    throw std::runtime_error("Memory save mode should be turned off");
	

	// initialization for backward pass
	// (because gradients will be accumulated from multiple layers)
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

	thrust::fill(this->outputs().begin(), 
		     (this->outputs().begin() + 
		      this->curMaxSeqLength() * this->parallelSequences() * this->size()),
		     0.0);
	
	// accumulating the outputs of previous layers
	BOOST_FOREACH (Layer<TDevice> *layer, this->PreLayers()) {
	    thrust::transform(
			      layer->outputs().begin(),
			      (layer->outputs().begin() + 
			       this->curMaxSeqLength() * this->parallelSequences() * this->size()),
			      this->outputs().begin(),
			      this->outputs().begin(),
			      thrust::plus<real_t>());	    
	}
    }

    
    // NN forward
    template <typename TDevice>
    void SkipForFeatTrans<TDevice>::computeForwardPass(const int timeStep, const int nnState)
    {
	// absolute time
	int effTimeS = timeStep     * this->parallelSequences();
	int effTimeE = (timeStep+1) * this->parallelSequences();
	
	// shift of the pointer to the data 
	int shiftIn  = 0; // value to assigned layer
	int shiftOut = this->outputBufPtrBias(timeStep * this->parallelSequences(), nnState);
	
	// physical layer, compute the output
	thrust::fill(this->outputs().begin() + effTimeS * this->size() - shiftOut, 
		     this->outputs().begin() + effTimeE * this->size() - shiftOut, 
		     0.0);

	//int cnt = 0;
	//accumulating the outputs of previous layers
	BOOST_FOREACH (Layer<TDevice> *layer, this->PreLayers()) {
		//if (this->PreLayers().size() && cnt == 22)
		shiftIn = layer->outputBufPtrBias(timeStep * this->parallelSequences(), nnState);
		thrust::transform(layer->outputs().begin() + effTimeS * this->size() - shiftIn,
				  layer->outputs().begin() + effTimeE * this->size() - shiftIn,
				  this->outputs().begin()  + effTimeS * this->size() - shiftOut,
				  this->outputs().begin()  + effTimeS * this->size() - shiftOut,
				  thrust::plus<real_t>());
		//cnt++;
	}
    }


    // NN backward
    template <typename TDevice>
    void SkipForFeatTrans<TDevice>::computeBackwardPass(const int nnState)
    {
	if (this->getSaveMemoryFlag())
	    throw std::runtime_error("Memory save mode should be turned off");
	
	// Both physical and virtual layers need to handle the gradients
	
	// at first, add the errors in both this->outputErrorsFromSkipLayer() and m_outputErrors
	thrust::transform(this->outputErrorsFromSkipLayer().begin(),
			  (this->outputErrorsFromSkipLayer().begin() + 
			   this->curMaxSeqLength() * this->parallelSequences() * this->size()),
			  this->outputErrors().begin(),
			  this->outputErrors().begin(),
			  thrust::plus<real_t>()
			  );

	// send erros to the all the previous layers
	BOOST_REVERSE_FOREACH (Layer<TDevice> *layer, this->PreLayers()) {
	    SkipLayer<TDevice>* tempLayer = dynamic_cast<SkipLayer<TDevice>*>(layer);
	    if(tempLayer){
		// this is an SkipAdd Layer, erros should be accumulated to 
		// this->outputErrorsFromSkipLayer()
		thrust::transform(this->outputErrors().begin(),
				  (this->outputErrors().begin() +
				   this->curMaxSeqLength()      * 
				   this->parallelSequences()    * this->size()),
				  tempLayer->outputErrorsFromSkipLayer().begin(),
				  tempLayer->outputErrorsFromSkipLayer().begin(),
				  thrust::plus<real_t>()
				  );
	    }else{
		// else, just copy the data to the outputErrors
		thrust::copy(this->outputErrors().begin(),
			     (this->outputErrors().begin() + 
			      this->curMaxSeqLength() * this->parallelSequences() * this->size()),
			     layer->outputErrors().begin());
	    }
	    
	}
    }

    // NN backward
    template <typename TDevice>
    void SkipForFeatTrans<TDevice>::computeBackwardPass(const int timeStep, const int nnState)
    {
	if (this->getSaveMemoryFlag())
	    throw std::runtime_error("Memory save mode should be turned off");

	// absolute time
	int effTimeS = timeStep     * this->parallelSequences();
	int effTimeE = (timeStep+1) * this->parallelSequences();

	// at first, add the errors in both this->outputErrorsFromSkipLayer() and m_outputErrors
	thrust::transform(this->outputErrorsFromSkipLayer().begin() + this->size() * effTimeS,
			  this->outputErrorsFromSkipLayer().begin() + this->size() * effTimeE,
			  this->outputErrors().begin()              + this->size() * effTimeS,
			  this->outputErrors().begin()              + this->size() * effTimeS,
			  thrust::plus<real_t>());


	// send erros to the all the previous layers
	BOOST_REVERSE_FOREACH (Layer<TDevice> *layer, this->PreLayers()) {
	    SkipLayer<TDevice>* tempLayer = dynamic_cast<SkipLayer<TDevice>*>(layer);
	    if(tempLayer){
		// this is an SkipAdd Layer, erros should be accumulated to 
		// this->outputErrorsFromSkipLayer()
		thrust::transform(
		     this->outputErrors().begin()  + effTimeS * this->size(),
		     this->outputErrors().begin()  + effTimeE * this->size(),
		     tempLayer->outputErrorsFromSkipLayer().begin() + effTimeS * this->size(),
		     tempLayer->outputErrorsFromSkipLayer().begin() + effTimeS * this->size(),
		     thrust::plus<real_t>());
	    }else{
		// else, just copy the data to the outputErrors
		thrust::copy(this->outputErrors().begin()  + effTimeS * this->size(),
			     this->outputErrors().begin()  + effTimeE * this->size(),
			     layer->outputErrors().begin() + effTimeS * this->size());
	    }   
	}
    }

    // return all the preceding layers
    // template <typename TDevice>
    // std::vector<Layer<TDevice>*> SkipForFeatTrans<TDevice>::PreLayers()
    // {
    // return this->PreLayers();
    //}

    template <typename TDevice>
    typename SkipForFeatTrans<TDevice>::real_vector& SkipForFeatTrans<TDevice>::outputFromGate()
    {	
	return SkipLayer<TDevice>::outputFromGate();
    }
    
    template <typename TDevice>
    const std::string& SkipForFeatTrans<TDevice>::type() const
    {
	// Fatal Error, one instance can only use one static string
	// This is different from the case in Feedforward/SkipPara
	static std::string s1("feattrans");
        return s1;
    }

    template <typename TDevice>
    void SkipForFeatTrans<TDevice>::exportLayer(const helpers::JsonValue &layersArray,
					 const helpers::JsonAllocator &allocator) const
    {
	SkipLayer<TDevice>::exportLayer(layersArray, allocator);
    }

    template <typename TDevice>
    void SkipForFeatTrans<TDevice>::reduceOutputBuffer()
    {
	if (m_virtualLayer){
	    // this->clearOutputBuffer() // this has been done
	}else{
	    this->resizeOutputBuffer(this->parallelSequences() * this->size());
	    this->setSaveMemoryFlag(true);
	    printf("\t[mem saved]");
	}
    }

    template <typename TDevice>
    int SkipForFeatTrans<TDevice>::outputBufPtrBias(const int timeStepTimesParallel,
						    const int nnState)
    {
	if (m_virtualLayer){
	    return this->precedingLayer().outputBufPtrBias(timeStepTimesParallel, nnState);
	}else if (this->getSaveMemoryFlag()){
	    return timeStepTimesParallel * this->size();
	}else{
	    return 0;
	}
    }

    template <typename TDevice>
    void SkipForFeatTrans<TDevice>::clearAllBuffers()
    {
	if (m_virtualLayer){

	}else{
	    this->clearOutputBuffer();	    
	}
    }

    template <typename TDevice>
    void SkipForFeatTrans<TDevice>::resizeAllBuffers(const int timeLength)
    {
	if (m_virtualLayer){
	    
	}else{
	    this->resizeOutputBuffer(timeLength * this->parallelSequences() * this->size());
	}	
    }    


    template <typename TDevice>
    void SkipForFeatTrans<TDevice>::logAllBuffers(helpers::vecPoolManager<TDevice> &vecPoolMng,
					      bool flag_add)
    {
	if (!m_virtualLayer){
	    Layer<TDevice>::logAllBuffers(vecPoolMng, flag_add);
	}
    }

    template <typename TDevice>
    void SkipForFeatTrans<TDevice>::swapAllBuffers(helpers::vecPoolManager<TDevice> &vecPoolMng,
					       bool flag_get)
    {
	if (!m_virtualLayer){
	    Layer<TDevice>::swapAllBuffers(vecPoolMng, flag_get);
	}	
    }    


    
    template class SkipForFeatTrans<Cpu>;
    template class SkipForFeatTrans<Gpu>;
    
}
