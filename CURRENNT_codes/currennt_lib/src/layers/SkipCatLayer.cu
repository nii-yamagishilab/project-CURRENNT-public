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


#include "SkipCatLayer.hpp"
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


namespace internal{
namespace{

    // copy segment
    struct CopyPartSkipCat
    {
	real_t *source;
	real_t *target;

	int srcDim;
	int srcS;     // the first dimension to be copied in source stream

	int copyDim;  // dimension of the data to be copied

	int tarDim;
	int tarS;     // the first dimension to store the copied data in target stream

	int srcShiftT;
	int tarShiftT;
	
	const char *patTypes;     // 

	bool accumulate;
	__host__ __device__ void operator() (const thrust::tuple<real_t&, int> &t) const
	{
	    int outputIdx = t.get<1>();
	    int timeIdx   = outputIdx / copyDim;
	    int dimIdx    = outputIdx % copyDim;

	    // skip dummy frame (for parallel sentence processing)
	    if (patTypes != NULL && patTypes[timeIdx] == PATTYPE_NONE){
		if (!accumulate)
		    target[timeIdx * tarDim + tarS + dimIdx] = 0;
		return;
	    }

	    if (accumulate)
		target[(timeIdx - tarShiftT) * tarDim + tarS + dimIdx] +=
		    source[(timeIdx - srcShiftT) * srcDim + srcS + dimIdx];
	    else
		target[(timeIdx - tarShiftT) * tarDim + tarS + dimIdx]  =
		    source[(timeIdx - srcShiftT) * srcDim + srcS + dimIdx];
	}
    };

}
}


namespace layers{

    // Construct the layer
    template <typename TDevice>
    SkipCatLayer<TDevice>::SkipCatLayer(
					const helpers::JsonValue &layerChild,
					const helpers::JsonValue &weightsSection,
					std::vector<Layer<TDevice>*> &precedingLayers,
					int maxSeqLength,
					int layerID)
	// use preLayers[0] as fake preceding layers
	: SkipLayer<TDevice>(layerChild, weightsSection, precedingLayers,
			     maxSeqLength, layerID, false)
    {
	// initialization
	this->PreLayers().clear();
	m_preSkipDim.clear();
	m_preSkipDimAccu.clear();

	
	// Link previous Skip layer
	if (precedingLayers.size() < 1)
	    throw std::runtime_error("Error no precedinglayers in skipadd/skipini");
	
	m_previousSkipStr = (layerChild->HasMember("preSkipLayer") ? 
			     ((*layerChild)["preSkipLayer"].GetString()) : "");
	
	if (m_previousSkipStr.size()){
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
	    
	    /*
	    boost::iterator_range<std::string::iterator> r;
	    BOOST_FOREACH (Layer<TDevice> *layer, precedingLayers) {
		r = boost::find_first(m_previousSkipStr, layer->name());
		if (r)
		    this->PreLayers().push_back(layer);
		    }
	    if (this->PreLayers().size() == 0 ||
		this->PreLayers().back()->name() != precedingLayers.back()->name()){
		// if the string doesn't specify the defaul previous layer
		this->PreLayers().push_back(precedingLayers.back());
		}*/
	}else{
	    throw std::runtime_error("Error: no preSkipLayer configuration in skipcat");
	}
	
	// Set the dimension to be catenated
	m_preSkipDim.clear();
	m_preSkipDimStr = (layerChild->HasMember("preSkipLayerDim") ? 
				((*layerChild)["preSkipLayerDim"].GetString()) : "");
	if (m_preSkipDimStr.size()){
	    misFuncs::ParseIntOpt(m_preSkipDimStr, m_preSkipDim);
	}else{
	    // default case, concatenate all the previous layers
	    m_preSkipDim.resize(this->PreLayers().size() * 2, 0);
	    int tmp = 1;
	    BOOST_FOREACH (Layer<TDevice> *layer, this->PreLayers()) {
		m_preSkipDim[tmp] = layer->size();
		tmp +=2;
	    }
	}
	
	// Check
	if (m_preSkipDim.size() != this->PreLayers().size() * 2)
	    throw std::runtime_error("Error: one in preSkiplayer, two in preSkipLayerDim");

	m_preSkipDimAccu.resize(this->PreLayers().size(), 0);
	int tmpDim = 0;
	for (int cnt =0 ; cnt < this->PreLayers().size(); cnt ++){
	    if (m_preSkipDim[2*cnt + 1] > this->PreLayers()[cnt]->size() ||
		m_preSkipDim[2*cnt] < 0 ||
		m_preSkipDim[2*cnt + 1] < m_preSkipDim[2*cnt]){
		printf("Error preSkipLayerDim for %s\n", this->PreLayers()[cnt]->name().c_str());
		throw std::runtime_error("Error: preSkipLayerDim configuration");
	    }
	    m_preSkipDimAccu[cnt] = tmpDim;
	    tmpDim += (m_preSkipDim[2*cnt + 1] - m_preSkipDim[2*cnt]);

	    if (this->getResolution() != this->PreLayers()[cnt]->getResolution()){
		throw std::runtime_error("SkipCat need layers with same resolution");
	    }
	}
	if (tmpDim != this->size()){
	    printf("Layer size %d, but sum of input size is %d\n", this->size(), tmpDim);
	    throw std::runtime_error("Error: layer size");
	}
	
	printf("\n\tReceive input from layer(s):");
	int cnt = 0;
	BOOST_FOREACH (Layer<TDevice> *layer, this->PreLayers()) {
	    printf(" %s [%d %d],", layer->name().c_str(), m_preSkipDim[cnt], m_preSkipDim[cnt+1]);
	    cnt += 2;
	}
	printf("\n");


    }	

    // Destructor
    template <typename TDevice>
    SkipCatLayer<TDevice>::~SkipCatLayer()
    {
    }
	
    // NN forward
    template <typename TDevice>
    void SkipCatLayer<TDevice>::computeForwardPass(const int nnState)
    {
	// initialization
	thrust::fill(this->outputs().begin(), 
		     (this->outputs().begin() + 
		      this->curMaxSeqLength() * this->parallelSequences() * this->size()),
		     0.0);
	
	// initialization for backward pass
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

	// accumulating the outputs of previous layers
	{{
	    internal::CopyPartSkipCat fn;
	    fn.target   = helpers::getRawPointer(this->outputs());
	    fn.tarDim   = this->size();
	    fn.srcShiftT= 0;
	    fn.tarShiftT= 0;
	    
	    fn.patTypes = helpers::getRawPointer(this->patTypes());
	    fn.accumulate = false;
	    
	    int n = this->curMaxSeqLength();
 	    n = n * this->parallelSequences();

	    int cnt = 0;
	    BOOST_FOREACH (Layer<TDevice> *layer, this->PreLayers()) {
		fn.tarS    = m_preSkipDimAccu[cnt/2];
		fn.source  = helpers::getRawPointer(layer->outputs());
		fn.srcDim  = layer->size();
		fn.srcS    = m_preSkipDim[cnt];
		fn.copyDim = m_preSkipDim[cnt+1] - m_preSkipDim[cnt];
		thrust::for_each(
		   thrust::make_zip_iterator(
			thrust::make_tuple(this->outputs().begin(), 
					   thrust::counting_iterator<int>(0))),
		   thrust::make_zip_iterator(
			thrust::make_tuple(this->outputs().begin() + n * fn.copyDim, 
					   thrust::counting_iterator<int>(0) + n * fn.copyDim)),
		fn);
		cnt += 2;
	    }
	}}
    }

    // NN forward
    template <typename TDevice>
    void SkipCatLayer<TDevice>::computeForwardPass(const int timeStep, const int nnState)
    {
	int effTimeS = timeStep     * this->parallelSequences();
	int effTimeE = (timeStep+1) * this->parallelSequences();

	// shifted time
	int shiftInTime  = 0; 
	int shiftOutTime = this->outputBufPtrBias(timeStep * this->parallelSequences(), nnState) /
	    this->size();

	// initialization without noise
	thrust::fill(this->outputs().begin() + effTimeS * this->size(), 
		     this->outputs().begin() + effTimeE * this->size(), 
		     0.0);

	// initialize the gradients
	if (timeStep == 0 && this->flagTrainingMode()){
	    if (this->getSaveMemoryFlag())
		throw std::runtime_error("Memory save mode should not be used in training");
	    thrust::fill(this->outputErrors().begin(), 
			 (this->outputErrors().begin() + 
			  this->curMaxSeqLength() * this->parallelSequences() * this->size()),
			 0.0);
	    thrust::fill(this->outputErrorsFromSkipLayer().begin(),
			 (this->outputErrorsFromSkipLayer().begin() + 
			  this->curMaxSeqLength() * this->parallelSequences() * this->size()),
			 0.0);
	}

	
	{{
	    internal::CopyPartSkipCat fn;
	    fn.target   = helpers::getRawPointer(this->outputs());
	    fn.tarDim   = this->size();
	    fn.tarShiftT= shiftOutTime;
	    
	    fn.patTypes = helpers::getRawPointer(this->precedingLayer().patTypes());
	    fn.accumulate = false;
	    
	    int cnt = 0;
	    BOOST_FOREACH (Layer<TDevice> *layer, this->PreLayers()) {

		shiftInTime = layer->outputBufPtrBias(timeStep*this->parallelSequences(), nnState)/
		    layer->size();
		
		fn.tarS     = m_preSkipDimAccu[cnt/2];
		fn.source   = helpers::getRawPointer(layer->outputs());
		fn.srcDim   = layer->size();
		fn.srcS     = m_preSkipDim[cnt];
		fn.srcShiftT= shiftInTime;
		fn.copyDim  = m_preSkipDim[cnt+1] - m_preSkipDim[cnt];
		
		thrust::for_each(
		  thrust::make_zip_iterator(
		     thrust::make_tuple(
		       this->outputs().begin()           + (effTimeS - shiftOutTime) * fn.copyDim, 
		       thrust::counting_iterator<int>(0) + effTimeS * fn.copyDim)),
		  thrust::make_zip_iterator(
		    thrust::make_tuple(
		       this->outputs().begin()           + (effTimeE - shiftOutTime) * fn.copyDim, 
		       thrust::counting_iterator<int>(0) + effTimeE * fn.copyDim)),
		fn);
		cnt += 2;
	    }
	}}

    }


    // NN backward
    template <typename TDevice>
    void SkipCatLayer<TDevice>::computeBackwardPass(const int nnState)
    {
	// 
	// at first, add the errors in both this->outputErrorsFromSkipLayer() and m_outputErrors
	thrust::transform(this->outputErrorsFromSkipLayer().begin(),
			  (this->outputErrorsFromSkipLayer().begin() + 
			   this->curMaxSeqLength() * this->parallelSequences() * this->size()),
			  this->outputErrors().begin(),
			  this->outputErrors().begin(),
			  thrust::plus<real_t>()
			  );

	// send erros to the all the previous layers
	{{
	    internal::CopyPartSkipCat fn;
	    fn.source   = helpers::getRawPointer(this->outputErrors());
	    fn.srcDim   = this->size();
	    fn.patTypes = helpers::getRawPointer(this->precedingLayer().patTypes());
	    fn.srcShiftT= 0;
	    fn.tarShiftT= 0;
	    
	    int n = this->precedingLayer().curMaxSeqLength();
 	    n = n * this->precedingLayer().parallelSequences();

	    int cnt  = 0;
	    BOOST_FOREACH (Layer<TDevice> *layer, this->PreLayers()) {

		int tmpEle = n * (m_preSkipDim[cnt+1] - m_preSkipDim[cnt]);
		fn.srcS    = m_preSkipDimAccu[cnt/2];
		fn.tarS    = m_preSkipDim[cnt];
		fn.copyDim = m_preSkipDim[cnt+1] - m_preSkipDim[cnt];
		
		SkipLayer<TDevice>* tempLayer = dynamic_cast<SkipLayer<TDevice>*>(layer);
		if(tempLayer){
		    fn.target  = helpers::getRawPointer(tempLayer->outputErrorsFromSkipLayer());
		    fn.tarDim  = tempLayer->size();
		    fn.accumulate = true;
		}else{
		    // if previous layer is not a SkipLayer, its gradients only comes
		    // from this layer, thus, it is safe to set the gradients to zeros
		    thrust::fill(layer->outputErrors().begin(),
				 layer->outputErrors().end(), 0.0);
		    fn.target  = helpers::getRawPointer(layer->outputErrors());
		    fn.tarDim  = layer->size();
		    fn.accumulate = false;
		}
		thrust::for_each(
			thrust::make_zip_iterator(
				thrust::make_tuple(this->outputErrors().begin(), 
						   thrust::counting_iterator<int>(0))),
			thrust::make_zip_iterator(
			        thrust::make_tuple(this->outputErrors().begin()      + tmpEle, 
						   thrust::counting_iterator<int>(0) + tmpEle)),
			fn);
		cnt += 2;
	    }
       }}
    }

    // NN backward
    template <typename TDevice>
    void SkipCatLayer<TDevice>::computeBackwardPass(const int timeStep, const int nnState)
    {
	if (this->getSaveMemoryFlag())
	    throw std::runtime_error("Memory save mode should be turned off");
	
	// absolute time
	int effTimeS = timeStep     * this->parallelSequences();
	int effTimeE = (timeStep+1) * this->parallelSequences();

	// 
	// at first, add the errors in both this->outputErrorsFromSkipLayer() and m_outputErrors
	thrust::transform(this->outputErrorsFromSkipLayer().begin() + this->size() * effTimeS,
			  this->outputErrorsFromSkipLayer().begin() + this->size() * effTimeE,
			  this->outputErrors().begin()              + this->size() * effTimeS,
			  this->outputErrors().begin()              + this->size() * effTimeS,
			  thrust::plus<real_t>());
	
	// send erros to the all the previous layers
	{{
	    internal::CopyPartSkipCat fn;
	    fn.source    = helpers::getRawPointer(this->outputErrors());
	    fn.srcDim    = this->size();
	    fn.patTypes  = helpers::getRawPointer(this->precedingLayer().patTypes());
	    fn.srcShiftT = 0;
	    fn.tarShiftT = 0;
	    
	    int n = this->precedingLayer().curMaxSeqLength();
 	    n = n * this->precedingLayer().parallelSequences();

	    int cnt  = 0;
	    int tmpDim = 0;
	    BOOST_FOREACH (Layer<TDevice> *layer, this->PreLayers()) {

		tmpDim = m_preSkipDim[cnt+1] - m_preSkipDim[cnt];
		
		fn.srcS    = m_preSkipDimAccu[cnt/2];
		fn.tarS    = m_preSkipDim[cnt];
		fn.copyDim = m_preSkipDim[cnt+1] - m_preSkipDim[cnt];
		
		SkipLayer<TDevice>* tempLayer = dynamic_cast<SkipLayer<TDevice>*>(layer);
		if(tempLayer){
		    fn.target  = helpers::getRawPointer(tempLayer->outputErrorsFromSkipLayer());
		    fn.tarDim  = tempLayer->size();
		    fn.accumulate = true;
		    
		}else{
		    // if previous layer is not a SkipLayer, its gradients only comes
		    // from this layer, thus, it is safe to set the gradients to zeros
		    thrust::fill(layer->outputErrors().begin() + layer->size() * effTimeS,
				 layer->outputErrors().begin() + layer->size() * effTimeE,
				 0.0);
		    
		    fn.target  = helpers::getRawPointer(layer->outputErrors());
		    fn.tarDim  = layer->size();
		    fn.accumulate = false;
		}
		thrust::for_each(
		  thrust::make_zip_iterator(
		     thrust::make_tuple(this->outputErrors().begin()       + tmpDim * effTimeS, 
					thrust::counting_iterator<int>(0)  + tmpDim * effTimeS)),
		  thrust::make_zip_iterator(
		      thrust::make_tuple(this->outputErrors().begin()      + tmpDim * effTimeE, 
					 thrust::counting_iterator<int>(0) + tmpDim * effTimeE)),
		  fn);
		cnt += 2;
	    }
       }}
    }

    // return all the preceding layers
    // template <typename TDevice>
    // std::vector<Layer<TDevice>*> SkipCatLayer<TDevice>::PreLayers()
    //{
    //	return this->PreLayers();
    //}

    template <typename TDevice>
    typename SkipCatLayer<TDevice>::real_vector& SkipCatLayer<TDevice>::outputFromGate()
    {
	
	return SkipLayer<TDevice>::outputFromGate();
    }
    
    template <typename TDevice>
    const std::string& SkipCatLayer<TDevice>::type() const
    {
	static std::string s;
	s = "skipcat";
        return s;
    }

    template <typename TDevice>
    void SkipCatLayer<TDevice>::exportLayer(const helpers::JsonValue &layersArray,
					 const helpers::JsonAllocator &allocator) const
    {
	SkipLayer<TDevice>::exportLayer(layersArray, allocator);
	if (m_previousSkipStr.size())
	    (*layersArray)[layersArray->Size() - 1].AddMember("preSkipLayer",
							      m_previousSkipStr.c_str(), allocator);
	if (m_preSkipDimStr.size())
	    (*layersArray)[layersArray->Size() - 1].AddMember("preSkipLayerDim",
							      m_preSkipDimStr.c_str(), allocator);

    }

    template <typename TDevice>
    void SkipCatLayer<TDevice>::reduceOutputBuffer()
    {
	this->resizeOutputBuffer(this->parallelSequences() * this->size());
	this->setSaveMemoryFlag(true);
	printf("\t[mem saved]");
    }

    template <typename TDevice>
    int SkipCatLayer<TDevice>::outputBufPtrBias(const int timeStepTimesParallel, const int nnState)
    {
	if (this->getSaveMemoryFlag()){
	    return timeStepTimesParallel * this->size();
	}else{
	    return 0;
	}
    }

    /*
    template <typename TDevice>
    typename SkipCatLayer<TDevice>::real_vector& SkipCatLayer<TDevice>::outputErrorsFromSkipLayer()
    {
        return this->outputErrorsFromSkipLayer();
    }
    */

    template <typename TDevice>
    void SkipCatLayer<TDevice>::clearAllBuffers()
    {
	this->clearOutputBuffer();	    
    }

    template <typename TDevice>
    void SkipCatLayer<TDevice>::resizeAllBuffers(const int timeLength)
    {
	this->resizeOutputBuffer(timeLength * this->parallelSequences() * this->size());
    }    

    template <typename TDevice>
    void SkipCatLayer<TDevice>::logAllBuffers(helpers::vecPoolManager<TDevice> &vecPoolMng,
					      bool flag_add)
    {
	Layer<TDevice>::logAllBuffers(vecPoolMng, flag_add);
    }

    template <typename TDevice>
    void SkipCatLayer<TDevice>::swapAllBuffers(helpers::vecPoolManager<TDevice> &vecPoolMng,
					       bool flag_get)
    {
	Layer<TDevice>::swapAllBuffers(vecPoolMng, flag_get);
    }    

    template class SkipCatLayer<Cpu>;
    template class SkipCatLayer<Gpu>;
    
}
