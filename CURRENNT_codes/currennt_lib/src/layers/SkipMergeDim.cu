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


#include "SkipMergeDim.hpp"
#include "../helpers/getRawPointer.cuh"
#include "../helpers/Matrix.hpp"
#include "../helpers/JsonClasses.hpp"
#include "../helpers/misFuncs.hpp"
#include "../activation_functions/Logistic.cuh"

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

    
    struct ComputeLayerOutputFn
    {
	int feature_dim;
	real_t *signal_1;
	real_t *signal_2;

	__host__ __device__ void  operator() (
					      const thrust::tuple<real_t&, int> &t) const
	{
	    // t.get<0>() out[ut
	    // t.get<1>() time index
	    t.get<0>() = 0.0;
	    for (int dim_idx = 0; dim_idx < feature_dim; dim_idx++)
		t.get<0>() += signal_1[t.get<1>() * feature_dim + dim_idx] *
		    signal_2[t.get<1>() * feature_dim + dim_idx];
	    
	}
    };

    struct ComputeLayerOutputGradFn
    {

	int feature_dim;
	real_t *grad_w;
	real_t *grad_x;
	real_t *input_error;
	
	const char *patTypes;
	
	__host__ __device__ void  operator() (
	      const thrust::tuple<const real_t&, const real_t&, int> &t) const
	{

	    int timeIdx = t.get<2>() / feature_dim;
	    
	    if (patTypes[timeIdx] == PATTYPE_NONE){
		grad_w[t.get<2>()] = 0.0;
		grad_x[t.get<2>()] = 0.0;
	    }else{
		grad_w[t.get<2>()] = input_error[timeIdx] * t.get<1>();
		grad_x[t.get<2>()] = t.get<0>() * input_error[timeIdx];
	    }
	}
    };

    
}
}


namespace layers{

    // Construct the layer
    template <typename TDevice>
    SkipMergeDim<TDevice>::SkipMergeDim(
					const helpers::JsonValue &layerChild,
					const helpers::JsonValue &weightsSection,
					std::vector<Layer<TDevice>*> &precedingLayers,
					int maxSeqLength,
					int layerID)
	// use preLayers[0] as fake preceding layers
	: SkipLayer<TDevice>(layerChild, weightsSection, precedingLayers,
			     maxSeqLength, layerID, false)
    {
	
	// Link previous layers
	m_previousSkipStr = (layerChild->HasMember("preSkipLayer") ? 
			     ((*layerChild)["preSkipLayer"].GetString()) : "");

	
	// Initial check
	if (precedingLayers.size() < 2)
	    throw std::runtime_error("Error: skipMergeDim needs 2 skip layers");
	if (m_previousSkipStr.size() == 0)
	    throw std::runtime_error("Error: skipMergeDim requires \"preSkipLayer\"");

	// previous layers are specified by preSkipLayer
	std::vector<std::string> tmpOpt;
	misFuncs::ParseStrOpt(m_previousSkipStr, tmpOpt, ",");
	if (tmpOpt.size() != 2)
	    throw std::runtime_error("Error: \"preSkipLayer\" should only contain 2 layers");

	// push the layers to be merged
	for (int cnt = 0 ; cnt < tmpOpt.size(); cnt++) {
	    BOOST_FOREACH (Layer<TDevice> *layer, precedingLayers) {
		if (layer->name() == tmpOpt[cnt]){
		    this->PreLayers().push_back(layer);
		    break;
		}
	    }
	}

	if (this->size() != 1){
	    printf("Error: skipMergeDim only supports size = 1\n");
	    throw std::runtime_error("Error: please check size in skipMergeDim layer");
	}
	    
	if (this->PreLayers().size() != 2){
	    printf("Error: skipMergeDim cannot find the two input layers\n");
	    throw std::runtime_error("Error: please check \"preSkipLayer\"");
	}

	if (this->PreLayers()[0]->size() != this->PreLayers()[1]->size()){
	    printf("Error: skipMergeDim's two input layers have different layer sizes\n");
	    throw std::runtime_error("Error: please check \"preSkipLayer\"");
	}

	if (this->PreLayers()[0]->getResolution() != this->PreLayers()[1]->getResolution() ||
	    this->PreLayers()[0]->getResolution() != this->getResolution()){
	    printf("Error: skipMergeDim's two input layers have different resolutions\n");
	    throw std::runtime_error("Error: please check and \"preSkipLayer\"");
	}

    }	

    // Destructor
    template <typename TDevice>
    SkipMergeDim<TDevice>::~SkipMergeDim()
    {
    }

    
    // NN forward
    template <typename TDevice>
    void SkipMergeDim<TDevice>::computeForwardPass(const int nnState)
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

	{
	    int n = this->curMaxSeqLength() * this->parallelSequences() * this->size();
	    
	    internal::ComputeLayerOutputFn fn;
	    fn.feature_dim = this->PreLayers()[0]->size();
	    fn.signal_1 = helpers::getRawPointer(this->PreLayers()[0]->outputs());
	    fn.signal_2 = helpers::getRawPointer(this->PreLayers()[1]->outputs());
	    
	    thrust::for_each(
	       thrust::make_zip_iterator(
					 thrust::make_tuple(this->outputs().begin(),
							    thrust::counting_iterator<int>(0))),
	       thrust::make_zip_iterator(
					 thrust::make_tuple(this->outputs().begin() + n,
							    thrust::counting_iterator<int>(0) + n)),
	       fn);
	}
    }

    // NN forward
    template <typename TDevice>
    void SkipMergeDim<TDevice>::computeForwardPass(const int timeStep, const int nnState)
    {
	// absolute time
	int effTimeS = timeStep     * this->parallelSequences();
	int effTimeE = (timeStep+1) * this->parallelSequences();
	
	// shift of the pointer to the data 
	int shiftIn  = 0; // value to assigned layer
	int shiftOut = this->outputBufPtrBias(timeStep * this->parallelSequences(), nnState);
	
	{
	    internal::ComputeLayerOutputFn fn;
	    fn.feature_dim = this->PreLayers()[0]->size();
	    fn.signal_1 = helpers::getRawPointer(this->PreLayers()[0]->outputs());
	    fn.signal_2 = helpers::getRawPointer(this->PreLayers()[1]->outputs());
	    
	    thrust::for_each(
	       thrust::make_zip_iterator(
		  thrust::make_tuple(
			this->_outputs().begin()  + effTimeS * this->size() - shiftIn,
			thrust::counting_iterator<int>(0) + effTimeS * this->size() - shiftIn)),
	       thrust::make_zip_iterator(
		  thrust::make_tuple(
			this->_outputs().begin() + effTimeS * this->size() - shiftOut,
			thrust::counting_iterator<int>(0) + effTimeS * this->size() - shiftOut)),
	       fn);
	}
    }


    // NN backward
    template <typename TDevice>
    void SkipMergeDim<TDevice>::computeBackwardPass(const int nnState)
    {
	if (this->getSaveMemoryFlag())
	    throw std::runtime_error("Memory save mode should be turned off");
		
	// at first, add the errors in both this->outputErrorsFromSkipLayer() and m_outputErrors
	thrust::transform(this->outputErrorsFromSkipLayer().begin(),
			  (this->outputErrorsFromSkipLayer().begin() + 
			   this->curMaxSeqLength() * this->parallelSequences() * this->size()),
			  this->outputErrors().begin(),
			  this->outputErrors().begin(),
			  thrust::plus<real_t>());


	{
	    SkipLayer<TDevice>* tempLayer = NULL;
	    
	    internal::ComputeLayerOutputGradFn fn;
	    fn.feature_dim = this->size();
	    fn.input_error = helpers::getRawPointer(this->outputErrors());
	    fn.patTypes    = helpers::getRawPointer(this->patTypes());
		    
	    tempLayer = dynamic_cast<SkipLayer<TDevice>*>(this->PreLayers()[0]);
	    if (tempLayer)
		fn.grad_w = helpers::getRawPointer(tempLayer->outputErrorsFromSkipLayer());
	    else
		fn.grad_w = helpers::getRawPointer(this->PreLayers()[0]->outputErrors());

	    tempLayer = dynamic_cast<SkipLayer<TDevice>*>(this->PreLayers()[1]);
	    if (tempLayer)
		fn.grad_x = helpers::getRawPointer(tempLayer->outputErrorsFromSkipLayer());
	    else
		fn.grad_x = helpers::getRawPointer(this->PreLayers()[1]->outputErrors());


	    int n = this->curMaxSeqLength() * this->parallelSequences() *
		this->PreLayers()[0]->size();

	    thrust::for_each(
	       thrust::make_zip_iterator(
		  thrust::make_tuple(
		     this->PreLayers()[0]->outputs().begin(),   
		     this->PreLayers()[1]->outputs().begin(),
		     thrust::counting_iterator<int>(0))),
	       thrust::make_zip_iterator(
		  thrust::make_tuple(
		     this->PreLayers()[0]->outputs().begin() + n,   
		     this->PreLayers()[1]->outputs().begin() + n,
		     thrust::counting_iterator<int>(0)       + n)),
	       fn);
	}
	
    }

    // NN backward
    template <typename TDevice>
    void SkipMergeDim<TDevice>::computeBackwardPass(const int timeStep, const int nnState)
    {
	if (this->getSaveMemoryFlag())
	    throw std::runtime_error("Memory save mode should be turned off");

	int effTimeS = timeStep     * this->parallelSequences();
	int effTimeE = (timeStep+1) * this->parallelSequences();

	// at first, add the errors in both this->outputErrorsFromSkipLayer() and m_outputErrors
	thrust::transform(this->outputErrorsFromSkipLayer().begin() + this->size() * effTimeS,
			  this->outputErrorsFromSkipLayer().begin() + this->size() * effTimeE,
			  this->outputErrors().begin()              + this->size() * effTimeS,
			  this->outputErrors().begin()              + this->size() * effTimeS,
			  thrust::plus<real_t>());

	{
	    SkipLayer<TDevice>* tempLayer = NULL;
	    
	    internal::ComputeLayerOutputGradFn fn;
	    fn.feature_dim = this->size();
	    fn.input_error = helpers::getRawPointer(this->outputErrors());
	    fn.patTypes    = helpers::getRawPointer(this->patTypes());
		    
	    tempLayer = dynamic_cast<SkipLayer<TDevice>*>(this->PreLayers()[0]);
	    if (tempLayer)
		fn.grad_w = helpers::getRawPointer(tempLayer->outputErrorsFromSkipLayer());
	    else
		fn.grad_w = helpers::getRawPointer(this->PreLayers()[0]->outputErrors());

	    tempLayer = dynamic_cast<SkipLayer<TDevice>*>(this->PreLayers()[1]);
	    if (tempLayer)
		fn.grad_x = helpers::getRawPointer(tempLayer->outputErrorsFromSkipLayer());
	    else
		fn.grad_x = helpers::getRawPointer(this->PreLayers()[1]->outputErrors());

	    thrust::for_each(
	       thrust::make_zip_iterator(
		  thrust::make_tuple(
		     this->PreLayers()[0]->outputs().begin() + effTimeS * this->size(),   
		     this->PreLayers()[1]->outputs().begin() + effTimeS * this->size(),
		     thrust::counting_iterator<int>(0)       + effTimeS * this->size())),
	       thrust::make_zip_iterator(
		  thrust::make_tuple(
		     this->PreLayers()[0]->outputs().begin() + effTimeE * this->size(),   
		     this->PreLayers()[1]->outputs().begin() + effTimeE * this->size(),
		     thrust::counting_iterator<int>(0)       + effTimeE * this->size())),
	       fn);
	}
	
    }

    template <typename TDevice>
    typename SkipMergeDim<TDevice>::real_vector& SkipMergeDim<TDevice>::outputFromGate()
    {	
	return SkipLayer<TDevice>::outputFromGate();
    }
    
    template <typename TDevice>
    const std::string& SkipMergeDim<TDevice>::type() const
    {
	static std::string s1("weighted_merge");
	return s1;
    }

    template <typename TDevice>
    void SkipMergeDim<TDevice>::exportLayer(const helpers::JsonValue &layersArray,
					 const helpers::JsonAllocator &allocator) const
    {
	SkipLayer<TDevice>::exportLayer(layersArray, allocator);
	if (m_previousSkipStr.size())
	    (*layersArray)[layersArray->Size() - 1].AddMember("preSkipLayer",
							      m_previousSkipStr.c_str(), allocator);	
    }
    
    template <typename TDevice>
    void SkipMergeDim<TDevice>::reduceOutputBuffer()
    {
	this->resizeOutputBuffer(this->parallelSequences() * this->size());
	this->setSaveMemoryFlag(true);
	printf("\t[mem saved]");
    }

    template <typename TDevice>
    int SkipMergeDim<TDevice>::outputBufPtrBias(const int timeStepTimesParallel,
						      const int nnState)
    {
	if (this->getSaveMemoryFlag()){
	    return timeStepTimesParallel * this->size();
	}else{
	    return 0;
	}
    }

    template <typename TDevice>
    void SkipMergeDim<TDevice>::clearAllBuffers()
    {
	this->clearOutputBuffer();	    
    }

    template <typename TDevice>
    void SkipMergeDim<TDevice>::resizeAllBuffers(const int timeLength)
    {
	this->resizeOutputBuffer(timeLength * this->parallelSequences() * this->size());	
    }    

    template <typename TDevice>
    void SkipMergeDim<TDevice>::logAllBuffers(helpers::vecPoolManager<TDevice> &vecPoolMng,
					      bool flag_add)
    {
	Layer<TDevice>::logAllBuffers(vecPoolMng, flag_add);
    }

    template <typename TDevice>
    void SkipMergeDim<TDevice>::swapAllBuffers(helpers::vecPoolManager<TDevice> &vecPoolMng,
					       bool flag_get)
    {
	Layer<TDevice>::swapAllBuffers(vecPoolMng, flag_get);
    }
    
    template class SkipMergeDim<Cpu>;
    template class SkipMergeDim<Gpu>;
    
}
