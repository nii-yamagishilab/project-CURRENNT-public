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


#include "SkipWeightAddLayer.hpp"
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

    // output = sig(z * z_scale + z_shift) * x + (1 - sig(z * z_scale + z_shift)) * y
    struct ComputeLayerOutputFn
    {
	real_t z_shift;
	real_t z_scale;
	__host__ __device__ void  operator() (
	    const thrust::tuple<const real_t&, const real_t&, const real_t&, real_t&> &t) const
	{
	    // t.get<0>() x
	    // t.get<1>() y
	    // t.get<2>() z
	    // t.get<3>() out
	    real_t sigz = activation_functions::Logistic::fn(t.get<2>() * z_scale + z_shift);
	    t.get<3>() = sigz * t.get<0>() + (1.0 - sigz) * t.get<1>();
	}
    };

    // grad_x = grad_output * sig(z)
    // grad_y = grad_output * (1 - sig(z))
    // grad_z = grad_output * (x - y) * (1-sig(z)) * sig(z)    
    struct ComputeLayerOutputGradFn
    {
	int layerSize;
	real_t z_shift;
	real_t z_scale;

	const char *patTypes;
	real_t *grad_x;
	real_t *grad_y;
	real_t *grad_z;
	
	__host__ __device__ void  operator() (
	      const thrust::tuple<const real_t&, const real_t&, const real_t&,
	                          const real_t&, int> &t) const
	{
	    // t.get<0>() x
	    // t.get<1>() y
	    // t.get<2>() z
	    // t.get<3>() grad_out
	    // t.get<4>() time index

	    int timeIdx = t.get<4>() / layerSize;

	    if (patTypes[timeIdx] == PATTYPE_NONE){
		grad_x[t.get<4>()] = 0.0;
		grad_y[t.get<4>()] = 0.0;
		grad_z[t.get<4>()] = 0.0;
	    }else{
		real_t sigz = activation_functions::Logistic::fn(t.get<2>() * z_scale + z_shift);
		grad_x[t.get<4>()] = t.get<3>() * sigz;
		grad_y[t.get<4>()] = t.get<3>() * (1.0 - sigz);
		grad_z[t.get<4>()] = t.get<3>() * (t.get<0>() - t.get<1>()) * sigz * (1.0 - sigz)
		    * z_scale;
	    }
	}
    };

    
}
}


namespace layers{

    // Construct the layer
    template <typename TDevice>
    SkipWeightAddLayer<TDevice>::SkipWeightAddLayer(
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

	// Link previous layers
	m_weightLayerStr  = (layerChild->HasMember("preWeightLayer") ? 
			     ((*layerChild)["preWeightLayer"].GetString()) : "");

	
	m_z_scale = (layerChild->HasMember("weightScale") ? 
		     ((*layerChild)["weightScale"].GetDouble()) : 1.0);

	m_z_shift = (layerChild->HasMember("weightShift") ? 
		     ((*layerChild)["weightShift"].GetDouble()) : 0.0);
	
	// Initial check
	if (precedingLayers.size() < 3)
	    throw std::runtime_error("Error: skipweightadd needs 3 skip layers");
	if (m_previousSkipStr.size() == 0)
	    throw std::runtime_error("Error: skipweightadd requires \"preSkipLayer\"");
	if (m_weightLayerStr.size() == 0)
	    throw std::runtime_error("Error: skipweightadd requires \"preWeightLayer\"");

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
	// push the weight layer
	BOOST_FOREACH (Layer<TDevice> *layer, precedingLayers) {
	    if (layer->name() == m_weightLayerStr){
		this->PreLayers().push_back(layer);
		break;
	    }
	}

	if (this->PreLayers().size() != 3){
	    printf("Error: skipweightadd cannot find the three input layers\n");
	    throw std::runtime_error("Error: please check \"preWeightLayer\" and \"preSkipLayer\"");
	}

	if (this->PreLayers()[0]->size() != this->PreLayers()[1]->size() ||
	    this->PreLayers()[2]->size() != this->PreLayers()[1]->size() ||
	    this->PreLayers()[0]->size() != this->size()){
	    printf("Error: skipweightadd's three input layers have different layer sizes\n");
	    throw std::runtime_error("Error: please check \"preWeightLayer\" and \"preSkipLayer\"");
	}

	if (this->PreLayers()[0]->getResolution() != this->PreLayers()[1]->getResolution() ||
	    this->PreLayers()[2]->getResolution() != this->PreLayers()[1]->getResolution() ||
	    this->PreLayers()[0]->getResolution() != this->getResolution()){
	    printf("Error: skipweightadd's three input layers have different resolutions\n");
	    throw std::runtime_error("Error: please check \"preWeightLayer\" and \"preSkipLayer\"");
	}

	
	printf("\n\tThis layer conducts sig(z*%f+%f)*x + (1-sig(z%f+%f))*y, where",
	       m_z_scale, m_z_shift, m_z_scale, m_z_shift);
	printf("\n\tx is given by %s", this->PreLayers()[0]->name().c_str());
	printf("\n\ty is given by %s", this->PreLayers()[1]->name().c_str());
	printf("\n\tz is given by %s", this->PreLayers()[2]->name().c_str());
	printf("\n");

    }	

    // Destructor
    template <typename TDevice>
    SkipWeightAddLayer<TDevice>::~SkipWeightAddLayer()
    {
    }

    
    // NN forward
    template <typename TDevice>
    void SkipWeightAddLayer<TDevice>::computeForwardPass(const int nnState)
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
	
	
	// processing output = sig(z) * x + (1-sig(z)) * y
	{
	    int n = this->curMaxSeqLength() * this->parallelSequences() * this->size();
	    internal::ComputeLayerOutputFn fn;
	    fn.z_shift = m_z_shift;
	    fn.z_scale = m_z_scale;
	    thrust::for_each(
	       thrust::make_zip_iterator(
		  thrust::make_tuple(this->PreLayers()[0]->outputs().begin(),   
				     this->PreLayers()[1]->outputs().begin(),
				     this->PreLayers()[2]->outputs().begin(),    
				     this->_outputs().begin())),
	       thrust::make_zip_iterator(
		  thrust::make_tuple(this->PreLayers()[0]->outputs().begin() + n,   
				     this->PreLayers()[1]->outputs().begin() + n,
				     this->PreLayers()[2]->outputs().begin() + n,    
				     this->_outputs().begin() + n)),
	       fn);
	}
    }

    // NN forward
    template <typename TDevice>
    void SkipWeightAddLayer<TDevice>::computeForwardPass(const int timeStep, const int nnState)
    {
	// absolute time
	int effTimeS = timeStep     * this->parallelSequences();
	int effTimeE = (timeStep+1) * this->parallelSequences();
	
	// shift of the pointer to the data 
	int shiftIn  = 0; // value to assigned layer
	int shiftOut = this->outputBufPtrBias(timeStep * this->parallelSequences(), nnState);
	
	{
	    internal::ComputeLayerOutputFn fn;
	    fn.z_shift = m_z_shift;
	    fn.z_scale = m_z_scale;
	    int n = this->curMaxSeqLength() * this->parallelSequences() * this->size();
	    thrust::for_each(
	       thrust::make_zip_iterator(
		  thrust::make_tuple(
		     this->PreLayers()[0]->outputs().begin() + effTimeS * this->size() - shiftIn,   
		     this->PreLayers()[1]->outputs().begin() + effTimeS * this->size() - shiftIn,
		     this->PreLayers()[2]->outputs().begin() + effTimeS * this->size() - shiftIn,    
		     this->_outputs().begin()  + effTimeS * this->size() - shiftIn)),
	       thrust::make_zip_iterator(
		  thrust::make_tuple(
		     this->PreLayers()[0]->outputs().begin() + effTimeS * this->size() - shiftOut,   
		     this->PreLayers()[1]->outputs().begin() + effTimeS * this->size() - shiftOut,
		     this->PreLayers()[2]->outputs().begin() + effTimeS * this->size() - shiftOut,
		     this->_outputs().begin() + effTimeS * this->size() - shiftOut)),
	       fn);
	}
    }


    // NN backward
    template <typename TDevice>
    void SkipWeightAddLayer<TDevice>::computeBackwardPass(const int nnState)
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
	    fn.layerSize = this->size();
	    fn.patTypes  = helpers::getRawPointer(this->patTypes());
	    fn.z_shift = m_z_shift;
	    fn.z_scale = m_z_scale;
	    
	    tempLayer = dynamic_cast<SkipLayer<TDevice>*>(this->PreLayers()[0]);
	    if (tempLayer)
		fn.grad_x = helpers::getRawPointer(tempLayer->outputErrorsFromSkipLayer());
	    else
		fn.grad_x = helpers::getRawPointer(this->PreLayers()[0]->outputErrors());

	    tempLayer = dynamic_cast<SkipLayer<TDevice>*>(this->PreLayers()[1]);
	    if (tempLayer)
		fn.grad_y = helpers::getRawPointer(tempLayer->outputErrorsFromSkipLayer());
	    else
		fn.grad_y = helpers::getRawPointer(this->PreLayers()[1]->outputErrors());

	    tempLayer = dynamic_cast<SkipLayer<TDevice>*>(this->PreLayers()[2]);
	    if (tempLayer)
		fn.grad_z = helpers::getRawPointer(tempLayer->outputErrorsFromSkipLayer());
	    else
		fn.grad_z = helpers::getRawPointer(this->PreLayers()[2]->outputErrors());

	    int n = this->curMaxSeqLength() * this->parallelSequences() * this->size();

	    thrust::for_each(
	       thrust::make_zip_iterator(
		  thrust::make_tuple(
		     this->PreLayers()[0]->outputs().begin(),   
		     this->PreLayers()[1]->outputs().begin(),
		     this->PreLayers()[2]->outputs().begin(), 
		     this->outputErrors().begin(),
		     thrust::counting_iterator<int>(0))),
	       thrust::make_zip_iterator(
		  thrust::make_tuple(
		     this->PreLayers()[0]->outputs().begin() + n,   
		     this->PreLayers()[1]->outputs().begin() + n,
		     this->PreLayers()[2]->outputs().begin() + n,
		     this->outputErrors().begin()            + n,
		     thrust::counting_iterator<int>(0)       + n)),
	       fn);
	}
	
    }
	
    template <typename TDevice>
    typename SkipWeightAddLayer<TDevice>::real_vector& SkipWeightAddLayer<TDevice>::outputFromGate()
    {	
	return SkipLayer<TDevice>::outputFromGate();
    }
    
    template <typename TDevice>
    const std::string& SkipWeightAddLayer<TDevice>::type() const
    {
	static std::string s1("skipweightadd");
	return s1;
    }

    template <typename TDevice>
    void SkipWeightAddLayer<TDevice>::exportLayer(const helpers::JsonValue &layersArray,
					 const helpers::JsonAllocator &allocator) const
    {
	SkipLayer<TDevice>::exportLayer(layersArray, allocator);

	(*layersArray)[layersArray->Size() - 1].AddMember("weightScale", m_z_scale, allocator);
	(*layersArray)[layersArray->Size() - 1].AddMember("weightShift", m_z_shift, allocator);	
	
	if (m_previousSkipStr.size())
	    (*layersArray)[layersArray->Size() - 1].AddMember("preSkipLayer",
							      m_previousSkipStr.c_str(), allocator);
	if (m_weightLayerStr.size())
	    (*layersArray)[layersArray->Size() - 1].AddMember("preWeightLayer",
							      m_weightLayerStr.c_str(), allocator);
	
    }
    
    template <typename TDevice>
    void SkipWeightAddLayer<TDevice>::reduceOutputBuffer()
    {
	this->resizeOutputBuffer(this->parallelSequences() * this->size());
	this->setSaveMemoryFlag(true);
	printf("\t[mem saved]");
    }

    template <typename TDevice>
    int SkipWeightAddLayer<TDevice>::outputBufPtrBias(const int timeStepTimesParallel,
						      const int nnState)
    {
	if (this->getSaveMemoryFlag()){
	    return timeStepTimesParallel * this->size();
	}else{
	    return 0;
	}
    }

    template <typename TDevice>
    void SkipWeightAddLayer<TDevice>::clearAllBuffers()
    {
	this->clearOutputBuffer();	    
    }

    template <typename TDevice>
    void SkipWeightAddLayer<TDevice>::resizeAllBuffers(const int timeLength)
    {
	this->resizeOutputBuffer(timeLength * this->parallelSequences() * this->size());	
    }    

    template class SkipWeightAddLayer<Cpu>;
    template class SkipWeightAddLayer<Gpu>;
    
}
