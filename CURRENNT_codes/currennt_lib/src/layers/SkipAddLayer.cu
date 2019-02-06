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


#include "SkipAddLayer.hpp"
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

    // Generating noise
    struct tempPrg
    {
	float a, b;
	int   seed;
	
	__host__ __device__
	tempPrg(float _a=-1.f, float _b=1.f, int _seed=123) : a(_a), b(_b), seed(_seed) {};

	__host__ __device__
	float operator()(const unsigned int n) const
	{
	    thrust::default_random_engine rng(seed);
	    thrust::uniform_real_distribution<float> dist(a, b);
	    rng.discard(n);
	    return dist(rng);
	}
    };

}
}


namespace layers{

    // Construct the layer
    template <typename TDevice>
    SkipAddLayer<TDevice>::SkipAddLayer(
					const helpers::JsonValue &layerChild,
					const helpers::JsonValue &weightsSection,
					std::vector<Layer<TDevice>*> &precedingLayers,
					int maxSeqLength,
					int layerID)
	// use preLayers[0] as fake preceding layers
	: SkipLayer<TDevice>(layerChild, weightsSection, precedingLayers,
			     maxSeqLength, layerID, false)
	, m_noiseRatio      (-1.0)
	, m_flagSkipInit    (true)
    {
	// Initial check
	if (precedingLayers.size() < 1)
	    throw std::runtime_error("Error no precedinglayers in skipadd/skipini");
	
	// Link previous layers
	m_previousSkipStr = (layerChild->HasMember("preSkipLayer") ? 
			     ((*layerChild)["preSkipLayer"].GetString()) : "");
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
	    // default cause, use only the previous 1 skip and previous normal output layer
	    if (precedingLayers.size()<2)
		this->PreLayers().assign(precedingLayers.begin(), precedingLayers.end());
	    else
		this->PreLayers().assign(precedingLayers.end()-2, precedingLayers.end());
	}
	
	printf("\n\tReceive input from layer(s):");
	BOOST_FOREACH (Layer<TDevice> *layer, this->PreLayers()) {
	    printf(" %s,", layer->name().c_str());
	    if (layer->size() != this->size()){
		printf("Error: %s vs %s", layer->name().c_str(), this->name().c_str());
		throw std::runtime_error("Error unequal layer size");
	    }
	}
	printf("\n");
	
	// m_outputErrorsFromSkipLayer = Cpu::real_vector(this->outputs().size(), (real_t)0.0);

	m_noiseRatio    = (layerChild->HasMember("noiseRatio") ? 
			   static_cast<real_t>((*layerChild)["noiseRatio"].GetDouble()) : -1.0);
	if (m_noiseRatio > 0)
	    printf("\n\tInject noise %f\n", m_noiseRatio);

	// Post-processing check
	// Note: here precedingLayers is checked.
	// In NeuralNetwork(), only skipadd will receive more than one layer in precedingLayer
	if (precedingLayers.size()<2)
	    m_flagSkipInit = true;  // this is the skipinit
	else
	    m_flagSkipInit = false; // this is the skipadd

	if (m_flagSkipInit == true && m_noiseRatio < 0 && this->flagTrainingMode()){
	    m_virtualLayer = true;     // this layer is just a virtual layer
	    this->clearOutputBuffer(); // clear the output memory (optional)
	}else{
	    m_virtualLayer = false;
	}
    }	

    // Destructor
    template <typename TDevice>
    SkipAddLayer<TDevice>::~SkipAddLayer()
    {
    }

    template <typename TDevice>
    typename SkipAddLayer<TDevice>::real_vector& SkipAddLayer<TDevice>::outputs()
    {
	if (m_virtualLayer)
	    return this->precedingLayer().outputs();
	else
	    return this->_outputs();
    }
    
    // NN forward
    template <typename TDevice>
    void SkipAddLayer<TDevice>::computeForwardPass(const int nnState)
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

	// processing
	if (m_virtualLayer){
	    // if virtual Layer, no need to do anything
	    return;
	}else{
	    // initialization
	    if (m_noiseRatio > 0){
		thrust::counting_iterator<unsigned int> index_sequence_begin(0);
		thrust::transform(
		    index_sequence_begin,
		    (index_sequence_begin +
		     this->curMaxSeqLength() * this->parallelSequences() * this->size()),
		    this->outputs().begin(),
		    internal::tempPrg(-1.0 * m_noiseRatio, m_noiseRatio,
				      (int)(misFuncs::GetRandomNumber() * 10000.0)));
	    }else{
		thrust::fill(this->outputs().begin(), 
			     (this->outputs().begin() + 
			      this->curMaxSeqLength() * this->parallelSequences() * this->size()),
			     0.0);
	    }
	
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
    }

    // NN forward
    template <typename TDevice>
    void SkipAddLayer<TDevice>::computeForwardPass(const int timeStep, const int nnState)
    {
	// absolute time
	int effTimeS = timeStep     * this->parallelSequences();
	int effTimeE = (timeStep+1) * this->parallelSequences();
	
	// shift of the pointer to the data 
	int shiftIn  = 0; // value to assigned layer
	int shiftOut = this->outputBufPtrBias(timeStep * this->parallelSequences(), nnState);
	
	if (m_virtualLayer){
	    // virtual layer, no need to do anything
	    return;
	}else{
	    // physical layer, compute the output
	    if (m_noiseRatio > 0){
		// initialize the output buffer with noise
		thrust::counting_iterator<unsigned int> index_sequence_begin(0);
		thrust::transform(index_sequence_begin    + effTimeS * this->size(),
				  index_sequence_begin    + effTimeE * this->size(),
				  this->outputs().begin() + effTimeS * this->size() - shiftOut,
				  internal::tempPrg(-1.0 * m_noiseRatio, m_noiseRatio));
	    }else{
		// initialization without noise
		thrust::fill(this->outputs().begin() + effTimeS * this->size() - shiftOut, 
			     this->outputs().begin() + effTimeE * this->size() - shiftOut, 
			     0.0);
	    }

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
    }


    // NN backward
    template <typename TDevice>
    void SkipAddLayer<TDevice>::computeBackwardPass(const int nnState)
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
			     layer->outputErrors().begin()
			     );
	    }
	    
	}
    }

    // NN backward
    template <typename TDevice>
    void SkipAddLayer<TDevice>::computeBackwardPass(const int timeStep, const int nnState)
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
			this->outputErrors().begin()                   + this->size() * effTimeS,
			this->outputErrors().begin()                   + this->size() * effTimeE,
			tempLayer->outputErrorsFromSkipLayer().begin() + this->size() * effTimeS,
			tempLayer->outputErrorsFromSkipLayer().begin() + this->size() * effTimeS,
			thrust::plus<real_t>());
	    }else{
		
		// else, just copy the data to the outputErrors
		thrust::copy(
			this->outputErrors().begin()   + this->size() * effTimeS,
			this->outputErrors().begin()   + this->size() * effTimeE,
			layer->outputErrors().begin()  + this->size() * effTimeS);
	    }
	    
	}
	
    }

    // return all the preceding layers
    // template <typename TDevice>
    // std::vector<Layer<TDevice>*> SkipAddLayer<TDevice>::PreLayers()
    // {
    // return this->PreLayers();
    //}

    template <typename TDevice>
    typename SkipAddLayer<TDevice>::real_vector& SkipAddLayer<TDevice>::outputFromGate()
    {	
	return SkipLayer<TDevice>::outputFromGate();
    }
    
    template <typename TDevice>
    const std::string& SkipAddLayer<TDevice>::type() const
    {
	// Fatal Error, one instance can only use one static string
	// This is different from the case in Feedforward/SkipPara
	static std::string s1("skipini");
	static std::string s2("skipadd");
        return (m_flagSkipInit ? s1 : s2);
    }

    template <typename TDevice>
    void SkipAddLayer<TDevice>::exportLayer(const helpers::JsonValue &layersArray,
					 const helpers::JsonAllocator &allocator) const
    {
	SkipLayer<TDevice>::exportLayer(layersArray, allocator);
	if (m_noiseRatio > 0)
	    (*layersArray)[layersArray->Size() - 1].AddMember("noiseRatio",
							      m_noiseRatio, allocator);
	if (m_previousSkipStr.size())
	    (*layersArray)[layersArray->Size() - 1].AddMember("preSkipLayer",
							      m_previousSkipStr.c_str(), allocator);

    }
    
    /*
    template <typename TDevice>
    typename SkipAddLayer<TDevice>::real_vector& SkipAddLayer<TDevice>::outputErrorsFromSkipLayer()
    {
        return this->outputErrorsFromSkipLayer();
    }
    */

    template <typename TDevice>
    void SkipAddLayer<TDevice>::reduceOutputBuffer()
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
    int SkipAddLayer<TDevice>::outputBufPtrBias(const int timeStepTimesParallel, const int nnState)
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
    void SkipAddLayer<TDevice>::clearAllBuffers()
    {
	if (m_virtualLayer){

	}else{
	    this->clearOutputBuffer();	    
	}
    }

    template <typename TDevice>
    void SkipAddLayer<TDevice>::resizeAllBuffers(const int timeLength)
    {
	if (m_virtualLayer){
	    
	}else{
	    this->resizeOutputBuffer(timeLength * this->parallelSequences() * this->size());
	}	
    }    

    template class SkipAddLayer<Cpu>;
    template class SkipAddLayer<Gpu>;
    
}
