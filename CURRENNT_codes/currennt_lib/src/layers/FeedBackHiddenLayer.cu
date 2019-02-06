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

#include "FeedBackHiddenLayer.hpp"

#include "../helpers/getRawPointer.cuh"
#include "../helpers/Matrix.hpp"
#include "../helpers/JsonClasses.hpp"
#include "../activation_functions/Logistic.cuh"
#include "../activation_functions/Tanh.cuh"

#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/for_each.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/fill.h>
#include <boost/foreach.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>
#include <vector>
#include <stdexcept>

#include "../Configuration.hpp"


namespace internal{
namespace {

    typedef activation_functions::Tanh     cell_act_fn_t;
        
    struct vectorFillForward_hidden
    {
	// Copy the output of preceding layer to the output of this layer
	// Copy the output of target layer to the output of this layer

	int dimInput1End;    // dimension of the output from previous layer
	int dimInput1Start;  // from which dimension of the previous layer to read
	int dimInput1Total;  // dimension of output of preceding layer
	int dimInput1Valid;
	
	int dimInput2End;    // dimension of output of target layer (to be fed back, in total dim)
	int dimInput2Start;  // from which dimension of the target to load (may not be 0)
	int dimInput2Total;  // dimension of output of this layer
	int dimInput2Valid;
	
	int dimOutput;
	int parallel;       // number of parallel sentences
	
	real_t *input1;     // preceding layer
	real_t *input2;     // target layer
	real_t *output;     // this layer

	// dispatched over Dim * T * Parallel
	__host__ __device__ void operator() (const thrust::tuple<const real_t&, int> &t)
	{
	    int outputEffIdx = t.get<1>();
	    int timeStep     = outputEffIdx / dimOutput;
	    int dimIdx       = outputEffIdx % dimOutput;

	    // Idx in the output of this layer
	    int outputIdx    = timeStep * dimOutput + dimIdx;
	    int lookBackTime = 0;
		
	    if (dimIdx >= dimInput1Valid){
		// copy from the target layer (feedback part)
		    
		// get the dimension index (across multiple time steps)
		dimIdx       = dimIdx - dimInput1Valid;
		    
		// get the time shift to be looked backwards
		lookBackTime = 1;
		    
		// get the dimension index in each time step
		dimIdx       = dimIdx % dimInput2Valid;
		    
		if (timeStep < lookBackTime)      
		    output[outputIdx] = 0.0; // if the previous step is unavailable
		else{
		    output[outputIdx] = input2[(timeStep - lookBackTime) * dimInput2Total +
					       dimIdx + dimInput2Start];
		}
	    }else{	
		output[outputIdx] = input1[timeStep * dimInput1Total + dimIdx];
	    }
	    
	}
    };
    
    
    struct vectorFillBackward_hidden
    {
	// Copy the output of preceding layer to the output of this layer
	// Copy the output of target layer to the output of this layer

	int dimInput1End;    // dimension of the output from previous layer
	int dimInput1Start;  // from which dimension of the previous layer to read
	int dimInput1Total;  // dimension of output of preceding layer
	int dimInput1Valid;
	
	int dimInput2End;    // dimension of output of target layer (to be fed back, in total dim)
	int dimInput2Start;  // from which dimension of the target to load (may not be 0)
	int dimInput2Total;  // dimension of output of this layer
	int dimInput2Valid;
	
	int dimOutput;
	int parallel;       // number of parallel sentences
	
	real_t *output1;     // preceding layer
	real_t *output2;     // target layer
	real_t *input;       // this layer

	// dispatched over Dim * T * Parallel
	__host__ __device__ void operator() (const thrust::tuple<const real_t&, int> &t)
	{
	    int outputEffIdx = t.get<1>();
	    int timeStep     = outputEffIdx / dimOutput;
	    int dimIdx       = outputEffIdx % dimOutput;

	    // Idx in the output of this layer
	    int outputIdx    = timeStep * dimOutput + dimIdx;
	    int lookBackTime = 0;
		
	    if (dimIdx >= dimInput1Valid){
		// copy from the target layer (feedback part)
		    
		// get the dimension index (across multiple time steps)
		dimIdx       = dimIdx - dimInput1Valid;
		    
		// get the time shift to be looked backwards
		lookBackTime = 1;
		    
		// get the dimension index in each time step
		dimIdx       = dimIdx % dimInput2Valid;
		    
		if (timeStep < lookBackTime){
		    // none
		}else{
		    // accumulate the gradient
		    output2[(timeStep - lookBackTime) * dimInput2Total +
			    dimIdx + dimInput2Start] += input[outputIdx];
		}
	    }else{
		// gradient to the input layer
		output1[timeStep * dimInput1Total + dimIdx] = input[outputIdx];
	    }
	    
	}
    };

    // dustbin.txt/Block 1226x01
    int ParseLayerOpt(const std::string options){
	std::vector<std::string> tempArgs;
	boost::split(tempArgs, options, boost::is_any_of("_"));
	return boost::lexical_cast<int>(tempArgs[0]);
    }
    
}
}

namespace layers{


    
    template <typename TDevice>
    FeedBackHiddenLayer<TDevice>::FeedBackHiddenLayer(const helpers::JsonValue &layerChild,
					  const helpers::JsonValue &weightsSection,
					  Layer<TDevice>           &precedingLayer,
					  int                       maxSeqLength,
					  int                       layerID
					  )
	: TrainableLayer<TDevice>(layerChild, weightsSection, 0, 0,
				  precedingLayer, maxSeqLength, layerID)
	, m_targetDim      (-1)
	, m_targetDimStart (-1)
	, m_targetDimEnd   (-1)
	, m_targetLayer (NULL)
    {
	m_targetBuffer.clear();
	
	const Configuration &config = Configuration::instance();
	
	// configuration for the previous state
	m_prevDimEnd    = ((layerChild->HasMember("previousDimEnd")) ? 
			  ((*layerChild)["previousDimEnd"].GetInt()) : precedingLayer.size());
	m_prevDimStart  = ((layerChild->HasMember("previousDimStart")) ? 
			  ((*layerChild)["previousDimStart"].GetInt()) : 0);
	if (m_prevDimStart < 0 || m_prevDimEnd > precedingLayer.size() ||
	    m_prevDimStart > m_prevDimEnd)
	    throw std::runtime_error("Error in previousDim and previousDimStart configuration");

	if (this->precedingLayer().getSaveMemoryFlag())
	    throw std::runtime_error("layer before feedback is reduced in mem");  

	m_targetLayerName = ((layerChild->HasMember("feedbackSourceLayer")) ? 
			     ((*layerChild)["feedbackSourceLayer"].GetString()) : "");

	if (m_targetLayerName.size() == 0)
	    throw std::runtime_error("feedback_hidden requires a feedbackLayer in network.jsn");
    }

    template <typename TDevice>
    FeedBackHiddenLayer<TDevice>::~FeedBackHiddenLayer()
    {
    }

    template <typename TDevice>
    void FeedBackHiddenLayer<TDevice>::exportLayer(const helpers::JsonValue     &layersArray, 
						   const helpers::JsonAllocator &allocator) const
    {
        TrainableLayer<TDevice>::exportLayer(layersArray, allocator);
        
	if (m_prevDimStart != 0 || m_prevDimEnd != this->precedingLayer().size()){
	    (*layersArray)[layersArray->Size() - 1].AddMember("previousDimStart", 
							      m_prevDimStart, allocator);
	    (*layersArray)[layersArray->Size() - 1].AddMember("previousDimEnd", 
							      m_prevDimEnd, allocator);
	}
	(*layersArray)[layersArray->Size() - 1].AddMember("feedbackSourceLayer",
							  m_targetLayerName.c_str(),
							  allocator);
    }

    template <typename TDevice>
    void FeedBackHiddenLayer<TDevice>::linkTargetLayer(Layer<TDevice> &targetLayer)
    {

	if (targetLayer.name() != m_targetLayerName){
	    return;
	}
	
	m_targetDim      = internal::ParseLayerOpt(targetLayer.layerAddInfor(1));
	m_targetLayer    = &targetLayer;

	m_targetDimStart = 0;
	m_targetDimEnd   = m_targetDim;

	// dim * look_back + dim * aggregate + preceding_layer
	int dimExpected = ((m_targetDimEnd - m_targetDimStart) +
			   (m_prevDimEnd - m_prevDimStart));
	
	if (dimExpected !=this->size()){
	    printf("Feedback dim + Feedforward dim = %d\n", dimExpected);
	    throw std::runtime_error("Error in network.jsn feedback layer size");
	}
	
	if (m_targetDimEnd > m_targetDim || m_targetDimStart > m_targetDim ||
	    m_targetDimEnd < m_targetDimStart){
	    throw std::runtime_error("Error in configuration of targetDimStart, targetDimEnd");
	}

	if (m_targetLayer->getResolution() != this->getResolution()){
	    throw std::runtime_error("FeedbackLayer uses a different time resolution from target");
	}
	    
	
	// print information
	printf("\nCreating the feedback link from %s:\n", m_targetLayerName.c_str());
	printf("\tReading previous layer [%d-%d] dim\n", m_prevDimStart, m_prevDimEnd);
	printf("\tFrom %s [%d-%d]", targetLayer.type().c_str(), m_targetDimStart, m_targetDimEnd);
	printf("\n");
    }

    template <typename TDevice>
    int FeedBackHiddenLayer<TDevice>::returnTargetLayerID()
    {
	if (m_targetLayer == NULL)
	    throw std::runtime_error("FeedBackHiddenLayer has no feedback layer");
	return m_targetLayer->getLayerID();
    }

    
    template <typename TDevice>
    void FeedBackHiddenLayer<TDevice>::loadSequences(const data_sets::DataSetFraction &fraction,
						     const int nnState)
    {
	TrainableLayer<TDevice>::loadSequences(fraction, nnState);
    }
    
    template <typename TDevice>
    const std::string& FeedBackHiddenLayer<TDevice>::type() const
    {
        static std::string s;
        if (s.empty()) s = "feedback_hidden";
        return s;
    }

    // computeForward: 
    //  in training stage, target data are known
    template <typename TDevice>
    void FeedBackHiddenLayer<TDevice>::computeForwardPass(const int nnState)
    {
	throw std::runtime_error("FeedBackHiddenLayer doesnot support computeForwardPass(nnstate)");
    }

    // computeForwardPass
    //  in synthesis stage, when the target must be predicted frame by frame
    template <typename TDevice>
    void FeedBackHiddenLayer<TDevice>::computeForwardPass(const int timeStep, const int nnState)
    {
	
	if (m_targetLayer == NULL){
	    throw std::runtime_error("Target layer is not linked");
	}	
	
	int effTimeStepS = timeStep     * this->parallelSequences();
	int effTimeStepE = (timeStep+1) * this->parallelSequences();

	thrust::fill(this->outputs().begin() + effTimeStepS * this->size(), 
		     this->outputs().begin() + effTimeStepE * this->size(), 0.0);
	
	{{
	    // The dimension of the concatenated feature (if no softmax exists)
	    int previousSize  = this->precedingLayer().size();
	    
	    // Concatenate the feature vector 
	    // (by treating the 1 dimensional softmax Index as a normal feature)
	    internal::vectorFillForward_hidden fn;
	    fn.dimInput1Start = m_prevDimStart;
	    fn.dimInput1End   = m_prevDimEnd;
	    fn.dimInput1Valid = m_prevDimEnd - m_prevDimStart;
	    fn.dimInput1Total = previousSize;     // the dimension from preceding layer
	    
	    fn.dimInput2Start = m_targetDimStart; // from which dimension to load from target layer
	    fn.dimInput2End   = m_targetDimEnd;   // the dimension of the output of target layer
	    fn.dimInput2Valid = m_targetDimEnd - m_targetDimStart;
	    fn.dimInput2Total = m_targetDim;      // dimension for 1 step
		
	    fn.dimOutput      = this->size();     
	    fn.parallel       = this->parallelSequences();
	    
	    fn.input1         = helpers::getRawPointer(this->precedingLayer().outputs());
	    fn.input2         = helpers::getRawPointer(m_targetLayer->feedbackOutputs(false));
	    fn.output         = helpers::getRawPointer(this->outputs());
	    
	    thrust::for_each(
	       thrust::make_zip_iterator(
		 thrust::make_tuple(
			this->outputs().begin()+ effTimeStepS * this->size(),
			thrust::counting_iterator<int>(0)+ effTimeStepS * this->size())),
	       thrust::make_zip_iterator(
		 thrust::make_tuple(
			this->outputs().begin()+ effTimeStepE * this->size(),
			thrust::counting_iterator<int>(0)+ effTimeStepE * this->size())),
			fn);
	    
	}}

    }

    // 
    template <typename TDevice>
    void FeedBackHiddenLayer<TDevice>::computeBackwardPass(const int nnState)
    {
	throw std::runtime_error("FeedBackHiddenLayer doesn't support computeBackwardPass(nnstate)");
    }
    
    // 
    template <typename TDevice>
    void FeedBackHiddenLayer<TDevice>::computeBackwardPass(const int timeStep, const int nnState)
    {

	int effTimeStepS = timeStep     * this->parallelSequences();
	int effTimeStepE = (timeStep+1) * this->parallelSequences();

	
	{{
	    // The dimension of the concatenated feature (if no softmax exists)
	    int previousSize  = this->precedingLayer().size();
	    
	    // Concatenate the feature vector 
	    // (by treating the 1 dimensional softmax Index as a normal feature)
	    internal::vectorFillBackward_hidden fn;
	    fn.dimInput1Start = m_prevDimStart;
	    fn.dimInput1End   = m_prevDimEnd;
	    fn.dimInput1Valid = m_prevDimEnd - m_prevDimStart;
	    fn.dimInput1Total = previousSize;     // the dimension from preceding layer
	    
	    fn.dimInput2Start = m_targetDimStart; // from which dimension to load from target layer
	    fn.dimInput2End   = m_targetDimEnd;   // the dimension of the output of target layer
	    fn.dimInput2Valid = m_targetDimEnd - m_targetDimStart;
	    fn.dimInput2Total = m_targetDim;      // dimension for 1 step
		
	    fn.dimOutput      = this->size();     
	    fn.parallel       = this->parallelSequences();
	    
	    fn.output1        = helpers::getRawPointer(this->precedingLayer().outputErrors());
	    fn.output2        = helpers::getRawPointer(m_targetLayer->outputErrors());
	    fn.input          = helpers::getRawPointer(this->outputErrors());
	    
	    thrust::for_each(
	       thrust::make_zip_iterator(
		 thrust::make_tuple(
			this->outputs().begin()+ effTimeStepS * this->size(),
			thrust::counting_iterator<int>(0)+ effTimeStepS * this->size())),
	       thrust::make_zip_iterator(
		 thrust::make_tuple(
			this->outputs().begin()+ effTimeStepE * this->size(),
			thrust::counting_iterator<int>(0)+ effTimeStepE * this->size())),
			fn);
	    
	}}
    }


    template class FeedBackHiddenLayer<Cpu>;
    template class FeedBackHiddenLayer<Gpu>;
    
}
