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
/****
 *
 *
 *
 ****/


#include "SkipParaLayer.hpp"
#include "../helpers/JsonClasses.hpp"

#include "../helpers/getRawPointer.cuh"
#include "../helpers/Matrix.hpp"
#include "../activation_functions/Tanh.cuh"
#include "../activation_functions/Logistic.cuh"
#include "../activation_functions/Identity.cuh"
#include "../activation_functions/Relu.cuh"
#include "../Configuration.hpp"

#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/for_each.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/fill.h>
#include <boost/foreach.hpp>
#include <boost/shared_ptr.hpp>
#include <vector>


// Definition for computation
// the exactly same block as feed-forward layer
namespace internal {
namespace {
    
    // Symbol:
    //    H(x): the output from the previous normal layer (output from the internal feed-forward block)
    //    x:    the output from the previous skip layer (input to the Highway block)
    //    T(x): the output from the gate of current block
    //    y:    the output of this block

    // output of the gate T(x)
    // T(x) = f(Wx+b)
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

    // output of the currennt layer
    // y = H(x)*T(x)+(1-T(x))*x = (H(x)-x)*T(x) + x
    struct ComputeLayerOutputFn
    {
	__host__ __device__ void  operator() (
		const thrust::tuple<const real_t&, const real_t&, const real_t&, real_t&> &t) const
	{
	    // t.get<0>() H(x)
	    // t.get<1>() T(x)
	    // t.get<2>() x
	    // t.get<3>() y
	    t.get<3>() = (t.get<0>() - t.get<2>()) * t.get<1>() + t.get<2>();
	}
    };

    // T'(x)
    template <typename TActFn>
    struct ComputeDeltaFn
    {
        // since calculating the derivatives is very cheap for our activation functions, 
        // we simple calculate the deltas of all timesteps, including dummies
        
        __host__ __device__ void operator() (const thrust::tuple<real_t&, const real_t&> &t) const
        {
            real_t delta = TActFn::deriv(t.get<1>()) * t.get<0>();
            t.get<0>() = delta;
        }
    };
    
    // the errors back propagated to the previous skip layer
    // let d_e/d_y = e, 
    // the error back-propagated should be:
    //    W*ge + e * [1 - T(x)]
    // W*ge will be assinged to t.get<4>() before this operator
    // then, e*[1-T(x)] will be accumulated
    // this error will be accumulated into the previous skip layer
    template <typename TActFn>
    struct ComputeAccumulateSkipBackError
    {
	__host__ __device__ void operator() (const thrust::tuple<const real_t&, 
					     const real_t&, real_t&> &t) const
	{
	    // t.get<0>() T(x) (delta)
	    // t.get<1>() e
	    // t.get<2>() error back propagated
	    real_t delta = TActFn::deriv(t.get<1>());
	    t.get<2>() = t.get<2>() + t.get<1>() - t.get<1>() * t.get<0>();
	}
    };

    // the erros back propagated to the input to the ActFn of the gate unit
    //    ge <- [H(x)-x] * T'(x) * e
    // change the value of e, e will not be used after this step
    template <typename TActFn>
    struct ComputeErrorToGateActFn
    {
	__host__ __device__ void operator() (const thrust::tuple<const real_t&, 
					     const real_t&, const real_t&, 
					     const real_t&, real_t&> &t) const
	{
	    // t.get<0>() H(x)
	    // t.get<1>() T(x) (delta)
	    // t.get<2>() x
	    // t.get<3>() e
	    // t.get<4>() ge 
	    real_t delta = TActFn::deriv(t.get<1>());
	    t.get<4>() = t.get<3>() * (t.get<0>() - t.get<2>()) * delta;
	}
    };
	
    struct ComputeBiasWeightUpdateFn
    {
        int    layerSize;
        int    patternsCount;
        real_t bias;

        const real_t *deltas;
        
        __host__ __device__ real_t operator() (const int &biasWeightIdx) const
        {
            const real_t *offDeltas = deltas + biasWeightIdx;

            real_t wu = 0;
            for (int i = 0; i < patternsCount; ++i) {
                wu += bias * *offDeltas;
                offDeltas += layerSize;
            }

            return wu;
        }
    };

    struct ComputeBiasWeightUpdateFn_online
    {
        int    layerSize;
        int    patternsCount;
        real_t bias;

        const real_t *deltas;
        const real_t *bias_grad;
        __host__ __device__ real_t operator() (const int &biasWeightIdx) const
        {
            const real_t *offDeltas = deltas + biasWeightIdx;

            real_t wu = bias_grad[biasWeightIdx];
            for (int i = 0; i < patternsCount; ++i) {
                wu += bias * *offDeltas;
                offDeltas += layerSize;
            }

            return wu;
        }
    };

} // anonymous namespace
} // namespace internal



namespace layers{

    // Construct the layer
    template <typename TDevice, typename TActFn>
    SkipParaLayer<TDevice, TActFn>::SkipParaLayer(
					const helpers::JsonValue &layerChild,
					const helpers::JsonValue &weightsSection,
					std::vector<Layer<TDevice>*> &precedingLayers,
					int maxSeqLength,
					int layerID)
	// use preLayers[0] as fake preceding layers
	: SkipLayer<TDevice>(layerChild, weightsSection, precedingLayers,
			     maxSeqLength, layerID, true)
    {
	// currently, only two previous layers are allowed: one from previous skiplayer, and another
	//  from normal feed-forward layer
	if (precedingLayers.size()>2){
	    throw std::runtime_error(
		std::string("SkipParaLayer only receive input from two previous layers"));
	}else if(precedingLayers.size()<2){
	    throw std::runtime_error(
		std::string("SkipParaLayer can not be directly linked to Input layer"));
	}
	
	// check whether the previous layer is directly connected to this layer, or H(x) = x
	// in this case, this skippara layer is not useful
	if (precedingLayers[0] == precedingLayers[1]){
	    throw std::runtime_error(
		std::string("Previous layer of this layer is a skip layer"));
	}

	// previous skip layer
	m_preSkipLayer = precedingLayers[0];
	this->PreLayers().push_back(precedingLayers[0]);
	
	if (this->size() != m_preSkipLayer->size()){
	    printf("Error: %s vs %s", m_preSkipLayer->name().c_str(), this->name().c_str());
	    throw std::runtime_error("Error unequal layer size");
	}
	if (this->size() != this->precedingLayer().size()){
	    printf("Error: %s vs %s", this->precedingLayer().name().c_str(), this->name().c_str());
	    throw std::runtime_error("Error unequal layer size");
	}
	
	
	// initialize the vector
	// m_outputErrorsFromSkipLayer = Cpu::real_vector(this->outputs().size(), (real_t)0.0);
	m_gateOutput                = Cpu::real_vector(this->outputs().size(), (real_t)0.0);
	m_gateErrors                = Cpu::real_vector(this->outputs().size(), (real_t)0.0);

	// Modify 0418 Fatal Error: I should check the weightSection at first
	if (weightsSection.isValid() && 
	    weightsSection->HasMember(this->name().c_str())){
	    // printf("\tRead saved gate bias");
	}else{
	    const Configuration &config = Configuration::instance();
	    // initializing the bias vector to a large negative value, let T(x) approach zero
	    thrust::fill(this->weights().begin() + this->size()*this->preSkipLayer()->size(),
			 this->weights().end(), config.highwayGateBias());
	}

	printf("\n\tReceive input from layer(s):");
	printf(" %s [x],", precedingLayers[0]->name().c_str());
	printf(" %s [H(x)],", precedingLayers[1]->name().c_str());
	printf("\n");

    }	

    // Destructor
    template <typename TDevice, typename TActFn>
    SkipParaLayer<TDevice, TActFn>::~SkipParaLayer()
    {
    }
	
    // NN forward
    template <typename TDevice, typename TActFn>
    void SkipParaLayer<TDevice, TActFn>::computeForwardPass(const int nnState)
    {
	
	// initialization for backward pass 
	// (put it here just for convience, it is complicated to initialize the errors
	//  in backward pass since this layer links to multiple layers)
	if (this->flagTrainingMode())
	    thrust::fill(this->outputErrors().begin(), 
			 (this->outputErrors().begin() + 
			  this->curMaxSeqLength() * this->parallelSequences() * this->size()),
			 0.0);
	
	// Do the Forward Pass
	// calculate the gate output (in the same way as feed-forward layer, but on the gate unit)
	// step1: linear transform
	// Wx
	{{
	    helpers::Matrix<TDevice> weightMatrix   (&this->weights(),               
						     this->preSkipLayer()->size(), 
						     this->size());
	    helpers::Matrix<TDevice> plOutputsMatrix(&this->preSkipLayer()->outputs(),
						     this->preSkipLayer()->size(), 
						     this->curMaxSeqLength() * 
						     this->parallelSequences());
	    helpers::Matrix<TDevice> outputsMatrix  (&this->gateOutput(),            
						     this->size(),               
						     this->curMaxSeqLength() * 
						     this->parallelSequences());
	    
	    outputsMatrix.assignProduct(weightMatrix, true, plOutputsMatrix, false);
	}}
		     
	// step2: non-linear transform
	// f(Wx+b)
	{{
		internal::ComputeOutputFn<TActFn> fn;
		fn.layerSize     = this->size();
		fn.bias          = this->bias();
		fn.biasWeights   = (helpers::getRawPointer(this->weights()) + 
				    this->size()*this->preSkipLayer()->size());
		
		thrust::transform(
				  this->gateOutput().begin(),
				  (this->gateOutput().begin() + 
				   this->curMaxSeqLength()*this->parallelSequences()*this->size()),
				  thrust::counting_iterator<int>(0),
				  this->gateOutput().begin(),
				  fn
				  );
	}}
	

	// compute the output based on the output of gate
	//  y = (H(x)-x)*T(x) + x
	{{
		internal::ComputeLayerOutputFn fn;
		int n = this->curMaxSeqLength() * this->parallelSequences() * this->size();
		thrust::for_each(
		      thrust::make_zip_iterator(
			thrust::make_tuple(this->precedingLayer().outputs().begin(),   
					   this->gateOutput().begin(),
					   this->preSkipLayer()->outputs().begin(),    
					   this->_outputs().begin())),
		      thrust::make_zip_iterator(
			thrust::make_tuple(this->precedingLayer().outputs().begin()+n, 
					   this->gateOutput().begin()+n,
					   this->preSkipLayer()->outputs().begin()+n,  
					   this->_outputs().begin()+n)),
		      fn
		      );
		
	}}
       
	// done.
    }
	
    // NN forward
    template <typename TDevice, typename TActFn>
    void SkipParaLayer<TDevice, TActFn>::computeForwardPass(const int timeStep, const int nnState)
    {

	if (this->precedingLayer().getSaveMemoryFlag() || this->preSkipLayer()->getSaveMemoryFlag())
	    throw std::runtime_error("SkipPara layer is not ready for reduced memory input");
	
	int effTimeS = timeStep     * this->parallelSequences();
	int effTimeE = (timeStep+1) * this->parallelSequences();

	// Do the Forward Pass
	// calculate the gate output (in the same way as feed-forward layer, but on the gate unit)
	// step1: linear transform
	// Wx
	{{
	    helpers::Matrix<TDevice> weightMatrix   (&this->weights(), 
						     this->preSkipLayer()->size(), 
						     this->size());
	    
	    helpers::Matrix<TDevice> plOutputsMatrix(&this->preSkipLayer()->outputs(),
						     this->preSkipLayer()->size(), 
						     this->parallelSequences(),
						     effTimeS * this->precedingLayer().size());
	    helpers::Matrix<TDevice> outputsMatrix  (&this->gateOutput(),            
						     this->size(),                
						     this->parallelSequences(),
						     effTimeS * this->size());
	    
	    outputsMatrix.assignProduct(weightMatrix, true, plOutputsMatrix, false);
	}}
		     
	// step2: non-linear transform
	// f(Wx+b)
	{{
		internal::ComputeOutputFn<TActFn> fn;
		fn.layerSize     = this->size();
		fn.bias          = this->bias();
		fn.biasWeights   = (helpers::getRawPointer(this->weights()) + 
				    this->size()*this->preSkipLayer()->size());
		
		thrust::transform(
			this->gateOutput().begin() + effTimeS * this->size(),
			this->gateOutput().begin() + effTimeE * this->size(),
			thrust::counting_iterator<int>(0),
			this->gateOutput().begin() + effTimeS * this->size(),
			fn);
	}}
	
	// compute the output based on the output of gate
	//  y = (H(x)-x)*T(x) + x
	{{
		internal::ComputeLayerOutputFn fn;
		
		thrust::for_each(
		      thrust::make_zip_iterator(
			thrust::make_tuple(
			    this->precedingLayer().outputs().begin() + effTimeS * this->size(),   
			    this->gateOutput().begin()               + effTimeS * this->size(),
			    this->preSkipLayer()->outputs().begin()  + effTimeS * this->size(),    
			    this->_outputs().begin()                 + effTimeS * this->size())),
		      thrust::make_zip_iterator(
			thrust::make_tuple(
			    this->precedingLayer().outputs().begin() + effTimeE * this->size(), 
			    this->gateOutput().begin()               + effTimeE * this->size(),
			    this->preSkipLayer()->outputs().begin()  + effTimeE * this->size(),  
			    this->_outputs().begin()                 + effTimeE * this->size())),
		      fn);
	}}
	// done.
    }

    // NN backward
    template <typename TDevice, typename TActFn>
    void SkipParaLayer<TDevice, TActFn>::computeBackwardPass(const int nnState)
    {
	// the same as the skipadd layer
	// at first, add the errors in both m_outputErrorsFromSkipLayer and m_outputErrors
	thrust::transform(this->outputErrorsFromSkipLayer().begin(),
			  (this->outputErrorsFromSkipLayer().begin() + 
			   this->curMaxSeqLength() * this->parallelSequences() * this->size()),
			  this->outputErrors().begin(),
			  this->outputErrors().begin(),
			  thrust::plus<real_t>()
			  );

	
	// get the errors back before the actFn of gate unit
	// ge <- e * [H(x)-x]*f'(x), where f(x) is the actFn of gate unit
	{{
		internal::ComputeErrorToGateActFn<TActFn> fn;
		int n = this->curMaxSeqLength() * this->parallelSequences() * this->size();

		thrust::for_each(
		         thrust::make_zip_iterator(
				thrust::make_tuple(this->precedingLayer().outputs().begin(), 
						   this->gateOutput().begin(),
						   this->preSkipLayer()->outputs().begin(), 
						   this->outputErrors().begin(),
						   this->gateErrors().begin())),
			 thrust::make_zip_iterator(
				thrust::make_tuple(this->precedingLayer().outputs().begin()+n, 
						   this->gateOutput().begin()+n,
						   this->preSkipLayer()->outputs().begin()+n, 
						   this->outputErrors().begin()+n,
						   this->gateErrors().begin()+n)),
			 fn
			 );
	}}

	std::vector<Layer<TDevice> *> prelayers;
	prelayers.push_back(this->preSkipLayer());
	prelayers.push_back(&this->precedingLayer());
	// send erros to the all the previous layers
	BOOST_REVERSE_FOREACH (Layer<TDevice> *layer, prelayers) {
	    //SkipParaLayer<TDevice, TActFn>* tempLayer =
	    // dynamic_cast<SkipParaLayer<TDevice, TActFn>*>(layer);
	    SkipLayer<TDevice>* tempLayer= dynamic_cast<SkipLayer<TDevice>*>(layer);
	    
	    if(tempLayer){
		// update the parameter of the gate
		// this is an SkipPara or SkipAdd Layer, 
		// erros should be accumulated to m_outputErrorsFromSkipLayer
		{{
		    // W * ge + (1-T)*e
		    // step1. accumulate W * ge to the tempLayer
		    helpers::Matrix<TDevice> weightsMatrix (&this->weights(),      
							    tempLayer->size(),   
							    this->size());
		    helpers::Matrix<TDevice> plErrorsMatrix(&tempLayer->outputErrorsFromSkipLayer(),
							    tempLayer->size(),   
							    this->curMaxSeqLength() * 
							    this->parallelSequences());
		    helpers::Matrix<TDevice> deltasMatrix  (&this->gateErrors(), 
							    this->size(), 
							    this->curMaxSeqLength() * 
							    this->parallelSequences());
		    plErrorsMatrix.assignProduct(weightsMatrix, false, deltasMatrix, false);

		    // step2. accumulate the (1 - T) * e 
		    internal::ComputeAccumulateSkipBackError<TActFn> fn;
		    int n = this->curMaxSeqLength() * this->parallelSequences() * this->size();
		    thrust::for_each(
		         thrust::make_zip_iterator(
			    thrust::make_tuple(this->gateOutput().begin(),   
					       this->outputErrors().begin(),
					       tempLayer->outputErrorsFromSkipLayer().begin())),
			 thrust::make_zip_iterator(
			    thrust::make_tuple(this->gateOutput().begin()+n, 
					       this->outputErrors().begin()+n,
					       tempLayer->outputErrorsFromSkipLayer().begin()+n)),
			 fn
			 );
		    
		}}
	    }else{
		// else, for the normal layer, progatate the erros 
		// errors * gateOutput(x) (d_e/d_y * T(x), elementwise)
		thrust::transform(this->outputErrors().begin(),
				  (this->outputErrors().begin() +
				   this->curMaxSeqLength()      * 
				   this->parallelSequences()    * 
				   this->size()),
				  this->gateOutput().begin(),
				  layer->outputErrors().begin(),
				  thrust::multiplies<real_t>());
	    }
	}
	

	// Now, this->outputErrors has become the errors before the activation funciton of gate unit
        // compute the input weight updates
        {{
            helpers::Matrix<TDevice> weightUpdatesMatrix(&this->_weightUpdates(), 
							 this->size(), this->size());
            helpers::Matrix<TDevice> plOutputsMatrix    (&this->preSkipLayer()->outputs(), 
							 this->preSkipLayer()->size(), 
							 this->curMaxSeqLength() * 
							 this->parallelSequences());
            helpers::Matrix<TDevice> deltasMatrix       (&this->gateErrors(),  
							 this->size(), 
							 this->curMaxSeqLength() * 
							 this->parallelSequences());

            weightUpdatesMatrix.assignProduct(plOutputsMatrix, false, deltasMatrix, true);
        }}

        // compute the bias weight updates
        {{
            internal::ComputeBiasWeightUpdateFn fn;
            fn.layerSize     = this->size();
            fn.patternsCount = this->curMaxSeqLength() * this->parallelSequences();
            fn.bias          = this->bias();
            fn.deltas        = helpers::getRawPointer(this->gateErrors());

            thrust::transform(
                thrust::counting_iterator<int>(0),
                thrust::counting_iterator<int>(0) + this->size(),
                this->_weightUpdates().begin() + this->preSkipLayer()->size() * this->size(),
                fn
                );
        }}

    }


    // NN backward
    template <typename TDevice, typename TActFn>
    void SkipParaLayer<TDevice, TActFn>::computeBackwardPass(const int timeStep, const int nnState)
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
		
	// get the errors back before the actFn of gate unit
	// ge <- e * [H(x)-x]*f'(x), where f(x) is the actFn of gate unit
	{{
	    internal::ComputeErrorToGateActFn<TActFn> fn;
	    int n = this->curMaxSeqLength() * this->parallelSequences() * this->size();

	    thrust::for_each(
	     thrust::make_zip_iterator(
	      thrust::make_tuple(
		this->precedingLayer().outputs().begin() + effTimeS * this->size(), 
		this->gateOutput().begin()               + effTimeS * this->size(),
		this->preSkipLayer()->outputs().begin()  + effTimeS * this->size(), 
		this->outputErrors().begin()             + effTimeS * this->size(),
		this->gateErrors().begin()               + effTimeS * this->size())),
	     thrust::make_zip_iterator(
	      thrust::make_tuple(
		this->precedingLayer().outputs().begin() + effTimeE * this->size(), 
		this->gateOutput().begin()               + effTimeE * this->size(),
		this->preSkipLayer()->outputs().begin()  + effTimeE * this->size(), 
		this->outputErrors().begin()             + effTimeE * this->size(),
		this->gateErrors().begin()               + effTimeE * this->size())),
	     fn);
	}}

	std::vector<Layer<TDevice> *> prelayers;
	prelayers.push_back(this->preSkipLayer());
	prelayers.push_back(&this->precedingLayer());
	
	// send erros to the all the previous layers
	BOOST_REVERSE_FOREACH (Layer<TDevice> *layer, prelayers) {
	    
	    //SkipParaLayer<TDevice, TActFn>* tempLayer =
	    // dynamic_cast<SkipParaLayer<TDevice, TActFn>*>(layer);
	    SkipLayer<TDevice>* tempLayer= dynamic_cast<SkipLayer<TDevice>*>(layer);

	    if(tempLayer){
		
		// update the parameter of the gate
		// this is an SkipPara or SkipAdd Layer, 
		// erros should be accumulated to m_outputErrorsFromSkipLayer
		{{
		    // W * ge + (1-T)*e
		    // step1. accumulate W * ge to the tempLayer
		    helpers::Matrix<TDevice> weightsMatrix (&this->weights(),      
							    tempLayer->size(),   
							    this->size());
		    
		    helpers::Matrix<TDevice> plErrorsMatrix(&tempLayer->outputErrorsFromSkipLayer(),
							    tempLayer->size(),   
							    this->parallelSequences(),
							    tempLayer->size() * effTimeS);
		    
		    helpers::Matrix<TDevice> deltasMatrix  (&this->gateErrors(), 
							    this->size(), 
							    this->parallelSequences(),
							    this->size() * effTimeS);
		    
		    plErrorsMatrix.assignProduct(weightsMatrix, false, deltasMatrix, false);

		    // step2. accumulate the (1 - T) * e 
		    internal::ComputeAccumulateSkipBackError<TActFn> fn;
		    
		    thrust::for_each(
		      thrust::make_zip_iterator(
			thrust::make_tuple(
			  this->gateOutput().begin()   + effTimeS * this->size(),   
			  this->outputErrors().begin() + effTimeS * this->size(),
			  tempLayer->outputErrorsFromSkipLayer().begin() + effTimeS * this->size())),
		      thrust::make_zip_iterator(
			thrust::make_tuple(
			  this->gateOutput().begin()   + effTimeE * this->size(), 
			  this->outputErrors().begin() + effTimeE * this->size(),
			  tempLayer->outputErrorsFromSkipLayer().begin() + effTimeE * this->size())),
		      fn);
		    
		}}
		
	    }else{
		// else, for the normal layer, progatate the erros 
		// errors * gateOutput(x) (d_e/d_y * T(x), elementwise)
		thrust::transform(
		   this->outputErrors().begin()  + effTimeS * this->size(),
		   this->outputErrors().begin()  + effTimeE * this->size(),
		   this->gateOutput().begin()    + effTimeS * this->size(),
		   layer->outputErrors().begin() + effTimeS * this->size(),
		  thrust::multiplies<real_t>());
	    }
	}
	

	// Now, this->outputErrors has become the errors before the activation funciton of gate unit
        // compute the input weight updates
        {{
            helpers::Matrix<TDevice> weightUpdatesMatrix(&this->_weightUpdates(), 
							 this->size(), this->size());
	    
            helpers::Matrix<TDevice> plOutputsMatrix    (&this->preSkipLayer()->outputs(), 
							 this->preSkipLayer()->size(), 
							 this->parallelSequences(),
							 effTimeS * this->preSkipLayer()->size());
            helpers::Matrix<TDevice> deltasMatrix       (&this->gateErrors(),  
							 this->size(), 
							 this->parallelSequences(),
							 effTimeS * this->size());

            weightUpdatesMatrix.assignProduct(plOutputsMatrix, false, deltasMatrix, true);
        }}

        // compute the bias weight updates
        {{
            internal::ComputeBiasWeightUpdateFn_online fn;
            fn.layerSize     = this->size();
            fn.patternsCount = this->parallelSequences();
            fn.bias          = this->bias();
            fn.deltas        = (helpers::getRawPointer(this->gateErrors()) +
				effTimeS * this->size());
	    fn.bias_grad     = (helpers::getRawPointer(this->_weightUpdates()) +
				this->preSkipLayer()->size() * this->size());
	    
            thrust::transform(
                thrust::counting_iterator<int>(0),
                thrust::counting_iterator<int>(0) + this->size(),
                this->_weightUpdates().begin() + this->preSkipLayer()->size() * this->size(),
                fn);
        }}

    }

    
    // return all the preceding layers
    template <typename TDevice, typename TActFn>
    Layer<TDevice>* SkipParaLayer<TDevice,TActFn>::preSkipLayer()
    {
	return m_preSkipLayer;
    }

    template <typename TDevice, typename TActFn>
    typename SkipParaLayer<TDevice,TActFn>::real_vector& SkipParaLayer<TDevice,TActFn>::gateOutput(){
	return m_gateOutput;
    }

    template <typename TDevice, typename TActFn>
    typename SkipParaLayer<TDevice,TActFn>::real_vector& SkipParaLayer<TDevice,TActFn>::outputFromGate(){
	return m_gateOutput;
    }
    
    template <typename TDevice, typename TActFn>
    typename SkipParaLayer<TDevice,TActFn>::real_vector& SkipParaLayer<TDevice,TActFn>::gateErrors(){
	return m_gateErrors;
    }

    template <typename TDevice, typename TActFn>
    const std::string& SkipParaLayer<TDevice,TActFn>::type() const
    {
	static std::string s;
	if (s.empty()){
	    if (typeid(TActFn) == typeid(activation_functions::Tanh))
                s = "skippara_tanh";
            else if (typeid(TActFn) == typeid(activation_functions::Logistic))
                s = "skippara_logistic";
            else if (typeid(TActFn) == typeid(activation_functions::Identity))
                s = "skippara_identity";
            else if (typeid(TActFn) == typeid(activation_functions::Relu))
                s = "skippara_relu";
            else
                throw std::runtime_error("Unsupported activation function");
	}
        return s;
    }

    template <typename TDevice, typename TActFn>
    void SkipParaLayer<TDevice,TActFn>::clearAllBuffers()
    {
    }

    template <typename TDevice, typename TActFn>
    void SkipParaLayer<TDevice,TActFn>::resizeAllBuffers(const int timeLength)
    {
    }    

    
    template class SkipParaLayer<Cpu, activation_functions::Tanh>;
    template class SkipParaLayer<Gpu, activation_functions::Tanh>;
    template class SkipParaLayer<Cpu, activation_functions::Logistic>;
    template class SkipParaLayer<Gpu, activation_functions::Logistic>;
    template class SkipParaLayer<Cpu, activation_functions::Identity>;
    template class SkipParaLayer<Gpu, activation_functions::Identity>;
    template class SkipParaLayer<Cpu, activation_functions::Relu>;
    template class SkipParaLayer<Gpu, activation_functions::Relu>;

    
}
