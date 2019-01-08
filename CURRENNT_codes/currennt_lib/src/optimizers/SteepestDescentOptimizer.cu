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

#include "SteepestDescentOptimizer.hpp"
#include "../layers/TrainableLayer.hpp"
#include "../layers/MDNLayer.hpp"
#include "../layers/InputLayer.hpp"
#include "../layers/Layer.hpp"
#include "../helpers/getRawPointer.cuh"
#include "../rapidjson/document.h"

#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>

namespace internal {
namespace {

    struct UpdateWeightFn_withMask
    {
        real_t learningRate;
        real_t momentum;

        const real_t *weights;

	/* Add 0413 weight Mask*/
	const real_t *weightMask;
	
	real_t       *weightUpdates;
        real_t       *weightDeltas;

        __host__ __device__ real_t operator() (const int &weightIdx)
        {

	    if (isnan(weightUpdates[weightIdx]) || isinf(weightUpdates[weightIdx]))
		weightUpdates[weightIdx] = 0.0;
	    
            // calculate and store the weight delta
            real_t delta = momentum * weightDeltas[weightIdx] - 
		learningRate * weightUpdates[weightIdx];
	    
            weightDeltas[weightIdx] = delta;

            // calculate the new weight
	    // Modify 0413 weight Mask
            // real_t newWeight = (weights[weightIdx] + delta;
	    real_t newWeight = ((weights[weightIdx] + delta)*weightMask[weightIdx]);
            return newWeight;
        }
    };

    struct UpdateWeightFn
    {
        real_t learningRate;
        real_t momentum;

        const real_t *weights;
        real_t       *weightUpdates;
        real_t       *weightDeltas;

        __host__ __device__ real_t operator() (const int &weightIdx)
        {
	    
	    if (isnan(weightUpdates[weightIdx]) || isinf(weightUpdates[weightIdx]))
		weightUpdates[weightIdx] = 0.0;
	    
            // calculate and store the weight delta
	    real_t delta = momentum * weightDeltas[weightIdx] - 
		learningRate * weightUpdates[weightIdx];
            	
            weightDeltas[weightIdx] = delta;

            // calculate the new weight
	    // Modify 0413 weight Mask
            real_t newWeight = weights[weightIdx] + delta;
	    return newWeight;
        }
    };
    
    /* Add 16-02-22 Wang: for WE updating */
    // functor to update the parameter
    struct UpdateWeWeightFn
    {
        real_t learningRate;
        real_t *weights;
        real_t *weightUpdates;
        real_t *weightMask;
	bool    useWeightMask;
        __host__ real_t operator() (const int &weightIdx)
        {

	    if (isnan(weightUpdates[weightIdx]) || isinf(weightUpdates[weightIdx]))
		weightUpdates[weightIdx] = 0.0;
	    
            // calculate and store the weight delta
            real_t delta =  -1 * learningRate * weightUpdates[weightIdx];

            // calculate the new weight
            real_t newWeight = weights[weightIdx] +
		(useWeightMask ? (delta * weightMask[weightIdx]) : delta);
            return newWeight;
        }
    };
        
    struct AdaGradAccumulate
    {
	real_t fracLength; 
	const real_t *weightMask;
        __host__ __device__ void operator() (const thrust::tuple<real_t&, real_t&, const int&> &t) const
        {
	    if (isnan(t.get<0>()) || isinf(t.get<0>()))
		t.get<0>() = 0.0;
	    
	    if (weightMask && weightMask[t.get<2>()] < 1.0)
		return;
	    real_t aveGradient = t.get<0>() / fracLength;

	    
	    t.get<1>() = t.get<1>() + (aveGradient * aveGradient);
        }
    };

    struct AdaGradAccumulateUpdate
    {
	real_t fracLength;
	const real_t *weightMask;
        __host__ __device__ void operator() (const thrust::tuple<real_t&, real_t&, const int&> &t) const
        {
	    if (isnan(t.get<0>()) || isinf(t.get<0>()))
		t.get<0>() = 0.0;

	    if (weightMask && weightMask[t.get<2>()] < 1.0)
		return;
	    real_t aveGradient = t.get<0>() / fracLength;

	    t.get<1>() = t.get<1>() + (aveGradient * aveGradient);
            t.get<0>() = aveGradient / sqrt(t.get<1>());
        }
    };

    struct AdamAccumulateUpdate
    {
	real_t  fracLength;
	real_t  beta1Accum;
	real_t  beta2Accum;
	real_t *mvBuffer;
	const real_t *weightMask;
	
        __host__ __device__ void operator() (const thrust::tuple<real_t&, const int&> &t) const
        {
	    if (isnan(t.get<0>()) || isinf(t.get<0>()))
		t.get<0>() = 0.0;

	    //
	    if (weightMask && weightMask[t.get<1>()] < 1.0)
		return;

	    // Average gradient over the sequence
	    real_t aveGradient = t.get<0>() / fracLength;
	    
	    // m_t = m_t_1 * beta1 + (1-beta1)*gradient;
    	    // v_t = v_t_1 * beta1 + (1-beta1)*gradient*gradient;
	    mvBuffer[2 * t.get<1>()]   = (mvBuffer[2 * t.get<1>()]   * OP_ADAMBETA1 +
					  aveGradient * (1 - OP_ADAMBETA1));
	    mvBuffer[2 * t.get<1>()+1] = (mvBuffer[2 * t.get<1>()+1] * OP_ADAMBETA2 +
					  aveGradient * aveGradient * (1 - OP_ADAMBETA2));
	    real_t tmpM = mvBuffer[2 * t.get<1>()]   / (1 - beta1Accum);
	    real_t tmpV = mvBuffer[2 * t.get<1>()+1] / (1 - beta2Accum);
	    
            t.get<0>() = tmpM / (sqrt(tmpV) + OP_ADAMEPSILON);
        }
    };
    
    struct divideOpe { 
	const real_t a; 
	divideOpe(real_t _a) : a(_a) {} 
	__host__ __device__ float operator()(const real_t& x) const 
	{ 
	    return x/a; 
	} 
    }; 

} // anonymous namespace
} // namespace internal


namespace optimizers {
    
    /* Add 16-02-22 Wang: for WE updating */
    // add the SGD optimizer for we
    template <typename TDevice>
    void SteepestDescentOptimizer<TDevice>::_updateWeInput(int fracLength)
    {
	if (m_weLearningRate < 0 ){
	    
	}else{
	    // The updating requires m_weBank, m_outerrors and m_weIdx, m_weDim, m_weIdx
	    // Because m_weBank is CPU::real_vector, we avoid using thrust and GPU computating
	    // currently, no momentum of we updating
	
	    // get the input layer
	    layers::InputLayer<TDevice> *layer = 
		dynamic_cast<layers::InputLayer<TDevice>*>(
				this->_neuralNetwork().layers().front().get());
	
	    // because dummy error is zero, no need to know where dummy starts, 
	    // just udpate using all the data

	    unsigned int inputSize  = layer->size();
	    unsigned int weIDDim    = layer->_weIDDim();
	    unsigned int weDim      = layer->_weDim();
	    
	    // Not using assignment here
	    // Cpu::real_vector weBank = layer->_weBank();
	    Cpu::real_vector weIdx  = layer->_weIdx();
	    Cpu::real_vector err    = layer->outputErrorsCpu();

	    internal::UpdateWeWeightFn fn;
	    fn.learningRate  = m_weLearningRate;
	    fn.useWeightMask = layer->flagWeMask();
	    
	    // updating now
	    for (int i=0;i<weIdx.size();i++){
	    
		if (weIdx[i]<0){
		    // note: when parallel sequences was utilized, 
		    // the data buffer size is like
		    // m_parallel * m_maxLength >= m_parallel * m_curMaxLength 
		    //                          >= sum_m_parallel(timesteps)
		    // dummy only work for the empty slots between 
		    // m_parallel*m_curMaxLength and sum_m_parallel(timesteps)
		    // thus, we need to use weIdx to skip the empty slotes 
		    // between m_parallel*m_maxLength and m_parallel*m_curMaxLength
		    // 
		    continue;
		}
		// locate the vector in weBank
		fn.weights        = helpers::getRawPointer(layer->_weBank())+(int)weIdx[i]*weDim;
		if (layer->flagWeMask())
		    fn.weightMask = helpers::getRawPointer(layer->_weMask())+(int)weIdx[i]*weDim;
		else
		    fn.weightMask = NULL;
		
		// locate the update vector in err (the err includes the dimension of normal input)
		fn.weightUpdates = helpers::getRawPointer(err)+i*inputSize+weIDDim;
		thrust::transform(thrust::counting_iterator<int>(0),
				  thrust::counting_iterator<int>(weDim),
				  layer->_weBank().begin()+weIdx[i]*weDim,
				  fn);
	    }
	    // debug
	    if(0){
		std::cout << "For debugging" << std::endl; 
	    }
	}
    }

    template <typename TDevice>
    void SteepestDescentOptimizer<TDevice>::_updateWeights(int fracLength)
    {
        /* Add 16-02-22 Wang: for WE updating */
	if (m_learningRate < 0){
	    // skip updateing the weights if learning rate is negative
	    
	}else{
	    
	   
	    // Update the parameter
	    internal::UpdateWeightFn_withMask updateWeightFn;
	    internal::UpdateWeightFn          updateWeightFn2;
	    
	    for (size_t i = 1; i < this->_neuralNetwork().layers().size(); ++i) {
        	layers::TrainableLayer<TDevice> *layer = 
		    dynamic_cast<layers::TrainableLayer<TDevice>*>(
					this->_neuralNetwork().layers()[i].get());
		layers::MDNLayer<TDevice> *mdnlayer = 
			dynamic_cast<layers::MDNLayer<TDevice>*>(
					this->_neuralNetwork().layers()[i].get());
		    
		if (!layer && !(mdnlayer && mdnlayer->flagTrainable()))
		    continue;
		
		// Adjust the gradient (only for trainble hidden layer, not MDN layer)
		if (this->_optOption()>0 && layer){
		    if (this->_optOption() == OPTIMIZATION_AVEGRAD){
			// average gradient over mini-batch
			thrust::transform(this->_curWeightUpdates()[i].begin(),
					  this->_curWeightUpdates()[i].end(),
					  this->_curWeightUpdates()[i].begin(),
					  internal::divideOpe((real_t)fracLength));
			    
		    }else if (this->_optOption() == OPTIMIZATION_ADAGRAD){
			// AdaGrad
			internal::AdaGradAccumulateUpdate adaUpdateFn;
			adaUpdateFn.fracLength = (real_t)fracLength;
			adaUpdateFn.weightMask = (layer->flagUseWeightMask()?
						  helpers::getRawPointer(layer->weightMask()):
						  NULL);
			thrust::for_each(
			     thrust::make_zip_iterator(
				thrust::make_tuple(this->_curWeightUpdates()[i].begin(),   
						   this->_weightStats()[i].begin(),
						   thrust::counting_iterator<int>(0))),
			     thrust::make_zip_iterator(
				thrust::make_tuple(this->_curWeightUpdates()[i].end(),
						   this->_weightStats()[i].end(),
						   thrust::counting_iterator<int>(0)+
						   m_weightDeltas[i].size())),
			     adaUpdateFn
			);
			/* Fatal error: 
			   m_weightDeltas.size() is used instead of m_weightDeltas[i].size()
			 */
			
		    }else if (this->_optOption() == OPTIMIZATION_ADAM){
			// Adam
			if (i == 1){
			    // Update the beta1 and beta2 every time
			    m_adamBeta1Accum = m_adamBeta1Accum * OP_ADAMBETA1;
			    m_adamBeta2Accum = m_adamBeta2Accum * OP_ADAMBETA2;
			}
			
			internal::AdamAccumulateUpdate adaUpdateFn;
			adaUpdateFn.fracLength = (real_t)fracLength;
			adaUpdateFn.beta1Accum = m_adamBeta1Accum;
			adaUpdateFn.beta2Accum = m_adamBeta2Accum;
			adaUpdateFn.mvBuffer   = helpers::getRawPointer(this->_weightStats()[i]);
			adaUpdateFn.weightMask = (layer->flagUseWeightMask()?
						  helpers::getRawPointer(layer->weightMask()):
						  NULL);
			
			thrust::for_each(
			     thrust::make_zip_iterator(
				thrust::make_tuple(this->_curWeightUpdates()[i].begin(),   
						   thrust::counting_iterator<int>(0))),
			     thrust::make_zip_iterator(
				thrust::make_tuple(this->_curWeightUpdates()[i].end(),
						   thrust::counting_iterator<int>(0)+
						   m_weightDeltas[i].size())),
			     adaUpdateFn
			);

			/* Fatal error: 
			   m_weightDeltas.size() is used instead of m_weightDeltas[i].size()
			 */

		    }else if(this->_optOption() == OPTIMIZATION_STOCHASTIC_ADAGRAD){
			// AdaGrad, but just accumulate the gradients
			internal::AdaGradAccumulate adaUpdateFn;
			adaUpdateFn.fracLength = (real_t)fracLength;
			adaUpdateFn.weightMask = (layer->flagUseWeightMask()?
						  helpers::getRawPointer(layer->weightMask()):
						  NULL);
			thrust::for_each(
			     thrust::make_zip_iterator(
				thrust::make_tuple(this->_curWeightUpdates()[i].begin(),   
						   this->_weightStats()[i].begin(),
						   thrust::counting_iterator<int>(0))),
			     thrust::make_zip_iterator(
				thrust::make_tuple(this->_curWeightUpdates()[i].end(),
						   this->_weightStats()[i].end(),
						   thrust::counting_iterator<int>(0)+
						   m_weightDeltas[i].size())),
			     adaUpdateFn
			);
			/* Fatal error: 
			   m_weightDeltas.size() is used instead of m_weightDeltas[i].size()
			 */

		    }else{
			// nothing
		    }
		}

		// if mask is utilized
		if (layer && layer->flagUseWeightMask()){
		    updateWeightFn.momentum      = m_momentum;
		    updateWeightFn.learningRate  = m_learningRate;
		    
		    if (layer->learningRate() > -0.5) // In fact, >= 0.0
			updateWeightFn.learningRate = layer->learningRate();

		    updateWeightFn.weights       = helpers::getRawPointer(layer->weights());
		    updateWeightFn.weightUpdates = helpers::getRawPointer(
							this->_curWeightUpdates()[i]);
		    updateWeightFn.weightDeltas  = helpers::getRawPointer(m_weightDeltas[i]);
		
		    // Add 0413 for Weight mask
		    updateWeightFn.weightMask    = helpers::getRawPointer(layer->weightMask());
		
		    thrust::transform(
				      thrust::counting_iterator<int>(0),
				      thrust::counting_iterator<int>((int)layer->weights().size()),
				      layer->weights().begin(),
				      updateWeightFn
				      );
		    
		// if mask is not used
		}else if(layer){
		    updateWeightFn2.momentum      = m_momentum;
		    updateWeightFn2.learningRate  = m_learningRate;
		    
		    if (layer->learningRate() > -0.5) // In fact, >= 0.0
			updateWeightFn2.learningRate = layer->learningRate();
		    

		    updateWeightFn2.weights       = helpers::getRawPointer(layer->weights());
		    updateWeightFn2.weightUpdates = helpers::getRawPointer(
							this->_curWeightUpdates()[i]);
		    updateWeightFn2.weightDeltas  = helpers::getRawPointer(m_weightDeltas[i]);
				
		    thrust::transform(
				    thrust::counting_iterator<int>(0),
				    thrust::counting_iterator<int>((int)layer->weights().size()),
				    layer->weights().begin(),
				    updateWeightFn2);
		    
		// trainable MDN layer
		}else if(mdnlayer && mdnlayer->flagTrainable()){
		    updateWeightFn2.momentum     = m_momentum;
		    updateWeightFn2.learningRate = m_learningRate;
		    //if (layer->learningRate() >= 0.0)
		    //     updateWeightFn2.learningRate = layer->learningRate();
		    

		    updateWeightFn2.weights       = helpers::getRawPointer(mdnlayer->weights());
		    updateWeightFn2.weightUpdates = helpers::getRawPointer(
							this->_curWeightUpdates()[i]);
		    updateWeightFn2.weightDeltas  = helpers::getRawPointer(m_weightDeltas[i]);
		    
		    //Cpu::real_vector temp1 = this->_curWeightUpdates()[i];
		    //Cpu::real_vector temp2 = mdnlayer->weights();
		    //temp1[0] = temp1[0];

		    thrust::transform(
				    thrust::counting_iterator<int>(0),
				    thrust::counting_iterator<int>((int)mdnlayer->weights().size()),
				    mdnlayer->weights().begin(),
				    updateWeightFn2);

		    //temp1 = this->_curWeightUpdates()[i];
		    //temp2 = mdnlayer->weights();
		    //temp1[0] = temp1[0];
		    
		}else{
		    throw std::runtime_error("Impossible Error");
		}
	    }
	}
    }

    template <typename TDevice>
    SteepestDescentOptimizer<TDevice>::SteepestDescentOptimizer(
        NeuralNetwork<TDevice> &neuralNetwork, data_sets::DataSet &trainingSet, 
	data_sets::DataSet &validationSet,
        data_sets::DataSet &testSet, int maxEpochs, int maxEpochsNoBest, 
	int validateEvery, int testEvery, 
        real_t learningRate, real_t momentum, real_t weLearningRate, 
	unsigned optOption,
	real_t adjustLRRate)
        : Optimizer<TDevice>(neuralNetwork, trainingSet, validationSet, testSet, 
			     maxEpochs, maxEpochsNoBest, validateEvery, testEvery,
			     optOption)
        , m_learningRate       (learningRate)
	, m_learningRateAdjust (adjustLRRate)
	, m_weLearningRate     (weLearningRate)
        , m_momentum           (momentum)
	, m_adamBeta1Accum     (1.0)
	, m_adamBeta2Accum     (1.0)
    {
        // intialize the weight deltas vectors with zeros
        m_weightDeltas = this->_curWeightUpdates();
        for (size_t i = 0; i < m_weightDeltas.size(); ++i)
            thrust::fill(m_weightDeltas[i].begin(), m_weightDeltas[i].end(), 0);
    }

    template <typename TDevice>
    SteepestDescentOptimizer<TDevice>::~SteepestDescentOptimizer()
    {
    }

    template <typename TDevice>
    void SteepestDescentOptimizer<TDevice>::exportState(const helpers::JsonDocument &jsonDoc) const
    {
        Optimizer<TDevice>::exportState(jsonDoc);

	if (m_momentum>0.0)
	    Optimizer<TDevice>::_exportWeights(jsonDoc, "steepest_descent_optimizer_weight_deltas", 
					       m_weightDeltas);
    }

    template <typename TDevice>
    void SteepestDescentOptimizer<TDevice>::importState(const helpers::JsonDocument &jsonDoc)
    {
        Optimizer<TDevice>::importState(jsonDoc);

	if (m_momentum>0.0)
	    Optimizer<TDevice>::_importWeights(jsonDoc, "steepest_descent_optimizer_weight_deltas", 
					       &m_weightDeltas);
    }

    template <typename TDevice>
    void SteepestDescentOptimizer<TDevice>::importParameter(const helpers::JsonDocument &jsonDoc)
    {
        Optimizer<TDevice>::importParameter(jsonDoc);
	// currently no need for momentum
        // Optimizer<TDevice>::_importWeights(jsonDoc, "steepest_descent_optimizer_weight_deltas", 
	// &m_weightDeltas);
    }

    template <typename TDevice>
    void SteepestDescentOptimizer<TDevice>::adjustLR()
    {
	//for(int i =0; i < decayTime; i++)
	//    m_learningRateDecay = m_learningRateDecay*0.1;
	//printf("\tAdjust the learning rate to %e", m_learningRate*m_learningRateDecay);
	m_learningRate *= m_learningRateAdjust;
    }
    
    template <typename TDevice>
    void SteepestDescentOptimizer<TDevice>::changeLR(real_t newLR)
    {
	m_learningRate = newLR;
    }
    
    template <typename TDevice>
    void SteepestDescentOptimizer<TDevice>::reinit()
    {
	// intialize the weight deltas vectors with zeros
        m_weightDeltas = this->_curWeightUpdates();
        for (size_t i = 0; i < m_weightDeltas.size(); ++i)
            thrust::fill(m_weightDeltas[i].begin(), m_weightDeltas[i].end(), 0);
	this->_reinit();

    }
    

    // explicit template instantiations
    template class SteepestDescentOptimizer<Cpu>;
    template class SteepestDescentOptimizer<Gpu>;

} // namespace optimizers
