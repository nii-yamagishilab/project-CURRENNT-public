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

#include "LayerNorm.hpp"
#include "../helpers/getRawPointer.cuh"
#include "../helpers/Matrix.hpp"
#include "../activation_functions/Tanh.cuh"
#include "../activation_functions/Logistic.cuh"
#include "../activation_functions/Identity.cuh"
#include "../activation_functions/Relu.cuh"

#include "../helpers/misFuncs.hpp"
#include "../helpers/JsonClasses.hpp"
#include "../Configuration.hpp"
#include "../MacroDefine.hpp"
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/for_each.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>

#include <typeinfo>

#define LAYERNORM_GAMMA_INITIAL 0.01

namespace internal {
namespace {

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

    template <typename TActFn>
    struct ComputeOutputFn_weightNorm
    {
        int     layerSize;
        real_t  bias;
	real_t *weightNormFactors;
        const real_t *biasWeights;

        __host__ __device__ real_t operator() (real_t a, const int &outputIdx) const
        {
            // calculate indices
            int blockIdx = outputIdx % layerSize; 

	    // x * g / |v|
	    a = a / weightNormFactors[1] * weightNormFactors[0];
	    
            // add the bias
            a += bias * biasWeights[blockIdx];

            // apply the activation function
            real_t b = TActFn::fn(a);

            // store the activation
            return b;
        }
    };


    
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
    

struct PrepareGrad
    {
	int    layerSize;
	bool   alphaNotBeta;
	const char *patTypes;   
	real_t     *grad;
	real_t     *data;
	
	__host__ __device__ void operator() (const thrust::tuple<real_t&, int> &t) const
	{
	    int dataIdx = t.get<1>();
	    int timeIdx = dataIdx / layerSize;
	    if (patTypes[timeIdx] == PATTYPE_NONE){
		t.get<0>() = 0.0; // skip dummy node
	    }else{
		if (alphaNotBeta)
		    t.get<0>() = grad[dataIdx] * data[dataIdx];
		else
		    t.get<0>() = grad[dataIdx];
	    }
	}
    };

    

    struct LayerNormGetMeanStd
    {
	
	int         layerSize;
	real_t     *inputData;
	real_t     *meanStdBuf;
	real_t      stdConst;
	const char *patTypes;
	
	
	__host__ __device__ void operator() (const thrust::tuple<real_t&, int> &t) const
	{
	    
	    int timeIdx = t.get<1>();
	    
	    meanStdBuf[timeIdx * 2 + 0] = 0.0; // mean
	    meanStdBuf[timeIdx * 2 + 1] = 0.0; // std

	    
	    if (patTypes[timeIdx] == PATTYPE_NONE){
		return;
	    }else{
		// Welford's online algorithm to calculate mean and std
		real_t delta_1;
		real_t delta_2;
		for (int dimIdx = 0; dimIdx < layerSize; dimIdx++){
		    delta_1 = inputData[timeIdx*layerSize + dimIdx] - meanStdBuf[timeIdx * 2 + 0];
		    meanStdBuf[timeIdx * 2 + 0] +=
			(inputData[timeIdx * layerSize  + dimIdx] - meanStdBuf[timeIdx * 2 + 0]) /
			(dimIdx + 1.0);
		    delta_2 = inputData[timeIdx*layerSize + dimIdx] - meanStdBuf[timeIdx * 2 + 0];
		    meanStdBuf[timeIdx * 2 + 1] += (delta_1 * delta_2);
		}
		meanStdBuf[timeIdx * 2 + 1] =
		    sqrt(meanStdBuf[timeIdx * 2 + 1] / layerSize  + stdConst);
	    }
	}
    };


    template <typename TActFn>
    struct ComputeLayerNorm_Transform
    {
	int layerSize;

	const char *patTypes;
	
	real_t *data;
	real_t *outdata;
	real_t *meanStdBuf;
	real_t *scale;
	
	__host__ __device__ void operator() (const thrust::tuple<real_t&, int> &t) const
	{
	    int dataIdx = t.get<1>();
	    int dimIdx  = dataIdx % layerSize;
	    int timeIdx = dataIdx / layerSize;

	    if (patTypes[timeIdx] == PATTYPE_NONE){
		// skip dummy node
	    }else{
		
		// normalized data \hat{x}
		data[dataIdx] =
		    (data[dataIdx] - meanStdBuf[timeIdx * 2 + 0])/meanStdBuf[timeIdx * 2 + 1];
		
		// y =f(\alpha \hat{x} + \beta)
		outdata[dataIdx] = TActFn::fn(data[dataIdx] * scale[dimIdx] +
					      scale[dimIdx + layerSize]);
	    }
	}
    };


    struct LayerNormGradStatistics
    {
	
	int         layerSize;
	real_t     *normedData;
	real_t     *inputGrad;
	real_t     *scale;
	real_t     *statsBuff;
	const char *patTypes;
	
	
	__host__ __device__ void operator() (const thrust::tuple<real_t&, int> &t) const
	{
	    
	    int timeIdx = t.get<1>();
	    real_t tmp = 0.0;
	    statsBuff[timeIdx * 2 + 0] = 0.0; // 
	    statsBuff[timeIdx * 2 + 1] = 0.0; // 

	    if (patTypes[timeIdx] == PATTYPE_NONE){
		return;
	    }else{
		for (int dimIdx = 0; dimIdx < layerSize; dimIdx++){
		    tmp =  inputGrad[timeIdx * layerSize + dimIdx] * scale[dimIdx] / layerSize;
		    statsBuff[timeIdx * 2 + 0] += tmp;
		    statsBuff[timeIdx * 2 + 1] += tmp * normedData[timeIdx * layerSize + dimIdx];
		}
	    }
	}
    };


    struct ComputeLayerNormGrad_output
    {
	
	int layerSize;

	const char *patTypes;
	
	real_t *errors;
	real_t *outNormed;
	real_t *stats;
	real_t *meanStd;
	real_t *scale;
	
	
	__host__ __device__ void operator() (const thrust::tuple<real_t&, int> &t) const
	{
	    int dataIdx      = t.get<1>();
	    int dimIdx       = dataIdx % layerSize;
	    int timeIdx      = dataIdx / layerSize;
	    
	    if (patTypes[timeIdx] == PATTYPE_NONE){
		errors[dataIdx] = 0.0;
	    }else{
		errors[dataIdx] =
		    (errors[dataIdx] * scale[dimIdx]  -
		     stats[timeIdx * 2 + 0] - stats[timeIdx * 2 + 1] * outNormed[dataIdx] ) /
		    meanStd[timeIdx * 2 + 1];
	    }
	}
    };

    
} // anonymous namespace
} // namespace internal


namespace layers {

    template <typename TDevice, typename TActFn>
    LayerNormLayer<TDevice, TActFn>::LayerNormLayer(const helpers::JsonValue &layerChild, 
						    const helpers::JsonValue &weightsSection, 
						    Layer<TDevice> &precedingLayer,
						    int             maxSeqLength,
						    int             layerID)
        : TrainableLayer<TDevice>(layerChild, weightsSection, 0, 2,
				  precedingLayer, maxSeqLength, layerID)
    {
	// check
	if (this->size() != this->precedingLayer().size())
	    throw std::runtime_error("layernorm layer size != preceding layer size");
	
	// Initialization 
	this->__layernorm_ini(weightsSection);
	
	// Allocate memory
	this->__allocateLocalMem();
    }

    template <typename TDevice, typename TActFn>
    LayerNormLayer<TDevice, TActFn>::~LayerNormLayer()
    {
    }

    template <typename TDevice, typename TActFn>
    void LayerNormLayer<TDevice, TActFn>::__allocateLocalMem()
    {
	// mean, std
	Cpu::real_vector tmp;
	tmp.resize(this->outputs().size() / this->size() * 2, 0.0); 
	m_stats  = tmp;
	
	tmp.resize(this->outputs().size() / this->size(), 1.0);
	m_oneVector = tmp;
	
	m_buff      = this->outputs();
	m_outNormed = this->outputs();
    }

    template <typename TDevice, typename TActFn>
    void LayerNormLayer<TDevice, TActFn>::__clearLocalMem()
    {
	// mean, std
	m_buff.clear();       m_buff.shrink_to_fit();
	m_outNormed.clear();  m_outNormed.shrink_to_fit();
	m_oneVector.clear();  m_oneVector.shrink_to_fit();
	m_stats.clear();      m_stats.shrink_to_fit();
    }
	
    template <typename TDevice, typename TActFn>
    void LayerNormLayer<TDevice, TActFn>::__layernorm_ini(const helpers::JsonValue
							    &weightsSection)
    {
	// const in sqr(std + stc_const)
	m_stdConst  = 0.00001;
	
	// initialize the batachnom parameters
	if (weightsSection.isValid() && weightsSection->HasMember(this->name().c_str())) {
	    // read the weight (which has been conducted in TrainableLayer)
	    
	}else{	    
	    // alpha = 1.0
	    thrust::fill(this->weights().begin(),
			 this->weights().begin() + this->size(),
			 LAYERNORM_GAMMA_INITIAL);
	    // beta
	    thrust::fill(this->weights().begin() + this->size(),
			 this->weights().end(),
			 0.0);
	}		
	printf("\n\tLayer norm is used before activation function\n");
	
    }
    
    template <typename TDevice, typename TActFn>
    const std::string& LayerNormLayer<TDevice, TActFn>::type() const
    {
        static std::string s;
        if (s.empty()) {
            if (typeid(TActFn) == typeid(activation_functions::Tanh))
                s = "layernorm_tanh";
            else if (typeid(TActFn) == typeid(activation_functions::Logistic))
                s = "layernorm_logistic";
            else if (typeid(TActFn) == typeid(activation_functions::Identity))
                s = "layernorm";
	    else if (typeid(TActFn) == typeid(activation_functions::Relu))
		s = "layernorm_relu";
            else
                throw std::runtime_error("Unsupported activation function");
        }    
        return s;
    }


    template <typename TDevice, typename TActFn>
    void LayerNormLayer<TDevice, TActFn>::__layernorm_computeForwardPass(const int nnState)
    {
	
	// copy previous layer's output to m_outNormed
	thrust::copy(this->precedingLayer().outputs().begin(),
		     this->precedingLayer().outputs().end(),
		     this->m_outNormed.begin());

	// Total number of frames
	int maxFrameNum = this->curMaxSeqLength() * this->parallelSequences();
	// Total number of elements
	int maxDataNum  = maxFrameNum * this->size();
	
	{{
	    
	    {
		internal::LayerNormGetMeanStd fn1;
		fn1.layerSize  = this->size();
		fn1.stdConst   = this->m_stdConst;
		fn1.inputData  = helpers::getRawPointer(this->m_outNormed);
		fn1.meanStdBuf = helpers::getRawPointer(this->m_stats);
		fn1.patTypes   = helpers::getRawPointer(this->patTypes());
		
		thrust::for_each(
		  thrust::make_zip_iterator(
			thrust::make_tuple(this->m_stats.begin(), 
					   thrust::counting_iterator<int>(0))),
		  thrust::make_zip_iterator(
			thrust::make_tuple(this->m_stats.begin()             + maxFrameNum, 
					   thrust::counting_iterator<int>(0) + maxFrameNum)),
		  fn1);
	    }
	   
	    {
	       // Step5: normalize and scale the data, and pass it through the activation function
	       internal::ComputeLayerNorm_Transform<TActFn> fn2;
	       fn2.layerSize  = this->size();
	       fn2.patTypes   = helpers::getRawPointer(this->patTypes());       
	       fn2.data       = helpers::getRawPointer(this->m_outNormed);
	       fn2.outdata    = helpers::getRawPointer(this->outputs());
	       fn2.scale      = helpers::getRawPointer(this->weights());
	       fn2.meanStdBuf = helpers::getRawPointer(this->m_stats);
	       	   
	       thrust::for_each(
		thrust::make_zip_iterator(
			thrust::make_tuple(this->outputs().begin(), 
					   thrust::counting_iterator<int>(0))),
		thrust::make_zip_iterator(
			thrust::make_tuple(this->outputs().begin()           + maxDataNum, 
					   thrust::counting_iterator<int>(0) + maxDataNum)),
		fn2);
	   }
	    
	}}
	// done
	
    }
    
    
    template <typename TDevice, typename TActFn>
    void LayerNormLayer<TDevice, TActFn>::computeForwardPass(const int nnState)
    {
	if (this->getSaveMemoryFlag())
	    throw std::runtime_error("Memory save mode should be turned off");
	this->__layernorm_computeForwardPass(nnState);
    }



    template <typename TDevice, typename TActFn>
    void LayerNormLayer<TDevice, TActFn>::__layernorm_computeForwardPass(
	 const int timeStep, const int nnState, const int effTimeStart, const int effTimeEnd,
	 const int shiftIn,  const int shiftOut)
    {
	// Note: this function has not been test in online generation mode

	
	if (this->flagTrainingMode())
	    throw std::runtime_error("Error: layernorm is not implemented for online training");

	thrust::copy(this->precedingLayer().outputs().begin() +
		     (effTimeStart * this->precedingLayer().size() - shiftIn),
		     this->precedingLayer().outputs().begin() +
		     (effTimeStart * this->precedingLayer().size() - shiftIn +
		      this->precedingLayer().size() * this->parallelSequences()),
		     this->m_outNormed.begin() + (effTimeStart * this->size() - shiftOut));
	
	{{
	    int maxFrameNum = this->curMaxSeqLength() * this->parallelSequences();
	    int maxDataNum  = maxFrameNum * this->size();

	    {
		internal::LayerNormGetMeanStd fn1;
		fn1.layerSize  = this->size();
		fn1.stdConst   = this->m_stdConst;
		fn1.inputData  = helpers::getRawPointer(this->m_outNormed);
		fn1.meanStdBuf = helpers::getRawPointer(this->m_stats);
		fn1.patTypes   = helpers::getRawPointer(this->patTypes());
		
		thrust::for_each(
		  thrust::make_zip_iterator(
		     thrust::make_tuple(
			this->m_stats.begin()             + effTimeStart-shiftOut/this->size(), 
			thrust::counting_iterator<int>(0) + effTimeStart-shiftOut/this->size())),
		  thrust::make_zip_iterator(
		    thrust::make_tuple(
			this->m_stats.begin()             + effTimeEnd -shiftOut/this->size(), 
			thrust::counting_iterator<int>(0) + effTimeEnd -shiftOut/this->size())),
		  fn1);
	    }
	   
	    {
	       // Step5: normalize and scale the data, and pass it through the activation function
	       internal::ComputeLayerNorm_Transform<TActFn> fn2;
	       fn2.layerSize  = this->size();
	       fn2.patTypes   = helpers::getRawPointer(this->patTypes());
	       fn2.data       = helpers::getRawPointer(this->m_outNormed);
	       fn2.outdata    = helpers::getRawPointer(this->outputs());
	       fn2.scale      = helpers::getRawPointer(this->weights());
	       fn2.meanStdBuf = helpers::getRawPointer(this->m_stats);
	       	   
	       thrust::for_each(
		thrust::make_zip_iterator(
		  thrust::make_tuple(
		    this->outputs().begin() + effTimeStart * this->size() - shiftOut, 
		    thrust::counting_iterator<int>(0) + effTimeStart * this->size() - shiftOut)),
		thrust::make_zip_iterator(
		  thrust::make_tuple(
		    this->outputs().begin() + effTimeEnd * this->size() - shiftOut, 
		    thrust::counting_iterator<int>(0) + effTimeEnd * this->size() - shiftOut)),
		fn2);
	   }
	    
	}}
	// done
    }

    
    template <typename TDevice, typename TActFn>
    void LayerNormLayer<TDevice, TActFn>::computeForwardPass(const int timeStep,
							       const int nnState)
    {
	int effTimeStart = timeStep * this->parallelSequences();
	int effTimeEnd   = (timeStep+1) * this->parallelSequences();
	
	// Pointer to the output of previous layer (input buffer)
	int shiftIn  = this->precedingLayer().outputBufPtrBias(timeStep * this->parallelSequences(),
							       nnState);
	// Pointer to the output of this layer
	int shiftOut = this->outputBufPtrBias(timeStep * this->parallelSequences(), nnState);

	this->__layernorm_computeForwardPass(timeStep, nnState, effTimeStart, effTimeEnd,
					     shiftIn, shiftOut);
	// done
    }


    template <typename TDevice, typename TActFn>
    void LayerNormLayer<TDevice, TActFn>::__layernorm_computeBackwardPass(const int nnState)
    {
	int maxFrameNum          = this->curMaxSeqLength() * this->parallelSequences();
	int maxDataNum           = maxFrameNum * this->size();

	thrust::fill(m_oneVector.begin(),            m_oneVector.end(),            1.0);
	thrust::fill(m_buff.begin(),                 m_buff.end(),                 0.0);

	{{
	    // Step1. Calculate \deltaE/\delta{\alpha}
	    // Step2. Calculate \deltaE/\delta{\beta}
	    internal::PrepareGrad fn1;
	    fn1.layerSize    = this->size();
	    fn1.alphaNotBeta = true;
	    fn1.patTypes     = helpers::getRawPointer(this->patTypes());
	    fn1.grad         = helpers::getRawPointer(this->outputErrors());
	    fn1.data         = helpers::getRawPointer(this->m_outNormed);
	    
	    thrust::for_each(
		thrust::make_zip_iterator(
			thrust::make_tuple(this->m_buff.begin(), 
					   thrust::counting_iterator<int>(0))),
		thrust::make_zip_iterator(
			thrust::make_tuple(this->m_buff.begin() + maxDataNum, 
					   thrust::counting_iterator<int>(0) + maxDataNum)),
		fn1);
	    
	    helpers::Matrix<TDevice> onevec    (&this->m_oneVector, maxFrameNum, 1);
	    helpers::Matrix<TDevice> data      (&this->m_buff,      this->size(), maxFrameNum);
	    helpers::Matrix<TDevice> gradAlpha (&this->_weightUpdates(), this->size(), 1, 0);
	   gradAlpha.assignProduct(data, false, onevec, false);
	
	   
	   fn1.alphaNotBeta = false;	   
	   thrust::for_each(
		thrust::make_zip_iterator(
			thrust::make_tuple(this->m_buff.begin(), 
					   thrust::counting_iterator<int>(0))),
		thrust::make_zip_iterator(
			thrust::make_tuple(this->m_buff.begin() + maxDataNum, 
					   thrust::counting_iterator<int>(0) + maxDataNum)),
		fn1);
	   
	   helpers::Matrix<TDevice> gradBeta (&this->_weightUpdates(), this->size(),
					      1, this->size());
	   gradBeta.assignProduct(data, false, onevec, false);
	   
	}}

	{{
	    // step2. accumulate intermediate statistics
	    internal::LayerNormGradStatistics fn1;
	    fn1.layerSize  = this->size();
	    fn1.normedData = helpers::getRawPointer(this->m_outNormed);
	    fn1.inputGrad  = helpers::getRawPointer(this->outputErrors());
	    fn1.scale      = helpers::getRawPointer(this->weights());
	    fn1.statsBuff  = helpers::getRawPointer(this->m_buff);
	    fn1.patTypes   = helpers::getRawPointer(this->patTypes());
		
	    thrust::for_each(
	       thrust::make_zip_iterator(
		    thrust::make_tuple(this->m_stats.begin(), 
				       thrust::counting_iterator<int>(0))),
	       thrust::make_zip_iterator(
		    thrust::make_tuple(this->m_stats.begin()             + maxFrameNum, 
				       thrust::counting_iterator<int>(0) + maxFrameNum)),
	       fn1);
	}}
	
	{{
	   // Step3. Calculate \deltaE/\delta{x}
	   internal::ComputeLayerNormGrad_output fn2;
	   fn2.layerSize = this->size();
	   fn2.patTypes  = helpers::getRawPointer(this->patTypes());
	   fn2.errors    = helpers::getRawPointer(this->outputErrors());
	   fn2.outNormed = helpers::getRawPointer(m_outNormed);
	   fn2.meanStd   = helpers::getRawPointer(m_stats);
	   fn2.scale     = helpers::getRawPointer(this->weights());
	   fn2.stats     = helpers::getRawPointer(m_buff);
	   
	   thrust::for_each(
		thrust::make_zip_iterator(
			thrust::make_tuple(m_outNormed.begin(), 
					   thrust::counting_iterator<int>(0))),
		thrust::make_zip_iterator(
			thrust::make_tuple(m_outNormed.begin()               + maxDataNum, 
					   thrust::counting_iterator<int>(0) + maxDataNum)),
		fn2);
	}}
	// done
    }

    
    
    template <typename TDevice, typename TActFn>
    void LayerNormLayer<TDevice, TActFn>::computeBackwardPass(const int nnState)
    {
	if (this->getSaveMemoryFlag())
	    throw std::runtime_error("Memory save mode should be turned off");

	int maxFrameNum          = this->curMaxSeqLength() * this->parallelSequences();
	int maxDataNum           = maxFrameNum * this->size();
	
	thrust::fill(this->_weightUpdates().begin(), this->_weightUpdates().end(), 0.0);

	
	// \partial E / \partial_this_layer_output
	{{
            internal::ComputeDeltaFn<TActFn> fn;

            thrust::for_each(
               thrust::make_zip_iterator(
		  thrust::make_tuple(this->outputErrors().begin(),
				     this->outputs().begin())),
	       thrust::make_zip_iterator(
		  thrust::make_tuple(this->outputErrors().begin() + maxDataNum,
				     this->outputs().begin()      + maxDataNum)),
                fn);
	}}

	
	// calculate the gradients
	this->__layernorm_computeBackwardPass(nnState);
	
	// propagate the gradients to the previous layer
	thrust::copy(this->outputErrors().begin(), this->outputErrors().end(),
		     this->precedingLayer().outputErrors().begin());

    }



    template <typename TDevice, typename TActFn>
    void LayerNormLayer<TDevice, TActFn>::computeBackwardPass(const int timeStep,
								const int nnState)
    {
	
	if (this->getSaveMemoryFlag())
	    throw std::runtime_error("Memory save mode should be turned off");
	throw std::runtime_error("Error: layerNorm is not implemented for online training");
	
    }


    
    template <typename TDevice, typename TActFn>
    void LayerNormLayer<TDevice, TActFn>::exportLayer(
	const helpers::JsonValue     &layersArray, 
	const helpers::JsonAllocator &allocator) const
    {
        TrainableLayer<TDevice>::exportLayer(layersArray, allocator);
    }


    template <typename TDevice, typename TActFn>
    void LayerNormLayer<TDevice, TActFn>::reduceOutputBuffer()
    {
	
	this->resizeOutputBuffer(this->parallelSequences() * this->size());
	m_oneVector.clear();
	m_oneVector.shrink_to_fit();
	m_buff.clear();
	m_buff.shrink_to_fit();
	m_outNormed = this->outputs();
	m_outNormed.shrink_to_fit();
	    
	Cpu::real_vector tmp;
	tmp.resize(this->outputs().size()/this->size() * 2, 0.0); 
	m_stats = tmp;
	m_stats.shrink_to_fit();
	
	this->setSaveMemoryFlag(true);
	printf("\t[mem saved]");
    }
    
    template <typename TDevice, typename TActFn>
    int LayerNormLayer<TDevice, TActFn>::outputBufPtrBias(const int timeStepTimesParallel,
							    const int nnState)
    {
	if (this->getSaveMemoryFlag()){
	    return timeStepTimesParallel * this->size();
	}else{
	    return 0;
	}
    }	


    template <typename TDevice, typename TActFn>
    void LayerNormLayer<TDevice, TActFn>::clearAllBuffers()
    {
	this->clearOutputBuffer();
	this->__clearLocalMem();
    }

    template <typename TDevice, typename TActFn>
    void LayerNormLayer<TDevice, TActFn>::resizeAllBuffers(const int timeLength)
    {
	this->resizeOutputBuffer(timeLength * this->parallelSequences() * this->size());
	this->__allocateLocalMem();
    }

    
    // explicit template instantiations
    template class LayerNormLayer<Cpu, activation_functions::Tanh>;
    template class LayerNormLayer<Gpu, activation_functions::Tanh>;
    template class LayerNormLayer<Cpu, activation_functions::Logistic>;
    template class LayerNormLayer<Gpu, activation_functions::Logistic>;
    template class LayerNormLayer<Cpu, activation_functions::Identity>;
    template class LayerNormLayer<Gpu, activation_functions::Identity>;
    template class LayerNormLayer<Cpu, activation_functions::Relu>;
    template class LayerNormLayer<Gpu, activation_functions::Relu>;

} // namespace layers
