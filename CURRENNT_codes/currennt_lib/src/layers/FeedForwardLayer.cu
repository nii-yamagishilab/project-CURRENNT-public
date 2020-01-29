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

#include "FeedForwardLayer.hpp"
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

#define BATCHNORM_GAMMA_INITIAL 0.01

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

    /*struct GradientAverage
    {
	real_t timeStep;
	real_t *gradients;
        __host__ __device__ void operator() (const int &index) const
        {
	    *(gradients + index) = *(gradients + index)/timeStep;
        }
	};*/

    // 
    struct BatchSize
    {
	// over time t * parallel sentence
	const char *patTypes;
	
	__host__ __device__ real_t operator() (const thrust::tuple<const real_t&, int> &t) const
	{
	    int timeIdx = t.get<1>();
	    if (patTypes[timeIdx] == PATTYPE_NONE)
		return 0.0;// skip dummy node
	    else
		return 1.0;
	}
    };
    
    struct PrepareForMeanStd
    {
	int layerSize;
	bool   meanNotVar;

	
	const char *patTypes;   
	real_t     *data;
	real_t     *outdata;
	real_t     *mean;
	__host__ __device__ void operator() (const thrust::tuple<real_t&, int> &t) const
	{
	    int dataIdx = t.get<1>();
	    int timeIdx = dataIdx / layerSize;
	    int dimIdx  = dataIdx % layerSize;
	    if (patTypes[timeIdx] == PATTYPE_NONE){
		// skip dummy node
		outdata[dataIdx] = 0.0; //
	    }else{
		if (meanNotVar)
		    outdata[dataIdx] = data[dataIdx]; //
		else
		    outdata[dataIdx] = (data[dataIdx]-mean[dimIdx]) * (data[dataIdx]-mean[dimIdx]);
	    }
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

    struct GetStd
    {
	real_t  stdConst;
	__host__ __device__ void operator() (const thrust::tuple<real_t&, int> &t) const
	{
	    int dimIdx = t.get<1>();
	    t.get<0>() = sqrt(t.get<0>() +stdConst);
	}
    };

    struct AveMeanStd
    {
	real_t  learningRate;
	real_t *meanStdBuf;
	real_t  cnt;
	__host__ __device__ void operator() (const thrust::tuple<real_t&, int> &t) const
	{
	    int dimIdx = t.get<1>();
	    
	    // if learning rate is close to 0, don't update the mean/std
	    if (learningRate < 0.00001 && learningRate > -0.00001)
		return;
	    else
		meanStdBuf[dimIdx] += (t.get<0>() - meanStdBuf[dimIdx]) / cnt;
	}
    };
    
    struct WeightSqr
    {
	__host__ __device__ real_t operator() (const thrust::tuple<const real_t&, int> &t) const
	{
	    return t.get<0>() * t.get<0>();
	}	
    };


    struct WeightNormGGrad
    {
	real_t *weightNormFactors;
	
	__host__ __device__ real_t operator() (const thrust::tuple<const real_t&, const real_t&> &t) const
	{
	    // \partial E / \partial W * V / |V|
	    return (t.get<0>() * t.get<1>()) / weightNormFactors[1];
	}	
    };

    struct WeightNormScaleGradofV
    {
	real_t     *weightNormFactors;
	real_t      gradOfG;
	
	__host__ __device__ void operator() (const thrust::tuple<real_t&, const real_t&> &t) const
	{
	    // t.get<0>() grad of w
	    // t.get<1>() v
	    t.get<0>() = (weightNormFactors[0] / weightNormFactors[1]) *
		(t.get<0>() - gradOfG  / weightNormFactors[1] * t.get<1>());
	}
    };		
    
    struct WeightNormScaleGradOfInput
    {
	real_t     *weightNormFactors;
	
	__host__ __device__ void operator() (const thrust::tuple<real_t&, int> &t) const
	{
	    t.get<0>() = t.get<0>() * (weightNormFactors[0] / weightNormFactors[1]);
	}
    };		

    template <typename TActFn>
    struct ComputeBatchNorm_Transform
    {
	int layerSize;

	const char *patTypes;   
	real_t *data;
	real_t *outdata;
	real_t *meanStd;
	real_t *meanStdBuf;
	real_t *scale;
	bool    trainFlag;
	__host__ __device__ void operator() (const thrust::tuple<real_t&, int> &t) const
	{
	    int dataIdx = t.get<1>();
	    int dimIdx  = dataIdx % layerSize;
	    int timeIdx = dataIdx / layerSize;
	    int varIdx  = dimIdx  + layerSize;
	    if (patTypes[timeIdx] == PATTYPE_NONE){
		// skip dummy node
	    }else{
		// \hat{x} = (x - \mu) / \sigma
		if (trainFlag)
		    data[dataIdx] = (data[dataIdx]-meanStd[dimIdx])/meanStd[varIdx];
		else
		    data[dataIdx] = (data[dataIdx]-meanStdBuf[dimIdx])/meanStdBuf[varIdx];

		// y =f(\alpha \hat{x} + \beta)
		outdata[dataIdx]   = TActFn::fn(data[dataIdx] * scale[dimIdx] + scale[varIdx]);
	    }
	}
    };
    
    struct ComputeBatchGradient_output
    {
	
	int layerSize;

	const char *patTypes;   
	real_t *errors;
	real_t *outNormed;
	real_t *meanStd;
	real_t *scale;
	real_t *scaleGrad;	
	real_t  batchSize;
	
	__host__ __device__ void operator() (const thrust::tuple<real_t&, int> &t) const
	{
	    int dataIdx      = t.get<1>();
	    int dimIdx       = dataIdx % layerSize;
	    int timeIdx      = dataIdx / layerSize;
	    
	    if (patTypes[timeIdx] == PATTYPE_NONE){
		errors[dataIdx] = 0.0;
	    }else{
		// gradient =
		// alpha / std * (\deltaE/\delta{y} - \deltaE/\deltaBeta / batchSize -
		//                \deltaE/\deltaAlpha * dataNormed / batchSize)
		errors[dataIdx] = ((errors[dataIdx] -
				    scaleGrad[dimIdx] * outNormed[dataIdx]/ batchSize -
				    scaleGrad[dimIdx + layerSize] / batchSize ) *
				   scale[dimIdx] / meanStd[dimIdx + layerSize]);
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

    // Additional weight due to batch normalization
    int weightForBatchNorm(const helpers::JsonValue &layerChild){
	if (layerChild->HasMember("batchnorm") && ((*layerChild)["batchnorm"].GetInt()))
	    return 3; // alpha, mean, std (the beta-bias has been allocated)
	else if (layerChild->HasMember("layernorm") && ((*layerChild)["layernorm"].GetInt()))
	    return 1; // alpha, mean, std (the beta-bias has been allocated)
	else if (layerChild->HasMember("weightnorm") && ((*layerChild)["weightnorm"].GetInt()))
	    return 2; // g, |v|, unnecessary memory space will be allocated
	else
	    return 0;
    }
    
    template <typename TDevice, typename TActFn>
    FeedForwardLayer<TDevice, TActFn>::FeedForwardLayer(const helpers::JsonValue &layerChild, 
							const helpers::JsonValue &weightsSection, 
							Layer<TDevice> &precedingLayer,
							int             maxSeqLength,
							int             layerID)
        : TrainableLayer<TDevice>(layerChild,
				  weightsSection,
				  1,
				  weightForBatchNorm(layerChild),
				  precedingLayer,
				  maxSeqLength,
				  layerID)
    {

	/* ----- Read configuration ---- */
	// Initialization 
	m_batchNorm = ((layerChild->HasMember("batchnorm") &&
			((*layerChild)["batchnorm"].GetInt())))? true : false;
	
	m_weightNorm = ((layerChild->HasMember("weightnorm") &&
			((*layerChild)["weightnorm"].GetInt())))? true : false;

	m_layerNorm = ((layerChild->HasMember("layernorm") &&
			((*layerChild)["layernorm"].GetInt())))? true : false;

	if (m_batchNorm && (m_weightNorm  || m_layerNorm)){
	    printf("\n\tOnly batchnorm will be used");
	    m_weightNorm = false;
	    m_layerNorm  = false;
	}else if (m_weightNorm && m_layerNorm){
	    printf("\n\tOnly layer norm will be used");
	    m_weightNorm = false;
	}else if (m_weightNorm){
	    printf("\n\tWarninig: weight normalization is not fully test");
	}

	/* ------- initialization ------- */
	if (m_batchNorm)
	    this->__batchnorm_ini(weightsSection);
	else if (m_layerNorm)
	    this->__layernorm_ini(weightsSection);
	else if (m_weightNorm)
	    this->__weightnorm_ini(weightsSection);

	/* ------- allocate memory ------ */
	this->__allocateLocalMem();
    }

    template <typename TDevice, typename TActFn>
    FeedForwardLayer<TDevice, TActFn>::~FeedForwardLayer()
    {
    }

    template <typename TDevice, typename TActFn>
    void FeedForwardLayer<TDevice, TActFn>::__allocateLocalMem()
    {
	// mean, std
	if (this->m_batchNorm){
	    Cpu::real_vector tmp;
	
	    // all-one vector for vector summation
	    tmp.resize(this->outputs().size()/this->size(), 1.0);
	    m_oneVector = tmp;
	    
	    // a tempopary buff
	    m_buff      = this->outputs();
	    m_outNormed = this->outputs();

	}else if (this->m_layerNorm){
	    
	    // mean, std
	    Cpu::real_vector tmp;
	    tmp.resize(this->outputs().size()/this->size() * 2, 0.0); 
	    m_stats  = tmp;

	    tmp.resize(this->outputs().size()/this->size(), 1.0);
	    m_oneVector = tmp;
	    
	    m_buff      = this->outputs();
	    m_outNormed = this->outputs();

	}else{
	    m_buff.clear();
	    m_outNormed.clear();
	    m_oneVector.clear();
	    m_stats.clear();
	}
    }

    template <typename TDevice, typename TActFn>
    void FeedForwardLayer<TDevice, TActFn>::__clearLocalMem()
    {
	// mean, std
	if (this->m_batchNorm){
	    m_buff.clear();       m_buff.shrink_to_fit();
	    m_outNormed.clear();  m_outNormed.shrink_to_fit();
	    m_oneVector.clear();  m_oneVector.shrink_to_fit();
	}else if (this->m_layerNorm){
	    m_buff.clear();       m_buff.shrink_to_fit();
	    m_outNormed.clear();  m_outNormed.shrink_to_fit();
	    m_oneVector.clear();  m_oneVector.shrink_to_fit();
	    m_stats.clear();      m_stats.shrink_to_fit();
	}else{
	    // do nothing
	}
    }

    template <typename TDevice, typename TActFn>
    void FeedForwardLayer<TDevice, TActFn>::__batchnorm_ini(const helpers::JsonValue
							    &weightsSection)
    {
	// Normalization:
	//  x_outNormed = (x - mean) / \sqrt(std + stdConst) * alpha + beta
	// Memory allocation:
	// [transformation_matrix, alpha, beta, batch_mean, batch_std]
	    	    
	// initialization
	m_stdConst  = 0.001; m_batchCnt  = 0.0; m_preEpoch  = 1;
	    
	// initialize the batachnom parameters
	if (weightsSection.isValid() && weightsSection->HasMember(this->name().c_str())) {
	    // read the weight (which has been conducted in TrainableLayer)		
	}else{
	    // initialize 
	    int transMatrixWeightNum = this->size() * this->precedingLayer().size();
	    // alpha = 1.0
	    thrust::fill(this->weights().begin() + transMatrixWeightNum,
			 this->weights().begin() + transMatrixWeightNum + this->size(),
			 BATCHNORM_GAMMA_INITIAL);
	    // beta, mean, std
	    thrust::fill(this->weights().begin() + transMatrixWeightNum + this->size(),
			 this->weights().end(),
			 0.0);
	}
	const Configuration &config = Configuration::instance();
	m_batchNormGenUseTrainMV = config.batchnorm_genmode();


	// mean, std
	Cpu::real_vector tmp;
	tmp.resize(this->size() * 2, 0.0); 
	m_stats  = tmp;
	
	printf("\n\tBatchnorm is used before activation function\n");
	if (m_batchNormGenUseTrainMV)
	    printf("\tBatchnorm will use training data mean/std in generation stage\n");
	
    }
	
    template <typename TDevice, typename TActFn>
    void FeedForwardLayer<TDevice, TActFn>::__layernorm_ini(const helpers::JsonValue
							    &weightsSection)
    {
	m_stdConst  = 0.00001;
	
	// initialize the batachnom parameters
	if (weightsSection.isValid() && weightsSection->HasMember(this->name().c_str())) {
	    // read the weight (which has been conducted in TrainableLayer)		
	}else{
	    // initialize 
	    int transMatrixWeightNum = this->size() * this->precedingLayer().size();
	    
	    // alpha = 1.0
	    thrust::fill(this->weights().begin() + transMatrixWeightNum,
			 this->weights().begin() + transMatrixWeightNum + this->size(),
			 BATCHNORM_GAMMA_INITIAL);
	    // beta
	    thrust::fill(this->weights().begin() + transMatrixWeightNum + this->size(),
			 this->weights().end(),
			 0.0);
	}		
	printf("\n\tLayer norm is used before activation function\n");
	
    }

    template <typename TDevice, typename TActFn>
    void FeedForwardLayer<TDevice, TActFn>::__weightnorm_ini(const helpers::JsonValue
							     &weightsSection)
    {
	printf("\n\tWeightnorm is used\n");
    }



    
    template <typename TDevice, typename TActFn>
    const std::string& FeedForwardLayer<TDevice, TActFn>::type() const
    {
        static std::string s;
        if (s.empty()) {
            if (typeid(TActFn) == typeid(activation_functions::Tanh))
                s = "feedforward_tanh";
            else if (typeid(TActFn) == typeid(activation_functions::Logistic))
                s = "feedforward_logistic";
            else if (typeid(TActFn) == typeid(activation_functions::Identity))
                s = "feedforward_identity";
	    else if (typeid(TActFn) == typeid(activation_functions::Relu))
		s = "feedforward_relu";
            else
                throw std::runtime_error("Unsupported activation function");
        }    
        return s;
    }



    
    template <typename TDevice, typename TActFn>
    void FeedForwardLayer<TDevice, TActFn>::__batchnorm_computeForwardPass(const int nnState)
    {
	
	// pointer to [alpha, beta, mean, std]
	int transMatrixWeightNum = this->size() * this->precedingLayer().size();
	    
	// Re-initialize the batch mean and variance
	if (this->flagTrainingMode() && m_preEpoch > 0 &&
	    m_preEpoch != this->getCurrTrainingEpoch()){
	    // Only re-initialize the mean and std for each epoch
	    // always update the mean, std for each epoch
	    m_batchCnt = 0;
	    thrust::fill(this->weights().begin() + transMatrixWeightNum + 2 * this->size(),
			 this->weights().end(),  0.0);
	    m_preEpoch = this->getCurrTrainingEpoch();
	}

	// Wx
	{{
		helpers::Matrix<TDevice> weightsMatrix  (&this->weights(),                  
							 this->precedingLayer().size(),
							 this->size());
		helpers::Matrix<TDevice> plOutputsMatrix(&this->precedingLayer().outputs(), 
							 this->precedingLayer().size(), 
							 this->curMaxSeqLength() * 
							 this->parallelSequences());
		helpers::Matrix<TDevice> outputsMatrix  (&this->m_outNormed,                 
							 this->size(),                  
							 this->curMaxSeqLength() * 
							 this->parallelSequences());
		outputsMatrix.assignProduct(weightsMatrix, true, plOutputsMatrix, false);
	}}	    

	// normalize the data
	m_batchCnt++;
	
	{{
	   int maxFrameNum = this->curMaxSeqLength() * this->parallelSequences();
	   int maxDataNum  = maxFrameNum * this->size();

	       
	   {
	       // Step1. calculate the batch size
	       //        For parallel sentences, there is dummy node. BatchSize should not count it.
	       internal::BatchSize fn0;
	       fn0.patTypes = helpers::getRawPointer(this->patTypes());
	       m_batchSize  = thrust::transform_reduce(
				thrust::make_zip_iterator(
				    thrust::make_tuple(
					this->m_buff.begin(), 
					thrust::counting_iterator<int>(0))),
				thrust::make_zip_iterator(
				    thrust::make_tuple(
					this->m_buff.begin()              + maxFrameNum, 
					thrust::counting_iterator<int>(0) + maxFrameNum)),
				fn0, (real_t)0.0, thrust::plus<real_t>());
	       
	       thrust::fill(this->m_oneVector.begin(), this->m_oneVector.end(), 1.0/m_batchSize);
	   }

	   {   
	       // Step2. accumulate the mean
	       internal::PrepareForMeanStd fn1;
	       fn1.layerSize  = this->size();
	       fn1.meanNotVar = true;
	       fn1.mean       = NULL;
	       fn1.patTypes   = helpers::getRawPointer(this->patTypes());
	       fn1.data       = helpers::getRawPointer(this->m_outNormed);
	       fn1.outdata    = helpers::getRawPointer(this->m_buff);	   
	       thrust::for_each(
		 thrust::make_zip_iterator(
			thrust::make_tuple(this->outputs().begin(), 
					   thrust::counting_iterator<int>(0))),
		 thrust::make_zip_iterator(
			thrust::make_tuple(this->outputs().begin()           + maxDataNum, 
					   thrust::counting_iterator<int>(0) + maxDataNum)),
		 fn1);
	   
	       helpers::Matrix<TDevice> onevec  (&this->m_oneVector, maxFrameNum,  1);
	       helpers::Matrix<TDevice> data    (&this->m_buff,      this->size(), maxFrameNum);
	       helpers::Matrix<TDevice> meanVec (&this->m_stats,     this->size(), 1);
	       meanVec.assignProduct(data, false, onevec, false);
	       
	       // Step3. accumulate the var
	       fn1.meanNotVar = false;
	       fn1.mean       = helpers::getRawPointer(this->m_stats);; 
	       thrust::for_each(
		thrust::make_zip_iterator(
			thrust::make_tuple(this->outputs().begin(), 
					   thrust::counting_iterator<int>(0))),
		thrust::make_zip_iterator(
			thrust::make_tuple(this->outputs().begin()           + maxDataNum, 
					   thrust::counting_iterator<int>(0) + maxDataNum)),
		fn1);
	       
	       helpers::Matrix<TDevice> data2   (&this->m_buff,  this->size(), maxFrameNum);
	       helpers::Matrix<TDevice> stdVec  (&this->m_stats, this->size(), 1, this->size());
	       stdVec.assignProduct(data2, false, onevec, false);
	       
	       internal::GetStd fn3;
	       fn3.stdConst = m_stdConst;
	       thrust::for_each(
		thrust::make_zip_iterator(
			thrust::make_tuple(m_stats.begin() + this->size(), 
					   thrust::counting_iterator<int>(0))),
		thrust::make_zip_iterator(
			thrust::make_tuple(m_stats.begin() + this->size() * 2, 
					   thrust::counting_iterator<int>(0) + this->size())),
		fn3);
	   }
	   
	   {
	       // Step4. accumulate the mean and std, for generation stage
	       //  online average into weight buffer
	       if (this->flagTrainingMode() && nnState == NN_STATE_GAN_NOGAN_TRAIN){
		   internal::AveMeanStd fn5;
		   fn5.learningRate = this->learningRate();
		   fn5.meanStdBuf = (helpers::getRawPointer(this->weights()) +
				     transMatrixWeightNum + this->size() * 2);
		   fn5.cnt        = m_batchCnt;
		   thrust::for_each(
		     thrust::make_zip_iterator(
			thrust::make_tuple(m_stats.begin(), 
					   thrust::counting_iterator<int>(0))),
		     thrust::make_zip_iterator(
			thrust::make_tuple(m_stats.begin() + this->size() * 2, 
					   thrust::counting_iterator<int>(0) + this->size() * 2)),
		     fn5);
	       }
	   }

	   {
	       // Step5: normalize and scale the data, and pass it through the activation function
	       internal::ComputeBatchNorm_Transform<TActFn> fn2;
	       fn2.layerSize = this->size();
	       fn2.patTypes  = helpers::getRawPointer(this->patTypes());
	       fn2.data      = helpers::getRawPointer(this->m_outNormed);
	       fn2.outdata   = helpers::getRawPointer(this->outputs());
	       fn2.scale     = helpers::getRawPointer(this->weights()) + transMatrixWeightNum;
	       fn2.meanStd   = helpers::getRawPointer(this->m_stats);
	       fn2.meanStdBuf= (helpers::getRawPointer(this->weights()) +
				transMatrixWeightNum + this->size() * 2);

	       if (this->flagTrainingMode() && (nnState == NN_STATE_GAN_NOGAN_TRAIN))
		   fn2.trainFlag = true;
	       else if (m_batchNormGenUseTrainMV == 0)
		   fn2.trainFlag = true;
	       else 
		   fn2.trainFlag = false;
	       
	       //fn2.trainFlag = true;
	       // Note: batchnorm here always assume this->m_outNormed (one or multiple sequences)
	       //  as a single batch. Thus in this computeForwardPass() mode, the data in the batch
	       //  are all visiable. Thus, normalization can be directly conducted on the input
	       //  data matrix no matter whether this is the training or val/generateion stages.
	       // However, in computeForwardPass(timeStep), generatation is conducted
	       //  step by step, and normalization must be conducted given the mean/std accumulated
	       //  during training stage.
	       
	   
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
    void FeedForwardLayer<TDevice, TActFn>::__layernorm_computeForwardPass(const int nnState)
    {
	
	// pointer to [alpha, beta, mean, std]
	int transMatrixWeightNum = this->size() * this->precedingLayer().size();
	    
	
	// Wx
	{{
	    helpers::Matrix<TDevice> weightsMatrix  (&this->weights(),                  
						     this->precedingLayer().size(),
						     this->size());
	    helpers::Matrix<TDevice> plOutputsMatrix(&this->precedingLayer().outputs(), 
						     this->precedingLayer().size(), 
						     this->curMaxSeqLength() * 
						     this->parallelSequences());
	    helpers::Matrix<TDevice> outputsMatrix  (&this->m_outNormed,                 
						     this->size(),                  
						     this->curMaxSeqLength() * 
						     this->parallelSequences());
	    outputsMatrix.assignProduct(weightsMatrix, true, plOutputsMatrix, false);
	}}	    

	
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
	       
	       fn2.scale      = helpers::getRawPointer(this->weights()) + transMatrixWeightNum;
	       
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
    void FeedForwardLayer<TDevice, TActFn>::__weightnorm_computeForwardPass(const int nnState)
    {
	int weight_size = this->precedingLayer().size() * this->size();
	
	{{
	     // Compute |v|
	     {		
		internal::WeightSqr fn0;
		m_weightNormVal  = thrust::transform_reduce(
				thrust::make_zip_iterator(
				    thrust::make_tuple(
					this->weights().begin(), 
					thrust::counting_iterator<int>(0))),
				thrust::make_zip_iterator(
				    thrust::make_tuple(
					this->weights().begin()           + weight_size, 
					thrust::counting_iterator<int>(0) + weight_size)),
				fn0, (real_t)0.0, thrust::plus<real_t>());
		m_weightNormVal = std::sqrt(m_weightNormVal);
		
		thrust::fill(this->weights().begin() + weight_size + this->size() + 1,
			     this->weights().begin() + weight_size + this->size() + 2,
			     m_weightNormVal);
	     }
	    
	    // The conventional feedforward part
	    // collect outputs from preceding layer
	    {
            helpers::Matrix<TDevice> weightsMatrix  (&this->weights(),                  
						     this->precedingLayer().size(), this->size());
	    
	    
            helpers::Matrix<TDevice> plOutputsMatrix(&this->precedingLayer().outputs(), 
						     this->precedingLayer().size(), 
						     this->curMaxSeqLength() * 
						     this->parallelSequences());

            helpers::Matrix<TDevice> outputsMatrix  (&this->_outputs(),                 
						     this->size(),                  
						     this->curMaxSeqLength() * 
						     this->parallelSequences());

            outputsMatrix.assignProduct(weightsMatrix, true, plOutputsMatrix, false);
	    }

	    // calculate the outputs of the layer
	    {
            internal::ComputeOutputFn_weightNorm<TActFn> fn;
            fn.layerSize        = this->size();
            fn.bias             = this->bias();
	    fn.weightNormFactors= (helpers::getRawPointer(this->weights()) + 
				   this->size() * this->precedingLayer().size() + this->size());
            fn.biasWeights      = (helpers::getRawPointer(this->weights()) + 
				   this->size() * this->precedingLayer().size());

            thrust::transform(
                this->_outputs().begin(),
                (this->_outputs().begin() + 
		 this->curMaxSeqLength() * this->parallelSequences() * this->size()),
                thrust::counting_iterator<int>(0),
                this->_outputs().begin(),
                fn);
	   }
       }}
	// done
    }


    template <typename TDevice, typename TActFn>
    void FeedForwardLayer<TDevice, TActFn>::__computeForwardPass(const int nnState)
    {	
	// The conventional feedforward part
	// collect outputs from preceding layer
	{{
            helpers::Matrix<TDevice> weightsMatrix  (&this->weights(),                  
						     this->precedingLayer().size(), this->size());
	    
	    
            helpers::Matrix<TDevice> plOutputsMatrix(&this->precedingLayer().outputs(), 
						     this->precedingLayer().size(), 
						     this->curMaxSeqLength() * 
						     this->parallelSequences());

            helpers::Matrix<TDevice> outputsMatrix  (&this->_outputs(),                 
						     this->size(),                  
						     this->curMaxSeqLength() * 
						     this->parallelSequences());

            outputsMatrix.assignProduct(weightsMatrix, true, plOutputsMatrix, false);
	 }}

	    
	// calculate the outputs of the layer
	{{
            internal::ComputeOutputFn<TActFn> fn;
            fn.layerSize        = this->size();
            fn.bias             = this->bias();
            fn.biasWeights      = (helpers::getRawPointer(this->weights()) + 
				   this->size() * this->precedingLayer().size());

            thrust::transform(
                this->_outputs().begin(),
                (this->_outputs().begin() + 
		 this->curMaxSeqLength() * this->parallelSequences() * this->size()),
                thrust::counting_iterator<int>(0),
                this->_outputs().begin(),
                fn
                );
	}}
    }

    
    template <typename TDevice, typename TActFn>
    void FeedForwardLayer<TDevice, TActFn>::computeForwardPass(const int nnState)
    {
	if (this->getSaveMemoryFlag())
	    throw std::runtime_error("Memory save mode should be turned off");
	
	if (m_batchNorm){
	    // batch norm
	    this->__batchnorm_computeForwardPass(nnState);
	    
	}else if (m_layerNorm){
	    // layer norm
	    this->__layernorm_computeForwardPass(nnState);
	    
	}else if (m_weightNorm){
	    // weight norm
	    this->__weightnorm_computeForwardPass(nnState);
		
	}else{
	    // conventional feedforward layer
	    this->__computeForwardPass(nnState);
	}
	// done
    }



    template <typename TDevice, typename TActFn>
    void FeedForwardLayer<TDevice, TActFn>::__batchnorm_computeForwardPass(
	 const int timeStep, const int nnState, const int effTimeStart, const int effTimeEnd,
	 const int shiftIn,  const int shiftOut)
    {

	// batch normalization mode
	if (this->flagTrainingMode())
	    throw std::runtime_error("Error: batchnorm is not for online training");

	int transMatrixWeightNum = this->size() * this->precedingLayer().size();
	
	// Wx
	{{
            helpers::Matrix<TDevice> weightsMatrix  (&this->weights(),                  
						     this->precedingLayer().size(), this->size());
	    
            helpers::Matrix<TDevice> plOutputsMatrix(&this->precedingLayer().outputs(), 
						     this->precedingLayer().size(), 
						     this->parallelSequences(),
						     (effTimeStart * this->precedingLayer().size()
						      - shiftIn));

            helpers::Matrix<TDevice> outputsMatrix  (&this->m_outNormed,
						     this->size(), 
						     this->parallelSequences(),
						     (effTimeStart * this->size()
						      - shiftOut));

            outputsMatrix.assignProduct(weightsMatrix, true, plOutputsMatrix, false);
	}}

	// Normalize using the averaged mean and std
	{{
	    internal::ComputeBatchNorm_Transform<TActFn> fn2;
	    fn2.layerSize = this->size();
	    fn2.patTypes  = helpers::getRawPointer(this->patTypes());
	    fn2.data      = helpers::getRawPointer(this->m_outNormed);
	    fn2.outdata   = helpers::getRawPointer(this->outputs());
	    fn2.scale     = helpers::getRawPointer(this->weights()) + transMatrixWeightNum;
	    fn2.meanStd   = helpers::getRawPointer(this->m_stats);
	    fn2.meanStdBuf= (helpers::getRawPointer(this->weights()) +
			     transMatrixWeightNum + this->size() * 2);
	    fn2.trainFlag = false;
	   
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
	}}
	// done
    }
    

    template <typename TDevice, typename TActFn>
    void FeedForwardLayer<TDevice, TActFn>::__layernorm_computeForwardPass(
	 const int timeStep, const int nnState, const int effTimeStart, const int effTimeEnd,
	 const int shiftIn,  const int shiftOut)
    {

	// batch normalization mode
	if (this->flagTrainingMode())
	    throw std::runtime_error("Error: layernorm is not implemented for online training");

	// to be checked
	
	// pointer to [alpha, beta, mean, std]
	int transMatrixWeightNum = this->size() * this->precedingLayer().size();
	
	// Wx
	{{
	    helpers::Matrix<TDevice> weightsMatrix  (&this->weights(),                  
						     this->precedingLayer().size(), this->size());
	    
	    helpers::Matrix<TDevice> plOutputsMatrix(&this->precedingLayer().outputs(), 
						     this->precedingLayer().size(), 
						     this->parallelSequences(),
						     (effTimeStart * this->precedingLayer().size()
						      - shiftIn));
	    
	    helpers::Matrix<TDevice> outputsMatrix  (&this->m_outNormed,                 
						     this->size(),                  
						     this->parallelSequences(),
						     (effTimeStart * this->size()
						      - shiftOut));
	    
	    outputsMatrix.assignProduct(weightsMatrix, true, plOutputsMatrix, false);
	}}	    

	
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
	       
	       fn2.scale      = helpers::getRawPointer(this->weights()) + transMatrixWeightNum;
	       
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
    void FeedForwardLayer<TDevice, TActFn>::__weightnorm_computeForwardPass(
	 const int timeStep, const int nnState, const int effTimeStart, const int effTimeEnd,
	 const int shiftIn,  const int shiftOut)
    {
	printf("Not implemented for weightNorm");
    }

    template <typename TDevice, typename TActFn>
    void FeedForwardLayer<TDevice, TActFn>::__computeForwardPass(
	 const int timeStep, const int nnState, const int effTimeStart, const int effTimeEnd,
	 const int shiftIn,  const int shiftOut)
    {
	// weight norm mode
	//  assume weight has been normalized in the training stage
	// normal mode
	    
	// collect outputs from preceding layer
	{{
            helpers::Matrix<TDevice> weightsMatrix  (&this->weights(),                  
						     this->precedingLayer().size(), this->size());
	    
	    
            helpers::Matrix<TDevice> plOutputsMatrix(&this->precedingLayer().outputs(), 
						     this->precedingLayer().size(), 
						     this->parallelSequences(),
						     (effTimeStart * this->precedingLayer().size()
						      - shiftIn));

            helpers::Matrix<TDevice> outputsMatrix  (&this->_outputs(),                 
						     this->size(), 
						     this->parallelSequences(),
						     (effTimeStart * this->size()
						      - shiftOut));

            outputsMatrix.assignProduct(weightsMatrix, true, plOutputsMatrix, false);
	}}

	    // calculate the outputs of the layer
	{{
            internal::ComputeOutputFn<TActFn> fn;
            fn.layerSize        = this->size();
            fn.bias             = this->bias();
            fn.biasWeights      = (helpers::getRawPointer(this->weights()) + 
				   this->size() * this->precedingLayer().size());

            thrust::transform(
		this->_outputs().begin() + effTimeStart * this->size() - shiftOut,
		this->_outputs().begin() + effTimeEnd   * this->size() - shiftOut,
		thrust::counting_iterator<int>(0),
		this->_outputs().begin() + effTimeStart * this->size() - shiftOut,
		fn);
	}}
	// 
    }
    
    template <typename TDevice, typename TActFn>
    void FeedForwardLayer<TDevice, TActFn>::computeForwardPass(const int timeStep,
							       const int nnState)
    {
	int effTimeStart = timeStep * this->parallelSequences();
	int effTimeEnd   = (timeStep+1) * this->parallelSequences();
	
	// Pointer to the output of previous layer (input buffer)
	int shiftIn  = this->precedingLayer().outputBufPtrBias(timeStep * this->parallelSequences(),
							       nnState);
	// Pointer to the output of this layer
	int shiftOut = this->outputBufPtrBias(timeStep * this->parallelSequences(), nnState);


	if (m_batchNorm){
	    this->__batchnorm_computeForwardPass(timeStep, nnState, effTimeStart, effTimeEnd,
						 shiftIn, shiftOut);
	}else if (m_layerNorm){
	    this->__layernorm_computeForwardPass(timeStep, nnState, effTimeStart, effTimeEnd,
						 shiftIn, shiftOut);
	}else if (m_weightNorm){
	    this->__weightnorm_computeForwardPass(timeStep, nnState, effTimeStart, effTimeEnd,
						 shiftIn, shiftOut);
	}else{
	    this->__computeForwardPass(timeStep, nnState, effTimeStart, effTimeEnd,
				       shiftIn, shiftOut);
	}
	// done
    }

    template <typename TDevice, typename TActFn>
    void FeedForwardLayer<TDevice, TActFn>::__batchnorm_computeBackwardPass(const int nnState)
    {
	int maxFrameNum          = this->curMaxSeqLength() * this->parallelSequences();
	int maxDataNum           = maxFrameNum * this->size();
	int transMatrixWeightNum = this->size() * this->precedingLayer().size();

	thrust::fill(m_oneVector.begin(),            m_oneVector.end(),            1.0);
	thrust::fill(m_buff.begin(),                 m_buff.end(),                 0.0);

	{{
	    // Step1. Calculate \deltaE/\delta{\alpha}
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
	    helpers::Matrix<TDevice> gradAlpha (&this->_weightUpdates(), this->size(), 1,
						transMatrixWeightNum);
	   gradAlpha.assignProduct(data, false, onevec, false);
	
	   // Step2. Calculate \deltaE/\delta{\beta}
	   fn1.alphaNotBeta = false;	   
	   thrust::for_each(
		thrust::make_zip_iterator(
			thrust::make_tuple(this->m_buff.begin(), 
					   thrust::counting_iterator<int>(0))),
		thrust::make_zip_iterator(
			thrust::make_tuple(this->m_buff.begin() + maxDataNum, 
					   thrust::counting_iterator<int>(0) + maxDataNum)),
		fn1);
	   
	   helpers::Matrix<TDevice> gradBeta (&this->_weightUpdates(), this->size(),1,
					      transMatrixWeightNum + this->size());
	   gradBeta.assignProduct(data, false, onevec, false);
	   
	}}

	{{
	   // Step3. Calculate \deltaE/\delta{x}
	   internal::ComputeBatchGradient_output fn2;
	   fn2.layerSize = this->size();
	   fn2.patTypes  = helpers::getRawPointer(this->patTypes());
	   fn2.errors    = helpers::getRawPointer(this->outputErrors());
	   fn2.outNormed = helpers::getRawPointer(m_outNormed);
	   fn2.meanStd   = helpers::getRawPointer(m_stats);
	   fn2.scale     = helpers::getRawPointer(this->weights())        + transMatrixWeightNum;
	   fn2.scaleGrad = helpers::getRawPointer(this->_weightUpdates()) + transMatrixWeightNum;
	   fn2.batchSize = m_batchSize;
	   
	   thrust::for_each(
		thrust::make_zip_iterator(
			thrust::make_tuple(m_outNormed.begin(), 
					   thrust::counting_iterator<int>(0))),
		thrust::make_zip_iterator(
			thrust::make_tuple(m_outNormed.begin() + maxDataNum, 
					   thrust::counting_iterator<int>(0) + maxDataNum)),
		fn2);
	}}
	// done
    }
    


    template <typename TDevice, typename TActFn>
    void FeedForwardLayer<TDevice, TActFn>::__weightnorm_computeBackwardPass_p1(const int nnState)
    {
	int maxFrameNum          = this->curMaxSeqLength() * this->parallelSequences();
	int maxDataNum           = maxFrameNum * this->size();
	int transMatrixWeightNum = this->size() * this->precedingLayer().size();
	
	Layer<TDevice> *pl = dynamic_cast<Layer<TDevice>*>(&this->precedingLayer());
	
	internal::WeightNormScaleGradOfInput fn2;
	fn2.weightNormFactors= (helpers::getRawPointer(this->weights()) + 
				transMatrixWeightNum + this->size());
		
	thrust::for_each(
	  thrust::make_zip_iterator(
		thrust::make_tuple(pl->outputErrors().begin(), 
				   thrust::counting_iterator<int>(0))),
	  thrust::make_zip_iterator(
		thrust::make_tuple(pl->outputErrors().begin()        + maxDataNum, 
				   thrust::counting_iterator<int>(0) + maxDataNum)),
	  fn2);
    }


    template <typename TDevice, typename TActFn>
    void FeedForwardLayer<TDevice, TActFn>::__weightnorm_computeBackwardPass_p2(const int nnState)
    {
	int maxFrameNum          = this->curMaxSeqLength() * this->parallelSequences();
	int maxDataNum           = maxFrameNum * this->size();
	int transMatrixWeightNum = this->size() * this->precedingLayer().size();

	{{
	    // gradient to scaling factor G
	    internal::WeightNormGGrad fn1;
	    fn1.weightNormFactors= (helpers::getRawPointer(this->weights()) + 
				    transMatrixWeightNum + this->size());

	    m_weightNormGrad =
		thrust::transform_reduce(
			thrust::make_zip_iterator(
				thrust::make_tuple(
					this->_weightUpdates().begin(), 
					this->weights().begin())),
				thrust::make_zip_iterator(
				    thrust::make_tuple(
					this->_weightUpdates().begin() + transMatrixWeightNum, 
					this->weights().begin()        + transMatrixWeightNum)),
				fn1, (real_t)0.0, thrust::plus<real_t>());


	    thrust::fill(this->_weightUpdates().begin()+transMatrixWeightNum+this->size(),
			 this->_weightUpdates().begin()+transMatrixWeightNum+this->size()+1,
			 m_weightNormGrad);

	       
	    internal::WeightNormScaleGradofV fn2;
	    fn2.weightNormFactors= (helpers::getRawPointer(this->weights()) + 
				    transMatrixWeightNum + this->size());
	    fn2.gradOfG = m_weightNormGrad;
	       
	    thrust::for_each(thrust::make_zip_iterator(
				    thrust::make_tuple(
					this->_weightUpdates().begin(), 
					this->weights().begin())),
				thrust::make_zip_iterator(
				    thrust::make_tuple(
					this->_weightUpdates().begin() + transMatrixWeightNum, 
					this->weights().begin()        + transMatrixWeightNum)),
				fn2);
	}}
    }

    
    template <typename TDevice, typename TActFn>
    void FeedForwardLayer<TDevice, TActFn>::__computeBackwardPass_bias(const int nnState)
    {
	// compute the bias weight updates
	{{
            internal::ComputeBiasWeightUpdateFn fn;
            fn.layerSize     = this->size();
            fn.patternsCount = this->curMaxSeqLength() * this->parallelSequences();
            fn.bias          = this->bias();
            fn.deltas        = helpers::getRawPointer(this->outputErrors());

            thrust::transform(
                thrust::counting_iterator<int>(0),
                thrust::counting_iterator<int>(0) + this->size(),
                this->_weightUpdates().begin() + this->precedingLayer().size() * this->size(),
                fn);
	}}
    }


    template <typename TDevice, typename TActFn>
    void FeedForwardLayer<TDevice, TActFn>::__layernorm_computeBackwardPass(const int nnState)
    {
	int maxFrameNum          = this->curMaxSeqLength() * this->parallelSequences();
	int maxDataNum           = maxFrameNum * this->size();
	int transMatrixWeightNum = this->size() * this->precedingLayer().size();

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
	    helpers::Matrix<TDevice> gradAlpha (&this->_weightUpdates(), this->size(), 1,
						transMatrixWeightNum);
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
	   
	   helpers::Matrix<TDevice> gradBeta (&this->_weightUpdates(), this->size(),1,
					      transMatrixWeightNum + this->size());
	   gradBeta.assignProduct(data, false, onevec, false);
	   
	}}

	{{
	    // step2. accumulate intermediate statistics
	    internal::LayerNormGradStatistics fn1;
	    fn1.layerSize  = this->size();
	    fn1.normedData = helpers::getRawPointer(this->m_outNormed);
	    fn1.inputGrad  = helpers::getRawPointer(this->outputErrors());
	    fn1.scale      = helpers::getRawPointer(this->weights())  + transMatrixWeightNum;
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
	   fn2.scale     = helpers::getRawPointer(this->weights())        + transMatrixWeightNum;
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
    void FeedForwardLayer<TDevice, TActFn>::computeBackwardPass(const int nnState)
    {
	if (this->getSaveMemoryFlag())
	    throw std::runtime_error("Memory save mode should be turned off");

	int maxFrameNum          = this->curMaxSeqLength() * this->parallelSequences();
	int maxDataNum           = maxFrameNum * this->size();
	int transMatrixWeightNum = this->size() * this->precedingLayer().size();
	
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

	
	// For batch-normalization, calculate the gradients w.r.t. gamma, beta, Wx+b
	if (m_batchNorm) {
	    this->__batchnorm_computeBackwardPass(nnState);
	}else if (m_layerNorm){
	    this->__layernorm_computeBackwardPass(nnState);
	}
	
	
	// Back-propagate the error to the preceding layer
	{{
	    // why only to Trainablelayer?
		//            TrainableLayer<TDevice> *pl = 
		// dynamic_cast<TrainableLayer<TDevice>*>(&this->precedingLayer());
		
	    Layer<TDevice> *pl = dynamic_cast<Layer<TDevice>*>(&this->precedingLayer());

	    if (pl) {
                helpers::Matrix<TDevice> weightsMatrix (&this->weights(),      
							pl->size(),   
							this->size());
                helpers::Matrix<TDevice> plErrorsMatrix(&pl->outputErrors(),   
							pl->size(),   
							this->curMaxSeqLength() * 
							this->parallelSequences());
                helpers::Matrix<TDevice> deltasMatrix  (&this->outputErrors(), 
							this->size(), 
							this->curMaxSeqLength() * 
							this->parallelSequences());
		
                plErrorsMatrix.assignProduct(weightsMatrix, false, deltasMatrix, false);
            }else{
		printf("\nGradients cannot be propagated after %s. ", this->name().c_str());
		throw std::runtime_error("Backpropagation error");
	    }
	    
	    if (m_weightNorm){
		this->__weightnorm_computeBackwardPass_p1(nnState);
	    }
	    
	}}

	
	// \partial E / \partial W
	{{
            helpers::Matrix<TDevice> weightUpdatesMatrix(&this->_weightUpdates(),           
							 this->precedingLayer().size(), 
							 this->size());

            helpers::Matrix<TDevice> plOutputsMatrix    (&this->precedingLayer().outputs(), 
							 this->precedingLayer().size(), 
							 this->curMaxSeqLength() * 
							 this->parallelSequences());
	    
            helpers::Matrix<TDevice> deltasMatrix       (&this->outputErrors(),             
							 this->size(),                  
							 this->curMaxSeqLength() * 
							 this->parallelSequences());

            weightUpdatesMatrix.assignProduct(plOutputsMatrix, false, deltasMatrix, true);
	}}

	
	if (m_batchNorm || m_layerNorm){
	    // bias has been updated
	    
	}else if (m_weightNorm){
	    
	    // gradients w.r.t weight
	    this->__weightnorm_computeBackwardPass_p2(nnState);
	    this->__computeBackwardPass_bias(nnState);
	    
	}else{
	    // conventional feedforward layer
	    this->__computeBackwardPass_bias(nnState);
	}

	// #2018101202
    }



    template <typename TDevice, typename TActFn>
    void FeedForwardLayer<TDevice, TActFn>::computeBackwardPass(const int timeStep,
								const int nnState)
    {
	
	if (this->getSaveMemoryFlag())
	    throw std::runtime_error("Memory save mode should be turned off");
	if (m_batchNorm)
	    throw std::runtime_error("Error: batchnorm is not for online training");
	if (m_weightNorm)
	    throw std::runtime_error("Error: weightnorm is not implemented for online training");
	if (m_layerNorm)
	    throw std::runtime_error("Error: layerNorm is not implemented for online training");
	
	// start and end time steps 
	int effTimeStart = timeStep     * this->parallelSequences();
	int effTimeEnd   = (timeStep+1) * this->parallelSequences();
	
	// initialize the gradients at the first step (the last time step in backward computation)
	if (timeStep == this->curMaxSeqLength() - 1)
	    thrust::fill(this->_weightUpdates().begin(), this->_weightUpdates().end(), 0.0);
 
	// compute deltas
	{{
            internal::ComputeDeltaFn<TActFn> fn;
            thrust::for_each(
               thrust::make_zip_iterator(
		thrust::make_tuple(this->outputErrors().begin() + effTimeStart * this->size(),
				   this->outputs().begin()      + effTimeStart * this->size())),
	       thrust::make_zip_iterator(
		thrust::make_tuple(this->outputErrors().begin() + effTimeEnd   * this->size(),
				   this->outputs().begin()      + effTimeEnd   * this->size())),
                fn);
	}}


	// get the pointer of the preceding layer
	Layer<TDevice> *pl = dynamic_cast<Layer<TDevice>*>(&this->precedingLayer());
	
	if (!pl){
	    printf("\nGradients cannot be propagated after %s. ", this->name().c_str());
	    throw std::runtime_error("Backpropagation error");		
	}
	
	// Back-propagate the error to the preceding layer
	{{
                helpers::Matrix<TDevice> plErrorsMatrix(&pl->outputErrors(),   
							pl->size(),    
							this->parallelSequences(),
							effTimeStart * pl->size());
                helpers::Matrix<TDevice> weightsMatrix (&this->weights(),      
							pl->size(),   
							this->size());
                helpers::Matrix<TDevice> deltasMatrix  (&this->outputErrors(), 
							this->size(),
							this->parallelSequences(),
							effTimeStart * this->size());
                plErrorsMatrix.assignProduct(weightsMatrix, false, deltasMatrix, false);
	}}

	// Remember to reset the gradients
	if (timeStep == this->curMaxSeqLength() - 1)
	    thrust::fill(this->_weightUpdates().begin(), this->_weightUpdates().end(), 0.0);
	
	// compute the input weight updates
	{{
            helpers::Matrix<TDevice> weightUpdatesMatrix(&this->_weightUpdates(),           
							 pl->size(), 
							 this->size());

            helpers::Matrix<TDevice> plOutputsMatrix    (&this->precedingLayer().outputs(), 
							 pl->size(), 
							 this->parallelSequences(),
							 effTimeStart * pl->size());
	    
            helpers::Matrix<TDevice> deltasMatrix       (&this->outputErrors(),             
							 this->size(), 
							 this->parallelSequences(),
							 effTimeStart * this->size());

            weightUpdatesMatrix.addProduct(plOutputsMatrix, false, deltasMatrix, true);
	}}

	// compute the bias weight updates
	{{
            internal::ComputeBiasWeightUpdateFn_online fn;
            fn.layerSize     = this->size();
            fn.patternsCount = this->parallelSequences();
            fn.bias          = this->bias();
            fn.deltas        = (helpers::getRawPointer(this->outputErrors()) +
				effTimeStart * this->size());
	    fn.bias_grad     = (helpers::getRawPointer(this->_weightUpdates()) +
				this->precedingLayer().size() * this->size());
	    
            thrust::transform(
                thrust::counting_iterator<int>(0),
                thrust::counting_iterator<int>(0) + this->size(),
                this->_weightUpdates().begin() + this->precedingLayer().size() * this->size(),
                fn);
	}}

    }


    
    template <typename TDevice, typename TActFn>
    void FeedForwardLayer<TDevice, TActFn>::exportLayer(
	const helpers::JsonValue     &layersArray, 
	const helpers::JsonAllocator &allocator) const
    {
        TrainableLayer<TDevice>::exportLayer(layersArray, allocator);
	if (m_batchNorm)
	    (*layersArray)[layersArray->Size() - 1].AddMember("batchnorm",
							      (int)m_batchNorm, allocator);
	if (m_layerNorm)
	    (*layersArray)[layersArray->Size() - 1].AddMember("layernorm",
							      (int)m_layerNorm, allocator);
	if (m_weightNorm)
	    (*layersArray)[layersArray->Size() - 1].AddMember("weightnorm",
							      (int)m_weightNorm, allocator);
	
    }


    template <typename TDevice, typename TActFn>
    void FeedForwardLayer<TDevice, TActFn>::reduceOutputBuffer()
    {
	
	this->resizeOutputBuffer(this->parallelSequences() * this->size());
	
	if (m_batchNorm){
	    // clear the memory buff for unnecessary buffers in memory save mode
	    m_oneVector.clear();
	    m_oneVector.shrink_to_fit();
	    m_buff.clear();
	    m_buff.shrink_to_fit();
	    m_outNormed = this->outputs();
	    m_outNormed.shrink_to_fit();
	}

	if (m_layerNorm){
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
	}
	
	this->setSaveMemoryFlag(true);
	printf("\t[mem saved]");
    }
    
    template <typename TDevice, typename TActFn>
    int FeedForwardLayer<TDevice, TActFn>::outputBufPtrBias(const int timeStepTimesParallel,
							    const int nnState)
    {
	if (this->getSaveMemoryFlag()){
	    return timeStepTimesParallel * this->size();
	}else{
	    return 0;
	}
    }	


    template <typename TDevice, typename TActFn>
    void FeedForwardLayer<TDevice, TActFn>::clearAllBuffers()
    {
	this->clearOutputBuffer();
	this->__clearLocalMem();
    }

    template <typename TDevice, typename TActFn>
    void FeedForwardLayer<TDevice, TActFn>::resizeAllBuffers(const int timeLength)
    {
	this->resizeOutputBuffer(timeLength * this->parallelSequences() * this->size());
	this->__allocateLocalMem();
    }

    
    template <typename TDevice, typename TActFn>
    void FeedForwardLayer<TDevice, TActFn>::logAllBuffers(
			helpers::vecPoolManager<TDevice> &vecPoolMng,
			bool flag_add)
    {
	// for output buffer
	Layer<TDevice>::logAllBuffers(vecPoolMng, flag_add);
	
	if (this->m_batchNorm){
	    // for m_buff
	    vecPoolMng.addOrRemoveNewVec(this->size(), flag_add);
	    // for outNormed
	    vecPoolMng.addOrRemoveNewVec(this->size(), flag_add);
	    // for oneVector
	    vecPoolMng.addOrRemoveNewVec(this->size()/this->size(), flag_add);

	}else if (this->m_layerNorm){
	    // for m_stats
	    vecPoolMng.addOrRemoveNewVec(this->size()/this->size() * 2, flag_add);
	    // for m_buff
	    vecPoolMng.addOrRemoveNewVec(this->size(), flag_add);
	    // for outNormed
	    vecPoolMng.addOrRemoveNewVec(this->size(), flag_add);
	    // for oneVector
	    vecPoolMng.addOrRemoveNewVec(this->size()/this->size(), flag_add);
	    
	}else{
	    // nothing for weight normalization
	}	
    }
    
    template <typename TDevice, typename TActFn>
    void FeedForwardLayer<TDevice, TActFn>::swapAllBuffers(
			helpers::vecPoolManager<TDevice> &vecPoolMng,
			bool flag_get)
    {
	Layer<TDevice>::swapAllBuffers(vecPoolMng, flag_get);

	if (this->m_batchNorm){
	    // for m_buff
	    vecPoolMng.getSwapVector(m_buff,
				     this->getLayerID(), this->size(), flag_get);
	    // for outNormed
	    vecPoolMng.getSwapVector(m_outNormed,
				     this->getLayerID(), this->size(), flag_get);
	    // for oneVector
	    vecPoolMng.getSwapVector(m_oneVector,
				     this->getLayerID(), 1,            flag_get);

	    if (flag_get)
		thrust::fill(m_oneVector.begin(), m_oneVector.end(), 1.0);
	    
	}else if (this->m_layerNorm){
	    // for m_buff
	    vecPoolMng.getSwapVector(m_stats,
				     this->getLayerID(), 2,            flag_get);
	    // for m_buff
	    vecPoolMng.getSwapVector(m_buff,
				     this->getLayerID(), this->size(), flag_get);
	    // for outNormed
	    vecPoolMng.getSwapVector(m_outNormed,
				     this->getLayerID(), this->size(), flag_get);
	    // for oneVector
	    vecPoolMng.getSwapVector(m_oneVector,
				     this->getLayerID(), 1,            flag_get);

	    if (flag_get){
		thrust::fill(m_oneVector.begin(), m_oneVector.end(), 1.0);
		thrust::fill(m_stats.begin(), m_stats.end(), 0.0);
	    }
	}else{
	    // nothing for weight normalization
	}
	
	
    }


    
    
    // explicit template instantiations
    template class FeedForwardLayer<Cpu, activation_functions::Tanh>;
    template class FeedForwardLayer<Gpu, activation_functions::Tanh>;
    template class FeedForwardLayer<Cpu, activation_functions::Logistic>;
    template class FeedForwardLayer<Gpu, activation_functions::Logistic>;
    template class FeedForwardLayer<Cpu, activation_functions::Identity>;
    template class FeedForwardLayer<Gpu, activation_functions::Identity>;
    template class FeedForwardLayer<Cpu, activation_functions::Relu>;
    template class FeedForwardLayer<Gpu, activation_functions::Relu>;

} // namespace layers




// #2018101201
	    /* This part is merged with the above part since pl is a general layer pointer
	    else{
		//Add 16-02-22 Wang: for WE updating 
		// If the input layer will udpate the word vectors
		// we need to propagate the errors back to the input layer
		// Add 17-05-01 for MiddleoutputLayer
		Layer<TDevice> *pl2 = dynamic_cast<Layer<TDevice>*>(&this->precedingLayer());
		if (this->precedingLayer().inputWeUpdate() ||
		    this->precedingLayer().type() == "middleoutput" ||
		    this->precedingLayer().type() == "featmatch"){
		    helpers::Matrix<TDevice> weightsMatrix (&this->weights(),      
							    pl2->size(),  
							    this->size());

		    helpers::Matrix<TDevice> plErrorsMatrix(&pl2->outputErrors(),  
							    pl2->size(),  
							    this->curMaxSeqLength() * 
							    this->parallelSequences());

		    helpers::Matrix<TDevice> deltasMatrix  (&this->outputErrors(), 
							    this->size(), 
							    this->curMaxSeqLength() * 
							    this->parallelSequences());
		    plErrorsMatrix.assignProduct(weightsMatrix, false, deltasMatrix, false);
		}else{
		    printf("\nGradients cannot be propagated after %s. ", this->name().c_str());
		    throw std::runtime_error("Backpropagation error");
		}
		}*/

// #2018101202
	/* Gradient averaging ?
	      if (this->_optOpt()){
	      {{
	      internal::GradientAverage fn;
	      fn.timeStep = (real_t)(this->curMaxSeqLength() * this->parallelSequences());
	      fn.gradients= helpers::getRawPointer(this->_weightUpdates());		
	      thrust::for_each(
	      thrust::counting_iterator<int>(0),
	      thrust::counting_iterator<int>(0) + this->_weightUpdates().size(),
	      fn);
	      }}
	      }*/
