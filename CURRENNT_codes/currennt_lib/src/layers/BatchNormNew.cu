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

#include "BatchNormNew.hpp"
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



    
} // anonymous namespace
} // namespace internal


namespace layers {

    template <typename TDevice, typename TActFn>
    BatchNormNewLayer<TDevice, TActFn>::BatchNormNewLayer(const helpers::JsonValue &layerChild, 
							const helpers::JsonValue &weightsSection, 
							Layer<TDevice> &precedingLayer,
							int             maxSeqLength,
							int             layerID)
        : TrainableLayer<TDevice>(layerChild, weightsSection, 0, 4,
				  precedingLayer, maxSeqLength, layerID)
    {
	// check
	if (this->size() != this->precedingLayer().size())
	    throw std::runtime_error("layernorm layer size != preceding layer size");
	
	/* ------- initialization ------- */
	this->__batchnorm_ini(weightsSection);
	
	/* ------- allocate memory ------ */
	this->__allocateLocalMem();
    }

    template <typename TDevice, typename TActFn>
    BatchNormNewLayer<TDevice, TActFn>::~BatchNormNewLayer()
    {
    }

    template <typename TDevice, typename TActFn>
    void BatchNormNewLayer<TDevice, TActFn>::__allocateLocalMem()
    {
	// mean, std
	Cpu::real_vector tmp;
	
	// all-one vector for vector summation
	tmp.resize(this->outputs().size()/this->size(), 1.0);
	m_oneVector = tmp;
	    
	// a tempopary buff
	m_buff      = this->outputs();
	m_outNormed = this->outputs();
    }

    template <typename TDevice, typename TActFn>
    void BatchNormNewLayer<TDevice, TActFn>::__clearLocalMem()
    {
	m_buff.clear();       m_buff.shrink_to_fit();
	m_outNormed.clear();  m_outNormed.shrink_to_fit();
	m_oneVector.clear();  m_oneVector.shrink_to_fit();	
    }

    template <typename TDevice, typename TActFn>
    void BatchNormNewLayer<TDevice, TActFn>::__batchnorm_ini(const helpers::JsonValue
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
	    // alpha = 1.0
	    thrust::fill(this->weights().begin(),
			 this->weights().begin() + this->size(),
			 BATCHNORM_GAMMA_INITIAL);
	    // beta, mean, std
	    thrust::fill(this->weights().begin() + this->size(),
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
    const std::string& BatchNormNewLayer<TDevice, TActFn>::type() const
    {
        static std::string s;
        if (s.empty()) {
            if (typeid(TActFn) == typeid(activation_functions::Tanh))
                s = "batchnorm_new_tanh";
            else if (typeid(TActFn) == typeid(activation_functions::Logistic))
                s = "batchnorm_new_logistic";
            else if (typeid(TActFn) == typeid(activation_functions::Identity))
                s = "batchnorm_new";
	    else if (typeid(TActFn) == typeid(activation_functions::Relu))
		s = "batchnorm_new_relu";
            else
                throw std::runtime_error("Unsupported activation function");
        }    
        return s;
    }


    
    template <typename TDevice, typename TActFn>
    void BatchNormNewLayer<TDevice, TActFn>::__batchnorm_computeForwardPass(const int nnState)
    {
	
	// pointer to [alpha, beta, mean, std]
	int transMatrixWeightNum = 0;
	    
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

	// copy previous layer's output to m_outNormed
	thrust::copy(this->precedingLayer().outputs().begin(),
		     this->precedingLayer().outputs().end(),
		     this->m_outNormed.begin());

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
    void BatchNormNewLayer<TDevice, TActFn>::computeForwardPass(const int nnState)
    {
	if (this->getSaveMemoryFlag())
	    throw std::runtime_error("Memory save mode should be turned off");
	
	this->__batchnorm_computeForwardPass(nnState);
	    
	
	// done
    }



    template <typename TDevice, typename TActFn>
    void BatchNormNewLayer<TDevice, TActFn>::__batchnorm_computeForwardPass(
	 const int timeStep, const int nnState, const int effTimeStart, const int effTimeEnd,
	 const int shiftIn,  const int shiftOut)
    {
	// Note: this function has not been test in online generation mode
	if (this->flagTrainingMode())
	    throw std::runtime_error("Error: batchnorm is not for online training");

	int transMatrixWeightNum = 0;
	thrust::copy(this->precedingLayer().outputs().begin() +
		     (effTimeStart * this->precedingLayer().size() - shiftIn),
		     this->precedingLayer().outputs().begin() +
		     (effTimeStart * this->precedingLayer().size() - shiftIn +
		      this->precedingLayer().size() * this->parallelSequences()),
		     this->m_outNormed.begin() + (effTimeStart * this->size() - shiftOut));

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
    void BatchNormNewLayer<TDevice, TActFn>::computeForwardPass(const int timeStep,
							       const int nnState)
    {
	int effTimeStart = timeStep * this->parallelSequences();
	int effTimeEnd   = (timeStep+1) * this->parallelSequences();
	
	// Pointer to the output of previous layer (input buffer)
	int shiftIn  = this->precedingLayer().outputBufPtrBias(timeStep * this->parallelSequences(),
							       nnState);
	// Pointer to the output of this layer
	int shiftOut = this->outputBufPtrBias(timeStep * this->parallelSequences(), nnState);


	this->__batchnorm_computeForwardPass(timeStep, nnState, effTimeStart, effTimeEnd,
					     shiftIn, shiftOut);
	// done
    }

    template <typename TDevice, typename TActFn>
    void BatchNormNewLayer<TDevice, TActFn>::__batchnorm_computeBackwardPass(const int nnState)
    {
	int maxFrameNum          = this->curMaxSeqLength() * this->parallelSequences();
	int maxDataNum           = maxFrameNum * this->size();
	int transMatrixWeightNum = 0;

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
    void BatchNormNewLayer<TDevice, TActFn>::computeBackwardPass(const int nnState)
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
	this->__batchnorm_computeBackwardPass(nnState);
	
	// propagate the gradients to the previous layer	
	thrust::copy(this->outputErrors().begin(), this->outputErrors().end(),
		     this->precedingLayer().outputErrors().begin());

	
    }



    template <typename TDevice, typename TActFn>
    void BatchNormNewLayer<TDevice, TActFn>::computeBackwardPass(const int timeStep,
								const int nnState)
    {
	
	if (this->getSaveMemoryFlag())
	    throw std::runtime_error("Memory save mode should be turned off");
	throw std::runtime_error("Error: batchnorm is not for online training");
	
    }


    
    template <typename TDevice, typename TActFn>
    void BatchNormNewLayer<TDevice, TActFn>::exportLayer(
	const helpers::JsonValue     &layersArray, 
	const helpers::JsonAllocator &allocator) const
    {
        TrainableLayer<TDevice>::exportLayer(layersArray, allocator);	
    }


    template <typename TDevice, typename TActFn>
    void BatchNormNewLayer<TDevice, TActFn>::reduceOutputBuffer()
    {
	
	this->resizeOutputBuffer(this->parallelSequences() * this->size());
	

	// clear the memory buff for unnecessary buffers in memory save mode
	m_oneVector.clear();
	m_oneVector.shrink_to_fit();
	m_buff.clear();
	m_buff.shrink_to_fit();
	m_outNormed = this->outputs();
	m_outNormed.shrink_to_fit();
	
	this->setSaveMemoryFlag(true);
	printf("\t[mem saved]");
    }
    
    template <typename TDevice, typename TActFn>
    int BatchNormNewLayer<TDevice, TActFn>::outputBufPtrBias(const int timeStepTimesParallel,
							    const int nnState)
    {
	if (this->getSaveMemoryFlag()){
	    return timeStepTimesParallel * this->size();
	}else{
	    return 0;
	}
    }	


    template <typename TDevice, typename TActFn>
    void BatchNormNewLayer<TDevice, TActFn>::clearAllBuffers()
    {
	this->clearOutputBuffer();
	this->__clearLocalMem();
    }

    template <typename TDevice, typename TActFn>
    void BatchNormNewLayer<TDevice, TActFn>::resizeAllBuffers(const int timeLength)
    {
	this->resizeOutputBuffer(timeLength * this->parallelSequences() * this->size());
	this->__allocateLocalMem();
    }

    
    // explicit template instantiations
    template class BatchNormNewLayer<Cpu, activation_functions::Tanh>;
    template class BatchNormNewLayer<Gpu, activation_functions::Tanh>;
    template class BatchNormNewLayer<Cpu, activation_functions::Logistic>;
    template class BatchNormNewLayer<Gpu, activation_functions::Logistic>;
    template class BatchNormNewLayer<Cpu, activation_functions::Identity>;
    template class BatchNormNewLayer<Gpu, activation_functions::Identity>;
    template class BatchNormNewLayer<Cpu, activation_functions::Relu>;
    template class BatchNormNewLayer<Gpu, activation_functions::Relu>;

} // namespace layers

