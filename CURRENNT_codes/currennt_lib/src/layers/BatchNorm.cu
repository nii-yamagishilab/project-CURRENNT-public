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

#ifdef _MSC_VER
#   pragma warning (disable: 4244) // thrust/iterator/iterator_adaptor.h(121): warning C4244: '+=' : conversion from '__int64' to 'int', possible loss of data
#endif


#include "BatchNorm.hpp"
#include "../helpers/getRawPointer.cuh"
#include "../helpers/Matrix.hpp"
#include "../helpers/JsonClasses.hpp"

#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/for_each.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>


#include "../Configuration.hpp"
#include <cmath>

#define FEEDBACKLAYER_DEBUG 0

namespace internal{
namespace {

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
	real_t     *buff;
	real_t     *data;
	
	__host__ __device__ void operator() (const thrust::tuple<real_t&, int> &t) const
	{
	    int dataIdx = t.get<1>();
	    int timeIdx = dataIdx / layerSize;
	    if (patTypes[timeIdx] == PATTYPE_NONE){
		buff[dataIdx] = 0.0; // skip dummy node
	    }else{
		if (alphaNotBeta)
		    buff[dataIdx] = grad[dataIdx] * data[dataIdx];
		else
		    buff[dataIdx] = grad[dataIdx];
	    }
	}
    };

    struct GetStd
    {
	real_t  stdConst;
	real_t *meanStd;
	__host__ __device__ void operator() (const thrust::tuple<real_t&, int> &t) const
	{
	    int dimIdx = t.get<1>();
	    meanStd[dimIdx] = sqrt(meanStd[dimIdx]+stdConst);
	}
    };

    struct AveMeanStd
    {
	real_t *meanStd;
	real_t *meanStdBuf;
	real_t  cnt;
	__host__ __device__ void operator() (const thrust::tuple<real_t&, int> &t) const
	{
	    int dimIdx = t.get<1>();
	    meanStdBuf[dimIdx] += (meanStd[dimIdx] - meanStdBuf[dimIdx]) / cnt;
	}
    };
    
    
    struct ComputeBatchMeanStd
    {
	int layerSize;
	int frameNM;            // frame * parallel
	real_t  stdConst;

	const char *patTypes;   
	real_t     *data;
	real_t     *meanStd;
	
	real_t     *meanStdBuf;	
	real_t      batchCnt;
	bool        trainFlag;
	
	__host__ __device__ void operator() (const thrust::tuple<real_t&, int> &t) const
	{
	    int dimIdx = t.get<1>();

	    real_t mean   = 0.0;
	    real_t var    = 0.0;
	    real_t delta  = 0.0;
	    real_t delta2 = 0.0;

	    int cnt = 0;
	    int idx = 0;

	    // online algorithm to compute mean, var
	    for (int n = 0; n < frameNM; n++){
		idx = layerSize * n + dimIdx;
		if (patTypes[n] == PATTYPE_NONE){
		    // skip dummy node
		}else{
		    cnt++;
		    delta  = data[idx] - mean;
		    mean  += delta/cnt;
		    delta2 = data[idx] - mean;
		    var   += delta * delta2;
		}
	    }
	    if (cnt < 2){
		mean = 0.0;
		var  = 1.0;
	    }else{
		var  = var / cnt;
	    }

	    // save mean, std, and number of frames
	    meanStd[dimIdx]                 = mean;
	    meanStd[dimIdx + layerSize]     = sqrt(var + stdConst);
	    meanStd[dimIdx + layerSize * 2] = cnt;

	    // If training, to accumulat the mean, std, for generation stage
	    if (trainFlag){
		meanStdBuf[dimIdx] += (mean - meanStdBuf[dimIdx]) / batchCnt;
		meanStdBuf[dimIdx + layerSize] += (meanStd[dimIdx + layerSize] -
						   meanStdBuf[dimIdx + layerSize]) / batchCnt;
	    }
	}
    }; 
    


    struct ComputeBatchNorm
    {
	int layerSize;

	const char *patTypes;   
	real_t *data;
	real_t *outdata;
	real_t *outNormed;
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
		    outNormed[dataIdx] = (data[dataIdx]-meanStd[dimIdx])/meanStd[varIdx];
		else
		    outNormed[dataIdx] = (data[dataIdx]-meanStdBuf[dimIdx])/meanStdBuf[varIdx];

		// y = \alpha \hat{x} + \beta
		outdata[dataIdx]   = outNormed[dataIdx] * scale[dimIdx] + scale[varIdx];
	    }
	}
    };


    
    struct ComputeBatchGradient_alphabeta
    {
	int     layerSize;
	int     frameNM;            // frame * parallel

	const char *patTypes;   
	real_t     *outNormed;
	real_t     *errors;
	real_t     *grad;
	
	__host__ __device__ void operator() (const thrust::tuple<real_t&, int> &t) const
	{
	    int dimIdx = t.get<1>();

	    real_t bufAlpha = 0.0;
	    real_t bufBeta  = 0.0;

	    int idx = 0;
	    for (int n = 0; n < frameNM; n++){
		idx = layerSize * n + dimIdx;
		if (patTypes[n] == PATTYPE_NONE){
		    // skip dummy node
		}else{
		    // sum_i \deltaE/\delta{y}_i * \hat{x}_i 
		    bufAlpha += errors[idx] * outNormed[idx];
		    // sum_i \deltaE/\delta{y}_i 
		    bufBeta  += errors[idx];
		}
	    }
	    grad[dimIdx]               = bufAlpha;
	    grad[dimIdx + layerSize]   = bufBeta;
	    
	    // slots for mean, std (to be output)
	    grad[dimIdx + 2*layerSize] = 0;
	    grad[dimIdx + 3*layerSize] = 0;
	    
	}
    };
    
    
    struct ComputeBatchGradient_output
    {
	
	int layerSize;

	const char *patTypes;   
	real_t *errors;
	real_t *grad;
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
		grad[dataIdx] = ((errors[dataIdx] -
				  scaleGrad[dimIdx] * outNormed[dataIdx]/ batchSize -
				  scaleGrad[dimIdx + layerSize] / batchSize ) *
				 scale[dimIdx] / meanStd[dimIdx + layerSize]);
	    }
	}
    };
    
}
}


namespace layers{
    
    template <typename TDevice>
    BatchNormLayer<TDevice>::BatchNormLayer(const helpers::JsonValue &layerChild, 
					    const helpers::JsonValue &weightsSection, 
					    Layer<TDevice> &precedingLayer,
					    int             maxSeqLength,
					    int             layerID)
        : TrainableLayer<TDevice>(layerChild, weightsSection, 0, 4,
				  precedingLayer, maxSeqLength, layerID)
    {
	// Trainable parameters: alpha + beta, for each dimension of previous output
	if (this->size() != precedingLayer.size()){
	    throw std::runtime_error("Error in batchnorm layer size (must = previous one)");
	}
	
	// initialization
	m_stdConst  = 0.001;
	m_outNormed = this->outputs();
	m_batchCnt  = 0.0;
	m_preEpoch  = 1;

	// mean, std, and number of frames for each dim
	Cpu::real_vector tmp;
	tmp.resize(this->size() * 3, 0.0); 
	m_stats     = tmp;

	// all-one vector for vector summation
	tmp.resize(this->outputs().size()/this->size(), 1.0);
	m_oneVector = tmp;

	// a tempopary buff
	m_buff = this->outputs();
	
	// initialize scale parameter
	if (weightsSection.isValid() && weightsSection->HasMember(this->name().c_str())) {
	    // read the scale parameter
	    
	}else{
	    // alpha = 1.0
	    thrust::fill(this->weights().begin(), this->weights().begin()+this->size(), 1.0);
	    // beta  = 0.0
	    // remaining for mean, std (to be accumulated across the data corpus)
	    thrust::fill(this->weights().begin() + this->size(), this->weights().end(), 0.0);
	}

	//const Configuration &config = Configuration::instance();
	//m_trainFlag = config.trainingMode();
	
	if (this->precedingLayer().getSaveMemoryFlag())
	    throw std::runtime_error("layer before batchnorm is reduced in mem");  

    }

    template <typename TDevice>
    BatchNormLayer<TDevice>::~BatchNormLayer()
    {
    }
    
    template <typename TDevice>
    const std::string& BatchNormLayer<TDevice>::type() const
    {
        static std::string s;
        if (s.empty()) s = "batchnorm";
        return s;
    }

    template <typename TDevice>
    void BatchNormLayer<TDevice>::computeForwardPass(const int nnState)
    {
	if (this->flagTrainingMode() && m_preEpoch > 0 &&
	    m_preEpoch != this->getCurrTrainingEpoch()){
	    // always update the mean, std for each epoch
	    m_batchCnt = 0;
	    thrust::fill(this->weights().begin() + 2 * this->size(), this->weights().end(), 0.0);
	    m_preEpoch = this->getCurrTrainingEpoch();
	}
	
	m_batchCnt++;
	{{
	   // Step1. calculate the batch size
	   //        For parallel sentences, there is dummy node. BatchSize should not count it.
	   internal::BatchSize fn0;
	   fn0.patTypes = helpers::getRawPointer(this->patTypes());
	   int tmp      = this->curMaxSeqLength() * this->parallelSequences();	   
	   m_batchSize  = thrust::transform_reduce(
		thrust::make_zip_iterator(
			thrust::make_tuple(this->outputs().begin(), 
					   thrust::counting_iterator<int>(0))),
		thrust::make_zip_iterator(
			thrust::make_tuple(this->outputs().begin() + tmp, 
					   thrust::counting_iterator<int>(0) + tmp)),
		fn0,
		(real_t)0.0,
		thrust::plus<real_t>());
	   thrust::fill(this->m_oneVector.begin(), this->m_oneVector.end(), 1.0/m_batchSize);
 
	   // Step2. accumulate the mean
	   internal::PrepareForMeanStd fn1;
	   fn1.layerSize = this->size();
	   fn1.meanNotVar= true;
	   fn1.mean      = NULL;
	   fn1.patTypes  = helpers::getRawPointer(this->patTypes());
	   fn1.data      = helpers::getRawPointer(this->precedingLayer().outputs());
	   fn1.outdata   = helpers::getRawPointer(m_buff);	   
	   tmp       = this->size() * this->curMaxSeqLength() * this->parallelSequences();
	   thrust::for_each(
		thrust::make_zip_iterator(
			thrust::make_tuple(this->outputs().begin(), 
					   thrust::counting_iterator<int>(0))),
		thrust::make_zip_iterator(
			thrust::make_tuple(this->outputs().begin() + tmp, 
					   thrust::counting_iterator<int>(0) + tmp)),
		fn1);
	   
	   helpers::Matrix<TDevice> onevec  (&this->m_oneVector, 
					     this->curMaxSeqLength() * this->parallelSequences(), 
					     1);
	   helpers::Matrix<TDevice> data    (&this->m_buff, 
					     this->size(),
					     this->curMaxSeqLength() * this->parallelSequences());
	   helpers::Matrix<TDevice> meanVec (&this->m_stats, 
					     this->size(),
					     1);
	   meanVec.assignProduct(data, false, onevec, false);

	   // Step3. accumulate the var
	   fn1.meanNotVar= false;
	   fn1.mean      = helpers::getRawPointer(this->m_stats);; 
	   thrust::for_each(
		thrust::make_zip_iterator(
			thrust::make_tuple(this->outputs().begin(), 
					   thrust::counting_iterator<int>(0))),
		thrust::make_zip_iterator(
			thrust::make_tuple(this->outputs().begin() + tmp, 
					   thrust::counting_iterator<int>(0) + tmp)),
		fn1);

	   helpers::Matrix<TDevice> data2   (&this->m_buff, 
					     this->size(),
					     this->curMaxSeqLength() * this->parallelSequences());
	   helpers::Matrix<TDevice> stdVec  (&this->m_stats, 
					     this->size(),
					     1,
					     this->size());
	   stdVec.assignProduct(data2, false, onevec, false);
	   internal::GetStd fn3;
	   fn3.stdConst = m_stdConst;
	   fn3.meanStd  = helpers::getRawPointer(m_stats) + this->size();
	   thrust::for_each(
		thrust::make_zip_iterator(
			thrust::make_tuple(m_stats.begin() + this->size(), 
					   thrust::counting_iterator<int>(0))),
		thrust::make_zip_iterator(
			thrust::make_tuple(m_stats.begin() + this->size() * 2, 
					   thrust::counting_iterator<int>(0) + this->size())),
		fn3);

	   // Step4. accumulate the mean and std, for generation stage
	   if (this->flagTrainingMode()){
	       internal::AveMeanStd fn5;
	       fn5.meanStd    = helpers::getRawPointer(m_stats);
	       fn5.meanStdBuf = helpers::getRawPointer(this->weights()) + this->size() * 2;
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
	   
	   
	   /*internal::ComputeBatchMeanStd fn;
	   fn.layerSize = this->size();
	   fn.frameNM   = this->curMaxSeqLength() * this->parallelSequences();
	   fn.stdConst  = m_stdConst;
	   
	   fn.patTypes  = helpers::getRawPointer(this->patTypes());
	   fn.data      = helpers::getRawPointer(this->precedingLayer().outputs());
	   fn.meanStd   = helpers::getRawPointer(m_stats);
	   
	   // one additional buffer to accumulate the mean, std over entire corpus
	   fn.meanStdBuf= helpers::getRawPointer(this->weights()) + this->size() * 2;
	   fn.trainFlag = m_trainFlag;
	   fn.batchCnt  = m_batchCnt;
	   thrust::for_each(
		thrust::make_zip_iterator(
			thrust::make_tuple(m_stats.begin(), 
					   thrust::counting_iterator<int>(0))),
		thrust::make_zip_iterator(
			thrust::make_tuple(m_stats.begin() + this->size(), 
					   thrust::counting_iterator<int>(0) + this->size())),
					   fn);*/
	   


	   // Step4: normalize and scale the data
	   internal::ComputeBatchNorm fn2;
	   fn2.layerSize = this->size();
	   fn2.patTypes  = helpers::getRawPointer(this->patTypes());
	   fn2.data      = helpers::getRawPointer(this->precedingLayer().outputs());
	   fn2.outdata   = helpers::getRawPointer(this->outputs());
	   fn2.outNormed = helpers::getRawPointer(m_outNormed);
	   fn2.scale     = helpers::getRawPointer(this->weights());
	   fn2.meanStd   = helpers::getRawPointer(m_stats);
	   fn2.meanStdBuf= helpers::getRawPointer(this->weights()) + this->size() * 2;
	   fn2.trainFlag = this->flagTrainingMode();
	   
	   tmp       = this->size() * this->curMaxSeqLength() * this->parallelSequences();
	   thrust::for_each(
		thrust::make_zip_iterator(
			thrust::make_tuple(this->outputs().begin(), 
					   thrust::counting_iterator<int>(0))),
		thrust::make_zip_iterator(
			thrust::make_tuple(this->outputs().begin() + tmp, 
					   thrust::counting_iterator<int>(0) + tmp)),
		fn2);
	}}
	
    }

    template <typename TDevice>
    void BatchNormLayer<TDevice>::computeForwardPass(const int timeStep, const int nnState)
    {
	if (this->flagTrainingMode())
	    throw std::runtime_error("Error: Batch norm is not for online propagation");

	int effTimeStart = timeStep * this->parallelSequences();
	int effTimeEnd   = (timeStep+1) * this->parallelSequences();

	// Step4: normalize and scale the data
	internal::ComputeBatchNorm fn2;
	fn2.layerSize = this->size();
	fn2.patTypes  = helpers::getRawPointer(this->patTypes());
	fn2.data      = helpers::getRawPointer(this->precedingLayer().outputs());
	fn2.outdata   = helpers::getRawPointer(this->outputs());
	fn2.outNormed = helpers::getRawPointer(m_outNormed);
	fn2.scale     = helpers::getRawPointer(this->weights());
	fn2.meanStd   = helpers::getRawPointer(m_stats);
	fn2.meanStdBuf= helpers::getRawPointer(this->weights()) + this->size() * 2;
	fn2.trainFlag = this->flagTrainingMode();
	   
	thrust::for_each(
	      thrust::make_zip_iterator(
		thrust::make_tuple(
		     this->outputs().begin() + effTimeStart * this->size(), 
		     thrust::counting_iterator<int>(0) + effTimeStart * this->size())),
	      thrust::make_zip_iterator(
		thrust::make_tuple(
		     this->outputs().begin() + effTimeEnd * this->size(), 
		     thrust::counting_iterator<int>(0) + effTimeEnd * this->size())),
	      fn2);
	
    }

    template <typename TDevice>
    void BatchNormLayer<TDevice>::computeBackwardPass(const int nnState)
    {
	{{

	   thrust::fill(m_oneVector.begin(), m_oneVector.end(), 1.0);
	   thrust::fill(this->_weightUpdates().begin(), this->_weightUpdates().end(), 0.0);
	   int tmp          = this->size() * this->curMaxSeqLength() * this->parallelSequences();

	   // Step1. Calculate \deltaE/\delta{\alpha}
	   internal::PrepareGrad fn1;
	   fn1.layerSize    = this->size();
	   fn1.alphaNotBeta = true;
	   fn1.patTypes     = helpers::getRawPointer(this->patTypes());
	   fn1.grad         = helpers::getRawPointer(this->outputErrors());
	   fn1.buff         = helpers::getRawPointer(m_buff);
	   fn1.data         = helpers::getRawPointer(m_outNormed);
	   
	   thrust::for_each(
		thrust::make_zip_iterator(
			thrust::make_tuple(this->outputs().begin(), 
					   thrust::counting_iterator<int>(0))),
		thrust::make_zip_iterator(
			thrust::make_tuple(this->outputs().begin() + tmp, 
					   thrust::counting_iterator<int>(0) + tmp)),
		fn1);
	   
	   helpers::Matrix<TDevice> onevec    (&this->m_oneVector, 
					       this->curMaxSeqLength() * this->parallelSequences(), 
					       1);
	   helpers::Matrix<TDevice> data      (&this->m_buff, 
					       this->size(),
					       this->curMaxSeqLength() * this->parallelSequences());
	   helpers::Matrix<TDevice> gradAlpha (&this->_weightUpdates(), 
					       this->size(),
					       1);
	   gradAlpha.assignProduct(data, false, onevec, false);

	   // Step2. Calculate \deltaE/\delta{\beta}
	   fn1.alphaNotBeta = false;	   
	   thrust::for_each(
		thrust::make_zip_iterator(
			thrust::make_tuple(this->outputs().begin(), 
					   thrust::counting_iterator<int>(0))),
		thrust::make_zip_iterator(
			thrust::make_tuple(this->outputs().begin() + tmp, 
					   thrust::counting_iterator<int>(0) + tmp)),
		fn1);
	   
	   helpers::Matrix<TDevice> gradBeta (&this->_weightUpdates(), 
					      this->size(),
					      1,
					      this->size());
	   gradBeta.assignProduct(data, false, onevec, false);
	   

	   /*
	   internal::ComputeBatchGradient_alphabeta fn;
	   fn.layerSize = this->size();
	   fn.frameNM   = this->curMaxSeqLength() * this->parallelSequences();
	   
	   fn.patTypes  = helpers::getRawPointer(this->patTypes());
	   fn.outNormed = helpers::getRawPointer(m_outNormed);
	   fn.errors    = helpers::getRawPointer(this->outputErrors());
	   fn.grad      = helpers::getRawPointer(this->_weightUpdates());
	   
	   thrust::for_each(
		thrust::make_zip_iterator(
			thrust::make_tuple(m_stats.begin(), 
					   thrust::counting_iterator<int>(0))),
		thrust::make_zip_iterator(
			thrust::make_tuple(m_stats.begin() + this->size(), 
					   thrust::counting_iterator<int>(0) + this->size())),
		fn);
	   */

	   // Step3. Calculate \deltaE/\delta{x}
	   internal::ComputeBatchGradient_output fn2;
	   fn2.layerSize = this->size();
	   fn2.patTypes  = helpers::getRawPointer(this->patTypes());
	   fn2.errors    = helpers::getRawPointer(this->outputErrors());
	   fn2.grad      = helpers::getRawPointer(this->precedingLayer().outputErrors());
	   fn2.outNormed = helpers::getRawPointer(m_outNormed);
	   fn2.meanStd   = helpers::getRawPointer(m_stats);
	   fn2.scale     = helpers::getRawPointer(this->weights());
	   fn2.scaleGrad = helpers::getRawPointer(this->_weightUpdates());
	   fn2.batchSize = m_batchSize;
	   
	   //tmp       = this->size() * this->curMaxSeqLength() * this->parallelSequences();
	   thrust::for_each(
		thrust::make_zip_iterator(
			thrust::make_tuple(m_outNormed.begin(), 
					   thrust::counting_iterator<int>(0))),
		thrust::make_zip_iterator(
			thrust::make_tuple(m_outNormed.begin() + tmp, 
					   thrust::counting_iterator<int>(0) + tmp)),
		fn2);

	}}
    
    }

    template <typename TDevice>
    void BatchNormLayer<TDevice>::computeBackwardPass(const int timeStep, const int nnState)
    {
	throw std::runtime_error("Error: Batchnorm is not for online computation");
    }
    
    template class BatchNormLayer<Cpu>;
    template class BatchNormLayer<Gpu>;
}
