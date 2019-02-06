/******************************************************************************
 * This file is an addtional component of CURRENNT. 
 * Xin WANG
 * National Institute of Informatics, Japan
 * 2016
 *
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

#include "KLPostOutputLayer.hpp"

#include "../Configuration.hpp"
#include "../helpers/getRawPointer.cuh"
#include "../helpers/NumericLimits.cuh"
#include "../helpers/min.cuh"
#include "../helpers/max.cuh"

#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>

#define KLDOUTPUTDATATYPE_LINEAR_UNI 1
#define KLDOUTPUTDATATYPE_LOG_UNI 2
#include <cmath>

namespace internal {
namespace {

    
    // For logData
    struct ComputeKLDFnLog
    {
        int layerSize;
	//int maxTime;
	
        const char   *patTypes;
	const real_t *mvData;
	real_t       *errorBuf;
	real_t        factor;
        __host__ __device__ real_t operator() (const thrust::tuple<real_t, real_t, int> &values) const
        {
            // unpack the tuple
            real_t target = values.get<0>();
            real_t output = values.get<1>();
            int outputIdx = values.get<2>();
	    
            // check if we have to skip this value
            int patIdx = outputIdx    / layerSize;
            if (patTypes[patIdx] == PATTYPE_NONE){
		*(errorBuf + outputIdx) = 0;
                return 0;
	    }
	    
	    int featDim   = outputIdx % layerSize;
	    
	    // get linear value
	    real_t mean   = *(mvData + featDim);
	    real_t var    = *(mvData + featDim + layerSize);
	    real_t tarLin = var*target + mean;
	    real_t outLin = var*output + mean;
	    
	    // upper unbounded
	    tarLin = (tarLin > -helpers::NumericLimits<real_t>::expLimit())?(exp(tarLin)):(0);
	    outLin = (outLin > -helpers::NumericLimits<real_t>::expLimit())?(exp(outLin)):(0);
	    
	    *(errorBuf + outputIdx) = (var * (outLin - tarLin))/factor;
            return (tarLin * (var * (target - output) - 1) + outLin)/factor;
        }
    };

    // For linear Data
    struct ComputeKLDFnLinear
    {
        int layerSize;
	//int maxTime;
	
        const char   *patTypes;
	const real_t *mvData;
	real_t       *errorBuf;
	real_t        factor;
        __host__ __device__ real_t operator() (const thrust::tuple<real_t, real_t, int> &values) const
        {
            // unpack the tuple
            real_t target = values.get<0>();
            real_t output = values.get<1>();
            int outputIdx = values.get<2>();
	    
            // check if we have to skip this value
            int patIdx = outputIdx    / layerSize;
            if (patTypes[patIdx] == PATTYPE_NONE){
		*(errorBuf + outputIdx) = 0;
                return 0;
	    }
	    
	    int featDim   = outputIdx % layerSize;
	    
	    // get linear value
	    real_t mean   = *(mvData + featDim);
	    real_t var    = *(mvData + featDim + layerSize);
	    real_t tarLin = var*target + mean;
	    real_t outLin = var*output + mean;
	    
	    // upper unbounded
	    real_t ratio = helpers::max(helpers::NumericLimits<real_t>::min(), tarLin/outLin);
	    
	    *(errorBuf + outputIdx) = (var * (1 - ratio))/factor;
	    
            return (tarLin * log(ratio) - tarLin + outLin)/factor;
        }
    };
    
    /*
    struct ComputeOutputErrorFn
    {        
    	int layerSize;
    	const char   *patTypes;
	const real_t *mvData;
    	__host__ __device__ real_t operator() 
	(const thrust::tuple<const real_t&, const real_t&, int> &t) const        
    	{   
	    // unpack the tuple
	    real_t actualOutput = t.get<0>();
	    real_t targetOutput = t.get<1>();
	    int    outputIdx    = t.get<2>();
	    
	    // calculate the pattern index
	    int patIdx          = outputIdx / layerSize;
	    // check if the pattern is a dummy
	    if (patTypes[patIdx] == PATTYPE_NONE)
		return 0;

	    // calculate the error
	    real_t error = actualOutput - targetOutput;
	    return error;
	}
	};*/
    
} // anonymous namespace
} // namespace anonymous


namespace layers {

    template <typename TDevice>
    KLPostOutputLayer<TDevice>::KLPostOutputLayer(const helpers::JsonValue &layerChild, 
						  Layer<TDevice> &precedingLayer,
						  int             maxSeqLength,
						  int             layerID)
        : PostOutputLayer<TDevice>(layerChild, precedingLayer,
				   precedingLayer.size(), maxSeqLength, layerID)
    {
	const Configuration &config = Configuration::instance();
	m_dataType = config.KLDOutputDataType();
	if(m_dataType != KLDOUTPUTDATATYPE_LOG_UNI && m_dataType != KLDOUTPUTDATATYPE_LINEAR_UNI) {
	    printf("Please specify --KLDTargetDataType for training kld output layer");
	    //throw std::runtime_error(std::string("1: log-domain, 2: linear domain data"));
	}else{
	    if (m_dataType == KLDOUTPUTDATATYPE_LOG_UNI)
		printf("Target Data type : log data");
	    else
		printf("Target Data type : linear data");
	}
	
	if (config.datamvPath().size()<1){
	    printf("Skip reading data.mv for KLD output layer");
	    //throw std::runtime_error(std::string("KLD output requires mean and var"));
	}
	
	m_lrFactor = config.lrFactor();
	printf("KLD scale factor: %f", m_lrFactor);

	// initialize the buffer
	m_errorBuf = this->_actualOutputs();
	
    }

    template <typename TDevice>
    KLPostOutputLayer<TDevice>::~KLPostOutputLayer()
    {
    }

    template <typename TDevice>
    const std::string& KLPostOutputLayer<TDevice>::type() const
    {
        static const std::string s("kld");
        return s;
    }

    template <typename TDevice>
    real_t KLPostOutputLayer<TDevice>::calculateError()
    {
		    
	if(m_dataType == KLDOUTPUTDATATYPE_LINEAR_UNI){
	    internal::ComputeKLDFnLinear fn;
	    fn.layerSize = this->size();
	    fn.patTypes  = helpers::getRawPointer(this->patTypes());
	    fn.mvData    = helpers::getRawPointer(this->_mvVector());
	    fn.errorBuf  = helpers::getRawPointer(m_errorBuf);
	    fn.factor    = m_lrFactor;
	    
	    //fn.maxTime   = this->maxSeqLength();
	    int n = this->curMaxSeqLength() * this->parallelSequences() * this->size();

	    real_t mse = (real_t) thrust::transform_reduce(
               thrust::make_zip_iterator(
		   thrust::make_tuple(this->_targets().begin(),   
				      this->_actualOutputs().begin(),   
				      thrust::counting_iterator<int>(0))),
               thrust::make_zip_iterator(
		   thrust::make_tuple(this->_targets().begin()+n, 
				      this->_actualOutputs().begin()+n, 
				      thrust::counting_iterator<int>(0)+n)),
	       fn,
	       (real_t)0,
	       thrust::plus<real_t>()
	       );
	    
	    // Debug
	    /*Cpu::real_vector target = this->_targets();
	    Cpu::real_vector output = this->_actualOutputs();
	    Cpu::real_vector mvData = this->_mvVector();
	    Cpu::pattype_vector patTyp = this->patTypes();
	    real_t mse_temp = 0;
	    for (int outputIdx = 0; outputIdx < n; outputIdx++){
		int featDim = outputIdx % this->size();
		int patIdx  = outputIdx / this->size();
		if (patTyp[patIdx] == PATTYPE_NONE)
		    continue;
		
		real_t mean   = mvData[featDim];
		real_t var    = mvData[featDim + this->size()];
		real_t tarLin = var*target[outputIdx] + mean;
		real_t outLin = var*output[outputIdx] + mean;
		real_t ratio  = tarLin / outLin;
		
		mse_temp += (tarLin * std::log(ratio) - tarLin + outLin) / m_lrFactor;
		if (mse_temp != mse_temp){
		    mse_temp = 0;
		}
		}*/
	    return mse;
	    
	}else if(m_dataType == KLDOUTPUTDATATYPE_LOG_UNI){
	    internal::ComputeKLDFnLog fn;
	    fn.layerSize = this->size();
	    fn.patTypes  = helpers::getRawPointer(this->patTypes());
	    fn.mvData    = helpers::getRawPointer(this->_mvVector());
	    fn.errorBuf  = helpers::getRawPointer(m_errorBuf);
	    fn.factor    = m_lrFactor;
	    
	    //fn.maxTime   = this->maxSeqLength();
	    int n = this->curMaxSeqLength() * this->parallelSequences() * this->size();

	    real_t mse = (real_t) thrust::transform_reduce(
               thrust::make_zip_iterator(
		   thrust::make_tuple(this->_targets().begin(),   
				      this->_actualOutputs().begin(),   
				      thrust::counting_iterator<int>(0))),
               thrust::make_zip_iterator(
		   thrust::make_tuple(this->_targets().begin()+n, 
				      this->_actualOutputs().begin()+n, 
				      thrust::counting_iterator<int>(0)+n)),
	       fn,
	       (real_t)0,
	       thrust::plus<real_t>()
	       );
	    return mse;
	    
	}else{
	    // can't be here
	    printf("Please specify --targetDataType");
	    throw std::runtime_error(std::string("Only log (2) and linear (1) data for KLD layer"));
	}
    }

    template <typename TDevice>
    void KLPostOutputLayer<TDevice>::computeForwardPass(const int nnState)
    {
    }

    template <typename TDevice>
    void KLPostOutputLayer<TDevice>::computeForwardPass(const int timeStep, const int nnState)
    {
    }

    template <typename TDevice>
    void KLPostOutputLayer<TDevice>::computeBackwardPass(const int nnState)
    {
     // calculate the errors
	/*internal::ComputeOutputErrorFn fn;
	  fn.layerSize = this->size();
	  fn.patTypes  = helpers::getRawPointer(this->patTypes());
	  fn.mvData    = helpers::getRawPointer(this->_mvVector());*/
	
	int n = this->curMaxSeqLength() * this->parallelSequences() * this->size();
	
	/*thrust::transform(
	  thrust::make_zip_iterator(
	  thrust::make_tuple(this->_actualOutputs().begin(),   
	  this->_targets().begin(),   
	  thrust::counting_iterator<int>(0))),
	  thrust::make_zip_iterator(
	  thrust::make_tuple(this->_actualOutputs().begin()+n, 
	  this->_targets().begin()+n, 
	  thrust::counting_iterator<int>(0)+n)),
	  this->_outputErrors().begin(),
	  fn
	  );*/
	thrust::copy(m_errorBuf.begin(), m_errorBuf.begin() + n, this->_outputErrors().begin());
    }

    template <typename TDevice>
    void KLPostOutputLayer<TDevice>::computeBackwardPass(const int timeStep, const int nnState)
    {
	if (timeStep == this->curMaxSeqLength())
	    this->computeBackwardPass(nnState);
    }
    // explicit template instantiations
    template class KLPostOutputLayer<Cpu>;
    template class KLPostOutputLayer<Gpu>;

} // namespace layers
