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
 *****************************************************************************//*

*/


#ifdef _MSC_VER
#   pragma warning (disable: 4244) // thrust/iterator/iterator_adaptor.h(121): warning C4244: '+=' : conversion from '__int64' to 'int', possible loss of data
#endif

#include "FilteringLayer.hpp"

#include "../helpers/getRawPointer.cuh"
#include "../helpers/Matrix.hpp"
#include "../helpers/min.cuh"
#include "../helpers/max.cuh"
#include "../helpers/safeExp.cuh"
#include "../helpers/NumericLimits.cuh"
#include "../helpers/JsonClasses.hpp"
#include "../helpers/misFuncs.hpp"

#include "../activation_functions/Tanh.cuh"
#include "../activation_functions/Logistic.cuh"
#include "../activation_functions/Identity.cuh"
#include "../activation_functions/Relu.cuh"

#include "../Configuration.hpp"

#include <boost/foreach.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>

#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>

#include <sstream>
#include <fstream>
#include <cmath>


#define FILTERING_LAYER_MODE_NONE_SELECTIVE 0
#define FILTERING_LAYER_MODE_SELECTIVE 1

namespace internal{
namespace {

    // Use one group of filters, do filtering
    struct causalFilteringForward_none_selective
    {
	int        filterLength;             // length of filter
	int        filterShareAcrossDim;     // whether the filter is shared across data dimension
	int        layerSize;                // layer size
	int        parallel;                 
	int        initSmooth;
	int        noncausal;
	int        maxLength;
	
	real_t     *inputData;
	real_t     *filterCoeffs;
	const char *patTypes;   

	// From 1 : T (of the previous layer)
	__host__ __device__ void operator() (const thrust::tuple<real_t&, int> &t) const
	{
	    int outputIdx  = t.get<1>();
	    int dimIdx     = outputIdx % layerSize;  // dimension index
	    int timeIdx    = outputIdx / layerSize;  // time index (regardless of parallel)
	    
	    int BlockIdx   = timeIdx / parallel;     // time index (considering parallel mode)
	    int BlockInIdx = timeIdx % parallel;     // index within a parallel block


	    if (patTypes[timeIdx] == PATTYPE_NONE){
		t.get<0>() = 0;
		return;
	    }		

	    // accumulate and update
	    real_t tmp = 0.0;
	    real_t filterCoeff = 1.0;  
	    real_t lastValidValue = 0.0;
	    
	    int    filter_idx_shift = 0;
	    int    data_time_idx;  // the relative time step index
	    int    data_mem_idx;   // the absolute time step in memory buffer

	    // for noncausal filter, shift the time index
	    if (noncausal)
		filter_idx_shift = filterLength / 2;
	    
	    for (int idx = 0 ; idx < filterLength; idx++){
		    
		// get the filter coefficient for one step
		// a_0 + a_1 z^-1 + ... + a_N z^-N
		// [a_0, a_1, ..., a_N]
		if (filterShareAcrossDim)
		    filterCoeff = filterCoeffs[idx];
		else
		    filterCoeff = filterCoeffs[dimIdx * filterLength + idx];

		// time index (of parallel block)
		data_time_idx = BlockIdx - idx + filter_idx_shift;
		// time index (abosolute time step in memory buffer)
		data_mem_idx = data_time_idx * parallel + BlockInIdx;
		
		if (data_time_idx >= 0 && data_mem_idx < maxLength &&
		    patTypes[data_mem_idx] != PATTYPE_NONE){
		    // when this time step is [0, max_length) and valid
		    tmp += inputData[data_mem_idx * layerSize + dimIdx] * filterCoeff;
		    lastValidValue = inputData[data_mem_idx * layerSize + dimIdx];
		}else if (initSmooth){
		    // If there is no data at the begining: use the first time step
		    // If there is no data at the end: use the current time step
		    if (data_time_idx < 0)
			tmp += (inputData[BlockInIdx * layerSize + dimIdx] * filterCoeff);
		    else
			tmp += (lastValidValue * filterCoeff);
		}else{
		    // do nothing
		}
	    }
	    t.get<0>() = tmp;
	}
    };
    
    
    struct causalFilteringBackward_none_selective
    {
	int        filterLength;
	int        filterShareAcrossDim;
	
	int        layerSize;
	int        maxLength;
	int        parallel;
	int        noncausal;
	
	real_t     *inputErrors;
	real_t     *filterCoeffs;
	const char *patTypes;   

	// From 1 : T (of the previous layer)
	__host__ __device__ void operator() (const thrust::tuple<real_t&, int> &t) const
	{
	    
	    int outputIdx  = t.get<1>();
	    int dimIdx     = outputIdx % layerSize;  // dimension index
	    int timeIdx    = outputIdx / layerSize;  // time index (regardless of parallel)
	    
	    int BlockIdx   = timeIdx / parallel;     // time index (considering parallel mode)
	    int BlockInIdx = timeIdx % parallel;     // index within a parallel block

	    if (patTypes[timeIdx] == PATTYPE_NONE){
		t.get<0>() = 0;
		return;
	    }		

	    // accumulate and update
	    real_t tmp = 0.0;
	    real_t filterCoeff;
	    
	    int    filter_idx_shift = 0;
	    int    data_time_idx;  // the relative time step index
	    int    data_mem_idx;   // the absolute time step in memory buffer

	    if (noncausal)
		filter_idx_shift = filterLength / 2;

	    for (int idx = 0 ; idx < filterLength; idx++){

		// get the filter coefficient for one step
		// a_0 + a_1 z^-1 + ... + a_N z^-N
		// [a_0, a_1, ..., a_N]
		if (filterShareAcrossDim)
		    filterCoeff = filterCoeffs[idx];
		else
		    filterCoeff = filterCoeffs[dimIdx * filterLength + idx];

		data_time_idx = BlockIdx + idx - filter_idx_shift;
		data_mem_idx = data_time_idx * parallel + BlockInIdx;

		if (data_mem_idx < maxLength && patTypes[data_mem_idx] != PATTYPE_NONE &&
		    data_time_idx >= 0){
		    tmp += inputErrors[data_mem_idx * layerSize + dimIdx] * filterCoeff;
		}else{
		    // Just ignore the gradients for the two-ends;
		}
	    }
	    t.get<0>() = tmp;
	    
	}
    };


    // Use weighted sum of filters
    struct causalFilteringForward_selective
    {
	int        filterLength;
	int        filterShareAcrossDim;
	
	int        outputLayerSize;
	int        inputLayerSize;
	int        filterNum;
	int        parallel;
	int        initSmooth;

	int        maxLength;
	int        noncausal;
	
	real_t     *inputData;
	real_t     *filterCoeffs;
	const char *patTypes;   

	// From 1 : T (of the previous layer)
	__host__ __device__ void operator() (const thrust::tuple<real_t&, int> &t) const
	{
	    int outputIdx  = t.get<1>();
	    int dimIdx     = outputIdx % outputLayerSize;  // dimension index
	    int timeIdx    = outputIdx / outputLayerSize;  // time index (regardless of parallel)
	    
	    int BlockIdx   = timeIdx / parallel;     // time index (considering parallel mode)
	    int BlockInIdx = timeIdx % parallel;     // index within a parallel block

	    if (patTypes[timeIdx] == PATTYPE_NONE){
		t.get<0>() = 0;
		return;
	    }		

	    // accumulate and update
	    real_t tmp = 0.0;
	    real_t filterCoeff  = 0.0;
	    real_t filterWeight = 0.0;
	    
	    int    filter_idx_shift = 0;
	    int    data_time_idx;  // the relative time step index
	    int    data_mem_idx;   // the absolute time step in memory buffer
	    
	    if (noncausal)
		filter_idx_shift = filterLength / 2;
	    
	    for (int idx = 0 ; idx < filterLength; idx++){

		// The input vector looks like:
		// [data_dim1, data_dim2, ..., data_dim_N, filter_w1, filter_w2, ...filter_wM]
		// M + N = inputLayerSize
		// N = outputLayerSize
		// M = filterNum
		
		filterCoeff = 0.0;
		
		// get weighted sum of filter weights
		for (int filterIdx = 0 ; filterIdx < filterNum; filterIdx++){

		    // weight of current time step
		    filterWeight = inputData[timeIdx * inputLayerSize +
					     inputLayerSize - filterNum + filterIdx];

		    // weighted sum of filter coeffs
		    if (filterShareAcrossDim)
			filterCoeff += (filterCoeffs[filterIdx * filterLength + idx] * filterWeight);
		    else
			filterCoeff += (filterCoeffs[(filterIdx * outputLayerSize + dimIdx) *
						     filterLength + idx] * filterWeight);
		}
		
		data_time_idx = BlockIdx - idx + filter_idx_shift;
		data_mem_idx = data_time_idx * parallel + BlockInIdx;
		
		if (data_time_idx >= 0 && data_mem_idx < maxLength &&
		    patTypes[data_mem_idx] != PATTYPE_NONE){
		    
		    // when this time step is [0, max_length) and valid
		    tmp += inputData[data_mem_idx * inputLayerSize + dimIdx] * filterCoeff;
		    
		}else if (initSmooth){
		    // If there is no data at the begining: use the first time step
		    // If there is no data at the end: use the current time step
		    if (data_time_idx < 0)
			tmp += (inputData[BlockInIdx * inputLayerSize + dimIdx] * filterCoeff);
		    else
			tmp += (inputData[timeIdx * inputLayerSize + dimIdx] * filterCoeff);
		}else{
		    // do nothing
		}
	    }
	    t.get<0>() = tmp;
	    
	}
    };
    
    
    struct causalFilteringBackward_selective
    {
	int        filterLength;
	int        filterShareAcrossDim;

	int        outputLayerSize;
	int        inputLayerSize;
	int        filterNum;
	int        noncausal;
	int        maxLength;
	int        parallel;
	
	real_t     *inputData;
	real_t     *inputErrors;
	real_t     *filterCoeffs;
	const char *patTypes;   

	// From 1 : T (of the previous layer)
	__host__ __device__ void operator() (const thrust::tuple<real_t&, int> &t) const
	{
	    int outputIdx  = t.get<1>();
	    int dimIdx     = outputIdx % inputLayerSize;  // dimension index
	    int timeIdx    = outputIdx / inputLayerSize;  // time index (regardless of parallel)
	    
	    int BlockIdx   = timeIdx / parallel;     // time index (considering parallel mode)
	    int BlockInIdx = timeIdx % parallel;     // index within a parallel block


	    if (patTypes[timeIdx] == PATTYPE_NONE){
		t.get<0>() = 0;
		return;
	    }		

	    // accumulate and update
	    real_t tmp = 0.0;
	    real_t filterCoeff  = 0.0;
	    real_t filterWeight = 0.0;
	    
	    int    filter_idx_shift = 0;
	    int    data_time_idx;  // the relative time step index
	    int    data_mem_idx;   // the absolute time step in memory buffer

	    if (noncausal)
		filter_idx_shift = filterLength / 2;

	    
	    if (dimIdx < (inputLayerSize - filterNum)){
		
		// gradients w.r.t input signal
		for (int idx = 0 ; idx < filterLength; idx++){

		    data_time_idx = BlockIdx + idx - filter_idx_shift;
		    data_mem_idx = data_time_idx * parallel + BlockInIdx;
		
		    // if the output time step is valid
		    if (data_mem_idx < maxLength && patTypes[data_mem_idx] != PATTYPE_NONE &&
			data_time_idx >= 0){

			filterCoeff = 0.0;
			
			for (int filter_idx = 0 ; filter_idx < filterNum; filter_idx++){

			    // weight of that (output) time step 
			    filterWeight = inputData[data_mem_idx * inputLayerSize +
						     inputLayerSize - filterNum + filter_idx];
			    
			    if (filterShareAcrossDim)
				filterCoeff += (filterCoeffs[filter_idx * filterLength + idx] *
						filterWeight);
			    else
				filterCoeff += (filterCoeffs[(filter_idx*outputLayerSize+dimIdx) *
							     filterLength + idx] *
						filterWeight);
			}
		
			// sum gradient
			tmp += (inputErrors[data_mem_idx * outputLayerSize + dimIdx] * filterCoeff);
		    }
		}
		t.get<0>() = tmp;
		
	    }else{

		// gradients w.r.t filter weights
		int filter_idx = dimIdx - outputLayerSize;

		for (int idx = 0 ; idx < filterLength; idx++){

		    data_time_idx = BlockIdx + idx - filter_idx_shift;
		    data_mem_idx = data_time_idx * parallel + BlockInIdx;

		    if (data_mem_idx >= maxLength || patTypes[data_mem_idx] == PATTYPE_NONE ||
			data_time_idx < 0)
			continue;
			    
		    if (filterShareAcrossDim){
			filterCoeff = filterCoeffs[filter_idx * filterLength + idx];
			for (int featDimIdx = 0 ; featDimIdx < outputLayerSize; featDimIdx ++){
			    tmp += (inputData[data_mem_idx * inputLayerSize + featDimIdx]
				    * filterCoeff
				    * inputErrors[timeIdx * outputLayerSize + featDimIdx]);
			}
			
		    }else{
			for (int featDimIdx = 0 ; featDimIdx < outputLayerSize; featDimIdx ++){
			    filterCoeff = filterCoeffs[(filter_idx * outputLayerSize + featDimIdx) *
						       filterLength + idx];
			    tmp += (inputData[data_mem_idx * inputLayerSize + featDimIdx]
				    * filterCoeff
				    * inputErrors[timeIdx * outputLayerSize + featDimIdx]);
			}
		    }
		}
		t.get<0>() = tmp;
	    }
	    
	}
    };
    
    
}
}

namespace layers {
    template <typename TDevice>
    FilteringLayer<TDevice>::FilteringLayer(const helpers::JsonValue &layerChild,
					    const helpers::JsonValue &weightsSection,
					    Layer<TDevice>           &precedingLayer,
					    int                       maxSeqLength,
					    int                       layerID)
	: TrainableLayer<TDevice>(layerChild, weightsSection, 0, 0,
				  precedingLayer, maxSeqLength, layerID)
    {
	// load the options from network.jsn
	this->__loadOpts(layerChild);
	// display the options
	this->__showOpts();
	
	if (this->getResolution() != this->precedingLayer().getResolution())
	    throw std::runtime_error("Error in filter layer: resolution != previous layer resolu");
	
    }

    template <typename TDevice>
    FilteringLayer<TDevice>::~FilteringLayer()
    {
    }

    template <typename TDevice>
    void FilteringLayer<TDevice>::__loadOpts(const helpers::JsonValue &layerChild)
    {
	// load options
	m_filter_across_dim = ((layerChild->HasMember("shareAcrossDim")) ? 
			       ((*layerChild)["shareAcrossDim"].GetInt()) : 1);
	
	m_filter_coeffs_str = ((layerChild->HasMember("filterCoeffs")) ? 
			       ((*layerChild)["filterCoeffs"].GetString()) : "");
	
	m_filter_length = ((layerChild->HasMember("filterLength")) ? 
			       ((*layerChild)["filterLength"].GetInt()) : 0);
	
	m_filter_initial_keep = ((layerChild->HasMember("initialCondSmooth")) ? 
			       ((*layerChild)["initialCondSmooth"].GetInt()) : 0);
	
	m_filter_noncausal = ((layerChild->HasMember("noncausal")) ? 
			       ((*layerChild)["noncausal"].GetInt()) : 0);
	
	if (this->size() == this->precedingLayer().size()){
	    // only 1 group of filter
	    m_filter_mode = FILTERING_LAYER_MODE_NONE_SELECTIVE;
	    m_filter_num = 1; 
	}else{
	    // multiplt groups of filters
	    m_filter_mode = FILTERING_LAYER_MODE_SELECTIVE;
	    // assume input data vectors contains the weights of each filter
	    // thus, input_signal_dim + num_filter = input_vector_dim
	    m_filter_num = this->precedingLayer().size() - this->size();
	}
	
	// parse the filter coefficients
	if (m_filter_coeffs_str.size()){
	    m_filter_coeffs.clear();
	    misFuncs::ParseFloatOpt(m_filter_coeffs_str, m_filter_coeffs_H);
	    m_filter_coeffs = m_filter_coeffs_H;
	    
	    // check, when shareAcrossDim is False, #coefficients should be N * feature dimension
	    if (m_filter_across_dim == 0){
		if (m_filter_coeffs_H.size() % this->size() != 0){
		    printf("\n\t %d filter coefficients for %d dimensions,", this->size(),
			   (int)m_filter_coeffs_H.size());
		    throw std::runtime_error("Error in filtering layer: filterCoeffs invalid");
		}
	    }
	}else{
	    throw std::runtime_error("Error in filtering layer: no filterCoeffs");
	}

	
	int tmp_filter_length = 0;
	// if m_filter_length is not specified, infer it
	if (m_filter_across_dim == 0)
	    tmp_filter_length = m_filter_coeffs_H.size() / this->size() / m_filter_num;
	else
	    tmp_filter_length = m_filter_coeffs_H.size() / m_filter_num;

	
	if (m_filter_length == 0){
	    m_filter_length = tmp_filter_length;
	}else if (m_filter_length != tmp_filter_length){
	    throw std::runtime_error("Error in filtering layer: filter mismatch");
	}else{
	    // nothing, the filter_length has been correctly configured
	}

    }
    
    template <typename TDevice>
    void FilteringLayer<TDevice>::__showOpts()
    {
	// show options
	// print information
	if (m_filter_mode == FILTERING_LAYER_MODE_NONE_SELECTIVE){
	    printf(" fixed filter, ");
	}else{
	    printf(" soft-weighted %d filters, ", m_filter_num);
	}
	
	if (m_filter_across_dim)
	    printf("\n\tone filter (length %d) across feature dimension,", m_filter_length);
	else
	    printf("\n\tone filter (length %d) for each feature dimension,", m_filter_length);

	if (m_filter_noncausal)
	    printf(" noncausal,");
	else
	    printf(" causal,");

	if (m_filter_initial_keep)
	    printf("\n\tboundary smoothed,");
	else
	    printf("\n\tboundary not smooth,");

    }
    
    template <typename TDevice>
    const std::string& FilteringLayer<TDevice>::type() const
    {
        static std::string s;
        if (s.empty()) s = "filtering";
        return s;
    }

    template <typename TDevice>
    void FilteringLayer<TDevice>::computeForwardPass(const int nnState)
    {
	int timeLength = this->curMaxSeqLength() * this->parallelSequences();
	
	if (m_filter_mode == FILTERING_LAYER_MODE_NONE_SELECTIVE){
	    // one group of filters
	    internal::causalFilteringForward_none_selective fn1;
	    fn1.filterLength         = this->m_filter_length;
	    fn1.layerSize            = this->size();
	    fn1.parallel             = this->parallelSequences();
	    fn1.filterShareAcrossDim = this->m_filter_across_dim;
	    fn1.initSmooth           = this->m_filter_initial_keep;
	    fn1.noncausal            = this->m_filter_noncausal;
	    fn1.maxLength            = timeLength;

	    fn1.filterCoeffs         = helpers::getRawPointer(this->m_filter_coeffs);
	    fn1.inputData            = helpers::getRawPointer(this->precedingLayer().outputs());
	    fn1.patTypes             = helpers::getRawPointer(this->patTypes());

	    int n = timeLength * this->size();
	    thrust::for_each(
               thrust::make_zip_iterator(
		  thrust::make_tuple(this->outputs().begin(),
				     thrust::counting_iterator<int>(0))),
	       thrust::make_zip_iterator(
		  thrust::make_tuple(this->outputs().begin()           + n,
				     thrust::counting_iterator<int>(0) + n)),
	       fn1);

	}else if (m_filter_mode == FILTERING_LAYER_MODE_SELECTIVE){
	    // weighted sum of filters
	    internal::causalFilteringForward_selective fn1;
	    fn1.filterLength    = this->m_filter_length;
	    fn1.filterShareAcrossDim = this->m_filter_across_dim;
	    
	    fn1.outputLayerSize = this->size();
	    fn1.inputLayerSize  = this->precedingLayer().size();
	    fn1.filterNum       = m_filter_num;
	    fn1.parallel        = this->parallelSequences();
	    fn1.maxLength       = timeLength;
	    
	    fn1.initSmooth      = this->m_filter_initial_keep;	    
	    fn1.noncausal       = this->m_filter_noncausal;
	    
	    fn1.filterCoeffs  = helpers::getRawPointer(this->m_filter_coeffs);
	    fn1.inputData     = helpers::getRawPointer(this->precedingLayer().outputs());
	    fn1.patTypes      = helpers::getRawPointer(this->patTypes());
	    
	    int n = timeLength * this->size();
	    thrust::for_each(
               thrust::make_zip_iterator(
		  thrust::make_tuple(this->outputs().begin(),
				     thrust::counting_iterator<int>(0))),
	       thrust::make_zip_iterator(
		  thrust::make_tuple(this->outputs().begin()           + n,
				     thrust::counting_iterator<int>(0) + n)),
	       fn1);

	}else{
	    throw std::runtime_error("Error in filtering layer: Unknown filter mode");
	}
    }

    template <typename TDevice>
    void FilteringLayer<TDevice>::computeForwardPass(const int timeStep, const int nnState)
    {
	throw std::runtime_error("Filtering computeForwardPass(timeStep) not implemented");
    }

    template <typename TDevice>
    void FilteringLayer<TDevice>::computeBackwardPass(const int nnState)
    {
	int timeLength = this->curMaxSeqLength() * this->parallelSequences();
	
	if (m_filter_mode == FILTERING_LAYER_MODE_NONE_SELECTIVE){    
	    internal::causalFilteringBackward_none_selective fn1;
	    fn1.filterLength = this->m_filter_length;
	    fn1.filterShareAcrossDim = this->m_filter_across_dim;
	    
	    fn1.layerSize    = this->size();
	    fn1.maxLength    = timeLength;
	    fn1.parallel     = this->parallelSequences();
	    fn1.noncausal    = this->m_filter_noncausal;
		    
	    fn1.filterCoeffs  = helpers::getRawPointer(this->m_filter_coeffs);
	    fn1.inputErrors   = helpers::getRawPointer(this->outputErrors());
	    fn1.patTypes      = helpers::getRawPointer(this->patTypes());
	    
	    int n = timeLength * this->size();
	    thrust::for_each(
              thrust::make_zip_iterator(
		thrust::make_tuple(this->precedingLayer().outputErrors().begin(),
				   thrust::counting_iterator<int>(0))),
	      thrust::make_zip_iterator(
		thrust::make_tuple(this->precedingLayer().outputErrors().begin() + n,
				   thrust::counting_iterator<int>(0) + n)),
	      fn1);

	}else if (m_filter_mode == FILTERING_LAYER_MODE_SELECTIVE){
	    
	    internal::causalFilteringBackward_selective fn1;
	    fn1.filterLength     = this->m_filter_length;
	    fn1.filterShareAcrossDim = this->m_filter_across_dim;
	    
	    fn1.outputLayerSize  = this->size();
	    fn1.inputLayerSize   = this->precedingLayer().size();
	    fn1.filterNum        = this->m_filter_num;
	    fn1.maxLength        = timeLength;
	    fn1.parallel         = this->parallelSequences();
	    fn1.noncausal        = this->m_filter_noncausal;
	    

	    fn1.filterCoeffs     = helpers::getRawPointer(this->m_filter_coeffs);
	    fn1.inputData        = helpers::getRawPointer(this->precedingLayer().outputs());
	    fn1.inputErrors      = helpers::getRawPointer(this->outputErrors());
	    fn1.patTypes         = helpers::getRawPointer(this->patTypes());
	    
	    int n = timeLength * this->precedingLayer().size();
	    thrust::for_each(
              thrust::make_zip_iterator(
		thrust::make_tuple(this->precedingLayer().outputErrors().begin(),
				   thrust::counting_iterator<int>(0))),
	      thrust::make_zip_iterator(
		thrust::make_tuple(this->precedingLayer().outputErrors().begin() + n,
				   thrust::counting_iterator<int>(0) + n)),
	      fn1);

	    
	}else{
	    throw std::runtime_error("Error in filtering layer: Unknown filter mode");
	}
	
    }
    
    template <typename TDevice>
    void FilteringLayer<TDevice>::computeBackwardPass(const int timeStep, const int nnState)
    {
	throw std::runtime_error("FilteringLayer computeBackwardPass(timeStep) not implemented");
    }
    
    template <typename TDevice>
    void FilteringLayer<TDevice>::exportLayer(
	const helpers::JsonValue     &layersArray, 
	const helpers::JsonAllocator &allocator) const
    {
	TrainableLayer<TDevice>::exportLayer(layersArray, allocator);
        (*layersArray)[layersArray->Size() - 1].AddMember("shareAcrossDim", m_filter_across_dim,
							  allocator);
	(*layersArray)[layersArray->Size() - 1].AddMember("filterCoeffs",
							  m_filter_coeffs_str.c_str(),
							  allocator);
	if (m_filter_initial_keep)
	    (*layersArray)[layersArray->Size() - 1].AddMember("initialCondSmooth",
							      m_filter_initial_keep,
							      allocator);
	if (m_filter_noncausal)
	    (*layersArray)[layersArray->Size() - 1].AddMember("noncausal",
							      m_filter_noncausal,
							      allocator);
    }

    template <typename TDevice>
    void FilteringLayer<TDevice>::reduceOutputBuffer()
    {
	this->resizeOutputBuffer(this->parallelSequences() * this->size());	
	this->setSaveMemoryFlag(true);
	printf("\t[mem saved]");
    }
    
    template <typename TDevice>
    int FilteringLayer<TDevice>::outputBufPtrBias(const int timeStepTimesParallel,
						  const int nnState)
    {
	if (this->getSaveMemoryFlag()){
	    return timeStepTimesParallel * this->size();
	}else{
	    return 0;
	}
    }	

    template <typename TDevice>
    void FilteringLayer<TDevice>::clearAllBuffers()
    {
	this->clearOutputBuffer();
    }

    template <typename TDevice>
    void FilteringLayer<TDevice>::resizeAllBuffers(const int timeLength)
    {
	this->resizeOutputBuffer(timeLength * this->parallelSequences() * this->size());
    }

    template class FilteringLayer<Cpu>;
    template class FilteringLayer<Gpu>;
}
