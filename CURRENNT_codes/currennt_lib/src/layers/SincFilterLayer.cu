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

#include "SincFilterLayer.hpp"
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

#define SINCFILTER_DEFAULT_FILTERL_ENGTH 30
#define PI_DEFINITION 3.141592653589793f
#define COEF_SUM_FLOOR 0.000001

namespace internal {
namespace {

    // Do filtering on input signal (based on function from FilteringLayer)
    struct causalFilteringForward
    {
	// t.get<0>() output buffer
	// t.get<1>() output buffer index

	int         output_dim;
	int         parallel;
	int         initSmooth;
	
	const char *patTypes;	
	
	int         filterLength;
	real_t     *filterCoeffs;
	
	real_t     *input;
	int         input_dim;
	int         input_dim_shift;


	// From 1 : T (of the previous layer)
	__host__ __device__ void operator() (const thrust::tuple<real_t&, int> &t) const
	{
	    int outputIdx  = t.get<1>();
	    int dimIdx     = outputIdx % output_dim;  // dimension index
	    int timeIdx    = outputIdx / output_dim;  // time index (regardless of parallel)

	   
	    int BlockIdx   = timeIdx / parallel;     // time index (considering parallel mode)
	    int BlockInIdx = timeIdx % parallel;     // index within a parallel block


	    if (patTypes[timeIdx] == PATTYPE_NONE){
		t.get<0>() = 0;
		return;
	    }		

	    // accumulate and update
	    real_t tmp = 0.0;
	    real_t filterCoeff;
	    
	    for (int idx = 0 ; idx < filterLength; idx++){

		// get the filter coefficient for one step
		// a_0 + a_1 z^-1 + ... + a_N z^-N
		// [a_0, a_1, ..., a_N]
		filterCoeff = filterCoeffs[timeIdx * filterLength + idx];
		
		if ((BlockIdx - idx) >= 0){
		    tmp += (input[((BlockIdx - idx) * parallel + BlockInIdx) * input_dim + dimIdx
				      + input_dim_shift]
			    * filterCoeff);
		}else if (initSmooth){
		    tmp += (input[BlockInIdx * input_dim + dimIdx + input_dim_shift]
			    * filterCoeff);
		}else{
		    // nothing
		}
	    }
	    t.get<0>() = tmp;
	    
	}
    };

    // Back-propagation to input signals (based on function from FilteringLayer)
    struct causalFilteringBackward
    {
	int         output_dim;
	int         parallel;
	const char *patTypes;
	
	
	int         filterLength;
	real_t     *filterCoeffs;
	
	int         maxLength;

	real_t     *inputErrors;
	int         input_dim;
	int         input_dim_shift;

	real_t     *outputErrors;

	// From 1 : T (of the previous layer)
	__host__ __device__ void operator() (const thrust::tuple<real_t&, int> &t) const
	{
	    
	    int outputIdx  = t.get<1>();
	    int dimIdx     = outputIdx % output_dim;  // dimension index
	    int timeIdx    = outputIdx / output_dim;  // time index (regardless of parallel)
	    
	    int BlockIdx   = timeIdx / parallel;     // time index (considering parallel mode)
	    int BlockInIdx = timeIdx % parallel;     // index within a parallel block


	    if (patTypes[timeIdx] == PATTYPE_NONE){
		inputErrors[timeIdx * input_dim + dimIdx + input_dim_shift] = 0.0;
		return;
	    }		

	    // accumulate and update
	    real_t tmp = 0.0;
	    real_t filterCoeff;
	    for (int idx = 0 ; idx < filterLength; idx++){

		// get the filter coefficient for one step
		// a_0 + a_1 z^-1 + ... + a_N z^-N
		// [a_0, a_1, ..., a_N]
		filterCoeff = filterCoeffs[timeIdx * filterLength + idx];
		
		if (((BlockIdx + idx) * parallel + BlockInIdx) < maxLength &&
		    patTypes[((BlockIdx + idx) * parallel + BlockInIdx)] != PATTYPE_NONE){
		    tmp += (outputErrors[((BlockIdx + idx) * parallel + BlockInIdx) * output_dim
					 + dimIdx] * filterCoeff);
		}
	    }
	    inputErrors[timeIdx * input_dim + dimIdx + input_dim_shift] = tmp;
	}
    };


    struct freqToFilters
    {
	// Foreach 1:T
	// t.get<1>(), time index

	real_t *lp_coef;
	real_t *hp_coef;
	real_t *w_buf;
	real_t *scale_buf;
	
	int     w_buf_dim;
	int     w_buf_shift;
	
	int     filter_length;  
	
	__host__ __device__ void operator() (const thrust::tuple<real_t&, int> &t) const
	{
	    int timeIdx = t.get<1>();
	    
	    // cut-off frequency
	    real_t freq = w_buf[timeIdx * w_buf_dim + w_buf_shift];

	    // filter range [-half_length, half_length]
	    int half_length = (filter_length - 1) / 2; 

	    
	    int    tap_idx;  // index of filter \in [-half_length, half_length]
	    real_t coef;     // coefficient value
	    real_t win_coef    = 0.0;   // window coefficiet;
		    
	    // scaling factor, to make sure the gain at 0 or pi is 1
	    //  for low-pass, just make sure Freq(w=0) = \sum coeff = 1
	    //  for high-pass, just make sure Freq(w=pi) = \sum coeff * (-1) ^ {n} = 1
	    real_t lp_coef_sum = 0.0;   // sum of the lp_coef
	    real_t hp_coef_sum = 0.0;   // sum of the hp_coef * (-1)^{n}

	    // step1. get raw cofficients
	    for (int buf_idx = 0; buf_idx < filter_length; buf_idx++){
		
		
		tap_idx = buf_idx - half_length;

		// hamming window
		win_coef  = 0.54 + 0.46 * cos(2.0 * PI_DEFINITION * tap_idx / filter_length);
		
		// low-pass filter 
		// h^lp[tap_idx] = sin(pi * freq * tap_idx) / (pi * tap_idx)
		if (tap_idx == 0)
		    coef = freq;
		else
		    coef = sin(PI_DEFINITION * freq * tap_idx) / (PI_DEFINITION * tap_idx);
		// save coef
		coef = coef * win_coef;
		lp_coef[timeIdx * filter_length + buf_idx] = coef;
		lp_coef_sum += coef;
		
		// high-pass filter
		// h^lp[tap_idx] = sin(pi*tap_idx)/(pi*tap_idx) -
		//                   sin(pi * freq * tap_idx) / (pi * tap_idx)
		if (tap_idx == 0)
		    coef = 1.0 - freq;
		else
		    coef = ((sin(PI_DEFINITION * tap_idx) - sin(PI_DEFINITION * freq * tap_idx))
			    / (PI_DEFINITION * tap_idx));
		
		// save coef
		coef = coef * win_coef;
		hp_coef[timeIdx * filter_length + buf_idx] = coef;
		if (tap_idx % 2 == 0)
		    hp_coef_sum += coef;
		else
		    hp_coef_sum -= coef;
	    }

	    // step2. scaling, to make sure the gain at 0 or pi is 1
	    //  for low-pass, just make sure Freq(w=0) = \sum coeff = 1
	    //  for high-pass, just make sure Freq(w=pi) = \sum coeff * (-1) ^ {n} = 1
	    if (lp_coef_sum < COEF_SUM_FLOOR) lp_coef_sum = COEF_SUM_FLOOR;
	    if (hp_coef_sum < COEF_SUM_FLOOR) hp_coef_sum = COEF_SUM_FLOOR;
	    scale_buf[timeIdx * 2 + 0] = lp_coef_sum;
	    scale_buf[timeIdx * 2 + 1] = hp_coef_sum;
	    
	    for (int buf_idx = 0; buf_idx < filter_length; buf_idx++){
		lp_coef[timeIdx * filter_length + buf_idx] =
		    lp_coef[timeIdx * filter_length + buf_idx] / lp_coef_sum;
		hp_coef[timeIdx * filter_length + buf_idx] =
		    hp_coef[timeIdx * filter_length + buf_idx] / hp_coef_sum;
	    }
	}
	
    };


    struct gradToFreq
    {
	// Foreach 1:T
	// t.get<0>() \partial_E / \partial_o_t
	// t.get<1>() timeIdx

	real_t *lp_sig_out;
	real_t *hp_sig_out;
	real_t *lp_coef;
	real_t *hp_coef;
	real_t *scale_buf;

	real_t *w_buf;
	real_t *w_grad_buf;
	
	int     w_buf_dim;
	int     w_buf_shift;
	
	int     filter_length;  
	int     parallel;

	const char *patTypes;
	
	__host__ __device__ void operator() (const thrust::tuple<real_t&, int> &t) const
	{
	    int timeIdx = t.get<1>();

	    int BlockIdx   = timeIdx / parallel;     // time index (considering parallel mode)
	    int BlockInIdx = timeIdx % parallel;     // index within a parallel block

	    // cut-off frequency
	    real_t freq = w_buf[timeIdx * w_buf_dim + w_buf_shift];

	    // half length of the filter
	    int half_length = (filter_length - 1) / 2;

	    // scaling factor (see function freqToFilters)
	    int scale_lp = scale_buf[timeIdx * 2 + 0];  // scaling factor, lp
	    int scale_hp = scale_buf[timeIdx * 2 + 1];  // scaling factor, hp

	    // void time step
	    if (patTypes[timeIdx] == PATTYPE_NONE){
		w_grad_buf[timeIdx * w_buf_dim + w_buf_shift] = 0.0;
		return;
	    }		

	    //
	    real_t grad = 0;          // grad w.r.t. w
	    real_t lp_sub_grad = 0;   // intermediate buffer
	    real_t hp_sub_grad = 0;   // intermediate buffer
	    real_t grad_lp_w = 0;     // \partial_h^{lp} / \partial_w
	    real_t grad_hp_w = 0;     // \partial_h^{hp} / \partial_w

	    real_t lp_coef_val = 0;   // value of lp filter coefficient
	    real_t hp_coef_val = 0;   // value of hp filter coefficient
	    
	    real_t tmp_value;
	    int    tap_idx;

	    // \partial_E / \partial_w_t = \partial_E / \partial_o_t *
	    //    (  \sum_m=0^{filter_length} o^{lp}_[t-m] * \partial_h^{lp}_m,t / \partial_w_t
	    //     + \sum_m=0^{filter_length} o^{hp}_[t-m] * \partial_h^{hp}_m,t / \partial_w_t )
	    // 
	    for (int buf_idx = 0; buf_idx < filter_length; buf_idx++){

		// drop the time index, ^{lp},^{hp}
		// \partial_h_m/ \partial_w = 
		//  sum_{n}\partial_h_m/\partial_h_n^{unscaled} * \partial_h_n^{unscaled}/\partial_w
		
		lp_sub_grad = 0;
		hp_sub_grad = 0;
		for (int buf_idx_2 = 0; buf_idx_2 < filter_length; buf_idx_2++){
		    tap_idx = buf_idx_2 - half_length;

		    // cos(pi w n) * hamming(n)
		    tmp_value = cos(PI_DEFINITION * freq * tap_idx) *
			(0.54 + 0.46 * cos(2.0 * PI_DEFINITION * tap_idx / filter_length));
		    
		    lp_sub_grad += tmp_value;
		    if (tap_idx % 2 ==0)
			hp_sub_grad += tmp_value;
		    else
			hp_sub_grad -= tmp_value;
			
		}

		// \partial_h^{lp} / \partial_w
		// \partial_h^{hp} / \partial_w
		tap_idx = buf_idx - half_length;
		tmp_value = cos(PI_DEFINITION * freq * tap_idx) *
		    (0.54 + 0.46 * cos(2.0 * PI_DEFINITION * tap_idx / filter_length));
		lp_coef_val = lp_coef[timeIdx * filter_length + buf_idx];
		hp_coef_val = hp_coef[timeIdx * filter_length + buf_idx];
		
		grad_lp_w = tmp_value / scale_lp - lp_coef_val / scale_lp / scale_lp * lp_sub_grad;
		grad_hp_w = hp_coef_val / scale_hp / scale_hp * hp_sub_grad - tmp_value / scale_hp;

		// lp part
		if ((BlockIdx - buf_idx) >= 0){
		    grad += lp_sig_out[(BlockIdx - buf_idx) * parallel + BlockInIdx] * grad_lp_w;
		    grad += hp_sig_out[(BlockIdx - buf_idx) * parallel + BlockInIdx] * grad_hp_w;
		}
		
	    }
	    w_grad_buf[timeIdx * w_buf_dim + w_buf_shift] = grad * t.get<0>();
	    
	}    

    };
   
    

} // anonymous namespace
} // namespace internal


namespace layers {

    
    template <typename TDevice>
    SincFilterLayer<TDevice>::SincFilterLayer(const helpers::JsonValue &layerChild, 
							const helpers::JsonValue &weightsSection, 
							Layer<TDevice> &precedingLayer,
							int             maxSeqLength,
							int             layerID)
        : TrainableLayer<TDevice>(layerChild, weightsSection, 0, 0,
				  precedingLayer, maxSeqLength, layerID)
    {
	
	// Initialization for batch normalization
	m_num_tap = (layerChild->HasMember("filterLength")?
		     (static_cast<int>((*layerChild)["filterLength"].GetInt())) :
		     SINCFILTER_DEFAULT_FILTERL_ENGTH);
	
	m_initSmooth = ((layerChild->HasMember("initialCondSmooth")) ? 
			((*layerChild)["initialCondSmooth"].GetInt()) : 0);

	if (m_num_tap % 2 == 0)
	    m_num_tap = m_num_tap / 2 * 2 + 1; 
	    
	if (this->size() != 1 || this->precedingLayer().size() != 3)
	    throw std::runtime_error("Error: sincFilterLayer only support 3d input, 1d output");
	
	this->__allocateLocalMem();
    }

    template <typename TDevice>
    SincFilterLayer<TDevice>::~SincFilterLayer()
    {
    }

    template <typename TDevice>
    void SincFilterLayer<TDevice>::__allocateLocalMem()
    {
	m_sig_lp_buf = this->outputs();
	m_sig_hp_buf = this->outputs();

	// initialize the coefficients
	Cpu::real_vector tmp;
	tmp.resize(this->outputs().size() * m_num_tap, 0.0);
	    
	m_lp_coeff = tmp;
	m_hp_coeff = tmp;

	tmp.resize(this->outputs().size() * 2, 0.0);
	m_coef_scale_buf = tmp;
    }

    template <typename TDevice>
    void SincFilterLayer<TDevice>::__clearLocalMem()
    {
	// clear the large memory buffer
	m_sig_lp_buf.clear();
	m_sig_lp_buf.shrink_to_fit();
	m_sig_hp_buf.clear();
	m_sig_hp_buf.shrink_to_fit();
	m_lp_coeff.clear();
	m_lp_coeff.shrink_to_fit();
	m_hp_coeff.clear();
	m_hp_coeff.shrink_to_fit();
	m_coef_scale_buf.clear();
	m_coef_scale_buf.shrink_to_fit();
    }

    template <typename TDevice>
    const std::string& SincFilterLayer<TDevice>::type() const
    {
        static std::string s;
        if (s.empty()) s = "sinc_filter";
        return s;
    }

    template <typename TDevice>
    void SincFilterLayer<TDevice>::__build_filter_coeff()
    {
	// create the filter coefficients based on the parameter
	int timeLength = this->curMaxSeqLength() * this->parallelSequences();
	
	// build low-pass and high-pass together
	{
	    internal::freqToFilters fn1;
	    fn1.lp_coef = helpers::getRawPointer(this->m_lp_coeff);
	    fn1.hp_coef = helpers::getRawPointer(this->m_hp_coeff);
	    fn1.scale_buf = helpers::getRawPointer(this->m_coef_scale_buf);
	    
	    fn1.filter_length = m_num_tap;
	    
	    fn1.w_buf   = helpers::getRawPointer(this->precedingLayer().outputs());
	    fn1.w_buf_dim = this->precedingLayer().size();
	    // freq is stored in the last dimension
	    fn1.w_buf_shift = this->precedingLayer().size()-1; 
		
	    thrust::for_each(
               thrust::make_zip_iterator(
		  thrust::make_tuple(this->outputs().begin(),
				     thrust::counting_iterator<int>(0))),
	       thrust::make_zip_iterator(
		  thrust::make_tuple(this->outputs().begin() + timeLength,
				     thrust::counting_iterator<int>(0)  + timeLength)),
	       fn1);
	}
	
    }
    
    
    template <typename TDevice>
    void SincFilterLayer<TDevice>::computeForwardPass(const int nnState)
    {
	if (this->getSaveMemoryFlag())
	    throw std::runtime_error("Memory save mode should be turned off");

	int timeLength = this->curMaxSeqLength() * this->parallelSequences();
	
	// Step1. convert input signal input filter coefficients
	this->__build_filter_coeff();
	
	// Step2. do filtering
	{
	    internal::causalFilteringForward fn1;
	    fn1.output_dim   = this->size();
	    fn1.parallel     = this->parallelSequences();
	    fn1.patTypes     = helpers::getRawPointer(this->patTypes());

	    fn1.input_dim    = this->precedingLayer().size();
	    fn1.input        = helpers::getRawPointer(this->precedingLayer().outputs());
	    fn1.filterLength = m_num_tap;
	    fn1.initSmooth   = m_initSmooth;
	    
	    int n = timeLength * this->size();
	    
	    // low-pass filtering part
	    fn1.filterCoeffs = helpers::getRawPointer(this->m_lp_coeff);
	    fn1.input_dim_shift = 0;   // assume the 1st dimension is the input for low-pass
	    
	    
	    thrust::for_each(
               thrust::make_zip_iterator(
		  thrust::make_tuple(m_sig_lp_buf.begin(),
				     thrust::counting_iterator<int>(0))),
	       thrust::make_zip_iterator(
		  thrust::make_tuple(m_sig_lp_buf.begin()              + n,
				     thrust::counting_iterator<int>(0) + n)),
	       fn1);
	    
	    // high-pass filtering
	    fn1.filterCoeffs = helpers::getRawPointer(this->m_hp_coeff);
	    fn1.input_dim_shift = 1;   // assume the 2nd dimension is the input for high-pass
	    
	    thrust::for_each(
               thrust::make_zip_iterator(
		  thrust::make_tuple(m_sig_hp_buf.begin(),
				     thrust::counting_iterator<int>(0))),
	       thrust::make_zip_iterator(
		  thrust::make_tuple(m_sig_hp_buf.begin()              + n,
				     thrust::counting_iterator<int>(0) + n)),
	       fn1);


	    thrust::fill(this->outputs().begin(), this->outputs().end(), 0.0);
	    
	    // sum the signal
	    thrust::transform(this->outputs().begin(),
			      this->outputs().begin() + n,
			      m_sig_hp_buf.begin(),
			      this->outputs().begin(),
			      thrust::plus<real_t>());
	    thrust::transform(this->outputs().begin(),
			      this->outputs().begin() + n,
			      m_sig_lp_buf.begin(),
			      this->outputs().begin(),
			      thrust::plus<real_t>());
	}
	// done
	
    }


    template <typename TDevice>
    void SincFilterLayer<TDevice>::computeForwardPass(const int timeStep,
							       const int nnState)
    {
	throw std::runtime_error("SincFilterLayer computeForwardPass(timeStep) not implemented");
    }

    
    template <typename TDevice>
    void SincFilterLayer<TDevice>::computeBackwardPass(const int nnState)
    {
	if (this->getSaveMemoryFlag())
	    throw std::runtime_error("Memory save mode should be turned off");

	int timeLength = this->curMaxSeqLength() * this->parallelSequences();
	
	// Step1. propagate back to input signals
	{
	    internal::causalFilteringBackward fn2;
	    
	    fn2.output_dim   = this->size();
	    fn2.parallel     = this->parallelSequences();
	    fn2.patTypes     = helpers::getRawPointer(this->patTypes());

	    fn2.filterLength = m_num_tap;
	    
	    fn2.maxLength    = timeLength;
	    
	    fn2.input_dim    = this->precedingLayer().size();
	    fn2.inputErrors  = helpers::getRawPointer(this->precedingLayer().outputErrors());
	    fn2.outputErrors = helpers::getRawPointer(this->outputErrors());

	    int n = timeLength * this->size();

	    // low-pass filtering part	    
	    fn2.filterCoeffs = helpers::getRawPointer(this->m_lp_coeff);
	    fn2.input_dim_shift = 0;   // assume the 1st dimension is the input for low-pass
	    
	    thrust::for_each(
               thrust::make_zip_iterator(
		  thrust::make_tuple(this->outputErrors().begin(),
				     thrust::counting_iterator<int>(0))),
	       thrust::make_zip_iterator(
		  thrust::make_tuple(this->outputErrors().begin()      + n,
				     thrust::counting_iterator<int>(0) + n)),
	       fn2);

	    // high-pass filtering part
	    fn2.filterCoeffs = helpers::getRawPointer(this->m_hp_coeff);
	    fn2.input_dim_shift = 1;   // assume the 2nd dimension is the input for high-pass
	    
	    thrust::for_each(
               thrust::make_zip_iterator(
		  thrust::make_tuple(this->outputErrors().begin(),
				     thrust::counting_iterator<int>(0))),
	       thrust::make_zip_iterator(
		  thrust::make_tuple(this->outputErrors().begin()      + n,
				     thrust::counting_iterator<int>(0) + n)),
	       fn2);
	}
	
	// Step2. compute gradients w.r.t cut-off frequency
	{
	    internal::gradToFreq fn3;
	    fn3.lp_sig_out = helpers::getRawPointer(this->m_sig_lp_buf);
	    fn3.hp_sig_out = helpers::getRawPointer(this->m_sig_hp_buf);
	    fn3.lp_coef    = helpers::getRawPointer(this->m_lp_coeff);
	    fn3.hp_coef    = helpers::getRawPointer(this->m_hp_coeff);
	    fn3.scale_buf  = helpers::getRawPointer(this->m_coef_scale_buf);
	    fn3.w_buf      = helpers::getRawPointer(this->precedingLayer().outputs());
	    fn3.w_grad_buf = helpers::getRawPointer(this->precedingLayer().outputErrors());
	    
	    fn3.w_buf_dim  = this->precedingLayer().size();
	    fn3.w_buf_shift = 2;
	    fn3.filter_length = this->m_num_tap;
	    fn3.parallel = this->parallelSequences();
	    
	    fn3.patTypes     = helpers::getRawPointer(this->patTypes());
	    
	    thrust::for_each(
               thrust::make_zip_iterator(
		  thrust::make_tuple(this->outputs().begin(),
				     thrust::counting_iterator<int>(0))),
	       thrust::make_zip_iterator(
		  thrust::make_tuple(this->outputs().begin()      + timeLength,
				     thrust::counting_iterator<int>(0) + timeLength)),
	       fn3);
	    
	}
	
    }


    template <typename TDevice>
    void SincFilterLayer<TDevice>::computeBackwardPass(const int timeStep,
								const int nnState)
    {
	throw std::runtime_error("SincFilterLayer computeBackwardPass(timeStep) not implemented");
    }


    template <typename TDevice>
    void SincFilterLayer<TDevice>::exportLayer(const helpers::JsonValue     &layersArray, 
						       const helpers::JsonAllocator &allocator) const
    {
        TrainableLayer<TDevice>::exportLayer(layersArray, allocator);
	(*layersArray)[layersArray->Size() - 1].AddMember("filterLength", m_num_tap, allocator);

	if (m_initSmooth)
	    (*layersArray)[layersArray->Size() - 1].AddMember("initialCondSmooth", m_initSmooth,
							      allocator);
    }


    template <typename TDevice>
    void SincFilterLayer<TDevice>::reduceOutputBuffer()
    {
	
	this->resizeOutputBuffer(this->parallelSequences() * this->size());
	this->__clearLocalMem();
	this->setSaveMemoryFlag(true);
	printf("\t[mem saved]");
    }
    
    template <typename TDevice>
    int SincFilterLayer<TDevice>::outputBufPtrBias(const int timeStepTimesParallel,
							    const int nnState)
    {
	if (this->getSaveMemoryFlag()){
	    return timeStepTimesParallel * this->size();
	}else{
	    return 0;
	}
    }	


    template <typename TDevice>
    void SincFilterLayer<TDevice>::clearAllBuffers()
    {
	this->clearOutputBuffer();
	this->__clearLocalMem();
    }

    template <typename TDevice>
    void SincFilterLayer<TDevice>::resizeAllBuffers(const int timeLength)
    {
	this->resizeOutputBuffer(timeLength * this->parallelSequences() * this->size());
	this->__allocateLocalMem();
    }

    
    // explicit template instantiations
    template class SincFilterLayer<Cpu>;
    template class SincFilterLayer<Gpu>;

} // namespace layers

