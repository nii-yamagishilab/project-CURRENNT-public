/******************************************************************************
 * This file is an addtional component of CURRENNT. 
 * Xin WANG
 * National Institute of Informatics, Japan
 * 2019
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

#include "SelfAttention.hpp"
#include "../helpers/getRawPointer.cuh"
#include "../helpers/Matrix.hpp"
#include "../helpers/min.cuh"
#include "../helpers/max.cuh"
#include "../helpers/safeExp.cuh"

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

namespace internal {
namespace {
    struct CalculateOffsetFn
    {
        int layerSize;

        const real_t *outputs;

        __host__ __device__ real_t operator() (const int &patIdx) const
        {

            // search for the min and max output
            real_t max = helpers::NumericLimits<real_t>::min();
            real_t min = helpers::NumericLimits<real_t>::max();

            const real_t *offOutputs = &outputs[patIdx * layerSize];

            for (int i = 0; i < layerSize; ++i) {
                real_t x = offOutputs[i];
                min = helpers::min(min, x);
                max = helpers::max(max, x);
            }

            // calculate the offset
            real_t offset = (real_t)0.5 * (min + max);

            return offset;
        }
    };

    struct CalculateExpFn
    {
        int layerSize;

        const real_t *offsets;

        __host__ __device__ void operator() (const thrust::tuple<real_t&, int> &t) const
        {
            // unpack the tuple
            real_t output = t.get<0>();
            int outputIdx = t.get<1>();

            // calculate the pattern index
            int patIdx = outputIdx / layerSize;

            // check if we can stop the calculation
            real_t offset = offsets[patIdx];

            // calculate the exponent
            real_t x = helpers::safeExp(output - offset);

            // store the result
            t.get<0>() = x;
        }
    };

    /*
    struct SumUpOutputsFn
    {
        int layerSize;

        const real_t *outputs;

        __host__ __device__ void operator() (const thrust::tuple<real_t&, int> &t) const
        {
            // unpack the tuple
            int patIdx = t.get<1>();

            // sum up the outputs
            const real_t *offOutputs = &outputs[patIdx * layerSize];

            real_t sum = 0;
            for (int i = 0; i < layerSize; ++i)
                sum += offOutputs[i];

            // store the result
            t.get<0>() = sum;
        }
	};*/

    struct NormalizeOutputsFn
    {
        int layerSize;

        const real_t *normFacts;

        __host__ __device__ void operator() (const thrust::tuple<real_t&, int> &t) const
        {
            // unpack the tuple
            int outputIdx = t.get<1>();

            // calculate the pattern index
            int patIdx = outputIdx / layerSize;

            // check if we can stop the calculation
            real_t normFact = normFacts[patIdx];

            // calculate the normalized value
            real_t x = t.get<0>() / normFact;

            // store the result
            t.get<0>() = x;
        }
    };

    struct CalculateErrorOffsetFn
    {
        int layerSize;

        const real_t *outputs;
        const real_t *outputErrors;

        __host__ __device__ real_t operator() (const int &patIdx) const
        {
            // calculate the offset
            const real_t *offOutputs      = &outputs     [patIdx * layerSize];
            const real_t *offOutputErrors = &outputErrors[patIdx * layerSize];

            real_t offset = 0;
            for (int i = 0; i < layerSize; ++i)
                offset += offOutputs[i] * offOutputErrors[i];

            return offset;
        }
    };
    

    struct CalculateErrorsFn
    {
        int layerSize;

        const real_t *errorOffsets;

        __host__ __device__ void operator() (const thrust::tuple<real_t&, const real_t&, int> &t) const
        {
            // unpack the tuple
            int outputIdx = t.get<2>();

            // calculate the pattern index
            int patIdx = outputIdx / layerSize;

            // check if we can stop the calculation
            real_t offset = errorOffsets[patIdx];
    
            // calculate the delta
            real_t error  = t.get<0>();
            real_t output = t.get<1>();

            real_t x = output * (error - offset);

            // store the result
            t.get<0>() = x;
        }
    };

} // anonymous namespace
} // namespace internal


namespace layers {

    template <typename TDevice>
    SelfAttentionLayer<TDevice>::SelfAttentionLayer(const helpers::JsonValue &layerChild, 
						    const helpers::JsonValue &weightsSection, 
						    Layer<TDevice> &precedingLayer,
						    int             maxSeqLength,
						    int             layerID)
        : TrainableLayer<TDevice>(layerChild, weightsSection,
				  3, 0, precedingLayer, maxSeqLength, layerID)
    {
	if (this->parallelSequences() > 1)
	    throw std::runtime_error("Self-attention not implemented for parallel_seq > 1");	
	this->__allocateLocalMem();
    }

    template <typename TDevice>
    SelfAttentionLayer<TDevice>::~SelfAttentionLayer()
    {
    }

    template <typename TDevice>
    void SelfAttentionLayer<TDevice>::__allocateLocalMem()
    {

	Cpu::real_vector tmp;
	tmp.resize(this->outputs().size(), 0.0);

	m_mat_v = tmp;
	m_mat_k = tmp;
	m_mat_q = tmp;

	m_grad_buf = tmp;
	
	tmp.resize((this->outputs().size()/this->size()/this->parallelSequences()) *
		   (this->outputs().size()/this->size()),
		   0.0);
	
	m_align = tmp;
	m_align_grad = tmp;


	tmp.resize(this->outputs().size()/this->size(),0.0);
	m_softmax_buf = tmp;

	m_one_vector = tmp;
	thrust::fill(m_one_vector.begin(), m_one_vector.end(), 1.0);
    }

    template <typename TDevice>
    void SelfAttentionLayer<TDevice>::__clearLocalMem()
    {
	// to be implemented
	return;
    }

    template <typename TDevice>
    const std::string& SelfAttentionLayer<TDevice>::type() const
    {
        static const std::string s = "self_attention";
        return s;
    }

    template <typename TDevice>
    void SelfAttentionLayer<TDevice>::computeForwardPass(const int nnState)
    {
	if (this->getSaveMemoryFlag())
	    throw std::runtime_error("Memory save mode should be turned off");

	// matrix size for W_v, W_q, and W_k
	int tmp_mat_size = this->precedingLayer().size() * this->size();

	// total number of time steps
	int frame_num_total = this->curMaxSeqLength() * this->parallelSequences();
	
	
	{{
	    // step1. compute W_v x -> V, W_q x -> Q, W_k x -> K


	    helpers::Matrix<TDevice> mat_pre_o(&this->precedingLayer().outputs(), 
					       this->precedingLayer().size(), 
					       frame_num_total);
	    // W_v x
	    helpers::Matrix<TDevice> mat_w_v(&this->weights(),                  
					     this->precedingLayer().size(), this->size());

            helpers::Matrix<TDevice> mat_v(&m_mat_v, this->size(), frame_num_total);
            mat_v.assignProduct(mat_w_v, true, mat_pre_o, false);

	    // W_q x
	    helpers::Matrix<TDevice> mat_w_q(&this->weights(),                  
					     this->precedingLayer().size(), this->size(),
					     tmp_mat_size * 1);

            helpers::Matrix<TDevice> mat_q(&m_mat_q, this->size(), frame_num_total);
            mat_q.assignProduct(mat_w_q, true, mat_pre_o, false);
	    // scale W_q
	    thrust::transform(m_mat_q.begin(), m_mat_q.end(), 
			      thrust::make_constant_iterator(1.0/frame_num_total),
			      m_mat_q.begin(), thrust::multiplies<real_t>());

	    // W_k x
	    helpers::Matrix<TDevice> mat_w_k(&this->weights(),                  
					     this->precedingLayer().size(), this->size(),
					     tmp_mat_size * 2);

            helpers::Matrix<TDevice> mat_k(&m_mat_k, this->size(), frame_num_total);
            mat_k.assignProduct(mat_w_k, true, mat_pre_o, false);

	    

	    // step2. calculate the alignment matrix
	    if (this->parallelSequences() == 1){

		// Q^T * K 
		helpers::Matrix<TDevice> mat_align(&m_align, frame_num_total, frame_num_total);
		mat_align.assignProduct(mat_q, true, mat_k, false);

		
		// softmax over each time step
		// 1. calculate the offset to center the activations for safer exponentiation
		{{
		    internal::CalculateOffsetFn fn;
		    fn.layerSize = frame_num_total;
		    fn.outputs   = helpers::getRawPointer(m_align);

		    thrust::transform(
			thrust::counting_iterator<int>(0),
			thrust::counting_iterator<int>(0) + frame_num_total,
			m_softmax_buf.begin(),
			fn);
		}}

		// 2. calculate the exponent
		{{
		    internal::CalculateExpFn fn;
		    fn.layerSize = frame_num_total;
		    fn.offsets   = helpers::getRawPointer(m_softmax_buf);

		    int n = frame_num_total * frame_num_total;

		    thrust::for_each(
                     thrust::make_zip_iterator(
		      thrust::make_tuple(m_align.begin(),
					 thrust::counting_iterator<int>(0))),
		     thrust::make_zip_iterator(
		      thrust::make_tuple(m_align.begin()+n,
					 thrust::counting_iterator<int>(0)+n)),
		     fn);
		}}

		// 3. sum up all outputs for each pattern
		{{
	        helpers::Matrix<TDevice> mat_one(&m_one_vector, 1, frame_num_total);
	    	helpers::Matrix<TDevice> mat_sum(&m_softmax_buf, 1, frame_num_total);
		helpers::Matrix<TDevice> mat_align(&m_align, frame_num_total, frame_num_total);
		mat_sum.assignProduct(mat_one, false, mat_align, false);
		}}

		// 4. normalize the outputs
		{{
		    internal::NormalizeOutputsFn fn;
		    fn.layerSize = frame_num_total;
		    fn.normFacts = helpers::getRawPointer(m_softmax_buf);

		    int n = frame_num_total * frame_num_total;

		    thrust::for_each(
                     thrust::make_zip_iterator(
			thrust::make_tuple(m_align.begin(),
					   thrust::counting_iterator<int>(0))),
		     thrust::make_zip_iterator(
			thrust::make_tuple(m_align.begin()+n,
					   thrust::counting_iterator<int>(0)+n)),
		     fn);
	       }}
	    }

	    // step.3 compute output m_mat_v * align
	    // The conventional feedforward part
	    // collect outputs from preceding layer
	    {{
	        helpers::Matrix<TDevice> mat_align(&m_align, frame_num_total, frame_num_total);
	    	helpers::Matrix<TDevice> mat_v    (&m_mat_v, this->size(), frame_num_total);
		helpers::Matrix<TDevice> mat_out  (&this->_outputs(), this->size(), frame_num_total);
		mat_out.assignProduct(mat_v, false, mat_align, false);
	    }}	     
	}}
	
	// done
    }


    template <typename TDevice>
    void SelfAttentionLayer<TDevice>::computeForwardPass(const int timeStep,
							       const int nnState)
    {
	throw std::runtime_error("Self-attention not support online generation");
    }

    template <typename TDevice>
    void SelfAttentionLayer<TDevice>::computeBackwardPass(const int nnState)
    {
	// matrix size for W_v, W_q, and W_k
	int tmp_mat_size = this->precedingLayer().size() * this->size();

	// total number of time steps
	int frame_num_total = this->curMaxSeqLength() * this->parallelSequences();

	// step1. gradient w.r.t Q^T K
	{{
	    // gradient w.r.t softmax(Q^T, K)
	    helpers::Matrix<TDevice> mat_v   (&m_mat_v, this->size(), frame_num_total);
	    helpers::Matrix<TDevice> mat_grad_o(&this->outputErrors(),this->size(),frame_num_total);
	    helpers::Matrix<TDevice> mat_grad_align(&m_align_grad, frame_num_total, frame_num_total);
	    mat_grad_align.assignProduct(mat_v, true, mat_grad_o, false);

	    // calculate the error offset for each pattern
	    {{
            internal::CalculateErrorOffsetFn fn;
            fn.layerSize    = frame_num_total;
            fn.outputs      = helpers::getRawPointer(m_align);
            fn.outputErrors = helpers::getRawPointer(m_align_grad);

            thrust::transform(
                thrust::counting_iterator<int>(0),
                thrust::counting_iterator<int>(0) + frame_num_total,
                m_softmax_buf.begin(),
                fn);
	    }}
	    

	    // calculate the errors
	    {{
            internal::CalculateErrorsFn fn;
            fn.layerSize    = frame_num_total;
            fn.errorOffsets = helpers::getRawPointer(m_softmax_buf);

            int n = frame_num_total * frame_num_total;

            thrust::for_each(
                thrust::make_zip_iterator(
			thrust::make_tuple(m_align_grad.begin(),
					   m_align.begin(),
					   thrust::counting_iterator<int>(0))),
                thrust::make_zip_iterator(
			thrust::make_tuple(m_align_grad.begin()+n,
					   m_align.begin()+n,
					   thrust::counting_iterator<int>(0)+n)),
                fn);
	    }}
	}}
	
	thrust::fill(this->_weightUpdates().begin(), this->_weightUpdates().end(), 0.0);
	
	// step2. gradient w.r.t W_v, x for W_v
	{{
	    // \partial_E / \partial_o * Align^T
	    helpers::Matrix<TDevice> mat_grad_o(&this->outputErrors(),this->size(),frame_num_total);
	    helpers::Matrix<TDevice> mat_align (&m_align, frame_num_total, frame_num_total);
	    helpers::Matrix<TDevice> mat_grad_buf(&m_grad_buf, this->size(), frame_num_total);
	    mat_grad_buf.assignProduct(mat_grad_o, false, mat_align, true);

	    helpers::Matrix<TDevice> mat_x(&this->precedingLayer().outputs(),
					   this->precedingLayer().size(),frame_num_total);
	    helpers::Matrix<TDevice> mat_grad_w_v (&this->_weightUpdates(),
						   this->precedingLayer().size(), this->size());
	    mat_grad_w_v.assignProduct(mat_x, false, mat_grad_buf, true);

	    helpers::Matrix<TDevice> mat_w_v(&this->weights(),                  
					     this->precedingLayer().size(), this->size());
	    helpers::Matrix<TDevice> mat_grad_x (&this->precedingLayer().outputErrors(),
						   this->precedingLayer().size(), frame_num_total);
	    mat_grad_x.assignProduct(mat_w_v, false, mat_grad_buf, false);


	    // gradient w.r.t W_q, x for W_q
	    
	    // \partial_E / \partial_{Q}  = mat_K * [\partial_E / \partial_{Q^T}{K}]^T
	    helpers::Matrix<TDevice> mat_align_grad(&m_align_grad, frame_num_total, frame_num_total);
	    helpers::Matrix<TDevice> mat_k   (&m_mat_k, this->size(), frame_num_total);
	    mat_grad_buf.assignProduct(mat_k, false, mat_align_grad, true);
	    thrust::transform(m_grad_buf.begin(), m_grad_buf.end(), 
			      thrust::make_constant_iterator(1.0/frame_num_total),
			      m_grad_buf.begin(), thrust::multiplies<real_t>());

	    // \partial_E / \partial_W^Q
	    helpers::Matrix<TDevice> mat_grad_w_q (&this->_weightUpdates(),
						   this->precedingLayer().size(), this->size(),
						   tmp_mat_size * 1);
	    mat_grad_w_q.assignProduct(mat_x, false, mat_grad_buf, true);

	    // \partial_E / \partial_Q * \partial_Q / \partial_x
	    
	    helpers::Matrix<TDevice> mat_w_q(&this->weights(),                  
					     this->precedingLayer().size(), this->size(),
					     tmp_mat_size * 1);
	    mat_grad_x.addProduct(mat_w_q, false, mat_grad_buf, false);


	    // gradient w.r.t W_k, x for W_k
	    
	    // \partial_E / \partial_{K} = \mat_Q * \partial_E / \partial_{Q^T}{K} 
	    helpers::Matrix<TDevice> mat_q   (&m_mat_q, this->size(), frame_num_total);
	    mat_grad_buf.assignProduct(mat_q, false, mat_align_grad, false);

	    // \partial_E / \partial_W^k
	    helpers::Matrix<TDevice> mat_grad_w_k (&this->_weightUpdates(),
						   this->precedingLayer().size(), this->size(),
						   tmp_mat_size * 1);
	    mat_grad_w_k.assignProduct(mat_x, false, mat_grad_buf, true);

	    // \partial_E / \partial_K * \partial_K / \partial_x
	    
	    helpers::Matrix<TDevice> mat_w_k(&this->weights(),                  
					     this->precedingLayer().size(), this->size(),
					     tmp_mat_size * 2);
	    mat_grad_x.addProduct(mat_w_k, false, mat_grad_buf, false);
	    
	}}
	
    }


    template <typename TDevice>
    void SelfAttentionLayer<TDevice>::computeBackwardPass(const int timeStep,
							  const int nnState)
    {
	throw std::runtime_error("self-attention doesn't support online mode");
    }

    
    template <typename TDevice>
    void SelfAttentionLayer<TDevice>::exportLayer(
	const helpers::JsonValue     &layersArray, 
	const helpers::JsonAllocator &allocator) const
    {
        TrainableLayer<TDevice>::exportLayer(layersArray, allocator);	
    }


    template <typename TDevice>
    void SelfAttentionLayer<TDevice>::reduceOutputBuffer()
    {
	throw std::runtime_error("self-attention doesn't support online mode");
    }
    
    template <typename TDevice>
    int SelfAttentionLayer<TDevice>::outputBufPtrBias(const int timeStepTimesParallel,
							    const int nnState)
    {
	if (this->getSaveMemoryFlag()){
	    return timeStepTimesParallel * this->size();
	}else{
	    return 0;
	}
    }	


    template <typename TDevice>
    void SelfAttentionLayer<TDevice>::clearAllBuffers()
    {
	this->clearOutputBuffer();
	this->__clearLocalMem();
    }

    template <typename TDevice>
    void SelfAttentionLayer<TDevice>::resizeAllBuffers(const int timeLength)
    {
	this->resizeOutputBuffer(timeLength * this->parallelSequences() * this->size());
	this->__allocateLocalMem();
    }

    
    // explicit template instantiations
    template class SelfAttentionLayer<Cpu>;
    template class SelfAttentionLayer<Gpu>;

} // namespace layers

