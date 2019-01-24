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

#include "SoftmaxLayer.hpp"
#include "../helpers/getRawPointer.cuh"
#include "../helpers/min.cuh"
#include "../helpers/max.cuh"
#include "../helpers/safeExp.cuh"
#include "../activation_functions/Identity.cuh"

#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/for_each.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>

#define SKIP_MARKER helpers::NumericLimits<real_t>::max()


namespace internal {
namespace {

    struct CalculateOffsetFn
    {
        int layerSize;

        const real_t *outputs;

        const char *patTypes;

        __host__ __device__ real_t operator() (const int &patIdx) const
        {
            // check if the pattern belongs to a sequence;
            // if not we return a certain number to avoid 
            // looking up patTypes for future calculations
            if (patTypes[patIdx] == PATTYPE_NONE)
                return SKIP_MARKER;

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
            if (offset == SKIP_MARKER)
                return;

            // calculate the exponent
            real_t x = helpers::safeExp(output - offset);

            // store the result
            t.get<0>() = x;
        }
    };

    struct SumUpOutputsFn
    {
        int layerSize;

        const real_t *outputs;

        __host__ __device__ void operator() (const thrust::tuple<real_t&, int> &t) const
        {
            // unpack the tuple
            int patIdx = t.get<1>();

            // check if the pattern belongs to a sequence
            if (t.get<0>() == SKIP_MARKER)
                return;

            // sum up the outputs
            const real_t *offOutputs = &outputs[patIdx * layerSize];

            real_t sum = 0;
            for (int i = 0; i < layerSize; ++i)
                sum += offOutputs[i];

            // store the result
            t.get<0>() = sum;
        }
    };

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
            if (normFact == SKIP_MARKER)
                return;

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

        const char *patTypes;

        __host__ __device__ real_t operator() (const int &patIdx) const
        {
            // check if the pattern belongs to a sequence;
            // if not we return a certain number to avoid 
            // looking up patTypes for future calculations
            if (patTypes[patIdx] == PATTYPE_NONE)
                return SKIP_MARKER;

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
            if (offset == SKIP_MARKER)
                return;

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

    template <typename TDevice, typename TFfActFn>
    SoftmaxLayer<TDevice, TFfActFn>::SoftmaxLayer(const helpers::JsonValue &layerChild, 
						  const helpers::JsonValue &weightsSection,
						  Layer<TDevice> &precedingLayer,
						  int             maxSeqLength,
						  int             layerID)
        : FeedForwardLayer<TDevice, TFfActFn>(layerChild,
					      weightsSection,
					      precedingLayer,
					      maxSeqLength,
					      layerID)
    {
        // resize the vector for temporary values
        m_patTmp.resize(this->patTypes().size());
    }

    template <typename TDevice, typename TFfActFn>
    SoftmaxLayer<TDevice, TFfActFn>::~SoftmaxLayer()
    {
    }

    template <typename TDevice, typename TFfActFn>
    const std::string& SoftmaxLayer<TDevice, TFfActFn>::type() const
    {
        static const std::string s = "softmax";
        return s;
    }

    template <typename TDevice, typename TFfActFn>
    void SoftmaxLayer<TDevice, TFfActFn>::computeForwardPass(const int nnState)
    {
        // compute the forward pass of the feedforward layer
        FeedForwardLayer<TDevice, TFfActFn>::computeForwardPass(nnState);

        // calculate the offset to center the activations for safer exponentiation
        {{
            internal::CalculateOffsetFn fn;
            fn.layerSize = this->size();
            fn.outputs   = helpers::getRawPointer(this->_outputs());
            fn.patTypes  = helpers::getRawPointer(this->patTypes());

            thrust::transform(
                thrust::counting_iterator<int>(0),
                thrust::counting_iterator<int>(0) + this->curMaxSeqLength() * this->parallelSequences(),
                m_patTmp.begin(),
                fn
                );
        }}

        // calculate the exponent
        {{
            internal::CalculateExpFn fn;
            fn.layerSize = this->size();
            fn.offsets   = helpers::getRawPointer(m_patTmp);

            int n = this->curMaxSeqLength() * this->parallelSequences() * this->size();

            thrust::for_each(
                thrust::make_zip_iterator(thrust::make_tuple(this->_outputs().begin(),   thrust::counting_iterator<int>(0))),
                thrust::make_zip_iterator(thrust::make_tuple(this->_outputs().begin()+n, thrust::counting_iterator<int>(0)+n)),
                fn
                );
        }}

        // sum up all outputs for each pattern
        {{
            internal::SumUpOutputsFn fn;
            fn.layerSize = this->size();
            fn.outputs   = helpers::getRawPointer(this->_outputs());

            int n = this->curMaxSeqLength() * this->parallelSequences();

            thrust::for_each(
                thrust::make_zip_iterator(thrust::make_tuple(m_patTmp.begin(),   thrust::counting_iterator<int>(0))),
                thrust::make_zip_iterator(thrust::make_tuple(m_patTmp.begin()+n, thrust::counting_iterator<int>(0)+n)),
                fn
                );
        }}

        // normalize the outputs
        {{
            internal::NormalizeOutputsFn fn;
            fn.layerSize = this->size();
            fn.normFacts = helpers::getRawPointer(m_patTmp);

            int n = this->curMaxSeqLength() * this->parallelSequences() * this->size();

            thrust::for_each(
                thrust::make_zip_iterator(thrust::make_tuple(this->_outputs().begin(),   thrust::counting_iterator<int>(0))),
                thrust::make_zip_iterator(thrust::make_tuple(this->_outputs().begin()+n, thrust::counting_iterator<int>(0)+n)),
                fn
                );
        }}
    }

    template <typename TDevice, typename TFfActFn>
    void SoftmaxLayer<TDevice, TFfActFn>::computeForwardPass(const int timeStep, const int nnState)
    {
	throw std::runtime_error("Not implemented");
    }

    template <typename TDevice, typename TFfActFn>
    void SoftmaxLayer<TDevice, TFfActFn>::computeBackwardPass(const int nnState)
    {
        // calculate the error offset for each pattern
        {{
            internal::CalculateErrorOffsetFn fn;
            fn.layerSize    = this->size();
            fn.outputs      = helpers::getRawPointer(this->_outputs());
            fn.outputErrors = helpers::getRawPointer(this->outputErrors());
            fn.patTypes     = helpers::getRawPointer(this->patTypes());

            thrust::transform(
                thrust::counting_iterator<int>(0),
                thrust::counting_iterator<int>(0) + this->curMaxSeqLength() * this->parallelSequences(),
                m_patTmp.begin(),
                fn
                );
        }}

        // calculate the errors
        {{
            internal::CalculateErrorsFn fn;
            fn.layerSize    = this->size();
            fn.errorOffsets = helpers::getRawPointer(m_patTmp);

            int n = this->curMaxSeqLength() * this->parallelSequences() * this->size();

            thrust::for_each(
                thrust::make_zip_iterator(thrust::make_tuple(this->outputErrors().begin(),   this->_outputs().begin(),   thrust::counting_iterator<int>(0))),
                thrust::make_zip_iterator(thrust::make_tuple(this->outputErrors().begin()+n, this->_outputs().begin()+n, thrust::counting_iterator<int>(0)+n)),
                fn
                );
        }}

        // compute the backward pass of the feedforward layer
        FeedForwardLayer<TDevice, TFfActFn>::computeBackwardPass(nnState);
    }


    // explicit template instantiations
    template class SoftmaxLayer<Cpu, activation_functions::Identity>;
    template class SoftmaxLayer<Gpu, activation_functions::Identity>;

} // namespace layers
