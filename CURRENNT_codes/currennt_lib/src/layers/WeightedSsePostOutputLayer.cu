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

#include "WeightedSsePostOutputLayer.hpp"
#include "../helpers/getRawPointer.cuh"

#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>


namespace internal {
namespace {


    struct ComputeWeightedSseFn
    {
        int layerSize;

        const char *patTypes;
        real_t *targets;
        real_t *outputs;

        __host__ __device__ real_t operator() (int index) const
        {
            // unpack the tuple
            real_t target = targets[index * 2];
            real_t output = outputs[index];
            real_t weight = targets[index * 2 + 1];

            // check if we have to skip this value
            int patIdx = index / layerSize;
            if (patTypes[patIdx] == PATTYPE_NONE)
                return 0;

            // calculate the error
            real_t diff = (output - target) * weight;
            return (diff * diff);
        }
    };

    struct ComputeOutputErrorFn
    {
        int layerSize;

        const char *patTypes;
        real_t *targets;
        real_t *outputs;

        __host__ __device__ real_t operator() (int index) const
        {
            // unpack the tuple
            real_t target = targets[index * 2];
            real_t output = outputs[index];
            real_t weight = targets[index * 2 + 1];

            // calculate the pattern index
            int patIdx = index / layerSize;

            // check if the pattern is a dummy
            if (patTypes[patIdx] == PATTYPE_NONE)
                return 0;

            // calculate the error
            real_t error = (output - target) * weight;

            return error;
        }
    };
    
} // anonymous namespace
} // namespace anonymous


namespace layers {

    template <typename TDevice>
    WeightedSsePostOutputLayer<TDevice>::WeightedSsePostOutputLayer(
				const helpers::JsonValue &layerChild,
				Layer<TDevice> &precedingLayer,
				int             maxSeqLength,
				int             layerID)
        : PostOutputLayer<TDevice>  (layerChild, precedingLayer,
				     precedingLayer.size() * 2, maxSeqLength, layerID)
    {
    }

    template <typename TDevice>
    WeightedSsePostOutputLayer<TDevice>::~WeightedSsePostOutputLayer()
    {
    }

    template <typename TDevice>
    const std::string& WeightedSsePostOutputLayer<TDevice>::type() const
    {
        static const std::string s("weightedsse");
        return s;
    }

    template <typename TDevice>
    real_t WeightedSsePostOutputLayer<TDevice>::calculateError()
    {
        internal::ComputeWeightedSseFn fn;
        fn.layerSize = this->size() / 2;
        fn.patTypes  = helpers::getRawPointer(this->patTypes());
        fn.targets = helpers::getRawPointer(this->_targets());
        fn.outputs = helpers::getRawPointer(this->_actualOutputs());

        int n = this->curMaxSeqLength() * this->parallelSequences() * this->size() / 2;

        real_t mse = (real_t)0.5 * thrust::transform_reduce(
            thrust::counting_iterator<int>(0),
            thrust::counting_iterator<int>(0) + n,
            fn,
            (real_t)0,
            thrust::plus<real_t>()
            );

        return mse;
    }

    template <typename TDevice>
    void WeightedSsePostOutputLayer<TDevice>::computeForwardPass(const int nnState)
    {
    }

    template <typename TDevice>
    void WeightedSsePostOutputLayer<TDevice>::computeForwardPass(const int timeStep,
								 const int nnState)
    {
    }

    template <typename TDevice>
    void WeightedSsePostOutputLayer<TDevice>::computeBackwardPass(const int nnState)
    {
        // calculate the errors
        internal::ComputeOutputErrorFn fn;
        fn.layerSize = this->size() / 2;
        fn.patTypes  = helpers::getRawPointer(this->patTypes());
        fn.targets = helpers::getRawPointer(this->_targets());
        fn.outputs = helpers::getRawPointer(this->_actualOutputs());

        int n = this->curMaxSeqLength() * this->parallelSequences() * this->size() / 2;

        thrust::transform(
            thrust::counting_iterator<int>(0),
            thrust::counting_iterator<int>(0) + n,
            this->_outputErrors().begin(),
            fn
            );
    }

    template <typename TDevice>
    void WeightedSsePostOutputLayer<TDevice>::computeBackwardPass(const int timeStep,
								  const int nnState)
    {
	if (timeStep == this->curMaxSeqLength())
	    this->computeBackwardPass(nnState);
    }
    
    // explicit template instantiations
    template class WeightedSsePostOutputLayer<Cpu>;
    template class WeightedSsePostOutputLayer<Gpu>;

} // namespace layers
