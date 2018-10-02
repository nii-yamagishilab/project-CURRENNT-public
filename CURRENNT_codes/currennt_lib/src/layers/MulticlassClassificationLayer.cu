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

#include "MulticlassClassificationLayer.hpp"
#include "../helpers/NumericLimits.cuh"
#include "../helpers/max.cuh"
#include "../helpers/getRawPointer.cuh"

#include <stdexcept>
#include <cassert>

#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/for_each.h>
#include <thrust/fill.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>

#define SKIP_MARKER helpers::NumericLimits<real_t>::max()


namespace internal {
namespace {

    struct ComputeCrossEntropyErrorFn
    {
        int layerSize;

        const real_t *outputs;

        __host__ __device__ real_t operator() (const thrust::tuple<int, int> &t) const
        {
            // unpack the tuple
            int targetClass = t.get<0>();
            int patIdx      = t.get<1>();

            // calculate the CEE
            if (targetClass == -1)
                return 0;
            else {
                int outputIdx     = outputIdx = patIdx * layerSize + targetClass;
                real_t targetProb = helpers::max(helpers::NumericLimits<real_t>::min(), outputs[outputIdx]);
                return log(targetProb);
            }
        }
    };

    struct CountCorrectClassificationsFn
    {
        int layerSize;

        const real_t *outputs;

        __host__ __device__ int operator() (const thrust::tuple<int, int> &t) const
        {
            // unpack the tuple
            int targetClass = t.get<0>();
            int patIdx      = t.get<1>();

            // check for dummy
            if (targetClass == -1)
                return 0;

            // determine the estimated target class
            const real_t *offOutputs = outputs + patIdx * layerSize;
            real_t maxProb = 0;
            int estClass   = 0;

            for (int i = 0; i < layerSize; ++i) {
                real_t out = offOutputs[i];
                if (out > maxProb) {
                    maxProb  = out;
                    estClass = i;
                }
            }

            // check if the we correctly classified the timestep
            if (targetClass == estClass)
                return 1;
            else
                return 0;
        }
    };

    struct ComputeOutputErrorFn
    {
        int layerSize;

        const real_t *outputs;
        real_t       *outputErrors;

        __host__ __device__ void operator() (const thrust::tuple<int, int> &t) const
        {
            // unpack the tuple
            int targetClass = t.get<0>();
            int patIdx      = t.get<1>();

            // check if we need to continue
            if (targetClass == -1)
                return;

            // calculate indices
            int outputIdx = patIdx * layerSize + targetClass;

            // calculate the error
            real_t targetProb = helpers::max(helpers::NumericLimits<real_t>::min(), outputs[outputIdx]);
            real_t error = - (1/targetProb);

            // store the error
            outputErrors[outputIdx] = error;
        }
    };

} // anonymous namespace
} // namespace anonymous


namespace layers {

    template <typename TDevice>
    MulticlassClassificationLayer<TDevice>::MulticlassClassificationLayer(
		const helpers::JsonValue &layerChild,
		Layer<TDevice> &precedingLayer,
		int maxSeqLength,
		int layerID)
        : PostOutputLayer<TDevice>(layerChild, precedingLayer, precedingLayer.size(),
				   maxSeqLength, layerID, false)
    {
        if (this->size() == 1)
            throw std::runtime_error("The multiclass classification post output layer cannot be used for an output layer size of 1");

        // resize the pattern target classes vector
        m_patTargetClasses.resize(this->patTypes().size());
    }

    template <typename TDevice>
    MulticlassClassificationLayer<TDevice>::~MulticlassClassificationLayer()
    {
    }

    template <typename TDevice>
    int MulticlassClassificationLayer<TDevice>::countCorrectClassifications()
    {
        internal::CountCorrectClassificationsFn fn;
        fn.layerSize = this->size();
        fn.outputs   = helpers::getRawPointer(this->_actualOutputs());

        int n = this->curMaxSeqLength() * this->parallelSequences();

        int correctClassifications = thrust::transform_reduce(
            thrust::make_zip_iterator(thrust::make_tuple(m_patTargetClasses.begin(),   thrust::counting_iterator<int>(0))),
            thrust::make_zip_iterator(thrust::make_tuple(m_patTargetClasses.begin()+n, thrust::counting_iterator<int>(0)+n)),
            fn,
            0,
            thrust::plus<int>()
            );

        return correctClassifications;
    }
    
    template <typename TDevice>
    const std::string& MulticlassClassificationLayer<TDevice>::type() const
    {
        static std::string s("multiclass_classification");
        return s;
    }

    template <typename TDevice>
    void MulticlassClassificationLayer<TDevice>::loadSequences(const data_sets::DataSetFraction &fraction, const int nnState)
    {
        PostOutputLayer<TDevice>::loadSequences(fraction, nnState);

        thrust::copy(fraction.targetClasses().begin(), fraction.targetClasses().end(), m_patTargetClasses.begin());
    }

    template <typename TDevice>
    real_t MulticlassClassificationLayer<TDevice>::calculateError()
    {
        // calculate the cross entropy error
        internal::ComputeCrossEntropyErrorFn fn;
        fn.layerSize = this->size();
        fn.outputs   = helpers::getRawPointer(this->_actualOutputs());

        int n = this->curMaxSeqLength() * this->parallelSequences();

        real_t error = thrust::transform_reduce(
            thrust::make_zip_iterator(thrust::make_tuple(m_patTargetClasses.begin(),   thrust::counting_iterator<int>(0))),
            thrust::make_zip_iterator(thrust::make_tuple(m_patTargetClasses.begin()+n, thrust::counting_iterator<int>(0)+n)),
            fn,
            (real_t)0,
            thrust::plus<real_t>()
            );

        return -error;
    }

    template <typename TDevice>
    void MulticlassClassificationLayer<TDevice>::computeForwardPass(const int nnState)
    {
    }

    template <typename TDevice>
    void MulticlassClassificationLayer<TDevice>::computeForwardPass(const int timeStep,
								    const int nnState)
    {
    }

    template <typename TDevice>
    void MulticlassClassificationLayer<TDevice>::computeBackwardPass(const int nnState)
    {
        int n = this->curMaxSeqLength() * this->parallelSequences();

        // set all errors to zero
        assert (n * this->size() <= this->_outputErrors().size());
        thrust::fill_n(this->_outputErrors().begin(), n * this->size(), (real_t)0);

        // calculate the errors
        internal::ComputeOutputErrorFn fn;
        fn.layerSize    = this->size();
        fn.outputs      = helpers::getRawPointer(this->_actualOutputs());
        fn.outputErrors = helpers::getRawPointer(this->_outputErrors());

        thrust::for_each(
            thrust::make_zip_iterator(thrust::make_tuple(m_patTargetClasses.begin(),   thrust::counting_iterator<int>(0))),
            thrust::make_zip_iterator(thrust::make_tuple(m_patTargetClasses.begin()+n, thrust::counting_iterator<int>(0)+n)),
            fn
            );
    }


    // explicit template instantiations
    template class MulticlassClassificationLayer<Cpu>;
    template class MulticlassClassificationLayer<Gpu>;

} // namespace layers
