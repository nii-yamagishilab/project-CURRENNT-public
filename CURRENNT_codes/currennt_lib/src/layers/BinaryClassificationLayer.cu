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

#include "BinaryClassificationLayer.hpp"
#include "../helpers/NumericLimits.cuh"
#include "../helpers/max.cuh"
#include "../helpers/getRawPointer.cuh"

#include <stdexcept>

//#include <thrust/reduce.h>
//#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/for_each.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>


namespace internal {
namespace {

    struct ComputeCrossEntropyErrorFn
    {
        const char *patTypes;

        __host__ __device__ real_t operator() (const thrust::tuple<real_t, real_t, int> &t) const
        {
            // unpack the tuple
            int outputIdx = t.get<2>();

            // check if we actually need to continue
            if (patTypes[outputIdx] == PATTYPE_NONE)
                return 0;

            // calculate the cross entropy error
            real_t target = t.get<0>();
            real_t output = t.get<1>();

            real_t act        = helpers::max(output, helpers::NumericLimits<real_t>::min());
            real_t targetProb = (target > 0 ? act : 1-act);
            real_t error      = -log(targetProb);

            return error;
        }
    };

    struct CountCorrectClassificationsFn
    {
        __host__ __device__ int operator() (const thrust::tuple<real_t, real_t, int> &t) const
        {
            // unpack the tuple
            real_t target  = t.get<0>();
            real_t output  = t.get<1>();
            int    patType = t.get<2>();

            // determine target and estimated class
            bool tgtClass = (target > (real_t)0.5);
            bool estClass = (output > (real_t)0.5);

            // count correct classification
            return (patType != PATTYPE_NONE) && (tgtClass == estClass);
        }
    };

    struct ComputeOutputErrorFn
    {
        const char *patTypes;

        __host__ __device__ void operator() (const thrust::tuple<real_t&, const real_t&, const real_t&, int> &t) const
        {
            // unpack the tuple
            int outputIdx = t.get<3>();

            // check if we actually need to continue
            if (patTypes[outputIdx] == PATTYPE_NONE)
                return;

            // calculate the error
            real_t target = t.get<1>();
            real_t output = t.get<2>();

            real_t act        = helpers::max(output, helpers::NumericLimits<real_t>::min());
            real_t targetProb = (target > 0 ? act : 1-act);
            real_t error      = (target > 0 ? -(1/targetProb) : (1/targetProb));

            // store the error
            t.get<0>() = error;
        }
    };

} // anonymous namespace
} // namespace anonymous


namespace layers {

    template <typename TDevice>
    BinaryClassificationLayer<TDevice>::BinaryClassificationLayer(
	  const helpers::JsonValue &layerChild,
	  Layer<TDevice> &precedingLayer,
	  int maxSeqLength,
	  int layerID)
        : PostOutputLayer<TDevice>(layerChild, precedingLayer,
				   precedingLayer.size(), maxSeqLength, layerID)
    {
        if (this->size() != 1)
            throw std::runtime_error("The binary classification post output layer cannot be used for an output layer size != 1");
    }

    template <typename TDevice>
    BinaryClassificationLayer<TDevice>::~BinaryClassificationLayer()
    {
    }

    template <typename TDevice>
    int BinaryClassificationLayer<TDevice>::countCorrectClassifications()
    {
        internal::CountCorrectClassificationsFn fn;

        int n = this->curMaxSeqLength() * this->parallelSequences();

        int correctClassifications = thrust::transform_reduce(
            thrust::make_zip_iterator(thrust::make_tuple(this->_targets().begin(),   this->_actualOutputs().begin(),   this->patTypes().begin())),
            thrust::make_zip_iterator(thrust::make_tuple(this->_targets().begin()+n, this->_actualOutputs().begin()+n, this->patTypes().begin()+n)),
            fn,
            0,
            thrust::plus<int>()
            );

        return correctClassifications;
    }
    
    template <typename TDevice>
    const std::string& BinaryClassificationLayer<TDevice>::type() const
    {
        static std::string s("binary_classification");
        return s;
    }

    template <typename TDevice>
    void BinaryClassificationLayer<TDevice>::loadSequences(const data_sets::DataSetFraction &fraction, const int nnState)
    {
	
        PostOutputLayer<TDevice>::loadSequences(fraction, nnState);
        // In this case, we can copy the integer vector of target classes,
	// since they are equal to the real target values (0/1)
        thrust::copy(fraction.targetClasses().begin(),
		     fraction.targetClasses().end(),
		     this->_targets().begin());
    }

    template <typename TDevice>
    real_t BinaryClassificationLayer<TDevice>::calculateError()
    {
        internal::ComputeCrossEntropyErrorFn fn;
        fn.patTypes = helpers::getRawPointer(this->patTypes());

        int n = this->curMaxSeqLength() * this->parallelSequences();

        real_t error = thrust::transform_reduce(
            thrust::make_zip_iterator(thrust::make_tuple(this->_targets().begin(),   this->_actualOutputs().begin(),   thrust::counting_iterator<int>(0))),
            thrust::make_zip_iterator(thrust::make_tuple(this->_targets().begin()+n, this->_actualOutputs().begin()+n, thrust::counting_iterator<int>(0)+n)),
            fn,
            (real_t)0,
            thrust::plus<real_t>()
            );

        return error;
    }

    template <typename TDevice>
    void BinaryClassificationLayer<TDevice>::computeForwardPass(const int nnState)
    {
    }

    template <typename TDevice>
    void BinaryClassificationLayer<TDevice>::computeForwardPass(const int timeStep,
								const int nnState)
    {
    }

    template <typename TDevice>
    void BinaryClassificationLayer<TDevice>::computeBackwardPass(const int nnState)
    {
        internal::ComputeOutputErrorFn fn;
        fn.patTypes = helpers::getRawPointer(this->patTypes());

        int n = this->curMaxSeqLength() * this->parallelSequences();

        thrust::for_each(
            thrust::make_zip_iterator(thrust::make_tuple(this->_outputErrors().begin(),   this->_targets().begin(),   this->_actualOutputs().begin(),   thrust::counting_iterator<int>(0))),
            thrust::make_zip_iterator(thrust::make_tuple(this->_outputErrors().begin()+n, this->_targets().begin()+n, this->_actualOutputs().begin()+n, thrust::counting_iterator<int>(0)+n)),
            fn
            );

    }


    // explicit template instantiations
    template class BinaryClassificationLayer<Cpu>;
    template class BinaryClassificationLayer<Gpu>;

} // namespace layers
