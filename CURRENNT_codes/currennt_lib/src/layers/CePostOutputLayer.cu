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

#include "CePostOutputLayer.hpp"
#include "../helpers/NumericLimits.cuh"
#include "../helpers/max.cuh"
#include "../helpers/limitedError.cuh"
#include "../helpers/getRawPointer.cuh"

//#include <thrust/reduce.h>
//#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <iostream>


namespace internal {
namespace {

    struct ComputeCeFn
    {
        int layerSize;

        const char *patTypes;

        __host__ __device__ real_t operator() (const thrust::tuple<real_t, real_t, int> &values) const
        {
            // unpack the tuple
            real_t target = values.get<0>();
            real_t output = values.get<1>();
            int outputIdx = values.get<2>();

            // check if we have to skip this value
            int patIdx = outputIdx / layerSize;
            if (patTypes[patIdx] == PATTYPE_NONE)
                return 0;

            // calculate the error
            // this is actually KL div.
            // it does not have the additive term (entropy of targets)
            // so it always goes to zero if the distributions are identical
            // but the derivative is the same as CE ...
            real_t ftarget = helpers::max(helpers::NumericLimits<real_t>::min(), target);
            output = helpers::max(helpers::NumericLimits<real_t>::min(), output);
            real_t div = target * log(ftarget / output);
            return div;
        }
    };

    struct ComputeOutputErrorFn
    {
        int layerSize;

        const char *patTypes;

        __host__ __device__ real_t operator() (const thrust::tuple<const real_t&, const real_t&, int> &t) const
        {
            // unpack the tuple
            real_t actualOutput = t.get<0>();
            real_t targetOutput = t.get<1>();
            int    outputIdx    = t.get<2>();

            // calculate the pattern index
            int patIdx = outputIdx / layerSize;

            // check if the pattern is a dummy
            if (patTypes[patIdx] == PATTYPE_NONE)
                return 0;

            // calculate the error
            actualOutput = helpers::max(helpers::NumericLimits<real_t>::min(), actualOutput);
            real_t bp_error = helpers::boundRange(-targetOutput / actualOutput, -100, +100);

            return bp_error;
        }
    };
    
} // anonymous namespace
} // namespace anonymous


namespace layers {

    template <typename TDevice>
    CePostOutputLayer<TDevice>::CePostOutputLayer(const helpers::JsonValue &layerChild, 
						  Layer<TDevice> &precedingLayer,
						  int             maxSeqLength,
						  int             layerID)
        : PostOutputLayer<TDevice>(layerChild, precedingLayer, precedingLayer.size(),
				   maxSeqLength, layerID)
    {
    }

    template <typename TDevice>
    CePostOutputLayer<TDevice>::~CePostOutputLayer()
    {
    }

    template <typename TDevice>
    const std::string& CePostOutputLayer<TDevice>::type() const
    {
        static const std::string s("ce");
        return s;
    }

    template <typename TDevice>
    real_t CePostOutputLayer<TDevice>::calculateError()
    {
        internal::ComputeCeFn fn;
        fn.layerSize = this->size();
        fn.patTypes  = helpers::getRawPointer(this->patTypes());

        int n = this->curMaxSeqLength() * this->parallelSequences() * this->size();

        real_t ce = thrust::transform_reduce(
            thrust::make_zip_iterator(thrust::make_tuple(this->_targets().begin(),   
							 this->_actualOutputs().begin(),   
							 thrust::counting_iterator<int>(0))),
            thrust::make_zip_iterator(thrust::make_tuple(this->_targets().begin()+n, 
							 this->_actualOutputs().begin()+n, 
							 thrust::counting_iterator<int>(0)+n)),
            fn,
            (real_t)0,
            thrust::plus<real_t>()
            );

        return ce;
    }

    template <typename TDevice>
    void CePostOutputLayer<TDevice>::computeForwardPass(const int nnState)
    {
    }
    template <typename TDevice>
    void CePostOutputLayer<TDevice>::computeForwardPass(const int timeStep, const int nnState)
    {
    }
    
    template <typename TDevice>
    void CePostOutputLayer<TDevice>::computeBackwardPass(const int nnState)
    {
        // calculate the errors
        internal::ComputeOutputErrorFn fn;
        fn.layerSize = this->size();
        fn.patTypes  = helpers::getRawPointer(this->patTypes());

        int n = this->curMaxSeqLength() * this->parallelSequences() * this->size();

        thrust::transform(
            thrust::make_zip_iterator(thrust::make_tuple(this->_actualOutputs().begin(),   
							 this->_targets().begin(),   
							 thrust::counting_iterator<int>(0))),
            thrust::make_zip_iterator(thrust::make_tuple(this->_actualOutputs().begin()+n, 
							 this->_targets().begin()+n, 
							 thrust::counting_iterator<int>(0)+n)),
            this->_outputErrors().begin(),
            fn
            );
    }


    // explicit template instantiations
    template class CePostOutputLayer<Cpu>;
    template class CePostOutputLayer<Gpu>;

} // namespace layers
