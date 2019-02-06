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

#include "RmsePostOutputLayer.hpp"
#include "../helpers/getRawPointer.cuh"
#include "../helpers/TypedMath.cuh"

#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>


namespace internal {
namespace {

    struct ComputeRmseFn
    {
        int layerSize;

        const real_t *actualOutputs;
        const real_t *targetOutputs;

        const char *patTypes;

        __host__ __device__ real_t operator() (const int &patIdx) const
        {
            // check if the pattern belongs to a sequence
            if (patTypes[patIdx] == PATTYPE_NONE)
                return 0;

            // offset arrays to the beginning of the pattern
            const real_t *offActualOutputs = &actualOutputs[patIdx * layerSize];
            const real_t *offTargetOutputs = &targetOutputs[patIdx * layerSize];

            // sum up the squared errors
            real_t sum = 0;
            for (int i = 0; i < layerSize; ++i) {
                real_t diff = offActualOutputs[i] - offTargetOutputs[i];
                sum += diff * diff;
            }

            // calculate the rmse
            real_t rmse = helpers::TypedMath<real_t>::sqrt(sum / layerSize);

            return rmse;
        }
    };

    struct ComputeOutputErrorFn
    {
        int layerSize;

        const real_t *rmses;

        __host__ __device__ real_t operator() (const thrust::tuple<const real_t&, const real_t&, int> &t) const
        {
            // unpack the tuple
            real_t actualOutput = t.get<0>();
            real_t targetOutput = t.get<1>();
            int    outputIdx    = t.get<2>();

            // calculate the pattern index
            int patIdx = outputIdx / layerSize;

            // get the RMSE for the current pattern
            real_t rmse = rmses[patIdx];

            // calculate the error
            real_t error = rmse * (actualOutput - targetOutput);

            return error;
        }
    };
    
} // anonymous namespace
} // namespace anonymous


namespace layers {

    template <typename TDevice>
    RmsePostOutputLayer<TDevice>::RmsePostOutputLayer(const helpers::JsonValue &layerChild,
						      Layer<TDevice> &precedingLayer,
						      int maxSeqLength,
						      int layerID)
        : PostOutputLayer<TDevice>(layerChild, precedingLayer, precedingLayer.size(),
				   maxSeqLength, layerID)
    {
        // resize the vector for RMSEs
        m_rmses.resize(this->patTypes().size());
    }

    template <typename TDevice>
    RmsePostOutputLayer<TDevice>::~RmsePostOutputLayer()
    {
    }

    template <typename TDevice>
    const std::string& RmsePostOutputLayer<TDevice>::type() const
    {
        static const std::string s("rmse");
        return s;
    }

    template <typename TDevice>
    real_t RmsePostOutputLayer<TDevice>::calculateError()
    {
        real_t rmse = thrust::reduce(
            m_rmses.begin(),
            m_rmses.begin() + this->curMaxSeqLength() * this->parallelSequences()
            );

        return rmse;
    }

    template <typename TDevice>
    void RmsePostOutputLayer<TDevice>::computeForwardPass(const int nnState)
    {
        // calculate the RMSE for each pattern
        internal::ComputeRmseFn fn;
        fn.layerSize     = this->size();
        fn.actualOutputs = helpers::getRawPointer(this->_actualOutputs());
        fn.targetOutputs = helpers::getRawPointer(this->_targets());
        fn.patTypes      = helpers::getRawPointer(this->patTypes());

        thrust::transform(
                thrust::counting_iterator<int>(0),
                thrust::counting_iterator<int>(0) + this->curMaxSeqLength() * this->parallelSequences(),
                m_rmses.begin(),
                fn
                );
    }

    template <typename TDevice>
    void RmsePostOutputLayer<TDevice>::computeForwardPass(const int timeStep, const int nnState)
    {
    }

    template <typename TDevice>
    void RmsePostOutputLayer<TDevice>::computeBackwardPass(const int nnState)
    {
        // calculate the errors
        internal::ComputeOutputErrorFn fn;
        fn.layerSize = this->size();
        fn.rmses     = helpers::getRawPointer(m_rmses);

        int n = this->curMaxSeqLength() * this->parallelSequences() * this->size();

        thrust::transform(
            thrust::make_zip_iterator(
		thrust::make_tuple(this->_actualOutputs().begin(),
				   this->_targets().begin(),
				   thrust::counting_iterator<int>(0))),
            thrust::make_zip_iterator(
		thrust::make_tuple(this->_actualOutputs().begin()+n,
				   this->_targets().begin()+n,
				   thrust::counting_iterator<int>(0)+n)),
            this->_outputErrors().begin(),
            fn);
    }

    template <typename TDevice>
    void RmsePostOutputLayer<TDevice>::computeBackwardPass(const int timeStep, const int nnState)
    {
	if (timeStep == this->curMaxSeqLength())
	    this->computeBackwardPass(nnState);
    }

    // explicit template instantiations
    template class RmsePostOutputLayer<Cpu>;
    template class RmsePostOutputLayer<Gpu>;

} // namespace layers
