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

#include "MaskLayer.hpp"
#include "TrainableLayer.hpp"
#include "../helpers/getRawPointer.cuh"
#include "../helpers/safeExp.cuh"
#include "../helpers/JsonClasses.hpp"

#include <thrust/transform.h>
#include <thrust/for_each.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>

#include <typeinfo>


namespace internal {
namespace {

    struct ComputeMaskFn
    {
        real_t *outputs;
        __host__ __device__ real_t operator() (int outputIdx) const
        {
            return helpers::safeExp(outputs[outputIdx * 2]) / (helpers::safeExp(outputs[outputIdx * 2]) + helpers::safeExp(outputs[outputIdx * 2 + 1]));
        }
    };

    struct ComputeMaskErrorFn
    {
        real_t *pl_outputs;
        real_t *pl_output_errors;
        __host__ __device__ void operator() (const thrust::tuple<const real_t&, const int&> &t) const
        {
            real_t err = t.get<0>();
            int freqIdx = t.get<1>();
            real_t tmp = helpers::safeExp(pl_outputs[freqIdx * 2]) + helpers::safeExp(pl_outputs[freqIdx * 2 + 1]);
            real_t tmp2 = helpers::safeExp(pl_outputs[freqIdx * 2] + pl_outputs[freqIdx * 2 + 1]) / (tmp * tmp);
            pl_output_errors[freqIdx * 2] = err * tmp2;
            pl_output_errors[freqIdx * 2 + 1] = -err * tmp2;
        }
    };

} // anonymous namespace
} // namespace internal


namespace layers {

    template <typename TDevice>
    MaskLayer<TDevice>::MaskLayer(
        const helpers::JsonValue &layerChild, 
        Layer<TDevice> &precedingLayer)
        : Layer<TDevice>(layerChild, precedingLayer.parallelSequences(), precedingLayer.maxSeqLength()),
        m_precedingLayer(precedingLayer)
    {
        //std::cout << "mask layer" << std::endl;
    }

    template <typename TDevice>
    MaskLayer<TDevice>::~MaskLayer()
    {
    }

    template <typename TDevice>
    const std::string& MaskLayer<TDevice>::type() const
    {
        static std::string s("mask");
        return s;
    }


    template <typename TDevice>
    void MaskLayer<TDevice>::computeForwardPass()
    {
        internal::ComputeMaskFn fn;
        fn.outputs = helpers::getRawPointer(m_precedingLayer.outputs());
        int n = this->curMaxSeqLength() * this->parallelSequences() * this->size();
        thrust::transform(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(0) + n, this->_outputs().begin(), fn);
    }

    template <typename TDevice>
    void MaskLayer<TDevice>::computeBackwardPass()
    {
        // TODO respect dummy flag to avoid zero multiplications
        int n = this->curMaxSeqLength() * this->parallelSequences() * this->size();
        internal::ComputeMaskErrorFn fn;
        /*std::cout << "my outputs " << std::endl;
        thrust::copy(this->outputs().begin(), this->outputs().end(), std::ostream_iterator<real_t>(std::cout, ";"));
        std::cout << std::endl;
        std::cout << "my errors " << std::endl;
        thrust::copy(this->outputErrors().begin(), this->outputErrors().end(), std::ostream_iterator<real_t>(std::cout, ";"));
        std::cout << std::endl;
        std::cout << "your outputs " << std::endl;
        thrust::copy(m_precedingLayer.outputs().begin(), m_precedingLayer.outputs().end(), std::ostream_iterator<real_t>(std::cout, ";"));
        std::cout << std::endl;*/
        fn.pl_outputs = helpers::getRawPointer(m_precedingLayer.outputs());
        fn.pl_output_errors = helpers::getRawPointer(m_precedingLayer.outputErrors());
        thrust::for_each(
            thrust::make_zip_iterator(thrust::make_tuple(this->outputErrors().begin(), thrust::counting_iterator<int>(0))),
            thrust::make_zip_iterator(thrust::make_tuple(this->outputErrors().begin() + n, thrust::counting_iterator<int>(0) + n)),
            fn
        );
        /*
        std::cout << "your errors " << std::endl;
        thrust::copy(m_precedingLayer.outputErrors().begin(), m_precedingLayer.outputErrors().end(), std::ostream_iterator<real_t>(std::cout, ";"));
        std::cout << std::endl;
        */
    }

    // explicit template instantiations
    template class MaskLayer<Cpu>;
    template class MaskLayer<Gpu>;

} // namespace layers
