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
 *****************************************************************************/

#include "./dataProcess.hpp"

#include <thrust/transform_reduce.h>

namespace internal{
namespace {

    // get absolute_value
    template<typename T>
    struct absolute_value
    {
	__host__ __device__ T operator()(const T &x) const
	{
	    return x < T(0) ? -x : x;
	}
    };

    // whether waveform value is too small 
    template<typename T>
    struct small_waveform_value
    {
	__host__ __device__ T operator()(const T &x) const
	{
	    return ((x > -0.0030 && x < 0.0030)?1:0); // 100 / 32768
	}
    };
    
}
}


bool checkWaveformValidity(Cpu::real_vector data, const int length)
{
	real_t smallWaveForm = thrust::transform_reduce(data.begin(),
							data.begin() + length,
							internal::small_waveform_value<real_t>(),
							0,
							thrust::plus<real_t>());
	if (smallWaveForm / length > 0.9)
	    return false;
	else
	    return true;
}

bool checkWaveformValidity(Gpu::real_vector data, const int length)
{

	real_t smallWaveForm = thrust::transform_reduce(data.begin(),
							data.begin() + length,
							internal::small_waveform_value<real_t>(),
							0,
							thrust::plus<real_t>());
	if (smallWaveForm / length > 0.9)
	    return false;
	else
	    return true;
}

