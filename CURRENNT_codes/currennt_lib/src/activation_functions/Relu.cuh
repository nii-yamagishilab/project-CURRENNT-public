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

#ifndef ACTIVATION_FUNCTIONS_RELU_CUH
#define ACTIVATION_FUNCTIONS_RELU_CUH

#include "../Types.hpp"

#define ACTIVIATION_FUNCTION_RELU_LeakyFactor 0.01

namespace activation_functions {

    struct Relu
    {
        static __host__ __device__ real_t fn(real_t x)
        {
            return (x+fabsf(x))*0.5;
	    //return (x>0)?(x):(exp(x)-1);
        }

        static __host__ __device__ real_t deriv(real_t y)
        {
	    return (y>0)?(1.0):(0.0);
            //return (y>0)?(1.0):(y+1.0);
        }
    };

} // activation_functions 


#endif // ACTIVATION_FUNCTIONS_RELU_CUH
