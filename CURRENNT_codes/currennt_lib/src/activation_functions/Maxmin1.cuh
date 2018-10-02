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

#ifndef ACTIVATION_FUNCTIONS_MAXMIN1_CUH
#define ACTIVATION_FUNCTIONS_MAXMIN1_CUH

#include "Logistic.cuh"


namespace activation_functions {

    struct Maxmin1
    {
        static __host__ __device__ real_t fn(real_t x)
        {
            return ((real_t)2.0 * Logistic::fn(x) - (real_t)1.0);
        }

        static __host__ __device__ real_t deriv(real_t y)
        {
            if (y == -1 || y == 1)
                return 0;

            return ((real_t)1.0 - (y * y)) / (real_t)2.0;
        }
    };

} // activation_functions 


#endif // ACTIVATION_FUNCTIONS_MAXMIN1_CUH
