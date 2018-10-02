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

#ifndef HELPERS_GETRAWPOINTER_CUH
#define HELPERS_GETRAWPOINTER_CUH

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>


namespace helpers {

    template <typename T>
    static inline T* getRawPointer(thrust::host_vector<T> &v)
    {
        return v.data();
    }

    template <typename T>
    static inline const T* getRawPointer(const thrust::host_vector<T> &v)
    {
        return v.data();
    }

    template <typename T>
    static inline T* getRawPointer(thrust::device_vector<T> &v)
    {
        return v.data().get();
    }

    template <typename T>
    static inline const T* getRawPointer(const thrust::device_vector<T> &v)
    {
        return v.data().get();
    }

} // namespace helpers


#endif // HELPERS_GETRAWPOINTER_CUH
