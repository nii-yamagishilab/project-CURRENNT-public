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

#ifndef HELPERS_PRINTVECTOR_CUH
#define HELPERS_PRINTVECTOR_CUH

#include "../Types.hpp"

#include <string>
#include <cstdio>


namespace helpers {

    template <typename TVector>
    static void printVector(const TVector &v, const std::string &desc, int chunks=1)
    {
        printVector<thrust::host_vector<typename TVector::value_type> >(v, desc, chunks);
    }

    template <>
    static void printVector<thrust::host_vector<int> >(const thrust::host_vector<int> &v, const std::string &desc, int chunks)
    {
        printf("\n");
        printf("%s (%d values):\n", desc.c_str(), (int)v.size());
        for (size_t i = 0; i < v.size(); ++i) {
            printf("    [%3d] = %5d\n", (int)i, (int)v[i]);
            if ((i + 1) % (v.size() / chunks) == 0)
                printf("\n");
        }
    }

    template <>
    static void printVector<thrust::host_vector<real_t> >(const thrust::host_vector<real_t> &v, const std::string &desc, int chunks)
    {
        printf("\n");
        printf("%s (%d values):\n", desc.c_str(), (int)v.size());
        for (size_t i = 0; i < v.size(); ++i) {
            printf("    [%3d] = %+.10lf    %+.4lf\n", (int)i, (double)v[i], (double)v[i]);
            if ((i + 1) % (v.size() / chunks) == 0)
                printf("\n");
        }
    }

} // namespace helpers


#endif // HELPERS_PRINTVECTOR_CUH
