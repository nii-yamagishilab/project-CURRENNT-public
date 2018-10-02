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

#ifndef HELPERS_CUBLAS_HPP
#define HELPERS_CUBLAS_HPP


namespace helpers {
namespace cublas {

    template <typename T>
    void multiplyMatrices(
        bool transposeA, bool transposeB,
        int m, int n, int k,
        const T *matrixA, int ldA,
        const T *matrixB, int ldB,
        T *matrixC, int ldC,
        bool addOldMatrixC = false
        );

} // namespace cublas
} // namespace helpers

#endif // HELPERS_CUBLAS_HPP
