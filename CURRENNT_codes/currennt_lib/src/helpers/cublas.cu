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

#include "cublas.hpp"

#include <cublas_v2.h>

#include <stdexcept>


namespace internal {
namespace {

    cublasHandle_t getCublasHandle()
    {
        static cublasHandle_t handle = 0;
        if (!handle) {
            cublasStatus_t res = cublasCreate(&handle);
            if (res != CUBLAS_STATUS_SUCCESS)
                throw std::runtime_error("Could not create CUBLAS handle");
        }

        return handle;
    }

} // anonymous namespace
} // namespace internal


namespace helpers {
namespace cublas {

    template <>
    void multiplyMatrices<float>(
        bool transposeA, bool transposeB,
        int m, int n, int k,
        const float *matrixA, int ldA,
        const float *matrixB, int ldB,
        float *matrixC, int ldC,
        bool addOldMatrixC
        )
    {
        float alpha = 1;
        float beta  = (addOldMatrixC ? 1.0f : 0.0f);

        cublasStatus_t res = cublasSgemm(
            /* handle */ internal::getCublasHandle(),
            /* transa */ transposeA ? CUBLAS_OP_T : CUBLAS_OP_N,
            /* transb */ transposeB ? CUBLAS_OP_T : CUBLAS_OP_N,
            /* m      */ m,
            /* n      */ n,
            /* k      */ k,
            /* alpha  */ &alpha,
            /* A      */ matrixA,
            /* lda    */ ldA,
            /* B      */ matrixB,
            /* ldb    */ ldB,
            /* beta   */ &beta,
            /* C      */ matrixC,
            /* ldc    */ ldC
            );

        if (res != CUBLAS_STATUS_SUCCESS)
            throw std::runtime_error("CUBLAS matrix multiplication failed");
    }

    template <>
    void multiplyMatrices<double>(
        bool transposeA, bool transposeB,
        int m, int n, int k,
        const double *matrixA, int ldA,
        const double *matrixB, int ldB,
        double *matrixC, int ldC,
        bool addOldMatrixC
        )
    {
        double alpha = 1;
        double beta  = (addOldMatrixC ? 1 : 0);

        cublasStatus_t res = cublasDgemm(
            /* handle */ internal::getCublasHandle(),
            /* transa */ transposeA ? CUBLAS_OP_T : CUBLAS_OP_N,
            /* transb */ transposeB ? CUBLAS_OP_T : CUBLAS_OP_N,
            /* m      */ m,
            /* n      */ n,
            /* k      */ k,
            /* alpha  */ &alpha,
            /* A      */ matrixA,
            /* lda    */ ldA,
            /* B      */ matrixB,
            /* ldb    */ ldB,
            /* beta   */ &beta,
            /* C      */ matrixC,
            /* ldc    */ ldC
            );

        if (res != CUBLAS_STATUS_SUCCESS)
            throw std::runtime_error("CUBLAS matrix multiplication failed");
    }

} // namespace cublas
} // namespace helpers
