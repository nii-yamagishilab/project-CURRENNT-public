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

#include "Matrix.hpp"
#include "getRawPointer.cuh"
#include "cublas.hpp"

#include <stdexcept>

#include <thrust/transform.h>

#define USE_CUBLAS 1


namespace internal {
namespace {

    struct MatrixMultiplyFn
    {
        int rowsA;
        int rowsB;
        int colsA;
        int colsB;

        const real_t *a;
        const real_t *b;

        __host__ __device__ real_t operator() (const int &idx) const
        {
            const real_t *offRowA = a + (idx % rowsA);
            const real_t *offColB = b + (idx / rowsA) * rowsB;

            real_t x = 0;
            for (int i = 0; i < colsA; ++i)
                x += offRowA[i * rowsA] * offColB[i];

            return x;
        }
    };

    struct AddMatrixMultiplyFn
    {
        int rowsA;
        int rowsB;
        int colsA;
        int colsB;

        const real_t *a;
        const real_t *b;

        __host__ __device__ real_t operator() (const int &idx, const real_t &oldX) const
        {
            const real_t *offRowA = a + (idx % rowsA);
            const real_t *offColB = b + (idx / rowsA) * rowsB;

            real_t x = 0;
            for (int i = 0; i < colsA; ++i)
                x += offRowA[i * rowsA] * offColB[i];

            return oldX + x;
        }
    };

    struct MatrixMultiplyTransposedAFn
    {
        int rowsA;
        int rowsB;
        int colsA;
        int colsB;

        const real_t *a;
        const real_t *b;

        __host__ __device__ real_t operator() (const int &idx) const
        {
            const real_t *offColA = a + (idx % colsA) * rowsA;
            const real_t *offColB = b + (idx / colsA) * rowsB;

            real_t x = 0;
            for (int i = 0; i < rowsA; ++i)
                x += offColA[i] * offColB[i];

            return x;
        }
    };

    struct AddMatrixMultiplyTransposedAFn
    {
        int rowsA;
        int rowsB;
        int colsA;
        int colsB;

        const real_t *a;
        const real_t *b;

        __host__ __device__ real_t operator() (const int &idx, const real_t &oldX) const
        {
            const real_t *offColA = a + (idx % colsA) * rowsA;
            const real_t *offColB = b + (idx / colsA) * rowsB;

            real_t x = 0;
            for (int i = 0; i < rowsA; ++i)
                x += offColA[i] * offColB[i];

            return oldX + x;
        }
    };

    struct MatrixMultiplyTransposedBFn
    {
        int rowsA;
        int rowsB;
        int colsA;
        int colsB;

        const real_t *a;
        const real_t *b;

        __host__ __device__ real_t operator() (const int &idx) const
        {
            const real_t *offRowA = a + (idx % rowsA);
            const real_t *offRowB = b + (idx / rowsA);

            real_t x = 0;
            for (int i = 0; i < colsA; ++i) {
                x += *offRowA * *offRowB;
                offRowA += rowsA;
                offRowB += rowsB;
            }

            return x;
        }
    };

    struct AddMatrixMultiplyTransposedBFn
    {
        int rowsA;
        int rowsB;
        int colsA;
        int colsB;

        const real_t *a;
        const real_t *b;

        __host__ __device__ real_t operator() (const int &idx, const real_t &oldX) const
        {
            const real_t *offRowA = a + (idx % rowsA);
            const real_t *offRowB = b + (idx / rowsA);

            real_t x = 0;
            for (int i = 0; i < colsA; ++i) {
                x += *offRowA * *offRowB;
                offRowA += rowsA;
                offRowB += rowsB;
            }

            return oldX + x;
        }
    };

} // anonymous namespace
} // namespace internal


namespace helpers {

    template <typename TDevice>
    Matrix<TDevice>::Matrix()
        : m_dataVector      (NULL)
        , m_dataVectorOffset(0)
        , m_data            (NULL)
        , m_rows            (0)
        , m_cols            (0)
    {
    }

    template <typename TDevice>
    Matrix<TDevice>::Matrix(real_vector *data, int rows, int cols, int dataOffset)
        : m_dataVector      (data)
        , m_dataVectorOffset(dataOffset)
        , m_data            (getRawPointer(*data) + dataOffset)
        , m_rows            (rows)
        , m_cols            (cols)
    {
        if (rows * cols > data->size() - dataOffset)
            throw std::runtime_error("Matrix exceeds available space in vector");
    }

    template <typename TDevice>
    Matrix<TDevice>::~Matrix()
    {
    }

    template <typename TDevice>
    void Matrix<TDevice>::assignProduct(const Matrix<TDevice> &a, bool transposeA, const Matrix<TDevice> &b, bool transposeB)
    {
        if (transposeA && !transposeB) {
            if (m_rows != a.m_cols || m_cols != b.m_cols || a.m_rows != b.m_rows)
                throw std::runtime_error("Invalid matrix dimensions");

            internal::MatrixMultiplyTransposedAFn fn;
            fn.rowsA = a.m_rows;
            fn.rowsB = b.m_rows;
            fn.colsA = a.m_cols;
            fn.colsB = b.m_cols;
            fn.a     = a.m_data;
            fn.b     = b.m_data;

            thrust::transform(
                thrust::counting_iterator<int>(0),
                thrust::counting_iterator<int>(0) + m_rows * m_cols,
                m_dataVector->begin() + m_dataVectorOffset,
                fn
                );
        }
        else if (!transposeA && !transposeB) {
            if (m_rows != a.m_rows || m_cols != b.m_cols || a.m_cols != b.m_rows)
                throw std::runtime_error("Invalid matrix dimensions");

            internal::MatrixMultiplyFn fn;
            fn.rowsA = a.m_rows;
            fn.rowsB = b.m_rows;
            fn.colsA = a.m_cols;
            fn.colsB = b.m_cols;
            fn.a     = a.m_data;
            fn.b     = b.m_data;

            thrust::transform(
                thrust::counting_iterator<int>(0),
                thrust::counting_iterator<int>(0) + m_rows * m_cols,
                m_dataVector->begin() + m_dataVectorOffset,
                fn
                );
        }
        else if (transposeA && transposeB) {
            throw std::runtime_error("Not implemented");
        }
        else /* if (!transposeA && transposeB) */ {
            if (m_rows != a.m_rows || m_cols != b.m_rows || a.m_cols != b.m_cols)
                throw std::runtime_error("Invalid matrix dimensions");

            internal::MatrixMultiplyTransposedBFn fn;
            fn.rowsA = a.m_rows;
            fn.rowsB = b.m_rows;
            fn.colsA = a.m_cols;
            fn.colsB = b.m_cols;
            fn.a     = a.m_data;
            fn.b     = b.m_data;

            thrust::transform(
                thrust::counting_iterator<int>(0),
                thrust::counting_iterator<int>(0) + m_rows * m_cols,
                m_dataVector->begin() + m_dataVectorOffset,
                fn
                );
        }
    }

    template <typename TDevice>
    void Matrix<TDevice>::addProduct(const Matrix<TDevice> &a, bool transposeA, const Matrix<TDevice> &b, bool transposeB)
    {
        if (transposeA && !transposeB) {
            if (m_rows != a.m_cols || m_cols != b.m_cols || a.m_rows != b.m_rows)
                throw std::runtime_error("Invalid matrix dimensions");

            internal::AddMatrixMultiplyTransposedAFn fn;
            fn.rowsA = a.m_rows;
            fn.rowsB = b.m_rows;
            fn.colsA = a.m_cols;
            fn.colsB = b.m_cols;
            fn.a     = a.m_data;
            fn.b     = b.m_data;

            thrust::transform(
                thrust::counting_iterator<int>(0),
                thrust::counting_iterator<int>(0) + m_rows * m_cols,
                m_dataVector->begin() + m_dataVectorOffset,
                m_dataVector->begin() + m_dataVectorOffset,
                fn
                );
        }
        else if (!transposeA && !transposeB) {
            if (m_rows != a.m_rows || m_cols != b.m_cols || a.m_cols != b.m_rows)
                throw std::runtime_error("Invalid matrix dimensions");

            internal::AddMatrixMultiplyFn fn;
            fn.rowsA = a.m_rows;
            fn.rowsB = b.m_rows;
            fn.colsA = a.m_cols;
            fn.colsB = b.m_cols;
            fn.a     = a.m_data;
            fn.b     = b.m_data;

            thrust::transform(
                thrust::counting_iterator<int>(0),
                thrust::counting_iterator<int>(0) + m_rows * m_cols,
                m_dataVector->begin() + m_dataVectorOffset,
                m_dataVector->begin() + m_dataVectorOffset,
                fn
                );
        }
        else if (transposeA && transposeB) {
            throw std::runtime_error("Not implemented");
        }
        else /* if (!transposeA && transposeB) */ {
            if (m_rows != a.m_rows || m_cols != b.m_rows || a.m_cols != b.m_cols)
                throw std::runtime_error("Invalid matrix dimensions");

            internal::AddMatrixMultiplyTransposedBFn fn;
            fn.rowsA = a.m_rows;
            fn.rowsB = b.m_rows;
            fn.colsA = a.m_cols;
            fn.colsB = b.m_cols;
            fn.a     = a.m_data;
            fn.b     = b.m_data;

            thrust::transform(
                thrust::counting_iterator<int>(0),
                thrust::counting_iterator<int>(0) + m_rows * m_cols,
                m_dataVector->begin() + m_dataVectorOffset,
                m_dataVector->begin() + m_dataVectorOffset,
                fn
                );
        }
    }

//#if (USE_CUBLAS == 1 && THRUST_DEVICE_BACKEND == THRUST_DEVICE_BACKEND_CUDA)
#if (USE_CUBLAS == 1)
    template <>
    void Matrix<Gpu>::assignProduct(const Matrix<Gpu> &a, bool transposeA, const Matrix<Gpu> &b, bool transposeB)
    {
        cublas::multiplyMatrices(
            transposeA, transposeB,
            m_rows, m_cols, (transposeA ? a.m_rows : a.m_cols),
            a.m_data, a.m_rows,
            b.m_data, b.m_rows,
            m_data,     m_rows,
            false
            );
    }

    template <>
    void Matrix<Gpu>::addProduct(const Matrix<Gpu> &a, bool transposeA, const Matrix<Gpu> &b, bool transposeB)
    {
        cublas::multiplyMatrices(
            transposeA, transposeB,
            m_rows, m_cols, (transposeA ? a.m_rows : a.m_cols),
            a.m_data, a.m_rows,
            b.m_data, b.m_rows,
            m_data,     m_rows,
            true
            );
    }
#endif

    // explicit template instantiations
    template class Matrix<Cpu>;
    template class Matrix<Gpu>;

} // namespace helpers
