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

#ifndef HELPERS_MATRIX_HPP
#define HELPERS_MATRIX_HPP

#include "../Types.hpp"


namespace helpers {

    template <typename TDevice>
    class Matrix
    {
        typedef typename TDevice::real_vector real_vector;

    private:
        real_vector *m_dataVector;
        int          m_dataVectorOffset;
        real_t      *m_data;
        int          m_rows;
        int          m_cols;

    public:
        Matrix();
        Matrix(real_vector *data, int rows, int cols, int dataOffset = 0);
        ~Matrix();

        void assignProduct(const Matrix<TDevice> &a, bool transposeA, const Matrix<TDevice> &b, bool transposeB);
        void addProduct   (const Matrix<TDevice> &a, bool transposeA, const Matrix<TDevice> &b, bool transposeB);
    };

} // namespace helpers

#endif // HELPERS_MATRIX_HPP
