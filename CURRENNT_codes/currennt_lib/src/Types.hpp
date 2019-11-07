/******************************************************************************
 * This file is an addtional component of CURRENNT. 
 * Xin WANG
 * National Institute of Informatics, Japan
 * 2018
 *
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

#ifndef TYPES_HPP
#define TYPES_HPP

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <vector_types.h>

#define PATTYPE_NONE   0 ///< pattern does not belong to the sequence
#define PATTYPE_FIRST  1 ///< first pattern/timestep in the sequence
#define PATTYPE_NORMAL 2 ///< pattern/timestep with a sequence (not first/last)
#define PATTYPE_LAST   3 ///< last pattern/timestep in the sequence

#define AUXDATATYPE_FLOAT 0 // used for the auxilliary data
#define AUXDATATYPE_INT   1 // used for the auxilliary data
#define AUXDATATYPE_CHAR  2 // used for the auxilliary data


/*************************************************************************//**
 * The floating point type used for all computations
 *****************************************************************************/
// real-valued float number
typedef float real_t;

// complex-valued float number
typedef float2 complex_t;

/*************************************************************************//**
 * Data types on the CPU
 *****************************************************************************/
struct Cpu
{
    enum { cublas_capable = false };

    // to store real_valued data array
    typedef thrust::host_vector<real_t> real_vector;
    
    // to store int data array
    typedef thrust::host_vector<int>    int_vector;

    // to store bool data array
    typedef thrust::host_vector<bool>   bool_vector;

    // to store char data array (indicator of sequence boundary)
    typedef thrust::host_vector<char>   pattype_vector;

    // to store complex-valued data array
    typedef thrust::host_vector<complex_t> complex_vector;
};


/*************************************************************************//**
 * Data types on the GPU
 *****************************************************************************/
struct Gpu
{
    enum { cublas_capable = true };

    typedef thrust::device_vector<real_t> real_vector;
    typedef thrust::device_vector<int>    int_vector;
    typedef thrust::device_vector<bool>   bool_vector;
    typedef thrust::device_vector<char>   pattype_vector;
    typedef thrust::device_vector<complex_t> complex_vector;
};


#endif // TYPES_HPP
