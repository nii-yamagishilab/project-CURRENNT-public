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

#ifndef DATAPROCESS_HPP
#define DATAPROCESS_HPP
 
#include <vector>
#include <string>
#include "../Types.hpp"
#include "../data_sets/DataSetFraction.hpp"

#define DATACHECKER_WAVEFORM_NONE   -1
#define DATACHECKER_WAVEFORM_SILENCE 1
 
 
bool checkWaveformValidity(Gpu::real_vector data, const int length);
bool checkWaveformValidity(Cpu::real_vector data, const int length);
     
#endif
