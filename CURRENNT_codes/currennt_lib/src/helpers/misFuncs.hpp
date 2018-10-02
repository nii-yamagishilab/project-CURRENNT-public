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


#ifndef MISFUNCS_HPP
#define MISFUNCS_HPP


#include <vector>
#include <string>
#include "../Types.hpp"


namespace misFuncs{
/* ***** Functions for string process ***** */
void   ParseStrOpt(const std::string stringOpt, std::vector<std::string> &optVec,
		   const std::string para="_");
void   ParseIntOpt(const std::string stringOpt,   Cpu::int_vector &optVec);
void   ParseFloatOpt(const std::string stringOpt, Cpu::real_vector &optVec);

/* ***** Functions for vector process ***** */
int    SumCpuIntVec(Cpu::int_vector &temp);
int    MaxCpuIntVec(Cpu::int_vector &temp);
void   PrintVecBinH(Cpu::real_vector &temp);
void   PrintVecBinH(Cpu::complex_vector &temp);
void   PrintVecBinH(Cpu::int_vector &temp);
void   PrintVecBinD(Gpu::real_vector &temp);
void   PrintVecBinD(Gpu::int_vector &temp);
void   PrintVecBinD(Gpu::complex_vector &temp);
void   AppendVecBin(Gpu::real_vector &temp);
void   AppendVecBin(Gpu::int_vector &temp);
void   AppendVecBin(Cpu::real_vector &temp);
void   AppendVecBin(Cpu::int_vector &temp);


/* ***** Functions for training process ***** */
int    flagUpdateDiscriminator(const int epoch, const int frac);

/* ***** Functions for numerical process ***** */
real_t GetRandomNumber();
bool   closeToZero(const real_t t1, const real_t lowBound = -0.0001,
		   const real_t upBound = 0.0001);
    
int getResoLength(const int maxSeqLength, const int timeResolution, const int parallel);

/* ***** Function for I/O ****** */
int ReadRealData(const std::string dataPath, Cpu::real_vector &data);

}
#endif
