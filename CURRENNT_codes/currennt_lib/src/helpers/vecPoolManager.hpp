/******************************************************************************
 * This file is an addtional component of CURRENNT. 
 * Xin WANG
 * National Institute of Informatics, Japan
 * 2016 - 2020
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

#ifndef HELPERS_VECPOOLMANAGER_HPP
#define HELPERS_VECPOOLMANAGER_HPP

#include "../Types.hpp"
#include <vector>
#include <set>

namespace helpers{
    	
    struct vecCnt{
	
	// vector dimension
	int vecSize;
	
	// temporary counter of vector number
	int tmp_requiredNum;

	// maximum number of vectors required
	int max_requiredNum;
    };

    struct vecInfor{

	// vector dimension
	int vecSize;

	// ID of layer who get the vector
	int layerID;
	
    };

    /**
     *  vector Pool manager
     */

    template <typename TDevice>
    class vecPoolManager
    {
	typedef typename TDevice::real_vector real_vector;
	typedef typename Cpu::real_vector     cpu_real_vec;
	typedef typename TDevice::int_vector  int_vector;
	typedef typename Cpu::int_vector      cpu_int_vec;

    private:
	// maximum length in training batch
	long int                  m_maxCurLength;
	
	// number of parallel utterances
	int                       m_parallelSeq;

	// buffer to count vectors
	std::vector<vecCnt>       m_vecCntBuff;

	// buffer of vectors
	std::vector<real_vector>  m_vecPool;

	// buffer of vector information
	std::vector<vecInfor>     m_vecPoolInfor; 
	
    public:

	// Constructor
	vecPoolManager(const long int maxCurLength,
		       const int parallelSeq);

	// Deconstructor
	~vecPoolManager();

	// count or discount a new vector
	void addOrRemoveNewVec(const int dimension,
			       const bool flag_add);

	// create vector pool
	void createVecPool();
	
	// swap vector;
	void getSwapVector(real_vector& dataVec,
			   const int layerID,
			   const int vecDim,
			   const bool flag_get);

	bool empty();
    };
}

#endif
