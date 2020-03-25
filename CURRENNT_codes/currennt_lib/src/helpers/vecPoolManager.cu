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

#include "vecPoolManager.hpp"

#include <vector>
#include <set>
#include <stdexcept>
#include <algorithm>
#include <cassert>
#include <fstream>

#include <boost/foreach.hpp>

#define __LOCAL_VECPOOL_DEBUG_FLAG false

namespace helpers{

    template <typename TDevice>
    vecPoolManager<TDevice>::vecPoolManager(const long int maxCurLength,
					    const int parallelSeq)
	: m_maxCurLength(maxCurLength)
	, m_parallelSeq(parallelSeq)
    {
	m_vecCntBuff.clear();
	m_vecPool.clear();
	m_vecPoolInfor.clear();
    }

    template <typename TDevice>
    vecPoolManager<TDevice>::~vecPoolManager()
    {
    }

    template <typename TDevice>
    void vecPoolManager<TDevice>::addOrRemoveNewVec(const int dimension,
						    const bool flag_add)
    {
	bool flag_found = false;
	
	if (flag_add){
	    // count a new vector 
	    for (size_t idx = 0; idx < m_vecCntBuff.size(); idx++){
		if (m_vecCntBuff[idx].vecSize == dimension){
		    flag_found = true;
		    m_vecCntBuff[idx].tmp_requiredNum += 1;

		    if (m_vecCntBuff[idx].tmp_requiredNum >
			m_vecCntBuff[idx].max_requiredNum)
			m_vecCntBuff[idx].max_requiredNum =
			    m_vecCntBuff[idx].tmp_requiredNum;
		}
	    }
	    // other wiss, add a new vector infor
	    if (flag_found == false){
		vecCnt tmp_vecCnt;
		tmp_vecCnt.vecSize = dimension;
		tmp_vecCnt.tmp_requiredNum = 1;
		tmp_vecCnt.max_requiredNum = 1;
		this->m_vecCntBuff.push_back(tmp_vecCnt);
		flag_found = true;
	    }
	    
	}else{
	    // discount an existing vector
	    for (size_t idx = 0; idx < m_vecCntBuff.size(); idx++){
		if (m_vecCntBuff[idx].vecSize == dimension){
		    flag_found = true;
		    m_vecCntBuff[idx].tmp_requiredNum -= 1;
		}
		if (m_vecCntBuff[idx].tmp_requiredNum < 0)
		    throw std::runtime_error("Error: remove vec vecPoolManager");
	    }
	}


	if (__LOCAL_VECPOOL_DEBUG_FLAG){
	    printf("\n");
	    for (size_t idx = 0; idx < m_vecCntBuff.size(); idx++)
		printf("%4d, %4d, %4d\t",
		       m_vecCntBuff[idx].vecSize,
		       m_vecCntBuff[idx].tmp_requiredNum,
		       m_vecCntBuff[idx].max_requiredNum);
	}
	
	if (flag_found == false)
	    throw std::runtime_error("Error: no vec in vecPoolManager");
	else
	    return;
    }

    template <typename TDevice>
    void vecPoolManager<TDevice>::getSwapVector(
	typename vecPoolManager<TDevice>::real_vector& dataVec,
	const int layerID, const int vecDim, const bool flag_get)
    {
	bool flag_found = false;
	
	if (flag_get){
	    // allocate memory to dataVec by swapping
	    for (size_t idx = 0; idx < m_vecPoolInfor.size(); idx++){
		if (m_vecPoolInfor[idx].vecSize == vecDim &&
		    m_vecPoolInfor[idx].layerID == -1){
		    
		    m_vecPoolInfor[idx].layerID = layerID;
		    dataVec.swap(m_vecPool[idx]);
		    
		    if (dataVec.empty() || (!m_vecPool[idx].empty()))
			throw std::runtime_error("Error get swap vec in vecPoolManager");
		    
		    flag_found = true;	
		    break;
		}
	    }
	}else{
	    // release memory of dataVec by swapping
	    for (size_t idx = 0; idx < m_vecPoolInfor.size(); idx++){
		if (m_vecPoolInfor[idx].layerID == layerID &&
		    m_vecPoolInfor[idx].vecSize == vecDim){
		    
		    // this vector will be swapped
		    dataVec.swap(m_vecPool[idx]);
		    m_vecPoolInfor[idx].layerID = -1;
		    
		    if ((!dataVec.empty()) || m_vecPool[idx].empty())
			throw std::runtime_error("Error get swap vec in vecPoolManager");
		    
		    flag_found = true;

		    // refill the buffer with zero
		    thrust::fill(m_vecPool[idx].begin(),
				 m_vecPool[idx].end(), 0.0);
		    break;
		}
	    }
	}

	if (__LOCAL_VECPOOL_DEBUG_FLAG){
	    printf("\n");
	    for (size_t idx = 0; idx < m_vecPoolInfor.size(); idx++)
		printf("%4d", m_vecPoolInfor[idx].vecSize);
	    printf("\n");
	    for (size_t idx = 0; idx < m_vecPoolInfor.size(); idx++)
		printf("%4d", m_vecPoolInfor[idx].layerID);
	    std::cout << std::flush;
	}
	
	if (flag_found == false)
	    throw std::runtime_error("Error get swap vec in vecPoolManager");
	else
	    return;
    }

    template <typename TDevice>
    void vecPoolManager<TDevice>::createVecPool()
    {

	if (false){
	    for (size_t idx = 0; idx < m_vecCntBuff.size(); idx++){
		printf("\n%d vectors of dimension %d",
		       m_vecCntBuff[idx].max_requiredNum,
		       m_vecCntBuff[idx].vecSize);
	    }
	}
	
	// create pool of vectors
	for (size_t idx = 0; idx < m_vecCntBuff.size(); idx++){

	    if (__LOCAL_VECPOOL_DEBUG_FLAG){
		// for debugging
		printf("\n%d vectors of dimension %d",
		       m_vecCntBuff[idx].max_requiredNum,
		       m_vecCntBuff[idx].vecSize);
	    }
	    
	    for (int cnt = 0; cnt < m_vecCntBuff[idx].max_requiredNum; cnt++){
		if (m_vecCntBuff[idx].vecSize > 0){
		    // allocate memory
		    real_vector tmp_vec =
			Cpu::real_vector(m_vecCntBuff[idx].vecSize *
					 m_maxCurLength *
					 m_parallelSeq);

		    // create vector infor
		    vecInfor tmp_vecInfor;
		    tmp_vecInfor.vecSize = m_vecCntBuff[idx].vecSize;
		    tmp_vecInfor.layerID = -1;

		    // push back to vector
		    m_vecPool.push_back(tmp_vec);
		    m_vecPoolInfor.push_back(tmp_vecInfor);
		}
	    }
	}
    }

    template <typename TDevice>
    bool vecPoolManager<TDevice>::empty()
    {
	return m_vecCntBuff.empty();
    }

    template class vecPoolManager<Gpu>;
    template class vecPoolManager<Cpu>;

}

