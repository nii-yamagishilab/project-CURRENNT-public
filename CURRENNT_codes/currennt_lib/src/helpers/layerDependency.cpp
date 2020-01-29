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

#include "layerDependency.hpp"
#include <cstdio>
#include <boost/foreach.hpp>

namespace helpers{

    /* 
     *   Methods for layerDep
     */
    
    // Initialization
    layerDep::layerDep(const int layerID)
    {
	m_layerID = layerID;
	m_fromwhich.clear();
	m_towhich.clear();
    }

    // Destructor
    layerDep::~layerDep()
    {
    }
    
    std::vector<int>& layerDep::get_towhich()
    {
	return m_towhich;
    }

    std::vector<int>& layerDep::get_fromwhich()
    {
	return m_fromwhich;
    }

    int layerDep::get_layerID()
    {
	return m_layerID;
    }

    void layerDep::add_towhich(std::vector<int> &outs)
    {
	m_towhich.insert(m_towhich.end(), outs.begin(), outs.end());
    }

    void layerDep::add_towhich(const int outs)
    {
	m_towhich.insert(m_towhich.end(), outs);
    }

    void layerDep::del_towhich(const int outs)
    {
	for (size_t idx=0; idx < m_towhich.size(); idx++)
	    if (m_towhich[idx] == outs)
		m_towhich[idx] = -1;
    }

    void layerDep::nul_towhich()
    {
	this->m_towhich.clear();
    }

    bool layerDep::empty_towhich()
    {
	bool any_val = true;
	for (size_t idx=0; idx < m_towhich.size(); idx++)
	    any_val = (m_towhich[idx]>0)?(false):(any_val);
	return any_val;
    }
    
    void layerDep::add_fromwhich(std::vector<int> &ins)
    {
	m_fromwhich.insert(m_fromwhich.end(), ins.begin(), ins.end());
    }

    void layerDep::add_fromwhich(const int ins)
    {
	m_fromwhich.insert(m_fromwhich.end(), ins);
    }

    void layerDep::del_fromwhich(const int ins)
    {
	for (size_t idx=0; idx < m_fromwhich.size(); idx++)
	    if (m_fromwhich[idx] == ins)
		m_fromwhich[idx] = -1;
    }
    
    void layerDep::nul_fromwhich()
    {
	this->m_fromwhich.clear();
    }

    bool layerDep::empty_fromwhich()
    {
	bool any_val = true;
	for (size_t idx=0; idx < m_fromwhich.size(); idx++)
	    any_val = (m_fromwhich[idx]>0)?(false):(any_val);
	return any_val;
    }



    /* 
     *   Methods for networkDepMng
     */

    //
    networkDepMng::networkDepMng()
    {
    }
    
    networkDepMng::~networkDepMng()
    {
    }

    std::vector<layerDep>& networkDepMng::get_layerDeps()
    {
	return m_layerDeps;
    }

    layerDep& networkDepMng::get_layerDep(const int layerID)
    {
	if (layerID < 0 || layerID >= m_layerDeps.size())
	    throw std::runtime_error("\nget_layerDep: input layerId is out of range");
	return m_layerDeps[layerID];
    }
    
    void networkDepMng::build(const int layerNums)
    {
	m_layerDeps.reserve(layerNums);
	for (int layerIdx = 0; layerIdx < layerNums; layerIdx++){
	    layerDep tmp_layerDep(layerIdx);
	    m_layerDeps.push_back(layerIdx);
	}
    }

    void networkDepMng::add_layerDep(const int layerId, std::vector<int> depend_layerIDs)
    {
	if (layerId < 0 || layerId >= m_layerDeps.size())
	    throw std::runtime_error("\nadd_layerDep: input layerId is out of range");
	m_layerDeps[layerId].add_fromwhich(depend_layerIDs);
	
	BOOST_FOREACH (int depend_layerID, depend_layerIDs){
	    if (depend_layerID < 0 || depend_layerID >= m_layerDeps.size())
		throw std::runtime_error("\nadd_layerDep: depend layerId is out of range");
	    m_layerDeps[depend_layerID].add_towhich(layerId);
	}
    }

    void networkDepMng::print_layerDep()
    {
	for (size_t layerIdx = 0; layerIdx < m_layerDeps.size(); layerIdx++){

	    if (m_layerDeps[layerIdx].empty_towhich() &&
		m_layerDeps[layerIdx].empty_fromwhich())
		continue;
	    
	    printf("\n%d: to ", m_layerDeps[layerIdx].get_layerID());
	    for (size_t layerIdx2 = 0;
		 layerIdx2 < m_layerDeps[layerIdx].get_towhich().size();
		 layerIdx2++){
		printf("%d,", m_layerDeps[layerIdx].get_towhich()[layerIdx2]);
	    }
	    printf("\tfrom ");
	    for (size_t layerIdx2 = 0;
		 layerIdx2 < m_layerDeps[layerIdx].get_fromwhich().size();
		 layerIdx2++){
		printf("%d,", m_layerDeps[layerIdx].get_fromwhich()[layerIdx2]);
	    }

	}
    }

    
}
