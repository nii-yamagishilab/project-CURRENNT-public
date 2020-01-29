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

#include "beamSearcher.hpp"

#include <vector>
#include <set>
#include <stdexcept>
#include <algorithm>
#include <cassert>
#include <fstream>

#include <boost/foreach.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#include <boost/random/mersenne_twister.hpp>


namespace beamsearch{

    //
    bool compareFunc(const sortUnit& a, const sortUnit& b){
	return a.prob >= b.prob;
    }

    template <typename TDevice>
    searchState<TDevice>::searchState()
	: m_stateID(-1)
	, m_prob(1.0)
	, m_timeStep(-1)
    {
	m_netState.clear();
	m_stateTrace.clear();
	m_probTrace.clear();
	m_netStateSize.clear();
    }
    
    template <typename TDevice>
    searchState<TDevice>::searchState(std::vector<int> &netStateSize,
				      const int maxSeqLength, const int stateNM)
	: m_stateID(-1)
	, m_prob(0.0)
	, m_timeStep(-1)
    {
	m_netState.resize(netStateSize.size());
	
	cpu_real_vec tmp;
	int tmpBuf = 0;
	for (int i = 0; i < netStateSize.size(); i++) {
	    tmpBuf += netStateSize[i];
	    tmp.resize(netStateSize[i], 0.0);
	    m_netState[i] = tmp;
	}

	m_netStateSize = netStateSize;

	cpu_real_vec tmp2(maxSeqLength * stateNM, 0.0);
	m_probTrace = tmp2;

	cpu_int_vec tmp3(maxSeqLength, 0);
	m_stateTrace = tmp3;
	
    }
    
    template <typename TDevice>
    searchState<TDevice>::~searchState()
    {
    }

    template <typename TDevice>
    const int searchState<TDevice>::getStateID()
    {
	return m_stateID;
    }

    template <typename TDevice>
    const real_t searchState<TDevice>::getProb()
    {
	return m_prob;
    }
    
    template <typename TDevice>
    const int searchState<TDevice>::getTimeStep()
    {
	return m_timeStep;
    }

    template <typename TDevice>
    int searchState<TDevice>::getStateID(const int id)
    {
	if (id >= m_stateTrace.size())
	    throw std::runtime_error("state ID is larger than expected");
	return m_stateTrace[id];
    }

    template <typename TDevice>
    typename searchState<TDevice>::cpu_int_vec& searchState<TDevice>::getStateTrace()
    {
	return m_stateTrace;
    }

    template <typename TDevice>
    typename searchState<TDevice>::cpu_real_vec& searchState<TDevice>::getProbTrace()
    {
	return m_probTrace;
    }

    template <typename TDevice>
    void searchState<TDevice>::liteCopy(searchState<TDevice>& sourceState)
    {
	m_stateID    = sourceState.getStateID();
	m_prob       = sourceState.getProb();
	m_timeStep   = sourceState.getTimeStep();
	thrust::copy(sourceState.getStateTrace().begin(),
		     sourceState.getStateTrace().end(), m_stateTrace.begin());
	thrust::copy(sourceState.getProbTrace().begin(),
		     sourceState.getProbTrace().end(), m_probTrace.begin());
    }

    template <typename TDevice>
    void searchState<TDevice>::fullCopy(searchState<TDevice>& sourceState)
    {
	m_stateID    = sourceState.getStateID();
	m_prob       = sourceState.getProb();
	m_timeStep   = sourceState.getTimeStep();
	thrust::copy(sourceState.getStateTrace().begin(),
		     sourceState.getStateTrace().end(), m_stateTrace.begin());
	thrust::copy(sourceState.getProbTrace().begin(),
		     sourceState.getProbTrace().end(), m_probTrace.begin());
	for (int i = 0; i < m_netStateSize.size(); i++){
	    this->setNetState(i, sourceState.getNetState(i));
	}
    }

    template <typename TDevice>
    real_t searchState<TDevice>::getProb(const int id)
    {
	if (id >= m_probTrace.size())
	    throw std::runtime_error("prob ID is larger than expected");
	return m_probTrace[id];
    }

    template <typename TDevice>
    typename searchState<TDevice>::cpu_real_vec& searchState<TDevice>::getNetState(
	const int id)
    {
	if (id >= m_netStateSize.size())
	    throw std::runtime_error("layer ID is larger than expected");
	return m_netState[id];
    }
    
    template <typename TDevice>
    void searchState<TDevice>::setStateID(const int stateID)
    {
	m_stateID = stateID;
    }

    template <typename TDevice>
    void searchState<TDevice>::setTimeStep(const int timeStep)
    {
	m_timeStep = timeStep;
    }

    template <typename TDevice>
    void searchState<TDevice>::setProb(const real_t prob)
    {
	m_prob = prob;
    }

    template <typename TDevice>
    void searchState<TDevice>::mulProb(const real_t prob)
    {
	if (prob < 1.1754944e-038f)
	    m_prob += (-1e30f);
	else
	    m_prob += std::log(prob);
    }
    
    template <typename TDevice>
    void searchState<TDevice>::setStateTrace(const int time, const int stateID)
    {
	if (time >= m_stateTrace.size())
	    throw std::runtime_error("setStateTrace, time is larger than expected");
	m_stateTrace[time] = stateID;
    }

    template <typename TDevice>
    void searchState<TDevice>::setProbTrace(const int time, const real_t prob)
    {
	if (time >= m_probTrace.size())
	    throw std::runtime_error("setProbTrace, time is larger than expected");
	m_probTrace[time] = prob;
    }

    template <typename TDevice>
    void searchState<TDevice>::setNetState(const int layerID, cpu_real_vec& state)
    {
	if (layerID >= m_netStateSize.size())
	    throw std::runtime_error("setNetState, time is larger than expected");
	if (m_netStateSize[layerID] > 0)
	    thrust::copy(state.begin(), state.begin()+m_netStateSize[layerID],
			 m_netState[layerID].begin());
    }

    template <typename TDevice>
    void searchState<TDevice>::print()
    {
	printf("%d:%d\t%f\t", m_timeStep, m_stateID, m_prob);
	//printf("%d", m_stateTrace.size());
	cpu_int_vec tmp = m_stateTrace;
	for (int i = 0; i <= m_timeStep; i++)
	    printf("%d ", tmp[i]);
	printf("\n");
    }



    
    template <typename TDevice>
    searchEngine<TDevice>::searchEngine(const int beamSize)
	: m_beamSize(beamSize)
	, m_stateLength(0)
	, m_validStateNum(0)
    {
	m_stateSeq.clear();
    }

    template <typename TDevice>
    searchEngine<TDevice>::~searchEngine()
    {
    }

    template <typename TDevice>
    void searchEngine<TDevice>::addState(searchState<TDevice> &state)
    {
	sortUnit tmp;
	m_stateSeq.push_back(state);
	m_sortUnit.push_back(tmp);
    }

    template <typename TDevice>
    void searchEngine<TDevice>::setState(const int id, searchState<TDevice> &state)
    {
	if (id > m_stateSeq.size())
	    throw std::runtime_error("beam search state not found");
	m_stateSeq[id].fullCopy(state);
    }
    
    template <typename TDevice>
    void searchEngine<TDevice>::setSortUnit(const int id, searchState<TDevice> &state)
    {
	if (id > m_sortUnit.size())
	    throw std::runtime_error("beam search state not found");
	m_sortUnit[id].prob = state.getProb();
	m_sortUnit[id].idx  = id;
    }

    template <typename TDevice>
    void searchEngine<TDevice>::setValidBeamSize(const int num)
    {
	m_validStateNum = num;
    }

    template <typename TDevice>
    int searchEngine<TDevice>::getBeamSize()
    {
	return m_beamSize;
    }

    template <typename TDevice>
    int searchEngine<TDevice>::getValidBeamSize()
    {
	return m_validStateNum;
    }

    template <typename TDevice>
    searchState<TDevice>& searchEngine<TDevice>::retrieveState(const int id)
    {
	if (id > m_stateSeq.size())
	    throw std::runtime_error("beam search state not found");
	return m_stateSeq[id];
    }
	
    template <typename TDevice>
    void searchEngine<TDevice>::sortSet(const int size)
    {
	m_validStateNum = (m_beamSize < size)?(m_beamSize):(size);
	std::sort(m_sortUnit.begin(), m_sortUnit.begin() + size, compareFunc);
	for (int i = 0; i < m_validStateNum; i++){
	    if ((m_beamSize + m_sortUnit[i].idx) < m_stateSeq.size())
		m_stateSeq[i] = m_stateSeq[m_beamSize + m_sortUnit[i].idx];
	    else{
		printf("beam search %d unit invalid", m_beamSize + m_sortUnit[i].idx);
		throw std::runtime_error("beam search sort error");
	    }
	}
    }

    template <typename TDevice>
    void searchEngine<TDevice>::printBeam()
    {
	for (int i = 0; i < m_validStateNum; i++)
	    m_stateSeq[i].print();
    }
}

template class beamsearch::searchEngine<Gpu>;
template class beamsearch::searchEngine<Cpu>;

template class beamsearch::searchState<Gpu>;
template class beamsearch::searchState<Cpu>;

