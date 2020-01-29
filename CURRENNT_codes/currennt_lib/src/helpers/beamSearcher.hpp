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

#ifndef HELPERS_BEANSEARCH_HPP
#define HELPERS_BEANSEARCH_HPP

#include "../Types.hpp"
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



/* ----- Definition for beam-search generation ----- */
/*   Internal class defined for NeuralNetwork only   */
/* --------------------------------------------------*/
namespace beamsearch{
    	
    struct sortUnit{
	real_t prob;
	int    idx;
    };

    bool compareFunc(const sortUnit& a, const sortUnit& b);
    
    // Search state
    template <typename TDevice>
    class searchState
    {
	typedef typename TDevice::real_vector real_vector;
	typedef typename Cpu::real_vector     cpu_real_vec;
	typedef typename TDevice::int_vector  int_vector;
	typedef typename Cpu::int_vector      cpu_int_vec;

    private:
	int              m_stateID;    // ID of the current state
	real_t           m_prob;       // probability
	int              m_timeStep;
	cpu_int_vec      m_stateTrace; // trace of the state ID
	cpu_real_vec     m_probTrace;  // trace of the probability distribution

	std::vector<int>          m_netStateSize;   // pointer in m_netState
	std::vector<cpu_real_vec> m_netState;     // hidden variables of the network
	
	
    public:
	searchState();
	searchState(std::vector<int> &netStateSize, const int maxSeqLength, const int stateNM);
	~searchState();

	const int      getStateID();
	const real_t   getProb();
	      int      getStateID(const int id);
	      real_t   getProb(const int id);
	const int      getTimeStep();
	cpu_int_vec&    getStateTrace();
	cpu_real_vec&   getProbTrace();
	cpu_real_vec&   getNetState(const int id);
	
	void setStateID(const int stateID);
	void setTimeStep(const int timeStep);
	void setProb(const real_t prob);
	void mulProb(const real_t prob);
	void setStateTrace(const int time, const int stateID);
	void setProbTrace(const int time, const real_t prob);
	void setNetState(const int layerID, cpu_real_vec& state);
	void liteCopy(searchState<TDevice>& sourceState);
	void fullCopy(searchState<TDevice>& sourceState);
	void print();
    };

    

    
    // Macro search state
    template <typename TDevice>
    class searchEngine
    {
	typedef typename TDevice::real_vector real_vector;
	typedef typename Cpu::real_vector cpu_real_vector;
	typedef typename TDevice::int_vector  int_vector;
	typedef typename Cpu::int_vector      cpu_int_vector;
	
    private:
	std::vector<searchState<TDevice> > m_stateSeq;
	std::vector<sortUnit> m_sortUnit;
	
	int m_beamSize;
	int m_stateLength;
	int m_validStateNum;
	
    public:
	searchEngine(const int beamSize);	
	~searchEngine();
	

	void setState(const int id, searchState<TDevice> &state);
	void setSortUnit(const int id, searchState<TDevice> &state);
	void setValidBeamSize(const int num);
	
	void addState(searchState<TDevice> &state);
	void sortSet(const int size);
	void printBeam();
	
	searchState<TDevice>& retrieveState(const int id);
	int  getBeamSize();
	int  getValidBeamSize();

    };

}

#endif
