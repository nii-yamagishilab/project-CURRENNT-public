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

#ifndef HELPERS_LAYERDEPENDENCY_HPP
#define HELPERS_LAYERDEPENDENCY_HPP

#include <vector>

namespace helpers{

    class layerDep
    {
    private:
	// current layer ID
	int m_layerID;

	// ID of the target layers (layers which take this layers output as input)
	std::vector<int> m_towhich;

	// ID of the source layers (layers whose output is taken by this layers as input)
	std::vector<int> m_fromwhich;   
	
    public:
	
	layerDep(const int layerID);
	~layerDep();

	// return ID of target layers
	std::vector<int>& get_towhich();
	
	// return ID of source layers
	std::vector<int>& get_fromwhich();
	
	// return ID of this layer
	int get_layerID();

	void add_towhich(std::vector<int> &outs);
	void add_towhich(const int outs);
	void del_towhich(const int outs);
	void nul_towhich();
	bool empty_towhich();
	
	void add_fromwhich(std::vector<int> &ins);
	void add_fromwhich(const int ins);
	void del_fromwhich(const int ins);
	void nul_fromwhich();
	bool empty_fromwhich();
    };

    
    // manage the dependency between layers of network
    class networkDepMng
    {
    private:
	std::vector<layerDep> m_layerDeps;

    public:
	networkDepMng();
	~networkDepMng();

	// build the network dependency map
	void build(const int layerNums);
	
	// add dependency to layerID
	void add_layerDep(const int layerID, std::vector<int> depend_layerIDs);
	
	// get the dependency
	std::vector<layerDep>& get_layerDeps();
	
	layerDep& get_layerDep(const int layerID);

	// print information
	void print_layerDep();
    };

}



#endif
