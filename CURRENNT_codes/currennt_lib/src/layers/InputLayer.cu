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

#include "InputLayer.hpp"
#include "../Configuration.hpp"
#include "../helpers/misFuncs.hpp"

#include <boost/random/normal_distribution.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#include <boost/random/mersenne_twister.hpp>

#include <boost/lexical_cast.hpp>
#include <thrust/transform.h>
#include <stdexcept>
#include <fstream>

#include "../helpers/getRawPointer.cuh"
#include <thrust/transform_reduce.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/sequence.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/execution_policy.h>




namespace internal {
namespace {

    // Block20170904x02
}
}


namespace layers {

    template <typename TDevice>
    InputLayer<TDevice>::InputLayer(const helpers::JsonValue &layerChild,
				    int parallelSequences,
				    int maxSeqLength,
				    int layerID)
        : Layer<TDevice>(layerChild,
			 parallelSequences,
			 maxSeqLength,
			 Configuration::instance().trainingMode(),
			 layerID,
			 NULL,
			 true)
	, m_weDim(0)
	, m_flagWeUpdate(false)
    {

	m_weMask.clear();
	m_weMaskFlag = false;

	if (this->getResolution() > 1)
	    throw std::runtime_error("Resolution for input layer should be 1");
    }

    template <typename TDevice>
    InputLayer<TDevice>::~InputLayer()
    {
    }

    template <typename TDevice>
    const std::string& InputLayer<TDevice>::type() const
    {
        static const std::string s("input");
        return s;
    }

    template <typename TDevice>
    void InputLayer<TDevice>::loadSequences(const data_sets::DataSetFraction &fraction,
					    const int nnState)
    {
	
	if (m_flagWeUpdate){
	    if (m_weIDDim > fraction.inputPatternSize()){
		throw std::runtime_error("WE dimension is larger than input data dimension");
	    }
	    if (this->size() != fraction.inputPatternSize()-1+m_weDim){
		printf("Input layer size should be: %d", fraction.inputPatternSize() -1 + m_weDim);
		throw std::runtime_error("Input's dimension -1 + weDim != input layer size");
	    }
	}else{
	    if (fraction.inputPatternSize() != this->size())
		throw std::runtime_error("Input layer size of != data input pattern size of ");
        }

        Layer<TDevice>::loadSequences(fraction, nnState);
	
	/* Add 16-02-22 Wang: for WE updating */
	// Original code of CURRENNT, just copy the input data from fraction to this->outputs()
	// thrust::copy(fraction.inputs().begin(),fraction.inputs().end(),this->_outputs().begin());

	if (m_flagWeUpdate){
	    // when embedded vectors are used

	    // Before loadSequences(), readWeBank() should have been called
	    int weidx = 0;
	    long unsigned int bias = 0;
	    long unsigned int fracTime = (fraction.inputs().size()/fraction.inputPatternSize());
	    Cpu::real_vector tempInput;
	    
	    if (fracTime > m_weIdx.size()) throw std::runtime_error("m_weIdx is too short\n");
	    thrust::fill(m_weIdx.begin(), m_weIdx.end(), -1);
	    
	    // Block20170904x01
	    tempInput.resize(this->size(), 0.0);
	    for (int i = 0; i < fracTime; i++){
		bias = i * fraction.inputPatternSize();
		
		// copy the normal input data
		thrust::copy(fraction.inputs().begin() + bias, 
			     fraction.inputs().begin() + bias + fraction.inputPatternSize(), 
			     tempInput.begin());
		
		// retrieve the embedded vector idx and save m_weIdx
		weidx = (long unsigned int)(fraction.inputs()[i * fraction.inputPatternSize() + 
							      m_weIDDim]);
		if (weidx * m_weDim > m_weBank.size()){
		    printf("Vector idx: %d\t", weidx);
		    throw std::runtime_error("vector idx larger than weBank size");
		}
		// store the idx of embedded vector
		m_weIdx[i] = weidx;
		
		// retrieve the embedded vector from m_weBank
		thrust::copy(m_weBank.begin()  + weidx     * m_weDim, 
			     m_weBank.begin()  + (weidx+1) * m_weDim, 
			     tempInput.begin() + fraction.inputPatternSize()  - 1);

		// Block#01
		// copy the we data into the input data (output of the InputLayer)
		thrust::copy(tempInput.begin(), tempInput.end(),
			     this->_outputs().begin() + i * this->size());
	    }
	}
	else
	{
	    // Normal case
	    thrust::copy(fraction.inputs().begin(), fraction.inputs().end(),
			 this->_outputs().begin());
	}
    }

    template <typename TDevice>
    void InputLayer<TDevice>::computeForwardPass(const int nnState)
    {
    }
    
    template <typename TDevice>
    void InputLayer<TDevice>::computeForwardPass(const int timeStep, const int nnState)
    {
    }

    template <typename TDevice>
    void InputLayer<TDevice>::computeBackwardPass(const int nnState)
    {
    }


    /* Add 16-02-22 Wang: for WE updating */
    // return the reference to m_weBank;
    template <typename TDevice>
    Cpu::real_vector& InputLayer<TDevice>::_weBank(){
	return m_weBank;
    }
    template <typename TDevice>
    Cpu::real_vector& InputLayer<TDevice>::_weIdx(){
	return m_weIdx;
    }

    // return the m_weDim
    template <typename TDevice>
    unsigned int& InputLayer<TDevice>::_weDim(){
	return m_weDim;
    }
    template <typename TDevice>
    unsigned int& InputLayer<TDevice>::_weIDDim(){
	return m_weIDDim;
    }
    
    // read the we data into m_weBank
    template <typename TDevice>
    bool InputLayer<TDevice>::readWeBank(const std::string weBankPath, 
					 const unsigned dim, const unsigned dimidx, 
					 const unsigned maxLength)
    {
	// 
	if (dim < 1) throw std::runtime_error("Dimention of weBank below 1");
	// save the information
	m_weDim                 = dim;
	m_flagWeUpdate          = true;
	m_weIDDim               = dimidx;

	// to store the word vector sequences for each frame
	m_weIdx    = Cpu::real_vector(maxLength, -1);
	
	// set the flag for We input
	this->_setInputWeUpdate(true);  
	
	// read the WE bank from IO
	printf("Initialize embedded vectors");
	printf("\n\tRead %d vectors, of dimension %d\n",
	       misFuncs::ReadRealData(weBankPath, m_weBank)/dim, dim);
	// Block#02	
	return true;
    }
    
    template <typename TDevice>
    bool InputLayer<TDevice>::flagInputWeUpdate()
    {
	return m_flagWeUpdate;
    }

    template <typename TDevice>
    bool InputLayer<TDevice>::saveWe(const std::string weFile)
    {
	if (m_flagWeUpdate && m_weBank.size()>0){
	    std::ofstream ofs(weFile.c_str(), std::ofstream::binary);
	    if (!ofs.good()){
		std::cout << "Fail to open " << weFile << std::endl;
		return false;
	    }
	    // we assume it is a CPU vector
	    std::vector<real_t> tempVec(m_weBank.begin(), m_weBank.end());
	    for(int i=0; i<tempVec.size(); i++){
		ofs.write((char *)&(tempVec[i]), sizeof(real_t));
	    }
	    ofs.close();
	}else{
	    std::cout << "No WeBank is available " << std::endl;
	    return false;
	}
	return true;
    }

    template <typename TDevice>
    void InputLayer<TDevice>::reInitWeight()
    {
	// nothing to be done here
    }
    
    template <typename TDevice>
    bool InputLayer<TDevice>::initWeNoiseOpt(const int weNoiseStartDim, const int weNoiseEndDim,
					     const real_t weNoiseDev)
    {
	this->m_weNoiseStartDim = weNoiseStartDim;
	this->m_weNoiseEndDim   = weNoiseEndDim;
	this->m_weNoiseDev      = weNoiseDev;
	if (this->m_weNoiseStartDim > this->m_weNoiseEndDim){
	    printf("Error: this->m_weNoiseStartDim > this->m_weNoiseEndDim\n");
	    return false;
	}
	if (this->m_weNoiseDev < 0.0){
	    printf("Error: this->m_weNoiseDev < 0.0 \n");
	    return false;
	}
	if (this->m_weNoiseStartDim > 0){
	    printf("WE noise: from %d to %d, %f\n", weNoiseStartDim, weNoiseEndDim, weNoiseDev);
	}
	if (this->m_weNoiseStartDim > 0)
	    throw std::runtime_error("noise on WE is deprecated");
	return true;
    }

    template <typename TDevice>
    int InputLayer<TDevice>::readWeMask(std::vector<real_t>::iterator b)
    {
	if (m_weBank.size()>0){
	    Cpu::real_vector tempVec;
	    tempVec.resize(m_weBank.size());
	    
	    std::vector<real_t>::iterator t = b;
	    for (int i = 0; i < m_weBank.size(); ++t, i++){
		if (*t <0 || *t >1)
		    throw std::runtime_error("DataMask is out of range [0, 1]");
		else
		    tempVec[i] = *t;
	    }	    
	    // copy the mask data into m_weMask
	    m_weMask     = tempVec;
	    // set the flag
	    m_weMaskFlag = true;
	}
	return m_weBank.size();
    }

    template <typename TDevice>
    void InputLayer<TDevice>::maskWe()
    {
	if (m_weMaskFlag)
	    thrust::transform(m_weBank.begin(),  m_weBank.end(),
			      m_weMask.begin(),  m_weBank.begin(),
			      thrust::multiplies<real_t>());
    }

    template <typename TDevice>
    Cpu::real_vector& InputLayer<TDevice>::_weMask()
    {
	return m_weMask;
    }

    template <typename TDevice>
    bool InputLayer<TDevice>::flagWeMask()
    {
	return m_weMaskFlag;
    }

    
    // explicit template instantiations
    template class InputLayer<Cpu>;
    template class InputLayer<Gpu>;

} // namespace layers



/*
Block#01
		// Optinal: add noise to the input
		// Note: this is different from the input_noise_sigma
		//       here, the noise will be added every epoch
		if (this->m_weNoiseStartDim > 0){		    
		    if (this->m_weNoiseStartDim >= this->size() ||
			this->m_weNoiseEndDim   >  this->size()){
			throw std::runtime_error("weNoiseDimenion error");
		    }
		    const Configuration &config = Configuration::instance();	    
		    static boost::mt19937 *gen = NULL;
		    if (!gen) {
			gen = new boost::mt19937;
			gen->seed(config.randomSeed()+100);
		    }
		    boost::random::normal_distribution<real_t> dist(0.0, this->m_weNoiseDev);
		    for (size_t j = this->m_weNoiseStartDim; j < this->m_weNoiseEndDim; ++j)
			tempInput[j] += dist(*gen);;
		}


Block#02
	
	std::ifstream ifs(weBankPath.c_str(), std::ifstream::binary | std::ifstream::in);
	if (!ifs.good()){
	    throw std::runtime_error(std::string("Fail to open ")+weBankPath);
	}

	// get the number of we data
	std::streampos numEleS, numEleE;
	long int numEle;
	numEleS = ifs.tellg();
	ifs.seekg(0, std::ios::end);
	numEleE = ifs.tellg();
	numEle  = (numEleE-numEleS)/sizeof(real_t);
	ifs.seekg(0, std::ios::beg);
	
	// read in the data
	m_weBank = Cpu::real_vector(numEle, 0);
	real_t tempVal;
	std::vector<real_t> tempVec;
	for (unsigned int i = 0; i<numEle; i++){
	    ifs.read ((char *)&tempVal, sizeof(real_t));
	    tempVec.push_back(tempVal);
	}
	thrust::copy(tempVec.begin(), tempVec.end(), m_weBank.begin());
	printf("Initialize embedded vectors");
	printf("\n\tRead %d vectors, of dimension %d\n", (int)numEle/dim, dim);
	ifs.close();
	

*/
