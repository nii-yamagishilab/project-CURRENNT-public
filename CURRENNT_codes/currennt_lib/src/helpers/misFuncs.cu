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

#include "./misFuncs.hpp"
#include <string>
#include <vector>
#include <boost/foreach.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>


#include "../Configuration.hpp"
#include <boost/random/normal_distribution.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#include <boost/random/mersenne_twister.hpp>

#include <stdexcept>
#include <fstream>

/* ***** Functions for string process ***** */
namespace misFuncs {
    
void ParseStrOpt(const std::string stringOpt, std::vector<std::string> &optVec,
		 const std::string para){
    std::vector<std::string> tempArgs;
    
    if (stringOpt.size()==0){
	optVec.clear();
	return;
    }
    
    boost::split(tempArgs, stringOpt, boost::is_any_of(para));
    for (int i =0 ; i<tempArgs.size(); i++)
	optVec.push_back(tempArgs[i]);
    return;
}

void ParseIntOpt(const std::string stringOpt, Cpu::int_vector &optVec){
    std::vector<std::string> tempArgs;
    std::vector<std::string> tempArgs2;
    std::vector<int> tmpresult;
    
    if (stringOpt.size()==0){
	optVec.clear();
	return;
    }
    
    boost::split(tempArgs, stringOpt, boost::is_any_of("_"));
    for (int i =0 ; i<tempArgs.size(); i++){
	boost::split(tempArgs2, tempArgs[i], boost::is_any_of("*"));
	if (tempArgs2.size() == 2){
	    int cnt = boost::lexical_cast<int>(tempArgs2[0]);
	    for (int j = 0; j < cnt; j++)
		tmpresult.push_back(boost::lexical_cast<int>(tempArgs2[1]));
	}else{
	    tmpresult.push_back(boost::lexical_cast<int>(tempArgs[i]));
	}
    }
    optVec.resize(tmpresult.size(), 0.0);
    for (int i=0;i<optVec.size();i++)
	optVec[i] = tmpresult[i];
}

void ParseFloatOpt(const std::string stringOpt, Cpu::real_vector &optVec){
    std::vector<std::string> tempArgs;
    std::vector<std::string> tempArgs2;
    std::vector<real_t> tmpresult;

    if (stringOpt.size()==0){
	optVec.clear();
	return;
    }
    
    boost::split(tempArgs, stringOpt, boost::is_any_of("_"));
    for (int i =0 ; i<tempArgs.size(); i++){
	boost::split(tempArgs2, tempArgs[i], boost::is_any_of("*"));
	if (tempArgs2.size() == 2){
	    int cnt = boost::lexical_cast<int>(tempArgs2[0]);
	    for (int j = 0; j < cnt; j++)
		tmpresult.push_back(boost::lexical_cast<real_t>(tempArgs2[1]));
	}else{
	    tmpresult.push_back(boost::lexical_cast<real_t>(tempArgs[i]));
	}
    }
    optVec.resize(tmpresult.size(), 0.0);
    for (int i=0;i<optVec.size();i++)
	optVec[i] = tmpresult[i];
}

int SumCpuIntVec(Cpu::int_vector &temp){
    int result = 0;
    for (int i = 0; i<temp.size(); i++)
	result += temp[i];
    return result;
}

int MaxCpuIntVec(Cpu::int_vector &temp){
    if (temp.size()>0){
	int max = temp[0];
	for (int i = 1; i<temp.size(); i++)
	    if (temp[i] > max){
		max = temp[i];
	    }
	return max;
    }else{
	printf("Input vector is void");
	return 0;
    }
}

void PrintVecBinD(Gpu::real_vector &temp){
    std::string filePath("./temp.bin");
    std::ofstream ofs(filePath.c_str(), std::ofstream::binary);
    if (!ofs.good()){
	std::cout << "Fail to open " << filePath << std::endl;
	return;
    }else{
	Cpu::real_vector tempBuf = temp;
	std::vector<real_t> tempVec(tempBuf.begin(), tempBuf.end());
	for(int i=0; i<tempVec.size(); i++){
	    ofs.write((char *)&(tempVec[i]), sizeof(real_t));
	}
	ofs.close();
	std::cout << "Save to " << filePath << std::endl;
    }
}

void PrintVecBinD(Gpu::complex_vector &temp){
    std::string filePath("./temp.bin");
    std::ofstream ofs(filePath.c_str(), std::ofstream::binary);
    if (!ofs.good()){
	std::cout << "Fail to open " << filePath << std::endl;
	return;
    }else{
	Cpu::complex_vector tempBuf = temp;
	std::vector<complex_t> tempVec(tempBuf.begin(), tempBuf.end());
	for(int i=0; i<tempVec.size(); i++){
	    ofs.write((char *)&(tempVec[i].x), sizeof(real_t));
	    ofs.write((char *)&(tempVec[i].y), sizeof(real_t));
	}
	ofs.close();
	std::cout << "Save to " << filePath << std::endl;
    }
}

void PrintVecBinD(Gpu::int_vector &temp){
    std::string filePath("./temp.bin");
    std::ofstream ofs(filePath.c_str(), std::ofstream::binary);
    if (!ofs.good()){
	std::cout << "Fail to open " << filePath << std::endl;
	return;
    }else{
	Cpu::real_vector tempBuf = temp;
	std::vector<real_t> tempVec(tempBuf.begin(), tempBuf.end());
	for(int i=0; i<tempVec.size(); i++){
	    ofs.write((char *)&(tempVec[i]), sizeof(real_t));
	}
	ofs.close();
	std::cout << "Save to " << filePath << std::endl;
    }
}

void PrintVecBinD(Gpu::real_vector *temp){
    std::string filePath("./temp.bin");
    std::ofstream ofs(filePath.c_str(), std::ofstream::binary);
    if (!ofs.good()){
	std::cout << "Fail to open " << filePath << std::endl;
	return;
    }else{
	Cpu::real_vector tempBuf = *temp;
	std::vector<real_t> tempVec(tempBuf.begin(), tempBuf.end());
	for(int i=0; i<tempVec.size(); i++){
	    ofs.write((char *)&(tempVec[i]), sizeof(real_t));
	}
	ofs.close();
	std::cout << "Save to " << filePath << std::endl;
    }
}

void PrintVecBinD(Gpu::complex_vector *temp){
    std::string filePath("./temp.bin");
    std::ofstream ofs(filePath.c_str(), std::ofstream::binary);
    if (!ofs.good()){
	std::cout << "Fail to open " << filePath << std::endl;
	return;
    }else{
	Cpu::complex_vector tempBuf = *temp;
	std::vector<complex_t> tempVec(tempBuf.begin(), tempBuf.end());
	for(int i=0; i<tempVec.size(); i++){
	    ofs.write((char *)&(tempVec[i].x), sizeof(real_t));
	    ofs.write((char *)&(tempVec[i].y), sizeof(real_t));
	}
	ofs.close();
	std::cout << "Save to " << filePath << std::endl;
    }
}

void PrintVecBinD(Gpu::int_vector *temp){
    std::string filePath("./temp.bin");
    std::ofstream ofs(filePath.c_str(), std::ofstream::binary);
    if (!ofs.good()){
	std::cout << "Fail to open " << filePath << std::endl;
	return;
    }else{
	Cpu::real_vector tempBuf = *temp;
	std::vector<real_t> tempVec(tempBuf.begin(), tempBuf.end());
	for(int i=0; i<tempVec.size(); i++){
	    ofs.write((char *)&(tempVec[i]), sizeof(real_t));
	}
	ofs.close();
	std::cout << "Save to " << filePath << std::endl;
    }
}

void PrintVecBinH(Cpu::real_vector &temp){
    std::string filePath("./temp.bin");
    std::ofstream ofs(filePath.c_str(), std::ofstream::binary);
    if (!ofs.good()){
	std::cout << "Fail to open " << filePath << std::endl;
	return;
    }else{
	std::vector<real_t> tempVec(temp.begin(), temp.end());
	for(int i=0; i<tempVec.size(); i++){
	    ofs.write((char *)&(tempVec[i]), sizeof(real_t));
	}
	ofs.close();
	std::cout << "Save to " << filePath << std::endl;
    }
}

void PrintVecBinH(Cpu::complex_vector &temp){
    std::string filePath("./temp.bin");
    std::ofstream ofs(filePath.c_str(), std::ofstream::binary);
    if (!ofs.good()){
	std::cout << "Fail to open " << filePath << std::endl;
	return;
    }else{
	std::vector<complex_t> tempVec(temp.begin(), temp.end());
	for(int i=0; i<tempVec.size(); i++){
	    ofs.write((char *)&(tempVec[i].x), sizeof(real_t));
	    ofs.write((char *)&(tempVec[i].y), sizeof(real_t));
	}
	ofs.close();
	std::cout << "Save to " << filePath << std::endl;
    }
}

void PrintVecBinH(Cpu::int_vector &temp){
    std::string filePath("./temp.bin");
    std::ofstream ofs(filePath.c_str(), std::ofstream::binary);
    if (!ofs.good()){
	std::cout << "Fail to open " << filePath << std::endl;
	return;
    }else{
	std::vector<real_t> tempVec(temp.begin(), temp.end());
	for(int i=0; i<tempVec.size(); i++){
	    ofs.write((char *)&(tempVec[i]), sizeof(real_t));
	}
	ofs.close();
	std::cout << "Save to " << filePath << std::endl;
    }
}


void AppendVecBin(Gpu::real_vector &temp){
    std::string filePath("./temp.bin");
    std::ofstream ofs(filePath.c_str(), std::ofstream::binary | std::ofstream::app);
    if (!ofs.good()){
	std::cout << "Fail to open " << filePath << std::endl;
	return;
    }else{
	Cpu::real_vector tempBuf = temp;
	std::vector<real_t> tempVec(tempBuf.begin(), tempBuf.end());
	for(int i=0; i<tempVec.size(); i++){
	    ofs.write((char *)&(tempVec[i]), sizeof(real_t));
	}
	ofs.close();
	std::cout << "Save to " << filePath << std::endl;
    }
}

void AppendVecBin(Gpu::int_vector &temp){
    std::string filePath("./temp.bin");
    std::ofstream ofs(filePath.c_str(), std::ofstream::binary | std::ofstream::app);
    if (!ofs.good()){
	std::cout << "Fail to open " << filePath << std::endl;
	return;
    }else{
	Cpu::real_vector tempBuf = temp;
	std::vector<real_t> tempVec(tempBuf.begin(), tempBuf.end());
	for(int i=0; i<tempVec.size(); i++){
	    ofs.write((char *)&(tempVec[i]), sizeof(real_t));
	}
	ofs.close();
	std::cout << "Save to " << filePath << std::endl;
    }
}

void AppendVecBin(Cpu::real_vector &temp){
    std::string filePath("./temp.bin");
    std::ofstream ofs(filePath.c_str(), std::ofstream::binary | std::ofstream::app);
    if (!ofs.good()){
	std::cout << "Fail to open " << filePath << std::endl;
	return;
    }else{
	std::vector<real_t> tempVec(temp.begin(), temp.end());
	for(int i=0; i<tempVec.size(); i++){
	    ofs.write((char *)&(tempVec[i]), sizeof(real_t));
	}
	ofs.close();
	std::cout << "Save to " << filePath << std::endl;
    }
}

void AppendVecBin(Cpu::int_vector &temp){
    std::string filePath("./temp.bin");
    std::ofstream ofs(filePath.c_str(), std::ofstream::binary | std::ofstream::app);
    if (!ofs.good()){
	std::cout << "Fail to open " << filePath << std::endl;
	return;
    }else{
	std::vector<real_t> tempVec(temp.begin(), temp.end());
	for(int i=0; i<tempVec.size(); i++){
	    ofs.write((char *)&(tempVec[i]), sizeof(real_t));
	}
	ofs.close();
	std::cout << "Save to " << filePath << std::endl;
    }
}

    
real_t GetRandomNumber(){
    static boost::mt19937 *gen = NULL;
    if (!gen) {
	gen = new boost::mt19937;
	gen->seed(Configuration::instance().randomSeed());
    }
    boost::random::uniform_real_distribution<real_t> dist(0, 1);
    return dist(*gen); 
}


int flagUpdateDiscriminator(const int epoch, const int frac){
    /*if (epoch % 2){
	return (frac % 2) == 0;
    }else{
	return (frac % 2) == 1;
	}*/
    return ((frac + 1) % 3);
}

bool closeToZero(const real_t t1, const real_t lowBound, const real_t upBound)
{
    return ((t1 > lowBound) && (t1 < upBound));
}

int ReadRealData(const std::string dataPath, Cpu::real_vector &data)
{
    // 
    std::ifstream ifs(dataPath.c_str(), std::ifstream::binary | std::ifstream::in);
    if (!ifs.good())
	throw std::runtime_error(std::string("Fail to open ")+dataPath);
    
    // get the number of we data
    std::streampos numEleS, numEleE;
    long int numEle;
    numEleS = ifs.tellg();
    ifs.seekg(0, std::ios::end);
    numEleE = ifs.tellg();
    numEle  = (numEleE-numEleS)/sizeof(real_t);
    ifs.seekg(0, std::ios::beg);
    
    // read in the data
    if (data.size() < numEle)
	data.resize(numEle, 0);
    
    real_t tempVal;
    for (unsigned int i = 0; i<numEle; i++){
	ifs.read ((char *)&tempVal, sizeof(real_t));
	data[i] = tempVal;
    }
    //thrust::copy(tempVec.begin(), tempVec.end(), data.begin());
    ifs.close();
    return numEle;
}

int getResoLength(const int maxSeqLength, const int timeResolution, const int parallel)
{
    if (timeResolution == 1)
	return maxSeqLength;
    else{
	int temp = maxSeqLength / parallel;
	return (temp / timeResolution + ((temp % timeResolution>0) ? 1:0)) * parallel;
    }
}

}
