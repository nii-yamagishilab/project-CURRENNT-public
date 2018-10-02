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
 *****************************************************************************//*

*/


#ifdef _MSC_VER
#   pragma warning (disable: 4244) // thrust/iterator/iterator_adaptor.h(121): warning C4244: '+=' : conversion from '__int64' to 'int', possible loss of data
#endif


#include "MDNLayer.hpp"
#include "MDNUnit.hpp"
#include "../helpers/getRawPointer.cuh"
#include "../helpers/Matrix.hpp"
#include "../helpers/min.cuh"
#include "../helpers/max.cuh"
#include "../helpers/safeExp.cuh"
#include "../helpers/JsonClasses.hpp"
#include "../helpers/misFuncs.hpp"

#include "../activation_functions/Tanh.cuh"
#include "../activation_functions/Logistic.cuh"
#include "../activation_functions/Identity.cuh"
#include "../activation_functions/Relu.cuh"

#include "../Configuration.hpp"
#include "../MacroDefine.hpp"

#include <boost/foreach.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>

#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/functional.h>
#include <sstream>
#include <fstream>
#include <cmath>


#define MIXTUREDYN_INITVARIANCE 0.01


namespace internal {
namespace {
    
    struct seqGradWeight
    {
	real_t *seqWeight;
	int     dataDim;  
	const char *patTypes;
	__host__ __device__ void operator() (const thrust::tuple<real_t&, int> &t) const
	{
	    int timeIdx   = t.get<1>() / dataDim;
	    if (patTypes != NULL && patTypes[timeIdx] == PATTYPE_NONE)
		return;
	    t.get<0>() = t.get<0>() * seqWeight[timeIdx];
	}
    };
    

}
}

namespace layers {
    
    Cpu::int_vector parseSecondOutputOpt(const std::string options)
    {
	Cpu::int_vector tmp;
	tmp.clear();
	if (options.size()){
	    for (int i = 0; i<options.size(); i++)
		tmp.push_back(options[i] - '0');
	}
	return tmp;
    }


    int parseMDNDyn(const int tmpTrainableFlag)
    {
	// A function used for historical reason
	// This function coverts the old mdnDyn options into the 
	//  the correct order of filter

	    // the original scheme
	    // 0 : non-trainable mixture
	    // 1 : 1st AR, time axis
	    // 2 : dynamic AR, parameter predicted by network
	    // 3 : 2st AR, time axis
	    // 4-5: 1-2 AR, dimension axis
	    // 6-7: 1-2 AR, dimension and time axis
	    // else: AR, time axis

	if (tmpTrainableFlag < 0){
	    printf("The value in --mdnDyn can only be [0-9]");
	    throw std::runtime_error("Error configuration --mdnDyn");
	}else if (tmpTrainableFlag <= 3){
	    // tmpTrainableFlag = 1 or 2
	    return (tmpTrainableFlag==1)?(1):(tmpTrainableFlag-1);
	}else if (tmpTrainableFlag <= 5){
	    // AR along the dimension axis
	    //dynDirection = MDNUNIT_TYPE_1_DIRECD;
	    return tmpTrainableFlag - 3;
	}else if (tmpTrainableFlag <= 7){
	    // AR long both time and dimension axes
	    // dynDirection = MDNUNIT_TYPE_1_DIRECB;
	    return tmpTrainableFlag - 5;
	}else{    
	    return tmpTrainableFlag - 5;
	}
    }
    
    /********************************************************
     MDNLayer

    *******************************************************/
    // definition of the MDN layer
    template <typename TDevice>
    MDNLayer<TDevice>::MDNLayer(const helpers::JsonValue &layerChild, 
				const helpers::JsonValue &weightsSection, 
				Layer<TDevice> &precedingLayer,
				int maxSeqLength,
				int layerID)
	: PostOutputLayer<TDevice>(layerChild, precedingLayer, -1, maxSeqLength, layerID)
    {
        const Configuration &config = Configuration::instance();
	
	if (this->precedingLayer().getSaveMemoryFlag())
	    throw std::runtime_error("The layer before MDN is reduced in mem");
	
        // parse the MDN vector
	int numEle;   // number of MDNUnit
	int unitS;    // index of start dimension (w.r.t output vector of preceding layer)
	int unitE;    // index of end dimension (w.r.t output vector of preceding layer)
	int mdnType;  // type of MDNUnit
	int unitSOut; // index of start dimension (w.r.t target feature vector)
	int unitEOut; // index of end dimension (w.r.t target feature vector)
	
	int outputSize = 0; // temporary buffer for layer size check

	// initialize
	m_mdnParaDim = 0;          // 
	m_flagNewAROption = false; // 
	
	// build the MDN unit
	MDNUnit<TDevice> *mdnUnit;
	
	// get the flag for variance tying
	m_tieVarMDNUnit   = config.getTiedVariance();
	
	// get the generation paramter (during generation stage)
	m_genPara         = config.mdnPara();

	// get the string for secondoutput configuration
	m_secondOutputOpt = parseSecondOutputOpt(config.secondOutputOpt());
	m_secondOutputDim = 0;
	
	// tmp Buff to read the configuration
	Cpu::int_vector flagTieVariance;
	Cpu::int_vector flagTrainable;
	Cpu::int_vector flagTrainable_arg;
	Cpu::int_vector optTanhAutoReg;
	Cpu::int_vector optTanhAutoReg_arg;
	Cpu::int_vector mdnSoftMaxGenMethods;
	Cpu::int_vector mdnSoftMaxFBMethods;
		
	flagTieVariance.clear();
	flagTrainable.clear();
	optTanhAutoReg.clear();
	mdnSoftMaxGenMethods.clear();
	mdnSoftMaxFBMethods.clear();
	
	/******************* Read config ********************/
	// 
	m_uvSigmoidStr      = ((layerChild->HasMember("uvSigmoidSoftmax")) ? 
			       ((*layerChild)["uvSigmoidSoftmax"].GetString()) : (""));
	m_oneSidedSmoothing = ((layerChild->HasMember("oneSidedSmoothGAN")) ? 
			       ((*layerChild)["oneSidedSmoothGAN"].GetInt()) : (0));
	m_mdnConfigPath     = ((layerChild->HasMember("mdnConfig")) ? 
			       ((*layerChild)["mdnConfig"].GetString()) : (""));
	m_conValSigStr      = ((layerChild->HasMember("continuousValuedSigmoid")) ? 
			       ((*layerChild)["continuousValuedSigmoid"].GetString()) : (""));
	m_flagNewAROption   = ((layerChild->HasMember("FlagNewAROption")) ? 
			       ((*layerChild)["FlagNewAROption"].GetBool()) : (m_flagNewAROption));
	
	// Block20170702x05
	
	// read in the configuration from .autosave
	// I should put this to the layerChild section
	if (weightsSection.isValid() && weightsSection->HasMember(this->name().c_str())) {
	    const rapidjson::Value &weightsChild = (*weightsSection)[this->name().c_str()];
            if (!weightsChild.IsObject())
                throw std::runtime_error(std::string("Weights section for layer '") + 
					 this->name() + "' is not an object");
	    if (!weightsChild.HasMember("config") || !weightsChild["config"].IsArray())
                throw std::runtime_error(std::string("Missing array 'config/") + 
					 this->name() + "/config'");
	    
	    // read in the mdnConfig vector
            const rapidjson::Value &inputWeightsChild    = weightsChild["config"];
            m_mdnConfigVec.reserve(inputWeightsChild.Size());;
	    for (rapidjson::Value::ConstValueIterator it = inputWeightsChild.Begin(); 
		 it != inputWeightsChild.End(); ++it)
                m_mdnConfigVec.push_back(static_cast<real_t>(it->GetDouble()));
            numEle = m_mdnConfigVec[0];
	    
	    // read in the flagTieVariance vector
	    if (weightsChild.HasMember("tieVarianceFlag") && 
		weightsChild["tieVarianceFlag"].IsArray()){
		const rapidjson::Value &BuftieVarianceFlag = weightsChild["tieVarianceFlag"];
		for (rapidjson::Value::ConstValueIterator it = BuftieVarianceFlag.Begin(); 
		     it != BuftieVarianceFlag.End(); ++it)
		    flagTieVariance.push_back(static_cast<int>(it->GetInt()));
	    }

	    // read in the trainable type vector
	    if (weightsChild.HasMember("trainableFlag") && 
		weightsChild["trainableFlag"].IsArray()){
		const rapidjson::Value &BuftieVarianceFlag = weightsChild["trainableFlag"];
		for (rapidjson::Value::ConstValueIterator it = BuftieVarianceFlag.Begin(); 
		     it != BuftieVarianceFlag.End(); ++it)
		    flagTrainable.push_back(static_cast<int>(it->GetInt()));
	    }
	    
	    // read in the trainable type vector
	    if (weightsChild.HasMember("tanhAutoReg") && 
		weightsChild["tanhAutoReg"].IsArray()){
		const rapidjson::Value &BuftieVarianceFlag = weightsChild["tanhAutoReg"];
		for (rapidjson::Value::ConstValueIterator it = BuftieVarianceFlag.Begin(); 
		     it != BuftieVarianceFlag.End(); ++it)
		    optTanhAutoReg.push_back(static_cast<int>(it->GetInt()));
	    }

	    // read in the secondOutputOpt
	    if (weightsChild.HasMember("feedbackOpt") &&
		weightsChild["feedbackOpt"].IsArray() && m_secondOutputOpt.size()==0){
		m_secondOutputOpt.clear();
		const rapidjson::Value &BuftieVarianceFlag = weightsChild["feedbackOpt"];
		for (rapidjson::Value::ConstValueIterator it = BuftieVarianceFlag.Begin(); 
		     it != BuftieVarianceFlag.End(); ++it)
		    m_secondOutputOpt.push_back(static_cast<int>(it->GetInt()));
	    }

	    // clear configPath
	    // config has been written into network.jsn/autosave
	    // no need to read mdnConfig anymore
	    m_mdnConfigPath = "";
				      
	// read in the configuration from mdn_config (binary file)
        }else{
	    
	    if (m_mdnConfigPath.size() == 0)
		m_mdnConfigPath = config.mdnFlagPath();
	    
	    std::ifstream ifs(m_mdnConfigPath.c_str(), 
			      std::ifstream::binary | std::ifstream::in);
	    if (!ifs.good())
		throw std::runtime_error(std::string("Can't open MDNConfig:" + m_mdnConfigPath));
	    
	    std::streampos numEleS, numEleE;
	    numEleS = ifs.tellg();
	    ifs.seekg(0, std::ios::end);
	    numEleE = ifs.tellg();
	    // get the total number of parameter
	    long int tmpnumEle  = (numEleE-numEleS)/sizeof(real_t);
	    ifs.seekg(0, std::ios::beg);
	
	    // get the total number of MDNUnit
	    real_t tempVal;
	    ifs.read((char *)&tempVal, sizeof(real_t));
	    numEle = (int)tempVal;                      
	    if (tmpnumEle != (numEle*5+1)){
		throw std::runtime_error("Number of parameter != 1st parameter * 5 + 1");
	    }	    

	    // get the configuration for each MDNUnit
	    m_mdnConfigVec.resize(1+numEle*5, 0.0);
	    m_mdnConfigVec[0] = (real_t)numEle;
	    for (int i=0; i<numEle; i++){
		ifs.read((char *)&tempVal, sizeof(real_t));
		m_mdnConfigVec[1+i*5] = tempVal;
		ifs.read((char *)&tempVal, sizeof(real_t));
		m_mdnConfigVec[2+i*5] = tempVal;
		ifs.read((char *)&tempVal, sizeof(real_t));
		m_mdnConfigVec[3+i*5] = tempVal;
		ifs.read((char *)&tempVal, sizeof(real_t));
		m_mdnConfigVec[4+i*5] = tempVal;
		ifs.read((char *)&tempVal, sizeof(real_t));
		m_mdnConfigVec[5+i*5] = tempVal;
	    }
	    ifs.close();
	    m_mdnConfigPath.clear();
	}

	// check configuration
	if (m_mdnConfigVec.size()  != numEle*5+1)
	    throw std::runtime_error("Error in reading the configuration of MDN");
	
	if (flagTieVariance.size() != flagTrainable.size() ||
	    (flagTieVariance.size() >0 && flagTieVariance.size() != numEle))
	    throw std::runtime_error("Error in tieVarianceFlag and trainableFlag (in model file)");
	
	if (m_secondOutputOpt.size()>0 && m_secondOutputOpt.size() != numEle)
	    throw std::runtime_error("Error in feedbackOpt. Length unequal to #. MDNUnits");
	else if(m_secondOutputOpt.size() == 0){
	    for (int i =0; i< numEle; i++) m_secondOutputOpt.push_back(MDNUNIT_FEEDBACK_OPT_0);
	}

	// Load the configuration for AR models
	
	if (config.mdnDyn().size() > 0){
	    // To be compatible with old configuration options
	    
	    // mdnDyn: read Trainable from input argument
	    if (config.mdnDyn().size() != numEle){
		// num1_num2_num3 format
		std::vector<std::string> tempArgs;
		boost::split(tempArgs, config.mdnDyn(), boost::is_any_of("_"));
		if (tempArgs.size() != numEle){
		    printf("mdnDyn length: %d, MDNUnits %d\n", 
			   (int)tempArgs.size(), (int)numEle);
		    throw std::runtime_error("Error in mdnDyn");
		}
		flagTrainable_arg.resize(config.mdnDyn().size(), 0);
		for (int i=0; i < tempArgs.size(); i++)
		    flagTrainable_arg[i] = boost::lexical_cast<int>(tempArgs[i]);
		
	    }else{
		flagTrainable_arg.resize(config.mdnDyn().size(), 0);
		for (int i=0; i < config.mdnDyn().size(); i++)
		    flagTrainable_arg[i] = config.mdnDyn()[i] - '0';
	    }

	    // tanhAutoReg: read Trainable from input argument
	    if (config.tanhAutoregressive().size() > 0){
		optTanhAutoReg_arg.resize(config.mdnDyn().size(), 0);
		if (config.tanhAutoregressive().size() != numEle){
		    for (int i=0; i < config.tanhAutoregressive().size(); i++)
			optTanhAutoReg_arg[i] = config.tanhAutoregressive()[0] - '0';
		}else{
		    for (int i=0; i < config.tanhAutoregressive().size(); i++)
			optTanhAutoReg_arg[i] = config.tanhAutoregressive()[i] - '0';
		}
	    }else{
		// default, make it all nontrainable unit
		optTanhAutoReg_arg.resize(numEle, 0);
	    }


	    
	}else if (config.arOrder().size() > 0){
	    // read new options
	    m_flagNewAROption = true;
	    
	    if (config.arOrder().size() != numEle){
		// num1_num2_num3 format
		std::vector<std::string> tempArgs;
		boost::split(tempArgs, config.arOrder(), boost::is_any_of("_"));
		if (tempArgs.size() != numEle){
		    printf("AROrder length: %d, #.MDNUnits %d\n", 
			   (int)tempArgs.size(), (int)numEle);
		    throw std::runtime_error("AROrder length is different from MDNUnits number");
		}
		flagTrainable_arg.resize(tempArgs.size(), 0);
		for (int i=0; i < tempArgs.size(); i++)
		    flagTrainable_arg[i] = boost::lexical_cast<int>(tempArgs[i]);
		
	    }else{
		flagTrainable_arg.resize(config.arOrder().size(), 0);
		for (int i=0; i < config.arOrder().size(); i++)
		    flagTrainable_arg[i] = config.arOrder()[i] - '0';
	    }

	    
	    if (config.arConfig().size() > 0){
		optTanhAutoReg_arg.resize(config.arOrder().size(), 0);
		if (config.arConfig().size() != numEle){

		    std::vector<std::string> tempArgs;
		    boost::split(tempArgs, config.arConfig(), boost::is_any_of("_"));
		    if (tempArgs.size() != numEle){
			printf("AROrConfig length: %d, #.MDNUnits %d\n", 
			       (int)tempArgs.size(), (int)numEle);
			throw std::runtime_error("ARConfig length is different from MDNUnits number");
		    }
		    optTanhAutoReg_arg.resize(tempArgs.size(), 0);
		    for (int i=0; i < tempArgs.size(); i++)
			optTanhAutoReg_arg[i] = boost::lexical_cast<int>(tempArgs[i]);

		}else{
		    for (int i=0; i < config.arConfig().size(); i++)
			optTanhAutoReg_arg[i] = config.arConfig()[i] - '0';
		}
	    }else{
		optTanhAutoReg_arg.resize(numEle, 0);
	    }
	    
	}else{
	    // default, make it all nontrainable unit
	    flagTrainable_arg.resize(numEle, 0);
	    optTanhAutoReg_arg.resize(numEle, 0);
	}

	
	

	if (config.mdnSoftMaxGenMethods().size()){
	    // read the sampling method for softmax units
	    misFuncs::ParseIntOpt(config.mdnSoftMaxGenMethods(), mdnSoftMaxGenMethods);
	    if (mdnSoftMaxGenMethods.size() != numEle)
		throw std::runtime_error("mdnSoftMaxGenMethods length != #MDNUnits");
	}
	if (config.scheduleSampOpts().size()){
	    // read the feedback method for softmax units
	    misFuncs::ParseIntOpt(config.scheduleSampOpts(), mdnSoftMaxFBMethods);
	    if (mdnSoftMaxFBMethods.size() != numEle)
		throw std::runtime_error("mdnSoftMaxGenMethods length != #MDNUnits");
	}

	printf("\n");	
	int weightsNum = 0;
	this->m_trainable = false;
	
	/******************** Create MDNUnit **************************/
	// create the MDNUnits
	for (int i=0; i<numEle; i++){
	    unitS    = (int)m_mdnConfigVec[1+i*5];  // start dim in previous layer's output
	    unitE    = (int)m_mdnConfigVec[2+i*5];  // end dim in previous layer's output
	    unitSOut = (int)m_mdnConfigVec[3+i*5];  // start dim in target feature vector
	    unitEOut = (int)m_mdnConfigVec[4+i*5];  // end dimension in target feature vector
	    mdnType  = (int)m_mdnConfigVec[5+i*5];  // MDNUnit type
	    
	    // binomial distribution (parametrized as sigmoid function)
	    if (mdnType==MDN_TYPE_SIGMOID){
		bool tmpFlag = (m_conValSigStr[0]=='y');
		mdnUnit = new MDNUnit_sigmoid<TDevice>(unitS, unitE, unitSOut, unitEOut, mdnType, 
						       precedingLayer, this->size(),
						       MDNUNIT_TYPE_0, m_secondOutputOpt[i],
						       tmpFlag);
		m_mdnParaDim += (unitE - unitS);
		outputSize += (unitE - unitS);
		printf("\tMDN sigmoid (conVal [%d])\n", (int)tmpFlag);
		
	    // multi-nomial distribution (parameterized by softmax function)
	    }else if(mdnType==MDN_TYPE_SOFTMAX){
		int uvSig = MDNUNIT_SOFTMAX_FLAT;
		int genMethod = config.mdnSoftMaxGenMethod();
		int fbMethod  = config.scheduleSampOpt();
		
		if (mdnSoftMaxGenMethods.size())
		    genMethod = mdnSoftMaxGenMethods[i];
		if (mdnSoftMaxFBMethods.size())
		    fbMethod  = mdnSoftMaxFBMethods[i];
		    
		if (i >= m_uvSigmoidStr.size())
		    uvSig = MDNUNIT_SOFTMAX_FLAT;  // no specification, as flat-softmax
		else if (m_uvSigmoidStr[i]=='s')
		    uvSig = MDNUNIT_SOFTMAX_UBLIND;// flat softmax blind to U/V
		else if (m_uvSigmoidStr[i]=='y')
		    uvSig = MDNUNIT_SOFTMAX_UV;    // hierarchical U/V
		else
		    uvSig = MDNUNIT_SOFTMAX_FLAT;  // default, flat-softmax
		
		mdnUnit = new MDNUnit_softmax<TDevice>(unitS, unitE, unitSOut, unitEOut, mdnType, 
						       precedingLayer, this->size(), genMethod,
						       fbMethod,
						       uvSig, config.mdnUVSigThreshold(),
						       MDNUNIT_TYPE_0, m_secondOutputOpt[i]);
		m_mdnParaDim += (unitE - unitS);
		outputSize += 1;
		
	    // Gaussian mixture distribution
	    }else if(mdnType > 0){
		
		// if the model (.autosave) has specified it, ignores the arguments
		bool tmpTieVarianceFlag = ((flagTieVariance.size()>0) ?
					   (flagTieVariance[i]>0)     :
					   (m_tieVarMDNUnit));

		// the Trainable flag, either by flagTrainable in model or arguments
		// if the model (.autosave) has specified it, ignores the arguments
		int  tmpTrainableFlag   = ((flagTrainable.size()>0)   ?
					   (flagTrainable[i])       :
					   (flagTrainable_arg[i]));
		
		// tanhAutoReg type
		int  tmpTanhAutoReg     = ((optTanhAutoReg.size()>0) ? 
					   (optTanhAutoReg[i])       :
					   (optTanhAutoReg_arg[i]));
		
		printf("\tMDN mixture: trainable: %2d, tieVariance %d, #parameter ", 
		       tmpTrainableFlag, tmpTieVarianceFlag);
		int featureDim;   
		int thisWeightNum; 
		
		if (tmpTrainableFlag == MDNUNIT_TYPE_0){
		    // baseline non-trainable units
		    featureDim  = unitEOut - unitSOut;
		    mdnUnit     = new MDNUnit_mixture<TDevice>(
					unitS, unitE, unitSOut, unitEOut, mdnType, 
					featureDim, precedingLayer, this->size(), 
					tmpTieVarianceFlag,
					MDNUNIT_TYPE_0,
					m_secondOutputOpt[i]);
		    printf("%d\n", 0);

		    if (tmpTieVarianceFlag){
			// K mixture weight, K*Dim mean, K*1 variance. 
			outputSize += ((unitE - unitS) - 2 * mdnType) / mdnType;
		    }else{
			// K mixture weight, K*Dim mean, K*Dim variance. 
			outputSize += ((unitE - unitS) - mdnType) / ( 2 * mdnType);
		    }
		    
		}else{
		    // only AR dependency over the time dimension
		    int dynDirection = MDNUNIT_TYPE_1_DIRECT;
		    
		    // the correct AR dependency order
		    int lookBackStep = ((m_flagNewAROption)?(tmpTrainableFlag):
					parseMDNDyn(tmpTrainableFlag));
		    
		    // trainable units
		    if (tmpTanhAutoReg == MDNARRMDN_ARDYNMAIC){
			// AR dynamic model

			// [unitS, unitEforMDN] specifies the dimension range for MDN_mixture
			// [unitEforMDN, unitE] specifies the dimension range for AR parameters
			int unitEforMDN = unitE - (lookBackStep + 1) * (unitEOut - unitSOut);

			if (unitEforMDN < 0){
			    throw std::runtime_error("Incorrect mdn.config for Dynamic SAR");
			}
			mdnUnit = new MDNUnit_mixture_dynSqr<TDevice>(
					unitS,    unitEforMDN, unitE,
					unitSOut, unitEOut,
					mdnType, 
					precedingLayer, this->size(), 
					tmpTieVarianceFlag,
					lookBackStep, tmpTrainableFlag);
			printf("%d, with dynamic link\n", 0);
			
			if (tmpTieVarianceFlag){
			    // K mixture weight, K*Dim mean, K*1 variance, Dim a, Dim b
			    outputSize += ((unitE - unitS) - 2 * mdnType -
					   (lookBackStep+1) * (unitEOut - unitSOut))/mdnType;
			}else{
			    // K mixture weight, K*Dim mean, K*Dim variance, Dim a, Dim b
			    outputSize += ((unitE - unitS) - mdnType -
					   (lookBackStep+1)*(unitEOut - unitSOut))/(2*mdnType);
			}
			
		    }else{
			// AR time-invariant model
			
			bool realPole = true;
			// for casecade of complex filters
			if (tmpTanhAutoReg==MDNARRMDN_CASECADECOMPLEX){
			    if (lookBackStep < 2){
				throw std::runtime_error("AR complex poles requires AR order>1");
			    }else if ((lookBackStep % 2) == 1){
				// if the order is an odd number
				// there must be a real pole (1-\alphaz^-1)
				// let's make it as (1 - \alphaz^-1 - 0z^-2)
				lookBackStep = lookBackStep + 1;
				// there is one real pole in the AR
				realPole     = true; 
			    }else{
				// there is no real pole
				realPole     = false;
			    }
			}
			
			// create the trainable unit
			featureDim    = unitEOut - unitSOut;
			thisWeightNum = layers::MixtureDynWeightNum(
						featureDim, mdnType, lookBackStep, dynDirection);
			
			mdnUnit = new MDNUnit_mixture_dyn<TDevice>(
					unitS, unitE, unitSOut, unitEOut, mdnType, 
					precedingLayer, this->size(), 
					tmpTieVarianceFlag, weightsNum, thisWeightNum,
					lookBackStep, tmpTrainableFlag, dynDirection, realPole,
					tmpTanhAutoReg, m_secondOutputOpt[i]);
			weightsNum += thisWeightNum;
			this->m_trainable = true;
			
			printf("\t%5d, AR order and direction: %d %d", 
			       thisWeightNum, lookBackStep, dynDirection);
			if (tmpTanhAutoReg){
			    printf(", with AR configuration %d", tmpTanhAutoReg);
			}
			printf("\n");

			// check sanity
			if (tmpTieVarianceFlag){
			    // K mixture weight, K*Dim mean, K*1 variance. 
			    outputSize += ((unitE - unitS) - 2 * mdnType) / mdnType;
			}else{
			    // K mixture weight, K*Dim mean, K*Dim variance. 
			    outputSize += ((unitE - unitS) - mdnType) / ( 2 * mdnType);
			}
		    }
		}
		m_mdnParaDim += (unitE - unitS);
		
		if (!mdnUnit->flagValid()){
		    throw std::runtime_error("Fail to initialize mdnUnit");
		}

	    }else{
		throw std::runtime_error("mdnUnit type invalid (>0, 0, -1)");
	    }
	    // accumulate the dim for feedback (if necessary)
	    m_secondOutputDim += mdnUnit->feedBackDim();
	    
	    m_mdnUnits.push_back(boost::shared_ptr<MDNUnit<TDevice> >(mdnUnit));
	}

	/********************  check               ****************/
	printf("\tMDN layer distribution parameter number: %d\n", m_mdnParaDim);
	if (m_mdnParaDim != precedingLayer.size()){
	    printf("MDN parameter dim %d is not equal to NN output dimension %d\n", 
		   m_mdnParaDim, precedingLayer.size());
	    throw std::runtime_error("");
	}
	if (outputSize != this->size()){
	    printf("Mismatch between target dimension %d and MDN configuration %d\n", 
		   outputSize, this->size());
	    printf("Did you use --tieVariance false for untied variance of MDN?\n");
	    throw std::runtime_error("");
	}
	
	
	/********************  Initialize the weight ****************/
	cpu_real_vector weights;
	if (this->m_trainable){
	    m_trainableNum = weightsNum;
	    
	    if (weightsSection.isValid() && weightsSection->HasMember(this->name().c_str())){
		if (!weightsSection->HasMember(this->name().c_str()))
		    throw std::runtime_error(std::string("Missing weights section for layer '") + 
					     this->name() + "'");
		const rapidjson::Value &weightsChild = (*weightsSection)[this->name().c_str()];
		if (!weightsChild.IsObject())
		    throw std::runtime_error(std::string("Weights section for layer '") + 
					     this->name() + "' is not an object");

		if (!weightsChild.HasMember("input") || !weightsChild["input"].IsArray())
		    throw std::runtime_error(std::string("Missing array 'weights/") + 
					     this->name() + "/input'");
		if (!weightsChild.HasMember("bias") || !weightsChild["bias"].IsArray())
		    throw std::runtime_error(std::string("Missing array 'weights/") + 
					     this->name() + "/bias'");
		if (!weightsChild.HasMember("internal") || !weightsChild["internal"].IsArray())
		    throw std::runtime_error(std::string("Missing array 'weights/") + 
					     this->name() + "/internal'");
        
		const rapidjson::Value &inputWeightsChild    = weightsChild["input"];
		const rapidjson::Value &biasWeightsChild     = weightsChild["bias"];
		const rapidjson::Value &internalWeightsChild = weightsChild["internal"];

		if (inputWeightsChild.Size() != weightsNum)
		    throw std::runtime_error(std::string("Invalid number of weights for layer '") 
					 + this->name() + "'");

		if (biasWeightsChild.Size() != 0)
		    throw std::runtime_error(std::string("bias part should be void for layer '") 
					     + this->name() + "'");

		if (internalWeightsChild.Size() != 0)
		    throw std::runtime_error(std::string("internal weights should be void layer'")
					 + this->name() + "'");
		
		weights.reserve(inputWeightsChild.Size() + 
				biasWeightsChild.Size()  + 
				internalWeightsChild.Size());

		for (rapidjson::Value::ConstValueIterator it = inputWeightsChild.Begin(); 
		     it != inputWeightsChild.End(); ++it)
		    weights.push_back(static_cast<real_t>(it->GetDouble()));
		
	    }else {
		// No other initialization methods implemented yet
		// Add 0923, we need random initialization here
		// The problem is that, for high order filter, we need to break the symmetry of
		// of the parameter
		weights.resize(weightsNum, 0.0);	
		if(config.arRMDNInitVar() > 0.0){
		    static boost::mt19937 *gen = NULL;
		    if (!gen) {
			gen = new boost::mt19937;
			gen->seed(config.randomSeed()+101);
		    }
		    boost::random::normal_distribution<real_t> dist(0.0, config.arRMDNInitVar());
		    for (size_t i = 0; i < weights.size(); ++i)
			weights[i] = dist(*gen);
		    printf("\n\tARRMDN para initialized as Gaussian noise (var: %f)", 
			   config.arRMDNInitVar());
		}else{
		    printf("\n\tARRMDN initialized as zero");
		}
	    }
	    printf("\n\tMDN trainable mixture is used."); 
	    printf("\tThe number of trainable parameter is %d\n", weightsNum);

	    // Allocate memory for trainable weights
	    m_sharedWeights       = weights;
	    m_sharedWeightUpdates = weights;
	    // Set the intial gradients to zero
	    thrust::fill(m_sharedWeightUpdates.begin(), m_sharedWeightUpdates.end(), (real_t)0.0);

	    // link the shared weights to each trainable MDNUnits
	    BOOST_FOREACH (boost::shared_ptr<MDNUnit<TDevice> > &mdnUnit, m_mdnUnits){
		mdnUnit->linkWeight(m_sharedWeights, m_sharedWeightUpdates);
	    }
	}

	/***************** Initialize memory for vectors ***************/
	// Allocate memory for second output
	if (m_secondOutputDim > 0){

	    // for feedback link
	    Cpu::real_vector tmp(this->outputs().size()/this->size() * m_secondOutputDim, 0.0);
	    m_secondOutput = tmp;

	    // for biased probability vector used in generation
	    m_probBiasDir = config.probDataDir();
	    m_probBiasDim = config.probDataDim();
	    if (m_probBiasDim > 0 && m_probBiasDim == m_secondOutputDim){
		printf("[Probablistic bias used]");
		m_probBiasVec = tmp;	    
	    }else if (m_probBiasDim > 0){
		//printf("\tWARNING: the prob bias must be %d in dimension\n", m_secondOutputDim);
		//printf("\tWARNING: prob bias data will not used\n");
		Cpu::real_vector tmp(this->outputs().size()/this->size(), 0.0);
		m_probBiasVec = tmp;
	    }else{
		m_probBiasDim = -1;
		m_probBiasVec.clear();
	    }
	}else{
	    m_secondOutput.clear();
	    m_probBiasVec.clear();
	    m_probBiasDim = -1;
	}

	// Allocate memory for mdnParameter
	Cpu::real_vector tmp(this->outputs().size() / this->size() * this->m_mdnParaDim, 0.0);
	this->m_mdnParaVec = tmp;

    }

    template <typename TDevice>
    MDNLayer<TDevice>::~MDNLayer()
    {
    }

    template <typename TDevice>
    const std::string& MDNLayer<TDevice>::type() const
    {
        static const std::string s("mdn");
        return s;
    }

    template <typename TDevice>
    real_t MDNLayer<TDevice>::calculateError()
    {
	real_t temp = 0.0;
	real_t temp2= 0.0;
	int i=0;
	
	BOOST_FOREACH (boost::shared_ptr<MDNUnit<TDevice> > &mdnUnit, m_mdnUnits){
	    if (this->flagMseWeight() && this->_mseWeightCPU()[i] < 0.0){
		// skip 
	    }else{
		temp2 = mdnUnit->calculateError(this->_targets());
		if (temp2 != temp2) printf("NaN: %d-th unit\t", i);
		temp += temp2;
	    }
	    ++i;
	}

	// the weighted likelihood (sequence level)
	// To to modified for more complex cases
	if (this->_flagSeqWeights()){
	    temp2 = 0;
	    for (i = 0; i < this->parallelSequences(); i++)
		temp2 += this->_seqWeightsCPU()[i];
	    temp2 = temp2/this->parallelSequences();
	    temp = temp * temp2;
	}
	//std::cout << temp/this->curMaxSeqLength() << std::endl;
	//misFuncs::AppendVecBin(this->precedingLayer().outputs());
	return temp;
    }

    template <typename TDevice>
    typename MDNLayer<TDevice>::real_vector& MDNLayer<TDevice>::mdnParaVec()
    {
        return m_mdnParaVec;
    }
    
    template <typename TDevice>
    int MDNLayer<TDevice>::mdnParaDim()
    {
	return m_mdnParaDim;
    }

    template <typename TDevice>
    void MDNLayer<TDevice>::computeForwardPass(const int nnState)
    {
	BOOST_FOREACH (boost::shared_ptr<MDNUnit<TDevice> > &mdnUnit, m_mdnUnits){
	    mdnUnit->computeForward();
	    
	    // if this MDN is in the middle of a network, generate the output
	    if (this->_postLayerType() == NN_POSTOUTPUTLAYER_NOTLASTMDN &&
		nnState != NN_STATE_GAN_NOGAN && nnState != NN_STATE_GAN_NOGAN_TRAIN)
		mdnUnit->getOutput(0.0001, (this->_targets()));
	}
    }

    template <typename TDevice>
    void MDNLayer<TDevice>::computeForwardPass(const int timeStep, const int nnState)
    {

	BOOST_FOREACH (boost::shared_ptr<MDNUnit<TDevice> > &mdnUnit, m_mdnUnits){
	    // Dirty code: m_probBiasVec is use to change the genMethod of mdnUnit
	    if (m_probBiasDim > 0)
		mdnUnit->setGenMethod(this->m_probBiasVec, timeStep);

	    mdnUnit->computeForward(timeStep);

	    // if this MDN is in the middle of a network, generate the output
	    if (this->_postLayerType() == NN_POSTOUTPUTLAYER_NOTLASTMDN && 
		nnState != NN_STATE_GAN_NOGAN && nnState != NN_STATE_GAN_NOGAN_TRAIN){
		mdnUnit->getOutput(timeStep,    0.0001, (this->_targets()));
		mdnUnit->getParameter(timeStep, helpers::getRawPointer(this->m_mdnParaVec));
	    }
	    // this->getOutput(timeStep, 0.0001); // by default, use 0.0001 as the parameter
	}
    }

    template <typename TDevice>
    void MDNLayer<TDevice>::computeBackwardPass(const int nnState)
    {
	thrust::fill(this->_outputErrors().begin(), this->_outputErrors().end(), (real_t)0.0);
	thrust::fill(m_sharedWeightUpdates.begin(), m_sharedWeightUpdates.end(), (real_t)0.0);
	
	int i = 0;
	int ganState = 0;
	
	if (m_oneSidedSmoothing)
	    ganState = PostOutputLayer<TDevice>::ganState();
	    
	BOOST_FOREACH (boost::shared_ptr<MDNUnit<TDevice> > &mdnUnit, m_mdnUnits){
	    if (this->flagMseWeight() && this->_mseWeightCPU()[i] < 0.0){
		//continue; // skip this unit if specified by the mseWeight
	    }else{
		mdnUnit->computeBackward(this->_targets(), ganState);
	    }
	    i++;
	}

	if (this->_flagSeqWeights()){
	    internal::seqGradWeight fn1;
	    fn1.seqWeight = helpers::getRawPointer(this->_seqWeights());
	    fn1.patTypes  = helpers::getRawPointer(this->patTypes());
	    fn1.dataDim   = this->precedingLayer().size();
	    	
	    int n = this->precedingLayer().curMaxSeqLength();
	    n = n * this->precedingLayer().parallelSequences();
	    n = n * this->precedingLayer().size();

	    thrust::for_each(
		thrust::make_zip_iterator(
			thrust::make_tuple(this->_outputErrors().begin(), 
					   thrust::counting_iterator<int>(0))),
		thrust::make_zip_iterator(
			thrust::make_tuple(this->_outputErrors().begin()    + n, 
					   thrust::counting_iterator<int>(0)+ n)),
		fn1);
	}
	
	PostOutputLayer<TDevice>::computeBackwardPass(nnState);
    }
    
    template <typename TDevice>
    void MDNLayer<TDevice>::exportConfig(const helpers::JsonValue     &weightsObject, 
					 const helpers::JsonAllocator &allocator) const
    {
	if (!weightsObject->IsObject())
            throw std::runtime_error("The JSON value is not an object");

	// config
	rapidjson::Value inputConfig(rapidjson::kArrayType);
	int inputConfigCount = this->m_mdnConfigVec.size();
	inputConfig.Reserve(inputConfigCount, allocator);
	for (int i = 0; i < inputConfigCount; i++)
	    inputConfig.PushBack(this->m_mdnConfigVec[i], allocator);
	
	rapidjson::Value weightsSection(rapidjson::kObjectType);
	weightsSection.AddMember("config", inputConfig, allocator);
	

	// weight
        // do nothing if we don't have any weights
	if (m_sharedWeights.empty()){
	    //weightsObject->AddMember(this->name().c_str(), weightsSection, allocator);
            //return;
	}else{

	    // create and fill the weight arrays
	    rapidjson::Value inputWeightsArray(rapidjson::kArrayType);
	    int inputWeightsCount = this->m_sharedWeights.size();
	    inputWeightsArray.Reserve(inputWeightsCount, allocator);
	    for (int i = 0; i < inputWeightsCount; ++i)
		inputWeightsArray.PushBack(m_sharedWeights[i], allocator);

	    rapidjson::Value biasWeightsArray(rapidjson::kArrayType);
	    int biasWeightsCount = 0;
	    biasWeightsArray.Reserve(biasWeightsCount, allocator);

	    rapidjson::Value internalWeightsArray(rapidjson::kArrayType);
	    //int internalWeightsCount = 0; 

	    // create and fill the weights subsection
	    weightsSection.AddMember("input",    inputWeightsArray,    allocator);
	    weightsSection.AddMember("bias",     biasWeightsArray,     allocator);
	    weightsSection.AddMember("internal", internalWeightsArray, allocator);
	    
	    
	}
	
	// Add additional options for MDN
	int mdnUnitCounts = this->m_mdnUnits.size();
	rapidjson::Value tieVariance(rapidjson::kArrayType);
	tieVariance.Reserve(mdnUnitCounts, allocator);
	rapidjson::Value trainableType(rapidjson::kArrayType);
	trainableType.Reserve(mdnUnitCounts, allocator);
	rapidjson::Value tanhRegAutoType(rapidjson::kArrayType);
	tanhRegAutoType.Reserve(mdnUnitCounts, allocator);
	rapidjson::Value secondOutputOpt(rapidjson::kArrayType);
	secondOutputOpt.Reserve(mdnUnitCounts, allocator);
	

	int cnt = 0;
	BOOST_FOREACH (const boost::shared_ptr<MDNUnit<TDevice> > &mdnUnit, m_mdnUnits){
	    tieVariance.PushBack((int)mdnUnit->flagVariance(), allocator);
	    trainableType.PushBack(mdnUnit->flagTrainable(), allocator);
	    tanhRegAutoType.PushBack((int)mdnUnit->tanhRegType(), allocator);
	    secondOutputOpt.PushBack((int)m_secondOutputOpt[cnt], allocator);
	    cnt++;
	}
	weightsSection.AddMember("tieVarianceFlag",  tieVariance, allocator);
	weightsSection.AddMember("trainableFlag",  trainableType, allocator);
	weightsSection.AddMember("tanhAutoReg",  tanhRegAutoType, allocator);
	weightsSection.AddMember("feedbackOpt",  secondOutputOpt, allocator);	
	// add the weights section to the weights object
	weightsObject->AddMember(this->name().c_str(), weightsSection, allocator);
	return;
    }
    
    template <typename TDevice>
    void MDNLayer<TDevice>::exportWeights(const helpers::JsonValue &weightsObject, 
					  const helpers::JsonAllocator &allocator) const
    {
	// we use exportConfig above instead of exportWeights to dump the weight
        if (!weightsObject->IsObject())
            throw std::runtime_error("The JSON value is not an object");
    }

    template <typename TDevice>
    void MDNLayer<TDevice>::reReadWeight(const helpers::JsonValue &weightsSection, 
					 const int readCtrFlag)
    {
	Cpu::real_vector weights;
	if (weightsSection.isValid() && weightsSection->HasMember(this->name().c_str())){
	    printf("read weight for layer %s", this->name().c_str());
	    const rapidjson::Value &weightsChild = (*weightsSection)[this->name().c_str()];
            if (!weightsChild.IsObject())
                throw std::runtime_error(std::string("Weights section for layer '") + 
					 this->name() + "' is not an object");
            if (!weightsChild.HasMember("input") || !weightsChild["input"].IsArray())
                throw std::runtime_error(std::string("Missing array 'weights/") + 
					 this->name() + "/input'");
            if (!weightsChild.HasMember("bias") || !weightsChild["bias"].IsArray())
                throw std::runtime_error(std::string("Missing array 'weights/") + 
					 this->name() + "/bias'");
            if (!weightsChild.HasMember("internal") || !weightsChild["internal"].IsArray())
                throw std::runtime_error(std::string("Missing array 'weights/") + 
					 this->name() + "/internal'");
	    const rapidjson::Value &inputWeightsChild    = weightsChild["input"];
            const rapidjson::Value &biasWeightsChild     = weightsChild["bias"];
            const rapidjson::Value &internalWeightsChild = weightsChild["internal"];

            if (inputWeightsChild.Size() != m_trainableNum)
                throw std::runtime_error(std::string("Invalid number of input weights for layer '") 
					 + this->name() + "'");
            if (biasWeightsChild.Size() != 0)
                throw std::runtime_error(std::string("Invalid number of bias weights for layer '") 
					 + this->name() + "'");
            if (internalWeightsChild.Size() != 0)
                throw std::runtime_error(std::string("Invalid number of internal for layer '") 
					 + this->name() + "'");

            weights.reserve(inputWeightsChild.Size());
            for (rapidjson::Value::ConstValueIterator it = inputWeightsChild.Begin(); 
		 it != inputWeightsChild.End(); 
		 ++it)
                weights.push_back(static_cast<real_t>(it->GetDouble()));	    
	    m_sharedWeights       = weights;
	    m_sharedWeightUpdates = weights;
	    thrust::fill(m_sharedWeightUpdates.begin(), m_sharedWeightUpdates.end(), (real_t)0.0);
	    
	}else{
	    printf("not read weight for layer %s", this->name().c_str());
	}
    }


    template <typename TDevice>
    void MDNLayer<TDevice>::getOutput(const real_t para)
    {
	// Modify 05-24 Add support to EM-style generation
	if (para < -3.0)
	{
	    throw std::runtime_error("mdn_samplePara can't be less than -3.0");
	}
	else if (para >= 0.0)
	{
	    // sampling from the distribution
	    bool tmpFlag = true;
	    BOOST_FOREACH (boost::shared_ptr<MDNUnit<TDevice> > &mdnUnit, m_mdnUnits)
	    {
		mdnUnit->getOutput(para, (this->_targets()));
		if(tmpFlag){
		    if (mdnUnit->varScale().size()>0)
			printf("sampling with variance scaled by varVector");
		    else
			printf("sampling with variance scaled by %f", para);
		    tmpFlag =false;
		}
	    }	     
	}
	else if (para > -1.50)
	{
	    // output the data parameter
	    printf("generating the parameters of MDN");
	    BOOST_FOREACH (boost::shared_ptr<MDNUnit<TDevice> > &mdnUnit, m_mdnUnits)
	    {
		mdnUnit->getParameter(helpers::getRawPointer(this->m_mdnParaVec));
	    }
	}
	else
	{
	    // EM generation
	    printf("EM-style generation\n");
	    int i = 0;
	    BOOST_FOREACH (boost::shared_ptr<MDNUnit<TDevice> > &mdnUnit, m_mdnUnits){
		printf("U%d",i++);
		mdnUnit->getEMOutput(para, this->_targets());
	    }
	}

        #ifdef DEBUG_LOCAL
	Cpu::real_vector temp=this->_targets();
	printf("Sampling: %f \n", temp[0]);
        #endif	
    }

    template <typename TDevice>
    void MDNLayer<TDevice>::getOutput(const int timeStep, const real_t para)
    {
	
	// for frame by frame, we always assume that parameter and output should be generated
	if (para > -1.50 || para < -3.0){
	    BOOST_FOREACH (boost::shared_ptr<MDNUnit<TDevice> > &mdnUnit, m_mdnUnits){

		// Dirty code: m_probBiasVec is use to change the genMethod of mdnUnit
		if (m_probBiasDim > 0)
		    mdnUnit->setGenMethod(this->m_probBiasVec, timeStep);

		mdnUnit->getOutput(timeStep, ((para>0)?(para):(0.0001)), (this->_targets()));
		mdnUnit->getParameter(timeStep,
				      helpers::getRawPointer(this->m_mdnParaVec));
	    }
	}else{
	    throw std::runtime_error("Frame-wise EM generation is not implemented");
	    // not implememented for EM generation
	}
    }
    
    template <typename TDevice>
    Cpu::real_vector MDNLayer<TDevice>::getMdnConfigVec()
    {
	return m_mdnConfigVec;
    }
    
    template <typename TDevice>
    void MDNLayer<TDevice>::initPreOutput(const MDNLayer<TDevice>::cpu_real_vector &mVec, 
					  const MDNLayer<TDevice>::cpu_real_vector &vVec)
    {
	BOOST_FOREACH (boost::shared_ptr<MDNUnit<TDevice> > &mdnUnit, m_mdnUnits){
	    mdnUnit->initPreOutput(mVec, vVec);
	}
    }
    
    template <typename TDevice>
    void MDNLayer<TDevice>::reInitWeight()
    {
	if (this->m_trainable){
	    thrust::fill(m_sharedWeights.begin(), m_sharedWeights.end(), 0.0);
	    thrust::fill(m_sharedWeightUpdates.begin(), m_sharedWeightUpdates.end(), 0.0);
	}
    }
    
    template <typename TDevice>
    MDNLayer<TDevice>::real_vector& MDNLayer<TDevice>::weights()
    {
	return m_sharedWeights;
    }

    template <typename TDevice>
    const MDNLayer<TDevice>::real_vector& MDNLayer<TDevice>::weights() const
    {
	return m_sharedWeights;
    }

    template <typename TDevice>
    MDNLayer<TDevice>::real_vector& MDNLayer<TDevice>::weightUpdates()
    {
	return m_sharedWeightUpdates;
    }
    
    template <typename TDevice>
    bool MDNLayer<TDevice>::flagTrainable() const
    {
	return m_trainable;
    }
    
    template <typename TDevice>
    void MDNLayer<TDevice>::setCurrTrainingEpoch(const int curTrainingEpoch)
    {
	Layer<TDevice>::setCurrTrainingEpoch(curTrainingEpoch);
	BOOST_FOREACH (boost::shared_ptr<MDNUnit<TDevice> > &mdnUnit, m_mdnUnits){
	    mdnUnit->setCurrTrainingEpoch(curTrainingEpoch);
	}
    }
    
    template <typename TDevice>
    int& MDNLayer<TDevice>::getCurrTrainingEpoch()
    {
	return Layer<TDevice>::getCurrTrainingEpoch();
    }

    template <typename TDevice>
    const std::string& MDNLayer<TDevice>::layerAddInfor(const int opt) const
    {
        static std::string s;
        if (s == "" && opt==1){
	    /*if (m_secondOutputOpt.size()<1 || m_secondOutputOpt.size() != m_mdnUnits.size())
	      throw std::runtime_error("Feedback is invalid");*/
            std::ostringstream Convert;
	    Convert << m_secondOutputDim << "_";
	    s = Convert.str();
	}
	/*else
	  s = "";*/
        return s;
    }

    template <typename TDevice>
    MDNLayer<TDevice>::real_vector& MDNLayer<TDevice>::feedbackOutputs(const bool flagTrain)
    {
	return this->m_secondOutput;
    }


    // The three functions retrieveFeedBackData
    // 1. retrieveFeedBackData(): only used in computeForward()
    //                            put training output as the feedback data
    // 2. retrieveFeedBackData(randNum, method): only used in computeForward() after the 1st one
    //                            set the feedback data to zero or 1/N
    // 3. retrieveFeedBackData(timeStep, method): used both in training and inference stages
    //           in training:  for schedule sampling in computeForward()
    //                         by olm -> retrieveFeedBackData(timeStep, methodCode)
    //           in inference: for all methods in computeForwardPassGen()
    //                         by olm -> retrieveFeedBackData(timeStep, methodCode)
    //                            olm -> retrieveFeedBackData(timeStep, 0)
    //      
    template <typename TDevice>
    void MDNLayer<TDevice>::retrieveFeedBackData()
    {
	int dimStart = 0;
	int cnt      = 0;
	
	BOOST_FOREACH (boost::shared_ptr<MDNUnit<TDevice> > &mdnUnit, m_mdnUnits){
	    // Use default method: set ground truth
	    mdnUnit->fillFeedBackData(this->m_secondOutput,  m_secondOutputDim,  dimStart,
				      this->_targets(), NN_FEEDBACK_GROUND_TRUTH);
	    dimStart += mdnUnit->feedBackDim();
	    cnt++;
	}
    }

    template <typename TDevice>
    void MDNLayer<TDevice>::retrieveFeedBackData(real_vector& randNum, const int method)
    {
	int dimStart = 0;
	int cnt      = 0;
	
	// The code here is dirty
	BOOST_FOREACH (boost::shared_ptr<MDNUnit<TDevice> > &mdnUnit, m_mdnUnits){
	    // method = -1 or -2 (zero or 1/N killing)
	    mdnUnit->fillFeedBackData(this->m_secondOutput,  m_secondOutputDim,  dimStart,
				      randNum, method);
	    dimStart += mdnUnit->feedBackDim();
	    cnt++;
	}
    }

    template <typename TDevice>
    void MDNLayer<TDevice>::retrieveFeedBackData(const int timeStep, const int method)
    {
	int dimStart = 0;
	int cnt      = 0;
	BOOST_FOREACH (boost::shared_ptr<MDNUnit<TDevice> > &mdnUnit, m_mdnUnits){
	    mdnUnit->fillFeedBackData(this->m_secondOutput,  m_secondOutputDim,  dimStart,
				      this->_targets(), timeStep, method);
	    dimStart += mdnUnit->feedBackDim();
	    cnt++;
	}
	dimStart     = 0;
	
	/* 2017/02/22 Used for bias modify the probability vector for feedback
	if (m_probBiasDim > 0){
	    BOOST_FOREACH (boost::shared_ptr<MDNUnit<TDevice> > &mdnUnit, m_mdnUnits){
		mdnUnit->biasProb(this->m_secondOutput,  m_secondOutputDim,  dimStart,
				  this->m_probBiasVec,   timeStep);
		dimStart += mdnUnit->feedBackDim();
		cnt++;
	    }
	}*/
    }

    template <typename TDevice>
    void MDNLayer<TDevice>::setFeedBackData(const int timeStep, const int state)
    {
	int dimStart = 0;
	int cnt      = 0;
	BOOST_FOREACH (boost::shared_ptr<MDNUnit<TDevice> > &mdnUnit, m_mdnUnits){
	    mdnUnit->setFeedBackData(this->m_secondOutput,  m_secondOutputDim,  dimStart,
				     state, timeStep);
	    dimStart += mdnUnit->feedBackDim();
	    cnt++;
	}
    }

    template <typename TDevice>
    real_t MDNLayer<TDevice>::retrieveProb(const int timeStep, const int state)
    {
	// This function is specifically used by MDNSoftmax !
	if (m_mdnUnits.size()>1)
	    throw std::runtime_error("Not implemented for retrieveProb");
	return m_mdnUnits[0]->retrieveProb(timeStep, state);
    }
    
    template <typename TDevice>
    void MDNLayer<TDevice>::loadSequences(const data_sets::DataSetFraction &fraction,
					  const int nnState)
    {
	PostOutputLayer<TDevice>::loadSequences(fraction, nnState);

	// Load additional probabilistic data for biased generation in model with feedback link
	// NOTE: this part should be moved to DataSet.cpp
	
	// Now, this part is used only for generation stage
	if (m_probBiasDim > 0 && Configuration::instance().generatingMode() > 0){
	    
	    if (fraction.numSequences() > 1)
		throw std::runtime_error("Please turn off parallel mode");

	    int i = 0;
	    std::string fileName = m_probBiasDir + "/" + fraction.seqInfo(i).seqTag + ".bin";
	    
	    std::ifstream ifs(fileName.c_str(), std::ifstream::binary | std::ifstream::in);
	    if (!ifs.good())
		throw std::runtime_error(std::string("Fail to open ")+fileName);
	   
	    std::streampos numEleS, numEleE;
	    long int numEle;
	    real_t tempVal;
	    numEleS = ifs.tellg();
	    ifs.seekg(0, std::ios::end);
	    numEleE = ifs.tellg();
	    numEle  = (numEleE-numEleS)/sizeof(real_t);
	    ifs.seekg(0, std::ios::beg);
	    std::vector<real_t> tempVec;
	    for (unsigned int i = 0; i<numEle; i++){
		ifs.read ((char *)&tempVal, sizeof(real_t));
		tempVec.push_back(tempVal);
	    }
	    if (numEle < fraction.maxSeqLength())
		throw std::runtime_error("Probability bias vector is too short");
	    
	    i = (numEle < m_probBiasVec.size())?(numEle):(m_probBiasVec.size());
	    thrust::copy(tempVec.begin(), tempVec.begin() + i, m_probBiasVec.begin());
	    ifs.close();
	    
	}
    }

    // export
    template <typename TDevice>
    void MDNLayer<TDevice>::exportLayer(const helpers::JsonValue &layersArray, 
					const helpers::JsonAllocator &allocator) const
    {
	PostOutputLayer<TDevice>::exportLayer(layersArray, allocator);
	if (m_uvSigmoidStr.size())
	    (*layersArray)[layersArray->Size() - 1].AddMember("uvSigmoidSoftmax",
							      m_uvSigmoidStr.c_str(),
							      allocator);
	if (m_conValSigStr.size())
	    (*layersArray)[layersArray->Size() - 1].AddMember("continuousValuedSigmoid",
							      m_conValSigStr.c_str(),
							      allocator);
	/*if (m_quanMergeStr.size())
	    (*layersArray)[layersArray->Size() - 1].AddMember("quantizeMerge",
							      m_quanMergeStr.c_str(),
							      allocator);*/
	if (m_oneSidedSmoothing)
	    (*layersArray)[layersArray->Size() - 1].AddMember("oneSidedSmoothGAN",
							      m_oneSidedSmoothing,
							      allocator);
	if (m_mdnConfigPath.size())
	    (*layersArray)[layersArray->Size() - 1].AddMember("mdnConfig",
							      m_mdnConfigPath.c_str(),
							      allocator);
	if (m_flagNewAROption)
	    (*layersArray)[layersArray->Size() - 1].AddMember("FlagNewAROption",
							      m_flagNewAROption,
							      allocator);
	
    }

    template class MDNLayer<Cpu>;
    template class MDNLayer<Gpu>;

}
