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
/****
 *
 *
 *
 ****/


#include "NormFlowLayer.hpp"
#include "../helpers/JsonClasses.hpp"
#include "../helpers/misFuncs.hpp"
#include "../helpers/getRawPointer.cuh"
#include "../helpers/Matrix.hpp"
#include "../helpers/safeExp.cuh"
#include "../activation_functions/Logistic.cuh"

#include "../Configuration.hpp"
#include "../MacroDefine.hpp"

#include "MDNLayer.hpp"

#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/for_each.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/fill.h>
#include <boost/foreach.hpp>
#include <boost/shared_ptr.hpp>
#include <vector>


#define NORMFLOW_TRANSFORM_SIGMOID_STD      1  // sigmoid (std)
#define NORMFLOW_TRANSFORM_EXP_STD          2  // exp (std)
#define NORMFLOW_TRANSFORM_SIGMOID_STD_REV  3  // sigmoid (std), reverse transform
#define NORMFLOW_TRANSFORM_VOLUME_PRESERVE  4  // std -> 1


#define NORMFLOW_PREVIOUS_LINEAR_LAYER_BIAS 2.0

#define NORMFLOW_FLOW_DIRECTION_CAUSAL     1
#define NORMFLOW_FLOW_DIRECTION_ANTICAUSAL 0


// Definition for computation
namespace internal {
namespace {
    
    // compute the output based on the output of gate
    //  normType:
    //   NORMFLOW_TRANSFORM_SIGMOID_STD
    //     output = (1 - sigmoid(raw_std)) * mean + sigmoid(raw_std) * input
    //   NORMFLOW_TRANSFORM_EXP_STD
    //     output = mean + exp(raw_std) * input
    struct Normflowtransform
    {
	real_t *paraRaw;        // mean and raw_std
	real_t *paraBuf;        // raw_std (become a positive number)
	int    *skipFlag;       // some dimensions are not modeled by GMM

	int normType;           //
	int featDim;
	int parallel;
	const char *patTypes;   // 

	__host__ __device__ void  operator() (
		const thrust::tuple<const real_t&, const int&, real_t&> &t) const
	{
	    int timeIdx = t.get<1>() / featDim;
	    int dimIdx  = t.get<1>() % featDim;

	    int stdIdx  = (timeIdx - parallel) * featDim * 2 + featDim + dimIdx;

	    if (skipFlag[dimIdx] || timeIdx < parallel){
		// if these dimensions can be skipped, or
		// if this is the first frame
		paraBuf[t.get<1>()] = (real_t)1.0;
		t.get<2>() = t.get<0>();		
	    }else{
		
		// dimensions that should be transformed
		if (normType == NORMFLOW_TRANSFORM_EXP_STD){
		    
		    // in case the previous layer's output is dropped out
		    if (paraRaw[stdIdx] == 0.0 && paraRaw[stdIdx - featDim] == 0.0)
			paraBuf[t.get<1>()] = 1.0;
		    else
			paraBuf[t.get<1>()] = helpers::safeExp(paraRaw[stdIdx]);

		    // out = std * input + mean, std = exp(\hat_std)
		    t.get<2>() = paraRaw[stdIdx - featDim] + paraBuf[t.get<1>()] * t.get<0>();

		}else if (normType == NORMFLOW_TRANSFORM_VOLUME_PRESERVE){
		    // std = 1.0 
		    paraBuf[t.get<1>()] = (real_t)1.0;
		    t.get<2>() = paraRaw[stdIdx - featDim] + paraBuf[t.get<1>()] * t.get<0>();

		}else if (normType == NORMFLOW_TRANSFORM_SIGMOID_STD){
		    // std = sigmoid(\hat_std)

		    // in case the previous layer's output is dropped out
		    if (paraRaw[stdIdx] == 0.0 && paraRaw[stdIdx - featDim] == 0.0)
			paraBuf[t.get<1>()] = 1.0;
		    else
			paraBuf[t.get<1>()] = activation_functions::Logistic::fn(paraRaw[stdIdx]);
		    
		    // out = std * input + (1 - std) * mean 
		    t.get<2>() = paraBuf[t.get<1>()] * t.get<0>() +
			((real_t)1.0 - paraBuf[t.get<1>()]) * paraRaw[stdIdx - featDim];
		    
		}else if (normType == NORMFLOW_TRANSFORM_SIGMOID_STD_REV){
		    
		    // std = sigmoid(\hat_std)
		    // paraBuf[t.get<1>()] = activation_functions::Logistic::fn(paraRaw[stdIdx]);
		    // for convenience, let std = 1.0/std = 1 + exp(-\hat_std)
		    // in case the previous layer's output is dropped out
		    if (paraRaw[stdIdx] == 0.0 && paraRaw[stdIdx - featDim] == 0.0)
			paraBuf[t.get<1>()] = 1.0;
		    else
			paraBuf[t.get<1>()] = (real_t)1.0 + helpers::safeExp(-paraRaw[stdIdx]);
		    
		    // out = (input - mu) / std + mu
		    t.get<2>() = (t.get<0>() - paraRaw[stdIdx - featDim]) * paraBuf[t.get<1>()] +
			paraRaw[stdIdx - featDim];
		}
	    }
	}
    };

    // compute the output based on the output of gate
    //  
    struct GradientsToInputParaLayer
    {
	real_t *gradOut;        // target buffer to store the gradients
	real_t *gradIn;         // d_E / d_o_(this_flow)
	real_t *preFlow;        // o_(previous_flow)
	real_t *paraRaw;
	real_t *paraBuf;
	
	int    *skipFlag;
	
	int featDim;
	int normType;
	int parallel;
	const char *patTypes;   // 

	__host__ __device__ void  operator() (
		const thrust::tuple<const real_t&, const int&> &t) const
	{
	    int timeIdx = t.get<1>() / featDim;
	    int dimIdx  = t.get<1>() % featDim;

	    int meanIdx     = timeIdx * featDim * 2 + dimIdx;
	    // Fatal error: wrong variable
	    // int stdIdx      = meanIdx + dimIdx;
	    int stdIdx      = meanIdx + featDim;             
	    
	    int outputIdx   = timeIdx * featDim + dimIdx;
	    int outShift    = parallel * featDim;
	    

	    if (skipFlag[dimIdx] ||
		patTypes[timeIdx] == PATTYPE_LAST || patTypes[timeIdx] == PATTYPE_NONE){
		// skip dimensions or no-valid or last frame
		gradOut[stdIdx]  = 0.0;  // std
		gradOut[meanIdx] = 0.0;  // mean
		
	    }else{
		// no skip dimensions	    

		
		if (normType == NORMFLOW_TRANSFORM_EXP_STD){
		    // gradients from E
		    // d_(E+Det) / d_mean_t = d_(E+Det) / d_o_(this_flow_output) * d_o/d_mean
		    //                      = d_E / d_o_(this_flow_output)
		    // Be careful: std_t is stored in paraBuf[t+1]
		    gradOut[meanIdx] = gradIn[outputIdx + outShift];

		    // d_(E+Det) / d_std
		    //  d_E / d_std = d_E / d_o_(this_flow) * o_(this_flow) * std
		    //  d_Det / d_std = -1
		    gradOut[stdIdx]  = gradIn[outputIdx + outShift] * 
			preFlow[outputIdx + outShift] * paraBuf[outputIdx + outShift] -
		        (real_t)1.0;

		}else if (normType == NORMFLOW_TRANSFORM_VOLUME_PRESERVE){
		    gradOut[meanIdx] = gradIn[outputIdx + outShift];
		    gradOut[stdIdx]  = 0.0;
		    
		}else if (normType == NORMFLOW_TRANSFORM_SIGMOID_STD){
		    // NORMFLOW_TRANSFORM_SIGMOID_STD
		    // d_(E+Det) / d_mean_t = d_(E+Det) / d_o_(this_flow_output) * d_o/d_mean
		    //                      = d_E / d_o_(this_flow_output) * (1 - std)
		    gradOut[meanIdx] = gradIn[outputIdx + outShift] *
			((real_t)1.0 - paraBuf[outputIdx + outShift]);

		    // Because of std = sgimoid (\hat_std), d_std / d_\hat_std = std * (1-std)
		    // d_(E+Det) / d_std
		    //  d_E / d_std = d_E / d_o_(this_flow) * (o_(this_flow) - mu) * std * (1-std)
		    // Det = -\sum_log(std)
		    //  d_Det / d_std = (std - 1)
		    gradOut[stdIdx] = gradIn[outputIdx + outShift] *
			paraBuf[outputIdx + outShift] *
			((real_t)1.0 - paraBuf[outputIdx + outShift]) * 
			(preFlow[outputIdx + outShift] - paraRaw[meanIdx]) +
			(paraBuf[outputIdx + outShift] - (real_t)1.0);
		    
		}else if (normType == NORMFLOW_TRANSFORM_SIGMOID_STD_REV){
		    
		    // d_(E+Det) / d_mean_t = d_E / d_o_(this_flow_output) * (1 - 1/std)
		    //                      = d_E / d_o_(this_flow_output) * (1 - paraBuff)
		    gradOut[meanIdx] = gradIn[outputIdx + outShift] *
			((real_t)1.0 - paraBuf[outputIdx + outShift]);

		    //  d_E / d_std = d_E / d_o * (o - mu) * (-1/std^2) * (std * (1-std))
		    //              = d_E / d_o * (o - mu) * (1-1/std)
		    //              = d_E / d_o * (o - mu) * (1-paraBuf)
		    // Det = -\sum_log(1/std) = \sum_log(std)
		    //  d_Det / d_std = (1 - std) = (1-1/paraBuf)
		    gradOut[stdIdx] = gradIn[outputIdx + outShift] *
			((real_t)1.0 - paraBuf[outputIdx + outShift]) * 
			(preFlow[outputIdx + outShift] - paraRaw[meanIdx]) +
			((real_t)1.0 - (real_t)1.0/paraBuf[outputIdx + outShift]);
		    
		}
		
	    }

	}
    };

    
    // compute the output based on the output of gate
    //  
    struct CollectGradFromMDN
    {
	real_t *gradMDN;

	int    *skipFlag;
	int    *gradPtr;

	int gradDim;
	int featDim;
	
	const char *patTypes;   // 

	__host__ __device__ void  operator() (
		const thrust::tuple<real_t&, const int&> &t) const
	{
	    int timeIdx = t.get<1>() / featDim;
	    int dimIdx  = t.get<1>() % featDim;

	    if (skipFlag[dimIdx] || patTypes[timeIdx] == PATTYPE_NONE){
		// skipped dimension or non-valid time step
		t.get<0>() = 0.0;
	    }else{
		int mixNum  = gradPtr[dimIdx * 3];      // how many mixtures
		int featDim = gradPtr[dimIdx * 3 + 1];  // dimension of feature
		int startD  = gradPtr[dimIdx * 3 + 2];  // position of the first dimension of mean
		
		real_t tmpGrad = 0.0;

		// The gradients w.r.t the output of final flow is -1.0 * gradints w.r.t GMM mean
		// So, just collect gradients from the "mean part" of GMM and multiply -1.0
		for (int i = 0; i < mixNum; i++)
		    tmpGrad += (gradMDN[gradDim * timeIdx + startD + i * featDim] * ((real_t)-1.0));
		
		t.get<0>() = tmpGrad;
	    }
	    

	}
    };


    // compute the output based on the output of gate
    //  
    struct ChangeFlowDirection
    {
	int    *seqLength;
	int     featDim;
	int     parallel;
	real_t *inputBuf;
	const char *patTypes;   // 

	__host__ __device__ void  operator() (
		const thrust::tuple<real_t&, const int&> &t) const
	{
	    int timeIdx  = t.get<1>() / featDim;
	    int dimIdx   = t.get<1>() % featDim;
	    int blockIdx = timeIdx / parallel;
	    int paralIdx = timeIdx % parallel;
	    
	    int thisSeqLength = seqLength[paralIdx];

	    if (patTypes[timeIdx] == PATTYPE_NONE){
		return;
	    }else{
		t.get<0>() = inputBuf[((thisSeqLength - 1 - blockIdx) * parallel + paralIdx) *
				      featDim + dimIdx];
	    }
	    
	}
    };

    
    // de-transformation
    //  input = (output - (1 - std) * mean)/std
    struct NormflowDeTransform
    {
	real_t *paraRaw;        // mean and raw_std
	int    *skipFlag;       
	
	int featDim;
	int parallel;
	int normType;
	
	const char *patTypes;   // 

	__host__ __device__ void  operator() (
		const thrust::tuple<const real_t&, const int&, real_t&> &t) const
	{
	    int timeIdx = t.get<1>() / featDim;
	    int dimIdx  = t.get<1>() % featDim;

	    int stdIdx  = (timeIdx - parallel) * featDim * 2 + featDim + dimIdx;
	    real_t std  = 0.0;

	    if (skipFlag[dimIdx]){
		t.get<2>() = t.get<0>();
	    }else{
		if (timeIdx < parallel){
		    t.get<2>() = t.get<0>();
		}else{
		    if (normType == NORMFLOW_TRANSFORM_EXP_STD){
			std = helpers::safeExp(paraRaw[stdIdx]);
			// out = (in - mu) / std
			t.get<2>() = (t.get<0>()-paraRaw[stdIdx-featDim])/std;
			
		    }else if (normType == NORMFLOW_TRANSFORM_VOLUME_PRESERVE){
			t.get<2>() = t.get<0>()-paraRaw[stdIdx-featDim];
			
		    }else if (normType == NORMFLOW_TRANSFORM_SIGMOID_STD){
			std = activation_functions::Logistic::fn(paraRaw[stdIdx]);
			// out = (in - (1-std) *mu) / std
			t.get<2>() = (t.get<0>()-((real_t)1.0 - std) *
				      paraRaw[stdIdx-featDim])/std;
		    }else if (normType == NORMFLOW_TRANSFORM_SIGMOID_STD_REV){
			
			std = activation_functions::Logistic::fn(paraRaw[stdIdx]);
			// out = in * std + (1 - std) * mu
			t.get<2>() = t.get<0>()*std+((real_t)1.0-std)*paraRaw[stdIdx-featDim];
		    }
		}
	    }
	    
	}
    };


    template<typename T>
    struct tmpLogFunc
    {
	__host__ __device__ T operator()(const T &x) const
	{
	    return helpers::safeLog(x);
	}
    };
    

} // anonymous namespace
} // namespace internal



namespace layers{

    // Construct the layer
    template <typename TDevice>
    NormFlowLayer<TDevice>::NormFlowLayer(
					const helpers::JsonValue &layerChild,
					const helpers::JsonValue &weightsSection,
					std::vector<Layer<TDevice>*> &precedingLayers,
					int maxSeqLength,
					int layerID)
	// use preLayers[0] as fake preceding layers
	: SkipLayer<TDevice>(layerChild, weightsSection, precedingLayers,
			     maxSeqLength, layerID, false)
	, m_targetLayer(NULL)
	, m_preFlowLayer(NULL)
	, m_initFlowFlag(false)
	, m_lastFlowFlag(false)
	, m_flowDirection(NORMFLOW_FLOW_DIRECTION_CAUSAL)
	, m_normType(NORMFLOW_TRANSFORM_SIGMOID_STD)
	, m_frameNum(1)
    {
	
	if (precedingLayers.size() < 1)
	    throw std::runtime_error("Impossible error: NormFlow has no previous layer");
	

	// Link previous layers
	m_previousFlowStr = (layerChild->HasMember("preFlowLayer") ? 
			     ((*layerChild)["preFlowLayer"].GetString()) : "");

	m_flowDirection   = (layerChild->HasMember("causal") ? 
			     ((*layerChild)["causal"].GetInt()) : NORMFLOW_FLOW_DIRECTION_CAUSAL);

	m_normType        = (layerChild->HasMember("normType") ? 
			     ((*layerChild)["normType"].GetInt()) : NORMFLOW_TRANSFORM_SIGMOID_STD);

	m_skipDim         = (layerChild->HasMember("skipDim") ? 
			     ((*layerChild)["skipDim"].GetString()) : "");
	
	// 
	if (m_previousFlowStr.size()){	    
	    // previous layers are specified by preSkipLayer
	    std::vector<std::string> tmpOpt;
	    misFuncs::ParseStrOpt(m_previousFlowStr, tmpOpt, ",");
	    for (int cnt = 0 ; cnt < tmpOpt.size(); cnt++) {
		BOOST_FOREACH (Layer<TDevice> *layer, precedingLayers) {
		    if (layer->name() == tmpOpt[cnt]){
			this->PreLayers().push_back(layer);
			break;
		    }
		}
	    }
	    if (this->PreLayers().size() != 1)
		throw std::runtime_error("NormFlow only receives input from another NormFlow");
	    
	    m_preFlowLayer = dynamic_cast<NormFlowLayer<TDevice>*>(this->PreLayers()[0]);
	    if (m_preFlowLayer == NULL)
		throw std::runtime_error("Previous flow layer type is invalid");
	    
	    if (m_preFlowLayer->name() == this->precedingLayer().name())
		throw std::runtime_error("Previous normflow layer is identical to previous layer");
	    
	    if (this->size() != m_preFlowLayer->size()){
		printf("Error: %s vs %s, ", m_preFlowLayer->name().c_str(), this->name().c_str());
		throw std::runtime_error("unequal layer size");
	    }
	    
	    if ((this->size() * 2) != this->precedingLayer().size()){
		printf("Error: %s vs %s, ",
		       this->precedingLayer().name().c_str(), this->name().c_str());
		throw std::runtime_error("previous layer size should be 2*normflow size");
	    }

	    if (this->getResolution() != m_preFlowLayer->getResolution()){
		printf("Error: %s vs %s, ", m_preFlowLayer->name().c_str(), this->name().c_str());
		throw std::runtime_error("Resolution mismatch");
	    }
	    if (this->getResolution() != this->precedingLayer().getResolution()){
		printf("Error: %s vs %s, ",
		       this->precedingLayer().name().c_str(), this->name().c_str());
		throw std::runtime_error("Resolution mismatch");
	    }
	    m_initFlowFlag = false;

	    // Data buffer 
	    m_paraBuffer   = Cpu::real_vector(this->outputs().size(), (real_t)0.0);

	    if (m_normType == NORMFLOW_TRANSFORM_SIGMOID_STD){
		printf("\n\tFlow type: o = (1-sigmoid(hat_std)) * mu + sigmoid(hat_std) * a");
	    }else if (m_normType == NORMFLOW_TRANSFORM_SIGMOID_STD_REV){
		printf("\n\tFlow type: o = (1/sigmoid(hat_std)) * (a - mu) + mu");	    
	    }else if (m_normType == NORMFLOW_TRANSFORM_EXP_STD){
		printf("\n\tFlow type: o = mu + exp(hat_std) * a");
	    }else if (m_normType == NORMFLOW_TRANSFORM_VOLUME_PRESERVE){
		printf("\n\tFlow type: o = mu + a");
	    }else{
		throw std::runtime_error("Unknown normType in network.jsn");
	    }

	}else{
	    
	    //
	    m_preFlowLayer = NULL;
	    m_initFlowFlag = true;
	    m_paraBuffer.clear();
	    
	    // This is the initial normalization flow
	    printf("\n\tInitial step of normalization flow");
	}
	
	//
	m_skipFlag.clear();
	
	// [startD, mixture_num, feature_dim]
	m_gradPtr.clear();

	
	m_seqLengthCpu = Cpu::int_vector(this->parallelSequences(), 0);
	m_seqLength = m_seqLengthCpu;
	
	// Print other info
	if (this->flagChangeFlowDirection()){
	    m_tempOutput = this->outputs();
	    printf("\n\tChange flow direction");
	}else
	    m_tempOutput.clear();

	if (m_skipDim.size()){
	    printf("\n\tSkip dim configuration: %s", m_skipDim.c_str());
	}
	
    }


    // Destructor
    template <typename TDevice>
    NormFlowLayer<TDevice>::~NormFlowLayer()
    {
    }

    template <typename TDevice>
    const bool& NormFlowLayer<TDevice>::flagInitFlow()
    {
	return m_initFlowFlag;
    }

    template <typename TDevice>
    const bool& NormFlowLayer<TDevice>::flagLastFlow()
    {
	return m_lastFlowFlag;
    }
    
    template <typename TDevice>
    void NormFlowLayer<TDevice>::initPreOutput()
    {

	// special initialization strategy (on each previous linear outptu layer)
	TrainableLayer<TDevice> *tLayer = dynamic_cast<layers::TrainableLayer<TDevice>*>(
						&(this->precedingLayer()));

	if (tLayer && ((tLayer->type() == "feedforward_identity") ||
		       (tLayer->type() == "feedforward_tanh") ||
		       (tLayer->type() == "feedforward_logistic"))){
	    printf("\nSpecial initialization for layers before normflow\t");
	    if (m_normType == NORMFLOW_TRANSFORM_SIGMOID_STD ||
		m_normType == NORMFLOW_TRANSFORM_SIGMOID_STD_REV){
		// set the weights to 0.0
		thrust::fill(tLayer->weights().begin(), tLayer->weights().end(), 0.0);

		// set the bias to a positive number
		thrust::fill(tLayer->weights().begin() +
			     tLayer->weights().size() - tLayer->size()/2,
			     tLayer->weights().end(),
			     NORMFLOW_PREVIOUS_LINEAR_LAYER_BIAS);
		
	    }else if (m_normType == NORMFLOW_TRANSFORM_EXP_STD ||
		      m_normType == NORMFLOW_TRANSFORM_VOLUME_PRESERVE){
		// set the weights to 0.0
		thrust::fill(tLayer->weights().begin(), tLayer->weights().end(), 0.0);

		// set the bias to a positive number
		thrust::fill(tLayer->weights().begin() +
			     tLayer->weights().size() - tLayer->size()/2,
			     tLayer->weights().end(),
			     0.0);
	    }
	}

    }

    // NN forward
    template <typename TDevice>
    void NormFlowLayer<TDevice>::computeForwardPass(const int nnState)
    {
	int uttLength = this->curMaxSeqLength() * this->parallelSequences();

	// For convenience, initialize the gradients buffer
	if (this->flagTrainingMode()){
	    thrust::fill(this->outputErrors().begin(), 
			 this->outputErrors().begin() + uttLength * this->size(),
			 0.0);
	    thrust::fill(this->outputErrorsFromSkipLayer().begin(), 
			 this->outputErrorsFromSkipLayer().begin() + uttLength * this->size(),
			 0.0);
	    thrust::fill(m_paraBuffer.begin(), m_paraBuffer.end(), 1.0);
	}
	
	if (this->flagInitFlow()){
	    // this is the initial flow
	    if (m_targetLayer == NULL)
		throw std::runtime_error("Initial flow doesn't find the target layer");

	    // copy the target output as the output of initial flow
	    thrust::copy(m_targetLayer->outputs().begin(),
			 m_targetLayer->outputs().begin() + uttLength * this->size(),
			 this->outputs().begin());
	    
	}else{
	    // this is not the initial flow
	    if (m_preFlowLayer == NULL)
		throw std::runtime_error("Norm flow doesn't find the previous flow layer");

	    // compute the output based on the output of gate
	    //  output = (1 - sigmoid(raw_std)) * mean + sigmoid(raw_std) * input
	    // or, other types of transformation
	    {{
		internal::Normflowtransform fn;

		fn.paraRaw   = helpers::getRawPointer(this->precedingLayer().outputs());
		fn.paraBuf   = helpers::getRawPointer(m_paraBuffer);
		fn.patTypes  = helpers::getRawPointer(this->patTypes());
		fn.skipFlag  = helpers::getRawPointer(m_skipFlag);
		fn.normType  = this->m_normType;
		fn.featDim   = this->size();
		fn.parallel  = this->parallelSequences();
		
		int n = uttLength * this->size();
		thrust::for_each(
		      thrust::make_zip_iterator(
			thrust::make_tuple(m_preFlowLayer->outputs().begin(),
					   thrust::counting_iterator<int>(0),
					   this->outputs().begin())),
		      thrust::make_zip_iterator(
			thrust::make_tuple(m_preFlowLayer->outputs().begin() + n,
					   thrust::counting_iterator<int>(0) + n,
					   this->outputs().begin()           + n)),
		      fn);
	    }}
	}
	

	// Check the flow direction
	// If this is the anti-causal inital flow, or this norm layer is
	//  different from the previous layer
	if (this->flagChangeFlowDirection()){
	    // reverse the output buffer
	    // cannot do it directly on this->outputs()
	    {{
		internal::ChangeFlowDirection fn;

		fn.seqLength = helpers::getRawPointer(m_seqLength);
		fn.featDim   = this->size();
		fn.parallel  = this->parallelSequences();
		fn.inputBuf  = helpers::getRawPointer(this->outputs());
		fn.patTypes  = helpers::getRawPointer(this->patTypes());
		
		int n = uttLength * this->size();
		thrust::for_each(
		      thrust::make_zip_iterator(
			thrust::make_tuple(m_tempOutput.begin(),
					   thrust::counting_iterator<int>(0))),
		      thrust::make_zip_iterator(
			thrust::make_tuple(m_tempOutput.begin() + n,
					   thrust::counting_iterator<int>(0) + n)),
		      fn);
	    }}
	    thrust::copy(m_tempOutput.begin(), m_tempOutput.end(), this->outputs().begin());
	}
	
	
	// Just copy from each normalization flow. The final flow will overwrite the results
	if (this->flagLastFlow())
	    thrust::copy(this->outputs().begin(),
			 this->outputs().begin() + uttLength * this->size(),
			 m_targetLayer->outputs().begin());
	
	// done
    }
    
    // NN forward
    template <typename TDevice>
    void NormFlowLayer<TDevice>::computeForwardPass(const int timeStep, const int nnState)
    {
	// To be implemented
	// absolute time
	int effTimeS = timeStep     * this->parallelSequences();
	int effTimeE = (timeStep+1) * this->parallelSequences();
	
	// shift of the pointer to the data 
	//int shiftIn  = 0; // value to assigned layer
	//int shiftOut = this->outputBufPtrBias(timeStep * this->parallelSequences(), nnState);
	
	// transformation
	if (this->flagInitFlow()){
	    // this is the initial flow
	    if (m_targetLayer == NULL)
		throw std::runtime_error("Initial flow doesn't find the target layer");

	    // shiftIn = m_targetLayer->outputBufPtrBias(effTimeS, nnState);
	    
	    // the first flow does nothing but copy the output to the output of network
	    if (timeStep == 0){
		int uttLength = this->curMaxSeqLength() * this->parallelSequences();
		// copy the generated output to the target output layer
		thrust::copy(this->outputs().begin(),
			     this->outputs().begin() + uttLength * this->size(),
			     m_targetLayer->outputs().begin());
	    }
	    

	    
	}else{
	    // this is not the initial flow
	    if (m_preFlowLayer == NULL)
		throw std::runtime_error("Norm flow doesn't find the previous flow layer");

	    // compute the output based on the output of gate
	    //  output = (1 - std) * mean + std * input
	    {{
		internal::NormflowDeTransform fn;

		fn.paraRaw   = helpers::getRawPointer(this->precedingLayer().outputs());
		fn.patTypes  = helpers::getRawPointer(this->patTypes());
		fn.skipFlag  = helpers::getRawPointer(this->m_skipFlag);
		
		fn.featDim   = this->size();
		fn.parallel  = this->parallelSequences();
		fn.normType  = this->m_normType;
		
		thrust::for_each(
		  thrust::make_zip_iterator(
		    thrust::make_tuple(
			this->outputs().begin()           + effTimeS * this->size(),
			thrust::counting_iterator<int>(0) + effTimeS * this->size(),
			m_preFlowLayer->outputs().begin() + effTimeS * this->size())),
		  thrust::make_zip_iterator(
		    thrust::make_tuple(
			this->outputs().begin()           + effTimeE * this->size(),
			thrust::counting_iterator<int>(0) + effTimeE * this->size(),
			m_preFlowLayer->outputs().begin() + effTimeE * this->size())),
		  fn);
	    }}

	}

    }

    // NN backward
    template <typename TDevice>
    void NormFlowLayer<TDevice>::computeBackwardPass(const int nnState)
    {
	int uttLength = this->curMaxSeqLength() * this->parallelSequences();
	if (this->flagInitFlow()){
	    // do nothing
	    return;
	}else{
	    // Step1: gathering gradients
	    if (this->flagLastFlow()){
		
		// Sum the gradients from the \par_E / \par_o_(last_flow)
		internal::CollectGradFromMDN fn;

		fn.gradMDN   = helpers::getRawPointer(
					m_targetLayer->precedingLayer().outputErrors());
		fn.skipFlag  = helpers::getRawPointer(m_skipFlag);
		fn.gradPtr   = helpers::getRawPointer(m_gradPtr);
		fn.patTypes  = helpers::getRawPointer(this->patTypes());
		fn.gradDim   = m_targetLayer->precedingLayer().size();
		fn.featDim   = this->size();

		
		int n = uttLength * this->size();
		thrust::for_each(
		      thrust::make_zip_iterator(
			thrust::make_tuple(this->outputErrors().begin(),
					   thrust::counting_iterator<int>(0))),
		      thrust::make_zip_iterator(
			thrust::make_tuple(this->outputErrors().begin()      + n,
					   thrust::counting_iterator<int>(0) + n)),
		      fn);
		
	    }else{
		// 
		// outputErrorsFromSkipLayer():
		//     d_(E) / d_o_(this_flow) * 1
		// outputErrors():
		//     d_(E) / d_mean_(next_flow) * d_mean_(next_flow) / d_o_(this_flow)
		//   + d_(E) / d_std_(next_flow) * d_std_(next_flow) / d_o_(this_flow)
		//   + d_(Det) / d_std_(next_flow) * d_std_(next_flow) / d_o_(this_flow)
		thrust::transform(
		      this->outputErrorsFromSkipLayer().begin(),
		      this->outputErrorsFromSkipLayer().begin()+uttLength * this->size(),
		      this->outputErrors().begin(),
		      this->outputErrors().begin(),
		      thrust::plus<real_t>());
	    }


	    // step1.5: reverse the temporal direction of gradients
	    // Check the flow direction
	    if (this->flagChangeFlowDirection()){
		{{
		    internal::ChangeFlowDirection fn;	
		    fn.seqLength = helpers::getRawPointer(m_seqLength);
		    fn.featDim   = this->size();
		    fn.parallel  = this->parallelSequences();
		    fn.inputBuf  = helpers::getRawPointer(this->outputErrors());
		    fn.patTypes  = helpers::getRawPointer(this->patTypes());
		
		    int n = uttLength * this->size();
		    thrust::for_each(
		      thrust::make_zip_iterator(
			thrust::make_tuple(m_tempOutput.begin(),
					   thrust::counting_iterator<int>(0))),
		      thrust::make_zip_iterator(
			thrust::make_tuple(m_tempOutput.begin() + n,
					   thrust::counting_iterator<int>(0) + n)),
		      fn);
		}}
		thrust::copy(m_tempOutput.begin(), m_tempOutput.end(),
			     this->outputErrors().begin());
	    }

	    
	    // Step2: deliver gradients to the previous flow
	    //   d_(E) / d_o_(previous_flow) = d_(E+Det) / d_o_(this_flow) * std_(this_flow)
	    //   or
	    //   d_(E) / d_o_(previous_flow) = d_(E+Det) / d_o_(this_flow) / std_(this_flow)
	    //   since we store 1/std in paraBuffer for the second case, we use the same function
	    SkipLayer<TDevice>* tempLayer = dynamic_cast<SkipLayer<TDevice>*>(m_preFlowLayer);
	    if(tempLayer){
		thrust::transform(this->outputErrors().begin(),
				  this->outputErrors().begin() +  uttLength * this->size(),
				  m_paraBuffer.begin(),
				  tempLayer->outputErrorsFromSkipLayer().begin(),
				  thrust::multiplies<real_t>());	
	    }else{
		throw std::runtime_error("Impossible: previous flow layer is invalid");
	    }

	    // Step3: deliver gradients to the input parameter layer
	    {{
		internal::GradientsToInputParaLayer fn;

		fn.gradOut   = helpers::getRawPointer(this->precedingLayer().outputErrors());
		fn.gradIn    = helpers::getRawPointer(this->outputErrors());
		fn.preFlow   = helpers::getRawPointer(m_preFlowLayer->outputs());
		fn.paraRaw   = helpers::getRawPointer(this->precedingLayer().outputs());
		fn.paraBuf   = helpers::getRawPointer(m_paraBuffer);
		fn.skipFlag  = helpers::getRawPointer(m_skipFlag);
		fn.patTypes  = helpers::getRawPointer(this->patTypes());

		fn.normType  = this->m_normType;
		fn.featDim   = this->size();
		fn.parallel  = this->parallelSequences();
		
		int n = uttLength * this->size();
		thrust::for_each(
		      thrust::make_zip_iterator(
			thrust::make_tuple(this->outputs().begin(),
					   thrust::counting_iterator<int>(0))),
		      thrust::make_zip_iterator(
			thrust::make_tuple(this->outputs().begin()           + n,
					   thrust::counting_iterator<int>(0) + n)),
		      fn);
	    }}	    
	    
	}
    }

    // NN backward
    template <typename TDevice>
    void NormFlowLayer<TDevice>::computeBackwardPass(const int timeStep, const int nnState)
    {
	throw std::runtime_error("NormFlowLayer computeBackwardPass(timeStep) not implemented");
    }

    
    template <typename TDevice>
    const std::string& NormFlowLayer<TDevice>::type() const
    {
	static std::string s;
	if (s.empty())
	    s = "normflow";
        return s;
    }

    template <typename TDevice>
    void NormFlowLayer<TDevice>::linkTargetLayer(Layer<TDevice> &targetLayer)
    {

	m_targetLayer = &targetLayer;
	
	if (m_targetLayer->getResolution() != this->getResolution()){
	    printf("Error: %s vs %s, ", targetLayer.name().c_str(), this->name().c_str());
	    throw std::runtime_error("Resolution mismatch");
	}
	
	if (m_targetLayer->size() != this->size()){
	    printf("Error: %s vs %s, ", m_targetLayer->name().c_str(), this->name().c_str());
	    throw std::runtime_error("unequal layer size");
	}
	
	if (m_targetLayer->type() != "mdn"){
	    throw std::runtime_error("normalization flow only works with MDN output layer");
	}
	
	layers::MDNLayer<TDevice>* tmpPtr =
	    dynamic_cast<layers::MDNLayer<TDevice>*>(m_targetLayer);
	
	if (tmpPtr){
	    Cpu::real_vector mdnConfigVec = tmpPtr->getMdnConfigVec();

	    Cpu::int_vector tmpSkipFlag = Cpu::int_vector(this->size(), 0.0);
	    Cpu::int_vector tmpGradPtr  = Cpu::int_vector(this->size() * 3, 0.0);
	    Cpu::int_vector skipDimCfg  = tmpSkipFlag;

	    // If some target dimensions are not processed by the normalization flow,
	    // parse the configuration string
	    if (m_skipDim.size())
		misFuncs::ParseIntOpt(m_skipDim, skipDimCfg);
	    
	    
	    if (!mdnConfigVec.empty()){
		
		for (int x = 0; x < (mdnConfigVec.size()-1)/5; x++){

		    int mdnType  = (int)mdnConfigVec[5+x*5];
		    int unitSOut = (int)mdnConfigVec[3+x*5];
		    int unitEOut = (int)mdnConfigVec[4+x*5];
			
		    if (mdnType == MDN_TYPE_SIGMOID || mdnType == MDN_TYPE_SOFTMAX){
			// If the target distribution is binar/multinomial,
			// normalizing flow cannot be used
			for (int y = unitSOut; y < unitEOut; y++)
			    tmpSkipFlag[y] = 1;	
		    }else{
			// If the target distribution is Gaussian/GMM
			int paraDimS = (int)mdnConfigVec[1+x*5];
			for (int y = unitSOut; y < unitEOut; y++){
			    
			    tmpGradPtr[y * 3 + 0] = mdnType;           // mixture number
			    tmpGradPtr[y * 3 + 1] = unitEOut-unitSOut; // feature dim
			    tmpGradPtr[y * 3 + 2] = paraDimS + mdnType + (y - unitSOut); // startD

			    // Check the skipDim configuration: whether this dimension should
			    // not be processed by the normalization flow
			    if (m_skipDim.size()){
				for (int i = 0; i<skipDimCfg.size()/2; i++)
				    if (y >= skipDimCfg[i * 2] && y < skipDimCfg[i*2+1])
					tmpSkipFlag[y] = 1;
			    }
			}
		    }
		}
	    }
	    
	    m_skipFlag = tmpSkipFlag;
	    m_gradPtr  = tmpGradPtr;
	    
	}else{
	    throw std::runtime_error("the target layer is not a MDN a layer");
	}
    }


    template <typename TDevice>
    void NormFlowLayer<TDevice>::exportLayer(const helpers::JsonValue &layersArray,
					     const helpers::JsonAllocator &allocator) const
    {
	SkipLayer<TDevice>::exportLayer(layersArray, allocator);
	if (m_previousFlowStr.size())
	    (*layersArray)[layersArray->Size() - 1].AddMember("preFlowLayer",
							      m_previousFlowStr.c_str(),
							      allocator);
	if (m_flowDirection != NORMFLOW_FLOW_DIRECTION_CAUSAL)
	    (*layersArray)[layersArray->Size() - 1].AddMember("causal",
							      m_flowDirection, allocator);
	if (m_normType != NORMFLOW_TRANSFORM_SIGMOID_STD)
	    (*layersArray)[layersArray->Size() - 1].AddMember("normType",
							      m_normType, allocator);
	if (m_skipDim.size())
	    (*layersArray)[layersArray->Size() - 1].AddMember("skipDim",
							      m_skipDim.c_str(), allocator);
	
    }

    template <typename TDevice>
    void NormFlowLayer<TDevice>::loadSequences(const data_sets::DataSetFraction &fraction,
					       const int nnState)
    {
	SkipLayer<TDevice>::loadSequences(fraction, nnState);
	m_frameNum = fraction.fracTimeLength();

	// load the sequence length in fraction
	if (fraction.numSequences() > this->parallelSequences())
	    throw std::runtime_error("Impossible error: too many sequences in one fraction");

	thrust::fill(m_seqLengthCpu.begin(), m_seqLengthCpu.end(), 0.0);
	for (int i = 0; i<fraction.numSequences(); i++)
	    m_seqLengthCpu[i] = fraction.seqInfo(i).length;
	m_seqLength = m_seqLengthCpu;    
    }

    template <typename TDevice>
    void NormFlowLayer<TDevice>::noticeLastFlow()
    {
	m_lastFlowFlag = true;
	if (m_flowDirection != NORMFLOW_FLOW_DIRECTION_CAUSAL){
	    printf("\n\tTht last normflow is changed to be causal");
	    m_flowDirection = NORMFLOW_FLOW_DIRECTION_CAUSAL;
	}
    }

    template <typename TDevice>
    void NormFlowLayer<TDevice>::loadNormedOutput()
    {
	int uttLength = this->curMaxSeqLength() * this->parallelSequences();

	if (this->flagLastFlow()){
	    // Generation stage
	    // copy the output from MDN and intialize each normalization flow buffer
	    thrust::copy(m_targetLayer->outputs().begin(),
			 m_targetLayer->outputs().begin() + uttLength * this->size(),
			 this->outputs().begin());
	}
	
	if (this->flagChangeFlowDirection()){
	    // reverse the output buffer
	    // cannot do it directly on this->outputs()
	    {{
		internal::ChangeFlowDirection fn;

		fn.seqLength = helpers::getRawPointer(m_seqLength);
		fn.featDim   = this->size();
		fn.parallel  = this->parallelSequences();
		fn.inputBuf  = helpers::getRawPointer(this->outputs());
		fn.patTypes  = helpers::getRawPointer(this->patTypes());
		
		int n = uttLength * this->size();
		thrust::for_each(
		      thrust::make_zip_iterator(
			thrust::make_tuple(m_tempOutput.begin(),
					   thrust::counting_iterator<int>(0))),
		      thrust::make_zip_iterator(
			thrust::make_tuple(m_tempOutput.begin() + n,
					   thrust::counting_iterator<int>(0) + n)),
		      fn);
	    }}
	    thrust::copy(m_tempOutput.begin(), m_tempOutput.end(), this->outputs().begin());
	}

	if (!this->flagInitFlow()){
	    thrust::copy(this->outputs().begin(),
			 this->outputs().end(),
			 m_preFlowLayer->outputs().begin());
	}
	
    }

    template <typename TDevice>
    bool NormFlowLayer<TDevice>::flagChangeFlowDirection()
    {
	// If intial_flow and anti-causal is used
	// Of if not inital_flow yet this layer is different from previous flow layer in direction
	if (this->flagInitFlow() &&
	    (this->flagCausalDependency() == NORMFLOW_FLOW_DIRECTION_ANTICAUSAL))
	    return true;
	else if ((!this->flagInitFlow()) &&
		 (this->flagCausalDependency() != m_preFlowLayer->flagCausalDependency()))
	    return true;
	else
	    return false;
    }

    template <typename TDevice>
    real_t NormFlowLayer<TDevice>::likelihoodJac()
    {
	m_likelihoodJac = thrust::transform_reduce(m_paraBuffer.begin(),
						   m_paraBuffer.end(),
						   internal::tmpLogFunc<real_t>(),
						   0.0,
						   thrust::plus<real_t>());
	return m_likelihoodJac;
    }

    template <typename TDevice>
    int NormFlowLayer<TDevice>::flagCausalDependency() const
    {
	return m_flowDirection;
    }
    
    template class NormFlowLayer<Cpu>;
    template class NormFlowLayer<Gpu>;

    
}
