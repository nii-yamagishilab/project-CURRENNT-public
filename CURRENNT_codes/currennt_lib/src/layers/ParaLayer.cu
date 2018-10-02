/******************************************************************************

*******************************************************************************/

#ifdef _MSC_VER
#   pragma warning (disable: 4244) // thrust/iterator/iterator_adaptor.h(121): warning C4244: '+=' : conversion from '__int64' to 'int', possible loss of data
#endif

#include "ParaLayer.hpp"
#include "../helpers/getRawPointer.cuh"
#include "../helpers/Matrix.hpp"
#include "../helpers/JsonClasses.hpp"
#include "../activation_functions/Tanh.cuh"
#include "../activation_functions/Logistic.cuh"
#include "../activation_functions/Identity.cuh"
#include "../activation_functions/Relu.cuh"

#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/for_each.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>

#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>

#include <typeinfo>

#define PARALAYERDEBUG    0

#define PARAFUNCTION_TYPE_0 0          // f(t) = b
#define PARAFUNCTION_TYPE_1 1          // f(t) = at+b
#define PARAFUNCTION_TYPE_0_NUMPARA 1  // f(t) = b, only one parameter
#define PARAFUNCTION_TYPE_1_NUMPARA 2  // f(t) = at+b, has two parameters

namespace internal {
namespace {

    /*struct fillTimeMatrix
    {
        int    effectiveLayerSize;
	int    layerSize;
	int    totalTime;
	int*   timeMatrixPtr;

        __host__ __device__ void operator() (const int &Idx) const
        {
	    int time = Idx / layerSize;
	    int dim  = Idx % layerSize;
	    *(timeMatrixPtr+Idx) = (dim < effectiveLayerSize)?(time):(totalTime - time - 1);
	}
	
	};*/
    
    struct setWeightZeroMatrix
    {
	// Generate a mask Matrix where unused elements are zero
	// 
	// Dispath over number of Parametric Dimension
	int    *paraConfig;
	//      Given by ReadParaOptions(), the format is
	//        NumberOfDimForPara_
	//            DimIdx_NumberOfElements_StartDim_EndDim_Type_StartDim_EndDim_Type_
	//            DimIdx_NumberOfElements_StartDim_EndDim_Type_StartDim_EndDim_Type_
	//      
	real_t *weightMatrixMask;
	//      A mask matrix, size: dimOfPrelayer * dimOfThisLayer
	//        
	int     paraConfigSize; // size of the paraConfig
	int     preLayerSize;   // size of the preceding layer
	int     curLayerSize;   // size of this layer
	bool    brnn;
	
	// Idx: which parametric dimension ?
	__host__ __device__ void operator() (const int &Idx) const
	{
	    int bias  = 2;                        // point to the first NumberOfElements
	    for (int i = 0; i<Idx; i++){
		bias += 3 * paraConfig[bias] + 2; // jump to the next NumberOfElements
	    }
	    
	    if (bias < paraConfigSize){
		int weightBias =  paraConfig[bias-1] * preLayerSize; // jump to the column
		
		// set the weight and bias for this output dim to zero
		for (int i = 0; i<preLayerSize; i++) 
		    *(weightMatrixMask + weightBias + i) = 0.0;
		*(weightMatrixMask + preLayerSize * curLayerSize + paraConfig[bias-1]) = 0.0;
		    
		// paraConfig[bias-1]: the dimension idx for this output dimension
		// paraConfig[bias-1] * preLayerSize: the first element of the column 
		// 
		for (int i = 0; i< paraConfig[bias]; i++){
		    int startDim = paraConfig[bias + i*3 + 1]; // first dimension of the 
		    int endDim   = paraConfig[bias + i*3 + 2]; // end
		    for (int j = startDim; j<endDim; j++){
			*(weightMatrixMask + weightBias + j) = 1.0;
			if (brnn)
			    *(weightMatrixMask + weightBias + j + preLayerSize/2) = 1.0;
		    }
		}
	    }
	}
    };

    struct paraTransform
    {
	// Given the output of preceding layer, the relative time matrix and configuration,
	// transform that output into the parametric results
	//
	// Dispatch T * NumberOfPara
	int     numPara;              // how many dimensions in output to be parameterized ?
	real_t *dataMatrix;           // output of preceding layer, size: dimension * T
	real_t *relativeTimeMatrix;   // timing matrix, size: dimension * T
	int     dataDim;              // #. rows of dataMatrix (dimension of previous layer)
	int    *paraConfig;           
	//      configuration, given by ReadParaOptions
	//
	bool    brnn;
	
	__host__ __device__ void operator() (const int &Idx) const
	{
	    int timeIdx = Idx / numPara;
	    int paraIdx = Idx % numPara;
	    
	    int startDim = paraConfig[2*paraIdx];
	    int paraType = paraConfig[2*paraIdx+1];
	    
	    int dataIdx = timeIdx * dataDim + startDim;
	    
	    // first order linear function at+b
	    if (paraType ==PARAFUNCTION_TYPE_1){
		// compute at+b
		*(dataMatrix + dataIdx) = ((*(dataMatrix + dataIdx)) * 
					   (*(relativeTimeMatrix + dataIdx)) + 
					   (*(dataMatrix + dataIdx + 1)));
		// set the remaining part to 0
		*(dataMatrix + dataIdx + 1) = 0;
		if (brnn){
		    dataIdx = dataIdx + dataDim/2;
		    // compute at+b
		    *(dataMatrix + dataIdx) = ((*(dataMatrix + dataIdx)) * 
					       (*(relativeTimeMatrix + dataIdx)) + 
					       (*(dataMatrix + dataIdx + 1)));
		    // set the remaining part to 0
		    *(dataMatrix + dataIdx + 1) = 0;
		}
	    }
	    // other parametrix representation can be written here
	}
    };
    
    struct paraGradient
    {
	// Given the gradients w.r.t to the transformed data, based on output of previous layer
	// calculate the gradients w.r.t the parameters in the equations
	// 
	// Dispatch T * NumberOfPara
	int     numPara;            // how many dimensions in output to be parameterized ?
	real_t *dataMatrix;         // not used now, reserved for more complex parametric funciton
	real_t *preGradMatrix;      // gradient propagated to the output of the previous layer
	real_t *relativeTimeMatrix; // timing matrix
	int    *paraConfig;         // configuration of paraConfig
	int     dataDim;            // #. rows of dataMatrix (dimension of previous layer)
	bool    brnn;

	__host__ __device__ void operator() (const int &Idx) const
	{
	    int timeIdx  = Idx / numPara;
	    int paraIdx  = Idx % numPara;
	    
	    int startDim = paraConfig[2*paraIdx];
	    int paraType = paraConfig[2*paraIdx+1];
	    
	    int dataIdx  = timeIdx * dataDim + startDim;
	    
	    // first order linear function at+b
	    if (paraType == PARAFUNCTION_TYPE_1){
		real_t grad                    = *(preGradMatrix + dataIdx);
		*(preGradMatrix + dataIdx)     = grad * (*(relativeTimeMatrix + dataIdx));
		*(preGradMatrix + dataIdx + 1) = grad;
		if (brnn){
		    dataIdx = dataIdx + dataDim / 2;
		    grad                           = *(preGradMatrix + dataIdx);
		    *(preGradMatrix + dataIdx)     = grad * (*(relativeTimeMatrix + dataIdx));
		    *(preGradMatrix + dataIdx + 1) = grad;

		}
	    }
	}
    };

    
    struct getRelativeTimeMatrix
    {
	// Generate the relative timing matrix based on the schedule of the previous Clock RNN layer
	// For example,
	//     phase1 phase1 phase1 phase1
	//     word1  word1  word2   word2
	//     frame1 frame2 frame3 frame4
	// The relative timing, w.r.t. to the begining of this segment is 
	//     0      1      2      3  (for all dimensions at phase level)
	//     0      1      0      1  (for all dimensions at word level)
	//     0      0      0      0  (for all dimensions at frame level)
	//
	// Dispatched over the dimensions of the output of the previous layer (Clock RNN layer)
	// 
	int*    bandConfig;     // configuration of the Clock RNN (previous layer)
	real_t* timeMatrixPtr;  // output timing matrix
	char*   timeStep;       // schedule of previous Clock RNN layer
	// Note, timeStep specify the begining of a segment for left-to-right case
	// for the right-to-left direction in brnn, the timeStep should be shifted by 1 frame
	// For example, for left-to-right case
	//     0      1      0       1
	//     word1  word1  word2   word2
	//     frame1 frame2 frame3 frame4
	// for right-to-left case
	//     1      0      1       0
	//     word1  word1  word2   word2
	//     frame1 frame2 frame3 frame4
        int    totalTime;      // total time of the currennt sentence
	int    bandNum;        // number of segments (band) 
	int    layerDim;       // dimension of the output of previous layer
	int    effectiveDim;   // if brnn, =layerDim/2; else, =layerDim
	
	
	// Dispatched over layerDim
        __host__ __device__ void operator() (const int &Idx) const
        {
	    // Idx is the dimension
	    
	    // which band this dimension is in
	    int  colStart    = 0;
	    int  colEnd      = 0;
	    int  tmp         = 0b01;
	    int  band        = 0;
	    
	    int effDim       = (Idx < effectiveDim)?(Idx):(Idx-effectiveDim);
	    for (; band<bandNum; band++){
		colStart = (band > 0)?(bandConfig[2*band-1]):(colStart);
		colEnd   = bandConfig[2*band+1];
		if (effDim >= colStart && effDim < colEnd)
		    break;
	    }
	    
	    // 
	    if (Idx < effectiveDim){
		int count = 0;
		for (int t = 0; t<totalTime; t++, count++){
		    if (timeStep[t] & (tmp<<band)){ 
			// normalize the time
			for (int i = 1; i<count; i++)
			    *(timeMatrixPtr + layerDim * (t-i) + Idx) /= (count-1);
			// boundary, set a new time starting point
			count = 0;
		    }
		    *(timeMatrixPtr + layerDim * t + Idx) = count;
		}
		for (int i = 1; i < count; i++)
		    *(timeMatrixPtr + layerDim * (totalTime-i) + Idx) /= (count-1);
		
	    }else{
		// for back-direction
		int count = 0;
		for (int t = totalTime-1; t>=0; t--, count++){
		    if (t== (totalTime-1) && (timeStep[0] & (tmp<<band))){
			count = 0;
		    }else if(timeStep[t+1] & (tmp<<band)){
			for (int i = 1; i<count; i++)
			    *(timeMatrixPtr + layerDim * (t+i) + Idx) /= (count-1);
			count = 0;
		    }
		    *(timeMatrixPtr + layerDim * t + Idx) = count;
		}
		for (int i = 1; i<count; i++)
		    *(timeMatrixPtr + layerDim * (-1+i) + Idx) /= (count-1);
		    
	    }
	}
	
    };

} // anonymous namespace
} // namespace internal


namespace layers {

    // Functions
    
    // GetParaNum: return the number of parameters for each type of parametric functions
    int GetParaNum(int ParaType){
	if (ParaType == PARAFUNCTION_TYPE_0){
	    return PARAFUNCTION_TYPE_0_NUMPARA;
	}else if (ParaType == PARAFUNCTION_TYPE_1){
	    return PARAFUNCTION_TYPE_1_NUMPARA;
	}else{
	    printf("paraConfig has invalid ParaType\n");
	    throw std::runtime_error("Currently only implemented ParaType = 0, 1");
	}
    }
    
    // InsertToVec: get the Dim_Type vector without duplicated elements
    // format: dim_paraType
    //  dim: the start dimension of one parametric function
    //  paraType: the type of the parametric funciton
    void InsertToVec(Cpu::int_vector &buffer, int startDim, int paraType){
	
	int paraNum = GetParaNum(paraType);
	
	if (buffer.size() == 0){
	    // buffer is empty now
	    buffer.push_back(startDim);
	    buffer.push_back(paraType);
	}else{
	    // 
	    if ((buffer.size()%2)!=0){
		throw std::runtime_error("Invalid Buffer in insertToVec");
	    }
	    int insertIdx = 0;
	    for (; insertIdx<(buffer.size()); insertIdx=insertIdx+2){
		
		// if the startDim is the same
		if (buffer[insertIdx] == startDim){
		    if (buffer[insertIdx+1] == paraType){
			// ignore duplidate specification
			break;
		    }else{
			// the same dimension was assigned with different parameter type
			printf("%d %d vs %d %d\n", 
			       buffer[insertIdx], buffer[insertIdx+1], startDim, paraType);
			throw std::runtime_error("Conflict para config, csase0");
		    }
		// the new element is on the right side of the element
		}else if (buffer[insertIdx] < startDim){
		    if ((buffer[insertIdx] + GetParaNum(buffer[insertIdx + 1])) > startDim){
			// the startDim is used another parameter function
			printf("%d %d vs %d %d\n", 
			       buffer[insertIdx], buffer[insertIdx+1], startDim, paraType);
			throw std::runtime_error("Conflict para config, case1");
		    }
		    
		    if ((insertIdx+2) == (buffer.size())){
			// insert to the end of buffer
			insertIdx = insertIdx + 2;
			break;
		    }
		    
		}else{
		    if ((startDim + paraNum) > buffer[insertIdx]){
			printf("%d %d vs %d %d\n", 
			       buffer[insertIdx], buffer[insertIdx+1], startDim, paraType);
			throw std::runtime_error("Conflict para config, case3");
		    }
		    if (insertIdx != 0){
			if ((buffer[insertIdx-2]+buffer[insertIdx-1])> startDim){
			    printf("%d %d vs %d %d\n", 
				   buffer[insertIdx], buffer[insertIdx+1], startDim, paraType);
			    throw std::runtime_error("Conflict para config, case4");
			}
		    }
		    insertIdx = insertIdx;
		    break;
		}
	    }
	    buffer.insert(buffer.begin()+insertIdx,   startDim);
	    buffer.insert(buffer.begin()+insertIdx+1, paraType);
	}
    }

    // Parse the parametric option. The output paraConfig is in format
    //    NumberOfDimForPara_
    //          Dim_NumberofElements_StartDim_EndDim_Type_StartDim_EndDim_Type_
    //          Dim_NumberofElements_StartDim_EndDim_Type_StartDim_EndDim_Type_
    // For example, 
    //    1 1 2 0 128 0 128 256 1
    //    one parametric dimension
    //        (1+1)th dimension, has 2 parametric parts, 0-128 of type 0, 128-256 of type 1
    //             
    // The other output is paraConfig2, it is in format
    //      startDim_Type_startDim_Type_
    // For example,
    //    0_0_1_0...127_0_128_1_130_1
    //    (0+1) dimension, type 0 parameterization
    //    (1+1) dimension, type 0 parameterization
    //    (128+1) dimension, type 1 parameterization
    //    (130+1) dimension, type 1 parameterization
    // Note, type1 requires 2 parameters, 
    // 129+1 is together with 128+1 dimension, as they are the parameters of one function
    void ReadParaOptions(const std::string options, 
			 Cpu::int_vector &paraConfig, Cpu::int_vector &paraConfig2)
    {
	// read in the option
	std::vector<std::string> tempArgs;
	std::vector<std::string> tempArgsSub;
	boost::split(tempArgs, options, boost::is_any_of("_"));
	if ((tempArgs.size() % 2) != 0){
	    printf("paraConfig should be outputDim_config_outputDim_config");
	    throw std::runtime_error("Error in ParaLayer");
	}
	
	paraConfig.clear();
	paraConfig.push_back(tempArgs.size()/2);
	
	paraConfig2.clear();
	for (int i=0; i < tempArgs.size(); i=i+2){
	    paraConfig.push_back(boost::lexical_cast<int>(tempArgs[i]));
	    boost::split(tempArgsSub, tempArgs[i+1], boost::is_any_of(","));
	    if ((tempArgsSub.size() % 3) != 0){
		printf("paraConfig's config should be startDim,endDim,order");
		throw std::runtime_error("Error in ParaLayer");
	    }
	    paraConfig.push_back(tempArgsSub.size() / 3);
	    for (int j=0;j<tempArgsSub.size();j=j+3){
		int startDim = boost::lexical_cast<int>(tempArgsSub[j]);
		int endDim   = boost::lexical_cast<int>(tempArgsSub[j+1]);
		int paraType = boost::lexical_cast<int>(tempArgsSub[j+2]);
		int paraNum  = GetParaNum(paraType);
		paraConfig.push_back(startDim);
		paraConfig.push_back(endDim);
		paraConfig.push_back(paraType);
		
		if (paraNum > 1 && paraType >= PARAFUNCTION_TYPE_1){
		    if ((endDim - startDim) % paraNum != 0){
			// paraNum: for 1-order and 2-order polynomial function
			throw std::runtime_error("startDim endDim doesn't match parameter number");
		    }else{
			int paraFuncNum = (endDim - startDim) / paraNum;
			for (int k = 0; k<paraFuncNum; k++){
			    InsertToVec(paraConfig2, startDim + k*paraNum, paraType);
			}
		    }
		}
	    }
	}
    }

    // Parse the option of ClockRNN
    // A lite version from RnnLayer
    void ReadClockRNNOptionsLite(const std::string options, Cpu::int_vector &m_crStep, 
				 const int size)
    {
	// read in the option
	std::vector<std::string> tempArgs;
	boost::split(tempArgs, options, boost::is_any_of("_"));
	if ((tempArgs.size() % 2) != 0){
	    printf("ClockRNN option should be TimeReso1_Dim1_TimeReso2_Dim2");
	    throw std::runtime_error("Error in RNNLayer");
	}
	m_crStep.resize(tempArgs.size(),-1);
	for (int i=0; i < tempArgs.size(); i++){
	    m_crStep[i] = boost::lexical_cast<int>(tempArgs[i]);
	}
	if (m_crStep[tempArgs.size()-1]!=size){
	    printf("ClockRNN options has unequal layer size: %d VS %d\n.Please check network.jsn", 
		   m_crStep[tempArgs.size()-1], size);
	    throw std::runtime_error("Error in RNNLayer");
	}
    }


    // class definition *************************************** 
    template <typename TDevice, typename TActFn>
    ParaLayer<TDevice, TActFn>::ParaLayer(const helpers::JsonValue &layerChild, 
					  const helpers::JsonValue &weightsSection,
					  Layer<TDevice> &precedingLayer,
					  int maxSeqLength,
					  int layerID)
        : FeedForwardLayer<TDevice, TActFn>(layerChild,
					    weightsSection,
					    precedingLayer,
					    maxSeqLength,
					    layerID)
    {
	
	// Read the configuration file
	m_paraConStr = ((layerChild->HasMember("paraconfig")) ? 
			((*layerChild)["paraconfig"].GetString()) : (""));
	
	// get ClockRNN state
	m_crStepStr  = ((layerChild->HasMember("clock")) ? 
			((*layerChild)["clock"].GetString()) : (""));
	
	if (m_paraConStr.size()>0){
	    Cpu::int_vector tmpConfig2;
	    ReadParaOptions(m_paraConStr, m_paraConfig, tmpConfig2);
	    m_paraConfigDev = m_paraConfig;
	    m_paraConfig2 = tmpConfig2;
	}else{
	    throw std::runtime_error("Could not find paraconfig in .jsn or .autosave file");
	}
	
	if (precedingLayer.type() == "brnn"){
	    m_brnn = true;
	}else if (precedingLayer.type() == "rnn"){
	    m_brnn = false;
	}else{
	    printf("ParaLayer can only follow a rnn or brnn clock layer\n");
	    throw std::runtime_error("Other cases are not implemented");
	}

	if (m_crStepStr.size()>0){
	    ReadClockRNNOptionsLite(m_crStepStr, m_crStep, 
				    this->precedingLayer().size()/(m_brnn?2:1));
	}else{
	    throw std::runtime_error("Could not find clock for ParaLayer in .jsn or .autosave");
	}
	m_crStepDevice = m_crStep;
	
	
	// create buffer for the relative time
	Cpu::real_vector tempTime(precedingLayer.maxSeqLength()*this->precedingLayer().size(), -1);
	m_relativeTime = tempTime;
	

	// set weight matrix elements to zero for those not used
	{{
	    this->m_weightMask.clear();	
	    this->m_weightMask.resize(this->weights().size(), 1.0);

	    // debug 
	    if (PARALAYERDEBUG){
		Cpu::real_vector temp = this->m_weightMask;
		printf("\n");
	    }

	    internal::setWeightZeroMatrix fn;
	    fn.paraConfig       = helpers::getRawPointer(m_paraConfigDev);
	    fn.weightMatrixMask = helpers::getRawPointer(this->m_weightMask);
	    fn.paraConfigSize   = m_paraConfig.size();
	    fn.preLayerSize     = this->precedingLayer().size();
	    fn.curLayerSize     = this->size();
	    fn.brnn             = this->m_brnn;
	    thrust::for_each(
	       thrust::counting_iterator<int>(0),
	       thrust::counting_iterator<int>(0) + m_paraConfig[0],
	       fn);
	    
	    // debug 
	    if (PARALAYERDEBUG){
		Cpu::real_vector temp = this->m_weightMask;
		printf("\n");
	    }
	    thrust::transform(this->m_weightMask.begin(), this->m_weightMask.end(),
			      this->weights().begin(),  this->weights().begin(),
			      thrust::multiplies<real_t>());
	}}
    }

    template <typename TDevice, typename TActFn>
    ParaLayer<TDevice, TActFn>::~ParaLayer()
    {
    }

    template <typename TDevice, typename TActFn>
    const std::string& ParaLayer<TDevice, TActFn>::type() const
    {
        static std::string s;

        if (s.empty()) {
            if (typeid(TActFn) == typeid(activation_functions::Identity))
                s = "paralayer";
	    else
                throw std::runtime_error("Unsupported activation function");
        }

        return s;
    }
    
    template <typename TDevice, typename TActFn>
    void ParaLayer<TDevice, TActFn>::loadSequences(const data_sets::DataSetFraction &fraction,
						   const int nnState)
    {
	TrainableLayer<TDevice>::loadSequences(fraction, nnState);
	
	if (fraction.auxDataDim()>0){
	    Cpu::pattype_vector clockTime = fraction.auxPattypeData();
	    if (clockTime.size() != this->curMaxSeqLength()){
		throw std::runtime_error("Error unequal length of clockTime size");
	    }
	    m_timeStep = clockTime;
	}else{
	    printf("ParaLayer requires Context-dependent clock schedule\n");
	    printf("Please check that Auxillary path, ext, type, dim are provided\n");
	    throw std::runtime_error("To be implemented for other cases");
	}

	// create the relative time for each dimension
	{{
	    // create the 1:T time matrix
	    /*internall::fillTimeMatrix fn1;
	    fn1.effectiveLayerSize = (this->m_brnn)?(this->size()/2):(this->size());
	    fn1.layerSize          = this->size();
	    fn1.totalTime          = this->curMaxSeqLength();
	    fn1.timeMatrixPtr      = helpers::getRawPointer(m_relativeTime);
	    thrust::for_each(
	      thrust::counting_iterator<int>(0),
	      thrust::counting_iterator<int>(0) + this->curMaxSeqLength() * this->size(),
	      fn1
	      );*/
		
	    internal::getRelativeTimeMatrix fn2;
	    fn2.bandConfig         = helpers::getRawPointer(m_crStepDevice);
	    fn2.timeMatrixPtr      = helpers::getRawPointer(m_relativeTime);
	    fn2.timeStep           = helpers::getRawPointer(m_timeStep);
	    fn2.totalTime          = this->curMaxSeqLength();
	    fn2.bandNum            = m_crStep.size()/2;
	    fn2.effectiveDim       = ((this->m_brnn)?
				      (this->precedingLayer().size()/2):
				      (this->precedingLayer().size()));
	    fn2.layerDim           = this->precedingLayer().size();
	    thrust::for_each(
			     thrust::counting_iterator<int>(0),
			     thrust::counting_iterator<int>(0) + this->precedingLayer().size(),
			     fn2
			     );

	    // debug 
	    if (PARALAYERDEBUG){
		Cpu::real_vector tempTime = m_relativeTime;
		int row,col;
		col = this->precedingLayer().size();
		row = this->curMaxSeqLength();
		for (int i = 0; i < row; i++){
		    for (int j=0; j< col; j++){
			printf("%3d ", (int)(100*tempTime[i*col + j]));
		    }
		    printf("\n");
		}
		printf("\n");
	    }
	}}
    }

    template <typename TDevice, typename TActFn>
    void ParaLayer<TDevice, TActFn>::computeForwardPass(const int nnState)
    {
	// transform the predicted parameters into feature trajectories
	{{
	    if (PARALAYERDEBUG){
		Cpu::real_vector temp = this->precedingLayer().outputs();
		printf("\n");
	    }
		
	    internal::paraTransform fn;
	    fn.numPara    = m_paraConfig2.size()/2;
	    fn.dataMatrix = helpers::getRawPointer(this->precedingLayer().outputs());
	    fn.paraConfig = helpers::getRawPointer(m_paraConfig2);
	    fn.dataDim    = this->precedingLayer().size();
	    fn.relativeTimeMatrix = helpers::getRawPointer(m_relativeTime);
	    fn.brnn       = this->m_brnn;
	    thrust::for_each(
	       thrust::counting_iterator<int>(0),
	       thrust::counting_iterator<int>(0) + this->curMaxSeqLength() * m_paraConfig2.size()/2,
	       fn);
	    
	    if (PARALAYERDEBUG){
		Cpu::real_vector temp = this->precedingLayer().outputs();
		printf("\n");
	    }   
	    
	}}
	
	// normal feedforward transformation
	this->FeedForwardLayer<TDevice,TActFn>::computeForwardPass(nnState);
    }

    template <typename TDevice, typename TActFn>
    void ParaLayer<TDevice, TActFn>::computeForwardPass(const int timeStep, const int nnState)
    {
	
    }

    template <typename TDevice, typename TActFn>
    void ParaLayer<TDevice, TActFn>::computeBackwardPass(const int nnState)
    {
	// normal backward computation
	this->FeedForwardLayer<TDevice,TActFn>::computeBackwardPass(nnState);
	
	
	// compute the gradient for the para representation layers
	// change the weight matrix
	{{
	    thrust::transform(this->m_weightMask.begin(),  this->m_weightMask.end(),
			      this->_weightUpdates().begin(),  this->_weightUpdates().begin(),
			      thrust::multiplies<real_t>());

	}}

	//
	// transform the predicted parameters into feature trajectories
	{{
	    internal::paraGradient fn;
	    fn.numPara       = m_paraConfig2.size()/2;
	    fn.dataMatrix    = helpers::getRawPointer(this->precedingLayer().outputs());
	    fn.preGradMatrix = helpers::getRawPointer(this->precedingLayer().outputErrors());
	    fn.paraConfig    = helpers::getRawPointer(m_paraConfig2);
	    fn.dataDim       = this->precedingLayer().size();
	    fn.relativeTimeMatrix = helpers::getRawPointer(m_relativeTime);
	    fn.brnn          = this->m_brnn;
	    thrust::for_each(
	       thrust::counting_iterator<int>(0),
	       thrust::counting_iterator<int>(0) + this->curMaxSeqLength() * m_paraConfig2.size()/2,
	       fn);
	       
	}} 
    }

    template <typename TDevice, typename TActFn>
    void ParaLayer<TDevice, TActFn>::exportLayer(const helpers::JsonValue &layersArray, 
						 const helpers::JsonAllocator &allocator) const
    {
        TrainableLayer<TDevice>::exportLayer(layersArray, allocator);
        (*layersArray)[layersArray->Size() - 1].AddMember("clock", m_crStepStr.c_str(), 
							  allocator);
	(*layersArray)[layersArray->Size() - 1].AddMember("paraconfig", m_paraConStr.c_str(), 
							  allocator);
    }

    // explicit template instantiations
    template class ParaLayer<Cpu, activation_functions::Tanh>;
    template class ParaLayer<Gpu, activation_functions::Tanh>;
    template class ParaLayer<Cpu, activation_functions::Logistic>;
    template class ParaLayer<Gpu, activation_functions::Logistic>;
    template class ParaLayer<Cpu, activation_functions::Identity>;
    template class ParaLayer<Gpu, activation_functions::Identity>;
    template class ParaLayer<Cpu, activation_functions::Relu>;
    template class ParaLayer<Gpu, activation_functions::Relu>;

} // namespace layers
