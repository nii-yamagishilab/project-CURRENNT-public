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

#include "MIDIInterfaceLayer.hpp"

#include "../helpers/getRawPointer.cuh"
#include "../helpers/Matrix.hpp"
#include "../helpers/JsonClasses.hpp"
#include "../helpers/misFuncs.hpp"
#include "../activation_functions/Logistic.cuh"
#include "../activation_functions/Tanh.cuh"
#include "../MacroDefine.hpp"

#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/for_each.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/fill.h>
#include <thrust/random.h>
#include <boost/foreach.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>
#include <vector>
#include <stdexcept>

#include "../Configuration.hpp"

#define PI_DEFINITION 3.141592654f
#define NN_MIDIINTERFACE_LAYER_MIDI_NUM 128
#define NN_MIDIINTERFACE_VELOCITY_NUM 128
#define NN_MIDIINTERFACE_OCTAVE_NOTE_NUM 12
#define NN_MIDIINTERFACE_MAXIMUM_F0 12543
#define NN_MIDIINTERFACE_MAXIMUM_HAR_NUM 128
namespace internal{
namespace{
    
    struct genNoise_Gaussian
    {
	float a, b;
	int   seed;
	
	__host__ __device__
	genNoise_Gaussian(float _a=-1.f, float _b=1.f, int _seed=123) : a(_a), b(_b), seed(_seed) {};

	__host__ __device__
	float operator()(const unsigned int n) const
	{
	    thrust::default_random_engine rng(seed);
	    thrust::normal_distribution<float> dist(a, b);
	    rng.discard(n);
	    return dist(rng);
	}
    };

    // Convert input piano-roll to sine waveforms
    // Only parallel over 128 midi channels, not good
    struct sinWaveGenerator_v1
    {
	real_t  freqSR;
	real_t  periodicMag;
	int     signalDim;
	int     parallel;
	int     seqLength;
	int     withHar;
	
	real_t     *pianoroll;
	real_t     *outputBuff;
	const char *patTypes;
	
	__host__ __device__ void operator() (const thrust::tuple<real_t&, int> &t) const
	{
	    
	    int dimIdx      = t.get<1>() % signalDim;
	    int paraBlock   = t.get<1>() / signalDim;
	    int midi_idx    = dimIdx;
	    int timeIdxPhy  = 0;
	    int harmonicNum = 1;
	    
	    // accumulate of phase, simply using the noise as initial phase
	    real_t spPhase = outputBuff[dimIdx];
	    real_t sigValue = 0.0;
	    
	    // frequenct of this piano-roll track
	    // from Librosa.midi_to_hz
	    // 440.0 * (2.0 ** ((np.asanyarray(notes) - 69.0)/12.0))
	    real_t freq = 440.0 * pow(2.0, (midi_idx - 69.0)/12.0);
	    
	    if (withHar){
		for (;;harmonicNum++){
		    if (harmonicNum * freq > NN_MIDIINTERFACE_MAXIMUM_F0){
			harmonicNum -= 1;
			break;
		    }
		}
	    }
	    
	    // Generate sine waveform step by step
	    for (int timeIdx = 0; timeIdx < seqLength; timeIdx++){
		
		// Time step index (with parallel utterances into consideration)
		timeIdxPhy = timeIdx * parallel + paraBlock;
		
		if (patTypes[timeIdxPhy] == PATTYPE_NONE)
		    continue;

		// accumulate phase + phase nosise
		spPhase += (2.0 * PI_DEFINITION * (freq / freqSR)); 
		if (spPhase > 2.0 * PI_DEFINITION){
		    spPhase = spPhase - 2.0 * PI_DEFINITION;
		}else if (spPhase < -2.0 * PI_DEFINITION){
		    spPhase = spPhase + 2.0 * PI_DEFINITION;
		}else{
		    // nothing
		}

		for (int harIdx = 1; harIdx <= harmonicNum; harIdx++){
		    sigValue = cos(spPhase * harIdx) * periodicMag;
		    outputBuff[timeIdxPhy * signalDim + dimIdx] =
			outputBuff[timeIdxPhy * signalDim + dimIdx] +
			sigValue * pianoroll[timeIdxPhy * signalDim + dimIdx] /
			NN_MIDIINTERFACE_VELOCITY_NUM / harmonicNum;
		}
	    }   
	}
    };

    // Convert input piano-roll to sine waveforms
    // Parallel over all time steps
    struct sinWaveGenerator_v2
    {
	real_t  freqSR;
	real_t  periodicMag;
	int     signalDim;
	int     parallel;
	int     withHar;

	real_t     *pianoroll;
	real_t     *phaNoise;
	const char *patTypes;
	
	__host__ __device__ void operator() (const thrust::tuple<real_t&, int> &t) const
	{

	    // MIDI dimension 
	    int dimIdx        = t.get<1>() % signalDim;

	    // parallel block
	    int parallelBlock = t.get<1>() / signalDim;
	    
	    if (patTypes[parallelBlock] == PATTYPE_NONE){
		t.get<0>() = 0.0;
		return;
	    }	
	    
	    // time step t
	    int timeIdxPhy  = parallelBlock / parallel;
	    
	    int midi_idx    = dimIdx;
	    int harmonicNum = 1;
	    
	    real_t spPhase = phaNoise[midi_idx * NN_MIDIINTERFACE_LAYER_MIDI_NUM];
	    real_t sigValue = 0.0;
	    
	    // frequenct of this piano-roll track
	    // from Librosa.midi_to_hz
	    // 440.0 * (2.0 ** ((np.asanyarray(notes) - 69.0)/12.0))
	    real_t freq = 440.0 * pow(2.0, (midi_idx - 69.0)/12.0);
	    
	    if (withHar){
		for (;;harmonicNum++){
		    if (harmonicNum * freq > NN_MIDIINTERFACE_MAXIMUM_F0 ||
			harmonicNum > NN_MIDIINTERFACE_MAXIMUM_HAR_NUM){
			harmonicNum -= 1;
			break;
		    }
		}
	    }
	    
	    for (int harIdx = 1; harIdx <= harmonicNum; harIdx++){
		spPhase  = phaNoise[midi_idx * NN_MIDIINTERFACE_MAXIMUM_HAR_NUM +
				    harIdx - 1];
		spPhase += ((timeIdxPhy / freqSR) * freq * harIdx);
		spPhase  = spPhase - nearbyintf(spPhase);
		sigValue = cos(spPhase  * 2.0 * PI_DEFINITION) * periodicMag;
		t.get<0>() = t.get<0>() + 
		    sigValue * pianoroll[t.get<1>()] /
		    NN_MIDIINTERFACE_VELOCITY_NUM / harmonicNum;
	    }
	}
    };



    struct sumSignal
    {
	int signalDim;
	int parallel;
	
	real_t *outputBuff;
	const char *patTypes;
	
	__host__ __device__ void operator() (const thrust::tuple<real_t&, int> &t) const
	{
	    int timeIdx   = t.get<1>();
		
	    t.get<0>() = 0.0;
	    if (patTypes[timeIdx] == PATTYPE_NONE){
		return;
	    }else{
		for (int dimIdx = 0; dimIdx < signalDim; dimIdx++)
		    t.get<0>() += outputBuff[timeIdx * signalDim + dimIdx];
	    }
	}
    };

}    
}


namespace layers{

	
    
    template <typename TDevice>
    MIDIInterfaceLayer<TDevice>::MIDIInterfaceLayer(
				const helpers::JsonValue &layerChild,
				const helpers::JsonValue &weightsSection,
				Layer<TDevice>           &precedingLayer,
				int                       maxSeqLength,
				int                       layerID)
	: TrainableLayer<TDevice>(layerChild, weightsSection, 0, 0,
				  precedingLayer, maxSeqLength, layerID)
	, m_noiseMag        (1.0)
	, m_periodicMag     (1.0)
	, m_freqSR          (-1.0)
    {
	// load options
	this->__loadOpts(layerChild);
	
	// set layer mode
	this->__setLayerMode();

	// allocate memory
	this->__allocateLocalMem();

	// print information
	this->__printOpts();
    }

    template <typename TDevice>
    MIDIInterfaceLayer<TDevice>::~MIDIInterfaceLayer()
    {
    }
    
    template <typename TDevice>
    void MIDIInterfaceLayer<TDevice>::__loadOpts(const helpers::JsonValue &layerChild)
    {

	m_noiseMag = (layerChild->HasMember("frequencyNoiseMag") ? 
		      static_cast<real_t>((*layerChild)["frequencyNoiseMag"].GetDouble()):1.0);
	
	m_periodicMag = (layerChild->HasMember("frequencyPeriodicMag") ? 
			 static_cast<real_t>((*layerChild)["frequencyPeriodicMag"].GetDouble()):1.0);

	m_freqSR  = (layerChild->HasMember("frequencySR")?
		     static_cast<real_t>((*layerChild)["frequencySR"].GetDouble()) : -1);

	m_withHarmonics = (layerChild->HasMember("frequencyWithHarmonics")?
			   static_cast<int>((*layerChild)["frequencyWithHarmonics"].GetInt()) : 0);
	
	m_MIDINoteNum = this->precedingLayer().size();

	if (m_MIDINoteNum != NN_MIDIINTERFACE_LAYER_MIDI_NUM){
	    printf("\n\tInput piano-roll is not %d dimensions",
		   NN_MIDIINTERFACE_LAYER_MIDI_NUM);    
	    throw std::runtime_error("\n\tError in CURRENNT: invalid piano-roll");
	}

	if (this->size() != 1 && this->size() != m_MIDINoteNum){
	    printf("\n\tmidi_interface layer size can only be 1 or %d", m_MIDINoteNum);
	    throw std::runtime_error("\n\tError in CURRENNT: invalid network.jsn");
	}
	
    }

    template <typename TDevice>
    void MIDIInterfaceLayer<TDevice>::__printOpts()
    {
	
	printf("\n\tMIDI interface: convert piano-roll to source signal\n");
	printf("\n\tPeriodic mag:%f", m_periodicMag);
	printf("\n\tNoise mag: %f", m_noiseMag);
	printf("\n\tWith harmonics: %d", m_withHarmonics);
	if (this->size() == 1)
	    printf("\n\tProduce a single track of waveform");
	else
	    printf("\n\tProduce one track for each MIDI note");
    }
    
    template <typename TDevice>
    void MIDIInterfaceLayer<TDevice>::__setLayerMode()
    {
	
    }
    
    template <typename TDevice>
    void MIDIInterfaceLayer<TDevice>::__allocateLocalMem()
    {
	m_periodicBuff.resize(this->outputs().size()/this->size() * m_MIDINoteNum, 0.0);
	m_phaseNoiseBuff.resize(NN_MIDIINTERFACE_LAYER_MIDI_NUM * NN_MIDIINTERFACE_MAXIMUM_HAR_NUM, 0.0);
	// m_noiseBuff.clear();
    }

    template <typename TDevice>
    void MIDIInterfaceLayer<TDevice>::__clearLocalMem()
    {
	m_periodicBuff.clear(); m_periodicBuff.shrink_to_fit();
	//m_noiseBuff.clear(); m_noiseBuff.shrink_to_fit();
    }

    template <typename TDevice>
    void MIDIInterfaceLayer<TDevice>::exportLayer(const helpers::JsonValue     &layersArray, 
						  const helpers::JsonAllocator &allocator) const
    {
        TrainableLayer<TDevice>::exportLayer(layersArray, allocator);
	
	(*layersArray)[layersArray->Size() - 1].AddMember("frequencyPeriodicMag",
							  m_periodicMag,
							  allocator);
	
	(*layersArray)[layersArray->Size() - 1].AddMember("frequencyNoiseMag",
							  m_noiseMag,
							  allocator);
	
	(*layersArray)[layersArray->Size() - 1].AddMember("frequencySR",
							  m_freqSR,
							  allocator);
	
	(*layersArray)[layersArray->Size() - 1].AddMember("frequencyWithHarmonics",
							  m_withHarmonics,
							  allocator);
    }

    template <typename TDevice>
    void MIDIInterfaceLayer<TDevice>::loadSequences(const data_sets::DataSetFraction &fraction,
						const int nnState)
    {
	TrainableLayer<TDevice>::loadSequences(fraction, nnState);
    }
    
    template <typename TDevice>
    const std::string& MIDIInterfaceLayer<TDevice>::type() const
    {
        static std::string s;
        if (s.empty()) s = "midi_interface";
        return s;
    }

    template <typename TDevice>
    void MIDIInterfaceLayer<TDevice>::computeForwardPass(const int nnState)
    {
	int timeLength      = this->curMaxSeqLength();
	int timeLengthTotal = timeLength * this->parallelSequences();
	int signalDimTotal  = m_MIDINoteNum * this->parallelSequences();
	int midiLenghTotal  = timeLengthTotal * m_MIDINoteNum;
	
	// output buffer
	thrust::fill(this->outputs().begin(), this->outputs().end(), 0.0);
	thrust::fill(m_periodicBuff.begin(), m_periodicBuff.end(), 0.0);
	
	// generate additive noise
	thrust::counting_iterator<unsigned int> index_sequence_begin(0);
	
	thrust::transform(index_sequence_begin,
			  index_sequence_begin +
			  timeLength * signalDimTotal,
			  m_periodicBuff.begin(),
			  internal::genNoise_Gaussian(0.0, m_noiseMag/3.0,
						      (int)(misFuncs::GetRandomNumber()*10000.0)));

	// for random phase
	thrust::transform(index_sequence_begin,
			  index_sequence_begin + m_phaseNoiseBuff.size(),
			  m_phaseNoiseBuff.begin(),
			  internal::genNoise_Gaussian(0.0, 1.0,
						      (int)(misFuncs::GetRandomNumber()*10000.0)));
	
	{
	internal::sinWaveGenerator_v2 fn1;
	fn1.signalDim   = m_MIDINoteNum;		
	fn1.freqSR      = (real_t)this->m_freqSR;
	fn1.parallel    = this->parallelSequences();
	fn1.periodicMag = this->m_periodicMag;
	fn1.withHar     = this->m_withHarmonics;
	
	fn1.pianoroll   = helpers::getRawPointer(this->precedingLayer().outputs());
	fn1.phaNoise    = helpers::getRawPointer(this->m_phaseNoiseBuff);
	fn1.patTypes    = helpers::getRawPointer(this->patTypes());
	
	thrust::for_each(
	  thrust::make_zip_iterator(
	    thrust::make_tuple(m_periodicBuff.begin(),
			       thrust::counting_iterator<int>(0))),
	  thrust::make_zip_iterator(
	    thrust::make_tuple(m_periodicBuff.begin()            + midiLenghTotal,
			       thrust::counting_iterator<int>(0) + midiLenghTotal)),
	  fn1);
	}

	if (this->size() == 1){
	    internal::sumSignal fn2;
	    fn2.signalDim = m_MIDINoteNum;
	    fn2.parallel   = this->parallelSequences();	    
	    fn2.patTypes   = helpers::getRawPointer(this->patTypes());
	    fn2.outputBuff = helpers::getRawPointer(this->m_periodicBuff);

	    thrust::for_each(
	     thrust::make_zip_iterator(
	      thrust::make_tuple(
		this->outputs().begin(),
		thrust::counting_iterator<int>(0))),
	     thrust::make_zip_iterator(
	      thrust::make_tuple(
		this->outputs().begin()           + timeLengthTotal,
		thrust::counting_iterator<int>(0) + timeLengthTotal)),
	     fn2);
	}else{
	    this->_outputs().swap(m_periodicBuff);
	}
    }

    template <typename TDevice>
    void MIDIInterfaceLayer<TDevice>::computeForwardPass(const int timeStep, const int nnState)
    {
	printf("\n\tmidi_interface::computeForwardPass(timeStep) not implemented");
	throw std::runtime_error("\n\tError: CURRENNT implementation error");

    }

    template <typename TDevice>
    void MIDIInterfaceLayer<TDevice>::computeBackwardPass(const int nnState)
    {	
	// do nothing
    }

    template <typename TDevice>
    void MIDIInterfaceLayer<TDevice>::computeBackwardPass(const int timeStep, const int nnState)
    {	
	// do nothing
    }

    template <typename TDevice>
    void MIDIInterfaceLayer<TDevice>::reduceOutputBuffer()
    {
	throw std::runtime_error("midi_interface not implemented");
    }
    
    template <typename TDevice>
    int MIDIInterfaceLayer<TDevice>::outputBufPtrBias(const int timeStepTimesParallel,
						      const int nnState)
    {
	if (this->getSaveMemoryFlag()){
	    return timeStepTimesParallel * this->size();
	}else{
	    return 0;
	}
    }	


    template <typename TDevice>
    void MIDIInterfaceLayer<TDevice>::clearAllBuffers()
    {
	this->clearOutputBuffer();
	this->__clearLocalMem();
    }

    template <typename TDevice>
    void MIDIInterfaceLayer<TDevice>::resizeAllBuffers(const int timeLength)
    {
	this->resizeOutputBuffer(timeLength * this->parallelSequences() * this->size());
	this->__allocateLocalMem();
    }

    template <typename TDevice>
    void MIDIInterfaceLayer<TDevice>::logAllBuffers(helpers::vecPoolManager<TDevice> &vecPoolMng,
						    bool flag_add)
    {
	// for output buffer
	Layer<TDevice>::logAllBuffers(vecPoolMng, flag_add);

	// noiseInput, phaseNoise
	vecPoolMng.addOrRemoveNewVec(m_MIDINoteNum, flag_add);
	//vecPoolMng.addOrRemoveNewVec(m_MIDINoteNum, flag_add);
    }
    
    template <typename TDevice>
    void MIDIInterfaceLayer<TDevice>::swapAllBuffers(helpers::vecPoolManager<TDevice> &vecPoolMng,
						 bool flag_get)
    {
	Layer<TDevice>::swapAllBuffers(vecPoolMng, flag_get);
	
	vecPoolMng.getSwapVector(m_periodicBuff, this->getLayerID(), m_MIDINoteNum, flag_get);
	//vecPoolMng.getSwapVector(m_noiseBuff, this->getLayerID(), m_MIDINoteNum, flag_get);
    }
    
    template class MIDIInterfaceLayer<Cpu>;
    template class MIDIInterfaceLayer<Gpu>;
    
}

