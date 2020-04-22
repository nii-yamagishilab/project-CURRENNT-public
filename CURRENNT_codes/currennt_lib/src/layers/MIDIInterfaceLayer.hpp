/******************************************************************************
 * This file is an addtional component of CURRENNT. 
 * Xin WANG
 * National Institute of Informatics, Japan
 * 2016 - 2020
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
/*
 */

#ifndef LAYERS_MIDIINTERFACELAYER_HPP
#define LAYERS_MIDIINTERFACELAYER_HPP


#include "TrainableLayer.hpp"

namespace layers{
    
    template <typename TDevice>
    class MIDIInterfaceLayer : public TrainableLayer<TDevice>
    {
	typedef typename TDevice::real_vector     real_vector;
	typedef typename Cpu::real_vector         cpu_real_vector;
	typedef typename TDevice::int_vector      int_vector;
	typedef typename TDevice::pattype_vector  pattype_vector;
	typedef typename Cpu::int_vector          cpu_int_vector;

    private:
	real_vector     m_periodicBuff;
	real_vector     m_phaseNoiseBuff;
	real_t          m_noiseMag;
	real_t          m_periodicMag;
	real_t          m_freqSR;
	int             m_MIDINoteNum;
	int             m_withHarmonics;
	
	void __loadOpts(const helpers::JsonValue &layerChild);

	void __printOpts();
	
	void __setLayerMode();
	
	void __allocateLocalMem();
	
	void __clearLocalMem();
	
    public:
	MIDIInterfaceLayer(
	    const helpers::JsonValue &layerChild,
	    const helpers::JsonValue &weightsSection,
            Layer<TDevice>           &precedingLayer,
	    int                       maxSeqLength,
	    int                       layerID
	);

	virtual ~MIDIInterfaceLayer();
	
	
	virtual const std::string& type() const;
	
	// NN forward
	virtual void computeForwardPass(const int nnState);
	
	// NN forward, per frame
	virtual void computeForwardPass(const int timeStep, const int nnState);
	
	// NN backward
	virtual void computeBackwardPass(const int nnState);

	// NN backward
	virtual void computeBackwardPass(const int timeStep, const int nnState);

	// export
	virtual void exportLayer(const helpers::JsonValue &layersArray, 
				 const helpers::JsonAllocator &allocator) const;
	//
	virtual void loadSequences(const data_sets::DataSetFraction &fraction, const int nnState);

	virtual void reduceOutputBuffer();
	
	virtual int outputBufPtrBias(const int timeStepTimesParallel, const int nnState);

	void clearAllBuffers();

	void resizeAllBuffers(const int timeLength);

	virtual void logAllBuffers(helpers::vecPoolManager<TDevice> &vecPoolMng,
				   bool flag_add);
	
	virtual void swapAllBuffers(helpers::vecPoolManager<TDevice> &vecPoolMng,
				    bool flag_get);	
    };
    
}

#endif
