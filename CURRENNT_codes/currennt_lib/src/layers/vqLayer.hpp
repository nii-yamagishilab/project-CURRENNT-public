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
/*
 */

#ifndef LAYERS_VQLAYER_HPP
#define LAYERS_VQLAYER_HPP


#include "TrainableLayer.hpp"

namespace layers{

    template <typename TDevice>
    class vqLayer : public TrainableLayer<TDevice>
    {
	typedef typename TDevice::real_vector     real_vector;
	typedef typename Cpu::real_vector         cpu_real_vector;
	typedef typename TDevice::int_vector      int_vector;
	typedef typename TDevice::pattype_vector  pattype_vector;
	typedef typename Cpu::int_vector          cpu_int_vector;

    private:
	void __loadOpts(const helpers::JsonValue &layerChild);
	void __setUp_encoder_inputs();
	void __setUp_allocateMem();
	void __showOpts();

	void __computeForwardPass_OneBestOneFold(const int nnState);
	void __computeForwardPass_NBestOneFold(const int nnState);
	void __computeForwardPass_OneBestNFold(const int nnState);
	
    public:

	const int        m_vqCodeBookSizeAll;     // size of the code book in total
	cpu_int_vector   m_vqCodeBookSizeVec; // size of each VQ code book
	std::string      m_vqCodeBookSizeStr;     // size of the VQ code book string
	std::string      m_preLayerNamesStr;
	std::string      m_codeBookStatusStr;     
	std::vector <TrainableLayer<TDevice>*> m_preLayers;
	
	real_vector  m_disMatrix;        // distance matrix
	real_vector  m_selectedIdxF;
	
	int_vector   m_selectedIdx;
	int_vector   m_codeDimStatus;    //

	real_vector  m_softmaxTmp;
	real_vector  m_softmaxRand;
	
	real_t       m_betaPara;
	real_t       m_codeError;
	real_t       m_lambdaPara;
	
	int          m_bestNcode;        //
	int          m_codebookNfold;    // 
	vqLayer(
	    const helpers::JsonValue &layerChild,
	    const helpers::JsonValue &weightsSection,
            Layer<TDevice>           &precedingLayer,
	    int                       maxSeqLength,
	    int                       layerID
	);

	virtual ~vqLayer();
	
	
	virtual const std::string& type() const;
	
	// NN forward
	virtual void computeForwardPass(const int nnState);
	
	// NN forward, per frame
	virtual void computeForwardPass(const int timeStep, const int nnState);
	
	// NN backward
	virtual void computeBackwardPass(const int nnState);

	// NN backward
	virtual void computeBackwardPass(const int timeStep, const int nnState);

	// Load sequences
        virtual void loadSequences(const data_sets::DataSetFraction &fraction, const int nnState);

	// export
	virtual void exportLayer(const helpers::JsonValue &layersArray, 
				 const helpers::JsonAllocator &allocator) const;

	virtual void reReadWeight(const helpers::JsonValue &weightsSection, const int layerSize, 
				  const int readCtrFlag);

	virtual real_t codeError() const;

	virtual real_vector& codeIdx();

	virtual int codeBookNum();
    };
    
}

#endif
