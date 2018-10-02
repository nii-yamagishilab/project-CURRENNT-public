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

#ifndef LAYERS_CNNLAYER_HPP
#define LAYERS_CNNLAYER_HPP

#include "TrainableLayer.hpp"
#include <boost/shared_ptr.hpp>

namespace layers {

    /******************************************************************************************//**
     * CNN layer 
     *********************************************************************************************/
    template <typename TDevice>
    class CNNLayer : public TrainableLayer<TDevice>
    {
	typedef typename TDevice::real_vector real_vector;
	typedef typename Cpu::real_vector cpu_real_vector;
	typedef typename TDevice::int_vector int_vector;
	typedef typename Cpu::int_vector cpu_int_vector;

    protected:
	int             m_winNumOrignal;  //
	
	cpu_int_vector  m_winWidth_H;     // filter dimension (width of the filter window)
	int_vector      m_winWidth_D;     // (dimension = this->size())
	std::string     m_winWidth_Opt;   //

	cpu_int_vector  m_winWidth_Cum_H;   // cumsum of the filter dimension
	int_vector      m_winWidth_Cum_D;   // cumsum of the filter dimension
	
	cpu_int_vector  m_winHeight_H;    // filter height
	int_vector      m_winHeight_D;    // (dimension = this->size())
	std::string     m_winHeight_Opt;

	cpu_int_vector  m_winStride_H;    //
	int_vector      m_winStride_D;    //
	std::string     m_winStride_Opt;  //
	
	cpu_int_vector  m_winIndex_H;     // index of the shifted filter in the original filter set
	int_vector      m_winIndex_D;     // (dimension = this->size())
 
	cpu_int_vector  m_winShiftIndex_H;// shift of the filter bias
	int_vector      m_winShiftIndex_D;// (dimension = this->size())
	cpu_int_vector  m_winShiftRevId_H;// shift of the filter bias 
	int_vector      m_winShiftRevId_D;// (dimension = this->size())

	
	cpu_int_vector  m_winColIndex_H;  // index of each column to the original filter weight
	int_vector      m_winColIndex_D;  // (dimension = winTotalL)
	cpu_int_vector  m_winRowIndex_H;  // index of each column to the original filter weight
	int_vector      m_winRowIndex_D;  // (dimension = winTotalL)
	cpu_int_vector  m_winColHeight_H; // same as m_winHeight, but duplicated for each column
	int_vector      m_winColHeight_D; // (dimension = winTotalL)
	cpu_int_vector  m_winWidthCol_H;
	int_vector      m_winWidthCol_D;
	cpu_int_vector  m_winShiftNum_H;
	int_vector      m_winShiftNum_D;


	real_vector     m_weightBuffer;   // buffer to store the large weight matrix
	cpu_int_vector  m_wCopyInfo_H;
	int_vector      m_wCopyInfo_D;    // buffer to store the information for copying weights
	int             m_weightNum;      // total number of weights of filter
	int             m_biasPos;
	int_vector      m_weightFilter_map;

	int             m_weightBufferNum;
	int             m_biasPosInBuffer;// position of bias in the weight buffer
	
	cpu_int_vector  m_winInterval_H;  // interval between window 
	int_vector      m_winInterval_D;
	std::string     m_winInterval_Opt;

	//int_vector      m_weightIdx;      // idx to access the weight of each window filter

	real_vector     m_conBuffer;        // data buffer
	int             m_winTotalL;        // sum of the width of filter

	int             m_causalFlag;       // whether the CNN filter is casual filter
	
	int             m_outputTanh;       //

	int             m_1DCNNOnly;        // whether the CNN is only 1-D


	void  __allocateLocalMem();
	void  __clearLocalMem();
	
    public:
	// initializer and destructor
	CNNLayer(const helpers::JsonValue &layerChild,
		 const helpers::JsonValue &weightsSection,
		 Layer<TDevice> &precedingLayer, int maxSeqLength, int layerID);

	virtual ~CNNLayer();

	virtual const std::string& type() const;
	
	virtual void computeForwardPass(const int nnState);

	virtual void computeForwardPass(const int timeStep, const int nnState);
	
	virtual void computeBackwardPass(const int nnState);

        virtual void loadSequences(const data_sets::DataSetFraction &fraction, const int nnState);

	// export
	virtual void exportLayer(const helpers::JsonValue &layersArray, 
				 const helpers::JsonAllocator &allocator) const;

	// memory save mode for generation in wavenet
	virtual void reduceOutputBuffer();

	virtual int  outputBufPtrBias(const int timeStepTimesParallel, const int nnState);

	void clearAllBuffers();

	void resizeAllBuffers(const int timeLength);
	
    };
    
    
}
#endif
