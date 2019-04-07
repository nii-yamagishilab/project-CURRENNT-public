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

#include "Maxpooling.hpp"

#include "../helpers/getRawPointer.cuh"
#include "../helpers/Matrix.hpp"
#include "../helpers/min.cuh"
#include "../helpers/max.cuh"
#include "../helpers/safeExp.cuh"
#include "../helpers/NumericLimits.cuh"
#include "../helpers/JsonClasses.hpp"
#include "../helpers/misFuncs.hpp"

#include "../activation_functions/Tanh.cuh"
#include "../activation_functions/Logistic.cuh"
#include "../activation_functions/Identity.cuh"
#include "../activation_functions/Relu.cuh"

#include "../Configuration.hpp"

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

#include <sstream>
#include <fstream>
#include <cmath>


namespace internal{
namespace {

    struct maxpoolingForward
    {
	int   featDim;
	int   parall;
	int   maxLength;
	int   maxWidth;
	int  *widthVec;
	int  *strideVec;

	real_t     *inputData;
	const char *patTypes;
	int        *maxIndex;
	
	__host__ __device__ void operator() (const thrust::tuple<real_t&, int> &t) const
	{
	    int outputIdx = t.get<1>();
	    int timeIdx   = outputIdx / featDim;
	    int dimIdx    = outputIdx % featDim;

	    int width     = widthVec[dimIdx];
	    int stride    = strideVec[dimIdx];
	    
	    if (patTypes[timeIdx] == PATTYPE_NONE)
		return;

	    // Whether this timestep is between the stride
	    if ((timeIdx % stride)){
		t.get<0>() = 0.0;
		return;
	    }

	    // for each output unit, find the maximum in the maxpooling window
	    int idx;
	    int maxIdx = 0;
	    real_t maxValue = -1 * helpers::NumericLimits<real_t>::max();
	    for (int i = -1 * width; i <= width; i++){
		idx = outputIdx + i * parall * featDim;
		if (idx >= 0 && idx < (maxLength * parall * featDim) &&
		    patTypes[timeIdx + i * parall] != PATTYPE_NONE &&
		    inputData[idx] > maxValue)
		{
		    maxValue = inputData[idx];
		    maxIdx   = i;
		}
	    }
	    t.get<0>() = maxValue;
	    
	    // for each input unit, loggs down for which unit it is used
	    maxIndex[(outputIdx + maxIdx * parall * featDim) * (2*maxWidth+1) - maxIdx + width] = 1;
	}
    };

    struct maxpoolingBackward
    {
	int  featDim;
	int  parall;
	int  maxLength;
	int  maxWidth;
	
	int  *widthVec;
	int  *strideVec;
	real_t     *inputData;
	const char *patTypes;
	int        *maxIndex;
	
	__host__ __device__ void operator() (const thrust::tuple<real_t&, int> &t) const
	{
	    int outputIdx = t.get<1>();
	    int timeIdx   = outputIdx / featDim;
	    int dimIdx    = outputIdx % featDim;

	    int width     = widthVec[dimIdx];
	    //int stride    = strideVec[dimIdx];
	    
	    if (patTypes[timeIdx] == PATTYPE_NONE){
		t.get<0>() = 0.0;
		return;
	    }
	    
	    // Whether this timestep is between the stride
	    //if ((timeIdx % stride)){
	    //t.get<0>() = 0.0;
	    //return;
	    //}

	    // for each output unit, find the maximum in the maxpooling window
	    int gradIdx;
	    int indxIdx;

	    for (int i = -1*width; i<=width; i++){
		indxIdx = outputIdx * (2 * maxWidth + 1) + i + width;
		gradIdx = outputIdx + i * parall * featDim;
		if (gradIdx >= 0 && gradIdx < (maxLength * parall * featDim) &&
		    patTypes[timeIdx + i * parall] != PATTYPE_NONE &&
		    maxIndex[indxIdx])
		    t.get<0>() += inputData[gradIdx];
		
	    }
	}	
    };
    
}
}

namespace layers {
    template <typename TDevice>
    MaxPoolingLayer<TDevice>::MaxPoolingLayer(const helpers::JsonValue &layerChild,
					      const helpers::JsonValue &weightsSection,
					      Layer<TDevice>           &precedingLayer,
					      int                       maxSeqLength,
					      int                       layerID)
	: TrainableLayer<TDevice>(layerChild, weightsSection, 0, 0,
				  precedingLayer, maxSeqLength, layerID)
    {
	
	m_width = ((layerChild->HasMember("width")) ? 
		   ((*layerChild)["width"].GetString()) : (""));
	m_stride = ((layerChild->HasMember("stride")) ? 
		   ((*layerChild)["stride"].GetString()) : (""));
	
	if (m_width.size() < 1 || m_stride.size() < 1){
	    throw std::runtime_error("missing width and stride in maxpooling of network.jsn");
	}	
	if (this->size() != precedingLayer.size()){
	    throw std::runtime_error("maxpooling layer size must be equal to the previous layer");
	}

	m_width_H.clear();
	misFuncs::ParseIntOpt(m_width, m_width_H);
	m_width_D = m_width_H;

	m_stride_H.clear();
	misFuncs::ParseIntOpt(m_stride, m_stride_H);
	m_stride_D = m_stride_H;

	if (m_width_H.size() != this->size() ||
	    m_stride_H.size()!= this->size()){
	    throw std::runtime_error("width and stride of maxpooling not equal to layer size");
	}
	
	m_maxWidth =  misFuncs::MaxCpuIntVec(m_width_H);
	m_maxPos.resize(precedingLayer.outputs().size() * (2 * m_maxWidth + 1), 0);

	if (this->precedingLayer().getSaveMemoryFlag())
	    throw std::runtime_error("layer before maxpooling is reduced in mem");

	throw std::runtime_error("Maxpooling is not fully implemented");
    }

    template <typename TDevice>
    MaxPoolingLayer<TDevice>::~MaxPoolingLayer()
    {
    }

    template <typename TDevice>
    const std::string& MaxPoolingLayer<TDevice>::type() const
    {
        static std::string s;
        if (s.empty()) s = "maxpooling";
        return s;
    }

    template <typename TDevice>
    void MaxPoolingLayer<TDevice>::computeForwardPass(const int nnState)
    {
	thrust::fill(m_maxPos.begin(), m_maxPos.end(), 0);
	{{
	    internal::maxpoolingForward fn;
	    fn.featDim   = this->size();
	    fn.maxWidth  = m_maxWidth;
	    fn.widthVec  = helpers::getRawPointer(this->m_width_D);
	    fn.strideVec = helpers::getRawPointer(this->m_stride_D);
	    fn.parall    = this->parallelSequences();
	    fn.maxLength = this->curMaxSeqLength();
	    fn.inputData = helpers::getRawPointer(this->precedingLayer().outputs());
	    fn.patTypes  = helpers::getRawPointer(this->patTypes());
	    fn.maxIndex  = helpers::getRawPointer(this->m_maxPos);

	    int n =this->curMaxSeqLength() * this->parallelSequences() * this->size();

	    thrust::for_each(
	     thrust::make_zip_iterator(
				       thrust::make_tuple(this->outputs().begin(),
					   thrust::counting_iterator<int>(0))),
	     thrust::make_zip_iterator(
				       thrust::make_tuple(this->outputs().begin()+n, 
					   thrust::counting_iterator<int>(0)+n)),
	     fn);    
	}}
    }

    template <typename TDevice>
    void MaxPoolingLayer<TDevice>::computeForwardPass(const int timeStep, const int nnState)
    {
	throw std::runtime_error("Not implemented");
    }

    template <typename TDevice>
    void MaxPoolingLayer<TDevice>::computeBackwardPass(const int nnState)
    {
	thrust::fill(this->precedingLayer().outputErrors().begin(),
		     this->precedingLayer().outputErrors().end(), 0);
	{{
	    internal::maxpoolingBackward fn;
	    fn.featDim = this->size();
	    fn.maxWidth  = m_maxWidth;
	    fn.widthVec  = helpers::getRawPointer(this->m_width_D);
	    fn.strideVec = helpers::getRawPointer(this->m_stride_D);
	    fn.parall  = this->parallelSequences();
	    fn.maxLength = this->curMaxSeqLength();
	    fn.inputData = helpers::getRawPointer(this->outputErrors());
	    fn.patTypes  = helpers::getRawPointer(this->patTypes());
	    fn.maxIndex  = helpers::getRawPointer(this->m_maxPos);

	    int n =this->curMaxSeqLength() * this->parallelSequences() * this->size();

	    thrust::for_each(
	     thrust::make_zip_iterator(
		thrust::make_tuple(this->precedingLayer().outputErrors().begin(),
				   thrust::counting_iterator<int>(0))),
	     thrust::make_zip_iterator(
		thrust::make_tuple(this->precedingLayer().outputErrors().begin()+n, 
					   thrust::counting_iterator<int>(0)+n)),
	     fn);    
	}}
    }
    
    template <typename TDevice>
    void MaxPoolingLayer<TDevice>::computeBackwardPass(const int timeStep, const int nnState)
    {
	throw std::runtime_error("maxpooling computBackwardPass(timeStep) not implemented");
    }
    
    template <typename TDevice>
    void MaxPoolingLayer<TDevice>::exportLayer(
	const helpers::JsonValue     &layersArray, 
	const helpers::JsonAllocator &allocator) const
    {
	TrainableLayer<TDevice>::exportLayer(layersArray, allocator);
	(*layersArray)[layersArray->Size() - 1].AddMember("width", m_width.c_str(), allocator);
        (*layersArray)[layersArray->Size() - 1].AddMember("stride",m_stride.c_str(),allocator);
    }

    template class MaxPoolingLayer<Cpu>;
    template class MaxPoolingLayer<Gpu>;
}
