/******************************************************************************
 * This file is an addtional component of CURRENNT. 
 * Xin WANG
 * National Institute of Informatics, Japan
 * 2019
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

#include "SpecialFeedBackLayer.hpp"

#include "../helpers/getRawPointer.cuh"
#include "../helpers/Matrix.hpp"
#include "../helpers/JsonClasses.hpp"
#include "../activation_functions/Logistic.cuh"
#include "../activation_functions/Tanh.cuh"

#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/for_each.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/fill.h>
#include <boost/foreach.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>
#include <vector>
#include <stdexcept>



namespace internal{
namespace {

}
}


namespace layers{


    template <typename TDevice>
    SpecialFeedBackLayer<TDevice>::SpecialFeedBackLayer(
				const helpers::JsonValue &layerChild,
				const helpers::JsonValue &weightsSection,
				Layer<TDevice>           &precedingLayer,
				int                       maxSeqLength,
				int                       layerID)
	: TrainableLayer<TDevice>(layerChild, weightsSection, 0, 0,
				  precedingLayer, maxSeqLength, layerID)
    {
	
	//
	m_feedback_src_str = ((layerChild->HasMember("source_layer")) ? 
			    ((*layerChild)["source_layer"].GetString()) : (""));

	m_feedback_src_ptr = NULL;
    }

    template <typename TDevice>
    SpecialFeedBackLayer<TDevice>::~SpecialFeedBackLayer()
    {
    }


    template <typename TDevice>
    void SpecialFeedBackLayer<TDevice>::exportLayer(
			const helpers::JsonValue     &layersArray, 
			const helpers::JsonAllocator &allocator) const
    {
	TrainableLayer<TDevice>::exportLayer(layersArray, allocator);
	(*layersArray)[layersArray->Size() - 1].AddMember("source_layer",
							  m_feedback_src_str.c_str(),
							  allocator);
    }
	    

    template <typename TDevice>
    void SpecialFeedBackLayer<TDevice>::linkTargetLayer(Layer<TDevice> &targetLayer)
    {
	if (targetLayer.name() == m_feedback_src_str){
	    m_feedback_src_ptr = &targetLayer;
	    printf("\n\tSpecial feedback layer %s", m_feedback_src_str.c_str());

	    if (targetLayer.size() != this->size())
		throw std::runtime_error("\nSpecialFeedBackLayer dimension mismatch");
	}
    }

    template <typename TDevice>
    void SpecialFeedBackLayer<TDevice>::computeForwardPass(const int nnState)
    {
	if (m_feedback_src_ptr == NULL)
	    throw std::runtime_error("\nSpecialFeedBackLayer receives no input");

	thrust::copy(m_feedback_src_ptr->outputs().begin(),
		     m_feedback_src_ptr->outputs().end(),
		     this->outputs().begin());
    }

    template <typename TDevice>
    void SpecialFeedBackLayer<TDevice>::computeForwardPass(const int timeStep,
							   const int nnState)
    {
	throw std::runtime_error("\nSpecialFeedBackLayer not for online generation");
    }

    template <typename TDevice>
    void SpecialFeedBackLayer<TDevice>::computeBackwardPass(const int nnState)
    {
	// do thing for backward propagation
    }

    template <typename TDevice>
    void SpecialFeedBackLayer<TDevice>::computeBackwardPass(const int timeStep,
							    const int nnState)
    {
	// do thing for backward propagation
    }

    template <typename TDevice>
    const std::string& SpecialFeedBackLayer<TDevice>::type() const
    {
        static std::string s;
        if (s.empty()) s = "simple_feedback";
        return s;
    }

    
    template class SpecialFeedBackLayer<Cpu>;
    template class SpecialFeedBackLayer<Gpu>;
    
}

