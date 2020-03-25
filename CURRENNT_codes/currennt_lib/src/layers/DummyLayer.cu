/******************************************************************************
 * This file is an addtional component of CURRENNT. 
 * Xin WANG
 * National Institute of Informatics, Japan
 * 2016 - 2019
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

#ifdef _MSC_VER
#   pragma warning (disable: 4244) 
#endif

#include "DummyLayer.hpp"

namespace layers {

    
    template <typename TDevice>
    DummyLayer<TDevice>::DummyLayer(const helpers::JsonValue &layerChild, 
				    const helpers::JsonValue &weightsSection, 
				    Layer<TDevice> &precedingLayer,
				    int             maxSeqLength,
				    int             layerID)
        : TrainableLayer<TDevice>(layerChild, weightsSection, 0, 0,
				  precedingLayer, maxSeqLength, layerID)
    {
    }

    template <typename TDevice>
    DummyLayer<TDevice>::~DummyLayer()
    {
    }


    template <typename TDevice>
    const std::string& DummyLayer<TDevice>::type() const
    {
        static std::string s;
        if (s.empty()) s = "dummy";
        return s;
    }
    
    
    template <typename TDevice>
    void DummyLayer<TDevice>::computeForwardPass(const int nnState)
    {	
    }


    template <typename TDevice>
    void DummyLayer<TDevice>::computeForwardPass(const int timeStep,
						 const int nnState)
    {
    }

    
    template <typename TDevice>
    void DummyLayer<TDevice>::computeBackwardPass(const int nnState)
    {
    }


    template <typename TDevice>
    void DummyLayer<TDevice>::computeBackwardPass(const int timeStep,
						  const int nnState)
    {
    }


    template <typename TDevice>
    void DummyLayer<TDevice>::reduceOutputBuffer()
    {
	this->resizeOutputBuffer(this->parallelSequences() * this->size());
	this->setSaveMemoryFlag(true);
	printf("\t[mem saved]");
    }
    
    template <typename TDevice>
    int DummyLayer<TDevice>::outputBufPtrBias(const int timeStepTimesParallel,
					      const int nnState)
    {
	if (this->getSaveMemoryFlag()){
	    return timeStepTimesParallel * this->size();
	}else{
	    return 0;
	}
    }	


    template <typename TDevice>
    void DummyLayer<TDevice>::clearAllBuffers()
    {
	this->clearOutputBuffer();
    }

    template <typename TDevice>
    void DummyLayer<TDevice>::resizeAllBuffers(const int timeLength)
    {
	this->resizeOutputBuffer(timeLength * this->parallelSequences() * this->size());
    }

    // explicit template instantiations
    template class DummyLayer<Cpu>;
    template class DummyLayer<Gpu>;

} // namespace layers

