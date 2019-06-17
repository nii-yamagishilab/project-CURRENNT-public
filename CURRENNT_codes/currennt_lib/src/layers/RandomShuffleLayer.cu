/******************************************************************************
 * This file is an addtional component of CURRENNT. 
 * Xin WANG
 * National Institute of Informatics, Japan
 * 2019
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


#include "RandomShuffleLayer.hpp"

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
#include <thrust/sort.h>
#include <thrust/sequence.h>

#include <boost/foreach.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>
#include <vector>
#include <stdexcept>
namespace internal{
namespace {
    
    // Generation uniform distribution noise
    struct genNoiseForShuffle
    {
	float a, b;
	int   seed;
	
	__host__ __device__
	genNoiseForShuffle(float _a=-1.f, float _b=1.f, int _seed=123) : a(_a), b(_b), seed(_seed) {};

	__host__ __device__
	float operator()(const unsigned int n) const
	{
	    thrust::default_random_engine rng(seed);
	    thrust::uniform_real_distribution<float> dist(a, b);
	    rng.discard(n);
	    return dist(rng);
	}
    };


    struct randomShuffle
    {
	int         layerSize;
	real_t     *inputData;
	
	__host__ __device__ void operator() (const thrust::tuple<real_t&, real_t&, int> &t) const
	{
	    t.get<0>() = inputData[(int)t.get<1>()];
	}
    };
    
    
}
}

namespace layers{

    // Construct the layer
    template <typename TDevice>
    RandomShuffleLayer<TDevice>::RandomShuffleLayer(const helpers::JsonValue &layerChild,
						    const helpers::JsonValue &weightsSection,
						    std::vector<Layer<TDevice>*> &precedingLayers,
						    int                       maxSeqLength,
						    int                       layerID)
	: SkipLayer<TDevice>(layerChild, weightsSection, precedingLayers,
			     maxSeqLength, layerID, false)
    {

	if (this->size() != this->precedingLayer().size())
	    throw std::runtime_error("Error: random_shuffle layer size != precedingLayer size");
	
	m_preShuffleLayerName = (layerChild->HasMember("preShuffleLayer") ? 
				 ((*layerChild)["preShuffleLayer"].GetString()) : "");
	
	// initialize the buffer
	m_dataIndex    = this->outputs();
	m_randomValue  = this->outputs();
	m_dataIndexRev = m_dataIndex;
	    
	// find th coupled random shuffle layer if possible
	if (m_preShuffleLayerName.size()){
	    BOOST_FOREACH (Layer<TDevice> *layer, precedingLayers) {
		if (layer->name() == m_preShuffleLayerName){
		    this->PreLayers().push_back(layer);
		    break;
		}
	    }
	}
	
	if (this->PreLayers().size()){

	    m_preShuffleLayer = dynamic_cast<RandomShuffleLayer<TDevice>*>(this->PreLayers()[0]);
	    if (m_preShuffleLayer == NULL)
		throw std::runtime_error("Error: random_shuffle cannot find previous shuffle layer");
	    
	    if (this->size() != m_preShuffleLayer->size())
		throw std::runtime_error("Error: random_shuffle layer size != pre shuffle layer");
	    
	    printf("\n\tRandomshuffle coupled with layer %s", m_preShuffleLayer->name().c_str());
	    // swap the index
	    m_dataIndex = m_preShuffleLayer->__dataIndexRev();
	    m_dataIndexRev = m_preShuffleLayer->__dataIndex();

	    
	}else{
	    printf("\n\tRandomshuffle uses no coupled layer");
	    // generate random number, with layerID as random seed
	    thrust::counting_iterator<unsigned int> index_sequence_begin(0);
	    thrust::transform(
		index_sequence_begin,
		index_sequence_begin + m_randomValue.size(),
		m_randomValue.begin(),
		internal::genNoiseForShuffle(0.0, 1.0, (int)this->getLayerID()));
	    
	    thrust::sequence(m_dataIndex.begin(), m_dataIndex.end());
	    m_dataIndexRev = m_dataIndex;

	    // shuffle within each frame
	    for (int frameIdx = 0; frameIdx < maxSeqLength; frameIdx++){
		
		// sort
		thrust::sort_by_key(m_randomValue.begin() + frameIdx * this->size(),
				    m_randomValue.begin() + (frameIdx+1) * this->size(),
				    m_dataIndex.begin() + frameIdx * this->size());
	
		// get the reverse index
		thrust::copy(m_dataIndex.begin() + frameIdx * this->size(),
			     m_dataIndex.begin() + (frameIdx + 1) * this->size(),
			     m_randomValue.begin() + frameIdx * this->size());
		
		thrust::sort_by_key(m_randomValue.begin() + frameIdx * this->size(),
				    m_randomValue.begin() + (frameIdx + 1) * this->size(),
				    m_dataIndexRev.begin()   + frameIdx * this->size());
	    }
	}
	
    }	

    // Destructor
    template <typename TDevice>
    RandomShuffleLayer<TDevice>::~RandomShuffleLayer()
    {
    }

    template <typename TDevice>
    const std::string& RandomShuffleLayer<TDevice>::type() const
    {
        static std::string s;
        if (s.empty()) s = "random_shuffle";
        return s;
    }

    template <typename TDevice>
    void RandomShuffleLayer<TDevice>::computeForwardPass(const int nnState)
    {
	int timeLength = this->curMaxSeqLength() * this->parallelSequences();

	/* Shuffle over time and dimension
	   Generating random vectors for each training utterance is too slow
	// sort with keys
	thrust::copy(m_randomValue.begin(), m_randomValue.end(), m_randomValueBuf.begin());
	thrust::sequence(m_dataIndex.begin(), m_dataIndex.end());
	m_dataIndexRev = m_dataIndex;

	// sort
	thrust::sort_by_key(m_randomValueBuf.begin(),
			    m_randomValueBuf.begin() + timeLength * this->size(),
			    m_dataIndex.begin());
	
	// get the reverse index
	thrust::copy(m_dataIndex.begin(), m_dataIndex.end(), m_randomValueBuf.begin());
	thrust::sort_by_key(m_randomValueBuf.begin(),
			    m_randomValueBuf.begin() + timeLength * this->size(),
			    m_dataIndexRev.begin());
	// if this layer is coupled with a previous random shuffle layer, swap the index
	if (this->PreLayers().size())
	    m_dataIndexRev.swap(m_dataIndex);
	    
	*/
	
	
	{{
	    internal::randomShuffle fn;
	    fn.layerSize = this->size();
	    fn.inputData = helpers::getRawPointer(this->precedingLayer().outputs());
	    
	    int n = timeLength * this->size();
	    thrust::for_each(
               thrust::make_zip_iterator(
		  thrust::make_tuple(this->outputs().begin(),
				     m_dataIndex.begin(),
				     thrust::counting_iterator<int>(0))),
	       thrust::make_zip_iterator(
		  thrust::make_tuple(this->outputs().begin()           + n,
				     m_dataIndex.begin()               + n,
				     thrust::counting_iterator<int>(0) + n)),
	       fn);
	    
	}}	
    }

    template <typename TDevice>
    void RandomShuffleLayer<TDevice>::computeForwardPass(const int timeStep, const int nnState)
    {
	throw std::runtime_error("Error: random shuffle not support online training/generation");
    }
    
    template <typename TDevice>
    void RandomShuffleLayer<TDevice>::computeBackwardPass(const int nnState)
    {
	int timeLength = this->curMaxSeqLength() * this->parallelSequences();
	{{
	    internal::randomShuffle fn;
	    fn.layerSize = this->size();
	    fn.inputData = helpers::getRawPointer(this->outputErrors());
	    
	    int n = timeLength * this->size();
	    thrust::for_each(
               thrust::make_zip_iterator(
		thrust::make_tuple(this->precedingLayer().outputErrors().begin(),
				   m_dataIndexRev.begin(),
				   thrust::counting_iterator<int>(0))),
	       thrust::make_zip_iterator(
		thrust::make_tuple(this->precedingLayer().outputErrors().begin() + n,
				   m_dataIndexRev.begin()                        + n,
				   thrust::counting_iterator<int>(0)             + n)),
	       fn);
	}}
	
    }

    template <typename TDevice>
    void RandomShuffleLayer<TDevice>::computeBackwardPass(const int timeStep, const int nnState)
    {
	throw std::runtime_error("Error: random shuffle not support online training");
    }
    
    
    template <typename TDevice>
    void RandomShuffleLayer<TDevice>::exportLayer(const helpers::JsonValue &layersArray,
					 const helpers::JsonAllocator &allocator) const
    {
	SkipLayer<TDevice>::exportLayer(layersArray, allocator);
	if (m_preShuffleLayerName.size())
	    (*layersArray)[layersArray->Size() - 1].AddMember("preShuffleLayer",
							      m_preShuffleLayerName.c_str(),
							      allocator);
    }

    template <typename TDevice>
    void RandomShuffleLayer<TDevice>::clearAllBuffers()
    {
    }

    template <typename TDevice>
    void RandomShuffleLayer<TDevice>::resizeAllBuffers(const int timeLength)
    {
    }    

    template <typename TDevice>
    typename RandomShuffleLayer<TDevice>::real_vector& RandomShuffleLayer<TDevice>::__dataIndex()
    {
	return m_dataIndex;
    }

    template <typename TDevice>
    typename RandomShuffleLayer<TDevice>::real_vector& RandomShuffleLayer<TDevice>::__dataIndexRev()
    {
	return m_dataIndexRev;
    }

    
    template <typename TDevice>
    std::vector<int> RandomShuffleLayer<TDevice>::dependLayerIDs()
    {
	return Layer<TDevice>::dependLayerIDs();
    }
    
    template class RandomShuffleLayer<Cpu>;
    template class RandomShuffleLayer<Gpu>;
    
}
