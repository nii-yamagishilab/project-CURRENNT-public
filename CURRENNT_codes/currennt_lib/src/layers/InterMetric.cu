/******************************************************************************
 * This file is an addtional component of CURRENNT. 
 * Copyright (c) 2019 Xin WANG
 * National Institute of Informatics, Japan
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

#include "InterMetric.hpp"
#include "../helpers/misFuncs.hpp"
#include "../helpers/JsonClasses.hpp"
#include "../Configuration.hpp"

#include <sstream>
#include <stdexcept>


namespace layers {

        

    template <typename TDevice>
    InterMetricLayer<TDevice>::InterMetricLayer(const helpers::JsonValue &layerChild, 
						Layer<TDevice> &precedingLayer,
						int  maxSeqLength,
						int  layerID,
						bool createOutputs)
        : Layer<TDevice>  (layerChild, precedingLayer.parallelSequences(), 
			   maxSeqLength,
			   Configuration::instance().trainingMode(),
			   layerID,
			   &precedingLayer,
			   createOutputs)
    {
	// nothing
	
    }

    template <typename TDevice>
    InterMetricLayer<TDevice>::~InterMetricLayer()
    {
    }

    template <typename TDevice>
    void InterMetricLayer<TDevice>::reInitWeight()
    {
	// do nothing
    }

    
    // explicit template instantiations
    template class InterMetricLayer<Cpu>;
    template class InterMetricLayer<Gpu>;

} // namespace layers
