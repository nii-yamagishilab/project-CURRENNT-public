/******************************************************************************
 * Copyright (c) 2013 Johannes Bergmann, Felix Weninger, Bjoern Schuller
 * Institute for Human-Machine Communication
 * Technische Universitaet Muenchen (TUM)
 * D-80290 Munich, Germany
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

#include "DataSetFraction.hpp"


namespace data_sets {

    DataSetFraction::DataSetFraction()
    {
    }

    DataSetFraction::~DataSetFraction()
    {
    }

    int DataSetFraction::inputPatternSize() const
    {
        return m_inputPatternSize;
    }

    int DataSetFraction::outputPatternSize() const
    {
        return m_outputPatternSize;
    }

    int DataSetFraction::externalInputSize() const
    {
        return m_exInputDim;
    }

    int DataSetFraction::externalOutputSize() const
    {
        return m_exOutputDim;
    }

    int DataSetFraction::maxSeqLength() const
    {
        return m_maxSeqLength;
    }

    int DataSetFraction::minSeqLength() const
    {
        return m_minSeqLength;
    }

    int DataSetFraction::maxExInputLength() const
    {
        return m_maxExInputLength;
    }

    int DataSetFraction::minExInputLength() const
    {
        return m_minExInputLength;
    }

    int DataSetFraction::maxExOutputLength() const
    {
        return m_maxExOutputLength;
    }

    int DataSetFraction::minExOutputLength() const
    {
        return m_minExOutputLength;
    }

    int DataSetFraction::numSequences() const
    {
        return (int)m_seqInfo.size();
    }

    const DataSetFraction::seq_info_t& DataSetFraction::seqInfo(int seqIdx) const
    {
        return m_seqInfo[seqIdx];
    }

    const Cpu::pattype_vector& DataSetFraction::patTypes() const
    {
        return m_patTypes;
    }

    const Cpu::real_vector& DataSetFraction::inputs() const
    {
        return m_inputs;
    }

    const Cpu::real_vector& DataSetFraction::outputs() const
    {
        return m_outputs;
    }

    const Cpu::int_vector& DataSetFraction::targetClasses() const
    {
        return m_targetClasses;
    }
    
    const Cpu::real_vector& DataSetFraction::auxRealData() const
    {
	return m_auxRealData;
    }
    
    const Cpu::pattype_vector& DataSetFraction::auxPattypeData() const
    {
	return m_auxPattypeData;
    }
    
    const Cpu::int_vector& DataSetFraction::auxIntData() const
    {
	return m_auxIntData;
    }
    
    const int& DataSetFraction::auxDataDim() const
    {
	return m_auxDataDim;
    }

    const Cpu::real_vector& DataSetFraction::exInputData()    const
    {
	return m_exInputData;
    }

    const Cpu::real_vector& DataSetFraction::exOutputData()    const
    {
	return m_exOutputData;
    }

    const int& DataSetFraction::getDataValidFlag() const
    {
	return m_flagDataValid;
    }

    void DataSetFraction::setDataValidFlag(const int flag)
    {
	m_flagDataValid = flag;
    }
    
    /*
    const Cpu::int_vector& DataSetFraction::txtData() const
    {
        return m_txtData;
    }
    int DataSetFraction::maxTxtLength() const
    {
        return m_maxTxtLength;
	}*/
  
    int DataSetFraction::fracTimeLength() const
    {
	return m_fracTotalLength;
    }


    const Cpu::pattype_vector& DataSetFraction::patTypesLowTimeRes() const
    {
	return m_patTypesLowTimeRes;
    }
    
    int   DataSetFraction::patTypesLowTimesResPos(const int resolution) const
    {
	for (int i = 0; i < m_resolutionBuffer.size(); i++){
	    if (m_resolutionBuffer[i].resolution == resolution)
		return m_resolutionBuffer[i].bufferPos;
	}
	return -1;
    }
    
    int   DataSetFraction::patTypesLowTimesResLen(const int resolution) const
    {
	for (int i = 0; i < m_resolutionBuffer.size(); i++){
	    if (m_resolutionBuffer[i].resolution == resolution)
		return m_resolutionBuffer[i].length;
	}
	return -1;
    }


    void DataSetFraction::printFracInfo(bool printToCerr) const
    {
	for (int i = 0; i<(int)m_seqInfo.size(); i++){
	    if (printToCerr)
		std::cerr << m_seqInfo[i].seqTag << "\t";
	    else
		std::cout << m_seqInfo[i].seqTag << "\t";
	}
	if (printToCerr)
	    std::cerr << std::endl;
	else
	    std::cout << std::endl;
	
    }
    
} // namespace data_sets
