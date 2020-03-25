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

#ifndef DATA_SETS_DATASET_HPP
#define DATA_SETS_DATASET_HPP

#include "DataSetFraction.hpp"

#include <boost/noncopyable.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/scoped_ptr.hpp>

#include <string>
#include <vector>
#include <fstream>


namespace data_sets {

    // the ******* nvcc hates boost headers :(
    struct thread_data_t;

    /******************************************************************************************//**
     * Contains input and/or output data of the neural network. This class is used to read input
     * data, training data, validation data or test data sets from a file and to write output data 
     * to a file.
     *********************************************************************************************/
    class DataSet : boost::noncopyable
    {
    public:
        struct sequence_t {
            int         originalSeqIdx;        // index of this segment in original sequence
            int         length;                // length of the segment
            std::string seqTag;                // name of the original sequence

            std::streampos inputsBegin;        // input data pointer to cache
            std::streampos targetsBegin;       // target data pointer to cache
	    
	    // Add 0620, Wang: support to the txt int data
	    //int            txtLength;        // length of the txt data for this sequence
	    //std::streampos txtDataBegin;     //  

	    // Add 1111, support to the auxillary data
	    int            auxDataDim;         // dimension of auxillary data
	    int            auxDataTyp;         // type of auxillary data
	    std::streampos auxDataBegin;       // auxillary data pointer to cache
	    
	    int            beginInUtt;       // the relative position of the start of the seq
	                                     // in the utterance
	    
	    // Add 170327, support to the external input data
	    int            exInputDim;       // external-stored input data dimension
	    int            exInputLength;    // length of external-stored input data
	    int            exInputStartPos;  // 
	    int            exInputEndPos;    // 
	    std::streampos exInputBegin;     //


	    // Add 171012, support to the external output data
	    int            exOutputDim;       // 
	    int            exOutputLength;    //
	    int            exOutputStartPos;  //
	    int            exOutputEndPos;    // 
	    std::streampos exOutputBegin;     //
        };

	struct seqOrdered_unt{
	    std::string  seqTag;
	    int          originalSeqIdx;
	};

	
    private:
        void _nextFracThreadFn();
        void _shuffleSequences();
        void _shuffleFractions();
        void _addNoise(Cpu::real_vector *v);
	void _cacheAssert(bool flag);
	
        Cpu::real_vector    _loadInputsFromCache(const sequence_t &seq);
        Cpu::real_vector    _loadOutputsFromCache(const sequence_t &seq);
	Cpu::real_vector    _loadExInputsFromCache(const sequence_t &seq);
	Cpu::real_vector    _loadExOutputsFromCache(const sequence_t &seq);
        Cpu::int_vector     _loadTargetClassesFromCache(const sequence_t &seq);
	
        boost::shared_ptr<DataSetFraction> _makeFractionTask(int firstSeqIdx);
        boost::shared_ptr<DataSetFraction> _makeFirstFractionTask();

	
	// Obsolete
	// Cpu::real_vector _loadTxtDataFromCache(const sequence_t &seq);
	
	// Interface to load auxillary data
	Cpu::real_vector    _loadAuxRealDataFromCache(const sequence_t &seq);
	Cpu::pattype_vector _loadAuxPattypeDataFromCache(const sequence_t &seq);
	Cpu::int_vector     _loadAuxIntDataFromCache(const sequence_t &seq);

	void _loadAuxDataOptions();
	void _loadExternalDataOptions();
	void _loadResolutionOptions();
	void _setCacheOptions(const std::string cachePath);
	void _loadTxtList(const std::string fileOrderedLst);
	void _orderSeq();
	
    private:
        bool   m_fractionShuffling;          // shuffle fraction?
        bool   m_sequenceShuffling;          // shuffle sequences?
        bool   m_isClassificationData;       // for classification
        real_t m_noiseDeviation;             // std of noise to be added to input data (obsolete)
        int    m_parallelSequences;          // number of parallelSequences
        int    m_totalSequences;             // total number of sequences
	
        unsigned long int    m_totalTimesteps;  // total number of time steps
	
        int    m_minSeqLength;               // minimum sequence length in data corpus
        int    m_maxSeqLength;               // maximum sequencd length in data corpus
        int    m_inputPatternSize;           // total dimension of input data vector
        int    m_outputPatternSize;          // total dimension of output data vector

        Cpu::real_vector m_outputMeans;      
        Cpu::real_vector m_outputStdevs;
	bool             m_outputMVFlag;
	
        std::fstream     m_cacheFile;
        std::string      m_cacheFileName;

        std::vector<sequence_t>  m_sequences;
	std::vector<seqOrdered_unt> m_seqOrderVec;
	
        boost::scoped_ptr<thread_data_t> m_threadData; // just because nvcc hates boost headers
        int    m_curFirstSeqIdx;
	
	int    m_quick_network_test_num;
	
	// Add 0620: Wang support to the txt input data
	// (Support for the txt data should be merged with the auxillary data)
	//int    m_maxTxtLength;                 // the maximum length of txt over corpus
	//int    m_txtDataPatternSize;           // dimension of the txt data 
	//int    m_totalTxtLength;               // the total length of txt data for this fraction
	//bool   m_hasTxtData;                   // whether contains the txt data?
	
	// Add 1111: Support to the auxillary data (external data not in .nc format)
	std::string m_auxDirPath;              // path to the directory where auxillary data exist
	std::string m_auxFileExt;              // extension of the auxillary data type
	int         m_auxDataTyp;              // the binary data type of the auxillary data
	int         m_auxDataDim;              // dimension of the auxillary data

	// Add 170327: external input file
	std::string m_exInputDir; // obsolete
	std::string m_exInputExt; // obsolete
	int         m_exInputDim; // obsolete
	
	int         m_exInputType;   // reserved
	bool        m_exInputFlag;
	int         m_exInputReso;
	
	int         m_exOutputType;  // reserved
	bool        m_exOutputFlag;
	int         m_exOutputReso;
	
	std::vector<std::string> m_exInputDirs;
	std::vector<std::string> m_exInputExts;
	Cpu::int_vector          m_exInputDims;
	
	std::vector<std::string> m_exOutputDirs;
	std::vector<std::string> m_exOutputExts;
	Cpu::int_vector          m_exOutputDims;
		
	Cpu::int_vector m_resolutionBuf;

	// flag to indicate multi-threading
	bool        m_flag_multithread;
    public:
        /**
         * Creates an empty data set
         */
        DataSet();

        /**
         * Loads the data set from a NetCDF file (filename.nc)
         *
         * @param ncfile   The filename of the NetCDF file
         * @param parSeq   Number of parallel sequences
         * @param fraction Fraction of all sequences to load
         * @param fracShuf Apply fraction shuffling
         * @param seqShuf  Apply sequence shuffling
         * @param noiseDev Static noise deviation
         */
        DataSet(const std::vector<std::string> &ncfiles, 
		int    parSeq,  
		real_t fraction       = 1, 
		int    truncSeqLength = 0,
		bool   fracShuf       = false, 
		bool   seqShuf        = false, 
		real_t noiseDev       = 0,
		std::string cachePath = "",
		std::string fileOrderedLst = ""
		);

        /**
         * Destructor
         */
        virtual ~DataSet();

        /**
         * Check if the data set contains classification data
         *
         * @return True if the data set contains classification data
         */
        bool isClassificationData() const;

        /**
         * Check if the data set is empty
         *
         * @return True if the data set is empty
         */
        bool empty() const;

        /**
         * Returns the next fraction or an empty pointer once after all fractions
         * have been processed.
         *
         * If the data set is split in 3 fractions, the consecutive calls to this
         * function will lead to the following returned values:
         *   1st call: pointer to 1st fraction
         *   2nd call: pointer to 2nd fraction
         *   3rd call: pointer to 3rd fraction
         *   4th call: empty pointer
         *   5th call: pointer to 1st fraction
         *   ...
         *
         * @return Next fraction or an empty pointer
         */
        boost::shared_ptr<DataSetFraction> getNextFraction();

	// similar to getNextFraction but without threading
	boost::shared_ptr<DataSetFraction> getNextFraction_nothreading();

        /**
         * Returns the local file name used to cache the data
         *
         * @return the local file name used to cache the data
         */
        std::string cacheFileName() const;

        /**
         * Returns the total number of sequences
         *
         * @return The total number of sequences
         */
        int totalSequences() const;
         
        /**
         * Returns the total number of timesteps
         *
         * @return The total number of timesteps
         */
        unsigned long int totalTimesteps() const;

        /**
         * Returns the length of the shortest sequence
         *
         * @return The length of the shortest sequence
         */
        int minSeqLength() const;

        /**
         * Returns the length of the longest sequence
         *
         * @return The length of the longest sequence
         */
        int maxSeqLength() const;

        /**
         * Returns the size of the input patterns
         *
         * @return The size of the input patterns
         */
        int inputPatternSize() const;

        /**
         * Returns the size of the output patterns
         *
         * @return The size of the output patterns
         */
        int outputPatternSize() const;

        /**
         * Returns the output means (per feature) indicated in the NC file
         *
         * @return vector of output means
         */
        Cpu::real_vector outputMeans() const;

        /**
         * Returns the output standard deviations (per feature) indicated in the NC file
         *
         * @return vector of output standard deviations
         */
        Cpu::real_vector outputStdevs() const;

	/**
         * Returns whether valid mean and std have been loaded
         *
         */
	bool outputMVFlag() const;
	
	/**
	 * Returns the maximum length of txt data in the corpus
	 *
	 */
	int maxTxtLength() const;
	
    };
    
    
    // Add 0514 Wang: add one data struct to read mean and variance only
    class DataSetMV
    {
    public:
	const int& inputSize() const;
	const int& outputSize() const;
	const Cpu::real_vector& inputM() const;
	const Cpu::real_vector& inputV() const;
	const Cpu::real_vector& outputM() const;
	const Cpu::real_vector& outputV() const;
	
	DataSetMV(const std::string &ncfile);
	DataSetMV();
	
	~DataSetMV();

    private:
	int    m_inputPatternSize;
        int    m_outputPatternSize;

        Cpu::real_vector m_inputMeans;
        Cpu::real_vector m_inputStdevs;

        Cpu::real_vector m_outputMeans;
        Cpu::real_vector m_outputStdevs;
	
    };
    
} // namespace data_sets


#endif // DATA_SETS_DATASET_HPP


