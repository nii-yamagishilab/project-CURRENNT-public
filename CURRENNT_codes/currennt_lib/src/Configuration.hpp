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

#ifndef CONFIGURATION_HPP
#define CONFIGURATION_HPP

#include "Types.hpp"

#include <string>


/******************************************************************************************//**
 * Creates the configuration for the program from the command line
 *********************************************************************************************/
class Configuration
{
public:
    enum optimizer_type_t {
        OPTIMIZER_STEEPESTDESCENT,
        OPTIMIZER_RPROP
    };

    enum distribution_type_t {
        DISTRIBUTION_NORMAL,
        DISTRIBUTION_UNIFORM,
	DISTRIBUTION_UNINORMALIZED
    };

    enum feedforwardformat_type_t {
        FORMAT_SINGLE_CSV,
        FORMAT_CSV,
        FORMAT_HTK
    };

private:
    static Configuration *ms_instance;

    std::string m_serializedOptions;

    bool m_trainingMode;
    bool m_hybridOnlineBatch;
    bool m_useCuda;
    bool m_listDevices;
    bool m_shuffleFractions;
    bool m_shuffleSequences;
    bool m_autosave;
    bool m_autosaveBest;

    optimizer_type_t         m_optimizer;
    distribution_type_t      m_weightsDistribution;
    feedforwardformat_type_t m_feedForwardFormat;

    bool m_revertStd;
    bool m_outputHtk;
    
    /* Add 16-02-22 Wang: for WE updating */
    bool        m_weUpdate;       // whether update the input we vector?
    unsigned    m_weIDDim;        // we index in which dimension of input vector (base 0)
    unsigned    m_weDim;          // dimension of the input we vector
    real_t      m_weLearningRate; // learning rate for we
    std::string m_weBank;         // path to the we data
    std::string m_trainedParameter;    // path of the trained model, only model weights  will be read
    std::string m_trainedParameterCtr; // a control string to decide which layer should be read in
    /* Add 20160902 WE noise options*/
    int         m_weNoiseStartDim;    // noise adding, from which dimension?
    int         m_weNoiseEndDim;      // noise adding, end at which dimension
    real_t      m_weNoiseDev;         // noise std.


    /* Add 0401 Wang: for mse weight */
    std::string m_mseWeightPath;  // path to the mse weight file

    /* Add 0402 Wang: for bias of Highway gate */
    real_t      m_highwayBias;         // initial value for highway bias

    /* Add 0408 Wang: to tap in the output of arbiary layer */
    int         m_outputTapLayer;         // the ID of the layer to be tapped in (start from 0)
    bool        m_outputGateOut;         // whether to get output from the gate 

    /* Add 0409 Wang: to decay the learning rate */
    real_t      m_lr_decay_rate;
    
    /* Add 0413 Wang: to mask the weight of network */
    std::string m_weightMaskPath;
    
    /* Add 0504 Wang: MDN flag vector */
    std::string m_mdnFlagPath;
    real_t      m_mdnSamplingPara;
    int         m_EMGenIter;

    /* Add 0514 Wang: data mv file*/
    std::string m_datamvPath;

    std::string m_dataTargetPath;
    std::string m_dataSourcePath;
    
    /* Add 0526 Wang: add variance related options for MDN*/
    real_t      m_varInitPara;
    real_t      m_vFloorPara;
    real_t      m_wInitPara;
    bool        m_tiedVariance;      // tie the variance across dimension in mixture model?
    
    /* Add 0620 Wang: add LstmCharW*/
    std::string m_chaBankPath;    // path to the binary data of txt character bank
    unsigned    m_chaDimLstmCharW;   // the size of the txt character bag
    
    /* Add 0602 Other options */
    std::string m_printWeightPath;
    int         m_printWeightOpt;
    std::string m_mdnVarScaleGen;       // the vector to scale each dimension of the output
    std::string m_mdnDyn;               // whether incorporate dynamic feature
    int         m_arrmdnLearning;       // an option for the learning rate of arrmdn
    std::string m_tanhAutoregressive;   // whether use the tanh-based function
    int         m_setDynFilterZero;     // 
    real_t      m_ARRMDNInitVar;        // the variance of Gaussian dis for initializing AR
    int         m_ARRMDNUpdateInterval; // after how many epochs update the n+1 order AR ? 

    std::string m_arOrder;
    std::string m_arConfig;
    
    
    /* Add 1012 KLD output*/
    int         m_KLDOutputDataType;    // 
    real_t      m_lrFactor;             // learning rate factor
    
    /* Add 1022 for Adagrad */
    unsigned    m_optimizerOption;      // 
    real_t      m_secondLearningRate;   // in case the optimizer requires another learning rate
    
    /* Add 1111 for auxillary data */
    std::string m_auxDataDir;          // options for the auxillary input data
    std::string m_auxDataExt;
    int         m_auxDataTyp;
    int         m_auxDataDim;

    std::string m_probDataDir;
    int         m_probDataDim;
    
    /* Add 1226 for feedback connection */
    std::string m_secondOutputOpt;     // for MDN to control second output

    /* Add 20170107 option for NN training */
    real_t   m_lstmForgetIni;          // bias to the output of LSTM forget gate in initial stage

    /* Add 20170124 options for feedback aggregation*/
    int      m_flagAggregateSyn;       //

    /* Add 20170129 options for reading weight mask*/
    int      m_weightMaskOpt;          // option to read and use the weight mask

    /* Add 20170131 schedule sampling */
    int      m_scheduleSampOpt;       //
    std::string m_scheduleSampOpts;   //
    int      m_scheduleSampPara;      //
    int      m_dropoutbeforeVAE;    
    
    real_t   m_mdnUVSigThreshold;
    int      m_mdnSoftMaxGenMethod;
    std::string m_mdnSoftMaxGenMethods;

    /* Add 20170326 external input*/
    std::string m_exInputDir;
    std::string m_exInputExt;
    int         m_exInputDim;
    
    /* Add 20170901 external inputs (multiple files) */
    std::string m_exInputDirs;
    std::string m_exInputExts;
    std::string m_exInputDims;
    int         m_exInputType;
    int         m_exInputReso;
    
    std::string m_exOutputDirs;
    std::string m_exOutputExts;
    std::string m_exOutputDims;
    int         m_exOutputType;    
    int         m_exOutputReso;
    
    /* Add 20170404 */
    int         m_verbose;
    int         m_fakeEpochNum;
    int         m_runningMode;

    /* Add 20170612 */
    int         m_mdnVarFixEpochNum;

    /* Add 20170711 */
    int         m_vaePlotManifold;

    /* Add 20171007 */
    std::string m_resolutions;

    /**/
    int         m_vaeEncoderOutputLayer;
    std::string m_vaeCodeInputDir;
    int         m_vaeGetVqIndex;
    int         m_vaePutVqIndex;

    int         m_seqWeightsFlag;

    int         m_waveNetSaveMemFlag;

    int         m_dataCheckerIdx;

    
    unsigned m_truncSeqLength;
    unsigned m_parallelSequences;
    unsigned m_maxEpochs;
    unsigned m_maxEpochsNoBest;
    unsigned m_validateEvery;
    unsigned m_testEvery;
    unsigned m_randomSeed;
    unsigned m_outputFeatureKind;

    real_t m_learningRate;
    real_t m_momentum;
    real_t m_weightsUniformMin;
    real_t m_weightsUniformMax;
    real_t m_weightsNormalSigma;
    real_t m_weightsNormalMean;
    real_t m_inputNoiseSigma;
    real_t m_weightNoiseSigma;
    real_t m_trainingFraction;
    real_t m_validationFraction;
    real_t m_testFraction;
    real_t m_featurePeriod;

    real_t m_softmaxTemperature;
    
    int m_inputLeftContext;
    int m_inputRightContext;
    int m_outputTimeLag;

    std::string m_networkFile;
    std::string m_networkGraphFile;
    std::string m_trainedNetwork;
    std::string m_feedForwardOutputFile;
    std::string m_autosavePrefix;
    std::string m_continueFile;
    std::string m_cachePath;
    
    std::vector<std::string> m_trainingFiles;
    std::vector<std::string> m_validationFiles;
    std::vector<std::string> m_testFiles;
    std::vector<std::string> m_feedForwardInputFiles;
    
public:
    /**
     * Parses the command line
     *
     * @param argc Number of strings in argv
     * @param argv Strings from the command line
     */
    Configuration(int argc, const char *argv[]);

    /**
     * Destructor
     */
    ~Configuration();

    /**
     * Returns the static instance
     *
     * @return The static instance
     */
    static const Configuration& instance();

    /**
     * Returns a string that contains all options
     * 
     * @return A string that contains all options
     */
    const std::string& serializedOptions() const;

    /**
     * Returns true if the NN shall be trained
     *
     * @return True if the NN shall be trained
     */
    bool trainingMode() const;
    bool generatingMode() const;
    
    const std::string& printWeightPath() const;
    const int& printWeightOpt() const;
    /**
     * Returns true if hybrid online/batch learning is enabled
     *
     * Hybrid online/batch learning means that the network weights are updated after every
     * block of parallel sequences. Example: if the user sets parallel_sequences=50, then
     * the weights are updated after each block of 50 sequences has been processed.
     *
     * If the number of parallel sequences is set to 1, we have true online learning with
     * weight updates after every sequence.
     *
     * @return True if hybrid online/batch learning is enabled
     */
    bool hybridOnlineBatch() const;

    /**
     * Returns true if shuffling of fractions in hybrid online/batch learning is enabled
     *
     * Each fraction contains of N parallel sequences except for the last block which can
     * contain less sequences. If this option is enabled, the order in which the fractions
     * are computed is randomized for each training epoch.
     *
     * @return True if shuffling of fractions is enabled
     */
    bool shuffleFractions() const;

    /**
     * Returns true if shuffling of sequences within and across fractions is enabled
     *
     * If this option is enabled, the sequences are shuffled before each training epoch, 
     * resulting in a completely randomized distribution of sequences across the fractions.
     *
     * @return True if shuffling of sequences is enabled
     */
    bool shuffleSequences() const;

    /**
     * Returns true if the nVidia CUDA technology shall be used to accelerate the computations
     *
     * @return True if CUDA shall be used
     */
    bool useCuda() const;

    bool listDevices() const;

    /**
     * Returns true if autosave is enabled
     *
     * @return True if autosave is enabled
     */
    bool autosave() const;
    
    /**
      * Returns true if autosave at best validation error is enabled
      *
      * @return true if autosave at best validation error is enabled
      */
    bool autosaveBest() const;

    /**
     * Returns the optimizer type
     *
     * @return The optimizer type
     */
    optimizer_type_t optimizer() const;

    /**
     * Returns the maximum number of parallel computed sequences
     *
     * @return The maximum number of parallel computed sequences
     */
    int parallelSequences() const;

    /**
     * Returns the maximum number of epochs during training
     *
     * @return The maximum number of epochs during training
     */
    int maxEpochs() const;

    /**
     * Returns the maximum number of training epochs in which no new lowest error could be achieved
     *
     * @return The maximum number of training epochs in which no new lowest error could be achieved
     */
    int maxEpochsNoBest() const;

    /**
     * Returns the number of training epochs after which the validation error shall be calculated
     *
     * @return The number of training epochs after which the validation error shall be calculated
     */
    int validateEvery() const;

    /**
     * Returns the number of training epochs after which the test error shall be calculated
     *
     * @return The number of training epochs after which the test error shall be calculated
     */
    int testEvery() const;

    /**
     * Returns the learning rate for the steepest descent optimizer
     *
     * @return The learning rate for the steepest descent optimizer
     */
    real_t learningRate() const;

    /**
     * Returns the momentum for the steepest descent optimizer
     *
     * @return The momentum for the steepest descent optimizer
     */
    real_t momentum() const;

    /**
     * Returns the path to the NN layout and weights file
     *
     * @return The path to the NN layout and weights file
     */
    const std::string& networkFile() const;

    const std::string& networkGraphFile() const;

    /**
     * Returns the path to the *.nc file containing the training sequences
     *
     * @return The path to the *.nc file containing the training sequences
     */
    const std::vector<std::string>& trainingFiles() const;

    /**
     * Returns the path for .nc cache files
     *
     * @return Path of .nc cache files
     */
    const std::string& cachePath() const;

    /**
     * Returns the path to the *.nc file containing the validation sequences
     *
     * @return The path to the *.nc file containing the validation sequences
     */
    const std::vector<std::string>& validationFiles() const;

    /**
     * Returns the path to the *.nc file containing the test sequences
     *
     * @return The path to the *.nc file containing the test sequences
     */
    const std::vector<std::string>& testFiles() const;

    /**
     * Returns the seed for the random number generator
     *
     * @return The seed for the random number generator
     */
    unsigned randomSeed() const;

    /**
     * Returns the sequence length to which the training set is truncated
     * 
     * @return sequence truncation length
     */
    unsigned truncateSeqLength() const;

    /**
     * Returns the distribution type of the initial weights
     *
     * @return The distribution type of the initial weights
     */
    distribution_type_t weightsDistributionType() const;

    /**
     * Returns the minimum value of the uniform distribution of the initial weights
     *
     * @return The minimum value of the uniform distribution of the initial weights
     */
    real_t weightsDistributionUniformMin() const;

    /**
     * Returns the maximum value of the uniform distribution of the initial weights
     *
     * @return The maximum value of the uniform distribution of the initial weights
     */
    real_t weightsDistributionUniformMax() const;

    /**
     * Returns the sigma of the normal distribution of the initial weights
     *
     * @return The sigma of the normal distribution of the initial weights
     */
    real_t weightsDistributionNormalSigma() const;

    /**
     * Returns the mean of the normal distribution of the initial weights
     *
     * @return The mean of the normal distribution of the initial weights
     */
    real_t weightsDistributionNormalMean() const;

    /**
     * Returns the sigma of the normal distribution of the input noise
     *
     * @return The sigma of the normal distribution of the input noise
     */
    real_t inputNoiseSigma() const;

    /**
     * Returns the amount of context frames appended from the left
     *
     * @return the amount of context frames appended from the left
     */
    int inputLeftContext() const;

    /**
     * Returns the amount of context frames appended from the right
     *
     * @return the amount of context frames appended from the right
     */
    int inputRightContext() const;

    /**
     * Returns the time lag of the output targets
     */
    int outputTimeLag() const;

    /**
     * Returns the sigma of the normal distribution of the weight noise
     *
     * @return The sigma of the normal distribution of the weight noise
     */
    real_t weightNoiseSigma() const;

    /**
     * Returns the fraction of the training set to use
     *
     * @return The fraction of the training set to use
     */
    real_t trainingFraction() const;

    /**
     * Returns the validation of the training set to use
     *
     * @return The validation of the training set to use
     */
    real_t validationFraction() const;

    /**
     * Returns the test of the training set to use
     *
     * @return The test of the training set to use
     */
    real_t testFraction() const;

    /**
     * Returns the path to the trained network file
     *
     * @return The path to the trained network file
     */
    const std::string& trainedNetworkFile() const;

    /**
     * Returns the forward pass output file format
     *
     * @return The forward pass output file format
     */
    feedforwardformat_type_t feedForwardFormat() const;

    /**
     * Returns whether output activations should be unstandardized in case of regression.
     *
     * @return whether output activations should be unstandardized in case of regression
     */
    bool revertStd() const;
    bool outputHtk() const;

    /**
     * Returns the (HTK) feature kind of the output (only for HTK output)
     *
     * @return the (HTK) feature kind of the output (only for HTK output)
     */
    unsigned outputFeatureKind() const;

    /**
     * Returns the feature period in seconds (only for HTK output)
     *
     * @return the feature period in seconds (only for HTK output)
     */
    real_t featurePeriod() const;

    /**
     * Returns the path to the forward pass input file
     *
     * @return The path to the forward pass input file
     */
    const std::vector<std::string>& feedForwardInputFiles() const;

    /**
     * Returns the path to the forward pass output file
     *
     * @return The path to the forward pass output file
     */
    const std::string& feedForwardOutputFile() const;

    /**
     * Returns the autosave filename prefix
     *
     * @return The autosave filename prefix
     */
    const std::string& autosavePrefix() const;

    /**
     * Returns the autosave file from which training will continue
     *
     * @return The autosave file from which training will continue
     */
    const std::string& continueFile() const;
    
    /* Add 16-02-22 Wang: for WE updating */
    /* Return the weIDDim and so on*/
    bool weUpdate() const;

    const std::string& weBankPath() const;
    
    const int&          weNoiseStartDim() const;

    const int&          weNoiseEndDim() const;      // noise adding, end at which dimension

    const real_t&       weNoiseDev() const;         // noise std.

    const std::string& mseWeightPath() const;

    const std::string& weightMaskPath() const;

    const int& weightMaskOpt() const;
    
    const std::string& trainedParameterPath() const;
    
    const std::string& trainedParameterCtr() const;
    
    const std::string& mdnFlagPath() const;

    const std::string& datamvPath() const;
    
    const std::string& dataTargetPath() const;

    const std::string& dataSourcePath() const;

    const std::string& mdnVarScaleGen() const;

    const int& EMIterNM() const;

    const real_t& getVarInitPara() const;

    const real_t& getVFloorPara() const;

    const real_t& getWInitPara() const;

    const bool& getTiedVariance() const;
    
    bool mdnFlag() const;

    const unsigned& weIDDim() const;
    
    const unsigned& weDim() const;
    
    const unsigned& txtChaDim() const;

    const std::string& chaBankPath() const;
    
    real_t weLearningRate() const;

    const real_t& highwayGateBias() const;

    const real_t& mdnPara() const;
    
    const int& outputFromWhichLayer() const;

    const bool& outputFromGateLayer() const;

    const real_t& lrDecayRate() const;

    const std::string& mdnDyn() const;

    const std::string& tanhAutoregressive() const;

    const std::string& arOrder() const;
    
    const std::string& arConfig() const;

    const int& zeroFilter() const;
    
    const int& arrmdnLearning() const;

    const real_t& arRMDNInitVar() const;

    const int& arRMDNUpdateInterval() const;
    
    const int& KLDOutputDataType() const;

    const real_t& lrFactor() const;

    const unsigned& optimizerOption() const;
    
    const real_t& optimizerSecondLR() const;

    const std::string& auxillaryDataDir() const;
    
    const std::string& auxillaryDataExt() const;
    
    const int& auxillaryDataTyp() const;
    
    const int& auxillaryDataDim() const;

    const std::string& secondOutputOpt() const;

    const real_t& lstmForgetIni() const;

    const int& probDataDim() const;

    const std::string& probDataDir() const;

    const int& aggregateOpt() const;

    const int& scheduleSampOpt() const;
    const std::string& scheduleSampOpts() const;

    const int& scheduleSampPara() const;
    const int& dropoutbeforeVAE() const;

    const real_t& mdnUVSigThreshold() const;

    const int& mdnSoftMaxGenMethod() const;
    const std::string& mdnSoftMaxGenMethods() const;
    
    const std::string& exInputDir() const;
    const std::string& exInputExt() const;
    const int& exInputDim() const;
    
    const std::string& exInputDirs() const;
    const std::string& exInputExts() const;
    const std::string& exInputDims() const;
    const int& exInputType() const;
    const int& exInputReso() const;
    
    const std::string& exOutputDirs() const;
    const std::string& exOutputExts() const;
    const std::string& exOutputDims() const;
    const int& exOutputType() const;
    const int& exOutputReso() const;
    
    const int& verboseLevel() const;

    const int& fakeEpochNum() const;

    const int& runningMode()  const;

    const int& mdnVarUpdateEpoch() const;

    const int& vaePlotManifold() const;

    const std::string& resolutions() const;

    const int& vaeEncoderOutputLayer() const;

    const std::string& vaeCodeInputDir() const;

    const int& vaeGetVQIndex() const;

    const int& seqWeightsFlag() const;
    
    const int& vaePutVQIndex() const;
    
    const real_t& softmaxTemperature() const;

    const int& waveNetMemSaveFlag() const;

    const int& dataCheckerIdx() const;
};


#endif // CONFIGURATION_HPP
