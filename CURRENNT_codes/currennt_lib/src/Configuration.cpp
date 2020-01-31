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

#include "Configuration.hpp"
#include "rapidjson/document.h"
#include "rapidjson/filestream.h"

#include <limits>
#include <fstream>
#include <sstream>

#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <boost/random/random_device.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/replace.hpp>

namespace po = boost::program_options;

#define DEFAULT_UINT_MAX std::numeric_limits<unsigned>::max(), "inf"

Configuration *Configuration::ms_instance = NULL;


namespace internal {

std::string serializeOptions(const po::variables_map &vm) 
{
    std::string s;

    for (po::variables_map::const_iterator it = vm.begin(); it != vm.end(); ++it) {
        if (it->second.value().type() == typeid(bool))
            s += (it->first + '=' + 
		  boost::lexical_cast<std::string>(boost::any_cast<bool>(it->second.value())));
        else if (it->second.value().type() == typeid(unsigned))
            s += (it->first + '=' + 
		  boost::lexical_cast<std::string>(boost::any_cast<unsigned>(it->second.value())));
        else if (it->second.value().type() == typeid(float))
            s += (it->first + '=' + 
		  boost::lexical_cast<std::string>(boost::any_cast<float>(it->second.value())));
        else if (it->second.value().type() == typeid(double))
            s += (it->first + '=' + 
		  boost::lexical_cast<std::string>(boost::any_cast<double>(it->second.value())));
        else if (it->second.value().type() == typeid(std::string))
            s += (it->first + '=' + boost::any_cast<std::string>(it->second.value()));
        else if (it->second.value().type() == typeid(int))
            s += (it->first + '=' + 
		  boost::lexical_cast<std::string>(boost::any_cast<int>(it->second.value())));

        s += ";;;";
    }

    return s;
}

void deserializeOptions(const std::string &autosaveFile, std::stringstream *ss)
{
    // open the file
    std::ifstream ifs(autosaveFile.c_str(), std::ios::binary);
    if (!ifs.good())
        throw std::runtime_error("Cannot open file");

    // calculate the file size in bytes
    ifs.seekg(0, std::ios::end);
    size_t size = ifs.tellg();
    ifs.seekg(0, std::ios::beg);

    // read the file into a buffer
    char *buffer = new char[size + 1];
    ifs.read(buffer, size);
    buffer[size] = '\0';

    // parse the JSON file
    rapidjson::Document jsonDoc;
    if (jsonDoc.Parse<0>(buffer).HasParseError())
        throw std::runtime_error(std::string("Parse error: ") + jsonDoc.GetParseError());

    // extract the options
    if (!jsonDoc.HasMember("configuration"))
        throw std::runtime_error("Missing string 'configuration'");

    std::string s = jsonDoc["configuration"].GetString();
    (*ss) << boost::replace_all_copy(s, ";;;", "\n");
}

} // namespace internal


Configuration::Configuration(int argc, const char *argv[])
{
    if (ms_instance)
        throw std::runtime_error("Static instance of class Configuration already created");
    else
        ms_instance = this;

    //std::cout << "ver:2018/10/24" << std::endl;
    //std::cout << "ver:2019/11/05" << std::endl;
    std::cout << "ver:2020/01/30" << std::endl;
    
    std::string optionsFile;
    std::string optimizerString;
    std::string weightsDistString;
    std::string feedForwardFormatString;

    std::string trainingFileList;
    std::string validationFileList;
    std::string testFileList;
    std::string feedForwardInputFileList;

    int dustbin;
    // create the command line options
    po::options_description commonOptions("Common options");
    commonOptions.add_options()
        ("help",                                                                              
	 "shows this help message")
	("version",
	 "shows the version")
        ("options_file",       
	 po::value(&optionsFile),                                       
	 "reads the command line options from the file")
        ("network",            
	 po::value(&m_networkFile)      ->default_value("network.jsn"), 
	 "sets the file containing the layout and weights of the neural network")
        ("network_graph",            
	 po::value(&m_networkGraphFile) ->default_value(""), 
	 "generate a graphic dot file, which can be used by dot to generate network graph")
        ("cuda",               
	 po::value(&m_useCuda)          ->default_value(true),          
	 "use CUDA to accelerate the computations")
        ("list_devices",       
	 po::value(&m_listDevices)      ->default_value(false),         
	 "display list of CUDA devices and exit")
        ("parallel_sequences", 
	 po::value(&m_parallelSequences)->default_value(1),             
	 "sets the number of parallel calculated sequences")
        ("random_seed",        
	 po::value(&m_randomSeed)       ->default_value(0u),            
	 "sets the seed for the random number generator (0 = auto)")
	("verbose",        
	 po::value(&m_verbose)       ->default_value(0),            
	 "information output level: 0 (default), 1")
	("quick_network_test",        
	 po::value(&m_quickTestNetwork)       ->default_value(0),            
	 "load only 1 utterance and test network: 0 (default), 1")
        ;

    po::options_description feedForwardOptions("Forward pass options");
    feedForwardOptions.add_options()
        ("ff_output_format", 
	 po::value(&feedForwardFormatString)->default_value("single_csv"),  
	 "output format for output layer activations (htk, csv or single_csv)")
        ("ff_output_file", 
	 po::value(&m_feedForwardOutputFile)->default_value(""), 
	 "sets the name of the output directory")
        ("ff_output_kind", 
	 po::value(&m_outputFeatureKind)->default_value(9),                   
	 "sets the parameter kind in case of HTK output (9: user, consult HTK book for details)")
        ("feature_period", 
	 po::value(&m_featurePeriod)->default_value(10),                      
	 "sets the feature period in case of HTK output (in seconds)")
        ("ff_input_file",  
	 po::value(&feedForwardInputFileList),                                
	 "sets the name(s) of the input file(s) in forward pass mode")
        ("revert_std",     
	 po::value(&m_revertStd)->default_value(true),                        
	 "de-normalize the generated data using mean and variance in data.nc (default: true)")
        ("output_htk",     
	 po::value(&m_outputHtk)->default_value(true),                        
	 "whether generate HTK output (big-endian, float32) (default: true)")
	/* Add 16-04-08 to tap in the output of arbitary layer */
	("output_from",    
	 po::value(&m_outputTapLayer)->default_value(-1),                     
	 "from which layer to get the output? (input layer is 0. Default: from the output layer) ")
	("output_from_gate",
	 po::value(&m_outputGateOut)->default_value(false),                  
	 std::string(
	      std::string("if the output layer is a gate layer, get output from gate instead") +
	      std::string("of transformation units? (default false)")).c_str())
	("output_a_few_frames",
	 po::value(&m_outputFrameNum)->default_value(-1),
	 std::string(
	      std::string("if we only want to save the first N frames of the output sequence,") +
	      std::string("use --output_a_few_frames N. (default -1, all frames)")).c_str())
	("mdnUVSigThreshold",
	 po::value(&m_mdnUVSigThreshold)->default_value(0.5),                  
	 std::string("Threhold for uvsigmoid (default 0.5)").c_str())
	("mdnSoftmaxGenMethod",
	 po::value(&m_mdnSoftMaxGenMethod)->default_value(0),
	 std::string(
	      std::string("Method to generate output from the softmax layer (global option):")+
	      std::string("\n\t0: one-hot (default)") + 
	      std::string("\n\t1: soft merge") + 
	      std::string("\n\t2: random sampling")).c_str())
	("mdnSoftmaxGenMethods",
	 po::value(&m_mdnSoftMaxGenMethods)->default_value(""),
	 std::string(
	      std::string("Method to generate output from softmax (X_X_X, for each softmax).")+
	      std::string("\n\t0: one-hot (default)") + 
	      std::string("\n\t1: soft merge") + 
	      std::string("\n\t2: random sampling") +
	      std::string("\n\t9: skip flag for none-softmax unit")).c_str())
	("fakeEpochNum",
	 po::value(&m_fakeEpochNum)->default_value(-1),
	 "Not used")
	("vaeGenMethod",
	 po::value(&m_vaePlotManifold)->default_value(0),
	 std::string(
	      std::string("Option for inference in VAE. When z is 2-Dim and vaeManifold=1, ") +
	      std::string("vae layer will read in the code z from fraction.outputs()")).c_str())
	("vaeEncoderOutputLayer",
	 po::value(&m_vaeEncoderOutputLayer)->default_value(-1),
	 std::string("From which layer the code (latent variables) should be generated?").c_str())
	("vaeCodeInputDir",
	 po::value(&m_vaeCodeInputDir)->default_value(""),
	 std::string("Directory of latent variables that will be fed into VAE decoder").c_str())
	("vaeGetVQIndex",
	 po::value(&m_vaeGetVqIndex)->default_value(0),
	 std::string("Whether to output VQIdx").c_str())
	("vaeInputCodeType",
	 po::value(&m_vaePutVqIndex)->default_value(2),
	 std::string("Type of input data to vqLayer? 0: code vector; 1: prob; 2: index").c_str())
	("ARmodelSpecialGenMode",
	 po::value(&m_ARmodelSpecialGenMode)->default_value(0),
	 std::string("Use AR model as encoding-decoding model (1); default not (0)").c_str())
	("waveNetMemSave",
	 po::value(&m_waveNetSaveMemFlag)->default_value(1),
	 std::string("Use memory-save mode for WaveNet in generation? (default yes)").c_str())
	("SoftmaxTemperature",
	 po::value(&m_softmaxTemperature) ->default_value(1),
	 "Temperature for softmax. Only used for generation")
	;
    
    po::options_description trainingOptions("Training options");
    trainingOptions.add_options()
        ("train",               
	 po::value(&m_trainingMode)     ->default_value(false),                 
	 "enables the training mode")
	("print_weight_to",        
	 po::value(&m_printWeightPath)  ->default_value(""),                 
	 "print the weight to binary file")
	("print_weight_opt",
	 po::value(&m_printWeightOpt)   ->default_value(1),
	 std::string(
	     std::string("option for printing weight. 0: only weights (default) and macro; ")+
	     std::string("1: weights, macro with layertype. For hts_engine.") +
	     std::string("2: translate *.autosave to *.jsn")).c_str())
        ("stochastic", 
	 po::value(&m_hybridOnlineBatch)->default_value(true),                          
	 "enables weight updates after every mini-batch of parallel calculated sequences")
        ("hybrid_online_batch", 
	 po::value(&m_hybridOnlineBatch)->default_value(true),                 
	 "same as --stochastic (for compatibility)")
        ("shuffle_fractions",   
	 po::value(&m_shuffleFractions) ->default_value(false),                 
	 "shuffles mini-batches in stochastic gradient descent")
        ("shuffle_sequences",   
	 po::value(&m_shuffleSequences) ->default_value(false),                 
	 "shuffles sequences within and across mini-batches")
        ("max_epochs",          
	 po::value(&m_maxEpochs)        ->default_value(DEFAULT_UINT_MAX),      
	 "sets the maximum number of training epochs")
        ("max_epochs_no_best",  
	 po::value(&m_maxEpochsNoBest)  ->default_value(20),                    
	 "sets the maximum number of epochs in which no new lowest error could be achieved")
        ("validate_every",      
	 po::value(&m_validateEvery)    ->default_value(1),                     
	 "sets the number of epochs until the validation error is computed")
        ("test_every",          
	 po::value(&m_testEvery)        ->default_value(1),                     
	 "sets the number of epochs until the test error is computed")
        ("optimizer",           
	 po::value(&optimizerString)    ->default_value("steepest_descent"),    
	 "sets the optimizer used for updating the weights")
        ("learning_rate",       
	 po::value(&m_learningRate)     ->default_value((real_t)1e-5, "1e-5"),  
	 "sets the learning rate for the steepest descent optimizer")
        ("momentum",            
	 po::value(&m_momentum)         ->default_value((real_t)0.9,  "0.9"),   
	 "sets the momentum for the steepest descent optimizer")
        ("weight_noise_sigma",  
	 po::value(&m_weightNoiseSigma) ->default_value((real_t)0),            
	 std::string(
	      std::string("sets the standard deviation of the weight noise added for") + 
	      std::string(" the gradient calculation on every batch")).c_str())
        ("save_network",        
	 po::value(&m_trainedNetwork)   ->default_value("trained_network.jsn"), 
	 "sets the file name of the trained network that will be produced")
	
	/* Add 16-02-22 Wang: for WE updating */
	("welearning_rate",     
	 po::value(&m_weLearningRate)   ->default_value((real_t)-1, "0"),         
	 "sets the learning rate for we.")
	("mseWeight",           
	 po::value(&m_mseWeightPath)    ->default_value(""),                      
	 "path to the weight for calculating the SSE and back-propagation (binary float data)")
	("LRDecayRate",       
	 po::value(&m_lr_decay_rate)    ->default_value(0.1),                     
	 "The rate to decay learning rate (default 0.1). Use Optimizer=4")
	/* Add 04-13 Wang: for weight mask*/
	("weight_mask",         
	 po::value(&m_weightMaskPath)   ->default_value(""),                      
	 std::string(
	      std::string("path to the network transformation matrix mask. The number of data") + 
	      std::string(" in this file should be equal to the network parameters")).c_str())
	("weight_mask_opt",         
	 po::value(&m_weightMaskOpt)   ->default_value(0),                      
	 std::string(
	      std::string("Option to read and use the weight mask\n") + 
	      std::string("\n\t0: the weight mask for normal NN weight (default)") +
	      std::string("\n\t1: the weight mask for embedded vectors") +
	      std::string("\n\t2: the weight mask for embedded vectors and NN weight")).c_str())
	
	/* Add 0504 Wang: for MDN flag*/
	("mdn_config",          
	 po::value(&m_mdnFlagPath)      ->default_value(""),                      
	 "path to the MDN flag. ")
	("mdn_samplePara",      
	 po::value(&m_mdnSamplingPara)  ->default_value((real_t)-4.0, "-4.0"),    
	 std::string(
	      std::string("parameter for MDN sampling. \n") + 
	      std::string("mdn_samplePara > 0: sampling with var scaled by mdn_samplePara.\n") +
	      std::string("mdn_samplePara = -1.0: generate the parameter of the distribution.\n")+ 
	      std::string("mdn_samplePara < -1.0: not use mdn and mdn generation.")).c_str())
	("mdn_EMGenIter",       
	 po::value(&m_EMGenIter)        ->default_value(5, "5"),                  
	 "Number of iterations for EM generation in MDN (default 5). ")
	("varInitPara",         
	 po::value(&m_varInitPara)      ->default_value(0.5, "0.5"), 
	 "Parameter to initialize the bias of MDN mixture unit (default 0.5)")
	("vFloorPara",          
	 po::value(&m_vFloorPara)       ->default_value(0.0001, "0.0001"), 
	 "Variance scale parameter for the variance floor (default 0.0001)")
	("wInitPara",           
	 po::value(&m_wInitPara)        ->default_value(1.0, "1.0"), 
	 "Internal use")
	("tieVariance",         
	 po::value(&m_tiedVariance)     ->default_value(false,"false"), 
	 std::string(
	      std::string("Whether the variance should be tied across dimension?(default false)")+ 
	      std::string("It will be ignored if tieVarianceFlag is in the model file (.autosave)")
	      ).c_str())
	("mdn_sampleParaVec",   
	 po::value(&m_mdnVarScaleGen)   ->default_value(""), 
	 std::string(
	      std::string("The binary vector of coef to scale variance of the mixture model.") + 
	      std::string("The length of vector should be equal to the dimension of output of")+
	      std::string(" the network. Sigmoid and softmax unit will ignore it")).c_str())
	("mdn_secondOutput",
	 po::value(&m_secondOutputOpt)  ->default_value(""),
	 std::string(
	      std::string("Control the second output of MDN for feedback. A string of 1/0.")  +
	      std::string("1: this MDNUnit use para to feedback. 0: use output")).c_str())
	("mdnDyn",            
	 po::value(&m_mdnDyn)           ->default_value(""), 
	 std::string("Obsolete. Please use ARORDER").c_str())
	("tanhAutoReg",       
	 po::value(&m_tanhAutoregressive) ->default_value("1"), 
	 std::string("Obsolete. Please use ARConfig").c_str())
	("AROrder",            
	 po::value(&m_arOrder)->default_value(""), 
	 std::string(
	      std::string("Order of AR dependecy\n")+
	      std::string("Format: n1_n2_...nk.._nK, where nk is the AR oder\n") +
	      std::string(" for the k-th MDN unit. If k-th MDN unit is sigmoid/softmax\n") +
	      std::string(" nk will have no effect on the MDN unit")).c_str())
	("ARConfig",       
	 po::value(&m_arConfig)->default_value("0"), 
	 std::string(
	      std::string("Configuration of AR model\n") + 
	      std::string("Format: c1_c2_ck_cK, where ck is the config for the k-th unit.\n") +
	      std::string(" If k-th MDN doesn't use AR dependnecy, ck has no effect.\n") +
	      std::string("Value (default 0)") +
	      std::string("\t0: time-invariant, plain form AR\n") +
	      std::string("\t1: time-invariant, AR filter as casecade of 1st-order filter\n") +
	      std::string("\t2: time-invariant, AR filter as casecade with complex poles\n") +
	      std::string("\t3: time-invariant, AR parameterized by reflection coefficients\n") +
	      std::string("\t4: time-variant,   AR parameterized by reflection coefficients")
		     ).c_str())
	("ReserverZeroFilter", 
	 po::value(&m_setDynFilterZero)   ->default_value(0), 
	 "Reserved option for MDN Mixture Dyn units. Don't use it if you don't know it.")
	("arrmdnLearning",     
	 po::value(&m_arrmdnLearning)     ->default_value(0), 
	 "An option to set the learning rate for ARRMDN. Don't use it if you don't know the code")
	("arrmdnInitVar",      
	 po::value(&m_ARRMDNInitVar)      ->default_value(0.01), 
	 "The variance of Gaussian distribution for initialization the AR parameter")
	("arrmdnUpdateInterval", 
	 po::value(&m_ARRMDNUpdateInterval)->default_value(-1), 
	 std::string(
	      std::string("Option for the classical form AR model learning.(default not use) ") + 
	      std::string("Increase the order of AR model every N epochs ")).c_str())
	("Optimizer",            
	 po::value(&m_optimizerOption)     ->default_value(0), 
	 std::string(
	      std::string("Optimization technique: ") + 
	      std::string("\n\t0: normal gradient descent SGD (default)") + 
	      std::string("\n\t1: AdaGrad (except the Trainable MDNLayer).") + 
	      std::string("\n\t2: Average SGC over the utterance.") + 
	      std::string("\n\t3: SGD then AdaGrad (together with --OptimizerSecondLR)")+
	      std::string("\n\t4: SGD decay the learning rate after validation failed")+
	      std::string("\n\t5: Adam  (except the Trainable MDNLayer) ")).c_str())
	("OptimizerSecondLR",    
	 po::value(&m_secondLearningRate)  ->default_value(0.01), 
	 "Optimizer==3, it requirs additional learning rate for AdaGrad (0.01 default)")
	("ScheduleSampOpt",
	 po::value(&m_scheduleSampOpt)  ->default_value(0),
	 "Training/generation Method for AR model. Default 0 (no dropout, no special tricl)")
	("ScheduleSampOpts",
	 po::value(&m_scheduleSampOpts)  ->default_value(""),
	 "ScheduleSampOpt for multiple MDNUnits X_X_X")
	("ScheduleSampPara",
	 po::value(&m_scheduleSampPara) ->default_value(0),
	 "Parameter for schedule sampling. Default 0")
	("dropoutbeforeVAE",
	 po::value(&m_dropoutbeforeVAE) ->default_value(0),
	 "Whether droptout should be used on encoder?. Default not")
	("runningMode",
	 po::value(&m_runningMode)      ->default_value(0),
	 "Training mode of CURRENNT.\n\t0: default\n\t1: skip layers with 0 LR during backprop")
	("mdnVarFixEpochNum",
	 po::value(&m_mdnVarFixEpochNum)->default_value(-1),
	 "Fix the variance of mdn (GMM) as 1 for this number of epochs. Default (not use)")
	("resolutions",
	 po::value(&m_resolutions)->default_value(""),
	 "resolution defined in network.jsn. Format: res1_res2_res3")
	("seqWeightsFlag",
	 po::value(&m_seqWeightsFlag)->default_value(0),
	 "Use weighted evaluated (sequence level)")
	("whichDataChecker",
	 po::value(&m_dataCheckerIdx)->default_value(-1),
	 "Which data checker should be used to pre-check the data? (default -1 (None))")
	("BatchNormGenMode",
	 po::value(&m_batchnorm_genmode)->default_value(0),
	 std::string(
	    std::string("0: use utterance mean/std to normalize data (non-AR model) (default);") +
	    std::string("1: use saved mean/std from training data")).c_str())
	("F0MeanForSourceModule",
	 po::value(&m_f0dataMean_signalgen)->default_value(-1),
	 std::string(
	    std::string("Mean of F0 data. This will be used by NSF source module to recover ") +
	    std::string("the linear F0 value if the input F0 has been normalized")).c_str())
	("F0StdForSourceModule",
	 po::value(&m_f0dataStd_signalgen)->default_value(-1),
	 std::string(
	    std::string("Std of F0 data. This will be used by NSF source module to recover ") +
	    std::string("the linear F0 value if the input F0 has been normalized")).c_str())
	;

    po::options_description autosaveOptions("Autosave options");
    autosaveOptions.add_options()
        ("autosave",        
	 po::value(&m_autosave)            ->default_value(false), 
	 "enables autosave after every epoch")
        ("autosave_best",        
	 po::value(&m_autosaveBest)        ->default_value(false), 
	 "enables autosave on best validation error")
        ("autosave_prefix", 
	 po::value(&m_autosavePrefix),                 
	 "prefix for autosave files; e.g. 'abc/mynet-' -> 'mynet-epoch005.autosave' in dir 'abc'")
        ("continue",        
	 po::value(&m_continueFile),                   
	 "continues training from an autosave file")
        ;

    po::options_description dataFilesOptions("Data file options");
    dataFilesOptions.add_options()
        ("train_file",        
	 po::value(&trainingFileList),                                 
	 "sets the *.nc file(s) containing the training sequences")
        ("val_file",          
	 po::value(&validationFileList),                               
	 "sets the *.nc file(s) containing the validation sequences")
        ("test_file",         
	 po::value(&testFileList),                                     
	 "sets the *.nc file(s) containing the test sequences")
        ("train_fraction",    
	 po::value(&m_trainingFraction)  ->default_value((real_t)1), 
	 "sets the fraction of the training set to use")
        ("val_fraction",      
	 po::value(&m_validationFraction)->default_value((real_t)1), 
	 "sets the fraction of the validation set to use")
        ("test_fraction",     
	 po::value(&m_testFraction)      ->default_value((real_t)1), 
	 "sets the fraction of the test set to use")
        ("truncate_seq",      
	 po::value(&m_truncSeqLength)    ->default_value(0),         
	 "enables training sequence truncation to given maximum length (0 to disable)")
        ("truncate_seq_n_segments",      
	 po::value(&m_truncNsegments)    ->default_value(0),         
	 "truncate sequence into n segments with roughly equal length (0 to disable)")	
        ("input_noise_sigma", 
	 po::value(&m_inputNoiseSigma)   ->default_value((real_t)0), 
	 "sets the standard deviation of the input noise for training sets")
        ("input_left_context", 
	 po::value(&m_inputLeftContext) ->default_value(0), 
	 "sets the number of left context frames (first frame is duplicated as necessary)")
        ("input_right_context", 
	 po::value(&m_inputRightContext)->default_value(0), 
	 "sets the number of right context frames (last frame is duplicated as necessary)")
        ("output_time_lag",   
	 po::value(&m_outputTimeLag)->default_value(0),              
	 "time lag for training targets (0 = predict current frame, 1 = predict previous, etc.)")
        ("cache_path",        
	 po::value(&m_cachePath)         ->default_value(""),        
	 "sets the cache path where the .nc data is cached for random access")
	
	/* Add 16-02-22 Wang: for WE updating */
	("weExternal",          
	 po::value(&m_weUpdate)          ->default_value(false),    
	 "whether update the input word embedding vectors (false)")
	("weIDDim",           
	 po::value(&m_weIDDim)           ->default_value(-1),       
	 "the WE index is the ?-th dimension of the input vector? (-1)")
	("weDim",             
	 po::value(&m_weDim)             ->default_value(0),        
	 "the dimension of the word embedding vectors (0)")
	("weBank",            
	 po::value(&m_weBank)            ->default_value(""),       
	 "the path to the word vectors")
	("trainedModel",      
	 po::value(&m_trainedModelPath)  ->default_value(""), 
	 "the path to the trained model paratemeter")
	("trainedModelCtr",   
	 po::value(&m_trainedParameterCtr) ->default_value(""), 
	 std::string(
	      std::string("trainedModel controller") +
	      std::string(". A string of 0/1/2/3 whose length is #layer of NN.") + 
	      std::string("\n 0: not read this layer") + 
	      std::string("\n 1: read this layer if number of weights matches") +
	      std::string("\n 2: assume column number is the same") + 
	      std::string("\n 3: assume row numbe is the same") + 
	      std::string("\n (default: void, read in all parameters in option 1)")).c_str())
	("datamv",            
	 po::value(&m_datamvPath)          ->default_value(""), 
	 "NOT USED ANY MORE. Please use target_data_ms")
	("target_data_ms",            
	 po::value(&m_dataTargetPath)          ->default_value(""), 
	 "Mean and std of target data, binary float32 vector, size = target_layer_size * 2")
	("source_data_ms",            
	 po::value(&m_dataSourcePath)          ->default_value(""), 
	 "Mean and std of input data, binary float32 vector, size = input_layer_size * 2")
	("txtChaDim",         
	 po::value(&m_chaDimLstmCharW)     ->default_value(0), 
	 "OBSOLETE: the dimension of the bag of character for LstmCharW")
	("txtBank",           
	 po::value(&m_chaBankPath)         ->default_value(""),       
	 "OBSOLETE:the path to the character vectors for LstmCharW")
	("weNoiseStartDim",   
	 po::value(&m_weNoiseStartDim)     ->default_value(-2), 
	 "OBSOLETE:the first dimension that will be added with noise in the input layer")
	("weNoiseEndDim", 
	 po::value(&m_weNoiseEndDim)       ->default_value(-1), 
	 "OBSOLETE:the last dimension that  will be addded with noise in the input layer")
	("weNoiseDev",    
	 po::value(&m_weNoiseDev)          ->default_value(0.1), 
	 "standard deviation of the noise that will be added to the word vectors (default 0.1)")
	("targetDataType", 
	 po::value(&dustbin)   ->default_value(-1),
	 "Please use KLDTargetDataType.")
	("KLDTargetDataType", 
	 po::value(&m_KLDOutputDataType)   ->default_value(-1),   
	 std::string(
	      std::string("the type of the target data.") +
	      std::string("\n\t1: linear domain, zero-mean, uni-var") +
	      std::string("\n\t2: log domain, zero-mean, uni-var\n")).c_str())
	("KLDLRfactor",    
	 po::value(&m_lrFactor)            ->default_value(1),   
	 "the factor to scale the training criterion and gradient for KLD. default 1.0")
	("AuxDataPath",                   
	 po::value(&m_auxDataDir)       ->default_value(""),
	 "Auxillary data path. Path to the directory of data")
	("AuxDataType",                   
	 po::value(&m_auxDataTyp)       ->default_value(-1),
	 "Auxillary data type: 0 float, 1 int, 2 char")
	("AuxDataExt",                   
	 po::value(&m_auxDataExt)       ->default_value(""),
	 "Auxillary data extension, e.g. .bin, .sp")
	("AuxDataDim",                   
	 po::value(&m_auxDataDim)       ->default_value(-1),
	 "Auxillary data dimension")
	("probDataDir",                   
	 po::value(&m_probDataDir)       ->default_value(""),
	 "Probabilistic file directory")
	("probDataDim",                   
	 po::value(&m_probDataDim)       ->default_value(-1),
	 "Probabilistic data dimension")
	("ExtInputDir",
	 po::value(&m_exInputDir) ->default_value(""),
	 "External input directory")
	("ExtInputExt",
	 po::value(&m_exInputExt) ->default_value(""),
	 "External inut extension")
	("ExtInputDim",
	 po::value(&m_exInputDim) ->default_value(0),
	 "External input dimension")
	("ExtInputDirs",
	 po::value(&m_exInputDirs) ->default_value(""),
	 "External input directory")
	("ExtInputExts",
	 po::value(&m_exInputExts) ->default_value(""),
	 "External inut extension")
	("ExtInputDims",
	 po::value(&m_exInputDims) ->default_value(""),
	 "External input dimension")
	("ExtOutputDirs",
	 po::value(&m_exOutputDirs) ->default_value(""),
	 "External output directory")
	("ExtOutputExts",
	 po::value(&m_exOutputExts) ->default_value(""),
	 "External output extension")
	("ExtOutputDims",
	 po::value(&m_exOutputDims) ->default_value(""),
	 "External output dimension")
	("ExtOutputType",
	 po::value(&m_exOutputType) ->default_value(1),
	 "External output type")
	("ExtInputType",
	 po::value(&m_exInputType) ->default_value(1),
	 "External Input type")
	("ExtOutputReso",
	 po::value(&m_exOutputReso) ->default_value(-1),
	 "External Output resolution configuration")
	("ExtInputReso",
	 po::value(&m_exInputReso) ->default_value(-1),
	 "External Input resolution configuration")
	("file_ordered_lst_trn",
	 po::value(&m_fileOrderedLstTrn) ->default_value(""),
	 "Training files will be ordered according to the text file list")
	("file_ordered_lst_val",
	 po::value(&m_fileOrderedLstVal) ->default_value(""),
	 "Validation files will be ordered according to the text file list")
	("file_ordered_lst_tst",
	 po::value(&m_fileOrderedLstTst) ->default_value(""),
	 "Test set files will be ordered according to the text file list")
        ;

    po::options_description weightsInitializationOptions("Weight initialization options");
    weightsInitializationOptions.add_options()
        ("weights_dist",         
	 po::value(&weightsDistString)   ->default_value("uniform"),            
	 "sets the distribution type of the initial weights (uniform or normal)")
        ("weights_uniform_min",  
	 po::value(&m_weightsUniformMin) ->default_value((real_t)-0.1, "-0.1"), 
	 "sets the minimum value of the uniform distribution")
        ("weights_uniform_max",  
	 po::value(&m_weightsUniformMax) ->default_value((real_t)+0.1, "0.1"),  
	 "sets the maximum value of the uniform distribution")
        ("weights_normal_sigma", 
	 po::value(&m_weightsNormalSigma)->default_value((real_t)0.1, "0.1"),   
	 "sets the standard deviation of the normal distribution")
        ("weights_normal_mean",  
	 po::value(&m_weightsNormalMean) ->default_value((real_t)0.0, "0"),     
	 "sets the mean of the normal distribution")
	/* Add 16-04-02 Wang: for initiliaizing the bias for gate of Highway network */
	("highway_gate_bias",    
	 po::value(&m_highwayBias) -> default_value((real_t)-1.50, "-1.50"),    
	 "The bias for the sigmoid function in the gate of highway block (default -1.50)")
	("lstm_forget_gate_bias",    
	 po::value(&m_lstmForgetIni) -> default_value((real_t)0.0, "0.0"),    
	 "The bias to the output of forget gate in LSTM during initialization (default 0.0)")
	("aggregate_syn_opt",
	 po::value(&m_flagAggregateSyn) ->default_value(0),
	 std::string(
	      std::string("Option for generation using aggregation in feedback layer") +
	      std::string("\n 0: no aggregation for generation (even it is used in training)") + 
	      std::string("\n 1: use aggregation if it is used. (default is 0)")).c_str())
        ;

    po::positional_options_description positionalOptions;
    positionalOptions.add("options_file", 1);

    // parse the command line
    po::options_description visibleOptions;
    visibleOptions.add(commonOptions);
    visibleOptions.add(feedForwardOptions);
    visibleOptions.add(trainingOptions);
    visibleOptions.add(autosaveOptions);
    visibleOptions.add(dataFilesOptions);
    visibleOptions.add(weightsInitializationOptions);

    po::options_description allOptions;
    allOptions.add(visibleOptions);

    po::variables_map vm;
    try {
        po::store(po::command_line_parser(argc, argv).options(allOptions).positional(positionalOptions).run(), vm);
        if (vm.count("options_file")) {
            optionsFile = vm["options_file"].as<std::string>();
            std::ifstream file(optionsFile.c_str(), std::ifstream::in);
            if (!file.is_open())
                throw std::runtime_error(std::string("Could not open options file '") + 
					 optionsFile + "'");
            po::store(po::parse_config_file(file, allOptions), vm);
        }
        po::notify(vm);
    }
    catch (const std::exception &e) {
        if (!vm.count("help")){
            std::cout << "Error while parsing the command line and/or options file: ";
	    std::cout << e.what() << std::endl;
	}

        std::cout << "Usage: currennt [options] [options-file]" << std::endl;
        std::cout << visibleOptions;

        exit(vm.count("help") ? 0 : 1);
    }

    if (vm.count("help")) {
        std::cout << "Usage: currennt [options] [options-file]" << std::endl;
        std::cout << visibleOptions;

        exit(0);
    }

    if (vm.count("version")){
	//std::cout << "2018/09/24: FFT3,2,1, equal noise intensity" << std::endl;
	//std::cout << "2019/02/06" << std::endl;
	exit(0);
    }
    
    // load options from autosave
    if (!m_continueFile.empty()) {
        try {
            std::stringstream ss;
            internal::deserializeOptions(m_continueFile, &ss);
            vm = po::variables_map();
            po::store(po::parse_config_file(ss, allOptions), vm);
            po::notify(vm);
        }
        catch (const std::exception &e) {
            std::cout << "Error while restoring configuration from autosave file: ";
	    std::cout << e.what() << std::endl;

            exit(1);
        }
    }

    // store the options for autosave
    m_serializedOptions = internal::serializeOptions(vm);

    // split the training file options
    boost::algorithm::split(m_trainingFiles, trainingFileList, 
			    boost::algorithm::is_any_of(";,"), 
			    boost::algorithm::token_compress_on);
    if (!validationFileList.empty())
        boost::algorithm::split(m_validationFiles, validationFileList, 
				boost::algorithm::is_any_of(";,"), 
				boost::algorithm::token_compress_on);
    if (!testFileList.empty())
        boost::algorithm::split(m_testFiles, testFileList, 
				boost::algorithm::is_any_of(";,"), 
				boost::algorithm::token_compress_on);
    if (!feedForwardInputFileList.empty())
        boost::algorithm::split(m_feedForwardInputFiles, feedForwardInputFileList, 
				boost::algorithm::is_any_of(";,"), 
				boost::algorithm::token_compress_on);

    // check the optimizer string
    if (optimizerString == "rprop")
        m_optimizer = OPTIMIZER_RPROP;
    else if (optimizerString == "steepest_descent")
        m_optimizer = OPTIMIZER_STEEPESTDESCENT;
    else {
        std::cout << "ERROR: Invalid optimizer. Possible values: steepest_descent, rprop.";
	std::cout << std::endl;
        exit(1);
    }

    // create a random seed
    if (!m_randomSeed)
        m_randomSeed = boost::random::random_device()();

    // check the weights distribution string
    if (weightsDistString == "normal")
        m_weightsDistribution = DISTRIBUTION_NORMAL;
    else if (weightsDistString == "uniform")
        m_weightsDistribution = DISTRIBUTION_UNIFORM;
    else if (weightsDistString == "uninorm")
	m_weightsDistribution = DISTRIBUTION_UNINORMALIZED;
    else {
        std::cout << "ERROR: Invalid initial weights distribution type.";
	std::cout << "Possible values: normal, uniform." << std::endl;
        exit(1);
    }

    // check the feedforward format string
    if (feedForwardFormatString == "single_csv")
        m_feedForwardFormat = FORMAT_SINGLE_CSV;
    else if (feedForwardFormatString == "csv")
        m_feedForwardFormat = FORMAT_CSV;
    else if (feedForwardFormatString == "htk")
        m_feedForwardFormat = FORMAT_HTK;
    else {
        std::cout << "ERROR: Invalid feedforward format string."; 
	std::cout << "Possible values: single_csv, csv, htk." << std::endl;
        exit(1);
    }

    // check data sets fractions
    if (m_trainingFraction <= 0 || 1 < m_trainingFraction) {
        std::cout << "ERROR: Invalid training set fraction. Should be 0 < x <= 1" << std::endl;
        exit(1);
    }
    if (m_validationFraction <= 0 || 1 < m_validationFraction) {
        std::cout << "ERROR: Invalid validation set fraction. Should be 0 < x <= 1" << std::endl;
        exit(1);
    }
    if (m_testFraction <= 0 || 1 < m_testFraction) {
        std::cout << "ERROR: Invalid test set fraction. Should be 0 < x <= 1" << std::endl;
        exit(1);
    }

    // print information about active command line options
    std::cout << "Configuration Infor:" << std::endl;
    if (m_trainingMode) {
        std::cout << "\tTraining Mode: Started in ";
	std::cout << (m_hybridOnlineBatch ? "hybrid online/batch" : "batch") << std::endl;

        if (m_shuffleFractions){
            std::cout << "\t\tMini-batches (parallel " << m_parallelSequences << " sequences each)";
	    std::cout << " will be shuffled during training." << std::endl;
	}
        if (m_shuffleSequences){
            std::cout << "\t\tSequences shuffled within and across mini-batches.\n" << std::endl;
	}
        if (m_inputNoiseSigma != (real_t)0){
	    if (m_exInputDim > 0 or m_exInputDims.size() > 0){
		m_inputNoiseSigma = 0.0;
		std::cout << "\t\tInput noise is disabled for external input" << std::endl;
	    }else{
		std::cout << "\t\tUsing input noise with std. of " << m_inputNoiseSigma << std::endl;
	    }
	}
        std::cout << "\t\tWritting network  to '" << m_trainedNetwork << "'." << std::endl;
        if (boost::filesystem::exists(m_trainedNetwork))
            std::cout << "\t\tWARNING: overwriting '" << m_trainedNetwork << "'" << std::endl;
	
    }else if(m_printWeightPath.size()>0){
	std::cout << "\tStarted in printing mode. ";
	std::cout << "Weight will be print to " << m_printWeightPath << std::endl;
	
    }else {
        std::cout << "\tStarted in forward pass mode." << std::endl;
        std::cout << "\tWritting output to '" << m_feedForwardOutputFile << "'." << std::endl;
        if (boost::filesystem::exists(m_feedForwardOutputFile))
            std::cout << "\t\tWARNING: overwriting '" << m_feedForwardOutputFile << std::endl;
    }

    if (m_trainingMode && !m_validationFiles.empty())
        std::cout << "\tValidation every " << m_validateEvery << " epochs." << std::endl;
    if (m_trainingMode && !m_testFiles.empty())
        std::cout << "\tTest  every " << m_testEvery << " epochs." << std::endl;

    if (m_trainingMode) {
        std::cout << "\n\tTraining epoch number maximum: ";
        if (m_maxEpochs != std::numeric_limits<unsigned>::max())
            std::cout << m_maxEpochs << std::endl;
        std::cout << "\n\tTraining epoch number no lowest validation error: ";
	std::cout << m_maxEpochsNoBest << std::endl;
    }
    
    if (m_autosave) {
	if (m_quickTestNetwork > 0)
	    m_autosave = false;
	else
	    std::cout << "\tAutosave after EVERY EPOCH enabled." << std::endl;
    }
    if (m_autosaveBest) {
	if (m_quickTestNetwork > 0)
	    m_autosaveBest = false;
	else
	    std::cout << "\tAutosave on BEST VALIDATION ERROR enabled." << std::endl;
    }

    if (m_useCuda){
        std::cout << "\tUtilizing the GPU on ";
	std::cout << m_parallelSequences << " sequences in parallel." << std::endl;
    }else
        std::cout << "\tWARNING: CUDA option not set. Computations on the CPU!" << std::endl;

    if (m_trainingMode) {
	std::cout << "\n\tInitialization method:" << std::endl;
        if (m_weightsDistribution == DISTRIBUTION_NORMAL){
            std::cout << "\t\tNormal dist. with mean, std:"; 
	    std::cout << m_weightsNormalMean << m_weightsNormalSigma;
        }else if (m_weightsDistribution == DISTRIBUTION_UNINORMALIZED)
	    std::cout << "\t\tUniform dist. with layer-wise range" << std::endl;
	else{
            std::cout << "\t\tUniform dist. with range [";
	    std::cout << m_weightsUniformMin << ", " << m_weightsUniformMax << "]";
	}
	std::cout << "\n\t\tRandom seed: " << m_randomSeed << std::endl;
	m_softmaxTemperature = 1.0;
    }

    if (m_mseWeightPath.size()>0){
	std::cout << "\tUsing MSE Weight: " << m_mseWeightPath  << std::endl;
    }

    
    // Checking 
    /* Add 16-02-22 Wang: for WE updating */
    if (m_weUpdate){
	// for checking:
	if (m_inputNoiseSigma > 0.0){
	    std::cout <<"\tWARNING: input vectors are used, input noise is turned off"<< std::endl;
	    m_inputNoiseSigma = 0.0;
	}
	if (m_weIDDim < 0 || m_weDim < 1 || m_weBank.size()<1){
	    std::cout << "\tERROR: Invalid configuration for WE updating" << std::endl;
	    exit(1);
	}
    }

    /*if (m_auxDataDir.size() > 0){
	std::cout << "\tUsing auxilary data. ";
	if (m_parallelSequences > 1){
	    std::cout << "Parallel training will be turned off." << std::endl;
	    m_parallelSequences = 1;
	}	
	std::cout << std::endl;
	}*/

    if (m_scheduleSampOpt == 6){
	m_parallelSequences = 1; // for beamsize generation in computeForwardPassGen()
    }
    
    if (m_feedForwardOutputFile.size() > 0 &&
	m_mdnVarFixEpochNum > 0 && m_mdnVarFixEpochNum < 999){
	std::cout << "\nGeneration mode, mdnVarFixEpochNum is not used";
	m_mdnVarFixEpochNum = -1;
    }

    if (m_dataCheckerIdx != -1){
	std::cout << "\nData checker is enabled:"  << m_dataCheckerIdx << std::endl;
    }

    if (m_datamvPath.size()){
	std::cout << "datamv is no longer supported. Please use --target_data_ms" << std::endl;
	exit(1);
    }
    std::cout << std::endl;
}

Configuration::~Configuration()
{
}

const Configuration& Configuration::instance()
{
    return *ms_instance;
}

const std::string& Configuration::serializedOptions() const
{
    return m_serializedOptions;
}

bool Configuration::trainingMode() const
{
    return m_trainingMode;
}

bool Configuration::generatingMode() const
{
    return !m_trainingMode;
}

bool Configuration::hybridOnlineBatch() const
{
    return m_hybridOnlineBatch;
}

bool Configuration::shuffleFractions() const
{
    return m_shuffleFractions;
}

bool Configuration::shuffleSequences() const
{
    return m_shuffleSequences;
}

bool Configuration::useCuda() const
{
    return m_useCuda;
}

bool Configuration::listDevices() const
{
    return m_listDevices;
}

bool Configuration::autosave() const
{
    return m_autosave;
}

bool Configuration::autosaveBest() const
{
    return m_autosaveBest;
}

Configuration::optimizer_type_t Configuration::optimizer() const
{
    return m_optimizer;
}

int Configuration::parallelSequences() const
{
    return (int)m_parallelSequences;
}

int Configuration::maxEpochs() const
{
    return (int)m_maxEpochs;
}

int Configuration::maxEpochsNoBest() const
{
    return (int)m_maxEpochsNoBest;
}

int Configuration::validateEvery() const
{
    return (int)m_validateEvery;
}

int Configuration::testEvery() const
{
    return (int)m_testEvery;
}

real_t Configuration::learningRate() const
{
    return m_learningRate;
}

real_t Configuration::momentum() const
{
    return m_momentum;
}

const std::string& Configuration::networkFile() const
{
    return m_networkFile;
}

const std::string& Configuration::networkGraphFile() const
{
    return m_networkGraphFile;
}

const std::vector<std::string>& Configuration::trainingFiles() const
{
    return m_trainingFiles;
}

const std::string& Configuration::cachePath() const
{
    return m_cachePath;
}


const std::vector<std::string>& Configuration::validationFiles() const
{
    return m_validationFiles;
}

const std::vector<std::string>& Configuration::testFiles() const
{
    return m_testFiles;
}

unsigned Configuration::randomSeed() const
{
    return m_randomSeed;
}

Configuration::distribution_type_t Configuration::weightsDistributionType() const
{
    return m_weightsDistribution;
}

real_t Configuration::weightsDistributionUniformMin() const
{
    return m_weightsUniformMin;
}

real_t Configuration::weightsDistributionUniformMax() const
{
    return m_weightsUniformMax;
}

real_t Configuration::weightsDistributionNormalSigma() const
{
    return m_weightsNormalSigma;
}

real_t Configuration::weightsDistributionNormalMean() const
{
    return m_weightsNormalMean;
}

real_t Configuration::inputNoiseSigma() const
{
    return m_inputNoiseSigma;
}

int Configuration::inputLeftContext() const
{
    return m_inputLeftContext;
}

int Configuration::inputRightContext() const
{
    return m_inputRightContext;
}

int Configuration::outputTimeLag() const
{   
    return m_outputTimeLag;
}

real_t Configuration::weightNoiseSigma() const
{
    return m_weightNoiseSigma;
}

real_t Configuration::trainingFraction() const
{
    return m_trainingFraction;
}

real_t Configuration::validationFraction() const
{
    return m_validationFraction;
}

real_t Configuration::testFraction() const
{
    return m_testFraction;
}

const std::string& Configuration::trainedNetworkFile() const
{
    return m_trainedNetwork;
}

Configuration::feedforwardformat_type_t Configuration::feedForwardFormat() const
{
    return m_feedForwardFormat;
}

real_t Configuration::featurePeriod() const
{
    return m_featurePeriod;
}

unsigned Configuration::outputFeatureKind() const
{
    return m_outputFeatureKind;
}

unsigned Configuration::truncateSeqLength() const
{
    return m_truncSeqLength;
}

const std::vector<std::string>& Configuration::feedForwardInputFiles() const
{
    return m_feedForwardInputFiles;

}

const std::string& Configuration::feedForwardOutputFile() const
{
    return m_feedForwardOutputFile;
}

const std::string& Configuration::autosavePrefix() const
{
    return m_autosavePrefix;
}

const std::string& Configuration::continueFile() const
{
    return m_continueFile;
}

/* Add 16-02-22 Wang: for WE updating */
const std::string& Configuration::weBankPath() const
{
    return m_weBank;
}

const std::string& Configuration::chaBankPath() const
{
    return m_chaBankPath;
}


const std::string& Configuration::mseWeightPath() const
{
    return m_mseWeightPath;
}

const std::string& Configuration::weightMaskPath() const
{
    return m_weightMaskPath;
}

const std::string& Configuration::trainedParameterPath() const
{
    return m_trainedModelPath;
}

const std::string& Configuration::trainedParameterCtr() const
{
    return m_trainedParameterCtr;
}
    
const unsigned& Configuration::weIDDim() const
{
    return m_weIDDim;
}
const unsigned& Configuration::weDim() const
{
    return m_weDim;
}

const unsigned& Configuration::txtChaDim() const
{
    return m_chaDimLstmCharW;
}

bool Configuration::weUpdate() const
{
    return m_weUpdate;
}


real_t Configuration::weLearningRate() const
{
    return m_weLearningRate;
}

bool Configuration::revertStd() const
{
    return m_revertStd;
}

bool Configuration::outputHtk() const
{
    return m_outputHtk;
}

const real_t& Configuration::highwayGateBias() const
{
    return m_highwayBias;
}

const int& Configuration::outputFromWhichLayer() const
{
    return m_outputTapLayer;
}


const bool& Configuration::outputFromGateLayer() const
{
    return m_outputGateOut;
}

const int& Configuration::outputFrameNum() const
{
    return m_outputFrameNum;
}

const real_t& Configuration::lrDecayRate() const
{
    return m_lr_decay_rate;
}

const std::string& Configuration::mdnFlagPath() const
{
    return m_mdnFlagPath;
}

bool Configuration::mdnFlag() const
{
    return m_mdnFlagPath.length()>0;
}

const std::string& Configuration::mdnDyn() const
{
    return m_mdnDyn;
}

const std::string& Configuration::arOrder() const
{
    return m_arOrder;
}
const std::string& Configuration::arConfig() const
{
    return m_arConfig;
}

const std::string& Configuration::datamvPath() const
{
    return m_datamvPath;
}

const std::string& Configuration::dataTargetPath() const
{
    return m_dataTargetPath;
}

const std::string& Configuration::dataSourcePath() const
{
    return m_dataSourcePath;
}

const real_t& Configuration::mdnPara() const
{
    return m_mdnSamplingPara;
}

const int& Configuration::EMIterNM() const
{
    return m_EMGenIter;
}

const real_t& Configuration::getVarInitPara() const
{
    return m_varInitPara;
}

const real_t& Configuration::getVFloorPara() const
{
    return m_vFloorPara;
}

const real_t& Configuration::getWInitPara() const
{
    return m_wInitPara;
}

const bool& Configuration::getTiedVariance() const
{
    return m_tiedVariance;
}

const std::string& Configuration::printWeightPath() const
{
    return m_printWeightPath;
}

const std::string& Configuration::mdnVarScaleGen() const
{
    return m_mdnVarScaleGen;
}

const std::string& Configuration::tanhAutoregressive() const
{
    return m_tanhAutoregressive;
}

const int& Configuration::zeroFilter() const
{
    return m_setDynFilterZero;
}

const int& Configuration::arrmdnLearning() const
{
    return m_arrmdnLearning;
}

const int& Configuration::weNoiseStartDim() const
{
    return m_weNoiseStartDim;
}

const int& Configuration::weNoiseEndDim() const
{
    return m_weNoiseEndDim;
}

const real_t& Configuration::weNoiseDev() const
{
    return m_weNoiseDev;
}

const real_t& Configuration::arRMDNInitVar() const
{
    return m_ARRMDNInitVar;
}

const int& Configuration::arRMDNUpdateInterval() const
{
    return m_ARRMDNUpdateInterval;
}


const int& Configuration::KLDOutputDataType() const
{
    return m_KLDOutputDataType;
}

const real_t& Configuration::lrFactor() const
{
    return m_lrFactor;
}


const unsigned& Configuration::optimizerOption() const
{
    return m_optimizerOption;
}


const real_t& Configuration::optimizerSecondLR() const
{
    return m_secondLearningRate;
}


const std::string& Configuration::auxillaryDataDir() const
{
    return m_auxDataDir;
}

const std::string& Configuration::auxillaryDataExt() const
{
    return m_auxDataExt;
}
    
const int& Configuration::auxillaryDataTyp() const
{
    return m_auxDataTyp;
}
    
const int& Configuration::auxillaryDataDim() const
{
    return m_auxDataDim;
}

const std::string& Configuration::secondOutputOpt() const
{
    return m_secondOutputOpt;
}

const int& Configuration::printWeightOpt() const
{
    return m_printWeightOpt;
}

const real_t& Configuration::lstmForgetIni() const
{
    return m_lstmForgetIni;
}

const int& Configuration::probDataDim() const
{
    return m_probDataDim;
}

const std::string& Configuration::probDataDir() const
{
    return m_probDataDir;
}

const int& Configuration::aggregateOpt() const
{
    return m_flagAggregateSyn;
}

const int& Configuration::weightMaskOpt() const
{
    return m_weightMaskOpt;
}

const int& Configuration::scheduleSampOpt() const
{
    return m_scheduleSampOpt;
}

const std::string& Configuration::scheduleSampOpts() const
{
    return m_scheduleSampOpts;
}

const int& Configuration::scheduleSampPara() const
{
    return m_scheduleSampPara;
}

const int& Configuration::dropoutbeforeVAE() const
{
    return m_dropoutbeforeVAE;
}

const real_t& Configuration::mdnUVSigThreshold() const
{
    return m_mdnUVSigThreshold;
}


const int& Configuration::mdnSoftMaxGenMethod() const
{
    return m_mdnSoftMaxGenMethod;
}

const std::string& Configuration::mdnSoftMaxGenMethods() const
{
    return m_mdnSoftMaxGenMethods;
}


const std::string& Configuration::exInputDir() const
{
    return m_exInputDir;
}
const std::string& Configuration::exInputExt() const
{
    return m_exInputExt;
}
const int& Configuration::exInputDim() const
{
    return m_exInputDim;
}

const std::string& Configuration::exInputDirs() const
{
    return m_exInputDirs;
}
const std::string& Configuration::exInputExts() const
{
    return m_exInputExts;
}
const std::string& Configuration::exInputDims() const
{
    return m_exInputDims;
}

const std::string& Configuration::exOutputDirs() const
{
    return m_exOutputDirs;
}
const std::string& Configuration::exOutputExts() const
{
    return m_exOutputExts;
}
const std::string& Configuration::exOutputDims() const
{
    return m_exOutputDims;
}


const int& Configuration::verboseLevel() const
{
    return m_verbose;
}

const int& Configuration::fakeEpochNum() const
{
    return m_fakeEpochNum;
}


const int& Configuration::runningMode()  const
{
    return m_runningMode;
}

const int& Configuration::mdnVarUpdateEpoch() const
{
    return m_mdnVarFixEpochNum;
}

const int& Configuration::vaePlotManifold() const
{
    return m_vaePlotManifold;
}

const std::string& Configuration::resolutions() const
{
    return m_resolutions;
}

const int& Configuration::vaeEncoderOutputLayer() const
{
    return m_vaeEncoderOutputLayer;
}


const std::string& Configuration::vaeCodeInputDir() const
{
    return m_vaeCodeInputDir;
}

const int& Configuration::vaeGetVQIndex() const
{
    return m_vaeGetVqIndex;
}

const int& Configuration::vaePutVQIndex() const
{
    return m_vaePutVqIndex;
}

const int& Configuration::exInputType() const
{
    return m_exInputType;
}
const int& Configuration::exOutputType() const
{
    return m_exOutputType;
}
const int& Configuration::exInputReso() const
{
    return m_exInputReso;
}
const int& Configuration::exOutputReso() const
{
    return m_exOutputReso;
}
const real_t& Configuration::softmaxTemperature() const
{
    return m_softmaxTemperature;
}

const int& Configuration::seqWeightsFlag() const
{
    return m_seqWeightsFlag;
}

const int& Configuration::waveNetMemSaveFlag() const{
    return m_waveNetSaveMemFlag;
}

const int& Configuration::dataCheckerIdx() const{
    return m_dataCheckerIdx;
}

const int& Configuration::batchnorm_genmode() const{
    return m_batchnorm_genmode;
}

const real_t& Configuration::f0dataMean_signalgen() const{
    return m_f0dataMean_signalgen;
}

const real_t& Configuration::f0dataStd_signalgen() const{
    return m_f0dataStd_signalgen;
}

const int& Configuration::ARmodelSpecialGenMode() const{
    return m_ARmodelSpecialGenMode;
}

const int& Configuration::quickTestNetwork() const{
    return m_quickTestNetwork;
}


const std::string& Configuration::fileOrderedLstTrn() const{
    return m_fileOrderedLstTrn;
}
const std::string& Configuration::fileOrderedLstVal() const{
    return m_fileOrderedLstVal;
}
const std::string& Configuration::fileOrderedLstTst() const{
    return m_fileOrderedLstTst;
}

unsigned Configuration::truncateSeqNSegments() const{
    return m_truncNsegments;
}
