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

#include "../../currennt_lib/src/Configuration.hpp"
#include "../../currennt_lib/src/NeuralNetwork.hpp"
#include "../../currennt_lib/src/layers/LstmLayer.hpp"
#include "../../currennt_lib/src/layers/MDNLayer.hpp"
#include "../../currennt_lib/src/layers/BinaryClassificationLayer.hpp"
#include "../../currennt_lib/src/layers/MulticlassClassificationLayer.hpp"
#include "../../currennt_lib/src/optimizers/SteepestDescentOptimizer.hpp"
#include "../../currennt_lib/src/helpers/JsonClasses.hpp"
#include "../../currennt_lib/src/rapidjson/prettywriter.h"
#include "../../currennt_lib/src/rapidjson/filestream.h"
#include "../../currennt_lib/src/helpers/misFuncs.hpp"

#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/date_time/posix_time/posix_time_duration.hpp>
#include <boost/filesystem.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
#include <boost/thread.hpp>
#include <boost/algorithm/string/replace.hpp>

#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <stdarg.h>
#include <sstream>
#include <cstdlib>
#include <iomanip>

/*  -------- Helper functions ----------- */

void swap32 (uint32_t *p)
{
  uint8_t temp, *q;
  q = (uint8_t*) p;
  temp = *q; *q = *( q + 3 ); *( q + 3 ) = temp;
  temp = *( q + 1 ); *( q + 1 ) = *( q + 2 ); *( q + 2 ) = temp;
}

void swap16 (uint16_t *p) 
{
  uint8_t temp, *q;
  q = (uint8_t*) p;
  temp = *q; *q = *( q + 1 ); *( q + 1 ) = temp;
}

void swapFloat(float *p)
{
  uint8_t temp, *q;
  q = (uint8_t*) p;
  temp = *q; *q = *( q + 3 ); *( q + 3 ) = temp;
  temp = *( q + 1 ); *( q + 1 ) = *( q + 2 ); *( q + 2 ) = temp;
}


/*  -------- Helper functions definitions ----------- */
enum data_set_type
{
    DATA_SET_TRAINING,
    DATA_SET_VALIDATION,
    DATA_SET_TEST,
    DATA_SET_FEEDFORWARD
};

// helper functions (implementation below)
void readJsonFile(rapidjson::Document *doc, const std::string &filename);
boost::shared_ptr<data_sets::DataSet> loadDataSet(data_set_type dsType);

template <typename TDevice> void printLayers(const NeuralNetwork<TDevice> &nn);
template <typename TDevice> void printOptimizer(const optimizers::Optimizer<TDevice> &optimizer);
template <typename TDevice> void saveNetwork(const NeuralNetwork<TDevice> &nn, 
					     const std::string &filename, const real_t nnlr, 
					     const real_t welr);
void createModifiedTrainingSet(data_sets::DataSet *trainingSet, int parallelSequences, 
			       bool outputsToClasses, boost::mutex &swapTrainingSetsMutex);

template <typename TDevice> void saveState(const NeuralNetwork<TDevice> &nn, 
					   const optimizers::Optimizer<TDevice> &optimizer, 
					   const std::string &infoRows, const real_t nnlr, 
					   const real_t welr);

template <typename TDevice> void restoreState(NeuralNetwork<TDevice> *nn, 
					      optimizers::Optimizer<TDevice> *optimizer, 
					      std::string *infoRows);

std::string printfRow(const char *format, ...);


/*  -------- Network helper functions definitions ----------- */
real_t getGenerationOpt(){

    // generationOpt:
    //      if mdnVarScale is specified 
    //          if config.mdnPara is -1, 
    //               this is MDN parameter generation with mdnVarScale specified
    //          else
    //               this is sampling, scaled by mdnVarScake
    //      else
    //          directly use the mdnPara()
    Configuration config = Configuration::instance();
    return ((config.mdnVarScaleGen().size()>0) ? 
	    ((config.mdnPara() > -1.5) ? config.mdnPara() : 1 ) : 
	    (config.mdnPara()));
}


/*  -------- main function definition ----------- */

template <typename TDevice>
int trainerMain(const Configuration &config)
{
    try {

	/*********** Initialize the network and data set *****************/
	
        // read the neural network description file 
        std::string networkFile = (config.continueFile().empty() ? 
				   config.networkFile() : config.continueFile());
        printf("Reading network from '%s'... ", networkFile.c_str());
        fflush(stdout);
	

	// netDoc: the normal pointer to network.jsn
	// netDocParameter: pointer to network parameter (trained_network.jsn or .autosave)
	// rapidjson::Document *netDocPtr(0); No Need to use netDocPtr
	rapidjson::Document netDocParameter;
        rapidjson::Document netDoc;

        readJsonFile(&netDoc, networkFile);
        printf("done.\n");
        printf("\n");
	
        // load data sets
        boost::shared_ptr<data_sets::DataSet> trainingSet    = 
	    boost::make_shared<data_sets::DataSet>();
        boost::shared_ptr<data_sets::DataSet> validationSet  = 
	    boost::make_shared<data_sets::DataSet>();
        boost::shared_ptr<data_sets::DataSet> testSet        = 
	    boost::make_shared<data_sets::DataSet>();
        boost::shared_ptr<data_sets::DataSet> feedForwardSet = 
	    boost::make_shared<data_sets::DataSet>();

        if (config.trainingMode()) {
            trainingSet = loadDataSet(DATA_SET_TRAINING);
            
            if (!config.validationFiles().empty())
                validationSet = loadDataSet(DATA_SET_VALIDATION);
            
            if (!config.testFiles().empty())
                testSet = loadDataSet(DATA_SET_TEST);
        }else if(config.printWeightPath().size()>0){
	    
        }else {
            feedForwardSet = loadDataSet(DATA_SET_FEEDFORWARD);
        }

        // calculate the maximum sequence length
        int maxSeqLength;
        if (config.trainingMode())
            maxSeqLength = std::max(trainingSet->maxSeqLength(), 
				    std::max(validationSet->maxSeqLength(),
					     testSet->maxSeqLength()));
	else if(config.printWeightPath().size()>0)
	    maxSeqLength = 0;
        else
            maxSeqLength = feedForwardSet->maxSeqLength();

        int parallelSequences = config.parallelSequences();
	

        // create the neural network
        printf("Creating the neural network...");
        fflush(stdout);
        int inputSize  = -1;
        int outputSize = -1;
        inputSize  = trainingSet->inputPatternSize();
        outputSize = trainingSet->outputPatternSize();

	/* Move 17-03-26: inside neuralNetwork */
	/* Add 16-02-22 Wang: for WE updating */
	// NeuralNetwork<TDevice> neuralNetwork(netDoc, parallelSequences, 
	//    maxSeqLength, inputSize, outputSize);
	//
	//if (config.weUpdate() && config.trainingMode()){
	//    inputSize = inputSize - 1 + config.weDim(); // change input size 
	//}

	// Re-Modify 03-02
	/* Don't need this anymore. 
	   Just directly use --network trained_network.jsn or .autosave
	if (!config.trainedParameterPath().empty()){
	    // just read in the network parameter (not any parameter else)
	    // this option can only be used through --trainedModel
	    readJsonFile(&netDocParameter, config.trainedParameterPath());
	    netDocPtr = &netDocParameter;
	}else{
	    netDocPtr = &netDoc;
	}
	*/
	
	NeuralNetwork<TDevice> neuralNetwork(netDoc, parallelSequences, maxSeqLength,
					     inputSize, outputSize);

	// Check the network configuration
        /*if (!trainingSet->empty() && trainingSet->outputPatternSize() != 
	    neuralNetwork.postOutputLayer().size())
            throw std::runtime_error(
		"Post output layer size != target size(training set)");
        if (!validationSet->empty() && validationSet->outputPatternSize() != 
	    neuralNetwork.postOutputLayer().size())
            throw std::runtime_error(
		"Post output layer size != target size(validation set)");
        if (!testSet->empty() && testSet->outputPatternSize() != 
	    neuralNetwork.postOutputLayer().size())
            throw std::runtime_error(
	    "Post output layer size != target size(test set)");*/

	printf("\nNetwork construction done.\n\n");
        printf("Network summary:\n");
        printLayers(neuralNetwork);
        printf("\n");

	/* Add 16-02-22 Wang: for WE updating */
	if (config.weUpdate()){
	    // Load the vector bank
	    neuralNetwork.initWeUpdate(config.weBankPath(), config.weDim(),
				       config.weIDDim(), maxSeqLength * parallelSequences);
	    
	    // Initialize the noise to be added for WE (optional)
	    if (!neuralNetwork.initWeNoiseOpt(config.weNoiseStartDim(),
					      config.weNoiseEndDim(),
					      config.weNoiseDev())){
		throw std::runtime_error("Error in configuration of weNoise");
	    }
	}
	
	/* Add 16-04-01 Wang: for MSE weight */
	if (config.mseWeightPath().size()>0)
	    neuralNetwork.initMseWeight(config.mseWeightPath());
	
	
	/* Add 0413 Wang: for weight mask */
	if (config.weightMaskPath().size()>0)
	    neuralNetwork.initWeightMask(config.weightMaskPath(),
					 config.weightMaskOpt());
	
	
	/* Add 0514 Wang: read data mean and variance (MV), initialize MDN */
	// step1: read MV if provided (optional)
	boost::shared_ptr<data_sets::DataSetMV> dataMV =
	    boost::make_shared<data_sets::DataSetMV>();
	if (config.datamvPath().size()>0){
	    printf("Read in data mean var %s", config.datamvPath().c_str());
	    dataMV =
		boost::make_shared<data_sets::DataSetMV>(config.datamvPath());
	    neuralNetwork.readMVForOutput(*dataMV);
	}
	
	// step2: initialize for MDN 
	// As data has been normalized, no need to read MV for MDN 
	if (config.trainingMode() && config.continueFile().empty())
	    neuralNetwork.initOutputForMDN(netDoc, *dataMV);
	// Note: config.continueFile().empty() make sure it is the first epoch


        // check if this is a classification task
        bool classificationTask = false;
        if (dynamic_cast<layers::BinaryClassificationLayer<TDevice>*>(
	     &neuralNetwork.postOutputLayer()) ||
	    dynamic_cast<layers::MulticlassClassificationLayer<TDevice>*>(
	     &neuralNetwork.postOutputLayer())) {
                classificationTask = true;
        }
        printf("\n");


	/********************* Train the network *************************/
        // Training Mode: 
        if (config.trainingMode()) {
            printf("Creating the optimizer... ");
            fflush(stdout);
            boost::scoped_ptr<optimizers::Optimizer<TDevice> > optimizer;
            optimizers::SteepestDescentOptimizer<TDevice> *sdo;

            switch (config.optimizer()) {
            case Configuration::OPTIMIZER_STEEPESTDESCENT:
                sdo = new optimizers::SteepestDescentOptimizer<TDevice>(
                    neuralNetwork, *trainingSet, *validationSet, *testSet,
                    config.maxEpochs(),
		    config.maxEpochsNoBest(), 
		    config.validateEvery(),
		    config.testEvery(),
                    config.learningRate(),
		    config.momentum(),
		    config.weLearningRate(),
		    config.optimizerOption(),
		    config.lrDecayRate()
                    );
                optimizer.reset(sdo);
                break;

            default:
                throw std::runtime_error("Unknown optimizer type");
            }

            //printf("done.\n");
            printOptimizer(config, *optimizer);
            std::string infoRows;

            
            if (!config.continueFile().empty()) {
		// continue from autosave?
                printf("Restoring state from '%s'... ", config.continueFile().c_str());
                fflush(stdout);
                restoreState(&neuralNetwork, &*optimizer, &infoRows);
                printf("done.\n\n");

		if (!config.trainedParameterPath().empty()){
		    printf("\ncurrennt --continue *.autosave uses model parameters in *.autosave");
		    printf("\n --trainedModel %s ignored", config.trainedParameterPath().c_str());
		}
            }else if (!config.trainedParameterPath().empty()){
		printf("Read network weights from %s\n", config.trainedParameterPath().c_str());
		std::vector<std::string> tmp_preTrained;
		misFuncs::ParseStrOpt(config.trainedParameterPath(), tmp_preTrained, ",");
		for (int cnt = 0 ; cnt < tmp_preTrained.size(); cnt++) {
		    readJsonFile(&netDocParameter, tmp_preTrained[cnt]);
		    neuralNetwork.importWeights(netDocParameter, config.trainedParameterCtr());
		}
	    }else{
		// training and initializing from scratch
		printf("Initializing and training the model from scratch\n");
	    }

	    
            printf("Starting training...");
	    printf("\nPrint error per sequence / per timestep / secondary error (optional)");
            printf("\n");
	    printf(" Epoch | Duration |           Training error         |");
	    printf("           Validation error       |");
	    printf("           Test error             |");
	    printf("New best \n");
            printf("-------+----------+----------------------------------+");
	    printf("----------------------------------+");
	    printf("----------------------------------+");
	    printf("---------\n");
	    std::cout << infoRows;

	    
	    // tranining loop
            bool finished   = false;	    
            while (!finished) {
		
                const char *errFormat = (classificationTask ? 
					 "%6.2lf%%%10.3lf |" : "%12.3lf / %9.3lf/ %8.3f|");
                const char *errSpace  = "                                  |";

                // train for one epoch and measure the time
                infoRows += printfRow(" %5d | ", optimizer->currentEpoch() + 1);                
                boost::posix_time::ptime startTime=boost::posix_time::microsec_clock::local_time();

		// train
                finished = optimizer->train();
		
		// Add 0511: if optimizer is blowed, decrease the learning_rate and start again
		if (optimizer->blowed()) {continue;}
		
                boost::posix_time::ptime endTime = boost::posix_time::microsec_clock::local_time();
                double duration = (double)(endTime - startTime).total_milliseconds() / 1000.0;
                infoRows += printfRow("%8.1lf |", duration);

		
		// print errors
                if (classificationTask)
                    infoRows += printfRow(errFormat, 
					  (double)optimizer->curTrainingClassError()*100.0, 
					  (double)optimizer->curTrainingError(),
					  (double)optimizer->curTrainingErrorSec());
                else
                    infoRows += printfRow(errFormat, 
					  (double)optimizer->curTrainingError(),
					  (double)optimizer->curTrainingErrorPerFrame(),
					  (double)optimizer->curTrainingErrorSec());
                
                if (!validationSet->empty() && 
		    optimizer->currentEpoch() % config.validateEvery() == 0) {
                    if (classificationTask)
                        infoRows += printfRow(errFormat, 
					      (double)optimizer->curValidationClassError()*100.0, 
					      (double)optimizer->curValidationError(),
					      (double)optimizer->curValidationErrorSec());
                    else
                        infoRows += printfRow(errFormat, 
					      (double)optimizer->curValidationError(),
					      (double)optimizer->curValidationErrorPerFrame(),
					      (double)optimizer->curValidationErrorSec());
                }
                else
                    infoRows += printfRow("%s", errSpace);

                if (!testSet->empty() && optimizer->currentEpoch() %
		    config.testEvery() == 0) {
                    if (classificationTask)
                        infoRows += printfRow(errFormat, 
					      (double)optimizer->curTestClassError()*100.0, 
					      (double)optimizer->curTestError(),
					      (double)optimizer->curTestErrorSec());
                    else
                        infoRows += printfRow(errFormat, 
					      (double)optimizer->curTestError(),
					      (double)optimizer->curTestErrorPerFrame(),
					      (double)optimizer->curTestErrorSec());
                }
                else
                    infoRows += printfRow("%s", errSpace);
		
		// check whether to terminate training
                if (!validationSet->empty()&&
		    optimizer->currentEpoch()%config.validateEvery()==0){
		    
                    if (optimizer->epochsSinceLowestValidationError() == 0) {
                        infoRows += printfRow("  yes %s\n",
					      optimizer->optStatus().c_str());
                        if (config.autosaveBest()) {
                            std::stringstream saveFileS;
			    
                            if (config.autosavePrefix().empty()) {
                                size_t pos = config.networkFile().find_last_of('.');
                                if (pos != std::string::npos && pos > 0)
                                    saveFileS << config.networkFile().substr(0, pos);
                                else
                                    saveFileS << config.networkFile();
                            }else{
                                saveFileS << config.autosavePrefix();
			    }
			    
                            saveFileS << ".best.jsn";
                            saveNetwork(neuralNetwork,
					saveFileS.str(), 
					config.learningRate(),
					config.weLearningRate());
                        }
                    }else{
			infoRows += printfRow("  no  %s\n", optimizer->optStatus().c_str());
		    }
		    
                }else{
                    infoRows += printfRow("        \n");
		}
		
                // autosave
		//  When finished, the last epoch***.autosave will load
		//  the weights of best_epoch rather than the weights of the last epoch.
		//  To avoid confusion, the last epoch*** will not be saved here
                if (config.autosave() && (!finished)){
                    saveState(neuralNetwork,
			      *optimizer,
			      infoRows, 
			      config.learningRate(), 
			      config.weLearningRate());
		}
            }
	    

	    // Finish training
            printf("\n");

            if (optimizer->epochsSinceLowestValidationError() == config.maxEpochsNoBest())
                printf("No new lowest error since %d epochs. Training stopped.\n", 
		       config.maxEpochsNoBest());
            else
                printf("Maximum number of training epochs reached. Training stopped.\n");

            if (!validationSet->empty())
                printf("Lowest validation error: %lf\n", optimizer->lowestValidationError());
            else
                printf("Final training set error: %lf\n", optimizer->curTrainingError());
            printf("\n");

            // save the trained network to the output file
            printf("Storing the trained network in '%s'... ",
		   config.trainedNetworkFile().c_str());
            saveNetwork(neuralNetwork,
			config.trainedNetworkFile(), 
			config.learningRate(),
			config.weLearningRate());
            printf("done.\n");

            std::cout << "Removing cache file(s) ..." << std::endl;
            if (trainingSet != boost::shared_ptr<data_sets::DataSet>())
                boost::filesystem::remove(trainingSet->cacheFileName());
            if (validationSet != boost::shared_ptr<data_sets::DataSet>())
                boost::filesystem::remove(validationSet->cacheFileName());
            if (testSet != boost::shared_ptr<data_sets::DataSet>())
                boost::filesystem::remove(testSet->cacheFileName());

	/********************* Convert the Json network file  *************************/
        }else if(config.printWeightPath().size()>0){
	    // print to binary format (0 for normal binary format, 1 for HTS-engine format)
	    if (config.printWeightOpt() < 2){
		printf("Print binary weight file in '%s' with option %d\n",
		       config.printWeightPath().c_str(),
		       config.printWeightOpt());
		neuralNetwork.printWeightMatrix(config.printWeightPath(),
						config.printWeightOpt());
		printf("done.\n");
		
	    // printconvert .autosave to HTS-engine format.trained_network
	    }else if (config.printWeightOpt() == 2){
		// save the trained network to the output file
		printf("Translate the network in '%s'... ",
		       config.printWeightPath().c_str());
		saveNetwork(neuralNetwork,
			    config.printWeightPath(),//config.trainedNetworkFile(), 
			    config.learningRate(),
			    config.weLearningRate());
		printf("done.\n");
	    }else if (config.printWeightOpt() == 3){
		// save network after loading weights from another network
		if (config.trainedParameterPath().size() == 0)
		    throw std::runtime_error("Please provide --trainedModel");
		
		readJsonFile(&netDocParameter,
			     config.trainedParameterPath());
		
		neuralNetwork.importWeights(netDocParameter,
					    config.trainedParameterCtr());

		printf("Save the network in '%s'... ",
		       config.trainedNetworkFile().c_str());
		saveNetwork(neuralNetwork,
			    config.printWeightPath(),// config.trainedNetworkFile(), 
			    config.learningRate(),   
			    config.weLearningRate());
		printf("done.\n");
	    }
	    
	/********************* Data Generation    *************************/
        }else {

	    real_t generationOpt     = getGenerationOpt();
	    int    outputPatternSize = neuralNetwork.outputPatternSize(generationOpt);
	    bool   unstandardize     = config.revertStd();
	    bool   htkoutput         = config.outputHtk();
	    int    outputlayerID     = config.outputFromWhichLayer();
	    
	    // Load data mean and std from data set
	    // Method 1:
            Cpu::real_vector outputMeans  = feedForwardSet->outputMeans();
            Cpu::real_vector outputStdevs = feedForwardSet->outputStdevs();

	    // Method 2:
	    //  Or load the mean and variation from 
	    if (config.datamvPath().size()>0){
		if (dataMV == NULL)
		    throw std::runtime_error("Can't read datamv");
		if (dataMV->outputM().size() != outputMeans.size())
		    throw std::runtime_error("output dimension mismatch datamv");
		outputMeans  = dataMV->outputM();
		outputStdevs = dataMV->outputV();
		/*for (int y = 0; y < outputMeans.size(); y++){
		    outputMeans[y]  = dataMV->outputM()[y];
		    outputStdevs[y] = dataMV->outputV()[y];
		    } */
		printf("Mean and var are over-written by %s.\n",
		       config.datamvPath().c_str());
	    }

	    // Method 3:
	    if (neuralNetwork.externalOutputMV(outputMeans, outputStdevs)){
		printf("Mean and var loaded from externalDataMV\n");
	    }

	    // Judge whether de-normalization can be launched
	    if (unstandardize && (!config.outputFromGateLayer()) &&
		outputMeans.size() == outputStdevs.size()        &&
		outputMeans.size() == outputPatternSize){

		if (neuralNetwork.isMDNLayer(outputlayerID)){
		    // for MDNLayer output
		    if (config.mdnPara() < -1.0 || config.mdnPara() >= 0.0){
			Cpu::real_vector mdnConfigVec = neuralNetwork.getMdnConfigVec();
			if (!mdnConfigVec.empty()){    
			    // if the unit is sigmoid or softmax, set the mean and std
			    for (int x = 0; x < (mdnConfigVec.size()-1)/5; x++){
				int mdnType  = (int)mdnConfigVec[5+x*5];
				if (mdnType == MDN_TYPE_SIGMOID || mdnType == MDN_TYPE_SOFTMAX){
				    int unitSOut = (int)mdnConfigVec[3+x*5];
				    int unitEOut = (int)mdnConfigVec[4+x*5];
				    for (int y = unitSOut; y < unitEOut; y++){
					outputMeans[y] = 0.0;
					outputStdevs[y] = 1.0;
				    }
				    printf("\n\t de-normalization is skipped for dimension ");
				    printf("from %d to %d\n", unitSOut+1, unitEOut);
				}
			    }
			}
			unstandardize = true;
		    }else{
			unstandardize = false;
		    }
		}
		unstandardize = true;
	    }else{
		unstandardize = false;
	    }
	    
	    
	    printf("Outputs from layer %d", config.outputFromWhichLayer());
	    if (config.outputFromGateLayer()) {printf(", gate output");}
	    if (config.mdnPara()>0 && neuralNetwork.isMDNLayer(outputlayerID))
		printf(", MDN with para=%f",config.mdnPara());
	    
	    if (htkoutput){
		printf(", HTK format (float32, big-endian)");
	    }else{
		printf(", bin format");
	    }
	    
	    if (unstandardize)
		printf(", de-normalized\n");
	    else
		printf(", NOT de-normalized\n");
	    
	    /*
	    // Prepare the mean and variance for data de-normalization
	    if (unstandardize && config.outputFromWhichLayer() < 0 && 
		(config.mdnPara() < -1.0 || config.mdnPara() > 0.0)){

		// when de-normalization is not used ?
		// 1. unstandardize == false
		// 2. generate from hidden layers
		// 3. MDN, output the distribution parameter
	       	// 4. MDN, output is from the sigmoid or softmax unit
		
            }else{
		printf("Outputs will NOT be scaled by mean and std specified in NC file.\n");
		}*/

	    int output_lag = config.outputTimeLag();
            if (config.feedForwardFormat() == Configuration::FORMAT_SINGLE_CSV) {
                // Block 20161111x01
		printf("WARNING: output only for HTK format");
            }else if (config.feedForwardFormat() == Configuration::FORMAT_CSV) {
                // Block 20161111x02
		printf("WARNING: output only for HTK format");
            }else if (config.feedForwardFormat() == Configuration::FORMAT_HTK) {
		
                // process all data set fractions
                int fracIdx = 0;
                boost::shared_ptr<data_sets::DataSetFraction> frac;
		    
                while (((frac = feedForwardSet->getNextFraction()))) {
		    // print fraction information
                    printf("Computing outputs for data fraction %d ... ", ++fracIdx);
		    for (int i = 0; i<frac->numSequences(); i++)
			printf("%s ", frac->seqInfo(i).seqTag.c_str());
                    fflush(stdout);
		    
		    
		    boost::posix_time::ptime sTime=boost::posix_time::microsec_clock::local_time();
		    neuralNetwork.notifyCurrentEpoch(config.fakeEpochNum());
		    neuralNetwork.updateNNStateForGeneration();
		    //neuralNetwork.loadSequences(*frac);
                    neuralNetwork.computeForwardPassGen(*frac, frac->maxSeqLength(), generationOpt);
		    boost::posix_time::ptime eTime=boost::posix_time::microsec_clock::local_time();

		    printf("\nTime (s): %f\n", (real_t)(eTime-sTime).total_milliseconds()/1000.0);

                    std::vector<std::vector<std::vector<real_t> > > outputs = 
			neuralNetwork.getOutputs(generationOpt); 

                    // write one output file per sequence
                    for (int psIdx = 0; psIdx < (int)outputs.size(); ++psIdx) {
                        if (outputs[psIdx].size() > 0) {
                            // replace_extension does not work in all Boost versions ...
                            //std::string seqTag = frac->seqInfo(psIdx).seqTag;
                            /*size_t dot_pos = seqTag.find_last_of('.');
                            if (dot_pos != std::string::npos && dot_pos > 0) {
                                seqTag = seqTag.substr(0, dot_pos);
                            }*/
                            //seqTag += ".htk";
                            //std::cout << seqTag << std::endl;
			    
			    std::string seqTagSuf;
			    if (htkoutput) {seqTagSuf = ".htk";} else{seqTagSuf = ".bin";}
                            boost::filesystem::path seqPath(frac->seqInfo(psIdx).seqTag+seqTagSuf);
                            std::string filename(seqPath.filename().string());
                            boost::filesystem::path oPath = 
				boost::filesystem::path(config.feedForwardOutputFile()) / 
				seqPath.relative_path().parent_path();
                            boost::filesystem::create_directories(oPath);
                            boost::filesystem::path filepath = oPath / filename;
                            std::ofstream file(filepath.string().c_str(), 
					       std::ofstream::out | std::ios::binary);

                            int nComps = outputs[psIdx][0].size();

                            // write header
			    if (htkoutput){
				unsigned tmp = (unsigned)outputs[psIdx].size();
				swap32(&tmp);
				file.write((const char*)&tmp, sizeof(unsigned));
				tmp = (unsigned)(config.featurePeriod() * 1e4);
				swap32(&tmp);
				file.write((const char*)&tmp, sizeof(unsigned));
				unsigned short tmp2 = (unsigned short)(nComps) * sizeof(float);
				swap16(&tmp2);
				file.write((const char*)&tmp2, sizeof(unsigned short));
				tmp2 = (unsigned short)(config.outputFeatureKind());
				swap16(&tmp2);
				file.write((const char*)&tmp2, sizeof(unsigned short));
			    }
			    

                            // write the patterns
                            for (int time=0; time<(int)outputs[psIdx].size(); ++time) 
			    {
                                for (int outIdx=0;outIdx<(int)outputs[psIdx][time].size();++outIdx)
				{
                                    float v;
				    v = (time < outputs[psIdx].size() - output_lag) ? 
					((float)outputs[psIdx][time+output_lag][outIdx]) :
					((float)outputs[psIdx][outputs[psIdx].size()-1][outIdx]);

                                    if (unstandardize) {
                                        v *= outputStdevs[outIdx];
                                        v += outputMeans[outIdx];
                                    }
				    
				    if (htkoutput)
					swapFloat(&v); 
				    
                                    file.write((const char*)&v, sizeof(float));
                                }
                            }
                            file.close();
                        }
                    }
                    printf(" done.\n");
                }
            }

	    printf("Outputs from layer %d", config.outputFromWhichLayer());
	    if (config.outputFromGateLayer()) {printf(", gate output");}
	    if (config.mdnPara()>0 && neuralNetwork.isMDNLayer(outputlayerID))
		printf(", MDN with para=%f",config.mdnPara());
	    /*
	    if (htkoutput){
		printf(", HTK format (float32, big-endian)");
	    }else{
		printf(", binary format");
	    }
	    
	    if (unstandardize && feedForwardSet->outputMVFlag())
		printf(", de-normalized\n");
	    else
		printf(", NOT de-normalized\n");
	    */
	    
            if (feedForwardSet != boost::shared_ptr<data_sets::DataSet>()) 
                std::cout << "Removing cache file: "<<feedForwardSet->cacheFileName()<<std::endl;
            boost::filesystem::remove(feedForwardSet->cacheFileName());
        } // evaluation mode
    }
    catch (const std::exception &e) {
        printf("FAILED: %s\n", e.what());
        return 2;
    }

    return 0;
}


int main(int argc, const char *argv[])
{
    // load the configuration
    Configuration config(argc, argv);

    // run the execution device specific main function
    if (config.useCuda()) {
        int count;
        cudaError_t err;
        if (config.listDevices()) {
            if ((err = cudaGetDeviceCount(&count)) != cudaSuccess) {
                std::cerr << "FAILED: " << cudaGetErrorString(err) << std::endl;
                return err;
            }
            std::cout << count << " devices found" << std::endl;
            cudaDeviceProp prop;
            for (int i = 0; i < count; ++i) {
                if ((err = cudaGetDeviceProperties(&prop, i)) != cudaSuccess) {
                    std::cerr << "FAILED: " << cudaGetErrorString(err) << std::endl;
                    return err;
                }
                std::cout << i << ": " << prop.name << std::endl;
            }
            return 0;
        }
        int device = 0;
        char* dev = std::getenv("CURRENNT_CUDA_DEVICE");
        if (dev != NULL) {
            device = std::atoi(dev);
        }
        cudaDeviceProp prop;
        if ((err = cudaGetDeviceProperties(&prop, device)) != cudaSuccess) {
            std::cerr << "FAILED: " << cudaGetErrorString(err) << std::endl;
            return err;
        }
        std::cout << "Using device #" << device << " (" << prop.name << ")" << std::endl;
        if ((err = cudaSetDevice(device)) != cudaSuccess) {
            std::cerr << "FAILED: " << cudaGetErrorString(err) << std::endl;
            return err;
        }
        return trainerMain<Gpu>(config);
    }
    else
        return trainerMain<Cpu>(config);
}


void readJsonFile(rapidjson::Document *doc, const std::string &filename)
{
    // open the file
    std::ifstream ifs(filename.c_str(), std::ios::binary);
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

    std::string docStr(buffer);
    delete buffer;

    // extract the JSON tree
    if (doc->Parse<0>(docStr.c_str()).HasParseError()){
	printf("\n\t\tPlease check lines around:\n\t\t");
	size_t start = (doc->GetErrorOffset()>20)?(doc->GetErrorOffset()-20):doc->GetErrorOffset();
	for (int t=0;t<50;t++)
	    printf("%c", docStr.c_str()[start + t]);
	printf("\n\n");
        throw std::runtime_error(std::string("Parse error: ") + doc->GetParseError());
    }
}


boost::shared_ptr<data_sets::DataSet> loadDataSet(data_set_type dsType)
{
    std::string type;
    std::vector<std::string> filenames;
    real_t fraction       = 1;
    bool fracShuf         = false;
    bool seqShuf          = false;
    real_t noiseDev       = 0;
    std::string cachePath = "";
    int truncSeqLength    = -1;
    
    std::string auxDataDir   = "";
    std::string auxDataExt   = "";
    int         auxDataDim   = -1;
    int         auxDataTyp   = -1;
    
    Configuration config = Configuration::instance();

    cachePath = Configuration::instance().cachePath();
    switch (dsType) {
    case DATA_SET_TRAINING:
        type           = "training set";
        filenames      = Configuration::instance().trainingFiles();
        fraction       = Configuration::instance().trainingFraction();
        fracShuf       = Configuration::instance().shuffleFractions();
        seqShuf        = Configuration::instance().shuffleSequences();
        noiseDev       = Configuration::instance().inputNoiseSigma();
        truncSeqLength = Configuration::instance().truncateSeqLength();
        break;

    case DATA_SET_VALIDATION:
        type           = "validation set";
        filenames      = Configuration::instance().validationFiles();
        fraction       = Configuration::instance().validationFraction();
        truncSeqLength = Configuration::instance().truncateSeqLength();
        break;

    case DATA_SET_TEST:
        type           = "test set";
        filenames      = Configuration::instance().testFiles();
        fraction       = Configuration::instance().testFraction();
        break;

    default:
        type           = "feed forward input set";
        filenames      = Configuration::instance().feedForwardInputFiles();
        noiseDev       = Configuration::instance().inputNoiseSigma();
        break;
    }

    // Auxillary data information
    auxDataDir         = Configuration::instance().auxillaryDataDir();
    auxDataExt         = Configuration::instance().auxillaryDataExt();
    auxDataDim         = Configuration::instance().auxillaryDataDim();
    auxDataTyp         = Configuration::instance().auxillaryDataTyp();
    
    // Load the data
    printf("Loading %s ", type.c_str());
    for (std::vector<std::string>::const_iterator fn_itr = filenames.begin();
         fn_itr != filenames.end(); ++fn_itr){
        printf("'%s' ", fn_itr->c_str());
    }
    printf("...");
    fflush(stdout);
    //std::cout << "truncating to " << truncSeqLength << std::endl;
    boost::shared_ptr<data_sets::DataSet> ds 
	= boost::make_shared<data_sets::DataSet>(
		filenames,
		Configuration::instance().parallelSequences(), 
		fraction,   truncSeqLength, 
		fracShuf,   seqShuf,
		noiseDev,   cachePath);
    
    printf("done.\n");
    printf("Loaded fraction:  %d%%\n",    (int)(fraction*100));
    printf("Sequences:        %d\n",      ds->totalSequences());
    printf("Sequence lengths: %d..%d\n",  ds->minSeqLength(),
	                                  ds->maxSeqLength());
    printf("Total timesteps:  %lu\n",     ds->totalTimesteps());

    // Note: Auxillary configuration will be used in DataSet.cpp.
    //       They are just shown here
    if (auxDataDir.size()>0){
	printf("Auxillary path:   %s\n", auxDataDir.c_str());
	printf("Auxillary ext :   %s\n", auxDataExt.c_str());
	printf("Auxillary type:   %d\n", auxDataTyp);
	printf("Auxillary dim:    %d\n", auxDataDim);
    }
    printf("\n");

    return ds;
}


template <typename TDevice>
void printLayers(const NeuralNetwork<TDevice> &nn)
{
    int weights = 0;
    printf("     Name\t\tType\n");
    for (int i = 0; i < (int)nn.layers().size(); ++i) {
        printf("(%d) %s\t\t%s ",  i, nn.layers()[i]->name().c_str(),
	       nn.layers()[i]->type().c_str());
        printf("[size: %d", nn.layers()[i]->size());

        const layers::TrainableLayer<TDevice>* tl = 
	    dynamic_cast<const layers::TrainableLayer<TDevice>*>(nn.layers()[i].get());
        if (tl) {
            printf(", bias: %.1lf, weights: %d",
		   (double)tl->bias(), (int)tl->weights().size());
            weights += (int)tl->weights().size();
        }else{
	    const layers::MDNLayer<TDevice>* mdnlayer = 
		dynamic_cast<const layers::MDNLayer<TDevice>*>(nn.layers()[i].get());
	    if (mdnlayer && mdnlayer -> flagTrainable()){
		printf(", weights: %d", (int)mdnlayer->weights().size());
		weights += (int)mdnlayer->weights().size();
	    }
	}
        printf("]\n");
    }

    printf("Total weights: %d\n", weights);
}


template <typename TDevice> 
void printOptimizer(const Configuration &config, const optimizers::Optimizer<TDevice> &optimizer)
{
    if (dynamic_cast<const optimizers::SteepestDescentOptimizer<TDevice>*>(&optimizer)) {
        //printf("Optimizer type: Steepest descent with momentum\n");
        printf("Max training epochs:       %d\n", config.maxEpochs());
        printf("Max epochs until new best: %d\n", config.maxEpochsNoBest());
        printf("Validation error every:    %d\n", config.validateEvery());
        printf("Test error every:          %d\n", config.testEvery());
        printf("Learning rate:             %g\n", (double)config.learningRate());
        printf("Momentum:                  %g\n", (double)config.momentum());
	
	if (config.continueFile().empty() && !config.trainedParameterPath().empty()){
	    printf("Model Parameter:           %s\n", config.trainedParameterPath().c_str());
	}
	if (config.weUpdate()){
	    printf("\nParameter for WE:\n");
	    printf("WE learning_rate:          %g\n", (double)config.weLearningRate());
	    printf("WE Bank:                   %s\n", config.weBankPath().c_str());
	    printf("WE Dim:                    %d\n", config.weDim());
	    printf("WE Start index:            %d\n", config.weIDDim());
	}
        printf("\n");
    }
}


template <typename TDevice> 
void saveNetwork(const NeuralNetwork<TDevice> &nn, const std::string &filename,
		 const real_t nnlr, const real_t welr)
{

    if (nnlr > 0){
	rapidjson::Document jsonDoc;
	jsonDoc.SetObject();
	nn.exportLayers (&jsonDoc);
	nn.exportWeights(&jsonDoc);
	
	FILE *file = fopen(filename.c_str(), "w");
	if (!file)
	    throw std::runtime_error("Cannot open file");

	rapidjson::FileStream os(file);
	rapidjson::PrettyWriter<rapidjson::FileStream> writer(os);
	jsonDoc.Accept(writer);

	fclose(file);
    }

    if (welr > 0){
	/* Add 16-02-22 Wang: for WE updating */
	// save WE
	//autosaveFilename << ".we";
	if (nn.flagInputWeUpdate()){
	    if (!nn.saveWe(filename+".we")){
		throw std::runtime_error("Fail to save we data");
	    }
	}    
    }
}


template <typename TDevice> 
void saveState(const NeuralNetwork<TDevice> &nn, 
	       const optimizers::Optimizer<TDevice> &optimizer, 
	       const std::string &infoRows,
	       const real_t nnlr, const real_t welr)
{

    if (nnlr > 0){
	// create the JSON document
	rapidjson::Document jsonDoc;
	jsonDoc.SetObject();

	// add the configuration options
	jsonDoc.AddMember("configuration", 
			  Configuration::instance().serializedOptions().c_str(), 
			  jsonDoc.GetAllocator());

	// add the info rows
	std::string tmp = boost::replace_all_copy(infoRows, "\n", ";;;");
	jsonDoc.AddMember("info_rows", tmp.c_str(), jsonDoc.GetAllocator());

	// add the network structure and weights
	nn.exportLayers (&jsonDoc);
	nn.exportWeights(&jsonDoc);

	// add the state of the optimizer
	optimizer.exportState(&jsonDoc);
    
	// open the file
	std::stringstream autosaveFilename;
	std::string prefix = Configuration::instance().autosavePrefix(); 
	autosaveFilename << prefix;
	if (!prefix.empty())
	    autosaveFilename << '_';
	autosaveFilename << "epoch";
	autosaveFilename << std::setfill('0') << std::setw(3) << optimizer.currentEpoch();
	autosaveFilename << ".autosave";
	std::string autosaveFilename_str = autosaveFilename.str();
	FILE *file = fopen(autosaveFilename_str.c_str(), "w");
	if (!file)
	    throw std::runtime_error("Cannot open file");
	
	// write the file
	rapidjson::FileStream os(file);
	rapidjson::PrettyWriter<rapidjson::FileStream> writer(os);
	jsonDoc.Accept(writer);
	fclose(file);
    }

    if (welr > 0){
	/* Add 16-02-22 Wang: for WE updating */
	// save WE
	// open the file
	std::stringstream autosaveFilename;
	std::string prefix = Configuration::instance().autosavePrefix(); 
	autosaveFilename << prefix;
	if (!prefix.empty())
	    autosaveFilename << '_';
	autosaveFilename << "epoch";
	autosaveFilename << std::setfill('0') << std::setw(3) << optimizer.currentEpoch();
	autosaveFilename << ".autosave";
	autosaveFilename << ".we";
	if (nn.flagInputWeUpdate()){
	    if (!nn.saveWe(autosaveFilename.str())){
		throw std::runtime_error("Fail to save we data");
	    }
	}
    }
}


template <typename TDevice> 
void restoreState(
 NeuralNetwork<TDevice> *nn,
 optimizers::Optimizer<TDevice> *optimizer, 
 std::string *infoRows)
{
    rapidjson::Document jsonDoc;
    readJsonFile(&jsonDoc, Configuration::instance().continueFile());

    // extract info rows
    if (!jsonDoc.HasMember("info_rows"))
        throw std::runtime_error("Missing value 'info_rows'");
    *infoRows = jsonDoc["info_rows"].GetString();
    boost::replace_all(*infoRows, ";;;", "\n");

    // extract the state of the optimizer
    optimizer->importState(jsonDoc);
}



std::string printfRow(const char *format, ...)
{
    // write to temporary buffer
    char buffer[100];
    va_list args;
    va_start(args, format);
    vsnprintf(buffer, sizeof(buffer), format, args);
    va_end(args);

    // print on stdout
    std::cout << buffer;
    fflush(stdout);

    // return the same string
    return std::string(buffer);
}

