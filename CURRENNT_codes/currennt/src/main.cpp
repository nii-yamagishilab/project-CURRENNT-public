/******************************************************************************
 * This file is modified by
 * Xin WANG
 * National Institute of Informatics, Japan
 * 2016 - 2020
 * 
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
#include "../../currennt_lib/src/optimizers/SteepestDescentOptimizer.hpp"
#include "../../currennt_lib/src/helpers/JsonClasses.hpp"
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


#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <stdarg.h>
#include <sstream>
#include <cstdlib>
#include <iomanip>

/**
 *  -------- Helper functions ----------- 
 */

enum data_set_type
{
    DATA_SET_TRAINING,
    DATA_SET_VALIDATION,
    DATA_SET_TEST,
    DATA_SET_FEEDFORWARD
};

// read json file
void readJsonFile(rapidjson::Document *doc, const std::string &filename);

// warpper to load netCDF data set
boost::shared_ptr<data_sets::DataSet> loadDataSet(data_set_type dsType);

// print optimizer information
void printOptimizer(const Configuration &config);

// wrapper to print one line information
std::string printfRow(const char *format, ...);

// print training error head
void printTrainingErrorHead(std::string &infoRows);

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
	
	/**
	 * Initialize the network and data set
	 *
	 **/
	
        // read the neural network description file 
        std::string networkFile = (config.continueFile().empty() ? 
				   config.networkFile() : config.continueFile());
        printf("Reading network from '%s'... ", networkFile.c_str());
        fflush(stdout);
	

	// netDoc: the normal pointer to network.jsn
	// netDocParameter: pointer to network parameter (*.jsn or *.autosave)
	// rapidjson::Document *netDocPtr(0); No Need to use netDocPtr
	rapidjson::Document netDocParameter;
        rapidjson::Document netDoc;

        readJsonFile(&netDoc, networkFile);
        printf("done.\n\n");

	/**
	 * Load data sets
	 *
	 **/
	
        boost::shared_ptr<data_sets::DataSet> trainingSet    = 
	    boost::make_shared<data_sets::DataSet>();
        boost::shared_ptr<data_sets::DataSet> validationSet  = 
	    boost::make_shared<data_sets::DataSet>();
        boost::shared_ptr<data_sets::DataSet> testSet        = 
	    boost::make_shared<data_sets::DataSet>();
        boost::shared_ptr<data_sets::DataSet> feedForwardSet = 
	    boost::make_shared<data_sets::DataSet>();

	int maxSeqLength;
        if (config.trainingMode()) {
	    // training mode
            trainingSet = loadDataSet(DATA_SET_TRAINING);
            
            if (!config.validationFiles().empty())
                validationSet = loadDataSet(DATA_SET_VALIDATION);
            
            if (!config.testFiles().empty())
                testSet = loadDataSet(DATA_SET_TEST);

	    maxSeqLength = std::max(trainingSet->maxSeqLength(), 
				    std::max(validationSet->maxSeqLength(),
					     testSet->maxSeqLength()));
	    
        }else if(config.printWeightPath().size()  > 0 ||
		 config.networkGraphFile().size() > 0){
	    // network conversion or network plotting mode
	    maxSeqLength = 0;
	    
        }else {
	    // default: forward propagation mode
            feedForwardSet = loadDataSet(DATA_SET_FEEDFORWARD);
	    maxSeqLength = feedForwardSet->maxSeqLength();
        }
            

	/**
	 * Create network and optimizer 
	 *
	 **/

	// input size (not used)
	int inputSize  = trainingSet->inputPatternSize();
	// output size (not used)
        int outputSize = trainingSet->outputPatternSize();
	// number of parallel sequences in one block
        int parallelSequences = config.parallelSequences();
        
	printf("Creating the neural network...");
        fflush(stdout);
	NeuralNetwork<TDevice> neuralNetwork(netDoc, parallelSequences, maxSeqLength,
					     inputSize, outputSize);
	printf("\nNetwork construction done.\n\n\n");
        neuralNetwork.printNetworkSummary();
	
	// B1
        // check if this is a classification task
        bool classificationTask = neuralNetwork.isClassificationNet();
	    

	/**
	 * Run the program
	 *
	 **/
	
        
        if (config.trainingMode()) {

	    /**
	     *  training mode
	     *
	     **/ 

	    // string to log training error
	    std::string infoRows;
	    
	    // create optimizer
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
		printOptimizer(config);
                break;

            default:
                throw std::runtime_error("CURRENNT error: unknown optimizer type");
            }

	    // initialize or load existing parameters
            if (!config.continueFile().empty()) {
		// continue from autosave and restore the training statistics
                printf("Restoring state from '%s'... ", config.continueFile().c_str());
                fflush(stdout);
                optimizer->restoreState(netDoc, infoRows);
                printf("done.\n\n");

		if (!config.trainedParameterPath().empty()){
		    printf("\n --trainedModel %s is ignored when training with --continue\n",
			   config.trainedParameterPath().c_str());
		}
		
            }else if (!config.trainedParameterPath().empty()){
		// selectively load parameter from --trainedModel
		printf("Read weights from %s\n", config.trainedParameterPath().c_str());
		std::vector<std::string> tmp_preTrained;
		misFuncs::ParseStrOpt(config.trainedParameterPath(),
				      tmp_preTrained, ",");

		// we can read multiple files
		for (int cnt = 0 ; cnt < tmp_preTrained.size(); cnt++) {
		    readJsonFile(&netDocParameter, tmp_preTrained[cnt]);
		    neuralNetwork.importWeights(
			netDocParameter, config.trainedParameterCtr());
		}
		
	    }else{
		// training and initializing from scratch
		printf("Initializing and training the model from scratch\n");
	    }

	    // tranining loop
            printTrainingErrorHead(infoRows);
	    bool finished   = false;	    
            while (!finished) {
		
                const char *errFormat =
		    (classificationTask ? 
		     "%6.2lf%%%10.3lf |" : "%12.3lf / %9.3lf/ %8.3f|");
                const char *errSpace  = "                                  |";

                // train for one epoch and measure the time
                infoRows += printfRow(" %5d | ", optimizer->currentEpoch() + 1);
		
                boost::posix_time::ptime startTime =
		    boost::posix_time::microsec_clock::local_time();

		// train
                finished = optimizer->train();
		
		// if optimizer is blowed, decrease learning_rate and start again
		if (optimizer->blowed()) {continue;}
		
                boost::posix_time::ptime endTime =
		    boost::posix_time::microsec_clock::local_time();
                double duration =
		    (double)(endTime - startTime).total_milliseconds() / 1000.0;

		// training time for one epoch
                infoRows += printfRow("%8.1lf |", duration);

		
		// print errors
                if (classificationTask){
                    infoRows +=
			printfRow(errFormat, 
				  (double)optimizer->curTrainingClassError()*100.0, 
				  (double)optimizer->curTrainingError(),
				  (double)optimizer->curTrainingErrorSec());
                }else{
                    infoRows +=
			printfRow(errFormat, 
				  (double)optimizer->curTrainingError(),
				  (double)optimizer->curTrainingErrorPerFrame(),
				  (double)optimizer->curTrainingErrorSec());
                }
		
                if (!validationSet->empty() && 
		    optimizer->currentEpoch() % config.validateEvery() == 0) {
                    if (classificationTask){
                        infoRows +=
			  printfRow(errFormat, 
				    (double)optimizer->curValidationClassError()*100.0, 
				    (double)optimizer->curValidationError(),
				    (double)optimizer->curValidationErrorSec());
                    }else{
                        infoRows +=
			  printfRow(errFormat, 
				    (double)optimizer->curValidationError(),
				    (double)optimizer->curValidationErrorPerFrame(),
				    (double)optimizer->curValidationErrorSec());
		    }
                }else{
                    infoRows += printfRow("%s", errSpace);
		}

                if (!testSet->empty() &&
		    optimizer->currentEpoch() % config.testEvery() == 0) {
                    if (classificationTask){
                        infoRows +=
			    printfRow(errFormat, 
				      (double)optimizer->curTestClassError()*100.0, 
				      (double)optimizer->curTestError(),
				      (double)optimizer->curTestErrorSec());
		    }else{
                        infoRows +=
			    printfRow(errFormat, 
				      (double)optimizer->curTestError(),
				      (double)optimizer->curTestErrorPerFrame(),
				      (double)optimizer->curTestErrorSec());
		    }
                }else{
                    infoRows += printfRow("%s", errSpace);
		}

		
		// check whether to terminate training
                if (!validationSet->empty()&&
		    optimizer->currentEpoch()%config.validateEvery()==0){
		    
                    if (optimizer->epochsSinceLowestValidationError() == 0) {
			
                        infoRows +=
			    printfRow("  yes %s\n", optimizer->optStatus().c_str());
			
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
			    neuralNetwork.saveNetwork(saveFileS.str(), 
						      config.learningRate(),
						      config.weLearningRate());
                        }
                    }else{
			infoRows +=
			    printfRow("  no  %s\n", optimizer->optStatus().c_str());
		    }
		    
                }else{
                    infoRows += printfRow("        \n");
		}
		
                if (config.autosave() && (!finished)){
                    optimizer->saveState(neuralNetwork,
					 infoRows, 
					 config.learningRate(), 
					 config.weLearningRate());
		}
		
            }
	    

	    // Finish training
            printf("\n");

            if (optimizer->epochsSinceLowestValidationError() ==
		config.maxEpochsNoBest())
                printf("No new lowest error since %d epochs. Training stopped.\n", 
		       config.maxEpochsNoBest());
            else
                printf("Maximum training epochs reached. Training stopped.\n");

            if (!validationSet->empty())
                printf("Lowest validation error: %lf\n",
		       optimizer->lowestValidationError());
            else
                printf("Final training set error: %lf\n",
		       optimizer->curTrainingError());
            printf("\n");

            // save the trained network to the output file
            printf("Storing the trained network in '%s'... ",
		   config.trainedNetworkFile().c_str());
            neuralNetwork.saveNetwork(config.trainedNetworkFile(), 
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


        }else if(config.printWeightPath().size()>0){
	    
	    /**
	     * 	Convert the Json network file
	     *
	     **/
	    
	    if (config.printWeightOpt() < 2){
		// print to binary format
		//  0 for normal binary format
		//  1 for HTS-engine format
		printf("Print binary weight file in '%s' with option %d\n",
		       config.printWeightPath().c_str(),
		       config.printWeightOpt());
		
		neuralNetwork.printWeightMatrix(config.printWeightPath(),
						config.printWeightOpt());
		printf("done.\n");
		
	    }else if (config.printWeightOpt() == 2){
		// save the trained network to the output file
		printf("Translate the network in '%s'... ",
		       config.printWeightPath().c_str());
		neuralNetwork.saveNetwork(config.printWeightPath(),
					  config.learningRate(),
					  config.weLearningRate());
		printf("done.\n");
		
	    }else if (config.printWeightOpt() == 3){
		// save network after loading weights from another network
		if (config.trainedParameterPath().size() == 0)
		    throw std::runtime_error("Config error: please give --trainedModel");
		
		readJsonFile(&netDocParameter,
			     config.trainedParameterPath());
		
		neuralNetwork.importWeights(netDocParameter,
					    config.trainedParameterCtr());

		printf("Save the network in '%s'... ",
		       config.printWeightPath().c_str());
		neuralNetwork.saveNetwork(config.printWeightPath(),
					  config.learningRate(),   
					  config.weLearningRate());
		printf("done.\n");
	    }

	
	}else if (config.networkGraphFile().size() > 0){
	    /**
	     * Network dot-plot file
	     * 
	     **/
	    // the dot plot should have been plotted during network initialization
	    // nothing to do here
	    printf("Generating dot graph file to %s.\n",
		   config.networkGraphFile().c_str());
	    printf("Please use dot to convert the graph file into a picture\n");
	    

        }else {

	    /**
	     * Generation 
	     **/
	    
	    // Get the configuration parameters
	    real_t generationOpt     = getGenerationOpt();
	    int    outputDim         = neuralNetwork.outputDimension(generationOpt);
	    bool   unstandardize     = config.revertStd();
	    bool   htkoutput         = config.outputHtk();
	    int    outputlayerID     = config.outputFromWhichLayer();
	    int    output_lag        = config.outputTimeLag();

	    if (config.feedForwardFormat() != Configuration::FORMAT_HTK)
		throw std::runtime_error("Please use --ff_output_format HTK");

	    if (parallelSequences > 1)
		throw std::runtime_error("Please use --parallel 1");
	    
	    // Load data mean and std from data set
	    // Method 1:
            Cpu::real_vector outputMeans  = feedForwardSet->outputMeans();
            Cpu::real_vector outputStdevs = feedForwardSet->outputStdevs();
	    // B2
	    // Method 3:
	    if (neuralNetwork.externalOutputMV(outputMeans, outputStdevs)){
		printf("Mean and var loaded from externalDataMV\n");
	    }

	    // Decide whether de-normalization can be used
	    if (unstandardize && (!config.outputFromGateLayer()) &&
		outputMeans.size() == outputStdevs.size()        &&
		outputMeans.size() == outputDim){
		
		if (neuralNetwork.isMDNLayer(outputlayerID)){
		    // for MDNLayer output
		    if (config.mdnPara() < -1.0 || config.mdnPara() >= 0.0){
			neuralNetwork.setDenormalizationMV(outputMeans, outputStdevs);
			unstandardize = true;
		    }else{
			unstandardize = false;
		    }
		}else{
		    unstandardize = true;
		}
	    }else{
		unstandardize = false;
	    }

	    
	    // Print generation information
	    printf("Outputs from layer %d", config.outputFromWhichLayer());
	    if (config.outputFromGateLayer()) {printf(", gate output");}
	    if (config.mdnPara()>0 && neuralNetwork.isMDNLayer(outputlayerID))
		printf(", MDN with para=%f",config.mdnPara());
	    if (htkoutput)
		printf(", HTK format (float32, big-endian)");
	    else
		printf(", bin format");
	    if (unstandardize)
		printf(", de-normalized\n");
	    else
		printf(", NOT de-normalized\n");

	    
	    // Generation
	    int fracIdx = 0;
	    boost::shared_ptr<data_sets::DataSetFraction> frac;
		    
	    while (((frac = feedForwardSet->getNextFraction()))) {
		    
		// print fraction information
		printf("Computing outputs for data fraction %d ... ", ++fracIdx);
		for (int i = 0; i<frac->numSequences(); i++)
		    printf("%s ", frac->seqInfo(i).seqTag.c_str());
		fflush(stdout);

		// network propagation
		boost::posix_time::ptime sTime =
		    boost::posix_time::microsec_clock::local_time();
		neuralNetwork.notifyCurrentEpoch(config.fakeEpochNum());
		neuralNetwork.updateNNStateForGeneration();
		neuralNetwork.computeForwardPassGen(*frac, frac->maxSeqLength(),
						    generationOpt);
		boost::posix_time::ptime eTime =
		    boost::posix_time::microsec_clock::local_time();
		printf("\nTime (s): %f\n",
		       (real_t)(eTime-sTime).total_milliseconds()/1000.0);

		// take the generated data
		std::vector<real_t> output_vec =
		    neuralNetwork.getOutputNew(generationOpt);
		    
		// write one output file per sequence
		if (output_vec.size() > 0) {

		    // get file name
		    std::string seqTagSuf;
		    if (htkoutput) {
			seqTagSuf = ".htk";
		    }else{
			seqTagSuf = ".bin";
		    }
		    boost::filesystem::path seqPath(frac->seqInfo(0).seqTag+seqTagSuf);
		    std::string filename(seqPath.filename().string());
		    boost::filesystem::path oPath = 
			boost::filesystem::path(config.feedForwardOutputFile()) / 
			seqPath.relative_path().parent_path();
		    boost::filesystem::create_directories(oPath);
		    boost::filesystem::path filepath = oPath / filename;
			    
		    std::ofstream file(filepath.string().c_str(), 
				       std::ofstream::out | std::ios::binary);


		    // write header
		    if (htkoutput){
			unsigned tmp = (unsigned)(output_vec.size()/outputDim);
			swap32(&tmp);
			file.write((const char*)&tmp, sizeof(unsigned));
			tmp = (unsigned)(config.featurePeriod() * 1e4);
			swap32(&tmp);
			file.write((const char*)&tmp, sizeof(unsigned));
			unsigned short tmp2 = (unsigned short)(outputDim)*sizeof(float);
			swap16(&tmp2);
			file.write((const char*)&tmp2, sizeof(unsigned short));
			tmp2 = (unsigned short)(config.outputFeatureKind());
			swap16(&tmp2);
			file.write((const char*)&tmp2, sizeof(unsigned short));
		    }

		    int timeIdx = 0;
		    int dimIdx = 0;
		    // write the patterns
		    for (int dataIdx = 0; dataIdx < output_vec.size(); ++dataIdx){
			timeIdx = dataIdx / outputDim;
			dimIdx  = dataIdx % outputDim;
			
			float v;
			v = ( output_vec.size() > (dataIdx + output_lag * outputDim) ) ? 
			    ((float)output_vec[dataIdx + output_lag * outputDim]) :
			    ((float)output_vec[output_vec.size() - outputDim + dimIdx]);
			
			if (unstandardize) {
			    v *= outputStdevs[dimIdx];
			    v += outputMeans[dimIdx];
			}
			
			if (htkoutput)
			    swapFloat(&v); 
				    
			file.write((const char*)&v, sizeof(float));
		    }
		    file.close();
		}else{
		    printf("\tno data to write\n");
		}
            }

	    // print information
	    if (config.outputFromWhichLayer() > 0)
		printf("Outputs from layer %d", config.outputFromWhichLayer());
	    if (config.outputFromGateLayer())
		printf(", gate output");
	    if (config.mdnPara()>0 && neuralNetwork.isMDNLayer(outputlayerID))
		printf(", MDN with para=%f",config.mdnPara());

	    // clear cache
            if (feedForwardSet != boost::shared_ptr<data_sets::DataSet>()){
                std::cout << "\nRemoving cache file: ";
		std::cout << feedForwardSet->cacheFileName() << std::endl;
	    }
            boost::filesystem::remove(feedForwardSet->cacheFileName());
        }
	
    }
    catch (const std::exception &e) {
        printf("FAILED in running CURENNT: %s\n", e.what());
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
                std::cerr << "FAILED in GPU initializing: " << cudaGetErrorString(err) << std::endl;
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
            std::cerr << "FAILED in GPU initialization: " << cudaGetErrorString(err) << std::endl;
            return err;
        }
        std::cout << "Using device #" << device << " (" << prop.name << ")" << std::endl;
        if ((err = cudaSetDevice(device)) != cudaSuccess) {
            std::cerr << "FAILED in GPU initialization: " << cudaGetErrorString(err) << std::endl;
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
    if (!ifs.good()){
	std::cout << "\nCannot open file" << filename << std::endl;
        throw std::runtime_error("Failed to read network file");
    }
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
	printf("\n\nFail to parse: %s", filename.c_str());
	printf("\nFile is incorrect in Json grammar. Please check lines:\n\t\t");
	size_t start = (doc->GetErrorOffset()>20)?
	    (doc->GetErrorOffset()-20):doc->GetErrorOffset();
	for (int t=0;t<50;t++)
	    printf("%c", docStr.c_str()[start + t]);
	printf("\n");
        throw std::runtime_error(doc->GetParseError());
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

    std::string fileOrderedList = "";
    
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
	fileOrderedList= Configuration::instance().fileOrderedLstTrn();
        break;

    case DATA_SET_VALIDATION:
        type           = "validation set";
        filenames      = Configuration::instance().validationFiles();
        fraction       = Configuration::instance().validationFraction();
        truncSeqLength = Configuration::instance().truncateSeqLength();
	fileOrderedList= Configuration::instance().fileOrderedLstVal();
        break;

    case DATA_SET_TEST:
        type           = "test set";
        filenames      = Configuration::instance().testFiles();
        fraction       = Configuration::instance().testFraction();
	fileOrderedList= Configuration::instance().fileOrderedLstTst();
        break;

    default:
        type           = "feed forward input set";
        filenames      = Configuration::instance().feedForwardInputFiles();
        noiseDev       = Configuration::instance().inputNoiseSigma();
	truncSeqLength = Configuration::instance().truncateSeqLength();
	fileOrderedList= Configuration::instance().fileOrderedLstTst();
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
		noiseDev,   cachePath,  fileOrderedList);
    
    printf("done.\n");
    printf("Loaded fraction:  %d%%\n",    (int)(fraction*100));
    printf("Sequences:        %d\n",      ds->totalSequences());
    printf("Sequence lengths: %d..%d\n",  ds->minSeqLength(),
	                                  ds->maxSeqLength());
    printf("Total timesteps:  %lu\n",     ds->totalTimesteps());

    if (fileOrderedList != "")
	printf("Utterances ordered by list:  %s\n", fileOrderedList.c_str());
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


void printOptimizer(const Configuration &config)
{
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

void printTrainingErrorHead(std::string &infoRows)
{
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
    return;
}


/**** Dustbin

      B1
	// Add 0514 Wang: read data mean and variance (MV), initialize MDN 
	//   Obsolete, no long use datamvPath()
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

	B2
	    // Method 2:
	    //  Or load the mean and variation from 
	    if (config.datamvPath().size()>0){
		if (dataMV == NULL)
		    throw std::runtime_error("Can't read datamv");
		if (dataMV->outputM().size() != outputMeans.size())
		    throw std::runtime_error("output dimension mismatch datamv");
		outputMeans  = dataMV->outputM();
		outputStdevs = dataMV->outputV();
		printf("Mean and var are over-written by %s.\n",
		       config.datamvPath().c_str());
	    }

*/
