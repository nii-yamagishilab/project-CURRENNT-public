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

#ifndef OPTIMIZERS_OPTIMIZER_HPP
#define OPTIMIZERS_OPTIMIZER_HPP

#include "../NeuralNetwork.hpp"
#include "../data_sets/DataSet.hpp"
#include "../MacroDefine.hpp"


namespace optimizers {

    /******************************************************************************************//**
     * Base class for weight optimizers
     *
     * @param TDevice The computation device (Cpu or Gpu)
     *********************************************************************************************/
    template <typename TDevice>
    class Optimizer
    {
        typedef typename TDevice::real_vector real_vector;

    private:
        NeuralNetwork<TDevice> &m_neuralNetwork;
        data_sets::DataSet     &m_trainingSet;
        data_sets::DataSet     &m_validationSet;
        data_sets::DataSet     &m_testSet;

        const int m_maxEpochs;
        const int m_maxEpochsNoBest;
        const int m_validateEvery;
        const int m_testEvery;

        bool   m_finished;
	
	// 0511 wang: check Nan and other;
	bool   m_blowed;
	int    m_blowedTime;
        int    m_curEpoch;
        int    m_epochsSinceLowestError;

        real_t m_lowestValidationError;
        real_t m_curTrainingError;
        real_t m_curValidationError;
        real_t m_curTestError;
        real_t m_curValidationClassError;
        real_t m_curTrainingClassError;
        real_t m_curTestClassError;

	real_t m_curTrainingSecError;
	real_t m_curValidationSecError;
	real_t m_curTestSecError;
	
	// Add 0512 showing the Error per frame
	real_t m_curTrainingErrorPerFrame;
        real_t m_curValidationErrorPerFrame;
        real_t m_curTestErrorPerFrame;

	
        std::vector<real_vector> m_curWeightUpdates;
        std::vector<real_vector> m_bestWeights;
	
	// Add 1024 for AdaGrad
	unsigned                 m_optOption;
	std::vector<real_vector> m_weightStats;              
	std::string              m_optStatus;
	
    private:
        void   _processDataSet(data_sets::DataSet &ds, bool calcWeightUpdates,
			       real_t &error, real_t &classError, real_t &secError);
        void   _storeWeights();
        void   _restoreWeights();

    protected:
        static void _exportWeights(const helpers::JsonDocument &jsonDoc, 
				   const char *arrayName, const std::vector<real_vector> &weights);
        static void _importWeights(const helpers::JsonDocument &jsonDoc, 
				   const char *arrayName, std::vector<real_vector> *weights);
	
        NeuralNetwork<TDevice>&   _neuralNetwork();
        std::vector<real_vector>& _curWeightUpdates();
	
	// Add 16-11-02: Add fracLength, as the number of frames
        virtual void              _updateWeights(int fracLength) =0;
	
	/* Add 16-02-22 Wang: for WE updating */
	virtual void              _updateWeInput(int fracLength) =0;
	
	
	// Add 10-24 for AdaGrad
	const unsigned& _optOption() const;
	std::vector<real_vector>& _weightStats();
	
    public:
        /**
         * Constructs the optimizer
         *
         * @param neuralNetwork   The neural network to operate on
         * @param trainingSet     The set of training sequences
         * @param validationSet   The set of validation sequences
         * @param testSet         The set of test sequences
         * @param maxEpochs       The maximum total number of epochs to train
         * @param maxEpochsNoBest The number of epochs in which no new lowest error could be
         *                        achieved before training is stopped
         * @param validateEvery   After how many epochs the validation error shall be calculated
         * @param testEvery       After how many epochs the test error shall be calculated
         */
        Optimizer(
            NeuralNetwork<TDevice> &neuralNetwork,
            data_sets::DataSet     &trainingSet,
            data_sets::DataSet     &validationSet,
            data_sets::DataSet     &testSet,
            int maxEpochs, 
            int maxEpochsNoBest,
            int validateEvery,
            int testEvery,
	    unsigned optOption
            );

        /**
         * Destructs the optimizer
         */
        virtual ~Optimizer();

        /**
         * Check if the training is finished
         *
         * @return True if the training is finished
         */
        bool finished() const;

        /**
         * Returns the current training epoch
         *
         * @return The current training epoch
         */
        int currentEpoch() const;

        /**
         * Returns the lowest error on the validation set
         *
         * @return The lowest error on the validation set
         */
        real_t lowestValidationError() const;

        /**
         * Returns the number of training epochs since the lowest error on the validation set
         *
         * @return The number of training epochs since the lowest error on the validation set
         */
        int epochsSinceLowestValidationError() const;

        /**
         * Returns the current training set error
         *
         * @return The current training set error
         */
        real_t curTrainingError() const;

        /**
         * Returns the current validation set error
         *
         * @return The current validation set error
         */
        real_t curValidationError() const;

        /**
         * Returns the current test set error
         *
         * @return The current test set error
         */
        real_t curTestError() const;

        /**
         * Returns the current training set classification error
         *
         * @return The current training set classification error
         */
        real_t curTrainingClassError() const;

        /**
         * Returns the current validation set classification error
         *
         * @return The current validation set classification error
         */
        real_t curValidationClassError() const;

        /**
         * Returns the current test set classification error
         *
         * @return The current test set classification error
         */
        real_t curTestClassError() const;

        /**
         * Optimizes the weights
         *
         * If either the maximum number of training epochs from the process configuration has been
         * reached or no new lowest error has been achieved since the last x epochs, the function
         * returnes true and the network will not be trained any further.
         *
         * @return True if the training is finished
         */
        bool train();
	

	// add 05-11 
	bool blowed();
        /**
         * Writes the current state to a JSON tree
         *
         * @param jsonDoc The JSON document
         */
        virtual void exportState(const helpers::JsonDocument &jsonDoc) const;
	
	virtual void adjustLR() =0;
	
	virtual void changeLR(real_t newLR) =0;
	
	virtual void reinit() =0;
	
	void _reinit();
	

        /**
         * Restores the state from a JSON tree
         *
         * @param jsonDoc The JSON document
         */
        virtual void importState(const helpers::JsonDocument &jsonDoc);

	virtual void importParameter(const helpers::JsonDocument &jsonDoc);

	/**
	 * Show the error per frame
	 */
        real_t curTrainingErrorPerFrame() const;

        real_t curValidationErrorPerFrame() const;

        real_t curTestErrorPerFrame() const;

	real_t curTrainingErrorSec() const;

        real_t curValidationErrorSec() const;

        real_t curTestErrorSec() const;

       	
	const std::string& optStatus() const;
    };

} // namespace optimizers


#endif // OPTIMIZERS_OPTIMIZER_HPP
