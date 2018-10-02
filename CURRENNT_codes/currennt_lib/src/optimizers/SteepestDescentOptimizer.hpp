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

#ifndef OPTIMIZERS_STEEPESTDESCENTOPTIMIZER_HPP
#define OPTIMIZERS_STEEPESTDESCENTOPTIMIZER_HPP

#include "Optimizer.hpp"

#include <vector>


namespace optimizers {

    /******************************************************************************************//**
     * Optimizer that uses steepest descent
     *
     * @param TDevice The computation device (Cpu or Gpu)
     *********************************************************************************************/
    template <typename TDevice>
    class SteepestDescentOptimizer : public Optimizer<TDevice>
    {
        typedef typename TDevice::real_vector real_vector;

    private:
        real_t                   m_learningRate;
	real_t                   m_learningRateAdjust;
        const real_t             m_momentum;
        std::vector<real_vector> m_weightDeltas;

        /* Add 16-02-22 Wang: for WE updating */
	const real_t m_weLearningRate;

	/* Add 17-05-19 Wang: for Adam*/
	real_t  m_adamBeta1Accum;
	real_t  m_adamBeta2Accum;

    protected:
        virtual void _updateWeights(int fracLength);
	
	/* Add 16-02-22 Wang: for WE updating */
	virtual void _updateWeInput(int fracLength);

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
         * @param learningRate    The learning rate
         * @param momentum        The momentum
         */
        SteepestDescentOptimizer(
            NeuralNetwork<TDevice> &neuralNetwork,
            data_sets::DataSet     &trainingSet,
            data_sets::DataSet     &validationSet,
            data_sets::DataSet     &testSet,
            int maxEpochs, 
            int maxEpochsNoBest, 
            int validateEvery,
            int testEvery,
            real_t learningRate,
            real_t momentum,
	    real_t weLearningRate,
	    unsigned optOption,
	    real_t adjustLRRate
            );

        /**
         * Destructs the optimizer
         */
        virtual ~SteepestDescentOptimizer();

        /**
         * @see Optimizer::exportState
         */
        virtual void exportState(const helpers::JsonDocument &jsonDoc) const;

        /**
         * @see Optimizer::importState
         */
        virtual void importState(const helpers::JsonDocument &jsonDoc);


        virtual void importParameter(const helpers::JsonDocument &jsonDoc);
	
	// 0511 Wang: adjust the learning rate
	virtual void adjustLR();
	
	virtual void changeLR(real_t newLR);

	virtual void reinit();
	
    };

} // namespace optimizers


#endif // OPTIMIZERS_STEEPESTDESCENTOPTIMIZER_HPP
