/******************************************************************************
 * This file is an addtional component of CURRENNT. 
 * Xin WANG
 * National Institute of Informatics, Japan
 * 2016
 *
 * This file is part of CURRENNT. 
 * Copyright (c) 2013 Johannes Bergmann, Felix Weninger, Bjoern Schuller
 * Institute for Human-Machine Communication
 * Technische Universitaet Muenchen (TUM)
 * D-80290 Munich, Germany
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


#ifndef MACRODEFINE_HPP
#define MACRODEFINE_HPP

/***    For Optimizer   ***/
#define  OPTIMIZATION_ADAGRAD   1            //  AdaGrad
#define  OPTIMIZATION_AVEGRAD   2            //  average the gradient per fraction of data
#define  OPTIMIZATION_STOCHASTIC_ADAGRAD 3   //  Stochastic gradient + AdaGrad
#define  OPTIMIZATION_SGD_DECAY 4            //  Stochastic gradient, and learning rate decay
#define  OPTIMIZATION_ADAM      5

#define  OP_ADAGRADFACTOR 0.000001           // epsilon for AdaGrad
#define  OP_ADAMBETA1     0.9                // Beta1 for Adam
#define  OP_ADAMBETA2     0.999              // Beta2 for Adam
#define  OP_ADAMEPSILON   0.00000001         // epsilon for Adam

#define OP_BLOWED_THRESHOLD 5 // tolerance of blowed network

/***    For printing information ***/
#define  OP_VERBOSE_LEVEL_0 0            // print nothing additional to cerr
#define  OP_VERBOSE_LEVEL_1 1            // print error per utterance to cerr
#define  OP_VERBOSE_LEVEL_2 2            // print sigmoid error rate to cerr (used by GAN)
#define  OP_VERBOSE_LEVEL_3 3

/*** For Feedback Model ***/
#define NN_FEEDBACK_SCHEDULE_MIN 0.000 // Minimal value for the schedule sampling prob parameter
#define NN_FEEDBACK_SCHEDULE_SIG 20    // K in 1/(1+exp((x-K))/Para)

// Schedule sampling and sequence model code
#define NN_FEEDBACK_GROUND_TRUTH 0     // use ground truth directly
#define NN_FEEDBACK_DROPOUT_1N   1     // dropout, set to 1/N
#define NN_FEEDBACK_DROPOUT_ZERO 2     // dropout, set to zero
#define NN_FEEDBACK_SC_SOFT      3     // schedule sampling, use soft vector
#define NN_FEEDBACK_SC_MAXONEHOT 4     // schedule sampling, use one hot vector of the max prob
#define NN_FEEDBACK_SC_RADONEHOT 5     // schedule sampling, use one random output
#define NN_FEEDBACK_BEAMSEARCH   6     // beam search (for generation)

// Softmax generation method
#define NN_SOFTMAX_GEN_BEST      0
#define NN_SOFTMAX_GEN_SOFT      1
#define NN_SOFTMAX_GEN_SAMP      2


/*** For GAN ***/
// Flags for GAN
#define NN_SIGMOID_GAN_DEFAULT         0
#define NN_SIGMOID_GAN_ONE_SIDED_FLAG  1

// Flags for NN state
#define NN_STATE_GAN_DIS_NATDATA       1
#define NN_STATE_GAN_DIS_GENDATA       2
#define NN_STATE_GAN_GEN               0
#define NN_STATE_GAN_GEN_FEATMAT       3
#define NN_STATE_GENERATION_STAGE      4
#define NN_STATE_GAN_NOGAN             5
#define NN_STATE_GAN_NOGAN_TRAIN       6

/*** For Normal layers ***/
#define NN_OPERATOR_LAYER_NOISE_TIMEREPEAT 1
#define NN_OPERATOR_LAYER_NOISE_DIMREPEAT  2
#define NN_OPERATOR_LAYER_NOISE_NOREPEAT   0

/*** For postoutput layers ***/
#define NN_POSTOUTPUTLAYER_LAST         1  // the true postoutput layer
#define NN_POSTOUTPUTLAYER_MIDDLEOUTPUT 2  // the middle postoutput for GAN
#define NN_POSTOUTPUTLAYER_FEATMATCH    3  // the middle postoutput (featmatch) for GAN
#define NN_POSTOUTPUTLAYER_NOTLASTMDN   4  // the postoutput (MDN) for acosutic model in GAN
#define NN_POSTOUTPUTLAYER_VAEKL        5  // the KL divergence layer

/*** For LoadExternalData function ***/
#define NN_INDEXLOADMETHOD_DEFAULT      0  //
#define NN_INDEXLOADMETHOD_1            1  //


#endif
