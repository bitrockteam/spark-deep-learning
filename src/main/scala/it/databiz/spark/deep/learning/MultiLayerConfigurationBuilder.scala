/**
 * Copyright (C) 2016  Databiz s.r.l.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
package it.databiz.spark.deep.learning

import it.databiz.spark.deep.learning.Conf._
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.layers.{ ConvolutionLayer, DenseLayer, OutputLayer, SubsamplingLayer }
import org.deeplearning4j.nn.conf.{ MultiLayerConfiguration, NeuralNetConfiguration, Updater }
import org.deeplearning4j.nn.weights.WeightInit
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.lossfunctions.LossFunctions

/**
 * A dedicated MultiLayerConfiguration Builder to use in order to train a Convolutional Neural Network
 * from the MNIST dataset, taking advantage of Apache Spark's cluster computing.
 *
 * Created by Vincibean <andrebessi00@gmail.com> on 20/03/16.
 */
object MultiLayerConfigurationBuilder extends MultiLayerConfiguration.Builder {

  def apply(): MultiLayerConfiguration.Builder =
    new NeuralNetConfiguration.Builder()
      .seed(seed) // Random number generator seed. Used for reproducibility between runs.
      .iterations(iterations) // Number of optimization iterations.
      .regularization(true) // Whether to use regularization (L1, L2, dropout, etc...)
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT) // Optimization algorithm to use.
      .activation(Activation.LEAKYRELU)
      .weightInit(WeightInit.XAVIER) // Weight initialization scheme. The Xavier algorithm automatically determines the scale of initialization based on the number of input and output neurons.
      .learningRate(0.02) // Learning rate.
      .updater(Updater.NESTEROVS) // Gradient updater
      .momentum(0.9)
      .regularization(true)
      .l2(1e-4) // L2 regularization coefficient.
      .list()
      .layer(0, new DenseLayer.Builder().nIn(height * width).nOut(500).build())
      .layer(1, new DenseLayer.Builder().nIn(500).nOut(100).build())
      .layer(
        2,
        new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
          .activation(Activation.SOFTMAX)
          .nIn(100) // Number of input neurons
          .nOut(outputNum) // Number of output neurons.
          .build()
      )
      .backprop(true) // Whether to use backpropagation.
      .pretrain(false) // Whether to pretrain the Convolutional Neural Network.

}
