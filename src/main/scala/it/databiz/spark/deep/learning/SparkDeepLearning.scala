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
import org.apache.spark.{ SparkConf, SparkContext }
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator
import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster
import org.slf4j.LoggerFactory

/**
 * Scala object containing the main method to use in order to train a Convolutional Neural Network
 * from the MNIST dataset, taking advantage of Apache Spark's cluster computing.
 *
 * Created by Vincibean <andrebessi00@gmail.com> on 19/03/16.
 */
object SparkDeepLearning {

  def main(args: Array[String]): Unit = {
    // Create logger
    val log =
      LoggerFactory.getLogger("it.databiz.spark.deep.learning.MnistExample")

    // Create Spark context
    val sc = new SparkContext(
      new SparkConf()
        .setMaster(s"local[$numCores]") // Edit this line if you intend to use a different master node.
        .setAppName("Spark-Deep-Learning")
    )

    // Load data into Spark
    log.info("--- Loading data --- ")
    val (trainingSet, testSet) = new MnistDataSetIterator(batchSize, numSamples, true)
      .splitDatasetAt(numForTraining)
    val sparkTrainingData = sc
      .parallelize(trainingSet, numCores * batchSize)
      .cache()

    // Set up network configuration:
    // prepare the neural network
    log.info("--- Preparing neural network --- ")
    val neuralNetwork = prepareNeuralNetwork(height, width, numChannels)

    /* How frequently should we average parameters (in number of minibatches)?
      Averaging too frequently can be slow (synchronization + serialization costs) whereas too infrequently can result
      learning difficulties (i.e., network may not converge) */
    val averagingFrequency: Int = 5

    //Set up the TrainingMaster. The TrainingMaster controls how learning is actually executed on Spark
    //Here, we are using standard parameter averaging
    //For details on these configuration options, see: https://deeplearning4j.org/spark#configuring
    val trainingMaster =
      new ParameterAveragingTrainingMaster.Builder(batchSize)
        .workerPrefetchNumBatches(numCores) //Asynchronously prefetch up to numCores batches
        .averagingFrequency(averagingFrequency)
        .batchSizePerWorker(batchSize)
        .build()

    // Create Spark multi layer network from configuration
    val sparkNeuralNetwork =
      new SparkDl4jMultiLayer(sc, neuralNetwork, trainingMaster)

    // Train network
    log.info("--- Starting neural network training ---")

    (0 to epochs).foreach { i =>
      // We are training with approximately 'batchSize' examples on each executor core
      val network = sparkNeuralNetwork.fit(sparkTrainingData)
      log.info(s"--- Epoch $i complete ---")
      log.info(network.evaluateOn(testSet).stats)
    }

    log.info("--- Training finished --- ")

    log.info("--- Saving model --- ")

    val writing = neuralNetwork.saveAsZip()

    if (writing.isSuccess) {
      log.info("--- Model saved --- ")
    } else {
      log.error("--- Could not save model ---")
    }

    //Delete the temp training files, now that we are done with them
    trainingMaster.deleteTempFiles(sc)

  }

  /**
   * Prepare and configure the Convolutional Neural Network that will trained on the MNIST dataset.
   *
   * @param height      the height of the images contained in the MNIST dataset
   * @param width       the width of the images contained in the MNIST dataset
   * @param numChannels the number of channels of the images contained in the MNIST dataset
   * @return a fully configured MultiLayerNetwork
   */
  def prepareNeuralNetwork(height: Int, width: Int, numChannels: Int): MultiLayerNetwork = {
    val modelBuilder =
      MultiLayerConfigurationBuilder().setInputType(InputType.convolutional(height, width, numChannels))
    val modelConfig   = modelBuilder.build()
    val neuralNetwork = new MultiLayerNetwork(modelConfig)
    neuralNetwork.init()
    neuralNetwork.setUpdater(null)
    neuralNetwork
  }

}
