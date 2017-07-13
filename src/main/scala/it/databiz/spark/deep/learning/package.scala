package it.databiz.spark.deep

import java.io.File

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator
import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.dataset.DataSet

import scala.collection.JavaConverters._
import scala.util.Try

/**
 * Container object of Scala implicits to use in order to train a Convolutional Neural Network
 * from the MNIST dataset, taking advantage of Apache Spark's cluster computing.
 *
 * Created by Vincibean <andrebessi00@gmail.com> on 26/03/16.
 */
package object learning {

  /**
   * MnistDataSetIterator wrapper, loads the dataset that should be used in order to
   * train a Convolutional Neural Network on the MNIST dataset, taking advantage of
   * Apache Spark's cluster computing.
   */
  implicit class MnistDataSetIteratorOps(mnistIterator: MnistDataSetIterator) {

    /**
     * Loads the MNIST dataset to be used in the MNIST example, shuffles it, then
     * splits it into training set and test set.
     *
     * @param numForTraining the number of instances in the dataset that should be used
     *                       for the training set. It must be a number between 1 and totalNumSamples.
     * @param numSamples     the number of all instances contained in the dataset.
     * @return the training set and the test set, each containing MNIST instances.
     */
    def splitDatasetAt(numForTraining: Int)(implicit numSamples: Int): (Seq[DataSet], Seq[DataSet]) = {
      require(0 < numForTraining && numForTraining < numSamples)
      val allData      = mnistIterator.asScala.toSeq
      val shuffledData = scala.util.Random.shuffle(allData)
      shuffledData.splitAt(numForTraining)
    }

  }

  /**
   * MultiLayerNetwork wrapper, provides a set of utility methods to Artificial Neural Networks.
   */
  implicit class MultiLayerNetworkOps(network: MultiLayerNetwork) {

    /**
     * Returns an Evaluation of the Convolutional Neural Network on the test set.
     *
     * @param testSet the MNIST test set on which to perform the evaluation.
     * @return an Evaluation of the Convolutional Neural Network on the test set.
     */
    def evaluateOn(testSet: Seq[DataSet]): Evaluation = {
      val eval = new Evaluation()
      testSet.foreach { ds =>
        val output = network.output(ds.getFeatureMatrix)
        eval.eval(ds.getLabels, output)
      }
      eval
    }

    /**
     * Saves the Artificial Neural Network's configurations and coefficients on disk.
     *
     * @return a Try monad indicating if the computation resulted in an Exception or not.
     */
    def saveAsZip(): Try[Unit] = Try {
      //Save the model
      val locationToSave = new File("MnistMultiLayerNetwork.zip") //Where to save the network. Note: the file is in .zip format - can be opened externally
      val saveUpdater    = true //Updater: i.e., the state for Momentum, RMSProp, Adagrad etc. Save this if you want to train your network more in the future
      ModelSerializer.writeModel(network, locationToSave, saveUpdater)
    }

  }

}
