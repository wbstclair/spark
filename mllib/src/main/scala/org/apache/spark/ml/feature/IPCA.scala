/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.ml.feature

import org.apache.hadoop.fs.Path
import org.apache.spark.annotation.Since
import org.apache.spark.ml._
import org.apache.spark.ml.linalg.{DenseMatrix, _}
import org.apache.spark.ml.param._
import org.apache.spark.ml.param.shared._
import org.apache.spark.ml.util._
import org.apache.spark.mllib.feature
import org.apache.spark.mllib.feature.{StandardScaler => NormalScaler, StandardScalerModel => NormalScalerModel}
import org.apache.spark.mllib.linalg.MatrixImplicits._
import org.apache.spark.mllib.linalg.VectorImplicits._
import org.apache.spark.rdd.RDD
import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{StructField, StructType}
import org.apache.spark.util.VersionUtils.majorVersion
//new imports
import org.apache.spark.mllib.linalg.distributed.{RowMatrix, IndexedRowMatrix, CoordinateMatrix}
import org.apache.spark.mllib.linalg.{DenseMatrix => OldDenseMatrix, DenseVector => OldDenseVector, Matrices => OldMatrices, Vector => OldVector, Vectors => OldVectors, SparseMatrix => OldSparseMatrix, SparseVector => OldSparseVector}
import breeze.linalg.{svd => brzSvd, DenseMatrix => BDM, DenseVector => BDV, Matrix => BM, sum => sumBreeze}
import java.util.Arrays
import scala.math.sqrt
import org.apache.spark.SparkContext

/**
 * Params for [[IPCA]] and [[IPCAModel]].
 */
private[feature] trait IPCAParams extends Params with HasInputCol with HasOutputCol {

  /**
   * The number of principal components.
   * @group param
   */
  final val k: IntParam = new IntParam(this, "k", "the number of principal components (> 0)",
    ParamValidators.gt(0))

  /** @group getParam */
  def getK: Int = $(k)

  /**
    * The weighting of past principal components. Implies weighting of future components by 1-alpha.
    * @group param
    */
  final val alpha: DoubleParam = new DoubleParam(this,"alpha", "the weight of past principal components (0 < alpha < 1)",
    ParamValidators.inRange(0,1,false,false))

  /** @group getParam */
  def getAlpha: Double = $(alpha)

  /** Validates and transforms the input schema. */
  protected def validateAndTransformSchema(schema: StructType): StructType = {
    SchemaUtils.checkColumnType(schema, $(inputCol), new VectorUDT)
    require(!schema.fieldNames.contains($(outputCol)),
      s"Output column ${$(outputCol)} already exists.")
    val outputFields = schema.fields :+ StructField($(outputCol), new VectorUDT, false)
    StructType(outputFields)
  }

}

/**
 * IPCA trains a model to project vectors to a lower dimensional space of the top `IPCA!.k`
 * principal components.
 */
class IPCA (override val uid: String)
  extends Estimator[IPCAModel] with IPCAParams with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("ipca"))

  /** @group setParam */
  def setInputCol(value: String): this.type = set(inputCol, value)

  /** @group setParam */
  def setOutputCol(value: String): this.type = set(outputCol, value)

  /** @group setParam */
  def setK(value: Int): this.type = set(k, value)

  /** @group setParam */
  def setAlpha(value: Double): this.type = set(alpha, value)

  /** @group getParam */
  def getNumFeatures: Int = numFeatures

  /**
    * The number of features which the model initially fit.
    */
  private var numFeatures: Int = 0

  /**
    * The total number of samples the model has fit by the model.
    * @group privateParam
    */
  private var samplesFit: Long = 0

  /**
    * Current eigenvalues, stored for incremental computation of new principal components.
    * @group privateParam
    */
  private var eigenvalues: OldDenseVector = null

  /**
    * Current eigenvectors, stored for incremental computation of new principal components.
    * @group privateParam
    */
  private var eigenvectors: OldDenseMatrix = null

  /**
    * Current StandardScalerModel which mean-normalizes input.
    * @group privateParam
    */
  private var meanNormalizer: NormalScalerModel = null

  /**
   * Computes an [[IPCAModel]] that contains the principal components of the input vectors.
   */
  override def fit(dataset: Dataset[_]): IPCAModel = {
    transformSchema(dataset.schema, logging = true)
    //implicit val sc: SparkContext = null
    val input: RDD[OldVector] = dataset.select($(inputCol)).rdd.map {
      case Row(v: Vector) => OldVectors.fromML(v)
    }

    val currentFeatures = input.first().size
    require($(k) <= currentFeatures,
      s"source vector size $currentFeatures must be no less than k=$k")
    require(currentFeatures <= 65535,
      s"source vector size $currentFeatures cannot be longer than 65535 to compute, due to constraints from RowMatrix")


    // Initial batch PCA computation

    // Set the number of features. Used to enforce requirements
    this.numFeatures = currentFeatures

    // Compute mean normalizer and transform data
    val firstNormalizer = new NormalScaler(withMean = true, withStd = false)
    this.meanNormalizer = firstNormalizer.fit(input)
    val normalizedRM = new RowMatrix(this.meanNormalizer.transform(input))

    // Initialize number of samples fit
    this.samplesFit = normalizedRM.numRows

    val initialSVD = normalizedRM.computeSVD(this.numFeatures,true)

    // Initial eigenvalues and eigenvectors
    this.eigenvectors = OldMatrices.dense(initialSVD.U.numRows.toInt, initialSVD.U.numCols.toInt, initialSVD.U
      .rows.map(_.toArray).collect.flatten).toDense
    this.eigenvalues = OldVectors.dense(initialSVD.s.toArray).toDense

    // Compute Explained Variance
    val eigenSum = initialSVD.s.toArray.sum
    val expVar = initialSVD.s.toArray.map(_ / eigenSum)

    val pc: OldDenseMatrix = OldMatrices.dense(this.numFeatures, $(k),
      Arrays.copyOfRange(this.eigenvectors.toArray, 0, this.numFeatures * $(k))).toDense
    val explainedVariance: OldDenseVector = OldVectors.dense(Array.fill[Double](this.numFeatures)(0)).toDense

    // Restrict to principal components for IPCAModel
   if ($(k) == this.numFeatures) {
     val pc = this.eigenvectors
     val explainedVariance = OldVectors.dense(expVar)
   } else {
     val pc = OldMatrices.dense(this.numFeatures, $(k),
       Arrays.copyOfRange(this.eigenvectors.toArray, 0, this.numFeatures * $(k)))
     // expVar is null! Why is it not being computed?
     val explainedVariance = OldVectors.dense(Arrays.copyOfRange(expVar, 0, $(k)))
   }
    copyValues(new IPCAModel(uid, pc, explainedVariance).setParent(this))
  }

  /**
    * Computes an [[IPCAModel]] that contains the principal components of the total fit data set by combining the input
    * vector with previous components. Should only be called after an initial fit.
    * @param dataset
    * @param sc
    * @return
    */
  def incrementFit(dataset: Dataset[_])(implicit sc: SparkContext): IPCAModel = {
    transformSchema(dataset.schema, logging = true)
    val input: RDD[OldVector] = dataset.select($(inputCol)).rdd.map {
      case Row(v: Vector) => OldVectors.fromML(v)
    }

    val currentFeatures = input.first().size
    require($(k) <= currentFeatures,
      s"source vector size $currentFeatures must be no less than k=$k")

    require(this.numFeatures > 0,
      s"incrementFit should only be called after fit. Use fit to set initial model before incrementing.")

    require(this.numFeatures == currentFeatures,
      s"source vector size $currentFeatures must be equal to the initially fit vector size numFeatures=$numFeatures")

    // Update the Mean Vector
    val tempMat = new RowMatrix(input)
    val newSamples = tempMat.numRows
    this.samplesFit = this.samplesFit + newSamples
    // Figure a way to do this without using summary statistics
    val summary = tempMat.computeColumnSummaryStatistics()

    // Compute the new mean vector
    val meanArray = (this.meanNormalizer.mean.toArray, summary.mean.toArray.map(_ * (1-$(alpha))))
      .zipped.map(_ + _)

    // Update the mean normalization with new mean
    this.meanNormalizer = new NormalScalerModel(null, OldVectors.dense(meanArray), false, true)
    // Mean Normalize new observations and apply alpha weighting
    val normalizedVectors = this.meanNormalizer.transform(input)
      .map(vector => OldVectors.dense(vector.toArray.map(_*sqrt(1-$(alpha)))))

    // Multiply previous eigenvectors by their corresponding eigenvalues
    val prevEigenvectors = new OldDenseMatrix(this.eigenvectors.numCols, this.eigenvectors.numRows,
      this.eigenvectors.transpose.toArray)
      .multiply(OldMatrices.diag(OldVectors.dense(this.eigenvalues.toArray.map(eigVal => sqrt(eigVal*$(alpha))))).toDense)

    // Convert previous eigenvectors to RDD
    val tempColumns = prevEigenvectors.toArray.grouped(prevEigenvectors.numRows)
    val tempRows = tempColumns.toSeq.transpose
    val prevEigenvectorsDenseSeq = tempRows.map(row => OldVectors.dense(row.toArray))
    val evRDD =  sc.parallelize(prevEigenvectorsDenseSeq, normalizedVectors.partitions.size)

    val combinedRDD = evRDD.union(normalizedVectors)

    // Compute gramian matrix and convert to breeze for SVD
    val combinedMatrix = new RowMatrix(combinedRDD)
    val approxMat = combinedMatrix.computeGramianMatrix
    val breezeApproxMat = new BDM(approxMat.numCols, approxMat.numRows, approxMat.toArray)

    // Compute SVD, excluding u
    val brzSvd.SVD(_,s,v) = brzSvd(breezeApproxMat)

    // Compute Explained Variance
    val eigenSum = sumBreeze(s)
    val expVar = s.map(_ / eigenSum).toArray

    this.eigenvalues = OldVectors.dense(Arrays.copyOfRange(s.data, 0,$(k))).toDense

    // Incremented eigenvalues and eigenvectors
    val freshEigenvectors = combinedMatrix.multiply(OldMatrices.dense(v.rows, v.rows, v.data))

    this.eigenvectors = OldMatrices.dense(this.numFeatures, $(k), Arrays.copyOfRange(freshEigenvectors
      .rows.map(_.toArray).collect.flatten,0,this.numFeatures*$(k))).toDense


    val pc: OldDenseMatrix = OldMatrices.dense(this.numFeatures, $(k),
      Arrays.copyOfRange(this.eigenvectors.toArray, 0, this.numFeatures * $(k))).toDense
    val explainedVariance: OldDenseVector = OldVectors.dense(Array.fill[Double](this.numFeatures)(0)).toDense

    // Restrict to principal components for IPCAModel
    if ($(k) == this.numFeatures) {
      val pc = this.eigenvectors
      val explainedVariance = OldVectors.dense(expVar)
    } else {
      val pc = OldMatrices.dense(this.numFeatures, $(k),
        Arrays.copyOfRange(this.eigenvectors.toArray, 0, this.numFeatures * $(k)))
      val explainedVariance = OldVectors.dense(Arrays.copyOfRange(expVar, 0, $(k)))
    }
    copyValues(new IPCAModel(uid, pc, explainedVariance).setParent(this))
  }


  override def transformSchema(schema: StructType): StructType = {
    validateAndTransformSchema(schema)
  }

  override def copy(extra: ParamMap): IPCA = defaultCopy(extra)
}

object IPCA extends DefaultParamsReadable[IPCA] {

  override def load(path: String): IPCA = super.load(path)
}

/**
 * Model fitted by [[IPCA]]. Transforms vectors to a lower dimensional space.
 *
 * @param pc A principal components Matrix. Each column is one principal component.
 * @param explainedVariance A vector of proportions of variance explained by
 *                          each principal component.
 */
class IPCAModel private[ml] (
     override val uid: String,
     val pc: DenseMatrix,
     val explainedVariance: DenseVector)
  extends Model[IPCAModel] with IPCAParams with MLWritable {

  import IPCAModel._

  /** @group setParam */
  def setInputCol(value: String): this.type = set(inputCol, value)

  /** @group setParam */
  def setOutputCol(value: String): this.type = set(outputCol, value)

  /**
   * Transform a vector by computed Principal Components.
   *
   * @note Vectors to be transformed must be the same length as the source vectors given
   * to `IPCA.fit()`.
   */
  override def transform(dataset: Dataset[_]): DataFrame = {
    transformSchema(dataset.schema, logging = true)
    val pcaModel = new feature.PCAModel($(k),
      OldMatrices.fromML(pc).asInstanceOf[OldDenseMatrix],
      OldVectors.fromML(explainedVariance).asInstanceOf[OldDenseVector])

    // TODO: Make the transformer natively in ml framework to avoid extra conversion.
    val transformer: Vector => Vector = v => pcaModel.transform(OldVectors.fromML(v)).asML

    val ipcaOp = udf(transformer)
    dataset.withColumn($(outputCol), ipcaOp(col($(inputCol))))
  }

  override def transformSchema(schema: StructType): StructType = {
    validateAndTransformSchema(schema)
  }

  override def copy(extra: ParamMap): IPCAModel = {
    val copied = new IPCAModel(uid, pc, explainedVariance)
    copyValues(copied, extra).setParent(parent)
  }

  override def write: MLWriter = new IPCAModelWriter(this)
}


object IPCAModel extends MLReadable[IPCAModel] {

  private[IPCAModel] class IPCAModelWriter(instance: IPCAModel) extends MLWriter {

    private case class Data(pc: DenseMatrix, explainedVariance: DenseVector)

    override protected def saveImpl(path: String): Unit = {
      DefaultParamsWriter.saveMetadata(instance, path, sc)
      val data = Data(instance.pc, instance.explainedVariance)
      val dataPath = new Path(path, "data").toString
      sparkSession.createDataFrame(Seq(data)).repartition(1).write.parquet(dataPath)
    }
  }

  private class IPCAModelReader extends MLReader[IPCAModel] {

    private val className = classOf[IPCAModel].getName

    /**
     * Loads a [[IPCAModel]] from data located at the input path. Note that the model includes an
     * `explainedVariance` member that is not recorded by Spark 1.6 and earlier. A model
     * can be loaded from such older data but will have an empty vector for
     * `explainedVariance`.
     *
     * @param path path to serialized model data
     * @return a [[PCAModel]]
     */
    override def load(path: String): IPCAModel = {
      val metadata = DefaultParamsReader.loadMetadata(path, sc, className)

      val dataPath = new Path(path, "data").toString
      val model = if (majorVersion(metadata.sparkVersion) >= 2) {
        val Row(pc: DenseMatrix, explainedVariance: DenseVector) =
          sparkSession.read.parquet(dataPath)
            .select("pc", "explainedVariance")
            .head()
        new IPCAModel(metadata.uid, pc, explainedVariance)
      } else {
        val Row(pc: OldDenseMatrix) = sparkSession.read.parquet(dataPath).select("pc").head()
        new IPCAModel(metadata.uid, pc.asML,
          Vectors.dense(Array.empty[Double]).asInstanceOf[DenseVector])
      }
      DefaultParamsReader.getAndSetParams(model, metadata)
      model
    }
  }

  @Since("1.6.0")
  override def read: MLReader[IPCAModel] = new IPCAModelReader

  @Since("1.6.0")
  override def load(path: String): IPCAModel = super.load(path)
}
