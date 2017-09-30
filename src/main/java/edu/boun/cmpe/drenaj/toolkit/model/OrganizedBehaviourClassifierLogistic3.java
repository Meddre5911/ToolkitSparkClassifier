package edu.boun.cmpe.drenaj.toolkit.model;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;

import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.feature.IndexToString;
import org.apache.spark.ml.feature.LabeledPoint;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.StringIndexerModel;
import org.apache.spark.ml.feature.VectorIndexer;
import org.apache.spark.ml.feature.VectorIndexerModel;
import org.apache.spark.ml.linalg.Vectors;
import org.apache.spark.mllib.util.MLUtils;
import org.apache.spark.rdd.RDD;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SQLContext;
import org.apache.spark.sql.SparkSession;

import edu.boun.cmpe.drenaj.toolkit.model.entity.EvaluationResult;
import edu.boun.cmpe.drenaj.toolkit.model.util.MetricsUtil;
import scala.Tuple2;

public class OrganizedBehaviourClassifierLogistic3 {

	public static void main(String[] args) {

		SparkSession sparkSession = SparkSession.builder().master("local[*]")
				.appName("LogisticRegressionClassificationExample").getOrCreate();

		SQLContext sqlContext = new SQLContext(sparkSession);
		SparkContext sc = sparkSession.sparkContext();
		JavaSparkContext jsc = new JavaSparkContext(sc);

		// Load and parse the data file, converting it to a DataFrame.

		JavaRDD<LabeledPoint> labeledDataRDD = jsc.textFile(
				TrainingDataPaths.PCA_NON_HASHTAG_HILLARY_TRUMPL_PATH,
				1).flatMap(new FlatMapFunction<String, LabeledPoint>() {
					@Override
					public Iterator<LabeledPoint> call(String line) throws Exception {
						List<LabeledPoint> labeledPoints = new ArrayList<>();
						if (!line.startsWith("ClassId")) {

							String[] split = line.split(",");
							String[] featuresStr = Arrays.copyOfRange(split, 1, split.length);
							double[] featuresDoubleValues = Arrays.stream(featuresStr).mapToDouble(Double::parseDouble)
									.toArray();

							LabeledPoint pos = new LabeledPoint(Double.valueOf(split[0]),
									Vectors.dense(featuresDoubleValues));

							labeledPoints.add(pos);
						}

						return labeledPoints.iterator();
					}
				});

		// Automatically identify categorical features, and index them.
		// Set maxCategories so features with > 4 distinct values are treated as
		// continuous.
		Dataset<Row> data = sqlContext.createDataFrame(labeledDataRDD, LabeledPoint.class);

		// Index labels, adding metadata to the label column.
		// Fit on whole dataset to include all labels in index.
		StringIndexerModel labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel")
				.fit(data);

		VectorIndexerModel featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures")
				.setMaxCategories(4).fit(data);

		Tuple2<RDD<LabeledPoint>, RDD<LabeledPoint>>[] kFold = MLUtils.kFold(labeledDataRDD.rdd(), 10, 77678969l,
				labeledDataRDD.classTag());

		EvaluationResult evaluationResult = new EvaluationResult();

		System.out.println("K Fold Length : " + kFold.length);
		StringBuilder sb = new StringBuilder();
		for (int i = 0; i < kFold.length; i++) {
			try {
				sb.append("\n-----  K Fold : " + i + "\n");
				Tuple2<RDD<LabeledPoint>, RDD<LabeledPoint>> trainingAndTestData = kFold[i];

				testModelInRandomForest(sqlContext, trainingAndTestData._1(), trainingAndTestData._2(), featureIndexer,
						labelIndexer, sb, evaluationResult);
			} catch (Exception e) {

			}
		}

		System.out.println("All Results : " + sb.toString());
		evaluationResult.printSummary(kFold.length);

		jsc.close();

	}

	public static void testModelInRandomForest(SQLContext sqlContext, RDD<LabeledPoint> trainingDataRDD,
			RDD<LabeledPoint> testDataRDD, VectorIndexerModel featureIndexer, StringIndexerModel labelIndexer,
			StringBuilder sb, EvaluationResult evaluationResult) {

		// Split the data into training and test sets (30% held out for testing)
		Dataset<Row> trainingData = sqlContext.createDataFrame(trainingDataRDD, LabeledPoint.class);
		sb.append("Training Data Count : " + trainingData.count() + "\n");
		Dataset<Row> testData = sqlContext.createDataFrame(testDataRDD, LabeledPoint.class);
		sb.append("Test Data Count : " + testData.count() + "\n");

		// Train a RandomForest model.

		LogisticRegression logisticRegression = new LogisticRegression().setMaxIter(10).setLabelCol("indexedLabel")
				.setFeaturesCol("indexedFeatures");

		// Convert indexed labels back to original labels.
		IndexToString labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel")
				.setLabels(labelIndexer.labels());

		// Chain indexers and forest in a Pipeline
		Pipeline pipeline = new Pipeline()
				.setStages(new PipelineStage[] { labelIndexer, featureIndexer, logisticRegression, labelConverter });

		// Train model. This also runs the indexers.
		PipelineModel model = pipeline.fit(trainingData);

		// Make predictions.
		Dataset<Row> predictions = model.transform(testData);

		// Select example rows to display.
		JavaRDD<Row> predictionsAndLabels = predictions.select("predictedLabel", "label").toJavaRDD();

		JavaRDD<Tuple2<Object, Object>> predictionAndLabels1 = predictionsAndLabels
				.map(new Function<Row, Tuple2<Object, Object>>() {

					@Override
					public Tuple2<Object, Object> call(Row v1) throws Exception {
						// TODO Auto-generated method stub
						return new Tuple2<Object, Object>(Double.valueOf(v1.getAs(0)), v1.getDouble(1));
					}
				});

		MetricsUtil.analyzeMultiClassMetrics(predictionAndLabels1, sb, evaluationResult);
//		MetricsUtil.analyzeBinaryClassMetrics(predictionAndLabels1, sb, evaluationResult);

	}

}
