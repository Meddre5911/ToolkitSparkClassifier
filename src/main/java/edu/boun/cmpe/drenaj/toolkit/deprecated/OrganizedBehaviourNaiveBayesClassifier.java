package edu.boun.cmpe.drenaj.toolkit.deprecated;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;

import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.mllib.classification.LogisticRegressionModel;
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS;
import org.apache.spark.mllib.evaluation.MulticlassMetrics;
import org.apache.spark.mllib.linalg.DenseVector;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.util.MLUtils;
import org.apache.spark.rdd.RDD;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SQLContext;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.mllib.classification.NaiveBayes;
import org.apache.spark.mllib.classification.NaiveBayesModel;

import scala.Tuple2;

public class OrganizedBehaviourNaiveBayesClassifier {

	public static void main(String[] args) {

		SparkSession sparkSession = SparkSession.builder().master("local")
				.appName("JavaRandomForestClassificationExample").getOrCreate();

		SQLContext sqlContext = new SQLContext(sparkSession);
		SparkContext sc = sparkSession.sparkContext();
		JavaSparkContext jsc = new JavaSparkContext(sc);

		// Load and parse the data file, converting it to a DataFrame.

		JavaRDD<LabeledPoint> labeledDataRDD = jsc.textFile(
				"/Users/erdembegenilmis/Desktop/workspaces/workspaceDirenaj/ToolkitSparkClassifier/data/organizedBehaviour/extractedFeatures_20170516234747446.csv",
				1).flatMap(new FlatMapFunction<String, LabeledPoint>() {
					@Override
					public Iterator<LabeledPoint> call(String line) throws Exception {
						List<LabeledPoint> labeledPoints = new ArrayList<>();
						if (!line.startsWith("ClassId")) {

							String[] split = line.split(",");
							String[] featuresStr = Arrays.copyOfRange(split, 1, split.length);
							double[] featuresDoubleValues = Arrays.stream(featuresStr).mapToDouble(Double::parseDouble)
									.toArray();

							LabeledPoint pos = new LabeledPoint(Double.valueOf(split[0]).doubleValue(),
									new DenseVector(featuresDoubleValues));

							labeledPoints.add(pos);
						}

						return labeledPoints.iterator();
					}
				});

		// Automatically identify categorical features, and index them.
		// Set maxCategories so features with > 4 distinct values are treated as
		// continuous.

		Tuple2<RDD<LabeledPoint>, RDD<LabeledPoint>>[] kFold = MLUtils.kFold(labeledDataRDD.rdd(), 10,
				8762387863872386238l, labeledDataRDD.classTag());

		System.out.println("K Fold Length : " + kFold.length);
		StringBuilder sb = new StringBuilder();
		for (int i = 0; i < kFold.length; i++) {
			try {
				sb.append("\n-----  K Fold : " + i + "\n");
				Tuple2<RDD<LabeledPoint>, RDD<LabeledPoint>> trainingAndTestData = kFold[i];

				testModelInLogisticRegression(sqlContext, trainingAndTestData._1(), trainingAndTestData._2(), sb);
			} catch (Exception e) {

			}
		}

		System.out.println("All Results : " + sb.toString());
		jsc.close();

	}

	public static void testModelInLogisticRegression(SQLContext sqlContext, RDD<LabeledPoint> trainingDataRDD,
			RDD<LabeledPoint> testDataRDD, StringBuilder sb) {

		NaiveBayesModel model = NaiveBayes.train(trainingDataRDD, 1.0);

		// Compute raw scores on the test set.
		JavaRDD<Tuple2<Object, Object>> predictionAndLabels = testDataRDD.toJavaRDD()
				.map(new Function<LabeledPoint, Tuple2<Object, Object>>() {
					public Tuple2<Object, Object> call(LabeledPoint p) {
						Double prediction = model.predict(p.features());
						return new Tuple2<Object, Object>(prediction, p.label());
					}
				});

		MulticlassMetrics metrics = new MulticlassMetrics(predictionAndLabels.rdd());

		// Split the data into training and test sets (30% held out for testing)
		Dataset<Row> trainingData = sqlContext.createDataFrame(trainingDataRDD, LabeledPoint.class);
		sb.append("Training Data Count : " + trainingData.count() + "\n");
		Dataset<Row> testData = sqlContext.createDataFrame(testDataRDD, LabeledPoint.class);
		sb.append("Test Data Count : " + testData.count() + "\n");

		double f_measure = metrics.fMeasure();
		double precision = metrics.precision();
		double recall = metrics.recall();
		double accuracy = metrics.accuracy();

		sb.append("Precision = " + precision + "\n");
		sb.append("Recall = " + recall + "\n");
		sb.append("F-measure = " + f_measure + "\n");
		sb.append("Accuracy = " + accuracy + "\n");

	}

}
