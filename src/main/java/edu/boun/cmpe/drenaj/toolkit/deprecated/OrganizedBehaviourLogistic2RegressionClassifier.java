package edu.boun.cmpe.drenaj.toolkit.deprecated;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;

import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.ml.classification.BinaryLogisticRegressionSummary;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.classification.LogisticRegressionSummary;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.ml.linalg.Vectors;
import org.apache.spark.mllib.classification.LogisticRegressionModel;
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS;
import org.apache.spark.mllib.linalg.DenseVector;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.util.MLUtils;
import org.apache.spark.rdd.RDD;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SQLContext;
import org.apache.spark.sql.SparkSession;

import edu.boun.cmpe.drenaj.toolkit.model.entity.EvaluationResult;
import edu.boun.cmpe.drenaj.toolkit.model.util.MetricsUtil;
import scala.Option;
import scala.Tuple2;

public class OrganizedBehaviourLogistic2RegressionClassifier {

	public static void main(String[] args) {

		SparkSession sparkSession = SparkSession.builder().master("local")
				.appName("JavavLogisticRegressionClassificationExample").getOrCreate();

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

		Tuple2<RDD<LabeledPoint>, RDD<LabeledPoint>>[] kFold = MLUtils.kFold(labeledDataRDD.rdd(), 10, 457687989080l,
				labeledDataRDD.classTag());

		EvaluationResult evaluationResult = new EvaluationResult();

		System.out.println("K Fold Length : " + kFold.length);
		StringBuilder sb = new StringBuilder();
		for (int i = 0; i < kFold.length; i++) {
			try {
				sb.append("\n-----  K Fold : " + i + "\n");
				Tuple2<RDD<LabeledPoint>, RDD<LabeledPoint>> trainingAndTestData = kFold[i];

				testModelInLogisticRegression(sqlContext, trainingAndTestData._1(), trainingAndTestData._2(), sb,
						evaluationResult);
			} catch (Exception e) {
				e.printStackTrace();
			}
		}

		System.out.println("All Results : " + sb.toString());
		evaluationResult.printSummary(kFold.length);
		jsc.close();

	}

	public static void testModelInLogisticRegression(SQLContext sqlContext, RDD<LabeledPoint> trainingDataRDD,
			RDD<LabeledPoint> testDataRDD, StringBuilder sb, EvaluationResult evaluationResult) {

		Vector generateInitialWeights = new LogisticRegressionWithLBFGS().setNumClasses(2).generateInitialWeights(trainingDataRDD);
		
		// // Run training algorithm to build the model.
		LogisticRegressionModel model = new LogisticRegressionWithLBFGS().setNumClasses(2).run(trainingDataRDD, generateInitialWeights);
//Option<Object> threshold = model.getThreshold();
		// Compute raw scores on the test set.
		JavaRDD<Tuple2<Object, Object>> predictionAndLabels = testDataRDD.toJavaRDD()
				.map(new Function<LabeledPoint, Tuple2<Object, Object>>() {
					public Tuple2<Object, Object> call(LabeledPoint p) {
						Double prediction = model.predict(p.features());
						if(p.label()==1d && prediction.doubleValue() == 1d){
							System.out.println("Prediction 1 yuppiii");
						}
						return new Tuple2<Object, Object>(prediction, p.label());
					}
				});

		MetricsUtil.analyzeBinaryClassMetrics(predictionAndLabels, sb, evaluationResult);
		MetricsUtil.analyzeMultiClassMetrics(predictionAndLabels, sb, evaluationResult);
	}

}
