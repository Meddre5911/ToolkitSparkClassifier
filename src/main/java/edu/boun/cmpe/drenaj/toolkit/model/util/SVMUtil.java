package edu.boun.cmpe.drenaj.toolkit.model.util;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.classification.SVMModel;
import org.apache.spark.mllib.classification.SVMWithSGD;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.rdd.RDD;

import edu.boun.cmpe.drenaj.toolkit.model.entity.EvaluationResult;
import scala.Tuple2;

public class SVMUtil {
	public static EvaluationResult evaluateCrossValidationWithThreshold(
			Tuple2<RDD<LabeledPoint>, RDD<LabeledPoint>>[] kFold, StringBuilder sb, Double threshold) {
		EvaluationResult evaluationResult = new EvaluationResult();
		System.out.println("K Fold Length : " + kFold.length);
		for (int i = 0; i < kFold.length; i++) {
			try {
				sb.append("\n-----  K Fold : " + i + "\n");
				Tuple2<RDD<LabeledPoint>, RDD<LabeledPoint>> trainingAndTestData = kFold[i];

				testModelInSVMClassifier(trainingAndTestData._1(), trainingAndTestData._2(), sb, threshold,
						evaluationResult);
			} catch (Exception e) {

			}
		}
		return evaluationResult;
	}

	private static void testModelInSVMClassifier(RDD<LabeledPoint> trainingDataRDD, RDD<LabeledPoint> testDataRDD,
			StringBuilder sb, Double threshold, EvaluationResult evaluationResult) {

		int numIterations = 100;
		SVMModel model = SVMWithSGD.train(trainingDataRDD, numIterations).setThreshold(threshold);

		// Compute raw scores on the test set.
		JavaRDD<Tuple2<Object, Object>> predictionAndLabels = testDataRDD.toJavaRDD()
				.map(new Function<LabeledPoint, Tuple2<Object, Object>>() {
					public Tuple2<Object, Object> call(LabeledPoint p) {
						Double score = model.predict(p.features());
						return new Tuple2<Object, Object>(score, p.label());
					}
				});

//		MetricsUtil.analyzeBinaryClassMetrics(predictionAndLabels, sb, evaluationResult);
		MetricsUtil.analyzeMultiClassMetrics(predictionAndLabels, sb, evaluationResult);

	}
}
