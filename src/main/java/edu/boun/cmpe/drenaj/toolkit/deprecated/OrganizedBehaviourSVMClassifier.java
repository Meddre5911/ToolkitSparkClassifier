package edu.boun.cmpe.drenaj.toolkit.deprecated;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.classification.SVMModel;
import org.apache.spark.mllib.classification.SVMWithSGD;
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics;
import org.apache.spark.mllib.linalg.DenseVector;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.util.MLUtils;
import org.apache.spark.rdd.RDD;
import org.apache.spark.sql.SQLContext;
import org.apache.spark.sql.SparkSession;

import edu.boun.cmpe.drenaj.toolkit.model.TrainingDataPaths;
import edu.boun.cmpe.drenaj.toolkit.model.concurrency.SVMClassifierTask;
import edu.boun.cmpe.drenaj.toolkit.model.entity.EvaluationResult;
import edu.boun.cmpe.drenaj.toolkit.model.entity.ThresholdFinderEntity;
import edu.boun.cmpe.drenaj.toolkit.model.util.SVMUtil;
import scala.Tuple2;

public class OrganizedBehaviourSVMClassifier {

	public static void main(String[] args) {

		SparkSession sparkSession = SparkSession.builder().master("local[*]")
				.appName("SVMClassificationExample").getOrCreate();

		SQLContext sqlContext = new SQLContext(sparkSession);
		SparkContext sc = sparkSession.sparkContext();
		JavaSparkContext jsc = new JavaSparkContext(sc);

		// Load and parse the data file, converting it to a DataFrame.

		JavaRDD<LabeledPoint> labeledDataRDD = jsc.textFile(
				TrainingDataPaths.ALL_DATA_PATH_ORGANIZED,
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

		Set<Double> allThresholds = new HashSet<>();

		StringBuilder sb = new StringBuilder();
		for (int i = 0; i < kFold.length; i++) {
			try {
				sb.append("\n-----  K Fold : " + i + "\n");
				Tuple2<RDD<LabeledPoint>, RDD<LabeledPoint>> trainingAndTestData = kFold[i];

				getThresholds(trainingAndTestData._1(), trainingAndTestData._2(), sb, allThresholds);
			} catch (Exception e) {

			}
		}

		ExecutorService executorService = Executors.newFixedThreadPool(20);

		ThresholdFinderEntity thresholdFinderEntity = new ThresholdFinderEntity();
		System.out.println("Threshold size : " + allThresholds.size());
		int i = 0;
		for (Double threshold : allThresholds) {
			System.out.println("Analysed Threshold Index " + ++i);
			SVMClassifierTask svmClassifierTask = new SVMClassifierTask(threshold, thresholdFinderEntity, kFold);
			executorService.submit(svmClassifierTask);
		}

		executorService.shutdown();
		while (!executorService.isTerminated()) {
		}

		System.out.println("All Results : " + sb.toString());
		System.out.println("Best Threshold : " + thresholdFinderEntity.getBestThreshold());
		EvaluationResult finalEvaluation = SVMUtil.evaluateCrossValidationWithThreshold(kFold, sb,
				thresholdFinderEntity.getBestThreshold());
		finalEvaluation.printSummary(kFold.length);

		jsc.close();

		try {
			executorService.awaitTermination(Long.MAX_VALUE, TimeUnit.MILLISECONDS);
		} catch (InterruptedException e) {
			e.printStackTrace();
		}

	}

	private static void getThresholds(RDD<LabeledPoint> trainingDataRDD, RDD<LabeledPoint> testDataRDD,
			StringBuilder sb, Set<Double> allThresholds) {
		int numIterations = 100;
		SVMModel model = SVMWithSGD.train(trainingDataRDD, numIterations);
		// Clear the default threshold.
		model.clearThreshold();

		// Compute raw scores on the test set.
		JavaRDD<Tuple2<Object, Object>> scoreAndLabels = testDataRDD.toJavaRDD()
				.map(new Function<LabeledPoint, Tuple2<Object, Object>>() {
					public Tuple2<Object, Object> call(LabeledPoint p) {
						Double score = model.predict(p.features());
						return new Tuple2<Object, Object>(score, p.label());
					}
				});

		BinaryClassificationMetrics metrics = new BinaryClassificationMetrics(JavaRDD.toRDD(scoreAndLabels));

		// F Score by threshold
		JavaRDD<Tuple2<Object, Object>> f1Score = metrics.fMeasureByThreshold().toJavaRDD();
		parseThresholds(f1Score, allThresholds);
		sb.append("F1 Score by threshold: " + f1Score.collect() + "\n");

	}

	public static void parseThresholds(JavaRDD<Tuple2<Object, Object>> thresholdRDD, Set<Double> allThresholds) {
		List<Tuple2<Object, Object>> collect = thresholdRDD.collect();
		for (Tuple2<Object, Object> tuple : collect) {
			double fMeasure = new Double(tuple._2().toString());
			if (fMeasure > 0.8d) {
				allThresholds.add(new Double(tuple._1().toString()));
			}
		}
	}

}
