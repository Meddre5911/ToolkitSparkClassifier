package edu.boun.cmpe.drenaj.toolkit.model.util;

import java.util.Map;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics;
import org.apache.spark.mllib.evaluation.MulticlassMetrics;

import edu.boun.cmpe.drenaj.toolkit.model.entity.EvaluationResult;
import edu.boun.cmpe.drenaj.toolkit.model.entity.ThresholdFinderEntity;
import scala.Tuple2;

public class MetricsUtil {

	public static void analyzeMultiClassMetrics(JavaRDD<Tuple2<Object, Object>> predictionAndLabels1, StringBuilder sb,
			EvaluationResult evaluationResult) {

		MulticlassMetrics metrics = new MulticlassMetrics(predictionAndLabels1.rdd());

		double[] labels = metrics.labels();

		double f_measure = metrics.fMeasure(1);
		double precision = metrics.precision(1);
		double recall = metrics.recall(1);
		double accuracy = metrics.accuracy();

		evaluationResult.addAccuracy(accuracy);
		evaluationResult.addFMeasure(f_measure);
		evaluationResult.addPrecision(precision);
		evaluationResult.addRecall(recall);

		sb.append("Precision = " + precision + "\n");
		sb.append("Recall = " + recall + "\n");
		sb.append("F-measure = " + f_measure + "\n");
		sb.append("Accuracy = " + accuracy + "\n");
	}

	public static void analyzeBinaryClassMetrics(JavaRDD<Tuple2<Object, Object>> predictionAndLabels, StringBuilder sb,
			EvaluationResult evaluationResult) {
		// Get evaluation metrics.
		BinaryClassificationMetrics metrics = new BinaryClassificationMetrics(JavaRDD.toRDD(predictionAndLabels));

		// Precision by threshold
		JavaRDD<Tuple2<Object, Object>> precision = metrics.precisionByThreshold().toJavaRDD();
		sb.append("Precision by threshold: " + precision.collect() + "\n");

		// Recall by threshold
		JavaRDD<Tuple2<Object, Object>> recall = metrics.recallByThreshold().toJavaRDD();
		sb.append("Recall by threshold: " + recall.collect() + "\n");

		// F Score by threshold
		JavaRDD<Tuple2<Object, Object>> f1Score = metrics.fMeasureByThreshold().toJavaRDD();
		sb.append("F1 Score by threshold: " + f1Score.collect() + "\n");

		JavaRDD<Tuple2<Object, Object>> f2Score = metrics.fMeasureByThreshold(2.0).toJavaRDD();
		sb.append("F2 Score by threshold: " + f2Score.collect() + "\n");

		// Precision-recall curve
		JavaRDD<Tuple2<Object, Object>> prc = metrics.pr().toJavaRDD();
		sb.append("Precision-recall curve: " + prc.collect() + "\n");

		// Thresholds
		JavaRDD<Double> thresholds = precision.map(new Function<Tuple2<Object, Object>, Double>() {
			@Override
			public Double call(Tuple2<Object, Object> t) {
				return new Double(t._1().toString());
			}
		});

		// ROC Curve
		JavaRDD<Tuple2<Object, Object>> roc = metrics.roc().toJavaRDD();
		sb.append("ROC curve: " + roc.collect() + "\n");

		// AUPRC
		sb.append("Area under precision-recall curve = " + metrics.areaUnderPR() + "\n");

		// AUROC
		sb.append("Area under ROC = " + metrics.areaUnderROC() + "\n");

		evaluationResult.addAreaUnderROC(metrics.areaUnderROC());
	}

	public static EvaluationResult getSummaryOfCrossValidation(EvaluationResult evaluationResult, double kFold) {
		EvaluationResult crossValidationSummary = new EvaluationResult();

		crossValidationSummary.setAccuracy(evaluationResult.getAccuracy() / kFold);
		crossValidationSummary.setPrecision(evaluationResult.getPrecision() / kFold);
		crossValidationSummary.setRecall(evaluationResult.getRecall() / kFold);
		crossValidationSummary.setF_measure(evaluationResult.getF_measure() / kFold);
		crossValidationSummary.setAreaUnderROC(evaluationResult.getAreaUnderROC() / kFold);

		return crossValidationSummary;
	}

	public synchronized static void checkPreviousEvaluations(EvaluationResult evaluationResult,
			ThresholdFinderEntity thresholdFinderEntity, double classifierThreshold) {
		if (evaluationResult.getF_measure() > thresholdFinderEntity.getMaxFMeasure()) {
			thresholdFinderEntity.setMaxFMeasure(evaluationResult.getF_measure());
			thresholdFinderEntity.setBestThreshold(classifierThreshold);
		}
	}

}
