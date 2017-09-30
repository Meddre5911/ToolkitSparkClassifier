package edu.boun.cmpe.drenaj.toolkit.model.concurrency;

import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.rdd.RDD;

import edu.boun.cmpe.drenaj.toolkit.model.entity.EvaluationResult;
import edu.boun.cmpe.drenaj.toolkit.model.entity.ThresholdFinderEntity;
import edu.boun.cmpe.drenaj.toolkit.model.util.MetricsUtil;
import edu.boun.cmpe.drenaj.toolkit.model.util.SVMUtil;
import scala.Tuple2;

public class SVMClassifierTask implements Runnable {

	private double threshold;
	private double kFoldCount;
	private ThresholdFinderEntity thresholdFinderEntity;
	private Tuple2<RDD<LabeledPoint>, RDD<LabeledPoint>>[] kFold;

	public SVMClassifierTask(double threshold, ThresholdFinderEntity thresholdFinderEntity,
			Tuple2<RDD<LabeledPoint>, RDD<LabeledPoint>>[] kFold) {
		this.threshold = threshold;
		this.thresholdFinderEntity = thresholdFinderEntity;
		this.kFold = kFold;
		this.kFoldCount = kFold.length;
	}

	@Override
	public void run() {
		EvaluationResult evaluationResult = SVMUtil.evaluateCrossValidationWithThreshold(kFold, new StringBuilder(),
				threshold);
		EvaluationResult crossValidationSummary = MetricsUtil.getSummaryOfCrossValidation(evaluationResult, kFoldCount);
		MetricsUtil.checkPreviousEvaluations(crossValidationSummary, thresholdFinderEntity, threshold);

	}

}
