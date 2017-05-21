package edu.boun.cmpe.drenaj.toolkit.deprecated;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;

import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.RandomForestClassificationModel;
import org.apache.spark.ml.classification.RandomForestClassifier;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.IndexToString;
import org.apache.spark.ml.feature.LabeledPoint;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.StringIndexerModel;
import org.apache.spark.ml.feature.VectorIndexer;
import org.apache.spark.ml.feature.VectorIndexerModel;
import org.apache.spark.ml.linalg.Vectors;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.tuning.CrossValidator;
import org.apache.spark.ml.tuning.CrossValidatorModel;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.mllib.util.MLUtils;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SQLContext;
import org.apache.spark.sql.SparkSession;

public class OrganizedBehaviourCrossValidatorClassifier {

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

							LabeledPoint pos = new LabeledPoint(Double.valueOf(split[0]),
									Vectors.dense(featuresDoubleValues));

							labeledPoints.add(pos);
						}

						return labeledPoints.iterator();
					}
				});

		labeledDataRDD.collect().forEach(System.out::println);

		Dataset<Row> data = sqlContext.createDataFrame(labeledDataRDD, LabeledPoint.class);

		// Index labels, adding metadata to the label column.
		// Fit on whole dataset to include all labels in index.
		StringIndexerModel labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel")
				.fit(data);

		// Automatically identify categorical features, and index them.
		// Set maxCategories so features with > 4 distinct values are treated as
		// continuous.
		VectorIndexerModel featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures")
				.setMaxCategories(4).fit(data);

		// Train a RandomForest model.
		RandomForestClassifier rf = new RandomForestClassifier().setLabelCol("indexedLabel")
				.setFeaturesCol("indexedFeatures");

		// Convert indexed labels back to original labels.
		IndexToString labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel")
				.setLabels(labelIndexer.labels());

		// Chain indexers and forest in a Pipeline
		Pipeline pipeline = new Pipeline()
				.setStages(new PipelineStage[] { labelIndexer, featureIndexer, rf, labelConverter });

		// Select (prediction, true label) and compute test error
		MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
				.setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy");

		int[] maxDepthArray = { 2, 4, 6 };
		int[] maxBinsArray = { 20, 60 };
		int[] numTreesArray = { 5, 20 };

		ParamMap[] paramGrid = new ParamGridBuilder().addGrid(rf.maxDepth(), maxDepthArray)
				.addGrid(rf.maxBins(), maxBinsArray).addGrid(rf.numTrees(), numTreesArray).build();

		CrossValidator crossValidator = new CrossValidator()
				// ml.Pipeline with ml.classification.RandomForestClassifier
				.setEstimator(pipeline)
				// ml.evaluation.MulticlassClassificationEvaluator
				.setEvaluator(evaluator).setEstimatorParamMaps(paramGrid).setNumFolds(10);

		CrossValidatorModel crossValidatorModel = crossValidator.fit(data);

		double[] avgMetrics = crossValidatorModel.avgMetrics();

		for (int i = 0; i < avgMetrics.length; i++) {
			System.out.println("Fold " + i + " accuracy : " + avgMetrics[i]);
		}

		jsc.close();

	}

}
