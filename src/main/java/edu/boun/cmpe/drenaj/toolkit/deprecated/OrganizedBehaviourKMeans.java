package edu.boun.cmpe.drenaj.toolkit.deprecated;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.ml.feature.LabeledPoint;
import org.apache.spark.ml.linalg.Vectors;
import org.apache.spark.mllib.clustering.KMeans;
import org.apache.spark.mllib.clustering.KMeansModel;
import org.apache.spark.mllib.linalg.Vector;

public class OrganizedBehaviourKMeans {

	public static void main(String[] args) {

		SparkConf conf = new SparkConf().setAppName("JavaKMeansExample").setMaster("local");
		JavaSparkContext jsc = new JavaSparkContext(conf);

		// $example on$
		// Load and parse data
		JavaRDD<String> data = jsc.textFile(
				"/Users/erdembegenilmis/Desktop/workspaces/workspaceDirenaj/ToolkitSparkClassifier/data/organizedBehaviour/extractedFeatures_20170306183548454.csv",
				1);
		JavaRDD<LabeledPoint> labeledDataRDD = data.flatMap(new FlatMapFunction<String, LabeledPoint>() {
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
		
		JavaRDD<Vector> parsedData = data.flatMap(new FlatMapFunction<String, Vector>() {
			@Override
			public Iterator<Vector> call(String line) throws Exception {
				List<Vector> featuresOfLine = new ArrayList<>();
				if (!line.startsWith("ClassId")) {

					String[] split = line.split(",");
					String[] featuresStr = Arrays.copyOfRange(split, 1, split.length);
					double[] featuresDoubleValues = Arrays.stream(featuresStr).mapToDouble(Double::parseDouble)
							.toArray();

					featuresOfLine.add(org.apache.spark.mllib.linalg.Vectors.dense(featuresDoubleValues));
				}

				 return featuresOfLine.iterator();
			}
		});

		// Cluster the data into two classes using KMeans
		int numClusters = 2;
		int numIterations = 100;
		KMeansModel clusters = KMeans.train(parsedData.rdd(), numClusters, numIterations);

		JavaRDD<Integer> predict = clusters.predict(parsedData);
		System.out.println("Predicted : ");
		predict.collect().forEach(System.out::println);

		System.out.println("Cluster centers:");
		for (Vector center : clusters.clusterCenters()) {
			System.out.println(" " + center);
		}
		double cost = clusters.computeCost(parsedData.rdd());
		System.out.println("Cost: " + cost);

		// Evaluate clustering by computing Within Set Sum of Squared Errors
		double WSSSE = clusters.computeCost(parsedData.rdd());
		System.out.println("Within Set Sum of Squared Errors = " + WSSSE);

		jsc.stop();
	}

}
