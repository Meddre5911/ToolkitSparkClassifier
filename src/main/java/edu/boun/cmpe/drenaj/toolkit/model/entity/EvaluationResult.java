package edu.boun.cmpe.drenaj.toolkit.model.entity;

public class EvaluationResult {
	
	private double f_measure;
	private double precision;
	private double recall;
	private double accuracy;
	private double areaUnderROC;

	public double getF_measure() {
		return f_measure;
	}

	public void setF_measure(double f_measure) {
		this.f_measure = f_measure;
	}

	public double getPrecision() {
		return precision;
	}

	public void setPrecision(double precision) {
		this.precision = precision;
	}

	public double getRecall() {
		return recall;
	}

	public void setRecall(double recall) {
		this.recall = recall;
	}

	public double getAccuracy() {
		return accuracy;
	}

	public void setAccuracy(double accuracy) {
		this.accuracy = accuracy;
	}

	public double getAreaUnderROC() {
		return areaUnderROC;
	}

	public void setAreaUnderROC(double areaUnderROC) {
		this.areaUnderROC = areaUnderROC;
	}

	public void addAccuracy(double accuracy) {
		this.accuracy += accuracy;
	}

	public void addRecall(double recall) {
		this.recall += recall;
	}

	public void addPrecision(double precision) {
		this.precision += precision;
	}

	public void addFMeasure(double f_measure) {
		this.f_measure += f_measure;
	}

	public void addAreaUnderROC(double areaUnderROC) {
		this.areaUnderROC += areaUnderROC;
	}

	public void printSummary(double kFold) {
		System.out.println("-------------- Summary ------------- ");
		System.out.println("General Accuracy : " + getAccuracy() / kFold);
		System.out.println("General Precision : " + getPrecision() / kFold);
		System.out.println("General Recall : " + getRecall() / kFold);
		System.out.println("General FMeasure : " + getF_measure() / kFold);
		System.out.println("General ROC Area : " + getAreaUnderROC() / kFold);
	}

}
