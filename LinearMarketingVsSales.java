

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.ml.regression.LinearRegressionModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class LinearMarketingVsSales {

	public static void main(String[] args) {
		
		Logger.getLogger("org").setLevel(Level.ERROR);
		Logger.getLogger("akka").setLevel(Level.ERROR);
		
		SparkSession spark = new SparkSession.Builder()
				.appName("LinearRegressionExample")
				.master("local")
				.getOrCreate();
		
		Dataset<Row> markVsSalesDf = spark.read()
			.option("header", "true")
			.option("inferSchema", "true")
			.format("csv")
			.load("/Users/akshaykokane/Documents/workspace/SparkWithJava/marketing-vs-sales.csv");
		

		Dataset<Row> mldf = markVsSalesDf.withColumnRenamed("sales", "label")
							.select("label", "marketing_spend");
		
		String[] featureCols = {"marketing_spend"};
		VectorAssembler assembler = new VectorAssembler()
									.setInputCols(featureCols)
									.setOutputCol("features");
		
		Dataset<Row> lblFeaturesDf = assembler.transform(mldf).select("label", "features");
		lblFeaturesDf = lblFeaturesDf.na().drop(); //drop any rows will null value
		
		//this is formal ml algorithm expects
		//lblFeaturesDf.show();
		
		//create liner regression model
		
		LinearRegression lr = new LinearRegression();
		
		LinearRegressionModel learningModel = lr.fit(lblFeaturesDf);
		
		learningModel.summary().predictions().show();
		
		/*
		 * OUTPUT
+--------+---------+------------------+
|   label| features|        prediction|
+--------+---------+------------------+
|280000.0|[30000.0]|211589.25864568225|
|279100.0|[40000.0]|261461.92379374604|
|220000.0|[40000.0]|261461.92379374604|
|168000.0|[27000.0]| 196627.4591012631|
|250200.0|[50000.0]|311334.58894180984|
|382800.0|[60000.0]|361207.25408987363|
|450400.0|[70000.0]| 411079.9192379374|
|412000.0|[70000.0]| 411079.9192379374|
|410500.0|[70000.0]| 411079.9192379374|
|450400.0|[80000.0]| 460952.5843860012|
|505300.0|[90000.0]|  510825.249534065|
+--------+---------+------------------+
		 * 
		 * */
		
		
	}
}
