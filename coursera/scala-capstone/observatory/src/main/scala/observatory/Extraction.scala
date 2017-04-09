package observatory

import java.nio.file.Paths

import org.apache.spark.sql._
import org.apache.spark.sql.types._

import java.time.LocalDate

/**
  * 1st milestone: data extraction
  */
object Extraction {

  import org.apache.spark.sql.SparkSession
  import org.apache.spark.sql.functions._

  val spark: SparkSession =
    SparkSession
      .builder()
      .appName("Capstone")
      .config("spark.master", "local")
      .getOrCreate()

  // For implicit conversions like converting RDDs to DataFrames
  import spark.implicits._

  /** @return The filesystem path of the given resource */
  def fsPath(resource: String): String = {
    // "src/main/resources" + resource
    Paths.get(getClass.getResource(resource).toURI).toString
  }

  def stations(stationsFile: String): DataFrame = {
    val rdd = spark.sparkContext.textFile(fsPath(stationsFile))
    val schema = StructType( List(
      StructField("stn", StringType, nullable=true),
      StructField("wban", StringType, nullable=true),
      StructField("latitude", DoubleType, nullable=true),
      StructField("longitude", DoubleType, nullable=true)
    ))
    val data = rdd.mapPartitionsWithIndex((i, it) => if (i == 0) it.drop(1) else it)
                  .map(_.split(","))
                  .filter(fields => fields(1).trim != "" && fields(2) != "")
                  .map(fields => Row.fromSeq(
                    List(fields(0).trim, fields(1).trim, fields(2).trim.toDouble, fields(3).trim.toDouble)
                  ))
    spark.createDataFrame(data, schema)
  }

  def temperatures(year: Int, temperaturesFile: String): DataFrame = {
    val rdd = spark.sparkContext.textFile(fsPath(temperaturesFile))
    val schema = StructType( List(
      StructField("stn", StringType, nullable=true),
      StructField("wban", StringType, nullable=true),
      StructField("year", IntegerType, nullable=false),
      StructField("month", IntegerType, nullable=true),
      StructField("day", IntegerType, nullable=true),
      StructField("temperature", DoubleType, nullable=true)
    ))
    val data = rdd.mapPartitionsWithIndex((i, it) => if (i == 0) it.drop(1) else it)
                  .map(_.split(","))
                  .map(fields => Row.fromSeq(
                    List(fields(0).trim, fields(1).trim,
                         year, fields(2).trim.toInt, fields(3).trim.toInt,
                         (fields(4).trim.toDouble - 32) / 9 * 5)
                  ))
    spark.createDataFrame(data, schema)
  }

  /**
    * @param year             Year number
    * @param stationsFile     Path of the stations resource file to use (e.g. "/stations.csv")
    * @param temperaturesFile Path of the temperatures resource file to use (e.g. "/1975.csv")
    * @return A sequence containing triplets (date, location, temperature)
    */
  def locateTemperatures(year: Int, stationsFile: String, temperaturesFile: String): Iterable[(LocalDate, Location, Double)] = {
    val stationsDF = stations(stationsFile)
    val temperaturesDF = temperatures(year, temperaturesFile)
    val joined = stationsDF.join(temperaturesDF, stationsDF("stn") === temperaturesDF("stn") && stationsDF("wban") === temperaturesDF("wban"))
    joined.collect().par.map {
      case Row(id: String, year: Int, month: Int, day: Int, latitude: Double, longitude: Double, temperature: Double) =>
        (LocalDate.of(year, month, day), Location(latitude, longitude), temperature)
    }.seq
  }

  /**
    * @param records A sequence containing triplets (date, location, temperature)
    * @return A sequence containing, for each location, the average temperature over the year.
    */
  def locationYearlyAverageRecords(records: Iterable[(LocalDate, Location, Double)]): Iterable[(Location, Double)] = {
    records.par
           .groupBy(_._2)
           .mapValues(r => r.foldLeft(0.0)((t, r1) => t + r1._3) / r.size)
           .seq
  }

}
