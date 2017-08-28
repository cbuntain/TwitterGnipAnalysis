package edu.umd.cs.hcil.analytics

import edu.umd.cs.hcil.spark.gnip.utils.JsonToActivity
import edu.umd.cs.hcil.spark.gnip.utils.ActivityStatusAdapter
import edu.umd.cs.hcil.spark.analytics.GeoExtractorByName
import edu.umd.hcil.vader.SentimentIntensityAnalyzer
import edu.umd.cs.hcil.spark.analytics.ActivityFrequency
import edu.umd.cs.hcil.spark.analytics.utils.TimeScale
import java.util.Date

import edu.umd.cs.hcil.spark.analytics.sentiment.PairwiseModeler
import org.apache.spark.ml.feature.CountVectorizerModel
import org.apache.spark.ml.regression.LinearRegressionModel
import org.apache.spark.sql.SparkSession


/**
  * Created by cbuntain on 8/11/17.
  */
class CountrySentiment {

}

object CountrySentiment {
  def main(args : Array[String]) : Unit = {

    val spark = SparkSession.builder().getOrCreate()
    val sc = spark.sparkContext

    val targetCountry = args(0)
    val sourcePath = args(1)
    val polSourcePath = args(2)
    val regressionModelPath = args(3)
    val cvModelPath = args(4)
    val shapePath = args(5)
    val stopwordPath = args(6)

    val stopwords = scala.io.Source.fromFile(stopwordPath).getLines.toArray

    val dataPath = sourcePath
    println("Reading from: " + dataPath)

    val polDataPath = polSourcePath
    println("Reading political data from: " + polDataPath)

    val countryShapes = GeoExtractorByName.
      generateStatePolyList(shapePath, targetCountry)

    println(countryShapes.length)

    val twitterMsgs = sc.textFile(dataPath)
    println("\t" + "Initial Partition Count: " + twitterMsgs.partitions.size)

    val messageCount = twitterMsgs.count()

    println("Messages: " + messageCount)

    // Convert each JSON line in the file to a status using Gnip4j
    //  Note that not all lines are Activity lines, so we catch any exception
    //  generated during this conversion and set to null since we don't care
    //  about non-status lines.
    val tweets = twitterMsgs.
      filter(line => line.length > 0).
      map(line => (line, JsonToActivity.jsonToStatus(line))).
      filter(activityTup =>
        activityTup._2 != null && activityTup._2.getBody != null)
    val tweetCount = tweets.count()

    println("Tweet Count: " + tweetCount)

    val geoLocatedStatuses = tweets
      .map(activityTuple => (activityTuple._1, ActivityStatusAdapter.createStatus(activityTuple._2)))
      .filter(statusTuple => statusTuple._2 != null && statusTuple._2.getGeoLocation != null)

    val geoTweetCount = geoLocatedStatuses.count()

    println("Parsed Tweet Count: " + geoTweetCount)

    val countryFiltered = GeoExtractorByName.
      filterByGeoShape(geoLocatedStatuses, countryShapes, sc)

    println("In Country: " + countryFiltered.count)

    val timeScale = TimeScale.DAILY

    // For each partition, use VADER to calculate the sentiment of
    //  the tweets in that partition, and save the date, user, and
    //  sentiment.
    val dailyUserSentiments = countryFiltered.
      map(tup => tup._2).
      mapPartitions(iterator => {
        val analyzer = SentimentIntensityAnalyzer.getDefaultAnalyzer

        iterator.map(status => {
          val user = status.getUser.getScreenName
          val postedDate = status.getCreatedAt
          val text = status.getText

          val d = ActivityFrequency.convertTimeToSlice(postedDate, timeScale)
          val sentiment = analyzer.polarity_scores(text).get("compound")

          (d, List((user, sentiment)))
        })
      })

    // Filter out neutral comments
    val nonzeroDailySentiments = dailyUserSentiments.filter(tup => tup._2(0)._2 != 0.0)

    // For each day, bin the sentiment values by user. We don't want
    //  loud users to influence sentiment value a lot
    val dailySentiment = nonzeroDailySentiments.reduceByKey((l, r) => l ++ r).
      mapValues(sentList => {
        val userSentMeans = sentList.groupBy(userSentiment => userSentiment._1).
          mapValues(l => l.map(tup => tup._2)).
          mapValues(l => l.reduce((l, r) => l + r) / l.size.toDouble)

        val mean = userSentMeans.values.reduce((l, r) => l + r) / userSentMeans.values.size
        val sigma = userSentMeans.values.map(l => Math.pow(l - mean, 2)).sum / userSentMeans.values.size

        (mean, sigma)
      }).collect

    for ( tup <- dailySentiment.sortBy(x => x._1) ) {
      println(tup._1.toString + "," + tup._2._1 + "," + tup._2._2)
    }

    // Daily activities
    val datedCounts = ActivityFrequency.
      activityCounter(countryFiltered.map(tup => tup._2), timeScale).
      mapValues(k => k._1).
      collect

    val politicalMsgs = sc.textFile(polDataPath)
    println("\t" + "Initial Partition Count: " + politicalMsgs.partitions.size)

    // Convert each JSON line in the file to a status using Gnip4j
    //  Note that not all lines are Activity lines, so we catch any exception
    //  generated during this conversion and set to null since we don't care
    //  about non-status lines.
    val polTweets = politicalMsgs.
      filter(line => line.length > 0).
      map(line => (line, JsonToActivity.jsonToStatus(line))).
      filter(activityTup =>
        activityTup._2 != null && activityTup._2.getBody != null)
    val polTweetCount = polTweets.count()

    println("Political Tweet Count: " + polTweetCount)

    val polStatuses = polTweets.
      map(activityTup => ActivityStatusAdapter.createStatus(activityTup._2)).
      repartition(16)

    val polStatusCount = polStatuses.count()

    println("Parsed Political Tweet Count: " + polStatusCount)

    // For each partition, use VADER to calculate the sentiment of
    //  the tweets in that partition, and save the date, user, and
    //  sentiment.
    val dailyPolUserSentiments = polStatuses.
      mapPartitions(iterator => {
        val analyzer = SentimentIntensityAnalyzer.getDefaultAnalyzer

        iterator.map(status => {
          val user = status.getUser.getScreenName
          val postedDate = status.getCreatedAt
          val text = status.getText

          val d = ActivityFrequency.convertTimeToSlice(postedDate, timeScale)
          val sentiment = analyzer.polarity_scores(text).get("compound")

          (d, List((user, sentiment)))
        })
      })

    // Filter out neutral comments
    val nonzeroDailyPolSentiments = dailyPolUserSentiments.filter(tup => tup._2(0)._2 != 0.0)

    // For each day, bin the sentiment values by user. We don't want
    //  loud users to influence sentiment value a lot
    val dailyPolSentiment = nonzeroDailyPolSentiments.reduceByKey((l, r) => l ++ r).
      mapValues(sentList => {
        val userSentMeans = sentList.groupBy(userSentiment => userSentiment._1).
          mapValues(l => l.map(tup => tup._2)).
          mapValues(l => l.reduce((l, r) => l + r) / l.size.toDouble)

        val mean = userSentMeans.values.reduce((l, r) => l + r) / userSentMeans.values.size
        val sigma = userSentMeans.values.map(l => Math.pow(l - mean, 2)).sum / userSentMeans.values.size

        (mean, sigma)
      }).collect

    for ( tup <- dailySentiment.sortBy(x => x._1) ) {
      println(tup._1.toString + "," + tup._2._1 + "," + tup._2._2)
    }

    // Use the linear regression model to score text
    val model = LinearRegressionModel.load(regressionModelPath)

    // For implicit conversions from RDDs to DataFrames
    import spark.implicits._

    val polStatusesDf = polStatuses.map(s => (s.getCreatedAt.getTime, s.getUser.getScreenName, s.getText))
      .toDF("date", "user", "text")
    val filteredTokens = PairwiseModeler.tokenFilter(polStatusesDf, "text", stopwords)
    val cvModel = CountVectorizerModel.load(cvModelPath)
    val features = cvModel.transform(filteredTokens).
      select("date", "user", "features").
      cache()


    val polLRSent = model.transform(features)
    val dailyPolLRUserSentiments = polLRSent
      .select("date", "user", "prediction")
      .rdd
      .map(r => {
        val d = ActivityFrequency.convertTimeToSlice(new Date(r.getLong(0)), timeScale)
        (d, List((r.getString(1), r.getDouble(2))))
      })

    // For each day, bin the sentiment values by user. We don't want
    //  loud users to influence sentiment value a lot
    val dailyPolLRSentiment = dailyPolLRUserSentiments.reduceByKey((l, r) => l ++ r).
      mapValues(sentList => {
        val userSentMeans = sentList.groupBy(userSentiment => userSentiment._1).
          mapValues(l => l.map(tup => tup._2)).
          mapValues(l => l.reduce((l, r) => l + r) / l.size.toDouble)

        val mean = userSentMeans.values.reduce((l, r) => l + r) / userSentMeans.values.size
        val sigma = userSentMeans.values.map(l => Math.pow(l - mean, 2)).sum / userSentMeans.values.size

        (mean, sigma)
      }).collect

    for ( tup <- dailyPolLRSentiment.sortBy(x => x._1) ) {
      println(tup._1.toString + "," + tup._2._1 + "," + tup._2._2)
    }

    // Daily activities
    val datedPolCounts = ActivityFrequency.
      activityCounter(polStatuses, timeScale).
      mapValues(k => k._1).
      collect

    val freqMap = datedCounts.toMap
    val sentMap = dailySentiment.toMap
    val polFreqMap = datedPolCounts.toMap
    val polSentMap = dailyPolSentiment.toMap
    val polLRSentMap = dailyPolLRSentiment.toMap

    val outputFile = new java.io.PrintWriter(new java.io.File(("%s.csv").format(targetCountry)))

    val header = "date,frequency,geo_sentiment_mean,geo_sentiment_var,pol_freq,pol_sentiment_mean,pol_sentiment_var,pol_pair_mean,pol_pair_var"
    println(header)
    outputFile.println(header)

    for ( date <- freqMap.keys.toList.sortBy(x => x) ) {

      val freq = freqMap(date)
      val sent = sentMap.getOrElse(date, (0.0, 0.0))
      val polFreq = polFreqMap.getOrElse(date, 0)
      val polSent = polSentMap.getOrElse(date, (0.0, 0.0))
      val polLRSent = polLRSentMap.getOrElse(date, (0.0, 0.0))

      val record = "%s,%d,%f,%f,%d,%f,%f,%f,%f".format(
        date.toString, freq, sent._1, sent._2, polFreq, polSent._1, polSent._2, polLRSent._1, polLRSent._2)

      println(record)
      outputFile.println(record)

    }

    outputFile.close()
  }
}