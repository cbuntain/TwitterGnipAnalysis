package edu.umd.cs.hcil.analytics

/**
  * Created by cbuntain on 8/2/17.
  */
object MultiTest {

  def main(args : Array[String]) : Unit = {
    val twitter_test = edu.umd.cs.hcil.spark.analytics.KLDivergence
    val gnip_test = edu.umd.cs.hcil.spark.gnip.RuleSelector

    println("Twitter: " + twitter_test.toString)
    println("Gnip: " + gnip_test.toString)
  }

}
