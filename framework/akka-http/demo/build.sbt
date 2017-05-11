name := "akka-http-learning"
organization := "io.github.huajianmao"
version := "0.1"
scalaVersion := "2.11.8"

scalacOptions := Seq("-unchecked", "-deprecation", "-encoding", "utf8")

libraryDependencies ++= {
  val akkaHttpVersion = "10.0.2"
  val scalaTestVersion = "3.0.1"
  val circeVersion = "0.8.0"

  Seq(
    "com.typesafe.akka" %% "akka-http" % akkaHttpVersion,
    "com.typesafe.akka" %% "akka-http-spray-json" % akkaHttpVersion,
    "de.heikoseeberger" %% "akka-http-circe" % "1.15.0",

    "com.typesafe.slick" %% "slick" % "3.2.0",

    "org.slf4j" % "slf4j-nop" % "1.7.25",

    "io.circe" %% "circe-core" % circeVersion,
    "io.circe" %% "circe-generic" % circeVersion,
    "io.circe" %% "circe-parser" % circeVersion,

    "com.zaxxer" % "HikariCP" % "2.6.1",
    "mysql" % "mysql-connector-java" % "6.0.6",

    "org.scalatest" %% "scalatest" % scalaTestVersion % "test"
  )
}
