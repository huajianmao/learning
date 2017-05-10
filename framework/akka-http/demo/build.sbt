name := "akka-http-learning"
organization := "io.github.huajianmao"
version := "0.1"
scalaVersion := "2.11.8"

scalacOptions := Seq("-unchecked", "-deprecation", "-encoding", "utf8")

libraryDependencies ++= {
  val akkaHttpVersion = "10.0.2"
  val scalaTestVersion = "3.0.1"
  Seq(
    "com.typesafe.akka" %% "akka-http" % akkaHttpVersion,
    "com.typesafe.akka" %% "akka-http-spray-json" % akkaHttpVersion,
    "com.typesafe.slick" %% "slick" % "3.2.0",

    "com.zaxxer" % "HikariCP" % "2.6.1",
    "mysql" % "mysql-connector-java" % "6.0.6",

    "org.scalatest" %% "scalatest" % scalaTestVersion % "test"
  )
}
