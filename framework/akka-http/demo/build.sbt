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

    "org.scalatest" %% "scalatest" % scalaTestVersion % "test"
  )
}
