name := "spark-deep-learning"

version := "1.1"

scalaVersion := "2.11.11"

libraryDependencies += "org.nd4j" % "nd4j-native-platform" % "0.9.1"

libraryDependencies += "org.deeplearning4j" % "dl4j-spark_2.11" % "0.9.1_spark_2"

libraryDependencies += "org.nd4j" % "nd4j-api" % "0.9.1"

libraryDependencies += "com.typesafe" % "config" % "1.3.1"

def latestScalafmt = "1.1.0"
commands += Command.args("scalafmt", "Run scalafmt cli.") {
  case (state, args) =>
    val Right(scalafmt) =
      org.scalafmt.bootstrap.ScalafmtBootstrap.fromVersion(latestScalafmt)
    scalafmt.main("--non-interactive" +: args.toArray)
    state
}

enablePlugins(JavaAppPackaging)
