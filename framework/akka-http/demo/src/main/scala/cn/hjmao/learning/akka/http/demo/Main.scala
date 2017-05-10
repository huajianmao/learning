package cn.hjmao.learning.akka.http.demo

import akka.actor.ActorSystem
import akka.event.{Logging, LoggingAdapter}
import akka.http.scaladsl.Http
import akka.stream.ActorMaterializer
import cn.hjmao.learning.akka.http.demo.api.API
import cn.hjmao.learning.akka.http.demo.model.db.DataSource
import cn.hjmao.learning.akka.http.demo.service.{AuthService, UserService}
import cn.hjmao.learning.akka.http.demo.utils.AppConfig

import scala.io.StdIn

object Main extends App with AppConfig {
  implicit val system = ActorSystem("akka-http-learning")
  implicit val materializer = ActorMaterializer()
  implicit val executionContext = system.dispatcher
  implicit val log: LoggingAdapter = Logging(system, getClass)

  val datasource = new DataSource("", dbUrl, dbUser, dbPassword)
  val userService = new UserService(datasource)
  val authService = new AuthService(datasource)(userService)
  val api = new API(userService, authService)
  val bindingFuture = Http().bindAndHandle(api.routes, host, port)

  println(s"Server online at http://${host}:${port}/\nPress RETURN to stop...")
  StdIn.readLine()

  bindingFuture.flatMap(_.unbind()).onComplete(_ => {datasource.db.close(); system.terminate()})
}
