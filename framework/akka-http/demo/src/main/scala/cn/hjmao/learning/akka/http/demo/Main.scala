package cn.hjmao.learning.akka.http.demo

import akka.actor.ActorSystem
import akka.http.scaladsl.Http
import akka.stream.ActorMaterializer
import cn.hjmao.learning.akka.http.demo.config.AppConf
import cn.hjmao.learning.akka.http.demo.config.route.Router

import scala.io.StdIn

object Main extends App with AppConf {
  implicit val system = ActorSystem("my-system")
  implicit val materializer = ActorMaterializer()
  implicit val executionContext = system.dispatcher

  val router = new Router()
  val bindingFuture = Http().bindAndHandle(router.routes, host, port)

  println(s"Server online at http://${host}:${port}/\nPress RETURN to stop...")
  StdIn.readLine()

  bindingFuture.flatMap(_.unbind()).onComplete(_ => system.terminate())
}
