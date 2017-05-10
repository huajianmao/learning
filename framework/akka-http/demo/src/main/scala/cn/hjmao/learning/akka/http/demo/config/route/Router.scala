package cn.hjmao.learning.akka.http.demo.config.route

import akka.http.scaladsl.model.{ContentTypes, HttpEntity}
import akka.http.scaladsl.server.Directives._
import cn.hjmao.learning.akka.http.demo.utils.CorsSupport

import scala.concurrent.ExecutionContext

/**
 * Created by hjmao on 17-5-10.
 */

class Router(implicit executionContext: ExecutionContext) extends CorsSupport {
  val userRouter = new UserRouter()
  val authRouter = new AuthRouter()
  val testRouter = new TestRouter()

  val routes = pathPrefix("v1") { corsHandler {
    testRouter.routes ~
    userRouter.routes ~
    authRouter.routes
  }}
}


class AuthRouter(implicit executionContext: ExecutionContext) {
  val routes = pathPrefix("auth") {
    get {
      complete(
        HttpEntity(ContentTypes.`text/html(UTF-8)`, "<h1>Say hello in AuthRouter</h1>")
      )
    }
  }
}

class UserRouter(implicit executionContext: ExecutionContext) {
  val routes = pathPrefix("user") {
    get {
      complete(
        HttpEntity(ContentTypes.`text/html(UTF-8)`, "<h1>Say hello in UserRouter</h1>")
      )
    }
  }
}

class TestRouter(implicit executionContext: ExecutionContext) {
  val routes = pathPrefix("test") {
    get {
      complete(
        HttpEntity(ContentTypes.`text/html(UTF-8)`, "<h1>Say hello in TestRouter</h1>")
      )
    }
  }
}
