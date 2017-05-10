package cn.hjmao.learning.akka.http.demo.api

import akka.http.scaladsl.model.{ContentTypes, HttpEntity}
import akka.http.scaladsl.server.Directives._
import cn.hjmao.learning.akka.http.demo.service.{AuthService, UserService}
import cn.hjmao.learning.akka.http.demo.utils.CorsSupport

import scala.concurrent.ExecutionContext

/**
 * Created by hjmao on 17-5-10.
 */

class API(userService: UserService, authService: AuthService)
         (implicit executionContext: ExecutionContext) extends CorsSupport {
  val userRouter = new UserAPI(userService)
  val authRouter = new AuthAPI(authService)
  val testRouter = new TestAPI()

  val routes = pathPrefix("v1") {
    pathPrefix("api") {
      corsHandler {
        testRouter.routes ~
        userRouter.routes ~
        authRouter.routes
      }
    }
  }
}


class UserAPI(userservice: UserService)(implicit executionContext: ExecutionContext) {
  import userservice._
  val routes = pathPrefix("user") {
    pathEndOrSingleSlash {
      get {
        complete(
          HttpEntity(ContentTypes.`text/html(UTF-8)`, "<h1>Going to list users</h1>")
        )
      }
    }
  }
}

class AuthAPI(authService: AuthService)(implicit executionContext: ExecutionContext) {
  val routes = pathPrefix("auth") {
    get {
      complete(
        HttpEntity(ContentTypes.`text/html(UTF-8)`, "<h1>Say hello in AuthRouter</h1>")
      )
    }
  }
}

class TestAPI()(implicit executionContext: ExecutionContext) {
  val routes = pathPrefix("test") {
    get {
      complete(
        HttpEntity(ContentTypes.`text/html(UTF-8)`, "<h1>Say hello in TestRouter</h1>")
      )
    }
  }
}
