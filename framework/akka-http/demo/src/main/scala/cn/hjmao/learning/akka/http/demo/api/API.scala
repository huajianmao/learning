package cn.hjmao.learning.akka.http.demo.api

import akka.http.scaladsl.model.StatusCodes
import akka.http.scaladsl.server.Directives._
import de.heikoseeberger.akkahttpcirce.FailFastCirceSupport

import cn.hjmao.learning.akka.http.demo.model.UserEntity
import cn.hjmao.learning.akka.http.demo.service.{AuthService, UserService}
import cn.hjmao.learning.akka.http.demo.utils.CorsSupport

import io.circe.generic.auto._
import io.circe.syntax._

import scala.concurrent.ExecutionContext

/**
 * Created by hjmao on 17-5-10.
 */

class API(userService: UserService, authService: AuthService)
         (implicit executionContext: ExecutionContext) extends CorsSupport {
  val userRouter = new UserAPI(authService, userService)
  val authRouter = new AuthAPI(authService)

  val routes = pathPrefix("v1") { pathPrefix("api") {
    corsHandler {
      authRouter.routes ~
      userRouter.routes
    }
  }}
}

class AuthAPI(val authService: AuthService)
             (implicit executionContext: ExecutionContext) extends FailFastCirceSupport with SecurityDirectives {

  import StatusCodes._
  import authService._

  val routes = pathPrefix("auth") {
    path("signin") {
      pathEndOrSingleSlash {
        post {
          entity(as[UsernamePassword]) { usernamePassword =>
            complete(signin(usernamePassword.username, usernamePassword.password).map(_.asJson))
          }
        }
      }
    } ~
    path("signup") {
      pathEndOrSingleSlash {
        post {
          entity(as[UserEntity]) { userEntity =>
            complete(Created -> signup(userEntity).map(_.asJson))
          }
        }
      }
    }
  }

  private case class UsernamePassword(username: String, password: String)
}

class UserAPI(val authService: AuthService, userService: UserService)
             (implicit executionContext: ExecutionContext) extends FailFastCirceSupport with SecurityDirectives {
  import userService._
  val routes = pathPrefix("users") {
    pathEndOrSingleSlash {
      get {
        complete(getUsers().map(_.asJson))
      }
    }
  }
}
