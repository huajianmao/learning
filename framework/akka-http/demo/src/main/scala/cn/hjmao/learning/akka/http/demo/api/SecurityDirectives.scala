package cn.hjmao.learning.akka.http.demo.api

/**
 * Created by hjmao on 17-5-11.
 */
import akka.http.scaladsl.server.directives.{BasicDirectives, FutureDirectives, HeaderDirectives, RouteDirectives}
import akka.http.scaladsl.server.Directive1
import cn.hjmao.learning.akka.http.demo.model.UserEntity
import cn.hjmao.learning.akka.http.demo.service.AuthService

trait SecurityDirectives {

  import BasicDirectives._
  import HeaderDirectives._
  import RouteDirectives._
  import FutureDirectives._

  def authenticate: Directive1[UserEntity] = {
    headerValueByName("Token").flatMap { token =>
      onSuccess(authService.authenticate(token)).flatMap {
        case Some(user) => provide(user)
        case None       => reject
      }
    }
  }

  protected val authService: AuthService
}
