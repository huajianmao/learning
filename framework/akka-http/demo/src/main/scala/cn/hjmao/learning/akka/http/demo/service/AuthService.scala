package cn.hjmao.learning.akka.http.demo.service


import cn.hjmao.learning.akka.http.demo.model.{TokenEntity, UserEntity}
import cn.hjmao.learning.akka.http.demo.model.db.{DataSource, TokenEntityTable}

import scala.concurrent.{ExecutionContext, Future}

/**
 * Created by hjmao on 17-5-10.
 */
class AuthService(val datasource: DataSource)
                 (userService: UserService)
                 (implicit executionContext: ExecutionContext) extends TokenEntityTable {
  import datasource._
  import datasource.driver.api._

  def signin(username: String, password: String): Future[Option[TokenEntity]] = {
    db.run(users.filter(u => u.username === username).result).flatMap { users =>
      users.find(user => user.password == password) match {
        case Some(user) => db.run(tokens.filter(_.username === user.username).result.headOption).flatMap {
          case Some(token) => Future.successful(Some(token))
          case None        => createToken(user).map(token => Some(token))
        }
        case None => Future.successful(None)
      }
    }
  }

  def signup(newUser: UserEntity): Future[TokenEntity] = {
    userService.createUser(newUser).flatMap(user => createToken(user))
  }

  def authenticate(token: String): Future[Option[UserEntity]] = {
    db.run((for {
      token <- tokens.filter(_.token === token)
      user <- users.filter(_.username === token.username)
    } yield user).result.headOption)
  }

  def createToken(user: UserEntity): Future[TokenEntity] = {
    db.run(tokens returning tokens += TokenEntity(username = user.username))
  }
}
