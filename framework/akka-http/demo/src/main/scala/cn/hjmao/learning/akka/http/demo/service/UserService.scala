package cn.hjmao.learning.akka.http.demo.service


import cn.hjmao.learning.akka.http.demo.model.{UserEntity, UserEntityUpdate}
import cn.hjmao.learning.akka.http.demo.model.db.{DataSource, UserEntityTable}

import scala.concurrent.{ExecutionContext, Future}

/**
 * Created by hjmao on 17-5-10.
 */
class UserService(val datasource: DataSource)
                 (implicit executionContext: ExecutionContext) extends UserEntityTable {
  import datasource._
  import datasource.driver.api._

  def getUsers(): Future[Seq[UserEntity]] = db.run(users.result)

  def getUserByUsername(username: String): Future[Option[UserEntity]] = {
    db.run(users.filter(_.username === username).result.headOption)
  }

  def createUser(user: UserEntity): Future[UserEntity] = {
    db.run(users returning users += user)
  }

  def updateUser(username: String, userUpdate: UserEntityUpdate): Future[Option[UserEntity]] = {
    getUserByUsername(username).flatMap {
      case Some(user) =>
        val updatedUser = userUpdate.merge(user)
        db.run(users.filter(_.username === username).update(updatedUser)).map(_ => Some(updatedUser))
      case None => Future.successful(None)
    }
  }

  def deleteUser(username: String): Future[Int] = {
    db.run(users.filter(_.username === username).delete)
  }
}
