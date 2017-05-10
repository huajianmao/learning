package cn.hjmao.learning.akka.http.demo.model.db

import cn.hjmao.learning.akka.http.demo.model.UserEntity

/**
 * Created by hjmao on 17-5-10.
 */
trait UserEntityTable {
  protected val datasource: DataSource
  import datasource.driver.api._

  class User(tag: Tag) extends Table[UserEntity](tag, "user") {
    def username = column[String]("username", O.PrimaryKey)
    def password = column[String]("password")

    def * = (username, password) <> ((UserEntity.apply _).tupled, UserEntity.unapply)
  }

  protected val users = TableQuery[User]
}
