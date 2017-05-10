package cn.hjmao.learning.akka.http.demo.model.db

import cn.hjmao.learning.akka.http.demo.model.TokenEntity

/**
 * Created by hjmao on 17-5-10.
 */
trait TokenEntityTable extends UserEntityTable {
  protected val datasource: DataSource
  import datasource.driver.api._

  class Token(tag: Tag) extends Table[TokenEntity](tag, "token") {
    def id = column[Option[Long]]("id", O.PrimaryKey, O.AutoInc)
    def username = column[String]("username")
    def token = column[String]("token")

    def userFk = foreignKey("USER_FK", username, users)(_.username, onUpdate = ForeignKeyAction.Restrict, onDelete = ForeignKeyAction.Cascade)

    def * = (id, username, token) <> ((TokenEntity.apply _).tupled, TokenEntity.unapply)
  }

  protected val tokens = TableQuery[Token]
}
