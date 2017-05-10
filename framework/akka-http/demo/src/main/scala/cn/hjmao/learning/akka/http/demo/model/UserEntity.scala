package cn.hjmao.learning.akka.http.demo.model

/**
 * Created by hjmao on 17-5-10.
 */
case class UserEntity(username: String, password: String) {
  require(!username.isEmpty, "username.empty")
  require(!password.isEmpty, "password.empty")
}

case class UserEntityUpdate(username: Option[String] = None, password: Option[String] = None) {
  def merge(user: UserEntity): UserEntity = {
    UserEntity(username.getOrElse(user.username), password.getOrElse(user.password))
  }
}
