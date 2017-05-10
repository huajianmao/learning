package cn.hjmao.learning.akka.http.demo.model.db

import com.zaxxer.hikari.{HikariConfig, HikariDataSource}

/**
 * Created by hjmao on 17-5-10.
 */
class DataSource(classname: String, url: String, user: String, password: String) {
  private val config = new HikariConfig()
  config.setJdbcUrl(url)
  config.setUsername(user)
  config.setPassword(password)

  private val datasource = new HikariDataSource(config)

  val driver = slick.jdbc.MySQLProfile
  import driver.api._
  val db = Database.forDataSource(datasource, None)
  db.createSession()
}
