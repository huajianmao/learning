package cn.hjmao.learning.akka.http.demo.utils

import com.typesafe.config.ConfigFactory

/**
 * Created by hjmao on 17-5-10.
 */
trait AppConfig {
  private val config = ConfigFactory.load()
  private val http = config.getConfig("http")
  private val database = config.getConfig("database")

  val host = http.getString("interface")
  val port = http.getInt("port")

//  val dbClassname = database.getString("classname")
  val dbUrl = database.getString("url")
  val dbUser = database.getString("user")
  val dbPassword = database.getString("password")
}
