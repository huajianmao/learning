package cn.hjmao.learning.akka.http.demo.config

import com.typesafe.config.ConfigFactory

/**
 * Created by hjmao on 17-5-10.
 */
trait AppConf {
  private val config = ConfigFactory.load()
  private val http = config.getConfig("http")

  val host = http.getString("interface")
  val port = http.getInt("port")
}
