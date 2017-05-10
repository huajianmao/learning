package cn.hjmao.learning.akka.http.demo.model

import java.util.UUID

/**
 * Created by hjmao on 17-5-10.
 */
case class TokenEntity(id: Option[Long] = None,
                       username: String,
                       token: String = UUID.randomUUID().toString.replaceAll("-", ""))
