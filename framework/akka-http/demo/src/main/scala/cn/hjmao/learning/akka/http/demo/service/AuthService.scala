package cn.hjmao.learning.akka.http.demo.service


import cn.hjmao.learning.akka.http.demo.model.db.DataSource

import scala.concurrent.ExecutionContext

/**
 * Created by hjmao on 17-5-10.
 */
class AuthService(val datasource: DataSource)
                 (userService: UserService)
                 (implicit executionContext: ExecutionContext) {

}
