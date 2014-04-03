我理解的Spark -- Overview
-------------------------

> 心血来潮，突然想读读Spark的源代码，
但是由于功力有限，读起来略感力不从心，
所以打算从零开始，慢慢啃这块骨头。
为了能够把自己看Spark过程中从零到整的一些理解作记录，
同时也想记录一下看自己能够坚持到什么时候，看自己在最后的时候能够理解Spark多少，
为此才开此分支，便于记录。
欢迎各种吐槽，一来可以帮我提高表达，二来帮忙减少对读者的祸害。

> 如果提供的资料有涉及到版权问题，麻烦提醒，我会及时删除。[Contact Me](mailto:huajianmao@gmail.com)


# 资料索引

## Spark Papers and Book
* *M. Zaharia*. [An Architecture for Fast and General Data Processing on Large Clusters](./doc/EECS-2014-12.pdf). **(PhD Disseration), University of California at Berkeley**, February 2014.
* *M. Zaharia, M. Chowdhury, T. Das, A. Dave, J. Ma, M. McCauley, M.J. Franklin, S. Shenker, I. Stoica*. [Resilient Distributed Datasets: A Fault-Tolerant Abstraction for In-Memory Cluster Computing](./doc/nsdi_spark.pdf), **NSDI 2012**, April 2012.
* *M. Zaharia, M. Chowdhury, M.J. Franklin, S. Shenker and I. Stoica*. [Spark: Cluster Computing with Working Sets](./doc/hotcloud_spark.pdf), **HotCloud 2010**, June 2010.

* *Karau, Holden*. [Fast Data Processing With Spark](./doc/Fast_Data_Processing_with_Spark.pdf). **Packt Publishing Ltd**, 2013. [Book, you may find the PDF version [here](http://ishare.iask.sina.com.cn/f/67674105.html)]

## Talks
* The Intro to Spark Internals Meetup talk ([Video](https://www.youtube.com/watch?v=49Hr5xZyTEA) or [here](http://pan.baidu.com/s/1kTHStAf), [PPT slides](./doc/dev-meetup-dec-2012.pptx)) is a good introduction to the internals
* [Spark Summit](http://spark-summit.org) ([2013](http://spark-summit.org/2013)|[2014](http://spark-summit.org/2014))
* [AMP Camp – Big Data Bootcamps](http://ampcamp.berkeley.edu/) ([AMP Camp 1](http://ampcamp.berkeley.edu/amp-camp-one-berkeley-2012/)|[AMP Camp 2](http://ampcamp.berkeley.edu/amp-camp-two-strata-2013/)|[AMP Camp 3](http://ampcamp.berkeley.edu/3/)|[AMP Camp 4](http://ampcamp.berkeley.edu/4/))
* [Apache Spark Channel on YouTube](https://www.youtube.com/channel/UCRzsq7k4-kT-h3TDUBQ82-w)

## Important Sites
* [Wikipedia - Apache Spark](http://en.wikipedia.org/wiki/Apache_spark)
* [Apache Spark website](http://spark.apache.org/)
* [Spark on Github](https://github.com/apache/spark)
* [Spark Wiki Homepage](https://cwiki.apache.org/confluence/display/SPARK/Wiki+Homepage)
* [Dev Maillist](http://apache-spark-developers-list.1001551.n3.nabble.com/)
* [Spark JIRA](https://spark-project.atlassian.net/browse/SPARK)


## Blogs
* [赛赛的网络日志-记录点滴...](http://jerryshao.me/tags.html#spark-ref)
* [Apache Spark - fxjwind](http://www.cnblogs.com/fxjwind/category/518904.html)
* [baishuo491的博客](http://baishuo491.iteye.com/blog/search?query=spark) on [Github](https://github.com/baishuo)

## 微博上的Spark
* [@hashjoin](http://weibo.com/hashjoin) on [Github](), *Databricks大数据公司创始人之一，UC Berkeley AMPLab/Spark/Shark*
* [@连城404](http://weibo.com/lianchengzju) on [Github](https://github.com/liancheng), *China Intel IoT joint Labs, and Databricks*
* [@明风Andy](http://weibo.com/mingfengandy), *淘宝技术部，数据挖掘与计算团队负责人*
* [@CrazyJvm](http://weibo.com/476691290)(Chen Chao) on [Github](https://github.com/CrazyJvm), *皮皮网，担任数据平台负责人*
* [@尹绪森](http://weibo.com/yinxusen) on [Github](https://github.com/yinxusen), *Intel Labs, Beijing, China*
* [@CodingCat](http://weibo.com/codingcat) (Nan Zhu) on [Github](https://github.com/codingcat), *Faimdata & McGill University, Montreal, Canada*
* [@Andrew-Xia](http://weibo.com/u/1410938285)(Xia Junluan) on [Github](https://github.com/xiajunluan), *Technical Architect and Engineering Manager at Intel Corporation, Shanghai*

## Scala
* [Scala Document](http://www.scala-lang.org/documentation/)
* [Scala School](http://twitter.github.io/scala_school/) from Twitter
* [Effective Scala](http://twitter.github.io/effectivescala/) Twitter's "best practices" for Scala. Useful for understanding idioms in Twitter's code.
* [Functional Programming Principles in Scala](https://www.coursera.org/course/progfun). This is a course about functional programming given by Martin Odersky himself. 
* [Principles of Reactive Programming](https://www.coursera.org/course/reactive). This is a course about concurrency and event-based asynchronous programming in Scala. 
* [Independent Courseware](http://scalacourses.com/). Take a series of online Scala courses ranging from beginner to advanced for a fee.
* [Try Scala In Your Browser!](http://www.simplyscala.com/) Simply Scala is a web site where you can interactively run the Scala interpreter in your browser! 
