package reductions

import scala.annotation._
import org.scalameter._
import common._

object ParallelParenthesesBalancingRunner {

  @volatile var seqResult = false

  @volatile var parResult = false

  val standardConfig = config(
    Key.exec.minWarmupRuns -> 40,
    Key.exec.maxWarmupRuns -> 80,
    Key.exec.benchRuns -> 120,
    Key.verbose -> true
  ) withWarmer(new Warmer.Default)

  def main(args: Array[String]): Unit = {
    val length = 100000000
    val chars = new Array[Char](length)
    val threshold = 10000
    val seqtime = standardConfig measure {
      seqResult = ParallelParenthesesBalancing.balance(chars)
    }
    println(s"sequential result = $seqResult")
    println(s"sequential balancing time: $seqtime ms")

    val fjtime = standardConfig measure {
      parResult = ParallelParenthesesBalancing.parBalance(chars, threshold)
    }
    println(s"parallel result = $parResult")
    println(s"parallel balancing time: $fjtime ms")
    println(s"speedup: ${seqtime / fjtime}")
  }
}

object ParallelParenthesesBalancing {

  /** Returns `true` iff the parentheses in the input `chars` are balanced.
   */
  def balance(chars: Array[Char]): Boolean = {
    def loop(delta: Int, idx: Int): Int = {
      if (delta < 0) Int.MinValue
      else if (idx >= chars.length) delta
      else if (chars(idx) == '(') loop(delta + 1, idx + 1)
      else if (chars(idx) == ')') loop(delta - 1, idx + 1)
      else loop(delta, idx + 1)
    }

    loop(0, 0) == 0
  }

  /** Returns `true` iff the parentheses in the input `chars` are balanced.
   */
  def parBalance(chars: Array[Char], threshold: Int): Boolean = {

    def traverse(idx: Int, until: Int, leftCount: Int, rightCount: Int): (Int, Int) = {
      if (until <= idx) (leftCount, rightCount)
      else if (chars(idx) == '(') traverse(idx + 1, until, leftCount + 1, rightCount)
      else if (chars(idx) == ')' && leftCount > 0) traverse(idx + 1, until, leftCount - 1, rightCount)
      else if (chars(idx) == ')' && leftCount <= 0) traverse(idx + 1, until, leftCount, rightCount + 1)
      else traverse(idx + 1, until, leftCount, rightCount)
    }

    def reduce(from: Int, until: Int): (Int, Int) = {
      if (until - from <= threshold) traverse(from, until, 0, 0)
      else {
        val middle = (until + from) / 2
        val (left, right) = parallel(reduce(from, middle), reduce(middle, until))
        val matched = if (left._1 > right._2) right._2 else left._1
        (left._1 + right._1 - matched, left._2 + right._2 - matched)
      }
    }

    reduce(0, chars.length) == (0, 0)
  }

  // For those who want more:
  // Prove that your reduction operator is associative!

}
