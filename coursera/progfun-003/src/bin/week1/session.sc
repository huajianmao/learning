package week1

object session {
  println("Welcome to the Scala worksheet")       //> Welcome to the Scala worksheet
  1 + 3                                           //> res0: Int(4) = 4
  def sqrt(x: Double): Double = {
    def abs(x: Double): Double = if (x >= 0) x else -x

    def sqrtIter(guess: Double): Double =
      if (isGoodEnough(guess)) guess
      else sqrtIter(improve(guess))

    def isGoodEnough(guess: Double): Boolean =
      if (abs(guess * guess - x) / x < 0.0001) true
      else false

    def improve(guess: Double): Double =
      return (guess + x / guess) / 2

    sqrtIter(1.0)
  }                                               //> sqrt: (x: Double)Double
}