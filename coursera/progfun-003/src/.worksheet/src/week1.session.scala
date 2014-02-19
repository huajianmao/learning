package week1

object session {;import org.scalaide.worksheet.runtime.library.WorksheetSupport._; def main(args: Array[String])=$execute{;$skip(75); 
  println("Welcome to the Scala worksheet");$skip(8); val res$0 = 
  1 + 3;System.out.println("""res0: Int(4) = """ + $show(res$0));$skip(424); 
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
  };System.out.println("""sqrt: (x: Double)Double""")}
}
