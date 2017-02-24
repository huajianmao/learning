
sealed trait List[+A]
case object Nil extends List[Nothing]
case class Cons[+A](head: A, tail: List[A]) extends List[A]

object List {
  def sum(ints: List[Int]): Int = ints match {
    case Nil => 0
    case Cons(head, tail) => head + sum(tail)
  }

  def product(ds: List[Double]): Double = ds match {
    case Nil => 1.0
    case Cons(0.0, _) => 0.0
    case Cons(x, xs) => x * product(xs)
  }

  def drop[A](l: List[A], n: Int): List[A] = {
    def loop(xs: List[A], n: Int): List[A] = {
      if (n <= 0) xs
      else {
        xs match {
          case Nil => Nil
          case Cons(x, xs) => loop(xs, n-1)
        }
      }
    }

    loop(l, n)
  }

  def apply[A](as: A*): List[A] = 
    if (as.isEmpty) Nil
    else Cons(as.head, apply(as.tail: _*))
}
