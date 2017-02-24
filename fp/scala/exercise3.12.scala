
sealed trait List[+A]
case object Nil extends List[Nothing]
case class Cons[+A](head: A, tail: List[A]) extends List[A]

object List {
  def foldRight[A, B](as: List[A], z: B)(f: (A, B) => B): B = as match {
    case Nil => z
    case Cons(x, xs) => f(x, foldRight(xs, z)(f))
  }

  def foldLeft[A, B](as: List[A], z: B)(f: (B, A) => B): B = {
    def fold[A, B](as: List[A], z: B, f: (B, A) => B, acc: B): B = as match {
      case Nil => acc
      case Cons(x, xs) => fold(xs, z, f, f(acc, x))
    }

    fold(as, z, f, z)
  }

  def sum(as: List[Int]): Int = foldLeft(as, 0)(_ + _)
  def product(as: List[Double]): Double = foldLeft(as, 1.0)(_ * _)
  def length[A](as: List[A]): Int = foldLeft(as, 0)((x, y) => 1 + x)

  def apply[A](as: A*): List[A] = 
    if (as.isEmpty) Nil
    else Cons(as.head, apply(as.tail: _*))
}
