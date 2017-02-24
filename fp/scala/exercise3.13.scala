
sealed trait List[+A]
case object Nil extends List[Nothing]
case class Cons[+A](head: A, tail: List[A]) extends List[A]

object List {
  def foldRight[A, B](as: List[A], z: B)(f: (A, B) => B): B = as match {
    case Nil => z
    case Cons(x, xs) => f(x, foldRight(xs, z)(f))
  }

  def reverse[A](as: List[A]): List[A] = {
    def loop(as: List[A], acc: List[A]): List[A] = as match {
      case Nil => acc
      case Cons(x, xs) => loop(xs, Cons(x, acc))
    }

    loop(as, Nil)
  }

  def foldLeft[A, B](as: List[A], z: B)(f: (B, A) => B): B = {
    def g(a: A, b: B): B = {
      f(b, a)
    }

    foldRight(List.reverse(as), z)((a, b) => f(b, a))
  }

  def apply[A](as: A*): List[A] = 
    if (as.isEmpty) Nil
    else Cons(as.head, apply(as.tail: _*))
}
