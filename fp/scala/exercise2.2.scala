def isSorted[A] (as: Array[A], ordered: (A, A) => Boolean): Boolean = {
  def loop(n: Int, as: Array[A], ordered: (A, A) => Boolean): Boolean = {
    if (n >= as.length - 1) true
    else if (ordered(as(n), as(n+1)) == false) false
    else loop(n+1, as, ordered)
  }

  loop(0, as, ordered)
}
