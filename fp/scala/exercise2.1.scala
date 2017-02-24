def fib(n: Int): Int = {
  def loop(n: Int, acc1: Int, acc2: Int): Int = {
    if (n < 2) acc1
    else loop(n-1, acc2, acc1 + acc2)
  }

  loop(n, 0, 1)
}
