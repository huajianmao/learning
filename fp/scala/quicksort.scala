def quicksort(xs: Array[Int]): Array[Int] = {
  if (xs.length <= 1) xs
  else {
    val pivolt = xs(xs.length / 2)
    Array.concat(
        quicksort(xs.filter(pivolt >)),
        xs.filter(pivolt ==),
        quicksort(xs.filter(pivolt <))
        )
  }
}
