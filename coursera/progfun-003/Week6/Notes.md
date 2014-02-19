# Collections

## Other Collections

### Sequences
*Vector*, like *List*, is a sequence implementation which is *linear*.
Similar but different with `::` in `List`, `:+` and `+:` are two operations of `Vector`.
* `x +: xs`: create a new vector with leading element `x`, followed by all elements of `xs`.
* `xs :+ x`: create a new vector with trailing element `x`, preceded by all elements of `xs`.
Note that the `:` always points to the sequence.

`List` and `Vector` are subclass of `Seq`, the class of all sequences, which itself is a subclass of `Iterable`.

#### Arrays and Strings
They come from Java.

#### Ranges
Ranges are represented as signle objects with three fields: *lower bound*, *upper bound*, *step value*.

``` Scala
val r: Range = 1 until 5   // 1, 2, 3, 4
val s: Range = 1 to 5      // 1, 2, 3, 4, 5
1 to 10 by 3               // 1, 4, 7, 10
6 to 1 by -2               // 6, 4, 2
```


## Combinatorial Search and For-Expressions

### Handling Nested Sequences
> Given a positive integer *n*, find all pairs of positive integers *i* and *j*, with `1 <= j < i < n` such that `i+j` is prime.


A useful law:
> `xs flatMap f = (xs map f).flatten`

``` Scala
(1 until n) flatMap (i => {
  (1 until i) map (j => (i, j))) filter ( pair => isPrime(pair._1 + pair._2))
})
```
### For Expression
A *for-expression* is of the form `for (x) yield e`, where `s` is a sequence of *generators* and *filters*, and `e` is an expression whose value is returned by an iteration.
`for ( p <- person if p.age > 20 ) yield p.name`
equals to `persons filter (p => p.age > 20) map (p => p.name)`

So the nested sequences in the last example could be revised to 
``` Scala
for {
  i <- 1 until n
  j <- 1 until i
  if isPrime(i + j)
} yield (i, j)
```


## Combinatorial Search

### Set vs. Sequences
* Sets are unordered; the elements of a set do not hava a predefined order in which they appear in the set.
* set do not have duplicate elements
* The fundamental operation on sets is contains `s contains 5`

The eight queens problem
> The eight queens problem is to place eight queens on a chessboard so that no queen is threatened by another.

``` Scala
def queens(n: Int) = {
  def placeQueens(k: Int): Set[List[Int]] = {
    if (k == 0) Set(List())
    else {
      for {
        queens <- placeQueens(k - 1)
        col <- 0 until n
        if isSafe(col, queens)
      } yield col :: queens
    }
  }
  placeQueens(n)
}

def isSafe(col: Int, queens: List[Int]): Boolean = {
  val row = queens.length
  val queensWithRow = (row-1 to 0 by -1) zip queens
  queensWithRow forall {
    case (r, c) => col != c && math.abs(col-c) != row - r
  }
}
```


## Queries with For
``` Scala
val bookSet = books.toSet
for {
  b1 <- bookSet
  b2 <- bookSet
  if b1 != b2

  a1 <- b1.authors
  a2 <- b2.authors
  if a1 == a2
} yield a1
```


## Translation of For

### For-Expressions and Higher-Order Functions
All the higher-order functions, like `map` `flatMap` `filter`, can be defined in terms of `for`:
``` Scala
def mapFun[T, U](xs: List[T], f: T => U): List[U] = for (x <- xs) yield f(x)

def flatMap[T, U](xs: List[T], f: T => Iterable[U]): List[U] = for (x <- xs; y <- f(x)) yield y

def filter[T](xs: List[T], p: T => Boolean): List[T] = for (x <- xs if p(x)) yield x
```

### Translation of For
In reality, the Scala compiler expresses *for-expressions* in terms of `map`, `flatMap` and a lazy variant of `filter`.

* `for (x <- e1) yield e2` is translated to `e1.map(x=>e2)`
* `for (x <- e1 if f; s) yield e2` is translated to `for (x <- e1.withFilter(x => f); s) yield e2`
* `for (x <- e1; y <- e2; s) yield e3` is translated to `e1.flatMap(x => for (y <- e2; s) yield e3)`

### Generalization of For
The translation of `for` is not limited to lists or sequences, or even collections.
It is based solely on the presence of the methods `map`, `flatMap` and `withFilter`.


## Map

A map of type `Map[Key, Value]` is a data structure that associates keys of type `Key` with values with type `Value`.
`val capitalOfCountry = Map("US"->"Washington", "Switzerland"->"Bern")`

NOTE: In fact, the syntax `key -> value` is just an alternative way to write the `pair(key, value)`.

### Maps are Iterables
class `Map[Key, Value]` extends the collection type `Iterable[(Key, Value)]`.

### Maps are Functions
Class `Map[Key, Value]` also extends the function type `Key => Value`, so maps can be used everywhere functions can.
`capitalOfCountry("US")  // "Washington"`

### Querying Map
``` Scala
capitalOfCountry("Andorra") // java.util.NoSuchElementException: key not found: Andorra

capitalOfCountry get "Andorra" // None
```

### Sorted and GroupBy
`orderBy` on a collection can be expressed by `sortWith` and `sorted`.
``` Scala
val fruit = List("apple", "pear", "orange", "pineapple")
fruit sortWith (_.length < _.length) // List("pear", "apple", "orange", "pineapple")
fruit.sorted                         // List("apple", "orange", "pear", "pineapple")
```

`groupBy` is available on Scala collections. It partitions a collection into a map of collections according to a discriminator function `f`.
`fruit groupBy (_.head)`

### Default Values
There is an operation `withDefaultValue` that turns a map into a *total function*:
``` Scala
val cap1 = capitalOfCountry withDefaultValue "<unknown>"
cap1("Andorra")   // "<unknown>"
```

### Implementation of Polynom
``` Scala
class Poly(terms0: Map[Int, Double]) {
  def this(bindings: (Int, Double)*) = this(bindings.toMap)
  val terms = terms0 withDefaultValue 0.0
  def + (other: Poly) = new Poly(terms ++ (other.terms map adjust))
  def adjust(term: (Int, Double)): (Int, Double) = {
    val (exp, coeff) = term
    exp -> (coeff + terms(exp))
  }
  override def toString =
    (for ((exp, coeff) <- terms.toList.sorted.reverse) yield coeff+"x^"+exp) mkString " + "
}
```
