# List

Merge Sort of List
``` Scala
def msort(xs: List[Int]): List[Int] = {
  val n = xs.length/2
  if (n == 0) xs
  else {
    def merge(xs: List[Int], ys: List[Int]) = {
      xs match {
        case Nil => ys
        case x :: xs1 => {
          ys match {
            case Nil => xs
            case y :: ys1 => {
              if (x < y) x :: merge(xs1, ys)
              else y :: merge(xs, ys1)
            }
          }
        }
      }
    }
    val (fst, snd) = xs splitAt n
    merge(msort(fst), msort(snd))
  }
}
```

## Pairs and Tuples

## Implicit Parameters
``` Scala
def msort[T](xs: List[T])(implicit ord: Ordering) =
def merge(xs: List[T], ys: List[T]) =
  ... if (ord.lt(x, y)) ...
  ... merge(msort(fst), msort(snd)) ...

msort(nums)
msort(fruits)
```

The compiler will search an implicit definition that
* is marked implicit
* has a type compatible with T
* is visible at the point of the function call, or is defined in a companion object associated with T .

## Higher Order List Functions
Functional languages allow programmers to write generic functions that implement patterns such as these using *higher-order functions*.
* transforming each element in a list in a certain way,
* retrieving a list of all elements satisfying a criterion,
* combining the elements of a list using an operator.

``` Scala
def scaleList(xs: List[Double], factor: Double): List[Double] = xs match {
  case Nil => xs
  case y :: ys => y * factor :: scaleList(ys, factor)
}
```

### Map
This scheme can be generalized to the method map of the List class.
``` Scala
abstract class List[T] { ...
  def map[U](f: T=>U): List[U] = this match {
    case Nil     => this
    case x :: xs => f(x) :: xs.map(f)
  }
}
```

### Filter
This pattern is generalized by the method `filter` of the List class:
``` Scala
abstract class List[T] {
  ...
  def filter(p: T=>Boolean): List[T] = this match {
    case Nil     => this
    case x :: xs => if (p(x)) x :: xs.filter(p) else xs.filter(p)
  }
}
```

## Reduction of Lists

Another comman operation on lists is to combine the elements of a list using a given operator.
``` Scala
sum(List(x1, ..., xn))      = 0 + x1 + ... + xn
product(List(x1, ...., xn)) = 1 * x1 * ... * xn
```


> Instead of `((x,y) => x*y)`, one can also write shorter: `(_ * _)`

Every `_` represents a new parameter, going from left to right.
``` Scala
def sum(xs: List[Int])     = (0 :: xs) reduceLeft (_ + _)
def product(xs: List[Int]) = (1 :: xs) reduceLeft (_ * _)
```
where `reduceLeft` inserts a given binary operator between adjacent elements of a list.

`folderLeft` is like `reduceLeft` but takes an *accumulator*, `z`, as an additional parameter, which is returned when `foldLeft` is called on an empty list.
``` Scala
(List(x1, ..., xn) foldLeft z)(op) = (...(z op x1) op ... ) op xn
```


``` Scala
abstract class List[T] { ...
  def reduceLeft(op: (T, T) => T): T = this match {
    case Nil => throw new Error(ŏNil.reduceLeftŏ)
    case x :: xs => (xs foldLeft x)(op)
  }
  def foldLeft[U](z: U)(op: (U, T) => U): U = this match {
    case Nil => z
    case x :: xs => (xs foldLeft op(z, x))(op)
  }
}
```

### Deduction of Reversing List

