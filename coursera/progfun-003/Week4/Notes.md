# Types and Pattern Matching

## Functions as Objects
The function type `A => B` is just an abbreviation for the class scala.Function1[A, B], which is roughly defined as follows:
In this way, functions are objects with `apply` methods.
``` Scala
package scala
trait Function1[A, B] {
  def apply(x: A): B
}
```
There are also traits Function2, Function3, ..., for functions which take more parameters (currently up to 22).

### Example
The anonymous function `(x: Int) => x * x` is equal to
``` Scala
{
  class AnonFunc extends Function1[Int, Int] {
    def apply(x: Int): Int = x * x
  }
  new AnonFunc
}
```
or
```Scala
new Function1[Int, Int] {
  def apply(x: Int) = x * x
}
```

### Functions and Methods
Suppose `def f(x:Int): Boolean = ...`,
`(x: Int) => f(x)` will be expanded to
```Scala
new Function1[Int, Boolean] {
  def apply(x: Int) = f(x)
}
```

## Polymophism
*subtyping* & *generics* for the two main areas: *bounds* and *variance*

### Type Bounds
> One might want to express that `assertAllPos` takes `Empty` sets to `Empty sets` and `NonEmpty` sets to `NonEmpty` sets.

A way to express this is `def assertAllPos [ S <: IntSet ] (r: S): S = ...`,
where `<: IntSet` is an *upper bound* of the type parameter `S`.
It means that `S` can be instantiated only to types that conform to IntSet.

* `S <: T` means: `S` is a subtype of `T`
* `S >: T` means: `S` is a supertype of `T`, or `T` is a subtype fo `S`

It is also possible to mix a lower bound with an upper bound.
```Scala
  [S >: NonEmpty <: IntSet]
```

### Covariance
> Given `NonEmpty <: IntSet`, is `List[NonEmpty] <: List[IntSet]`? -- It depends

Opposite:
```Scala
NonEmpty[] a = new NonEmpty[] {new NonEmpty(1, Empty, Empty)}
IntSet[] b = a
b[0] = Empty
NonEmpty s = a[0]
```

We call type for which this relationship holds *covariant* because their subtyping relationship varies exactly like the type parameter.

#### The Liskov Substitution Principle
> If `A <: B`, then everything one can to do with a value of type `B` one should also be able to do with a value of type `A`.

## Variance

```Scala
C[A] <: C[B]          // C is covariant
C[A] >: C[B]          // C is contravariant
neither C[A] nor C[B] is a subtype of the other   // C is nonvariant
```

> if `A2 <: A1` and `B1 <: B2`, then `A1=>B1 <: A2=>B2`

Scala lets you declare the variance of a type by annotating the type parameter:
``` Scala
class C[+A] {...}     // C is covariant
class C[-A] {...}     // C is contravariant
class C[A] {...}      // C is nonvariant
```

> Functions are contravariant in their argument type(s) and covariant in their result type.

``` Scala
package scala
trait Function1[-T, +U] {
  def apply(x: T): U
}
```

### Variance Check

* covariant type parameters can only appear in method results.
* contravariant type parameters can only appear in method.
* invariant type parameters can appear anywhere.

Lower Bound
```Scala
def prepend [U >: T] (elem: U): List[U] = new Cons(elem, this)
```


## Decomposition

## Pattern Matching

*Pattern matching* is a generalization of switch from C/Java to class hierarchies.
> Many functional languages automate the situation whose sole purpose of test and accessor functions is to *reverse* the construction process:
* which subclass was used?
* What were the arguments of the constructor?
This is called *pattern matching*

### Case Classes
``` Scala
trait Expr
case class Number(n: Int) extends Expr
case class Sum(e1: Expr, e2: Expr) extends Expr
```

It also implicitly defines companion objects with apply methods.


``` Scala 
e match {
  case pat1 => expr1,
  ...
  case patn => exprn
}
```

## Lists

* Lists are immutable, the elements of a list cannot be changed
* Lists are recursive.
* Lists are homogeneous: the elements of a list must all have the same type.

The construction operation `::` (pronounced *cons*):
`x :: xs` gives a new list with the first element `x`, followed by the elements of `xs`.

> Operators ending in ':' associate to the right.


