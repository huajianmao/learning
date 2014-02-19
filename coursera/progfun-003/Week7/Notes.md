# Lazy Evaluation

## Streams

Streams are similar to lists, but their trait is *evaluated only on demand*.
> Avoid computing the tail of a sequence until it is needed for the evaluation result (which might be never).

`x #:: xs == Stream.cons(x, xs)`

``` Scala
trait Stream[+A] extends seq[A] {
  def isEmpty: Boolean
  def head: A
  def tail: Stream[A]
  ...
}
```

### Implementation of Streams
Concrete implementations of streams are defined in the Stream companion object.
``` Scala
object Stream {
  def cons[T](hd: T, tl: => Stream[T]) = new Stream[T] {
    def isEmpty = false
    def head = hd
    def tail = tl
  }
  ...
}
```
The second parameter of Stream.cons is a *by-name parameter*, so it is not evaluated at the point of call but will be evaluated each time someone calls tail on a Stream object.


## Lazy Evaluation
Storing the result of first evaluation of tail and re-using the stored result instead of recomputing tail.

We call this scheme *lazy evaluation* (as opposed to *by-name evaluation* in the case where everything is recomputed, and *strict evaluation* for normal parameters and `val` definitions.)

Scala uses *strict evaluation* by default, but allows lazy evaluation of value definitions with the *lazy val* form: `lazy val x = expr`

Using a lazy value for tail, `Stream.cons` can be implemented more efficientyly:
``` Scala
def cons[T](hd: T, tl: => Stream[T]) = new Stream[T] {
  def head = hd
  lazy val tail = tl
}
```

### Infinite Streams
The stream of all integers starting from a number:
``` Scala
def from(n: Int): Stream[Int] = n #:: from(n+1)

val nats = from(0)
nats map (_ * 4)
```

### The Sieve of Eratosthenes
``` Scala
def sieve(s: Stream[Int]): Stream[Int] = {
  s.head #:: sieve(s.tail filter (_ % s.head !=0))
}

val primes = sieve(from(2))

(primes take N).toList
```

## Case Study

### Guiding Principles for Good Design
* Name everything you can.
* Put operations into natural scopes.
* Keep degrees of freedom for future refinements.


