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


### 
