# Data and Abstraction

## Class Hierarchies

### Abstract Classes
Abstract classes can contain members which are missing an implementation (like `incl` and `contains` in the following `IntSet` example).

Consequently, no instances of an abstract class can be created with the operator new .

``` Scala
abstract class IntSet {
  def incl(x: Int): IntSet
  def contains(x: Int): Boolean
}
```

### Class Extensions

``` Scala
class Empty extends IntSet {
  def contains(x: Int): Boolean = false
  def incl(x: Int): IntSet = new NonEmpty(x, new Empty, new Empty)
}

class NonEmpty(elem: Int, left: IntSet, right: IntSet) extends IntSet {
  def contains(x: Int): Boolean = {
    if (x === elem) true
    else if (x < elem) left contains x
    else if (x > elem) right contains x
    else false
  }
  
  def incl(x: Int): IntSet = {
    if (x < elem) new NonEmpty(elem, left incl x, right)
    else if (x > elem) new NonEmpty(elem, left, right incl x)
    else this
  }
}
```
`Empty` and `NonEmpty` both *extend* the class `IntSet`.

This implies that the types `Empty` and `NonEmpty` *conform* to the type IntSet
* an object of the *subclass* type `Empty` or `NonEmpty` can be used wherever an object of the *superclass* type `IntSet` is required.

> The direct or indirect superclasses of a class C are called base classes of C .

### Implementation and Overriding
The definitions of `contains` and `incl` in the classes Empty and NonEmpty **implement** the *abstract* functions in the *base* trait `IntSet`.

It is also possible to redefine an existing, non-abstract definition in a subclass by using `override`.
``` Scala
abstract class Base {
  def foo = 1
  def bar: Int
}

class Sub extends Base {
  override def foo = 2
  def bar = 3
}
```
