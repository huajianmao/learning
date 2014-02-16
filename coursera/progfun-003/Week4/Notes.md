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

### Object Definitions
**Object** is used to defines a *Singleton object*.
In the following example, no other `Empty` instances can be (or need to be) created.
Singleton objects are *values*, so `Empty` evaluates to itself.

```Scala
object Empty extends IntSet {
  def contains(x: Int): Boolean = false
  def incl(x: Int): IntSet = new NonEmpty(x, Empty, Empty)
}
```

### Programs
As It is possible to create standalone applications in Scala.
Each such application contains an object with a `main` method.
``` Scala
object Hello {
  def main(args: Array[String]) = println("Hello world!")
}
```

### Dynamic Binding
> Object-oriented languages (including Scala) implement dynamic method dispatch.

This means that the code invoked by a method call depends on the runtime type of the object that contains the method.



## How Classes are Organized
### Packages
Same to Java.

#### Imports 
Similar to Java

You can import from either a package or an object.

##### Forms of Imports
``` Scala
import week3.{Rational, Hello}  // Import the two 
import week3._                  // In Java * is used as the wildcard
```

##### Automatic Imports
Some entities are automatically imported in any Scala program.
* All members of package `scala`
* All members of package `java.lang`
* All members of the *singleton* object `scala.Predef`


### Traits
**In Java, as well as in Scala, a class can only have one superclass.**

> But what if a class has several natural supertypes to which it conforms or from which it wants to inherit code?
---- Use **Traits**.

* Classes, objects and traits can inherit from at most one class but *arbitrary many* traits.
* Traits resemble interfaces in Java, but are more powerful because they can contains fields and concrete methods.
* On the other hand, traits cannot have (value) parameters, only classes can.

``` Scala
trait Planar {
  def height: Int
  def width: Int
  def surface = height * width
}

class Square extends Shape with Planar with Movable ...
```


### Scala’s Class Hierarchy
Please refer to the figure at the page 9 of the 02_Lecture_3.2_-_How_Classes_Are_Organized.pdf of the class

#### Top Types
`Any`: the base type of all types; with Methods: `==`, `!=`, `equals`, `hashCode`, `toString`
`AnyRef`: The base type of all *reference* types; Alias of `java.lang.Object`
`AnyVal`: The base type of all *primitive* types.

#### The Nothing Type
Nothing is at the bottom of Scala’s type hierarchy. It is a subtype of every other type.
* To signal abnormal termination
* As an element type of empty collections

### Exceptions  
* Scala’s exception handling is similar to Java’s.
* The expression `throw Exc` aborts evaluation with the exception `Exc`.
* The type of this expression is `Nothing`.

### Tye Null Type
* Every reference class type also has `null` as a value.
* The type of `null` is `Null`.
* `Null` is a subtype of every class that inherits from Object ; 
* `Null` is incompatible with subtypes of `AnyVal`.
* 


## Polymorphism

### Cons-Lists
``` Scala
class Cons(val head: Int, val tail: IntList) extends IntList

<=>

class Cons(_head: Int, _tail: IntList) extends IntList {
  val head = _head
  val tail = _tail
}
```

### Complete Definition of List
``` Scala
trait List[T] {
  def isEmpty: Boolean
  def head: T
  def tail: List[T]
}
class Cons[T](val head: T, val tail: List[T]) extends List[T] {
  def isEmpty = false
}
class Nil[T] extends List[T] {
  def isEmpty = true
  def head = throw new NoSuchElementException(ŏNil.headŏ)
  def tail = throw new NoSuchElementException(ŏNil.tailŏ)
}
```
### Generic Functions
Like classes, functions can have type parameters.
``` Scala
def singleton[T](elem: T) = new Cons[T](elem, new Nil[T])
```

### Type Inference
In most of the time, Scala compiler can usually deduce the correct type parameters from the value arguments of a function call. So, in most cases, type parameters can be left out.
For example, `singleton[Int](1)` is the same as `singleton(1)`

### Types and Evaluation
*Type Erasure*

### Polymorphism
Same as Java.


