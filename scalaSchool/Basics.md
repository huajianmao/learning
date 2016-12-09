# Basics

## Partial application and Curried functions

## Expressions

> Scala is highly expression-oriented: most things are expressions rather than statements.

### Aside: Functions vs Methods
* Generally, a plain method generated less overhead than a function (which technically is an object with an `apply` method).
* It is complicated.
..* [Scala Functions vs Methods](http://jim-mcbeath.blogspot.com/2009/05/scala-functions-vs-methods.html)
..* [Functions vs methods in Scala - Stack Overflow](http://stackoverflow.com/questions/4839537/functions-vs-methods-in-scala)
..* [Difference between method and function in Scala](http://stackoverflow.com/questions/2529184/difference-between-method-and-function-in-scala)

## Inheritance

``` Scala
class ScientificCalculator(brand: String) extends Calculator(brand) {
  def log(m: Double, base: Double) = math.log(m) / math.log(base)
}
```

Type alias
``` Scala
class ConcurrentPool[K, V] {
  type Queue = ConcurrentLinkedQueue[V]
  type Map   = ConcurrentHashMap[K, Queue]
  ...
}
```

## Abstract Class & Trait
> **When do you want a Trait instead of an Abstract Class?** If you want to define an interface-like type, you might find it difficult to choose between a trait or an abstract class. Either one lets you define a type with some behavior, asking extenders to define some other behavior. Some rules of thumb:

* Favor using traits. It’s handy that a class can extend several traits; a class can extend only one class.
* If you need a constructor parameter, use an abstract class. Abstract class constructors can take parameters; trait constructors can’t. For example, you can’t say trait t(i: Int) {}; the i parameter is illegal.

## apply methods
`apply` methods give you a nice *syntactic sugar* for when a class or object has one main use.

``` Scala
class Foo {}

object FooMaker {
  def apply() = new Foo
}

val newFoo = FooMaker()
// the result would be
// newFoo: Foo = Foo@547015bc
```

## Objects
Objects are used to hold single instances of a class.

## Functions are Objects
> A function is a set of traits.

``` Scala
object addOne extends Function1[Int, Int] {
  def apply(m: Int): Int = m + 1
}
addOne(1)
```

A nice short-hand for `extends Function1[Int, Int]` is `extends (Int => Int)`.

## Pattern Matching

When to use pattern matching?
Pattern matches (x match { ...) are pervasive in well written Scala code: they conflate conditional execution, destructuring, and casting into one construct. Used well they enhance both clarity and safety.})

Use pattern matching to implement type switches:
``` Scala
obj match {
  case str: String => ...
  case addr: SocketAddress => ...
}
```

Pattern matching works best when also combined with destructuring (for example if you are matching case classes); instead of 
``` Scala
animal match {
  case dog: Dog => "dog (%s)".format(dog.breed)
  case _ => animal.species
}
```
write
``` Scala
animal match {
  case Dog(breed) => "dog (%s)".format(breed)
  case other => other.species
}
```

Please refer to [when to use pattern matching](http://twitter.github.com/effectivescala/#Functional programming-Pattern matching) and [pattern matching formatting](http://twitter.github.com/effectivescala/#Formatting-Pattern matching).

## Case classes
* `case class`es are used to conveniently store and match on the contents of a class.
* case classes automatically have equality and nice toString methods based on the constructor arguments.
* case classes are designed to be used with pattern matching
``` Scala
val hp20b = Calculator("hp", "20B")
val hp30b = Calculator("hp", "30B")

def calcType(calc: Calculator) = calc match {
  case Calculator("hp", "20B") => "financial"
  case Calculator("hp", "48G") => "scientific"
  case Calculator("hp", "30B") => "business"
  case _ => "Calculator of unknown type"
}
```

## Exceptions
Exceptions are available in Scala via a `try-catch-finally` syntax that uses pattern matching.
