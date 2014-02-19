# Higher Order Functions

> Functions that take other functions as parameters or that return functions as results are called *higher order functions*

Example:
``` Scala
def sum(f: Int=>Int, a: Int, b: Int): Int =
  if (a > b) 0
  else f(a) + sum(f, a+1, b)
```
Type `Int=>Int` is a function type which takes an Integer (the left `Int`) as the parameter and returns an Integer (the second `Int`).

## Anonymous Functions
``` Scala
(x: Int) => x * x * x
```
Here `(x: Int)` is the *parameter* of the function, and `x * x * x` is it's body.

### Anonymous Functions are Syntactic Sugar
An anonymous function `(x1:T1, x2:T2, ..., xn:Tn) => E` can always be expressed using `def` as follows:
`def f(x1:T1, x2:T2, ..., xn:Tn) = E;f`

## Currying
> The definition of functions that return function is so useful in function programming that there is a special syntax for it in Scala.

``` Scala
def sum(f: Int=>Int) (a: Int, b: Int): Int =
  if (a>b) 0 else f(a) + sum(f)(a+1, b)
```
In this way, `sum(f)` is a valid expression by avoiding passing the other arguments.

In general, a definition of a function with multiple parameter lists
`def f(args1)...(argsn) = E` where `n>1`, is equivalent to
`def f(args1)...(argsn-1) = {def g(argsn)=E;g}` where g is a fresh identifier.

By repeating the process n times
`def f(args1)...(argsn-1)(argsn) = E` is shown to be equivalent to
`def f=(args1 => (args2 => ... (argsn=>E)...))`.

This style of definition and function application is called *curring*.

IMPORTANT: functional types assocaiate to the right. This is to say that
`Int => Int => Int` is equivalent to `Int => ( Int => Int)`.

### Exercise

``` Scala
def mapReduce(f:Int=>Int, combine:(Int, Int)=>Int, zero: Int)(a:Int, b:Int): Int = {
  if (a>b) zero
  else combine(f(a), mapReduce(f, combine, zero)(a+1, b))
}

def product(f:Int=>Int)(a:Int, b:Int) = mapReduce(f, (x, y)=> x*y, 1)(a, b)

def factorial(n:Int) = product(x=>x)(1, n)
```
### Summary
The functions are essential abstractions because they allow us to introduce general methods to perform computations as explicit and named elements in our programming language.

These abstractions can be combined with higher-order functions to create new abstractions.

> As a programmer, one must look for opportunities to abstract and reuse.

## Functions and Data

### Class
In Scala, we do this by defining a class:
```Scala
class Rational(x: Int, y: Int) {
  def numer = x
  def denom = y
}
```

### Object
We call *the elements of a class type* objects. We create an object *by prefixing an application of the constructor of
the class with the operator `new`*.

### Members of an Object

### Methods
One can go further and also package functions operating on a data abstraction in the data abstraction itself. Such functions are called *methods*.

`private` members are the memebers which could only be accessed from inside the `class`.

### Data Abstraction
> This ability to choose different implementations of the data without affecting clients is called data abstraction.

### Self Reference
> Inside of a class, the name `this` represents the *object* on which the current method is executed.

### Preconditions
`require` is a predefined function, which takes a condition and an optional message string.
> require is used to enforce a precondition on the caller of a function

If the condition passed to require is false , an IllegalArgumentException is thrown with the given message string.
``` Scala
class Rational(x: Int, y: Int) {
  require(y > 0, ”denominator must be positive”)
  ...
}
```

`assert` also takes a condition and an optional message string as parameters.
>assert is used as to check the code of the function itself.

Like require , a failing assert will also throw an exception, but it’s a different one: AssertionError for assert , IllegalArgumentException for require .
``` Scala
val x = sqrt(y)
assert(x >= 0)
```

### Constructors
In Scala, a class implicitly introduces a constructor. This one is called the primary constructor of the class.
* takes the parameters of the class
* and executes all statements in the class body (such as the require a couple of slides back).

### Auxiliary Constructors
``` Scala
class Rational(x: Int, y: Int) {
  def this(x: Int) = this(x, 1)
  ...
}
```

### Operator
Any method *with a parameter* can be used like an *infix* operator.
`r add s` is equal to `r.add(s)`

In Scala, operators can be used as identifiers.
Thus, an identifier can be:
* Alphanumeric: starting with a letter, followed by a sequence of letters or numbers.
* Symbolic: starting with an operator symbol, followed by other operator symbols.
* The underscore character '_' counts as a letter.
* Alphanumeric identifiers can also end in an underscore followed by some operator symbols.

Example:
``` Scala
x1  *   +?%&   vector_++    counter_=

def unary_- : Rational = new Rational(-numer, denom)

def - (that: Rational) = this + -that
```

#### Precedence of an operator
`(all letters)` < `|` < `^` < `&` < `<`,`>` < `=`,`!` < `:` < `+`,`-` < `*`,`/`,`%` < `(all other special characters)`


