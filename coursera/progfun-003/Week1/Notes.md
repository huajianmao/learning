# Programming Paradigms

Paradigm
> In science, a **paradigm** describes distinct concepts or thought patterns in some scientific discipline.
* imperative programming
* functional programming
* logic programming

## What is Functional Programming Language?

In a *restricted* sense, a functional programming language is one which does not have mutable variables, assignments, or imperative control structures.

In a *wider* sense, a functional programming language enables the construction of elegant programs that focus on functions.

## Why Functional Programming?
* Simpler reasoning principles
* Better modularity
* Good for exploiting parallelism for multicore and cloud computing.

## Elements of Programming
* Primitive expressions representing the simplest elements
* Ways to combine expressions
* Ways to abstract expressions, which introduce a name for an expression by which it can then be referred to.

## REPL(Read-Eval-Print-Loop)
An interactive shell lets one write expressions and responds with their value.

## Scala Function Definition Style
```Scala
def func(var1: TYPE, var2: TYPE, ...): RETTYPE = ...
```

## The Substitution Modle
The idea underlying the model is that all evaluation does is *reduce* an expression to a value.

The substitution model is formalized in the [λ-calculus]( http://en.wikipedia.org/wiki/Lambda_calculus ), which gives a foundation for functional programming.

## Scala's Evaluation Strategy
Scala normally uses call-by-value strategy.

**But if the type of a function parameter starts with `=>` it uses call-by-name.**

## Value Definitions
* `def`: it is *by-name*, its right hand side is evaluated on each use.
* `val`: it is *by-value*, its right hand side is evaluated at the point of the definition itself.

## Block in Scala

## Lexical Scoping
Definition of outer blocks are visible inside a block unless they are shadowed.

## Semicolons and infix operators
There are two ways to overcome the multiline expressions problem.
* We could write the multi-line expression in parentheses, because semicolons are never inserted inside `(...)`:
```Scala
(someLongExpression
 + someOtherExpression)
```
* Or we can write the operator on the first line, because this tells the Scala compiler that expression is not yet finished:
```Scala
someLongExpression +
someOtherExpression
```

## Tail Recursion
Implementation Consideration
> If a function calls itself as its last action, the function's stack frame can be reused. This is called *tail recursion*.

>IN General, if the last action of a function consists of calling a function (which may be the same), one stack frame would be sufficient for both functions. Such calls are called *tail-calls*.

