# Syntax Summary

## Types

``` Scala
Type         = SimpleType | FunctionType
FunctionType = SimpleType => Type
             | ([Types]) => Type
SimpleType   = Ident
Types        = Type {, Types}
```

A type can be
* A numeric type: Int, Double (and Byte, Short, Char, Long, Float),
* The Boolean Type with the values `true` and `false`,
* The String type,
* A *function type*, like `Int => Int`, `(Int, Int) => Int`.

## Expressions
``` Scala
Expr          = InfixExpr | FunctionExpr
              | if (Expr) Expr else Expr
InfixExpr     = PrefixExpr | InfixExpr Operator InfixExpr
Operator      = ident
PrefixExpr    = [+|-|!|~] SimpleExpr
SimpleExpr    = ident | literal | SimpleExpr . ident | Block
FunctionExpr  = Bindings => Expr
Bindings      = ident [: SimpleType] | ([Binding {, Binding}])
Binding       = ident [: Type]
Block         = { {Def;} Expr}
```
An expression can be:
* An *identifier* such as `x`, `isGoodEnough`,
* A *literal*, like `0`, `1.0`, `"abc"`,
* A *function application*, like `sqrt(x)`,
* An *operator application*, like `-x`, `y+x`
* A *selection*, like `math.abs`,
* A *conditional expression*, like `if (x<0) -x else x`,
* A *block*, like `{ val x = math.abs(y); x*2}`,
* An *anonymous function*, like `x=>x+1`.

## Definitions
``` Scala
Def           = FunDef
FunDef        = def ident {([Parameters])}
                [: Type] = Expr

ValDef        = val ident [: Type] = Expr
Parameter     = ident : [=>] Type
Parameters    = Parameter {, Parameter}
```

A definition can be:
* A *function definition*, like `def square(x: Int) = x*x`,
* A *value definition*, like `val y = square(2)`

A parameter can be:
* A *call-by-value* parameter, like `(x: Double)`,
* A *call-by-name* parameter, like `(y: => Double)`.

