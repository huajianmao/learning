# Week 3

## Classification and Representation
### Classification
 - Binary classification
 - Multi-Class classification
 - Logistic Regression is a classification method

### Hypothesis Representation
 - sigmoid function / logistic function
 - h(x) = p(y=1|x;theta)

### Decision Boundary
 - Linear decision boundary
 - Non-linear decision boundaries
 - The decision boundary is the line that separates the area where y = 0 and where y = 1.

## Logistic Regression Model
### Cost function
 - `J(theta)`: non-convex / convex
 - global minimum with convex cost function.
 - Cost(h(x), y) = -log(h(x)) if y = 1
                 = -log(1-h(x)) if y = 0
 - penalty based

### Simplified Cost Function and Gradient Descent
 - cost(h(x), y) = -ylog(h(x)) - (1-y)log(1-h(x))

### Advanced Optimization
 - log(x)函数求导 -> 1/x
 - sigmoid函数求导 -> sigmoid * (1 - sigmoid)
 - Gradient descent
 - Conjugate gradient
 - BFGS
 - L-BFGS
 - fminunc(@costFunc, initialTheta, options)

## Multiclass Classification
 - one-vs-all: Train a logistic regression classifier for each class to predict the probability that y = i
 - To make a prediction on a new x, pick the class ￼that maximizes h(x)

## Review

## Solving the Problem of Overfitting
### The Problem of Overfitting
 - what it is
 - under-fitting, high bias
 - over-fitting, high variance: too many features

### Cost Function

### Regularized Linear Regression

### Regularized Logistic Regression

## Review
