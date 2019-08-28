# Week 1

## Introduction
### Welcome

 > The AI dream of someday building machines as intelligent as you or me.

 > To mimic how the human brain learns.

Machine Learning
 - Grew out of work in AI
 - New capability for computers

Examples:
 - Database mining
 - Applications can't program by hand
 - Self-customizing programs
 - Understand human using

### What is Machine Learning
 About **what** and **when** to use machine learning.

-------------
#### Definitions
Arthur Samuel(1959) [Checkers playing program]:
 > Field of study that gives computers the ability to learn without being explicitly programmed.

Tom Mitchell(1998):
 > A computer program is said to learn from experience E with respect to some task T and some performance measure P, if its performance on T, as measured by P, improves with experience E.

---------

In general, any machine learning problem can be assigned to one of two broad classifications:
 - Supervised learning
 - Unsupervised learning

Others:
 - Reinforcement learning
 - recommender systems.


> how if you're actually trying to develop a machine learning system, how to make those best practices type decisions about the way in which you build your system.

### Supervised Learning
 > with "right answers". Given the "right ansser" for each exampple in the data.
 >
 > We are given a data set and already know what our correct output should look like, having the idea that there is a relationship between the input and the output.
 - Regression: to predict a continuos value
 - Classification: to predict a discrete value

> when we talk about an algorithm called the Support Vector Machine, there will be a neat mathematical trick that will allow a computer to deal with an infinite number of features.

### Unsupervised Learning
 > has no labels
 >
 > Unsupervised learning allows us to approach problems with little or no idea what our results should look like. We can derive structure from data where we don't necessarily know the effect of the variables.

 > Cluster

[ ] is there any cluster characteristics in the Stroke data?

> Octave


## Model and Cost Function

### Model Representation
 > given a training set, to learn a function h : X → Y
 > so that h(x) is a “good” predictor for the corresponding value of y
 >
 > hypothesis (function *h*)
 >
 > ![](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/H6qTdZmYEeaagxL7xdFKxA_2f0f671110e8f7446bb2b5b2f75a8874_Screenshot-2016-10-23-20.14.58.png?expiry=1566432000000&hmac=rYWNUVp3JKFkllzhh41Q8cIj8TTShHpFaJdMzcOwC9c)
 >
 > **Regression problem**: When the target variable that we’re trying to predict is continuous
 >
 > **Classification problem**:  When y can take on only a small number of discrete values

### Cost Function

> Key question: 怎样算拟合得好？
> 
> -> 
>
> We can measure the accuracy of our hypothesis function by using a cost function.
> 
> **Cost function** (object function)

The **square cost function** is probably the most commonly used one for regression problems.

#### Intuition I
 - What the cost function is doing
 - Why we want to use it

#### Intuition II
 > contour plot
 
 >  what we really want is an efficient algorithm for automatically finding the value of theta zero and theta one, that minimizes the cost function J.


 ## Parameter Learning

 ### Gradient descent
  > Gradient descent is used all over the place in machine learning.
  [local optimum]

 - learning rate (`alpha`)
 - derivative term (partial derivatives)
 - simultaneously update theta 0 and theta 1

#### Gradient descent intuition
 - if `alpha` is too small, gradient descent can be slow
 - if `alpha` is too large, gradient descent can overshoot the minimum.
 - Gradient descent can converge to a local minimum, even with the learning rate `alpha` fixed.

#### Gradient Descent for Linear Regression
 - Convex function
 - **"Batch" Gradient Descent**: Each step of gradient descent uses all the training examples

## Linear Algebra Review
### Matrices and Vectors
 - Dimension of matrix: number of rows x number of columns
 - Matrix Elements: A_ij, the `i`th row, `j`th column
 - Vector: An `n x 1` matrix
 - Vector element: y_i, the `i`th element
 - 1-indexed vs. 0-indexed: default is 1-indexed in this course
 - Uppercase letters to notate a matrix, lowercase letters to notate a vector

### Addition and Scalar Multiplication
 - Matrix Addition
 - Scalar Multiplication
 - scalar x M = M x scalar
 - Combination of Operands
 - A + scalar: each element of A adding scalar

### Matrix Vector Multiplication
 - Matrix-Matrix Multiplication
 - 应用：如何把一个问题转化为 线性代数问题，然后利用线性代数库进行问题批量求解

### Matrix multiplication properties
 - matrix multiplication is **NOT** commutative. `A * B != B * A`
 - matrix multiplication is associative. `(A * B) * C = A * (B * C)`
 - Identity Matrix I. `I * A = A * I` (`eye(2) in Octave`)
 

### Inverse and Transpose