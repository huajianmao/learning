# Week 2

## Linear Regression with Multiple Variables
### Environment Setup Instructions
#### Multiple Features
 - (如果feature中有非数值feature如何处理？)
 - number of features
 - number of samples
 - zero feature
 - hθ(x)=θ0+θ1x1+θ2x2+θ3x3+⋯+θnxn

#### Gradient Descent for Multiple Variables
 - hypothesis, parameters and cost function
 - ![](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/MYm8uqafEeaZoQ7hPZtKqg_c974c2e2953662e9578b38c7b04591ed_Screenshot-2016-11-09-09.07.04.png?expiry=1567036800000&hmac=Pqckk4gkkh678KGN6Z7Tkwk6ADWspzCBJNbSqHM4iP4)

#### Gradient Descent in Practice I - Feature Scaling
 - Idea: Make sure features are on a similar scale to overcome the data skew problem
 - Get every feature into approximately a `-1 <= xi <= 1` range.
 - Mean normalization: Replace xi with (xi - ui) divided by the range of the x values to make features have approximately zero mean (Do not apply to x0 = 1)

#### Gradient Descent in Practice II - Learning Rate
 - How to make sure gradient descent is working correctly
 - J(θ) should decrease when the number of iteration increase
 - Declare convergence if J(θ) decreases by less than 10^-3 in one iteration.
 - If J(θ) increases, you may use smaller `alpha`.
 - If `alpha` is too small: slow convergence
 - If `alpha` is too large: J(θ) may not decrease on every iteration; may not converge.
 - To choose `alpha`, try `..., 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, ...`

#### Features and Polynomial Regression
 - We can improve our features and the form of our hypothesis function in a couple different ways.
 - We can combine multiple features into one. 
 - We can change the behavior or curve of our hypothesis function by making it a quadratic, cubic or square root function (or any other form).

### Computing Parameters Analytically
#### Normal Equation
 - Method to solve for `theta` analytically
 - 最小值处导数为0
 - 仅有方阵有逆矩阵
 - suitable for situation when `n` is small
 - complexity: `O(n^3)`

#### Normal Equation Noninvertibility
 - pinv works for both `invertible` and `noninvertible`
 - Too many features, delete some features or use regularization

### Submitting Programming Assignments
#### Working on and Submitting Programming Assignments

### Review

## Octave / Matlab Tutorial
### Octave / Matlab Tutorial
#### Basic Operations
 - prototype with Octave first ten implement with other languages
 - Octave, Python, NumPy
 - ones / zeros / rand / randn / eye / disp / hist / 1:0.1:2
 - help

#### Moving Data Around
 - size / length
 - load filename / who / whos
 - clear
 - save filename variable
 - A([1 3], :) / A(2, :) / A(:, 2)
 - A = [A, [100; 102; 103]]
 - A(:)
 - C = [A B] % concatenate A and B
 - C = [A; B]

#### Computing on Data
 - A * C / A .* B / A .^ 2 / 1 ./ A / log(v) / exp(v) / abs(v)
 - v + ones(length(v), 1) / v + 1
 - A'
 - max(v) / max(A)  % max value of each column
 - max(A, [], 1) / max(A, [], 2)
 - a < 3 / find(a < 3)
 - magic(3)
 - sum / prod / floor / ceil
 - sum(A, 1) / sum(A, 2)
 - flipud
 - pinv(A) / inv(A)

#### Plotting Data
 - plot
 - hold on
 - xlabel('time') / ylable('value') / legend('sin', 'cos') / title('my plot')
 - print -dpng 'myPlog.png'
 - figure
 - subplot(1, 2, 1);
 - axis([0.5 1 -1 1])
 - iamgesc(A), colorbar, colormap gray;

#### Control Statements: for, while, if statement
 - for i=1:9, / if i == 6, elseif ..., else
 - exit / quit
 - function y = func_name(x)
 - function [y1, y2] = squreAndCubeThisNumber(x)
 - search path / addpath

#### Vectorization
 - 

### Review
