# Week 6 

## Advice for Applying Machine Learning
### Deciding What to Try Next
 - how to choose one of the most promising avenues to spend your time pursuing.

 - Get more training examples
 - Try smaller sets of features
 - Try getting additional features
 - Try adding polynomial features
 - Try decreasing lambda
 - Try increasing lambda

#### Machine learning diagnostic
 > A test that you can run to gain insight what is/isn't working with a learning algorithm, and gain guidance as to how best to improve tis performance.

### Evaluating a Hypothesis
 - Training set
 - Cross Validation set
 - Test set

 1. Optimize the parameters in Θ using the training set for each polynomial degree.
 2. Find the polynomial degree d with the least error using the cross validation set.
 3. Estimate the generalization error using the test set with Jtest(Θ(d)), (d = theta from polynomial with lower error);

## Machine Learning System Design
### Building a Spam Classifier

#### Error Analysis

The recommended approach to solving machine learning problems is to:
 - Start with a simple algorithm, implement it quickly, and test it early on your cross validation data.
 - Plot learning curves to decide if more data, more features, etc. are likely to help.
 - Manually examine the errors on examples in the cross validation set and try to spot a trend where most of the errors were made.
