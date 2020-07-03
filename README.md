# LogisticRegression
Logistic regression is used to classify binary objects. In this example we interpolate the appropriate weight coefficient variables to classify a 16X16 image as a handwritten digit as either a four or a nine. The data set is provided by the United States Postal Serves (USPS). The pixel values of the image are evaluated as our regressor value and are between 0 and 255. The images are compressed from a 16X16 matrix to a 256-feature vector.

The code source code here is a modified version of my previous linear regression code.

## Objective
The linear regression equation can be applied to this problem, as this can be used to evaluate non-linear functions by introducing non-linear features.

The derivative of the objective with respect to the variable `w` is

```
w = (X^T*X)^-1X^T*y
```

This can then be plugged into the sigmoid function.

```
P(y = 0 | X_i; w) = theta(W^T*X_i) = 1 / (1 + e^(-W^T*X_i))
```

The sigmoid function allows us to evaluate small changes in the system between zero and one. The decision boundary used in this example is at 0.5 anything below 0.5 maps to 0, and anything above maps to 1. When P(Y=0) predicts the probability of an image being a four, and P(Y=1) predicts the probability of an image being a nine.

![](/images/nine_four.jpg)

## Logistic Regression and the Gradient

![](https://blog.paperspace.com/content/images/2018/05/68747470733a2f2f707669676965722e6769746875622e696f2f6d656469612f696d672f70617274312f6772616469656e745f64657363656e742e676966.gif)

Batch gradient descent is used to find the optimal decision boundary for W.

The idea is to minimize the logarithmic loss function:
```
- sum (i = 1 to n) log[P(Y = y_i | X_i, W)]
```

```
P(Y = y_i | X_i, W) = 
[ theta(W^T * X_i)    if Y_i = 1
  1-theta(W^T * X_i)  if Y_i = 0
]
```

Not going to get into the odds and how this math is set up here, but our final objective to minimize can then be

```
L(w) = - sum (i = 1 to n) [Y_i log(theta(W^T * X_i)) + (1 - Y_i) log(1-theta(W^T * X_i))]
```

The gradient of this function can be found by:

![](https://i.stack.imgur.com/v4iYn.png)

## Results

With the current hyper parameters, the predictive results are 92 percent accurate on the testing dataset. Epochs (ITR) is currently set to 100. And the learning rate (ALPHA) is set to 0.1. These can easily be changed by defining them during compile time.

## Normalization
The data is normalized before being processed. 
The normalization function is:
```
    z_i = (x_i - min) / (max - min)
```
Without normalizing the data the values applied to sigmoid function would be extream.

![](https://miro.medium.com/max/2972/1*vXpodxSx-nslMSpOELhovg.png)

## Setting Up
There is one external library needed before executing this source code. The Eigen library located at [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page)

Download the latest Eigen library and unpack it in the cloned repository. Add the path of the Eigen library to the Makefile.

## For Visual Studio Users
You can add the path of eigen to Configuration Properties -> VC++ Directories -> Include Directories

### Example path setup for Linux/Mac users
```
CXXFLAGS = -std=c++11 -Wall -fpic -O2 -I ./eigen/
```

## Running for Linux/Mac users
Once the path to eigen has been connected to the make file. Simply run the make command.
```
foo@m1$ make
```