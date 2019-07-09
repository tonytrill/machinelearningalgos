import numpy as np 
from numpy import random

class LinearRegression:
    def __init__(self, epochs, learning_rate, verbose):
        self.w = []
        self.b = 0
        self.epochs = epochs
        self.alpha = learning_rate
        if verbose == None:
            self.verbose = False
        else:
            self.verbose = verbose
    
    """
    Get the predicition of the linear regression model
    """
    def predict(self,x):
        return(np.dot(x,self.w) + self.b)
    
    """
    Calculate the Mean Squared Error of two vectors.
    MSE = (1/N) * SUM from 1 to N (actual - predicted)**2 
    """
    def mse(self, actual, predictions):
        assert len(actual) == len(predictions), "Arrays Not Same Length"
        return np.sum((actual - predictions)**2)*(1/len(actual))

    """
    Calculate the gradient of the Mean Squared Error with respect to w (the weights)
    """
    def gradient_w(self, x, y):
        return (np.dot((y-(np.dot(x, self.w) + self.b)),-2*x)).mean()

    """
    Calculate the gradient of the Mean Squared Error with respect to b (the intercept value)
    """
    def gradient_b(self, x, y):
        return (-2*(y-(np.dot(x,self.w) + self.b))).mean()

    """
    Update the weights by adding the gradient of the loss function and the learning rate (alpha)
    """
    def update_weights(self, x, y):
        self.w = self.w - self.gradient_w(x,y) * self.alpha

    """
    Update the intercept by adding the gradient of the loss function and the learning rate (alpha)
    """
    def update_intercept(self, x, y):
         self.b = self.b - self.gradient_b(x,y) * self.alpha

    """ 
    Runs Gradient Descent for the number of epochs given in class creation
    Will print out training results every 10 epochs if verbose is True.
    """
    def fit(self, x, y):
        # Initialize 0 weights and intercept as the number of columns in the data
        self.w = np.zeros(x.shape[1])
        for epoch in range(self.epochs):
            self.update_weights(x,y)
            print(self.w, self.b)
            self.update_intercept(x,y)
            if self.verbose == True and epoch % 10 == 0:
                print("epoch number: " + str(epoch) + " ran!")
                print("MSE: " + str(self.mse(y, self.predict(x))))
    
    """
    This is the closed form solution of Linear Regression.
    y = wx
    y = w_0x_0 + w_1x1 + w_2x_2 + .....
    where x_0 = 1 for all instances to get the intercept w_0 or b in our class.
    weights = (X_tranposed * X)^-1 * (X_transposed * y)
    """
    def closed_form_solution(self, x, y):
        # Append a column of 1's to the data in order to get the intercept value!
        intercept = np.ones(shape=(x.shape[0], 1))
        x = np.append(intercept, x, axis=1)
        # Closed form caluclation.
        weights = np.linalg.inv(x.T @ x) @ (x.T @ y)
        self.b = weights[0]
        self.w = weights[1:]


if __name__ == "__main__":
    # Generate some fake data to pass into the learner
    np.random.seed(0)
    x = np.random.rand(100, 4)
    y = 2 * x[...,0] + 3 * x[...,1] +  + 4 * x[...,2] + x[...,3] + 2
    y.shape = (y.shape[0], 1)
    random.seed(123)
    test = LinearRegression(epochs=10, learning_rate=.1, verbose=True)
    test.closed_form_solution(x,y)
    print(test.w, test.b)
    #print(test.fit(x,y))
