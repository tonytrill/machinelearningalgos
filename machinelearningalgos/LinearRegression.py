import numpy as np 
from numpy import random

class LinearRegression:
    def __init__(self, epochs, learning_rate, verbose):
        self.w = []
        self.b = []
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
        return(np.dot(self.w,x) + self.b)
    
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
        return np.sum(np.dot(-2*x,(y-(np.dot(self.w, x) + self.b))))*(1/len(y))

    """
    Calculate the gradient of the Mean Squared Error with respect to b (the intercept value)
    """
    def gradient_b(self, x, y):
        return np.sum(-2*(y-(np.dot(self.w, x) + self.b)))*(1/len(y))

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
        self.w = np.zeros(len(x))
        self.b = np.zeros(len(x))
        for epoch in range(self.epochs):
            self.update_weights(x,y)
            self.update_intercept(x,y)
            if self.verbose == True and epoch % 10 == 0:
                print("epoch number: " + str(epoch) + " ran!")
                print("MSE: " + str(self.mse(y, self.predict(x))))


if __name__ == "__main__":
    # Generate some fake data to pass into the learner
    x1 = 3*random.normal(size=100)
    x2 = 10*random.normal(size=100)
    x3 = random.normal(size=100)
    x = np.array([x1,x2,x3])
    y = 2 + x1 + x2  + x3
    print(y)
    test = LinearRegression(epochs=50, learning_rate=0.2, verbose=True)
    test.fit(x,y)
    print(test.w, test.b)