import numpy as np 

class LinearRegression:
    def __init__(self, epochs, learning_rate, verbose):
        self.w = []
        self.b = []
        self.epochs = epochs
        self.alpha = learning_rate
        if verbose == None:
            self.verbose = False
        else:
            self.verbose == verbose
    
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
        for epoch in range(self.epochs):
            self.update_weights(x,y)
            self.update_intercept(x,y)
            if self.verbose == True and epoch % 10 == 0:
                print("epoch number: " + str(epoch) + " ran!")
                print("MSE: " + str(self.mse(y, self.predict(x))))
