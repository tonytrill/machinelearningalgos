import numpy as np 
from numpy import random

class LinearRegression:
    def __init__(self, epochs, learning_rate, verbose):
        self.w = None
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
        return(np.dot(x,self.w))
    
    """
    Calculate the Mean Squared Error of two vectors.
    MSE = (1/(2*N)) * (x * w - y).T * (x*w - y) 
    """
    def cost(self, x, y):
        return(1/(2*len(y)))*np.transpose((x@self.w - y))@(x@self.w - y)
    """
    Update the weights by adding the gradient of the loss function and the learning rate (alpha)
    """
    def update_weights(self, x, y):
        self.w = self.w - (1/len(y)) * x.T.dot(self.predict(x) - y) * self.alpha

    """ 
    Runs Gradient Descent for the number of epochs given in class creation
    Will print out training results every 100 epochs if verbose is True.
    """
    def fit(self, x, y):
        # Append a column of 1's to the data in order to get the intercept value!
        intercept = np.ones(shape=(x.shape[0], 1))
        x = np.append(intercept, x, axis=1)
        # Initialize 0 weights and intercept as the number of columns in the data
        self.w = np.ones(shape=(x.shape[1],1))
        for epoch in range(self.epochs):
            self.update_weights(x,y)
            if self.verbose == True and epoch % 100 == 0:
                print("epoch number: " + str(epoch) + " ran!")
                print(self.w)
                print("MSE: " + str(self.cost(x,y)))
    
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
        self.w = np.linalg.inv(x.T @ x) @ (x.T @ y)
        


if __name__ == "__main__":
    # Generate some fake data to pass into the learner
    np.random.seed(0)
    x = np.random.rand(100, 1)
    #y = 2 * x[...,0] + 3 * x[...,1] +  + 4 * x[...,2] + x[...,3] + 2
    y = 10 * x[...,0]  + 2
    y.shape = (y.shape[0], 1)
    random.seed(123)
    # Train a model using Gradient Descent
    test2 = LinearRegression(epochs=5000, learning_rate=0.1, verbose=False)
    test2.fit(x,y)
    print(test2.w)
    # Check Solution by Closed Form Solution
    test = LinearRegression(epochs=10, learning_rate=0.01, verbose=True)
    test.closed_form_solution(x,y)
    print(test.w)