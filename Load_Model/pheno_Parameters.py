from scipy.optimize import minimize
import numpy as np
from matplotlib import pyplot as plt
import os

#(where X is an array(DoY)). p : a array of 6 parameters( min, max, slope1, SoS, slope2 , EoS)
def dbl_sigmoid_function(p, X):
    sigma1 = 1./(1+np.exp(-p[2]*(X-p[3])))
    sigma2 = 1./(1+np.exp(p[4]*(X-p[5])))
    y = p[0] + (p[1]-p[0])*((sigma1 + sigma2) -1)
    return y
def cost_function(p, X,Y,func=dbl_sigmoid_function):
    y_pred = func(p, X)
    cost = 0.5*(y_pred-Y)**2
    return cost.sum()/len(X)

class double_sigmoid:
    def __init__(self, X, Y,p):
        self.X = X
        self.Y = Y
        self.p = p #[0.3, 0.4, 0.05, 100, 0.05, 250] #min, max, slope1, SoS, slope2 , EoS 
        self.learning_rate = 0.001
        
    def predict(self, x=[]):
        if not len(x):
            x= self.X
        sigma1 = 1./(1+np.exp(-self.p[2]*(x-self.p[3])))
        sigma2 = 1./(1+np.exp(self.p[4]*(x-self.p[5])))
        y_pred = self.p[0] + (self.p[1]-self.p[0])*((sigma1 + sigma2-1))
        return y_pred
    
    def compute_cost(self):
        y_pred = self.predict(self.X)
        cost = 0.5*(y_pred-self.Y)**2
        return cost.sum()/len(self.X)

    def update_coeffs(self):
        
        self.Y_pred = self.predict()
        Y = self.Y
        Y_pred = self.Y_pred
        m = len(Y)
        lr = self.learning_rate 
        derivative = self.derivatives()
        
        ## Applying gradient Descent
        for i in range(6):
            self.p[i] = self.p[i] - (lr/m)*np.sum((Y_pred-Y)*derivative[i])
            
    def derivatives(self):
        y = self.Y_pred
        min, max, slope1, SoS, slope2 , EoS = self.p
        sigma1 = 1./(1+np.exp(-self.p[2]*(self.X-self.p[3])))
        sigma2 = 1./(1+np.exp(self.p[4]*(self.X-self.p[5])))
        derivative = [0]*6
        
        derivative[0] = -(y-max)/(max-min)
        derivative[1] = (y-min)/(max-min)
        derivative[2] = ((sigma1)*(sigma1-1))*(max-min)*(self.X-SoS)
        derivative[3] = ((sigma1)*(1.- sigma1))*(max-min)*slope1
        derivative[4] = ((sigma2)*(1.- sigma2))*(max-min)*(self.X-EoS)
        derivative[5] = ((sigma2)*(sigma2-1))*(max-min)*slope2
        
        return derivative

def get_perameter(X,Y,p):
    #optimised_p = minimize(cost_function, p, args=(X, Y)).x
    #return optimised_p
    regressor = double_sigmoid(X, Y,p)
    iterations = 0
    costs = []

    while 1:
        Y_pred = regressor.predict()
        cost = regressor.compute_cost()
        if len(costs) and (abs(cost-costs[-1])<1e-9):
            break
        costs.append(cost)
        regressor.update_coeffs()    
        iterations += 1
        
    return regressor.p

def extract_phenology_Parameters(DoY,CC,p):
    return get_perameter(DoY,CC,p)

def plot_CC(X,Y,p,cc_name,saveCC=False,path_location=""):
    X1=np.arange(1,365)
    plt.figure(figsize=(15, 6))
    plt.errorbar(X, Y, fmt="o",color = 'orange', mfc="none")
    plt.ylabel(cc_name)
    plt.xlabel("Day of Year")
    plt.plot(X1, dbl_sigmoid_function(p, X1), '-', color = 'blue')
    
    if saveCC:
        # check path
        if not (path_location[-4:]=='.JPG' or path_location[-4:]=='.jpg' or path_location[-4:]=='.png'
        or path_location[-4:]=='.PNG' or path_location[-4:]=='JPEG' or path_location[-4:]=='jpeg'):

            assert os.path.exists(path_location), path_location+ " does not exist or is not valid"
            path_location+='/'+cc_name+"_plot.jpg"

        plt.savefig(path_location)
    else:
        plt.show()
    plt.close()
    
