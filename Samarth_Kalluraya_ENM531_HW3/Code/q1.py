# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 14:53:09 2019  

@author: samarth
"""

import autograd.numpy as np
import matplotlib.pyplot as plt
from pyDOE import lhs
from scipy.stats import norm
from scipy.special import legendre
from Adam import AdamOptimizer
from autograd import grad

def loss_func(parameters,data):
    parameters=np.squeeze(parameters)
    x=data[:,:-1]
    y=data[:,-1]
    wt=parameters[:-1]
    x_parameters=np.matmul(x, wt)
    var=parameters[-1]
    loss=(1/(2*var))*np.sum((x_parameters-y)**2)+(x.shape[0]/2)*np.log(2*np.pi*var)
    return loss

def fourier_basis(x_val,M,N):       #define basis function
    phi_zeros=np.zeros([N,1])
    phi_ones=np.ones([N,1])    
    X_f=np.concatenate((phi_zeros, phi_ones), axis=1)
    for i in range(1,M+1):
        X_f=np.concatenate((X_f, np.sin(i*np.pi*x_val)), axis=1)
        X_f=np.concatenate((X_f, np.cos(i*np.pi*x_val)), axis=1)
    return X_f

if __name__ == "__main__":
    
    learning_rate=0.001
    batch_size_variable=64
    iterations=10000
    
    N = 500     # N is the number of training points.
    M=16        # M is the number of features
    mn=0        # adam update parameter
    vn=0        # adam update parameter
    b1=0.9      # adam update parameter
    b2=0.999    # adam update parameter
    
   
    # Create random input and output data
    sample_X = 2* lhs(1, N)-1
    noise_var=0.2
    Y=2*np.sin(2*np.pi*sample_X)+np.sin(8*np.pi*sample_X)+0.5*np.sin(16*np.pi*sample_X)
    Y=Y+noise_var*np.std(Y)*np.random.randn(N,1) 
    
    #generate fourier basis
    X=fourier_basis(sample_X,M,N)
    
    #( X, Y, M, N, batch_size, loss_func,mn,vn,b1,b2,learning_rate,iterations)
    adam=AdamOptimizer(X,Y,M,N,batch_size_variable,loss_func,mn,vn,b1,b2,learning_rate,iterations)
    
    #set initial parameter values
    adam.parameterize() 
    
    #run the adam optimizer
    parameters=adam.find_params()
    
    ######test model on new set of data points
    
    #generate new data points
    X_star = np.linspace(-1,1,N)[:,None]
    test_X=fourier_basis(X_star,M,N)      
    
    #exact solution
    Y_desired=2*np.sin(2*np.pi*X_star)+np.sin(8*np.pi*X_star)+0.5*np.sin(16*np.pi*X_star)
    
    #predicted solution
    predict_Y=np.matmul(test_X, parameters[:-1])
        
    
    #print(batch_size_variable)
    #store loss values of training data in the 'loss' variable
    loss=adam.loss_array
    
    
    # Plot
    plt.figure(1, figsize=(25,10))
    plt.subplot(1,2,1)
    plt.plot(X_star, predict_Y, linewidth=3.0, label = 'predicted', c="r")
    plt.plot(X_star, Y_desired, linewidth=3.0, label = 'desired', c="y")
    plt.plot(sample_X,Y,'.', label = 'training data', c="c")
    plt.legend()
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.axis('tight')
    
    plt.figure(2, figsize=(15,6))
    plt.subplot(1,2,1)
    x_axis=np.arange(adam.iter)
    plt.plot(x_axis, adam.loss_array, linewidth=3.0, label = 'loss ')
    plt.legend()
    plt.xlabel('$iterations$')
    plt.ylabel('$loss$')
    plt.axis('tight')
    
    
    