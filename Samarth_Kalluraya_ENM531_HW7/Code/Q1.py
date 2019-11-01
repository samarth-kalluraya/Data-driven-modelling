# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 14:34:32 2019

@author: samar
    """

import numpy as np
import matplotlib.pyplot as plt
from pyDOE import lhs

from Q1_models import GPRegression

#np.random.seed(1234)

if __name__ == "__main__":    
    
    N = 50
    D = 1
    lb = -1.0*np.ones(D)
    ub = 1.0*np.ones(D)
    noise = 0.01
    
    def f(x):
#        return x*np.sin(2.0*np.pi*x)
        return -1*((x<0.0) - 1.0)

    
    # Training data    
    X = lb + (ub-lb)*lhs(D, N)
    y = f(X)
    y = y + noise*np.std(y)*np.random.randn(N,D)
    
    # Test data
    nn = 200
    X_star = np.linspace(lb, ub, nn)
    y_star = f(X_star)
    
    # Define model
    model = GPRegression(X, y)
    
    # Train 
    model.train()
    
    # Predict
    y_pred, y_var = model.predict(X_star)
    y_var = np.abs(np.diag(y_var))
           
    # Check accuracy                  
    error = np.linalg.norm(y_pred-y_star,2)/np.linalg.norm(y_star,2)
    print("Relative L2 error u: %e" % (error))
    
    # Draw samples from the prior and posterior
    Y0 = model.draw_prior_samples(X_star, 100)
    YP = model.draw_posterior_samples(X_star, 100)
    
    # Plot predictions
    plt.figure(1, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
    plt.plot(X_star, y_star, 'b-', label = "Exact", linewidth=2)
    plt.plot(X_star, y_pred, 'r--', label = "Prediction", linewidth=2)
    lower = y_pred - 2.0*np.sqrt(y_var[:,None])
    upper = y_pred + 2.0*np.sqrt(y_var[:,None])
    plt.fill_between(X_star.flatten(), lower.flatten(), upper.flatten(), 
                     facecolor='orange', alpha=0.5, label="Two std band")
    plt.plot(X,y,'bo', markersize = 12, alpha = 0.5, label = "Data")
    plt.legend(frameon=False,loc='upper left')
    ax = plt.gca()
    ax.set_xlim([lb[0], ub[0]])
    plt.xlabel('$x$')
    plt.ylabel('$f(x)$')
    
    # Plot samples
    plt.figure(2, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
    plt.subplot(2,1,1)
    plt.plot(X_star,Y0)
    ax = plt.gca()
    ax.set_xlim([lb[0], ub[0]])
    plt.ylabel('$f(x)$')
    plt.title("Prior samples")
    plt.subplot(2,1,2)
    plt.plot(X_star,YP)
    plt.plot(X,y,'bo', markersize = 12, alpha = 0.5, label = "Data")
    ax = plt.gca()
    ax.set_xlim([lb[0], ub[0]])
    plt.xlabel('$x$')
    plt.ylabel('$f(x)$')
    plt.title("Posterior samples")
    plt.show()