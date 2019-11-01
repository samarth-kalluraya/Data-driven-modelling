# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 14:36:34 2019

@author: samar
"""

import autograd.numpy as np
import matplotlib.pyplot as plt
from pyDOE import lhs
from Q2_model import Multifidelity_GP

np.random.seed(1234)

def f_H(x):
    return (6.0*x-2.0)**2 * np.sin(12.*x-4.0)

def f_L(x):
    return 0.5*f_H(x) + 10.0*(x-0.5) - 5.0

def Normalize(X, X_m, X_s):
    return (X-X_m)/(X_s)
def Denormalize(X, X_m, X_s):
    return X*X_s+X_m

if __name__ == "__main__":    
    
    N_H = 4
    N_L = 13
    D = 1
    lb = 0.0*np.ones(D)
    ub = 1.0*np.ones(D)
    noise_L = 0.00
    noise_H = 0.00
    
    Normalize_input_data = 1
    Normalize_output_data = 1
    
    # Training data    
    X_L = lb + (ub-lb)*lhs(D, N_L)
    y_L = f_L(X_L) + noise_L*np.random.randn(N_L,D)
    
    X_H = np.array([[0.0,0.4,0.6,1.0]]).T
    y_H = f_H(X_H) + noise_H*np.random.randn(N_H,D)
    
    # Test data
    nn = 200
    X_star = np.linspace(lb, ub, nn)
    y_star_H = f_H(X_star)
    y_star_L = f_L(X_star)

     #  Normalize Input Data
    if Normalize_input_data == 1:
        X = np.vstack((X_L,X_H))
        X_m = np.mean(X, axis = 0)
        X_s = np.std(X, axis = 0)   
        X_L = Normalize(X_L, X_m, X_s)
        X_H = Normalize(X_H, X_m, X_s)
        lb = Normalize(lb, X_m, X_s)
        ub = Normalize(ub, X_m, X_s)
        X_star = Normalize(X_star, X_m, X_s)
        
    #  Normalize Output Data
    if Normalize_output_data == 1:
        y = np.vstack((y_L,y_H))
        y_m = np.mean(y, axis = 0)
        y_s = np.std(y, axis = 0)   
        y_L = Normalize(y_L, y_m, y_s)
        y_H = Normalize(y_H, y_m, y_s)
        y_star_H = Normalize(y_star_H, y_m, y_s)
        y_star_L = Normalize(y_star_L, y_m, y_s)

    
    # Define model
    model = Multifidelity_GP(X_L, y_L, X_H, y_H)
    
    # Train 
    model.train()
    
    # Predict
    y_pred_H, y_var_H = model.predict_H(X_star)
    y_var_H = np.abs(np.diag(y_var_H))
     
    y_pred_L, y_var_L = model.predict_L(X_star)
    y_var_L = np.abs(np.diag(y_var_L))
        

    # Check accuracy                       
    error = np.linalg.norm(y_pred_H-y_star_H,2)/np.linalg.norm(y_star_H,2)
    print("Relative L2 error high fidelty: %e" % (error))
    
    error = np.linalg.norm(y_pred_L-y_star_L,2)/np.linalg.norm(y_star_L,2)
    print("Relative L2 error low fidelty: %e" % (error))

    if Normalize_input_data == 1:
        X_L = Denormalize(X_L, X_m, X_s)
        X_H = Denormalize(X_H, X_m, X_s)
        lb = Denormalize(lb, X_m, X_s)
        ub = Denormalize(ub, X_m, X_s)
        X_star = Denormalize(X_star, X_m, X_s)
        
    #  DeNormalize Output Data
    if Normalize_output_data == 1:
        y_L = Denormalize(y_L, y_m, y_s)
        y_H = Denormalize(y_H, y_m, y_s)
        y_star_H = Denormalize(y_star_H, y_m, y_s)
        y_star_L = Denormalize(y_star_L, y_m, y_s)
        y_pred_H = Denormalize(y_pred_H, y_m, y_s)
        y_pred_L = Denormalize(y_pred_L, y_m, y_s)
        y_var_H = Denormalize(y_var_H, 0, y_s**2)
        y_var_L = Denormalize(y_var_L, 0, y_s**2)

#    # Plot
    plt.figure(1, facecolor = 'w')
    plt.plot(X_star, y_star_H, 'b-', label = "Exact", linewidth=2)
    plt.plot(X_star, y_pred_H, 'r--', label = "Prediction", linewidth=2)
    lower = y_pred_H - 2.0*np.sqrt(y_var_H[:,None])
    upper = y_pred_H + 2.0*np.sqrt(y_var_H[:,None])
    plt.fill_between(X_star.flatten(), lower.flatten(), upper.flatten(), 
                     facecolor='orange', alpha=0.5, label="Two std band")
    plt.plot(X_star, y_star_L, 'm-', label = "Exact", linewidth=2)
    plt.plot(X_star, y_pred_L, 'g--', label = "Prediction", linewidth=2)
    lower = y_pred_L - 2.0*np.sqrt(y_var_L[:,None])
    upper = y_pred_L + 2.0*np.sqrt(y_var_L[:,None])
    plt.fill_between(X_star.flatten(), lower.flatten(), upper.flatten(), 
                     facecolor='orange', alpha=0.5, label="Two std band")
    plt.plot(X_H,y_H,'bo', label = "High-fidelity data")
    plt.plot(X_L,y_L,'ms', label = "Low-fidelity data")
    plt.legend(frameon=False,loc='upper left')
    ax = plt.gca()
    ax.set_xlim([lb[0], ub[0]])
    plt.xlabel('$x$')
    plt.ylabel('$f(x)$')
#    plt.show()
    plt.savefig('Q2_'+str(N_L))