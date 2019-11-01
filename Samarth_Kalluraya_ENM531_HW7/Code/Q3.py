# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 18:17:23 2019

@author: samar
"""

"""
Solves the equation:
    u_xxxx = f    
"""
    
import numpy as np
from Q3_kernels import Helmholtz_kernels
from pyDOE import lhs
import matplotlib.pyplot as plt

from Q3_models_autograd import HelmGP

np.random.seed(1234)

if __name__ == "__main__": 
    
    N_f = 10
    N_u = 3
    D = 1
    lb = -1.0*np.ones(D)
    ub = 1.0*np.ones(D)
    noise_f = 0.01
    
    # Forcing term
    def f(x):
        f = -(np.pi**2)*np.sin(np.pi*x)-5*np.sqrt(2)*np.sin(np.pi*x)
        return f
    
    # Exact solution
    def u(x):
        u = np.sin(np.pi*x)
        return u
    
    # Boundary condtions data
    X_u = lb + (ub-lb)*lhs(D, N_u)
    y_u = u(X_u)
    
    # Forcing training data
    X_f = lb + (ub-lb)*lhs(D, N_f)
    y_f = f(X_f) + noise_f*np.random.randn(N_f,D)
    
    # Test data
    nn = 500
    X_star = np.linspace(lb, ub, nn)
    u_star = u(X_star)
    f_star = f(X_star)
 
    # Compute required kernels
    k_uu, k_uf, k_ff = Helmholtz_kernels()
        
    # Define model
    model = HelmGP(X_u, y_u, X_f, y_f, k_uu, k_uf, k_ff)
    
    # Training
    model.train()
    
#    # New acquisition point
#    UCB_min = model.DE_UCB_min(X_star)
#    UCB_max = model.DE_UCB_max(X_star)
#    v_min, v_max = UCB_min.min(), UCB_max.max()
#    min_id, max_id = np.argmin(UCB_min), np.argmax(UCB_min)    
#    if (np.abs(v_min) > np.abs(v_max)):
#        new_X = X_star[min_id,:]
#    else:
#        new_X = X_star[max_id,:]
    
    # Prediction
    u_pred, u_var = model.predict_u(X_star)
    f_pred, f_var = model.predict_f(X_star)    
    u_var = np.abs(np.diag(u_var))[:,None]
    f_var = np.abs(np.diag(f_var))[:,None]
    
    # Plotting
    plt.figure(1)
    plt.subplot(1,2,1)
    plt.plot(X_star, u_star, 'b-', label = "Exact", linewidth=2)
    plt.plot(X_star, u_pred, 'r--', label = "Prediction", linewidth=2)
    lower = u_pred - 2.0*np.sqrt(u_var)
    upper = u_pred + 2.0*np.sqrt(u_var)
    plt.fill_between(X_star.flatten(), lower.flatten(), upper.flatten(), 
                     facecolor='orange', alpha=0.5, label="Two std band")
    ax = plt.gca()
    ax.set_xlim([lb[0], ub[0]])
    plt.xlabel('$x$')
    plt.ylabel('$u(x)$')
    
    plt.subplot(1,2,2)
    plt.plot(X_star, f_star, 'b-', label = "Exact", linewidth=2)
    plt.plot(X_star, f_pred, 'r--', label = "Prediction", linewidth=2)
    lower = f_pred - 2.0*np.sqrt(f_var)
    upper = f_pred + 2.0*np.sqrt(f_var)
    plt.fill_between(X_star.flatten(), lower.flatten(), upper.flatten(), 
                     facecolor='orange', alpha=0.5, label="Two std band")
    plt.plot(X_f,y_f,'bo', markersize = 8, alpha = 0.5, label = "Data")
    ax = plt.gca()
    ax.set_xlim([lb[0], ub[0]])
    plt.xlabel('$x$')
    plt.ylabel('$f(x)$')
    print(np.linalg.norm(f_pred-f_star,2)/np.linalg.norm(f_star,2))
    print(np.linalg.norm(u_pred-u_star,2)/np.linalg.norm(u_star,2))
    print(np.exp(model.hyp[-2]))
    print(100*((np.exp(model.hyp[-2])-5*np.sqrt(2))/5*np.sqrt(2)))
    plt.show()