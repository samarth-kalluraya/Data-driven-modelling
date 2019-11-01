# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 13:42:54 2019

@author: samarth
"""

import numpy as np
import matplotlib.pyplot as plt
from pyDOE import lhs
from scipy.stats import norm
from scipy.special import legendre
from models import BayesianLinearRegression

if __name__ == "__main__": 
    
    # N is the number of training points.
    N = 500
    M=8

    alpha = 5.0
    beta = 0.1
    
    # Create random input and output data
    X = 2* lhs(1, N)
    noise=np.reshape(np.random.normal(0,0.5,N),(N,1))
    y = (np.exp(X))*np.sin(2*np.pi*X)+noise
    #y = X**5 + noise_var*np.random.randn(N,1)
    
    #define basis functions
    phi_zeros=np.zeros([N,1])
    phi_ones=np.ones([N,1])
    
    phi_iden=X
    
    phi_monom=phi_ones
    for i in range(1,M+1):
        phi_monom=np.concatenate((phi_monom, X**i), axis=1)
    
    phi_fourier=np.concatenate((phi_zeros, phi_ones), axis=1)
    for i in range(1,M+1):
        phi_fourier=np.concatenate((phi_fourier, np.sin(i*np.pi*X)), axis=1)
        phi_fourier=np.concatenate((phi_fourier, np.cos(i*np.pi*X)), axis=1)
        
    leg_poly=legendre(0)
    phi_legen=leg_poly(X)
    for i in range(1,M+1):
        leg_poly=legendre(i)
        phi_legen=np.concatenate((phi_legen, leg_poly(X)), axis=1)
    
    
    # Define models
    m_iden = BayesianLinearRegression(phi_iden, y, alpha, beta)
    m_monom = BayesianLinearRegression(phi_monom, y, alpha, beta)
    m_fourier = BayesianLinearRegression(phi_fourier, y, alpha, beta)
    m_legen = BayesianLinearRegression(phi_legen, y, alpha, beta)
    
    # Fit MLE and MAP estimates for w
    w_MLE_iden = m_iden.fit_MLE()
    w_MAP_iden, Lambda_inv_iden = m_iden.fit_MAP()
    
    w_MLE_monom = m_monom.fit_MLE()
    w_MAP_monom, Lambda_inv_monom = m_monom.fit_MAP()
    
    w_MLE_fourier = m_fourier.fit_MLE()
    w_MAP_fourier, Lambda_inv_fourier = m_fourier.fit_MAP()
    
    w_MLE_legen = m_legen.fit_MLE()
    w_MAP_legen, Lambda_inv_legen = m_legen.fit_MAP()
    
    # Predict at a set of test points
    X_star = np.linspace(0,2,N)[:,None]
    
    phi_star_iden=X_star
    
    phi_star_monom=phi_ones
    for i in range(1,M+1):
        phi_star_monom=np.concatenate((phi_star_monom, X_star**i), axis=1)
    
    phi_star_fourier=np.concatenate((phi_zeros, phi_ones), axis=1)
    for i in range(1,M+1):
        phi_star_fourier=np.concatenate((phi_star_fourier, np.sin(i*np.pi*X_star)), axis=1)
        phi_star_fourier=np.concatenate((phi_star_fourier, np.cos(i*np.pi*X_star)), axis=1)
        
    leg_poly=legendre(0)
    phi_star_legen=leg_poly(X_star)
    for i in range(1,M+1):
        leg_poly=legendre(i)
        phi_star_legen=np.concatenate((phi_star_legen, leg_poly(X_star)), axis=1)
    
    
    y_pred_MLE_iden = np.matmul(phi_star_iden, w_MLE_iden)
    y_pred_MAP_iden = np.matmul(phi_star_iden, w_MAP_iden)
      
    y_pred_MLE_monom = np.matmul(phi_star_monom, w_MLE_monom)
    y_pred_MAP_monom = np.matmul(phi_star_monom, w_MAP_monom)
    
    y_pred_MLE_fourier = np.matmul(phi_star_fourier, w_MLE_fourier)
    y_pred_MAP_fourier = np.matmul(phi_star_fourier, w_MAP_fourier)
    
    y_pred_MLE_legen = np.matmul(phi_star_legen, w_MLE_legen)
    y_pred_MAP_legen = np.matmul(phi_star_legen, w_MAP_legen)
  
    
    # Draw sampes from the predictive posterior
    num_samples = 500
    
    mean_star_iden, var_star_iden = m_iden.predictive_distribution(phi_star_iden)
    samples_iden = np.random.multivariate_normal(mean_star_iden.flatten(), var_star_iden, num_samples)
     
    mean_star_monom, var_star_monom = m_monom.predictive_distribution(phi_star_monom)
    samples_monom = np.random.multivariate_normal(mean_star_monom.flatten(), var_star_monom, num_samples)
    
    mean_star_fourier, var_star_fourier = m_fourier.predictive_distribution(phi_star_fourier)
    samples_fourier = np.random.multivariate_normal(mean_star_fourier.flatten(), var_star_fourier, num_samples)
    
    mean_star_legen, var_star_legen = m_legen.predictive_distribution(phi_star_legen)
    samples_legen = np.random.multivariate_normal(mean_star_legen.flatten(), var_star_legen, num_samples)
   
    
    # Plot
    plt.figure(1, figsize=(25,10))
    plt.subplot(1,2,1)
    plt.plot(X_star, y_pred_MLE_iden, linewidth=6.0, label = 'MLE')
    plt.plot(X_star, y_pred_MAP_iden, linewidth=3.0, label = 'MAP')
    for i in range(0, num_samples):
        plt.plot(X_star, samples_iden[i,:], 'k', linewidth=0.01)
    plt.plot(X,y,'o', label = 'Data')
    plt.legend()
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.axis('tight')

    # Plot
    plt.figure(2, figsize=(25,10))
    plt.subplot(1,2,1)
    plt.plot(X_star, y_pred_MLE_monom, linewidth=3.0, label = 'MLE')
    plt.plot(X_star, y_pred_MAP_monom, linewidth=3.0, label = 'MAP')
    for i in range(0, num_samples):
        plt.plot(X_star, samples_monom[i,:], 'k', linewidth=0.01)
    plt.plot(X,y,'o', label = 'Data')
    plt.legend()
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.axis('tight')

    # Plot
    plt.figure(3, figsize=(25,10))
    plt.subplot(1,2,1)
    plt.plot(X_star, y_pred_MLE_fourier, linewidth=6.0, label = 'MLE')
    plt.plot(X_star, y_pred_MAP_fourier, linewidth=3.0, label = 'MAP')
    for i in range(0, num_samples):
        plt.plot(X_star, samples_fourier[i,:], 'k', linewidth=0.01)
    plt.plot(X,y,'o', label = 'Data')
    plt.legend()
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.axis('tight')

    # Plot
    plt.figure(4, figsize=(25,10))
    plt.subplot(1,2,1)
    plt.plot(X_star, y_pred_MLE_legen, linewidth=3.0, label = 'MLE')
    plt.plot(X_star, y_pred_MAP_legen, linewidth=3.0, label = 'MAP')
    for i in range(0, num_samples):
        plt.plot(X_star, samples_legen[i,:], 'k', linewidth=0.01)
    plt.plot(X,y,'o', label = 'Data')
    plt.legend()
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.axis('tight')


    # for the case N=500 and M=4 plot the features corresponding to each basis
    MM=4
    phi_star_id4=X_star
    
    phi_star_mon4=phi_ones
    for i in range(1,MM+1):
        phi_star_mon4=np.concatenate((phi_star_mon4, X_star**i), axis=1)
    
    phi_star_four4=np.concatenate((phi_zeros, phi_ones), axis=1)
    for i in range(1,MM+1):
        phi_star_four4=np.concatenate((phi_star_four4, np.sin(i*np.pi*X_star)), axis=1)
        phi_star_four4=np.concatenate((phi_star_four4, np.cos(i*np.pi*X_star)), axis=1)
        
    leg_poly=legendre(0)
    phi_star_leg4=leg_poly(X_star)
    for i in range(1,MM+1):
        leg_poly=legendre(i)
        phi_star_leg4=np.concatenate((phi_star_leg4, leg_poly(X_star)), axis=1)
        
    #Plot
    plt.figure(5, figsize=(18,6))
    plt.subplot(1,2,1)
    plt.plot(X_star, phi_star_id4, linewidth=6.0, label = 'identity basis')
    plt.legend()
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.axis('tight')

    # Plot
    plt.figure(6, figsize=(18,6))
    plt.subplot(1,2,1)
    for i in range(0, 5):
        plt.plot(X_star, phi_star_mon4[:,i],  linewidth=3.0)
    plt.legend(('x0','x1','x2','x3','x4'))
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.axis('tight')

    # Plot
    plt.figure(7, figsize=(18,6))
    plt.subplot(1,2,1)
    for i in range(0, 10):
        plt.plot(X_star, phi_star_four4[:,i],  linewidth=3.0)
    plt.legend(('0','1','sin(pi*x)','cos(pi*x)','sin(2pi*x)','cos(2pi*x)','sin(3pi*x)','cos(3pi*x)','sin(4pi*x)','cos(4pi*x)'))
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.axis('tight')

    # Plot
    plt.figure(8, figsize=(18,6))
    plt.subplot(1,2,1)
    for i in range(0, MM+1):
        plt.plot(X_star, phi_star_leg4[:,i],  linewidth=3.0)
    plt.legend(('deg1','deg2','deg3','deg4','deg5'))
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.axis('tight')
