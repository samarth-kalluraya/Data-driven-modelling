# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 18:17:55 2019

@author: samar
"""

import sympy as sp
import autograd.numpy as np
from autograd.scipy.special import erf
from sympy.printing.lambdarepr import NumPyPrinter

     
def Helmholtz_kernels():
    
    def operator(k, x, gamma):
        return sp.diff(k,x,2)-gamma*k
    
    ############################## Symbolic ##############################
    
    autograd_modules = [{'And': np.logical_and, 'Or': np.logical_or, 
                         'erf': erf}, np]

    # Define kernels
    x, x_prime = sp.symbols('x x_prime')
    sigma, theta = sp.symbols('sigma theta')
    gamma=sp.symbols('gamma')
    
    k_uu = sigma*sp.exp(-0.5*((x-x_prime)**2/theta))
    k_uf = operator(k_uu, x_prime, gamma)
    
    k_ff = operator(operator(k_uu, x_prime, gamma), x, gamma)
    
    ############################## Lambdify ##############################
    
    ########################## Row 1 ##################################
    
    lambda_k_uu = sp.lambdify((x, x_prime, sigma, theta), k_uu, 
                              dummify=True, modules=autograd_modules, 
                              printer=NumPyPrinter)
    
    
    lambda_k_uf = sp.lambdify((x, x_prime, sigma, theta, gamma), k_uf, 
                              dummify=True, modules=autograd_modules, 
                              printer=NumPyPrinter)
    
    ########################## Row 2 ##################################
    
   
    lambda_k_ff = sp.lambdify((x, x_prime, sigma, theta, gamma), k_ff, 
                              dummify=True, modules=autograd_modules, 
                              printer=NumPyPrinter)
    
    
    ############################## Vectorization ##############################
    
    ########################## Row 1 ##################################

    def k_uu(x, x_prime, hyp):
        sigma = np.exp(hyp[0])
        theta = np.exp(hyp[1])
                    
        N = x.shape[0]
        N_prime = x_prime.shape[0]
    
        x = np.matmul(x, np.ones((1,N_prime)))
        x_prime = np.matmul(np.ones((N,1)),x_prime.T)

        return lambda_k_uu(x, x_prime, sigma, theta)
    
    
    def k_uf(x, x_prime, hyp):
        sigma = np.exp(hyp[0])
        theta = np.exp(hyp[1])
        gamma = np.exp(hyp[2])
            
        N = x.shape[0]
        N_prime = x_prime.shape[0]
    
        x = np.matmul(x, np.ones((1,N_prime)))
        x_prime = np.matmul(np.ones((N,1)),x_prime.T)

        return lambda_k_uf(x, x_prime, sigma, theta, gamma)
    
    
    ########################## Row 2 ##################################


    def k_ff(x, x_prime, hyp):
        sigma = np.exp(hyp[0])
        theta = np.exp(hyp[1])
        gamma = np.exp(hyp[2])
                    
        N = x.shape[0]
        N_prime = x_prime.shape[0]
    
        x = np.matmul(x, np.ones((1,N_prime)))
        x_prime = np.matmul(np.ones((N,1)),x_prime.T)

        return lambda_k_ff(x, x_prime, sigma, theta, gamma)   

    
    ########################## Return ##################################

    return k_uu, k_uf, k_ff
           
           