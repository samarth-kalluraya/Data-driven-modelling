# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 13:59:28 2019

@author: samar
"""

import matplotlib.pyplot as plt
import numpy as np

def get_beta0(y, x, beta1, tau, mu0, tau0):
    n=len(y)
    temp_prec=tau0+tau*n
    temp_mean=tau0*mu0+tau*(np.sum(y-beta1*x))
    mean=temp_mean/temp_prec
    temp_beta0=np.random.normal(mean,1/np.sqrt(temp_prec))
    return temp_beta0

def get_beta1(y, x, beta0, tau, mu1, tau1):
    temp_prec = tau1+tau * np.sum(x * x)
    temp_mean = tau1*mu1+tau*np.sum((y-beta0)*x)
    mean=temp_mean/temp_prec
    temp_beta1=np.random.normal(mean,1/np.sqrt(temp_prec))
    return temp_beta1

def get_tau(y, x, beta0, beta1, alpha, beta):
    alpha_new = alpha + N / 2
    temp_beta = y-beta0-beta1*x
    beta_new = beta + np.sum(temp_beta**2)/2
    temp_tau=np.random.gamma(alpha_new, 1 / beta_new)
    return temp_tau

def gibbs_sample(y, x, iters, init, hypers):
    assert len(y) == len(x)
    beta0 = init[0]
    beta1 = init[1]
    tau = init[2]
    
    store_data = np.zeros((iters, 3)) ## store_data to store values of beta0, beta1, tau
    
    for it in range(iters):
        beta0 = get_beta0(y, x, beta1, tau, hypers[0], hypers[1])
        beta1 = get_beta1(y, x, beta0, tau, hypers[2], hypers[3])
        tau = get_tau(y, x, beta0, beta1, hypers[4], hypers[5])
        store_data[it,:] = np.array((beta0, beta1, tau))
    return store_data
  
actual_beta0 = -1
actual_beta1 = 2
actual_tau = 1

N = 50
x = np.random.uniform(0, 4, N)
y = np.random.normal(actual_beta0 + actual_beta1 * x, 1 / np.sqrt(actual_tau))

# initial values
init=np.array([0,0,2]) #[beta0,beta1,tau]
## hyper parameters
hypers=np.array([0,1,0,1,2,1]) #[mu0,tau0,mu1,tau1,alpha,beta]

iters = 5000
store_data = gibbs_sample(y, x, iters, init, hypers)

iteration=x_values = np.linspace(1, iters, iters)

beta0_median=np.median(store_data[:,0])
beta1_median=np.median(store_data[:,1])
tau_median=np.median(store_data[:,2])
y_predict=np.random.normal(beta0_median + beta1_median * x, 1 / np.sqrt(tau_median))
plt.figure(1, figsize=(10,6))
plt.plot(x, y,"o")
plt.plot(x, y_predict,"o")
plt.legend(("actual points","predicted points"))

plt.figure(2, figsize=(10,6))
for i in range(50):
    y_line= store_data[4950+i,0]+store_data[4950+i,1]*x    
    plt.plot(x, y_line)

  
plt.figure(3, figsize=(10,6))
plt.hist(store_data[:,0], normed=True, bins=100)
plt.hist(store_data[:,1], normed=True, bins=100)
plt.hist(store_data[:,2], normed=True, bins=100)
plt.legend(("w0","w1","gamma"))
plt.ylabel('Probability');
#
#
#
#
#
#
