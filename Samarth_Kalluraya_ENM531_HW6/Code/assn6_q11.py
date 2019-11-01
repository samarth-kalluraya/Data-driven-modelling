# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 17:22:38 2019

@author: samar
"""
import matplotlib.pyplot as plt
import numpy as np
import math


def gaussian(x, mu, sig):
    return (np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.))))*(1/(sig*math.sqrt(2*np.pi)))

def p_of_x(x, mu1, sig1, mu2, sig2):
    return (np.exp(-np.power(x - mu1, 2.) / (2 * np.power(sig1, 2.))))*(1/(sig1*math.sqrt(2*np.pi)))+(np.exp(-np.power(x - mu2, 2.) / (2 * np.power(sig2, 2.))))*(1/(sig2*math.sqrt(2*np.pi)))


mu1, sigma1 = 30, 10 # mean and standard deviation
mu2, sigma2 = 80, 20 # mean and standard deviation
mu_q, sigma_q = 60, 20 # mean and standard deviation for q(x)
N=10 #number of points to be iid from q(x)
x_values = np.linspace(-3, 150, 1000)

#Monte Carlo mean
set1 = np.random.normal(mu1, sigma1, 50000)
set2 = np.random.normal(mu2, sigma2, 50000)
set1=set1[:,None]
set2=set2[:,None]

data_set=np.concatenate((set1, set2), axis=0)

mean_data=np.mean(data_set)
#print(mean_data)
var1=np.var(data_set)
print(mean_data)
print(var1)
print("...")
#Importance sampling
iid_from_q=np.random.normal(mu_q, sigma_q, N)
iid_from_q=iid_from_q[:,None]
w_tilda=p_of_x(iid_from_q,mu1,sigma1,mu2,sigma2)/gaussian(iid_from_q, mu_q, sigma_q)
mean_w_tilda=np.mean(w_tilda)

#w_hat=w_tilda/mean_w_tilda
fxwt=iid_from_q*w_tilda
mean_imp=np.mean(fxwt)/mean_w_tilda
print(mean_imp)

fxwt_var=w_tilda*(iid_from_q-mean_imp)**2
var2=np.mean(fxwt_var)/mean_w_tilda
print(var2)