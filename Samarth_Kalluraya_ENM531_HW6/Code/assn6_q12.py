# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 15:41:40 2019

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

mu_q, sigma_q = 50, 30 # mean and standard deviation for q(x)

N=20000 #number of points to be iid from q(x)


x_values = np.linspace(-50, 150, 5000)
c=round(np.amax(p_of_x(x_values,mu1,sigma1,mu2,sigma2)/gaussian(x_values, mu_q, sigma_q)))
px=p_of_x(x_values,mu1,sigma1,mu2,sigma2)
gx=gaussian(x_values, mu_q, sigma_q)

sample=np.array([])
sample_count=0
i=0
while sample_count<N:
    i=i+1
    iid_from_q=np.random.normal(mu_q, sigma_q)
    Y=np.random.uniform(0,c*gaussian(iid_from_q, mu_q, sigma_q))
    if Y<p_of_x(iid_from_q,mu1,sigma1,mu2,sigma2):
        sample=np.append(sample,[iid_from_q],axis=0)
        sample_count=sample_count+1



set1 = np.random.normal(mu1, sigma1, 10000)
set2 = np.random.normal(mu2, sigma2, 10000)
set1=set1[:,None]
set2=set2[:,None]

data_set=np.concatenate((set1, set2), axis=0)

#
plt.figure(1, figsize=(15,6))
plt.hist(data_set, normed=True, bins=100)
plt.ylabel('Probability');
plt.hist(sample, normed=True, bins=100)
plt.legend(("samples from p(x)","samples obtained by rejection sampling",))
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
x_values = np.linspace(-3, 150, 1000)
norm1=gaussian(x_values,30,10)
norm2=gaussian(x_values,80,20)
norm12=p_of_x(x_values,80,20,30,10)
q=gaussian(x_values,55,20)
#plt.plot(x_values, 4*q, linewidth=3.0, label = 'norm 1 ')
plt.plot(x_values, norm12, linewidth=3.0, label = 'norm 12 ')
plt.plot(x_values, norm12/2, linewidth=3.0, label = 'norm 12 ')
plt.legend(("p(x)","normalized p(x)"))
#%%%%%%%%%%%%%%%%%%%%%%%%%
print(np.mean(sample))
print(np.mean(data_set))