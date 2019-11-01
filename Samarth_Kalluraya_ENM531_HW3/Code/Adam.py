# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 16:56:42 2019

@author: samar
"""

import autograd.numpy as np
from autograd import grad
from autograd import jacobian

import math

class AdamOptimizer:
    
    def __init__(self, X, Y, M, N, batch_size, loss_func,mn,vn,b1,b2,learning_rate,iter):
      
      self.X = X     #feature matrix
      self.Y = Y     #Output matrix
      self.M = M     #number of features
      self.N = N     #number of data points
      self.batch_size=batch_size   
      self.loss_func=loss_func
      self.b2 = b2
      self.b1 = b1 
      self.vn = vn
      self.mn = mn
      self.learning_rate=learning_rate
      self.data=np.concatenate((self.X, self.Y), axis=1)   #data=[feature matrix, output column]
      self.grad_func=grad(self.loss_func,argnum=0)
      self.loss_array=[]
      self.iter=iter

    #generate initial parameter guess
    def parameterize(self): 
      params=np.zeros([2*self.M+2,1])
      var=np.array([[0.001]])
      params=np.concatenate((params,var),axis=0)
      self.params = params
      return params
    
    #randomize the data points
    def randomize_batches(self):
        np.random.shuffle(self.data)
    
    #returns batches of desired batch size            
    def make_batch(self):
        return [self.data[i:i + self.batch_size, :] for i in range(0, self.N, self.batch_size)]
    
    #implements adams optimizer
    def find_params(self):
        i=1
        while i<self.iter+1:
            self.randomize_batches()
            for batchdata in self.make_batch():
                self.params=np.squeeze(self.params)
                gn=self.grad_func(self.params,batchdata)        #generate gradient
                gn2=gn**2
                
                self.mn=self.b1*self.mn+(1-self.b1)*gn
                self.vn=self.b2*self.vn+(1-self.b2)*gn2
                mnhat=(self.mn/(1-self.b1**i))
                vnhat=(self.vn/(1-self.b2**i))
                self.params=self.params-(self.learning_rate*mnhat)/(np.sqrt(vnhat)+0.00000001)
                
                self.params[-1]=np.exp(np.log(self.params[-1]))         #ensure variance is positive
                
                #store value of loss
                self.loss_array=np.concatenate((self.loss_array,[self.loss_func(self.params,self.data)]),axis=0)
                
                i=i+1
                
        return(self.params) 
        
       
    
            