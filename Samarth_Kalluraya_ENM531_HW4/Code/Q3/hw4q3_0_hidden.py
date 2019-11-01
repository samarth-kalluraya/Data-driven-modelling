# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 11:35:50 2019

@author: samar
"""

import autograd.numpy as np
import matplotlib.pyplot as plt
from pyDOE import lhs
from scipy.stats import norm
from scipy.special import legendre
from autograd import grad
from sklearn.metrics import confusion_matrix
#import pandas as pd

def get_data():                                                            #get all data input and output from the csv file
    all_data=np.genfromtxt("Data_for_UCI_named.csv", delimiter=",")
    data=all_data[1:,:-1]
    n=np.size(data,0)
    for i in range(n):
        if data[i,-1]<0:
            data[i,-1]=0
        else:
            data[i,-1]=1
    return data

def get_training_data(data):           #get all data input  and output for training from the csv file
    n=np.size(data,0)
    n=2*n/3
    n=int(np.round(n))
    train_data=data[:n,:]
    return train_data

def get_test_data(data):           #get all data input for testing from the csv file
    n=np.size(data,0)
    n=2*n/3
    n=int(np.round(n))
    test_data=data[n:,:-1]
    return test_data

def get_test_data_result(data):     #get all data output for testing from the csv file
    n=np.size(data,0)
    n=2*n/3
    n=int(np.round(n))
    y=data[n:,-1]
    return y

def normalize(data):            #standardize the dat
    man=data.mean(0)
    std=data.std(0)
    y=data-man
    z=y/std
    return z
        
def sigmoid(x):             #sigmoid function
    temp=1/(1+np.exp(-x))
    return temp

def relu(c):                #RELU function
    c=np.maximum(c,0)
    return c

def forward_pass(x,w1):                                  #function takes input data and weights and computes the forward pass
    n=x.shape[0]
    ones_nx1=np.ones([n,1])
    x=np.concatenate((x, ones_nx1), axis=1)    
    y=sigmoid(np.matmul(x,w1))
    return y

def loss_func(w1,data):                             #function takes all data and weights and computes the loss
    x=data[:,:-1]
    y=data[:,-1]
    y_pred=forward_pass(x,w1)
    loss=-(np.sum(y*np.log(y_pred)+(1-y)*np.log(1-y_pred)))
    return loss

def initialize_weights(x_y,z):                               #Initializes the weights  

    w1=np.random.normal(0,(2/(x_y.shape[1]+z.shape[1])),(x_y.shape[1]+1,z.shape[1]))
    return w1
    
class Tunable_Parameters:
    def __init__(self,mn,vn,b1,b2,batch_size, learning_rate,iteration):                     #Model tuning parameters
        self.mn1=mn
        self.vn1=vn
        self.b1=b1
        self.b2=b2
        self.b=batch_size
        self.l=learning_rate
        self.iter=iteration
        
        
class AdamOptimizer:              #implements neural network and adam optimizer
    def __init__(self, x, y, w1, loss_func, tunable_params):                       #*****************
          assert (isinstance(tunable_params, Tunable_Parameters))
          #Tunable_Parameters=(self,mn,vn,b1,b2,b=batch_size, l=learning_rate,iter=iteration):
          self.tp=tunable_params
          self.X=x           #X=feature matrix
          self.Y=y            #Y=result matrix
          self.data = np.concatenate((self.X,self.Y),axis=1) 
          self.N = np.size(self.X,0)
    
          self.loss_func=loss_func
          #weights
          self.w1=w1
          
      #gradient functions
          self.grad_func_w1=grad(self.loss_func,argnum=0)
          self.loss_array=[]


#  
    def randomize_batches(self):        #shuffles the data order
        np.random.shuffle(self.data)
#                
    def make_batch(self):               #creates batches
        return [self.data[i:i + self.tp.b, :] for i in range(0, self.N, self.tp.b)]
    
    def update_weights(self,weight,mn,vn,gn,j):     #runs adam optimizer and updates weights, and returns mn and vn of respective weights
        mn=self.tp.b1*mn+(1-self.tp.b1)*gn
        vn=self.tp.b2*vn+(1-self.tp.b2)*(gn**2)
        mnhat=(mn/(1-self.tp.b1**j))
        vnhat=(vn/(1-self.tp.b2**j))
        weight=weight-(self.tp.l*mnhat)/(np.sqrt(vnhat)+0.00000001)
        return (weight,mn,vn)
#    
        
        
    def find_weights(self):                       #trains the model on the neural network
        i=1
        while i<self.tp.iter+1:
            self.randomize_batches()
            for batchdata in self.make_batch():
                self.w1=np.squeeze(self.w1)
                
                gn1=self.grad_func_w1(self.w1, batchdata)        #generate gradient
                
                (self.w1,self.tp.mn1,self.tp.vn1)=self.update_weights(self.w1,self.tp.mn1,self.tp.vn1,gn1,i)        #Update weight, mn, vn
                
                self.loss_array=np.concatenate((self.loss_array,[self.loss_func(self.w1,self.data)]),axis=0)   #store value of loss at iteration        .. 
#                #print(self.loss_func(self.params,self.data))
                i=i+1
                if i%100==0:
                    print(i)
#                    print(self.loss_array[-1])
                    
    def get_weights(self):
        return self.w1
# 
#
if __name__ == "__main__":
    learning_rate=0.001
    batch_size_variable=32
    iterations=20000
    mn=0
    vn=0
    b1=0.9
    b2=0.999
    
    #neural network
    Layer1=0
    
    tuning_params=Tunable_Parameters(mn,vn,b1,b2,batch_size_variable,learning_rate,iterations)  #model tuning parameters
    
     
    all_data=get_data()                             #get all data
    training_data=get_training_data(all_data)       #get training data
    
    train_x=training_data[:,:-1]        #get inputs for training data
    train_x=normalize(train_x)          #standardize data
    train_y_exact=training_data[:,-1]   #get exact outputs for training data
    train_y_exact=train_y_exact[:,None]
    
    (w1)=initialize_weights(train_x,train_y_exact)                      #initialize weights
    
    
    #( data, loss_func,mn,vn,b1,b2,batch_size, learning_rate,iteration)
    adam=AdamOptimizer(train_x, train_y_exact, w1, loss_func, tuning_params)                  #create object
    adam.find_weights()                                                                               #train
    (final_w1)=adam.get_weights()                      #get trained weights
    
    
    
    test_data_x=get_test_data(all_data)                #get test data
    test_data_x=normalize(test_data_x)                 #standardize data
    test_data_desired=get_test_data_result(all_data)   #get exact outputs for test data
    
    test_data_predict=forward_pass(test_data_x,final_w1,final_w2,final_w3,final_w4)                     #run forward pass using trained weights
    
    test_data_size=np.size(test_data_predict,0)
    for i in range(test_data_size):
        if test_data_predict[i]<0.5:
            test_data_predict[i]=0
        else:
            test_data_predict[i]=1
            
    confuse=confusion_matrix(test_data_desired, test_data_predict)
    accuracy=confuse.trace()/confuse.sum()
    print(accuracy)
    
    plt.figure(1, figsize=(15,6))
    plt.subplot(1,2,1)
    x_axis=np.arange(np.size(adam.loss_array,0))
    plt.plot(x_axis, adam.loss_array, linewidth=3.0, label = 'loss ')
    plt.legend()
    plt.xlabel('$iterations$')
    plt.ylabel('$loss$')
    plt.axis('tight')