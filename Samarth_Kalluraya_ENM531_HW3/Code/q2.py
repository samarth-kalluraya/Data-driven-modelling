# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 11:35:50 2019

@author: samarth
"""

import autograd.numpy as np
import matplotlib.pyplot as plt
from pyDOE import lhs
from scipy.stats import norm
from scipy.special import legendre
from autograd import grad
from sklearn.metrics import confusion_matrix

#obtain data from csv file
def get_data():
    all_data=np.genfromtxt("Data_for_UCI_named.csv", delimiter=",")
    data=all_data[1:,:-1]
    n=np.size(data,0)
#    df = pd.read_csv("Data_for_UCI_named.csv")
#    saved_column = df['stabf'] 
    #y=np.zeros([n,1])
    bias_ones=np.ones([n,1])
    data=np.concatenate((bias_ones, data), axis=1)
    for i in range(n):
        if data[i,-1]<0:
            data[i,-1]=0
        else:
            data[i,-1]=1
    return data

#get first 2/3 data for training model
def get_training_data(data):
    n=np.size(data,0)
    n=2*n/3
    n=int(np.round(n))
    train_data=data[:n,:]
    return train_data

#get last 1/3 input data for testing model
def get_test_data(data):
    n=np.size(data,0)
    n=2*n/3
    n=int(np.round(n))
    test_data=data[n:,:-1]
    return test_data

#get last 1/3 output data for checking model
def get_test_data_result(data):
    n=np.size(data,0)
    n=2*n/3
    n=int(np.round(n))
    y=data[n:,-1]
    return y

#sigmoid funtion
def sigmoid(w,x):
    wx = np.matmul(x,w)
    temp=1/(1+np.exp(-wx))
    return temp

def loss_func(parameters,data):
    parameters=np.squeeze(parameters)
    x=data[:,:-1]
    y=data[:,-1]
    ai=sigmoid(parameters,x)
    loss=-(np.sum(y*np.log(ai)+(1-y)*np.log(1-ai)))
    return loss

class AdamOptimizer:
    
    def __init__(self, data, loss_func,mn,vn,b1,b2,batch_size, learning_rate,iteration):
      self.data = data
      self.X=self.data[:,:-1]           #X=feature matrix
      self.Y=self.data[:,-1]            #Y=result matrix
      self.M = np.size(self.X,1)        #number of features
      self.N = np.size(self.data,0)     #number of data points
      self.batch_size=batch_size        #batch size
      self.loss_func=loss_func
      self.b2 = b2         # adam update parameter
      self.b1 = b1         # adam update parameter
      self.vn = vn         # adam update parameter
      self.mn = mn         # adam update parameter
      
      self.learning_rate=learning_rate
      self.grad_func=grad(self.loss_func,argnum=0)
      self.loss_array=[]
      self.iter=iteration

    #generate initial parameter guess
    def parameterize(self): 
      params=np.zeros([self.M,1])
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
                
                #update parameters
                self.params=self.params-(self.learning_rate*mnhat)/(np.sqrt(vnhat)+0.00000001)
                
                #store value of loss
                self.loss_array=np.concatenate((self.loss_array,[self.loss_func(self.params,self.data)]),axis=0)
                #print(self.loss_func(self.params,self.data))
                i=i+1
                print(i)
        return(self.params) 
# 
#
if __name__ == "__main__":
    learning_rate=0.001
    batch_size_variable=32
    iterations=20000
    mn=0         # adam update parameter
    vn=0         # adam update parameter
    b1=0.9         # adam update parameter
    b2=0.999         # adam update parameter99
    
    #get data from csv file
    all_data=get_data()
    
    #get training data 
    training_data=get_training_data(all_data)
    
    #( data, loss_func,mn,vn,b1,b2,batch_size, learning_rate,iteration)
    adam=AdamOptimizer( training_data, loss_func,mn,vn,b1,b2,batch_size_variable, learning_rate,iterations)
    paras=adam.parameterize()     ##initialize parameters\
    
    #implement adam optimizer
    parameters=adam.find_params()
    
    #get test data 
    test_data=get_test_data(all_data)
    test_data_desired=get_test_data_result(all_data)
    
    #predict the output
    test_data_predict=np.matmul(test_data, parameters)
    
    #determine if the output is stable or unstable
    test_data_size=np.size(test_data_predict,0)
    for i in range(test_data_size):
        if test_data_predict[i]<0.5:   
            test_data_predict[i]=0   #stable
        else:
            test_data_predict[i]=1   #unstable
    
    #find confusion matrix        
    confuse=confusion_matrix(test_data_desired, test_data_predict)
    
    #find accuracy
    accuracy=confuse.trace()/confuse.sum()
    print(accuracy)
    
    #find final loss value
    loss=adam.loss_array[-1]
    
    #plot loss vs iteration
    plt.figure(1, figsize=(15,6))
    plt.subplot(1,2,1)
    x_axis=np.arange(np.size(adam.loss_array,0))
    plt.plot(x_axis, adam.loss_array, linewidth=3.0, label = 'loss ')
    plt.legend()
    plt.xlabel('$iterations$')
    plt.ylabel('$loss$')
    plt.axis('tight')