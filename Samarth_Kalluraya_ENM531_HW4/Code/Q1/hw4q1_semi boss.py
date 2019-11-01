# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 19:19:31 2019
new class model ... gn1 gn2 gn3 w1 w2 w3
@author: samarth
"""

import autograd.numpy as np
import matplotlib.pyplot as plt
from pyDOE import lhs
from autograd import grad
import copy


def lhs_xy(n):          #geenerate obseravtions using latin hypercube sampling
    x=4*lhs(2, n)+50
    return x

def exact_output(data):     #given (x,y) find z=cos(pi*x)cos(pi*y)
    z=np.cos(np.pi*data[:,0])*np.cos(np.pi*data[:,1])
    z=z[:,None]
    return z

def normalize(data):        #standardize data with mean=0 
    mean=data.mean(0)
    std=data.std(0)
    y=data-mean
    z=y/std
    return z
        
def forward_pass(xy,w1,w2,w3):      #compute the forward pass gicen the input data and the weights of each layer
    
    HL1=np.tanh(np.matmul(xy,w1[:-1,:])+w1[-1,:])           #next_layer=prev_layer*weights+bias
    HL2=np.tanh(np.matmul(HL1,w2[:-1,:])+w2[-1,:])
    z=(np.matmul(HL2,w3[:-1])+w3[-1])
    return z

#compute the loss function gicen the input and output data and the weights of each layer
def loss_func(w1,w2,w3,data):       
    xy=data[:,:-1]
    z=data[:,2]
    w1=np.squeeze(w1)
    w2=np.squeeze(w2)
    w3=np.squeeze(w3)
    z_pred=forward_pass(xy,w1,w2,w3)
    loss=np.sum((z_pred-z)**2)
    return loss

class Tunable_Parameters:       #Model parameters
    def __init__(self,mn,vn,b1,b2,batch_size, learning_rate,iteration):
        self.mn=mn
        self.vn=vn
        self.b1=b1
        self.b2=b2
        self.b=batch_size
        self.l=learning_rate
        self.iter=iteration
        
class AdamOptimizer:        #Optimizer
    def __init__(self, tunable_params):
      assert (isinstance(tunable_params, Tunable_Parameters))
      #Tunable_Parameters=(self,mn,vn,b1,b2,b=batch_size, l=learning_rate,iter=iteration):
      self.tp=copy.deepcopy(tunable_params)

    #update weights
    def update_weights(self,weight,gn,j):   #(weight, gradient, iteration number)
        self.tp.mn=self.tp.b1*self.tp.mn+(1-self.tp.b1)*gn
        self.tp.vn=self.tp.b2*self.tp.vn+(1-self.tp.b2)*(gn**2)
        mnhat=(self.tp.mn/(1-self.tp.b1**j))
        vnhat=(self.tp.vn/(1-self.tp.b2**j))
        weight=weight-(self.tp.l*mnhat)/(np.sqrt(vnhat)+0.00000001)
        return (weight)

               
class NeuralNetwork:    
    #implements Neural Network
    def __init__(self, loss_func, dim_array, tunable_params):
        
        assert (isinstance(tunable_params, Tunable_Parameters))
        #Tunable_Parameters=(self,mn,vn,b1,b2,b=batch_size, l=learning_rate,iter=iteration):
        self.tp=tunable_params
        self.loss_func=loss_func
        self.dim_array=dim_array
        
        self.grad_func_w1=grad(self.loss_func,argnum=0)     #gradient of loss function with respect to w1
        self.grad_func_w2=grad(self.loss_func,argnum=1)     #gradient of loss function with respect to w2
        self.grad_func_w3=grad(self.loss_func,argnum=2)     #gradient of loss function with respect to w3
        self.loss_array=[]
        
        self.a1=AdamOptimizer(tunable_params)       #tuning parameters for w1
        self.a2=AdamOptimizer(tunable_params)       #tuning parameters for w2
        self.a3=AdamOptimizer(tunable_params)       #tuning parameters for w3
        
    def xavier(self,in_dim, out_dim):       #Xavier initialization
        w=np.random.normal(0,(2/(in_dim+out_dim)),(in_dim+1,out_dim))
        return w
    
    def initialize_weights(self):           #Initialize weights and biases 
        #Weights and biases are stored in a single matrix w=[[weights],[biases]]
        self.w1=(self.xavier(self.dim_array[0],self.dim_array[1]))
        self.w2=(self.xavier(self.dim_array[1],self.dim_array[2]))
        self.w3=(self.xavier(self.dim_array[2],self.dim_array[3]))
        return self.w1,self.w2,self.w3
    
    #returns batches of desired batch size                 
    def make_batch(self,data):
        return [data[i:i + self.tp.b, :] for i in range(0, data.shape[0], self.tp.b)]
    
    #training function
    def find_weights(self,data):
        i=1
        while i<self.tp.iter+1:
            np.random.shuffle(data)
            for batchdata in self.make_batch(data):
                
                gn1=self.grad_func_w1(self.w1,self.w2,self.w3,batchdata)        #generate gradient
                gn2=self.grad_func_w2(self.w1,self.w2,self.w3,batchdata)        #generate gradient
                gn3=self.grad_func_w3(self.w1,self.w2,self.w3,batchdata)        #generate gradient
                
                self.w1=self.a1.update_weights(self.w1,gn1,i)       #Update weights
                self.w2=self.a2.update_weights(self.w2,gn2,i)       #Update weights
                self.w3=self.a3.update_weights(self.w3,gn3,i)       #Update weights

                #store value of loss
                self.loss_array=np.concatenate((self.loss_array,[self.loss_func(self.w1,self.w2,self.w3,data)]),axis=0)
#                print(self.loss_func(self.params,self.data))
                i=i+1
                if i%100==0:
                    print(i)
                    
    def get_weights(self):
        return self.w1,self.w2,self.w3
    
    
if __name__ == "__main__":
    observations=500        #training points
    learning_rate=0.0002    #learning rate
    batch_size_variable=observations
    iterations=20000
    mn=0         # adam update parameter
    vn=0         # adam update parameter
    b1=0.9         # adam update parameter
    b2=0.999         # adam update parameter99
    
    #neural network
    Layer1=50
    Layer2=50
    
    
    #(self,mn,vn,b1,b2,batch_size, learning_rate,iteration) 
    tuning_params=Tunable_Parameters(mn,vn,b1,b2,batch_size_variable,learning_rate,iterations)
    
    x_y=lhs_xy(observations)    #generate training data
    exact_z=exact_output(x_y)   #generate exact solution for training data
    x_y=normalize(x_y)          #standardize data
    
    train_data=np.concatenate((x_y,exact_z),axis=1) #store input and output in a single matrix
    
    dimension_array=[x_y.shape[1],Layer1,Layer2,exact_z.shape[1]]   #nodes in each layer
    
    NN=NeuralNetwork(loss_func,dimension_array,tuning_params)       #create neural network object
    
    (w1,w2,w3)=NN.initialize_weights()              #initialize weights
    NN.find_weights(train_data)                     #train model to find weights
    (final_w1,final_w2,final_w3)=NN.get_weights()   #store trained weights
    
    test_xy=lhs_xy(200)                 #gnerate test data
    test_exact_z=exact_output(test_xy)  #generate exact solution for test data
    test_xy=normalize(test_xy)          #standardize data
    
    #run forward pass for test data on trained weights
    test_pred_z=forward_pass(test_xy,final_w1,final_w2,final_w3)
    
    #find L2 error 
    L2=np.sqrt(np.sum((test_exact_z-test_pred_z)**2))/np.sqrt(np.sum((test_exact_z)**2))
    print(L2)
    
    #plot Exact output vs Predicted output
    plt.figure(1, figsize=(15,6))
    plt.subplot(1,2,1)
    plt.scatter(test_exact_z, test_pred_z)
    plt.legend()
    plt.xlabel('$Exact output$')
    plt.ylabel('$Predicted output$')

    plt.figure(2, figsize=(15,6))
    plt.subplot(1,2,1)
    x_axis=np.arange(NN.tp.iter)
    plt.plot(x_axis, NN.loss_array, linewidth=3.0, label = 'loss ')
    plt.legend()
    plt.xlabel('$iterations$')
    plt.ylabel('$loss$')
    plt.axis('tight')
    
    loss=NN.loss_array