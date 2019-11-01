# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 20:14:03 2019

@author: samar
"""


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from pyDOE import lhs

pi=tf.constant(np.pi)

class NeuralNetwork:
    # Initialize the class
    def __init__(self, X1, X2, u1, dimension_array, lower_bound, upper_bound):
        
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        
        self.x1 = X1
        self.x2 = X2
        self.u1 = u1
        
        self.dimension_array = dimension_array
        
        # Initialize weights and biases
        self.weights, self.biases = self.initialize_weights(dimension_array)
        
        # tf placeholders and graph
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        
        
        self.x1_tf = tf.placeholder(tf.float32, shape=[None, self.x1.shape[1]])
        self.x2_tf = tf.placeholder(tf.float32, shape=[None, self.x2.shape[1]])
        self.u1_tf = tf.placeholder(tf.float32, shape=[None, self.u1.shape[1]])
                
        self.u1_predict = self.net_u(self.x1_tf)
        self.f_predict = self.net_f(self.x2_tf)
        
        #loss function
        self.loss_func = tf.reduce_mean(tf.square(self.u1_tf - self.u1_predict)) + \
                    tf.reduce_mean(tf.square(self.f_predict))
        
        self.optimizer_Adam = tf.train.AdamOptimizer()
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss_func)
        
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def initialize_weights(self, dimension_array):        #initialize weights according to xavier initialization
        weights = []
        biases = []
        n = len(dimension_array) 
        for i in range(0,n-1):
            W = self.xavier(size=[dimension_array[i], dimension_array[i+1]])
            b = tf.Variable(tf.zeros([1,dimension_array[i+1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)        
        return weights, biases
        
    def xavier(self, size):     #xavier initialization
        in_dim = size[0]
        out_dim = size[1]        
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)
    
    def forward_pass(self, X, weights, biases):         #function takes data weights and biases and runs forward pass
        n = len(weights) + 1
        H = 2.0*(X - self.lower_bound)/(self.upper_bound - self.lower_bound) - 1.0
        for i in range(0,n-2):
            W = weights[i]
            b = biases[i]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y
            
    def net_u(self, x):                     #Run neural network on boundary conditions
        u = self.forward_pass(x, self.weights, self.biases)
        return u
    
    def net_f(self, x):         #calculate value of f with lambda=1
        u = self.net_u(x)
        u_x = tf.gradients(u, x)[0]
        u_xx = tf.gradients(u_x, x)[0]
        f = u_xx - u + (pi**2+1)*(tf.sin(pi*x))
        return f

        
    def train(self, nIter):     #train model
        tf_dict = {self.x1_tf: self.x1, self.x2_tf: self.x2,  self.u1_tf: self.u1}
        for i in range(nIter):
            self.sess.run(self.train_op_Adam, tf_dict)
            # Print
            if i % 50 == 0:
                loss_value = self.sess.run(self.loss_func, tf_dict)
                print(i)
                print(loss_value)
        
    def predict(self, test_X):      #predict output for given data
        tf_dict = {self.x1_tf: test_X}
        u_star = self.sess.run(self.u1_predict, tf_dict)
        return u_star

    
if __name__ == "__main__": 
    dimension_array = [1, 50, 50, 1]
    data_points=200     #Nf
    lower_bound=-1
    upper_bound=1
    iterations=1000
    
    x1=np.array([[-1],[1]])     #boundary conditions for x
    u1=np.array([[0],[0]])      #u(x) at boundary conditions
    x2=2*lhs(1, data_points)-1  #generate Nf data points
    
    NN = NeuralNetwork(x1,x2, u1, dimension_array, lower_bound, upper_bound)
    NN.train(iterations)     #train model 
    
    #Test model
    test_X=np.transpose(np.atleast_2d(np.linspace(-1,1,120)))  #generate 120 data points for testing
    u_predict = NN.predict(test_X)      #use the model to predict u(x)
    u_exact=np.sin(np.pi*test_X)        #Find exact values of u(x)

 
        #find L2 error 
    L2=np.sqrt(np.sum((u_exact-u_predict)**2))/np.sqrt(np.sum((u_exact)**2))
    print(L2)
    
    #plot exact u(x) vs predicted u(x)
    plt.figure(1, figsize=(15,6))
    plt.subplot(1,2,1)
    plt.scatter(u_exact, u_predict)
    plt.legend()
    plt.xlabel('$exact_solution$')
    plt.ylabel('$predicted_solution$')
    
    #Plot exact u(x) and predicted u(x) against data points
    plt.figure(2, figsize=(8,6))
    plt.subplot(1,2,1)
    plt.plot(test_X, u_exact, linewidth=4.0,  label = 'u_exact', c="r")
    plt.plot(test_X, u_predict, linewidth=3.0, label = 'u_predict', c="b")
    plt.legend()
    plt.xlabel('$x$')
    plt.ylabel('$u(x)$')