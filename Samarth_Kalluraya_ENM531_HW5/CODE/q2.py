# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 13:03:26 2019

@author: samar
"""


import torch
import numpy as np
import torch.utils.data
import torch.nn.functional as F
from torch.autograd import Variable
import timeit
from scipy.integrate import odeint

import matplotlib.pyplot as plt

class RNN:
    # Initialize the class
    def __init__(self, X, Y, hidden_dim):
        

        self.dtype = torch.FloatTensor
        
        # X has the form lags x data x dim
        # Y has the form data x dim

        # Define PyTorch variables
        X = torch.from_numpy(X).type(self.dtype)
        Y = torch.from_numpy(Y).type(self.dtype)
        self.X = Variable(X, requires_grad=False)
        self.Y = Variable(Y, requires_grad=False)
                 
        self.X_dim = X.shape[-1]
        self.Y_dim = Y.shape[-1]
        self.hidden_dim = hidden_dim
        self.lags = X.shape[0]
        
        # Initialize network weights and biases        
        self.Uo,self.bo,self.Wo,self.Us,self.bs,self.Ws,self.Ui,self.bi,self.Wi,self.Uf,self.bf,self.Wf, self.V, self.c = self.initialize_RNN()
                
        # Store loss values
        self.training_loss = []
      
        # Define optimizer
        self.optimizer = torch.optim.Adam([self.Uo,self.bo,self.Wo,self.Us,self.bs,self.Ws,self.Ui,self.bi,self.Wi,self.Uf,self.bf,self.Wf, self.V, self.c], lr=1e-3)
    
    
    # Initialize network weights and biases using Xavier initialization
    def initialize_RNN(self):      
        # Xavier initialization
        def xavier_init(size):
            in_dim = size[0]
            out_dim = size[1]
            xavier_stddev = np.sqrt(2.0/(in_dim + out_dim))
            return Variable(xavier_stddev*torch.randn(in_dim, out_dim).type(self.dtype), requires_grad=True)
        
        Uo = xavier_init(size=[self.X_dim, self.hidden_dim])
        bo = Variable(torch.zeros(1,self.hidden_dim).type(self.dtype), requires_grad=True)
        Wo = Variable(torch.eye(self.hidden_dim).type(self.dtype), requires_grad=True)
         
        Us = xavier_init(size=[self.X_dim, self.hidden_dim])
        bs = Variable(torch.zeros(1,self.hidden_dim).type(self.dtype), requires_grad=True)
        Ws = Variable(torch.eye(self.hidden_dim).type(self.dtype), requires_grad=True)
         
        Ui = xavier_init(size=[self.X_dim, self.hidden_dim])
        bi = Variable(torch.zeros(1,self.hidden_dim).type(self.dtype), requires_grad=True)
        Wi = Variable(torch.eye(self.hidden_dim).type(self.dtype), requires_grad=True)
        
        Uf = xavier_init(size=[self.X_dim, self.hidden_dim])
        bf = Variable(torch.zeros(1,self.hidden_dim).type(self.dtype), requires_grad=True)
        Wf = Variable(torch.eye(self.hidden_dim).type(self.dtype), requires_grad=True)
        
        V = xavier_init(size=[self.hidden_dim, self.Y_dim])
        c = Variable(torch.zeros(1,self.Y_dim).type(self.dtype), requires_grad=True)
        
        return Uo, bo, Wo, Us, bs, Ws, Ui, bi, Wi, Uf, bf, Wf, V, c
       
           
    # Evaluates the forward pass
    def forward_pass(self, X):
        H = torch.zeros(X.shape[1], self.hidden_dim).type(self.dtype)
        St = torch.zeros(X.shape[1], self.hidden_dim).type(self.dtype)
        
        for i in range(0, self.lags):
            ft = F.sigmoid((torch.matmul(H,self.Wf) + torch.matmul(X[i,:,:],self.Uf) + self.bf) )
            it = F.sigmoid((torch.matmul(H,self.Wi) + torch.matmul(X[i,:,:],self.Ui) + self.bi) )
            S_t = F.tanh((torch.matmul(H,self.Ws) + torch.matmul(X[i,:,:],self.Us) + self.bs) )
            ot = F.sigmoid((torch.matmul(H,self.Wo) + torch.matmul(X[i,:,:],self.Uo) + self.bo) )
            St = ft*St+it*S_t
            H = ot*F.tanh(St)       
        H = torch.matmul(H,self.V) + self.c
        return H
    
    
    # Computes the mean square error loss
    def compute_loss(self, X, Y):
        loss = torch.mean((Y - self.forward_pass(X))**2)
        return loss
        
    # Fetches a mini-batch of data
    def fetch_minibatch(self,X, y, N_batch):
        N = X.shape[1]
        idx = torch.randperm(N)[0:N_batch]
        X_batch = X[:,idx,:]
        y_batch = y[idx,:]        
        return X_batch, y_batch
    
    
    # Trains the model by minimizing the MSE loss
    def train(self, nIter = 10000, batch_size = 128):
        
        start_time = timeit.default_timer()
        for it in range(nIter):
            # Fetch mini-batch
            X_batch, Y_batch = self.fetch_minibatch(self.X, self.Y, batch_size)
            
            loss = self.compute_loss(X_batch, Y_batch)
            
            # Store loss value
            self.training_loss.append(loss)
            
            # Backward pass
            loss.backward()
            
            # update parameters
            self.optimizer.step()
            
            # Reset gradients for next step
            self.optimizer.zero_grad()
            
            # Print
            if it % 100 == 0:
                elapsed = timeit.default_timer() - start_time
                print('It: %d, Loss: %.3e, Time: %.2f' % 
                      (it, loss.cpu().data.numpy(), elapsed))
                start_time = timeit.default_timer()
    
    
   # Evaluates predictions at test points    
    def predict(self, X_star):
        X_star = torch.from_numpy(X_star).type(self.dtype)
        y_star = self.forward_pass(X_star)
        y_star = y_star.cpu().data.numpy()
        return y_star
    


#####################################

np.random.seed(1234)
    
if __name__ == "__main__":
    
    #create lags in  dataset
    def create_dataset(data, lags):
        N = len(data)-lags
        data_dim = data.shape[1]
        X = np.zeros((lags, N, data_dim))
        Y = np.zeros((N, data_dim))
        for i in range(0,N):
            X[:,i,:] = data[i:(i+lags), :]
            Y[i,:] = data[i + lags, :]
        return X, Y
    
    #standardize the data
    def normalize(data):            
        man=data.mean(0)
        std=data.std(0)
        y=data-man
        z=y/std
        return z,man,std

    #define the ode equation
    def population(h,t):
        a=1
        b=0.1
        c=1.5
        d=0.75
        x=h[0]
        y=h[1]
        dxdt=x*(a-b*y)
        dydt=-y*(c-d*x)
        return [dxdt, dydt]
    
    #define initial conditions
    xo=10
    yo=5
    t=np.linspace(0,60,2000)
    # generate the dataset
    dataset=odeint(population,[xo,yo],t)
    #standardize the data
    data,mean_data,std_data=normalize(dataset)
    # Use 2/3 of all data as training Data
    train_size = int(len(data) * (2.0/3.0))
    train = data[0:train_size,:]
    
    # reshape X and Y
    # X has the form lags x data x dim
    # Y has the form data x dim
    lags = 8
    X, Y = create_dataset(train, lags)
    
    # Model creation
    hidden_dim = 20
    model = RNN(X, Y, hidden_dim)
    # Model train
    model.train(nIter = 20000, batch_size = 128)
    
    # Prediction
    pred = np.zeros((len(data)-lags, Y.shape[-1]))
    X_tmp =  np.copy(X[:,0:1,:])
    for i in range(0, len(data)-lags):
        pred[i] = model.predict(X_tmp)
        X_tmp[:-1,:,:] = X_tmp[1:,:,:] 
        X_tmp[-1,:,:] = pred[i]
    
    #denormalize the predicted data
    denorm_pred=pred*std_data+mean_data
    
    #plot the signals
    plt.figure(1)
    plt.plot(dataset[lags:,0], 'c-', linewidth = 1, label = "Exact X")
    plt.plot(dataset[lags:,1], 'y-', linewidth = 1, label = "Exact Y")
    plt.plot(denorm_pred[:,0], 'r--', linewidth = 1, label = "Prediction X")
    plt.plot(denorm_pred[:,1], 'm--', linewidth = 1, label = "Prediction Y")
    plt.plot(X.shape[1]*np.ones((2,1)), np.linspace(-.75,65,2), 'k--', linewidth=2)
    plt.axis('tight')
    plt.xlabel('$t$')
    plt.ylabel('$x and y$')
    plt.legend(loc='lower left')
    
    #calculate the L2 error
    act_xy=data[lags+train_size:,:]
    act_x=act_xy[:,0]
    act_y=act_xy[:,1]
    pred_x=pred[train_size:,0]
    pred_y=pred[train_size:,1]
    ltx=np.linalg.norm(act_x-pred_x)/np.linalg.norm(act_x)
    lty=np.linalg.norm(act_y-pred_y)/np.linalg.norm(act_y)