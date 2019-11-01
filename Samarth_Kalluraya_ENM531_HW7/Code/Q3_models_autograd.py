# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 18:18:15 2019

@author: samar
"""


import autograd.numpy as np
from autograd import value_and_grad
from scipy.optimize import minimize
from scipy.stats import norm

class HelmGP:
    # Initialize the class
    def __init__(self, X_u, y_u, X_f, y_f, 
                 k_uu, k_uf, k_ff): 
        
        self.D = X_f.shape[1]
        
        self.X_u = X_u
        self.y_u = y_u
        
        self.X_f = X_f
        self.y_f = y_f
        
        self.hyp = self.init_params()        
        
        self.k_uu = k_uu
        self.k_uf = k_uf
        
        self.k_ff = k_ff
                       
        self.jitter = 1e-8
        
        self.likelihood(self.hyp)
        print("Total number of parameters: %d" % (self.hyp.shape[0]))


    # Initialize hyper-parameters        
    def init_params(self):
        # Kernel hyper-parameters
        hyp = np.log(np.ones(self.D+1))
        # Noise variance
        logsigma_n = np.array([1, -4.0])
#        gamma=np.array([1.0])####################################################################
#        hyp = np.concatenate([hyp, gamma, logsigma_n])
        hyp = np.concatenate([hyp, logsigma_n])
        return hyp
    
    
    # Computes the negative log-marginal likelihood
    def likelihood(self, hyp):
        X_u = self.X_u
        y_u = self.y_u
        
        X_f = self.X_f
        y_f = self.y_f
        
        y = np.vstack((y_u, y_f))

        N = y.shape[0]
        N_f = y_f.shape[0]
        
        theta = hyp[:-1]
        sigma_n = np.exp(hyp[-1])
               
        K_uu = self.k_uu(X_u, X_u, theta)
        K_uf = self.k_uf(X_u, X_f, theta)
    
        K_ff = self.k_ff(X_f, X_f, theta) + np.eye(N_f)*sigma_n
        
        K = np.vstack((np.hstack((K_uu, K_uf)),
                       np.hstack((K_uf.T, K_ff))))
        
        L = np.linalg.cholesky(K + np.eye(N)*self.jitter) 
        self.L = L
        
        alpha = np.linalg.solve(np.transpose(L), np.linalg.solve(L,y))    
        NLML = 0.5*np.matmul(np.transpose(y),alpha) + \
               np.sum(np.log(np.diag(L))) + 0.5*np.log(2.*np.pi)*N  
        return NLML[0,0]
    
            
    #  Prints the negative log-marginal likelihood at each training step         
    def callback(self,params):
        print("Log likelihood {}".format(self.likelihood(params)))
        

    # Minimizes the negative log-marginal likelihood using L-BFGS
    def train(self):
        result = minimize(value_and_grad(self.likelihood), self.hyp, jac=True, 
                          method='L-BFGS-B', callback=self.callback)
        self.hyp = result.x
        
        
    # Return posterior mean and variance at a set of test points
    def predict_u(self,X_star):
        X_u = self.X_u
        y_u = self.y_u
        
        X_f = self.X_f
        y_f = self.y_f
        
        y = np.vstack((y_u, y_f))

        L = self.L
                
        theta = self.hyp[:-1]
        
        K_uu = self.k_uu(X_star, X_u, theta)
        K_uf = self.k_uf(X_star, X_f, theta)
        psi = np.hstack((K_uu, K_uf))
        
        alpha = np.linalg.solve(np.transpose(L), np.linalg.solve(L,y))
        pred_u_star = np.matmul(psi,alpha)

        beta = np.linalg.solve(np.transpose(L), np.linalg.solve(L,psi.T))
        var_u_star = self.k_uu(X_star, X_star, theta) - np.matmul(psi,beta)
        
        if isinstance(pred_u_star, np.ndarray) == False:
            pred_u_star = pred_u_star._value
        if isinstance(var_u_star, np.ndarray) == False:
            var_u_star = var_u_star._value
        
        return pred_u_star, var_u_star
    
    
    # Return posterior mean and variance at a set of test points
    def predict_f(self,X_star):
        X_u = self.X_u
        y_u = self.y_u
        
        X_f = self.X_f
        y_f = self.y_f
        
        y = np.vstack((y_u, y_f))

        L = self.L

        theta = self.hyp[:-1]
        
        K_uf = self.k_uf(X_u, X_star, theta)
        K_ff = self.k_ff(X_star, X_f, theta)
        psi = np.hstack((K_uf.T, K_ff))
        
        alpha = np.linalg.solve(np.transpose(L), np.linalg.solve(L,y))
        pred_u_star = np.matmul(psi,alpha)

        beta = np.linalg.solve(np.transpose(L), np.linalg.solve(L,psi.T))
        var_u_star = self.k_ff(X_star, X_star, theta) - np.matmul(psi,beta)
        
        if isinstance(pred_u_star, np.ndarray) == False:
            pred_u_star = pred_u_star._value
        if isinstance(var_u_star, np.ndarray) == False:
            var_u_star = var_u_star._value
    
        return pred_u_star, var_u_star
    
    
    def DE_UCB_max(self, X_star):      
        u, v_u = self.predict_u(X_star)
        f, v_f = self.predict_f(X_star)      
        v_u = np.abs(np.diag(v_u))[:,None]
        v_f = np.abs(np.diag(v_f))[:,None]
        nn = np.linalg.norm(np.sqrt(v_u))/np.linalg.norm(np.sqrt(v_f))
        acq = u + (np.sqrt(v_f)*nn + np.sqrt(v_u))
        if isinstance(acq, np.ndarray) == False:
            acq = acq._value
        return acq
    
    
    def DE_UCB_min(self, X_star):      
        u, v_u = self.predict_u(X_star)
        f, v_f = self.predict_f(X_star)      
        v_u = np.abs(np.diag(v_u))[:,None]
        v_f = np.abs(np.diag(v_f))[:,None]
        nn = np.linalg.norm(np.sqrt(v_u))/np.linalg.norm(np.sqrt(v_f))
        acq = u - (np.sqrt(v_f)*nn + np.sqrt(v_u))
        if isinstance(acq, np.ndarray) == False:
            acq = acq._value
        return acq
 