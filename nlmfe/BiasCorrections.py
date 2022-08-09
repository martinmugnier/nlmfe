# -*- coding: utf-8 -*-
# Ensure "normal" division
from __future__ import division

# Load library dependencies
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
import func

class BiasCorrections():
    
    """
    BiasCorrections:
        This class implements analytical corrections introduced in 
        Fernandez Val & Weidner (2016).
    
    INPUTS:
    ------
    `A'               : N x T x K (Numpy) array of exogeneous variables;
    `b'               : N x T (Numpy) array of a scalar endogeneous variable;
    `model'           : string ("logit" or "probit");
                             
    FUNCTIONS:
    ---------
    `compute_matrix_res_proj_cov'
    `bias_correct''
    
    """
    
    def __init__(self, A, b, model='logit'):
        self.A = A
        self.b = b
        self.n, self.T, self.K = A.shape
        if model=='logit':
            self.f = func.logit_pdf
            self.F = func.logit_cdf
            self.fprime = lambda x: self.F(x)*(1-self.F(x))*(1-2*self.F(x))
        elif model=='probit':
            self.f = func.probit_pdf
            self.F = func.probit_cdf
            self.fprime = lambda x: -x*self.f(x)
        
        self.H = lambda x: self.f(x) / (self.F(x)*(1-self.F(x)))
        self.omega = lambda x: self.H(x)*self.f(x)
                        
    def compute_matrix_res_proj_cov(self, A, omega):
        res = np.zeros((self.n, self.T, self.K))
        mat = sparse.hstack([sparse.kron(sparse.eye(self.n), 
            sparse.csc_matrix(np.ones(self.T).reshape(self.T,1))),
            sparse.kron(sparse.csc_matrix(np.ones(self.n).reshape(self.n,1)),
                            sparse.eye(self.T))])
        mat_to_inv = sparse.csc_matrix.dot(mat.transpose(), \
                                            sparse.diags(omega)).dot(mat)
        for k in range(self.K):
            coef_k = spsolve(mat_to_inv, sparse.csc_matrix.dot(
                mat.transpose().dot(sparse.diags(omega)), sparse.csc_matrix(
                A[:,:,k].reshape(self.n*self.T,1)))).reshape(self.n + self.T)
            res[:,:,k] = A[:,:,k]-coef_k[:self.n,None]-coef_k[None, self.n:]
        return res
    
    def bias_correct(self, estimates, avar=True, APE=False, L=3, APE_vec=0):
        # compute indexes
        A = self.A
        b = self.b
        beta = estimates[:self.K]
        alpha = np.insert(estimates[self.K: self.K + (self.n - 1)], 0, 0) 
        xi = estimates[self.K + (self.n-1):]
        mat = A.dot(beta) + alpha[:,None] + xi[None,:]
        # compute weights
        weights = self.omega(mat)
        X_tilde_hat = self.compute_matrix_res_proj_cov(A, 
                                    np.sqrt(weights).reshape(self.n*self.T))
        mat_fprime = self.fprime(mat)
        mat_F = self.F(mat)
        H = self.H(mat)
        trimed_est = np.zeros((self.n, self.K))
        for j in range(L):
            for t in range(j+1,self.T):
                trimed_est += (self.T)/(self.T-j) * (H[:,t-j]*(b[:,t-j]
                    - mat_F[:,t-j])*weights[:,t])[:,None]*X_tilde_hat[:,t,:]
        B_hat = - (1/(2*self.n)) * np.sum((np.sum((H*mat_fprime)[:,:,None]
                    *X_tilde_hat, axis=1) + 2 * trimed_est)
                                    /np.sum(weights, axis=1)[:, None], axis=0)
        D_hat = - (1/(2*self.T)) * np.sum(np.sum((H*mat_fprime)[:,:,None]
            *X_tilde_hat, axis=1) / np.sum(weights,axis=1)[:, None], axis=0)
        W_hat = np.mean(weights[:,:,None, None]*X_tilde_hat[:,:,:,None]
                        *X_tilde_hat[:,:,None,:], axis=(0,1))
        W_hat_inv = np.linalg.inv(W_hat)
        beta_tilde =  beta + W_hat_inv.dot(B_hat)/self.T \
                       + W_hat_inv.dot(D_hat)/self.n
        
        if APE:
            if True*APE==0:
                print("Need vector of estimated APE's.")
            else:
                delta = APE_vec # need to implement the partial optimization pb
                B_delt = 0 # TBD
                D_delt = 0 # TBD
                delta_tilde = 0 #TBD
        if avar:
            return beta_tilde, W_hat_inv
        else:
            return beta_tilde
            
            
        