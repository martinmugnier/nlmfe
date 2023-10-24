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
        This class implements the analytical bias corrections introduced in 
        Fernandez Val & Weidner (2016).
    
    INPUTS:
    ------
    `A'               : N x T x K (Numpy) array of exogeneous variables;
    `b'               : N x T (Numpy) array of a scalar endogeneous variable;
    `model'           : string ("logit" or "probit");
                             
    FUNCTIONS:
    ---------
    `compute_weighted_FE_proj'
    `bias_correct'
    
    """
    
    def __init__(self, A, b, model='logit'):
        self.A = A
        self.b = b
        self.N, self.T, self.K = A.shape
        if model=='logit':
            self.f = func.logit_pdf
            self.F = func.logit_cdf
            self.fprime = lambda x: self.f(x)*(1-2*self.F(x))
            self.fprimeprime = lambda x: self.f(x)*((1-2*self.F(x))**2
                                                    -2*self.f(x))
        elif model=='probit':
            self.f = func.probit_pdf
            self.F = func.probit_cdf
            self.fprime = lambda x: -x*self.f(x)
            self.fprimeprime = lambda x: (x**2-1)*self.f(x)
        
        self.H = lambda x: self.f(x) / (self.F(x)*(1-self.F(x)))
        self.omega = lambda x: self.H(x)*self.f(x)
                        
    def compute_weighted_FE_proj(self, A, omega):
        
        """
        compute_weighted_FE_proj:
            compute the projection of matrix A on the space spanned by the 
            fixed effects under a metric given by the definite positive matrix 
            omega. 
            
        INPUTS:
        ------
        `A'            : N x T x K Numpy array;
        `omega'        : NT Numpy array.
        
        OUTPUTS:
        -------
        `res'      : projection of A.
        
        """
        res = np.zeros((self.N, self.T, self.K))
        mat = sparse.hstack([sparse.kron(sparse.eye(self.N), 
            sparse.csc_matrix(np.ones(self.T).reshape(self.T,1))),
            sparse.kron(sparse.csc_matrix(np.ones(self.N).reshape(self.N,1)),
                            sparse.eye(self.T))])
        mat_to_inv = sparse.csc_matrix.dot(mat.transpose(), \
                                            sparse.diags(omega)).dot(mat)
        for k in range(self.K):
            coef_k = spsolve(mat_to_inv, sparse.csc_matrix.dot(
                mat.transpose().dot(sparse.diags(omega)), sparse.csc_matrix(
                A[:,:,k].reshape(self.N*self.T,1)))).reshape(self.N+self.T)
            res[:,:,k] = coef_k[:self.N,None] + coef_k[None,self.N:]
        return res
    
    def bias_correct(self, estimates, avar=True, APE=False, L=3, APE_dum=[],
                     entire_pop=False):
        
        """
        bias_correct:
            compute bias-corrected slope estimate, bias-corrected average 
            partial effects (APE) estimate, an estimate of the asymptotic 
            variance-covariance matrix of the bias-corrected slope estimate,
            and an estimate of the matrix of standard errors of the 
            average partial effects estimates. 
            
        INPUTS:
        ------
        `estimates'    : K+(N-1)+T x 1 (Numpy array) of uncorrected estimates;
        `avar'         : 0/1, return asymptotic variances;
        `APE'          : 0/1, return vector of estimated APEs;
        `L'            : {0,1,2,3...}, trimming parameter;
        `APE_dum'      : subset of {1,...,K}, declare dummy variables;
        `entire_pop'   : 0/1, declare if the entire population is observed.
        
        OUTPUTS:
        -------
        `beta_tilde'      : bias corrected slope estimate;
        `W_hat_inv'       : variance-covariance matrix;
        `delta_tilde'     : bias corrected APEs;
        `delat_tilde_se'  : standard errors for bias-corrected APEs.
        
        """
        # compute indexes
        A = self.A
        b = self.b
        beta = estimates[:self.K]
        alpha = np.insert(estimates[self.K: self.K + (self.N - 1)], 0, 0) 
        xi = estimates[self.K + (self.N-1):]
        mat = A.dot(beta) + alpha[:,None] + xi[None,:]
        # compute weights
        weights = self.omega(mat)
        X_tilde_hat = A - self.compute_weighted_FE_proj(A, 
                                    np.sqrt(weights).reshape(self.N*self.T))
        mat_fprime = self.fprime(mat)
        mat_F = self.F(mat)
        H = self.H(mat)
        trimed_est = np.zeros((self.N, self.K))
        for j in range(L):
            for t in range(j+1,self.T):
                trimed_est += (self.T)/(self.T-j) * (H[:,t-j]*(b[:,t-j]
                    - mat_F[:,t-j])*weights[:,t])[:,None]*X_tilde_hat[:,t,:]
        B_hat = - (1/(2*self.N)) * np.sum((np.sum((H*mat_fprime)[:,:,None]
                    *X_tilde_hat, axis=1) + 2 * trimed_est)
                                    /np.sum(weights, axis=1)[:, None], axis=0)
        D_hat = - (1/(2*self.T)) * np.sum(np.sum((H*mat_fprime)[:,:,None]
            *X_tilde_hat, axis=1) / np.sum(weights,axis=1)[:, None], axis=0)
        W_hat = np.mean(weights[:,:,None, None]*X_tilde_hat[:,:,:,None]
                        *X_tilde_hat[:,:,None,:], axis=(0,1))
        W_hat_inv = np.linalg.inv(W_hat)
        beta_tilde =  beta + W_hat_inv.dot(B_hat)/self.T \
                       + W_hat_inv.dot(D_hat)/self.N
                       
        if APE:
            # TBA: Add decomposition by unit-specific or time-specific APE
            # -, alpha_bs, xi_bs = compute fixed effects estimates with beta 
                                # set to the bias-corrected estimate.
            # mat_bs = update single index to its bias-corrected value. 
            mat_bs = mat # for now use uncorrected estimates (see FVW 2016)
            delta = np.mean(self.f(mat_bs))*beta_tilde 
            delta_weighted_der = -(self.fprime(mat_bs)[:,:,None]
                                *beta_tilde[None, None,:])/weights[:,:,None]
            delta_second_der = self.fprimeprime(mat_bs)[:,:,None]\
                                    *beta_tilde[None,None,:]
            for k in APE_dum:
                mat_bs_without_k = mat_bs - A[:,:,k]*beta_tilde[k]
                delta[k] = self.F(beta_tilde[k]+mat_bs_without_k) \
                                -self.F(mat_bs_without_k)
                delta_weighted_der[:,:,k] = - (self.f(beta_tilde[k]
                                                      +mat_bs_without_k)
                            - self.f(mat_bs_without_k))/weights
                delta_second_der[:,:,k] = self.fprime(beta_tilde[k]
                    +mat_bs_without_k) - self.fprime(mat_bs_without_k)
            psi_hat = self.compute_weighted_FE_proj(delta_second_der, 
                                np.sqrt(weights).reshape(self.N*self.T))
            psi_tilde_hat = delta_weighted_der - psi_hat
            trimed_est_delta = np.zeros((self.N, self.K))
            for j in range(L):
                for t in range(j+1,self.T):
                    trimed_est += (self.T)/(self.T-j) * (H[:,t-j]*(b[:,t-j]
                     - mat_F[:,t-j])*weights[:,t])[:,None]*psi_tilde_hat[:,t,:]
            B_delt = - (1/(2*self.N)) * np.sum((np.sum(delta_second_der -
                (H*mat_fprime)[:,:,None]*psi_hat, axis=1) + 2*trimed_est_delta)
                                / np.sum(weights, axis=1)[:,None], axis=0) 
            D_delt = - (1/(2*self.T)) * np.sum(np.sum(delta_second_der -
                (H*mat_fprime)[:,:,None]*psi_hat, axis=1) 
                        / np.sum(weights,axis=1)[:,None], axis=0)
            delta_tilde = delta + B_delt/self.T + D_delt/self.N
            Delta_mat = self.f(mat_bs)[:,:,None]*beta_tilde[None,None,:] \
                                - delta[None,None,:]
            D_beta = np.reshape(np.mean(delta_weighted_der*X_tilde_hat
                                        *(-weights)), self.K)
            Gamma_mat = np.zeros((self.N,self.T,self.K))
            for i in range(self.N):
                for t in range(self.T):
                    Gamma_mat[i,t,:]= (np.transpose(D_beta).dot(W_hat_inv).dot(
                            *(H*(b-mat_F))[i,t]*X_tilde_hat[i,t]) - 
                            (H*(b-mat_F))[i,t]*psi_hat[i,t])
            term1 = 0
            term2 = 0
            term3 = 0
            if (1-entire_pop):
                for t in range(self.T):
                    for tau in range(self.T):
                        term1 += Delta_mat[:,t,:,None]*Delta_mat[:,tau,None,:]
                    term3 += Gamma_mat[:,t,:,None]*Gamma_mat[:,t,None,:]
                for i in range(self.N):
                    for j in np.delete(np.arange(self.N), i):
                        term2 += np.sum(Delta_mat[i,:,:,None]
                                        *Delta_mat[j,:,None,:], axis=0)
            delta_tilde_se = np.sqrt(np.sum(term1+term2+term3, axis=0)) \
                /(self.N*self.T)
        if avar*APE:
            return beta_tilde, W_hat_inv, delta_tilde, delta_tilde_se
        elif avar*(1-APE):
            return beta_tilde, W_hat_inv
        elif (1-avar)*APE:
            return beta_tilde, delta_tilde
        else:
            return beta_tilde
