# -*- coding: utf-8 -*-

import numpy as np
from scipy.stats import norm


flatten = lambda l : [item for sublist in l for item in sublist]

def multi_outer(A,B):
    
    """
    multi_outer:
        This function computes rowwise outer products between matrices A and B.
        
    INPUTS:
    ------
    `A'       : N x K (Numpy) array;
    `B '     : N x K (Numpy) array.
    
    OUTPUT:
    ------
    `C''       : N x K x K tensor C with C[i,:,:] = A[i,:]*B[i,:].T.
    """
    
    return A[:,:,None]*B[:,None,:]

def logit_cdf(x):
    
    """
    logit_cdf:
        logistic cumulative distribution function (cdf).
        
    INPUT:
    -----
    `x'       : evaluation point (or array).
    
    OUTPUT:
    ------
    `y'       : evaluation of logistic cdf at x.
    """
    
    return 1 / (1 + np.exp(-x))

def logit_pdf(x):
    
    """
    logit_pdf:
        logistic probability density function (pdf).
    
    INPUT:
    -----
    `x'       : evaluation point (or array).
    
    OUTPUT:
    ------
    `y'       : evaluation of logistic pdf at x.
    """
    
    return logit_cdf(x) * (1-logit_cdf(x))

def probit_cdf(x):
    
    """
    probit_cdf:
        probit cumulative distribution function (cdf).
        
    INPUT:
    -----
    `x'       : evaluation point (or array).
    
    OUTPUT:
    ------
    `y'       : evaluation of probit cdf at x.
    """
    
    return norm.cdf(x)

def probit_pdf(x):
    
    """
    probit_pdf:
        probit probability density function (cdf).
        
    INPUT:
    -----
    `x'       : evaluation point (or array).
    
    OUTPUT:
    ------
    `y'       : evaluation of probit pdf at x.
    """
    
    return norm.pdf(x)

# log-likelihood functions

def bin_log_loglik(b, mat):
    
    """
    bin_logloglik:
        compute log likelihood in the binary logistic model.
    
    INPUTS:
    ------
    `b'         : N x T (Numpy) array of binary outcomes;
    `mat'       : N x T (Numpy) array of single indexes.
    
    OUTPUT:
    ------
    `y'         : log-likelihood (real number). 
    """
    
    M = logit_cdf(mat)
    return np.nansum(np.log(b*M + (1-b)*(1-M)))

def bin_prob_loglik(b, mat):
    
    """
    bin_prob_loglik:
        compute log likelihood in the binary probit model.
    
    INPUTS:
    ------
    `b'         : N x T (Numpy) array of binary outcomes;
    `mat'       : N x T (Numpy) array of single indexes.
    
    OUTPUT:
    ------
    `y'         : log-likelihood (real number).
    """
    
    M = probit_cdf(mat)
    return np.nansum(np.log(b*M + (1-b)*(1-M)))

def count_poisson_loglik(b, mat):
    
    """
    count_poisson_loglik:
        compute log likelihood in the count Poisson model. 
    
    INPUTS:
    ------
     `b'         : N x T (Numpy) array of binary outcomes;
    `mat'       : N x T (Numpy) array of single indexes.
    
    OUTPUT:
    ------
    `y'         : log-likelihood (real number).
    """
    
    return np.nansum(b*mat - np.exp(mat))
    
# log-likelihood gradients

    # fixed-effects
def grad_log_loglik_fe(b_i, mat_i, axis=0):
    
    """
    grad_log_loglik_fe:
        compute the log likelihood FE's gradient in the binary logistic model.
    
    INPUTS:
    ------
    `b_ i'       : N x T (Numpy) array of binary outcomes;
    `mat_i'     : N x T (Numpy) array of single indexes;
    `axis'       : 0/1 to compute gradient of time/individual FE.
    
    OUTPUT:
    ------
    `y'         : FE's gradient of the loglikelihood (real number).
    """
    
    return np.nansum(b_i-logit_cdf(mat_i), axis)

def grad_prob_loglik_fe(b_i, mat_i, axis=0):
    
    """
    grad_prob_loglik_fe:
        compute the log likelihood FE's gradient in the binary probit model.
    
    INPUTS:
    ------
    `b_ i'       : N x T (Numpy) array of binary outcomes;
    `mat_i'     : N x T (Numpy) array of single indexes;
    `axis'       : 0/1 to compute gradient of time/individual FE.
    
    OUTPUT:
    ------
    `y'         : FE's gradient of the loglikelihood (real number).
    """
    
    M = probit_cdf(mat_i) 
    m = probit_pdf(mat_i)
    return np.nansum((m/(M*(1-M)))*(b_i-M), axis)

def grad_poiss_loglik_fe(b_i, mat_i, axis=0):
    
    """
    grad_poiss_loglik_fe:
        compute the log likelihood FE's gradient in the count Poisson model.
    
    INPUTS:
    ------
    `b_ i'       : N x T (Numpy) array of binary outcomes;
    `mat_i'     : N x T (Numpy) array of single indexes;
    `axis'       : 0/1 to compute gradient of time/individual FE.
    
    OUTPUT:
    ------
    `y'         : FE's gradient of the loglikelihood (real number).
    """
    
    return np.nansum(b_i - np.exp(mat_i), axis)
  
    # slope coefficient(s) FPMLE++
def grad_log_loglik_beta(b, A, mat, axis=(0,1)):
    
    """
    grad_log_loglik_beta:
        compute the log likelihood common parameter's gradient
        in the binary logistic model.
    
    INPUTS:
    ------
    `b'         : N x T (Numpy) array of binary outcomes;
    `A'         : N x T x K (Numpy) array of covariates;
    `mat'       : N x T (Numpy) array of single indexes;
    `axis'     : (0,1)/1 to compute gradient for homogeneous/heterogeneous
                  slopes.
    
    OUTPUT:
    ------
    `y'         : K x 1 or N x K vector (gradient w.r.t the slope parameter). 
    """

    return  np.nansum(A*(b - logit_cdf(mat))[:,:,None], axis)

def grad_prob_loglik_beta(b, A, mat, axis=(0,1)):
    
    """
    grad_prob_loglik_beta:
        compute the log likelihood common parameter's gradient
        in the binary probit model.
    
    INPUTS:
    ------
    `b'         : N x T (Numpy) array of binary outcomes;
    `A'         : N x T x K (Numpy) array of covariates;
    `mat'       : N x T (Numpy) array of single indexes;
    `axis'     : (0,1)/1 to compute gradient for homogeneous/heterogeneous
                  slopes.
    
    OUTPUT:
    ------
    `y'         : K x 1 or N x K vector (gradient w.r.t the slope parameter).
    """
    
    M = probit_cdf(mat)
    m = probit_pdf(mat)
    grad = np.nansum(A * ((m/(M*(1-M))) * (b-M))[:,:,None], axis)
    return grad

def grad_poiss_loglik_beta(b, A, mat, axis=(0,1)):
    
    """
    grad_poiss_loglik_beta:
        compute the log likelihood common parameter's gradient
        in the count Poisson model.
    
    INPUTS:
    ------
    `b'         : N x T (Numpy) array of binary outcomes;
    `A'         : N x T x K (Numpy) array of covariates;
    `mat'       : N x T (Numpy) array of single indexes;
    `axis'     : (0,1)/1 to compute gradient for homogeneous/heterogeneous
                  slopes.
    
    OUTPUT:
    ------
    `y'         : K x 1 or N x K vector (gradient w.r.t the slope parameter).
    """
    
    return np.nansum(A * (b - np.exp(mat))[:,:,None], axis)



    # heterogeneous slope coefficient FPMLE
def grad_log_loglik_beta_indiv(b, A, mat):
    
    """
    grad_log_loglik_beta_indiv:
        compute the log likelihood common parameter's gradient
        in the binary logistic model (FPMLE only).
    
    INPUTS:
    ------
    `b'         : T x 1 (Numpy) array of binary outcomes;
    `A'         : T x K (Numpy) array of covariates;
    `mat'       : T x 1 (Numpy) array of single indexes.
    
    OUTPUT:
    ------
    `y'         : K x 1 vector (gradient w.r.t the slope parameter). 
    """

    return  np.nansum(A*(b - logit_cdf(mat))[:,None], axis=0)

def grad_prob_loglik_beta_indiv(b, A, mat):
    
    """
    grad_prob_loglik_beta_indiv:
        compute the log likelihood common parameter's gradient
        in the binary probit model.
    
    INPUTS:
    ------
    `b'         : T x 1 (Numpy) array of binary outcomes;
    `A'         : T x K (Numpy) array of covariates;
    `mat'       : T x 1 (Numpy) array of single indexes.
    
    OUTPUT:
    ------
    `y'         : K x 1 vector (gradient w.r.t the slope parameter). 
    """
    
    M = probit_cdf(mat)
    m = probit_pdf(mat)
    grad = np.nansum(A * ((m/(M*(1-M))) * (b-M))[:,None], axis=0)
    return grad

def grad_poiss_loglik_beta_indiv(b, A, mat):
    
    """
    grad_poiss_loglik_beta_indiv:
        compute the log likelihood common parameter's gradient
        in the count Poisson model.
    
    INPUTS:
    ------
    `b'         : T x 1 (Numpy) array of binary outcomes;
    `A'         : T x K (Numpy) array of covariates;
    `mat'       : T x 1 (Numpy) array of single indexes.
    
    OUTPUT:
    ------
    `y'         : K x 1 vector (gradient w.r.t the slope parameter).
    """
    
    return np.nansum(A * (b - np.exp(mat))[:,None], axis=0)

# log-likelihood hessians

    # fixed-effects
def hess_log_loglik_fe(mat_i, axis=0):
    
    """
    hess_log_loglik_fe:
        compute the log likelihood FE's hessian in the binary logistic model.
    
    INPUTS:
    ------
    `b_i'        : N x T (Numpy) array of binary outcomes;
    `mat _i'     : N x T (Numpy) array of single indexes;
    `axis'       : 0/1 to compute hessian of time/individual FE.
    
    OUTPUT:
    ------
    y          : FE's hessian of the loglikelihood (real number).   
    """
    
    M = logit_cdf(mat_i)
    return - np.nansum(M*(1-M), axis)

def hess_prob_loglik_fe(mat_i, axis=0):
    
    """
    hess_prob_loglik_fe:
        compute the log likelihood FE's hessian in the binary probit model.
    
    INPUTS:
    ------
    `b_i'        : N x T (Numpy) array of binary outcomes;
    `mat _i'     : N x T (Numpy) array of single indexes;
    `axis'       : 0/1 to compute hessian of time/individual FE.
    
    OUTPUT:
    ------
    y          : FE's hessian of the loglikelihood (real number).   
    """
    
    M = probit_cdf(mat_i)
    m = probit_pdf(mat_i)
    return - np.nansum(m**2/(M*(1-M)), axis)

def hess_poiss_loglik_fe(mat_i, axis=0):
    
    """
    hess_poiss_loglik_fe:
        compute the log likelihood FE's hessian in the count Poisson model.
    
    INPUTS:
    ------
    `b_i'        : N x T (Numpy) array of binary outcomes;
    `mat _i'     : N x T (Numpy) array of single indexes;
    `axis'       : 0/1 to compute hessian of time/individual FE.
    
    OUTPUT:
    ------
    y          : FE's hessian of the loglikelihood (real number).   
    """
    
    return - np.nansum(np.exp(mat_i), axis)

#  slope coefficient(s) FPMLE++
def hess_log_loglik_beta(A, mat, axis=(0,1)):
    
    """
    hess_log_loglik_beta:
        compute the log likelihood slope parameter's hessian
        in the binary logistic model.
    
    INPUTS:
    ------
    `A'         : N x T x K (Numpy) array of covariates;
    `mat'       : N x T (Numpy) array of single indexes;
    `axis'      : (0,1)/1 to compute hessian for homogeneous/heterogeneous
                  slopes.
    
    OUTPUT:
    ------
    `y'         : K x 1 or N x K vector (hessian w.r.t the slope parameter).
    """
    
    M = logit_cdf(mat)
    hess = A[:,:,:,None]*A[:,:,None,:] * (M*(1-M))[:,:, None, None]
    return - np.nansum(hess, axis)


def hess_prob_loglik_beta(A, mat, axis=(0,1)):
    
    """
    hess_prob_loglik_beta:
        compute the log likelihood slope parameter's hessian
        in the binary probit model.
    
    INPUTS:
    ------
    `A'         : N x T x K (Numpy) array of covariates;
    `mat'       : N x T (Numpy) array of single indexes;
    `axis'      : (0,1)/1 to compute hessian for homogeneous/heterogeneous
                  slopes.
    
    OUTPUT:
    ------
    `y'         : K x 1 or N x K vector (hessian w.r.t the slope parameter).
    """
    
    M = probit_cdf(mat)
    m = probit_pdf(mat)
    hess = A[:,:,:,None]*A[:,:,None,:] * (m**2/ (M*(1-M)))[:,:, None, None]
    return  - np.nansum(hess, axis)


def hess_poiss_loglik_beta(A, mat, axis=(0,1)):
    
    """
    hess_poiss_loglik_beta:
        compute the log likelihood slope parameter's hessian
        in the count Poisson model.
    
    INPUTS:
    ------
    `A'         : N x T x K (Numpy) array of covariates;
    `mat'       : N x T (Numpy) array of single indexes;
    `axis'      : (0,1)/1 to compute hessian for homogeneous/heterogeneous
                  slopes.
    
    OUTPUT:
    ------
    `y'         : K x 1 or N x K vector (hessian w.r.t the slope parameter).
    """
    
    hess = A[:,:,:,None]*A[:,:,None,:] * np.exp(mat)[:,:, None, None]
    return - np.nansum(hess, axis)

    # heterogeneous slope coefficients FPMLE
def hess_log_loglik_beta_indiv(A, mat):
    
    """
    hess_log_loglik_beta_indiv:
        compute the log likelihood common parameter's hessian
        in the binary Logistic model.
    
    INPUTS:
    ------
    b         : N x T (Numpy) array of integer outcomes
    A         : N x T x K (Numpy) array of covariates
    mat       : N x T (Numpy) array of single indexes
    
    OUTPUT:
    ------
    y         : K x 1 vector (gradient w.r.t the common parameter)     
    """
    
    M = logit_cdf(mat)
    hess = A[:,:,None]*A[:,None,:] * (M*(1-M))[:, None, None]
    return - np.nansum(hess,axis=0)

def hess_prob_loglik_beta_indiv(A, mat):
    
    """
    hess_prob_loglik_beta_indiv:
        compute the log likelihood common parameter's hessian
        in the binary Probit model.
    
    INPUTS:
    ------
    b         : N x T (Numpy) array of integer outcomes
    A         : N x T x K (Numpy) array of covariates
    mat       : N x T (Numpy) array of single indexes
    
    OUTPUT:
    ------
    y         : K x 1 vector (gradient w.r.t the common parameter)     
    """
    
    M = probit_cdf(mat)
    m = probit_pdf(mat)
    hess = A[:,:,None]*A[:,None,:] * (m**2/ (M*(1-M)))[:, None, None]
    return  - np.nansum(hess, axis=0)

def hess_poiss_loglik_beta_indiv(A, mat):
    
    """
    hess_poiss_loglik_beta_indiv:
        compute the log likelihood common parameter's hessian
        in the binary Probit model.
    
    INPUTS:
    ------
    b         : N x T (Numpy) array of integer outcomes
    A         : N x T x K (Numpy) array of covariates
    mat       : N x T (Numpy) array of single indexes
    
    OUTPUT:
    ------
    y         : K x 1 vector (gradient w.r.t the common parameter)     
    """
    
    hess = A[:,:,None]*A[:,None,:] * np.exp(mat)[:, None, None]
    return - np.nansum(hess, axis=0)



# Poisson model
def newton_raphson(model, tol=1e-3, max_iter=1000, display=True):
        i = 0
        error = 100  # Initial error value
    
        # Print header of output
        if display:
            header = f'{"Iteration_k":<13}{"Log-likelihood":<16}{"θ":<60}'
            print(header)
            print("-" * len(header))
    
        # While loop runs while any value in error is greater
        # than the tolerance until max iterations are reached
        while np.any(error > tol) and i < max_iter:
            H, G = model.H(), model.G()
            β_new = model.β - (np.linalg.inv(H) @ G)
            error = β_new - model.β
            model.β = β_new
    
            # Print iterations
            if display:
                β_list = [f'{t:.3}' for t in list(model.β.flatten())]
                update = f'{i:<13}{model.logL():<16.8}{β_list}'
                print(update)
    
            i += 1
    
        #print(f'Number of iterations: {i}')
        #print(f'β_hat = {model.β.flatten()}')
    
        # Return a flat array for β (instead of a k_by_1 column vector)
        return model.β.flatten()