# -*- coding: utf-8 -*-
# ensure "normal" division
from __future__ import division

# load library dependencies
import numpy as np
from scipy.optimize import minimize
from concurrent.futures import ThreadPoolExecutor, as_completed
import func

class TwoWayFPMLE():
    
    """
    TwoWayFPMLE:
        This class implements fast Gauss-Seidel fixed-point algorithms (FPMLE 
        and FPMLE++) for estimating semiparametric nonlinear panel data models 
        with two-way fixed effects as described in Mugnier & Wang (2022, 
        Working Paper) -- ``Identification and (Fast) Estimation of Large 
        Nonlinear Panel Models with Two-Way Fixed Effects''. 
        The implementation is as described in the paper. Notation attempts to 
        follow that used in the paper. 
        In particular: 
        `N'  : number of individuals;
        `T'  : number of time periods (must be balanced);
        `K'  : number of observed covariates.
        Missing cells in unbalanced panel data can be imputed  with NaN 
        by using the PanelPreProcess() class. Estimation simply ignores them.
        
    AUTHOR:    Martin Mugnier, martin.mugnier@economics.ox.ac.uk, October 2023
    ------
    
    INPUTS:
    ------
    `A'            : N x T x K (Numpy) array of exogeneous variables;
    `b'            : N x T (Numpy) array of a scalar endogeneous variable;
    `indiv_ref'    : index of individual fixed effect set to zero;
    `het_exog'     : indices of heterogeneous slopes;
    `model'        : character string ('logit', 'probit', 'poisson');
    `max_workers'  : maximum number of processors used for parallel computing;
    `options_set'  : options for Newton-Raphson algorithms (FPMLE only).
    
    FUNCTIONS:
    ---------
    `objective'
    `compute_indiv_effect'
    `compute_time_effect'
    `inner_loop_indiv'
    `fast_inner_loop_indiv'
    `inner_loop_time'
    `fast_inner_loop_time'
    `outer_loop_slope_coeff'
    `fast_outer_loop_slope_coeff'
    `fit'
    
    """
    
    def __init__(self, A, b, indiv_ref=0, het_exog=None, model='logit', 
                 max_workers=4, solver='Newton-CG', options_set = 
                 {'xtol': 1e-6, 'maxiter': 1000, 'disp': False}, verbose=True):
        
        self.A = A
        self.b = b
        self.indiv_ref = indiv_ref
        self.het_exog = het_exog
        self.max_workers = max_workers
        self.solver = solver
        self.options_set = options_set
        self.model = model
        
        # check input format
        if type(A).__module__ != 'numpy':
            print('The design matrix A must be a N x T x K (Numpy) array.')  
            
        if type(b).__module__ != 'numpy':
            print('The outcome matrix b must be a N x T (Numpy) array.')
            
        if len(A.shape) != 3:
            print('The design matrix A must have shape (N, T, K).')  
        
        else:
            self.N, self.T, self.K = A.shape
            
        if het_exog is None:
            self.het = False
            if verbose:
                print('Fully homogeneous slope model.')
            
        else:
            self.het = True 
            self.K_het = len(het_exog)
            self.hom_exog = np.setdiff1d(np.arange(self.K), self.het_exog)
            
            if self.K_het == self.K*verbose:
                print('Fully heterogeneous slopes model.')
            
            elif verbose:
                print('Mixed heterogeneous and homogeneous slopes model.')
                
        if len(b.shape) != 2:
            print('The outcome matrix b must have shape (N, T).')
            
        if model == 'logit':
            if (np.abs(2*b-1)==np.ones(b.shape)).all() == False:
                print('Warning: the outcome matrix contains NaNs or other \
                      than 0/1 entries.')
            self.loglik = func.bin_log_loglik
            self.grad_loglik_fe = func.grad_log_loglik_fe
            self.grad_loglik_beta = func.grad_log_loglik_beta
            self.grad_log_loglik_beta_indiv = func.grad_log_loglik_beta_indiv
            self.hess_loglik_fe = func.hess_log_loglik_fe
            self.hess_loglik_beta = func.hess_log_loglik_beta
            self.hess_loglik_beta_indiv = func.hess_log_loglik_beta_indiv 
            keep_i = (0<np.sum(self.b, axis=1))*(np.sum(self.b, axis=1)<self.T)
            keep_t = (0<np.sum(self.b, axis=0))*(np.sum(self.b, axis=0)<self.N)
            if np.sum(keep_i)<self.N:
                print('Dropped ', np.sum(1-keep_i), 
                      ' individuals with no outcome variations')
            if np.sum(keep_t)<self.T:
                print('Dropped ', np.sum(1-keep_t), 
                      ' time periods with no outcome variations')
            self.A = self.A[keep_i][:,keep_t,:]
            self.b = self.b[keep_i][:,keep_t]
            self.N, self.T, self.K = self.A.shape # make it more userfriendly (e.g., to track dropped ID)
            
        elif model == 'probit':
            if (np.abs(2*b-1)==np.ones(b.shape)).all() == False:
                print('Warning: the outcome matrix contains NaNs or other \
                      than 0/1 entries.')
            self.loglik = func.bin_prob_loglik
            self.grad_loglik_fe = func.grad_prob_loglik_fe
            self.grad_loglik_beta = func.grad_prob_loglik_beta
            self.grad_log_loglik_beta_indiv = func.grad_prob_loglik_beta_indiv
            self.hess_loglik_fe = func.hess_prob_loglik_fe
            self.hess_loglik_beta = func.hess_prob_loglik_beta
            self.hess_loglik_beta_indiv = func.hess_prob_loglik_beta_indiv
            keep_i = (0<np.sum(self.b, axis=1))*(np.sum(self.b, axis=1)<self.T)
            keep_t = (0<np.sum(self.b, axis=0))*(np.sum(self.b, axis=0)<self.N)
            if np.sum(keep_i)<self.N:
                print('Dropped ', np.sum(1-keep_i), 
                      ' individuals with no outcome variations')
            if np.sum(keep_t)<self.T:
                print('Dropped ', np.sum(1-keep_t), 
                      ' time periods with no outcome variations')
            self.A = self.A[keep_i][:,keep_t,:]
            self.b = self.b[keep_i][:,keep_t]
            self.N, self.T, self.K = self.A.shape # make it more userfriendly (e.g., to track dropped ID)
        
        elif model == 'poisson':
            self.loglik = func.count_poisson_loglik
            self.grad_loglik_fe = func.grad_poiss_loglik_fe
            self.grad_loglik_beta = func.grad_poiss_loglik_beta
            self.grad_log_loglik_beta_indiv = func.grad_poiss_loglik_beta_indiv
            self.hess_loglik_fe = func.hess_poiss_loglik_fe
            self.hess_loglik_beta = func.hess_poiss_loglik_beta
            self.hess_loglik_beta_indiv = func.hess_poiss_loglik_beta_indiv
            keep_i = 0<np.sum(self.b, axis=1)
            keep_t = 0<np.sum(self.b, axis=0)
            if np.sum(keep_i)<self.N:
                print('Dropped ', np.sum(1-keep_i), 
                      ' individuals with zero outcome at all time periods.')
            if np.sum(keep_t)<self.T:
                print('Dropped ', np.sum(1-keep_t), 
                      ' time periods with zero outcome for all individuals')
            self.A = self.A[keep_i][:,keep_t,:]
            self.b = self.b[keep_i][:,keep_t]
            self.N, self.T, self.K = self.A.shape # make it more userfriendly (e.g., to track dropped ID)
            
            
        else:
            print("Model must be either 'logit', 'probit' or 'poisson'.")

    def objective(self, alpha, xi, beta): 
        
        """
        objective:
            compute the log-likelihood function.
            
        INPUTS:
        ------
        `alpha'    : (N-1) x 1 (Numpy) array of individual effects;
        `xi'       : T x 1 (Numpy) array of time effects;
        `beta'     : K x 1 or N x K (Numpy) array of slope coefficients.
        
        OUTPUT:
        ------
        loglik
        
        """
        if self.het:
            idx = np.einsum('ijk,ik->ij', self.A, beta) + np.insert(alpha,
                            self.indiv_ref, 0)[:, None] + xi[None, :]
        else:
            idx = self.A.dot(beta)  + np.insert(alpha, self.indiv_ref, 
                                0)[:,None] + xi[None,:]
        return - self.loglik(self.b, idx) 
    
    def compute_indiv_effect(self, batch_indiv, alpha, xi, beta):
        
        """
        compute_indiv_effect:
            perform STEP1 of FPMLE for a given batch of individuals.
            
        INPUTS:
        ------
        `batch_indiv' : batch of individuals;
        `alpha'       : (N-1) x 1 (Numpy) array, current guess for alpha;
        `xi'          : T x 1 (Numpy) array, current guess for xi;
        `beta'        : K x 1 (Numpy) array OR N x K array, current guess for 
                        beta.
        
        OUTPUTS:
        -------
        `res_alpha'    : updated individual effects;
        `res_theta'    : updated slopes.
        
        """
        A = self.A
        b = self.b
        res_alpha = np.zeros(len(batch_indiv))
        
        if not self.het:
            # set-up objective functions (alpha_i)
            def objective_i(alpha_i, i):
                idx_i = A[int(i)].dot(beta) + alpha_i + xi
                return - self.loglik(b[int(i)], idx_i) 
            
            def grad_obj_i(alpha_i, i):
                idx_i = A[int(i)].dot(beta) + alpha_i + xi
                return np.array([-self.grad_loglik_fe(b[int(i),:], idx_i) ])
    
            def hess_obj_i(alpha_i, i):
                idx_i = A[int(i)].dot(beta) + alpha_i + xi
                return np.array([-self.hess_loglik_fe(idx_i)]) 
            
            # run optimization w.r.t. alpha_i
            for idx, i in enumerate(batch_indiv):
                res_optim = minimize(objective_i, alpha[int(i)], args=(i), 
                    method=self.solver, jac=grad_obj_i, hess=hess_obj_i, 
                    options=self.options_set)
                res_alpha[idx] = res_optim.x 
            res_theta = beta.copy()
            
        else: 
            
            # set-up optimization functions (alpha_i and beta_i)
            def objective_i_het(alpha_i, i):
                idx_i = A[int(i)].dot(beta[int(i)]) + alpha_i + xi
                return - self.loglik(b[int(i)], idx_i) 
            
            def objective_beta_i_het(beta_i, i):
                theta_i = np.zeros(self.K)
                theta_i[self.het_exog] = beta_i
                theta_i[self.hom_exog] = beta[int(i), self.hom_exog]
                idx_i = A[int(i)].dot(theta_i) + alpha[int(i)] + xi
                return - self.loglik(b[int(i)], idx_i) 

            def grad_obj_i(alpha_i, i):
                idx_i = A[int(i)].dot(beta[int(i)]) + alpha_i + xi
                return np.array([-self.grad_loglik_fe(b[int(i)], idx_i)])
    
            def grad_beta_obj_i(beta_i, i):
                theta_i = np.zeros(self.K)
                theta_i[self.het_exog] = beta_i
                theta_i[self.hom_exog] = beta[int(i), self.hom_exog]
                idx_i = A[int(i)].dot(theta_i) + alpha[int(i)] + xi
                return -self.grad_log_loglik_beta_indiv(b[int(i)], 
                        A[int(i),:,self.het_exog].T,  idx_i)

            def hess_obj_i(alpha_i, i):
                idx_i = A[int(i)].dot(beta[int(i)]) + alpha_i + xi
                return np.array([-self.hess_loglik_fe(idx_i)])
            
            def hess_beta_obj_i(beta_i, i):
                theta_i = np.zeros(self.K)
                theta_i[self.het_exog] = beta_i
                theta_i[self.hom_exog] = beta[int(i), self.hom_exog]
                idx_i = A[int(i)].dot(theta_i) + alpha[int(i)] + xi
                return -self.hess_loglik_beta_indiv(A[int(i),
                        :, self.het_exog].T, idx_i)
            
            # run optimization w.r.t. alpha_i and beta_i
            res_beta = np.zeros((len(batch_indiv), self.K_het))
            for idx, i in enumerate(batch_indiv):
                # run optimization w.r.t. alpha_i
                res_optim = minimize(objective_i_het, alpha[int(i)], 
                    args=(i), method=self.solver, jac=grad_obj_i, 
                    hess=hess_obj_i, options=self.options_set)    
                res_alpha[idx] = res_optim.x
                # run optimization w.r.t. beta_i
                res_optim = minimize(objective_beta_i_het, beta[int(i),
                    self.het_exog], args=(i), 
                    method=self.solver, jac=grad_beta_obj_i, 
                    hess=hess_beta_obj_i, options=self.options_set)
                res_beta[idx,:] = res_optim.x
            res_theta = np.zeros((self.N, self.K))
            res_theta[batch_indiv[0].astype(int):batch_indiv[0].astype(int)
                      +len(batch_indiv), self.het_exog] = res_beta
            res_theta[:,self.hom_exog] = beta[:,self.hom_exog]
        return res_alpha, res_theta
    
    def compute_time_effect(self, batch_time, alpha, xi, beta):
        
        """
        compute_time_effect:
            perform STEP2 of FPMLE for a given batch of time periods
            
        INPUTS:
        ------
        `batch_time' : batch of updated time periods;
        `alpha'         : (N-1) x 1 (Numpy array), current guess for alpha;
        `xi'            : T x 1 (Numpy array), current guess for xi;
        `beta'          : K x 1 (Numpy) array or N x K array, current guess for 
                          beta.
        
        OUTPUTS:
        -------
        `res'   : updated time effects.
        
        """
        A = self.A
        b = self.b
        res = np.zeros(len(batch_time))
  
        # set-up optimization w.r.t. xi_t
        def objective_t(xi_t, t):
            if self.het:
                idx_t = np.einsum('ik,ik->i', A[:,int(t)], beta) \
                    + np.insert(alpha, self.indiv_ref, 0) + xi_t
            else:
                idx_t = A[:,int(t)].dot(beta) + np.insert(alpha, 
                    self.indiv_ref, 0) + xi_t
            return - self.loglik(b[:,int(t)], idx_t) 
        
        def grad_obj_t(xi_t, t):
            if self.het:
                idx_t = np.einsum('ik,ik->i', A[:,int(t)], beta) \
                    + np.insert(alpha, self.indiv_ref, 0) + xi_t
            else:
                idx_t = A[:,int(t)].dot(beta) + np.insert(alpha, 
                    self.indiv_ref, 0) + xi_t
            return np.array([-np.nansum(self.grad_loglik_fe(b[:, int(t)], 
                                                            idx_t))])
        
        def hess_obj_t(xi_t, t):
            if self.het:
                idx_t = np.einsum('ik,ik->i', A[:,int(t)], beta) \
                    + np.insert(alpha, self.indiv_ref, 0) + xi_t
            else:
                idx_t = A[:,int(t)].dot(beta) + np.insert(alpha, 
                    self.indiv_ref, 0) + xi_t
            return np.array([-self.hess_loglik_fe(idx_t)]) 
        
        # run optimization w.r.t. xi_t
        for idx, t in enumerate(batch_time):
            res_optim = minimize(objective_t, xi[int(t)], args=(t),
                        method='Newton-CG', jac=grad_obj_t, hess=hess_obj_t,
                        options=self.options_set)
            res[idx] = res_optim.x
        return res
    
    def inner_loop_indiv(self, alpha, xi, beta):
        
        """
        inner_loop_indiv:
            run STEP 1 of FPMLE.
            
        INPUTS:
        ------
        `alpha'       : (N-1) x 1 (Numpy array), current guess for alpha;
        `xi'          : T x 1 (Numpy array), current guess for xi;
        `beta'        : K x 1 (Numpy) array or N x K array, current guess for 
                        beta.
        
        OUTPUTS:
        -------
        `res'   : updated individual effects (and heterogeneous slopes).
        
        """
        # parallelize tasks
        indiv_per_worker = np.floor((self.N)/self.max_workers)
        alpha_updated_list = []
        beta_updated_list = []
        batch_id = []
        alpha_hat = np.zeros(self.N)
        
        if self.het:
            beta_hat = np.zeros((self.N, self.K))
            
        with ThreadPoolExecutor(self.max_workers) as executor:
           
            if self.max_workers >= (self.N):
                future_to_compute_indiv = {executor.submit(
                    self.compute_indiv_effect, [j], alpha, xi, beta): 
                        j for j in range(self.N)}
            
            elif (self.N)/self.max_workers - indiv_per_worker == 0:
                batches = np.split(np.arange(self.N), self.max_workers)
                future_to_compute_indiv = {executor.submit(
                    self.compute_indiv_effect, batches[j], alpha, xi, beta): j 
                    for j in range(len(batches))}
            
            else:
                batches = np.split(np.arange(self.max_workers*indiv_per_worker), 
                                   indiv_per_worker)
                batches.append(np.arange(self.max_workers*indiv_per_worker, 
                                         self.N))
                future_to_compute_indiv = {executor.submit(
                    self.compute_indiv_effect, batches[j], alpha, xi, beta): 
                            j for j in range(len(batches))}
        
        for future in as_completed(future_to_compute_indiv):
            batch_id.append(future_to_compute_indiv[future])
            alpha_updated_list.append(future.result()[0])
            if self.het:
                beta_updated_list.append(future.result()[1])
       
        for j_loc, j in enumerate(batch_id):
            alpha_hat[batches[j].astype(int)] = alpha_updated_list[j_loc]
        
        if self.het:
            beta_hat = np.nansum(np.array(beta_updated_list), axis=0)
            beta_hat[:,self.hom_exog] = beta_hat[:,self.hom_exog] \
                        / self.max_workers 
        
        else:
            beta_hat = beta.copy()
        return np.delete(alpha_hat, self.indiv_ref), beta_hat
    
    def fast_inner_loop_indiv(self, alpha, xi, beta, nu=1e-3, hess=False):
        """
        fast_inner_loop_indiv:
            run STEP 1 of FPMLE++.
            
        INPUTS:
        ------
        `alpha'       : (N-1) x 1 (Numpy array), current guess for alpha;
        `xi'          : T x 1 (Numpy array), current guess for xi;
        `beta'        : K x 1 (Numpy) array or N x K array, current guess for 
                        beta.
        `nu'          : gradient step (positive small number);
        `hess'        : 0/1, use of hessian steps.
        
        OUTPUTS:
        -------
        `res'   : updated individual effects (and heterogeneous slopes).
        
        """
        A = self.A
        b = self.b
        if self.het:
            idx = np.einsum('ijk,ik->ij', A, beta) + np.insert(alpha, 
                            self.indiv_ref, 0)[:,None] + xi[None,:]
        else:
            idx = A.dot(beta) + np.insert(alpha, self.indiv_ref, 0)[:,None] \
            + xi[None,:]
        # compute the alpha_i's
        if hess:
            # apply a single Newton-Raphson step
            try:
                alpha_hat = alpha - (self.grad_loglik_fe(np.delete(b, 
                        self.indiv_ref, 0), np.delete(idx, self.indiv_ref, 0),
                        axis=1) /self.hess_loglik_fe(np.delete(idx, 
                        self.indiv_ref, 0), axis=1))
            except:
                    print("Numerical instability alpha_i's hessian.")
                    alpha_hat = alpha + nu*self.grad_loglik_fe(np.delete(b, 
                        self.indiv_ref, 0), np.delete(idx, 
                        self.indiv_ref, 0), axis=1)                                                           
        else:
            # apply a single Newton-Raphson step
            alpha_hat = alpha + nu*self.grad_loglik_fe(np.delete(b, 
                        self.indiv_ref, 0), np.delete(idx, 
                        self.indiv_ref, 0), axis=1)
        # compute the beta_i's 
        if self.het: 
            idx = np.einsum('ijk,ik->ij', A, beta) + np.insert(alpha_hat, 
                            self.indiv_ref, 0)[:,None] + xi[None,:]
            if hess:
                # apply a single Newton-Raphson step
                try:
                    theta_hat = beta[:,self.het_exog] - np.einsum('ijk,ik->ij', 
                    np.linalg.inv(self.hess_loglik_beta(A[:,:,self.het_exog],
                    idx, axis=1)), self.grad_loglik_beta(b, 
                                        A[:,:,self.het_exog], idx, axis=1)) 
                except:
                    print("Numerical instability of beta_i's hessian.")
                    theta_hat = beta[:,self.het_exog] \
                        + nu*self.grad_loglik_beta(b, A[:,:,self.het_exog],
                                            idx, axis=1)
            else:
                theta_hat = beta[:,self.het_exog] \
                        + nu*self.grad_loglik_beta(b, A[:,:,self.het_exog],
                                            idx, axis=1)
            beta_hat = np.zeros((self.N, self.K))
            beta_hat[:, self.het_exog] = theta_hat
            beta_hat[:,self.hom_exog] = beta[:,self.hom_exog] 
        else:
            beta_hat = beta.copy()
        return alpha_hat, beta_hat
    
    def inner_loop_time(self, alpha, xi, beta):
       
        """
        inner_loop_time:
            run STEP 2 of FPMLE.
            
        INPUTS:
        ------
        `alpha'       : (N-1) x 1 (Numpy array), current guess for alpha;
        `xi'          : T x 1 (Numpy array), current guess for xi;
        `beta'        : K x 1 (Numpy) array or N x K array, current guess for 
                        beta.
        
        OUTPUTS:
        -------
        `res'   : updated time effects.
        
        """
        # parallelize tasks
        time_periods_per_worker = np.floor(self.T/self.max_workers)
        xi_updated_list = []
        batch_id = []
        xi_hat = np.zeros(self.T)
        with ThreadPoolExecutor(self.max_workers) as executor:
            if self.max_workers >= (self.T):
                future_to_compute_time = {executor.submit(
                    self.compute_time_effect, [j], alpha, xi, beta): 
                        j for j in range(self.T)}
            elif (self.T-1)/self.max_workers - time_periods_per_worker == 0:
                batches = np.split(np.arange(1, self.T), self.max_workers)
                future_to_compute_time = {executor.submit(
                    self.compute_time_effect, batches[j],alpha, xi, beta
                    ): j for j in range(len(batches))}
            else:
                batches = np.split(np.arange(self.max_workers
                            *time_periods_per_worker), time_periods_per_worker)
                batches.append(np.arange(self.max_workers
                            * time_periods_per_worker, self.T))
                future_to_compute_time = {executor.submit(
                    self.compute_time_effect, batches[j], alpha, xi, beta
                    ): j for j in range(len(batches))}
        for future in as_completed(future_to_compute_time):
            batch_id.append(future_to_compute_time[future])
            xi_updated_list.append(future.result())
        for j_loc, j in enumerate(batch_id):
            xi_hat[batches[j].astype(int)] = xi_updated_list[j_loc]   
        return xi_hat
    
    def fast_inner_loop_time(self, alpha, xi, beta, nu=1e-3, hess=True):
        
        """
        fast_inner_loop_time:
            run STEP 2 of FPMLE++.
            
        INPUTS:
        ------
        `alpha'       : (N-1) x 1 (Numpy array), current guess for alpha;
        `xi'          : T x 1 (Numpy array), current guess for xi;
        `beta'        : K x 1 (Numpy) array or N x K array, current guess for 
                        beta;
        `nu'          : learning rate (positive small number);
        `hess'        : 0/1, use of hessian steps.
        
        OUTPUTS:
        -------
        `res'   : updated time effects.
        
        """
        if not self.het:
            idx = self.A.dot(beta) + np.insert(alpha, self.indiv_ref, 
                0)[:,None] + xi[None,:]
            
            # apply a single Newton-Raphson step
            if hess:
                try:
                    xi_hat = xi - (self.grad_loglik_fe(self.b, idx, axis=0)
                                / self.hess_loglik_fe(idx, axis=0))
                except:
                    print("Numerical instability of xi_t's hessian.")
                    xi_hat = xi + nu*self.grad_loglik_fe(self.b, idx, 
                                                             axis=0)
            else:
                xi_hat = xi + nu*self.grad_loglik_fe(self.b, idx, axis=0)
        
        else:
            idx = np.einsum('ijk,ik->ij', self.A, beta) + np.insert(alpha, 
                                self.indiv_ref, 0)[:,None] + xi[None,:]
            
            # apply a single Newton-Raphson step
            if hess:
                try:
                    xi_hat = xi - (self.grad_loglik_fe(self.b, idx, axis=0)
                                /self.hess_loglik_fe(idx, axis=0))
                except:
                    print("Numerical instability of xi_t's hessian.")
                    xi_hat = xi + nu*self.grad_loglik_fe(self.b, idx, axis=0)             
            else:
                xi_hat = xi + nu*self.grad_loglik_fe(self.b, idx, axis=0)     
        return xi_hat
    
    def outer_loop_slope_coeff(self, alpha, xi, beta):
        
        """
        outer_loop_slope_coeff:
            run STEP 3 of FPMLE.
            
        INPUTS:
        ------
        `alpha'       : (N-1) x 1 (Numpy array), current guess for alpha;
        `xi'          : T x 1 (Numpy array), current guess for xi;
        `beta'        : K x 1 (Numpy) array or N x K array, current guess for 
                        beta.
        
        OUTPUTS:
        -------
        `res'   : updated (homogeneous) slopes.
        
        """
        A = self.A
        b = self.b
        if not self.het:
            def objective_outer_loop(beta):
                idx = A.dot(beta) + np.insert(alpha, self.indiv_ref, 
                        0)[:,None] + xi[None,:]
                return - self.loglik(b, idx)
            
            def grad_outer_loop(beta):
                idx = A.dot(beta) + np.insert(alpha, self.indiv_ref, 
                        0)[:,None] + xi[None,:]
                return - self.grad_loglik_beta(b, A, idx) 
            
            def hess_outer_loop(beta):
                idx = A.dot(beta) + np.insert(alpha, self.indiv_ref, 
                        0)[:,None] +xi[None,:]
                return - self.hess_loglik_beta(A, idx) 
        
            beta_sol = minimize(objective_outer_loop, beta, method='Newton-CG', 
                            jac=grad_outer_loop, hess=hess_outer_loop, 
                            options=self.options_set)
            res = beta_sol.x
        else:
            def objective_outer_loop(beta_opt):
                theta = np.zeros((self.N, self.K))
                theta[:, self.hom_exog] = beta_opt
                theta[:, self.het_exog] = beta[:, self.het_exog]
                idx = np.einsum('ijk,ik->ij', self.A, theta) + np.insert(alpha, 
                                self.indiv_ref, 0)[:,None] + xi[None,:]
                return - self.loglik(b, idx) 
            
            def grad_outer_loop(beta_opt):
                theta = np.zeros((self.N, self.K))
                theta[:, self.hom_exog] = beta_opt
                theta[:, self.het_exog] = beta[:, self.het_exog]
                idx = np.einsum('ijk,ik->ij', self.A, theta) + np.insert(alpha, 
                                self.indiv_ref, 0)[:,None] + xi[None,:]
                return - self.grad_loglik_beta(b, A[:,:,self.hom_exog], idx) 
            
            def hess_outer_loop(beta_opt):
                theta = np.zeros((self.N, self.K))
                theta[:, self.hom_exog] = beta_opt
                theta[:, self.het_exog] = beta[:, self.het_exog]
                idx = np.einsum('ijk,ik->ij', self.A, theta) + np.insert(alpha, 
                                self.indiv_ref, 0)[:,None] + xi[None,:]
                return - self.hess_loglik_beta(A[:,:,self.hom_exog], idx) 
        
            beta_sol = minimize(objective_outer_loop, beta[0, self.hom_exog], 
                            method='Newton-CG', jac=grad_outer_loop, 
                            hess=hess_outer_loop, options=self.options_set)
            res = np.zeros((self.N ,self.K))
            res[:, self.hom_exog] = beta_sol.x
            res[:, self.het_exog] = beta[:, self.het_exog]     
        return res
    
    def fast_outer_loop_slope_coeff(self, alpha, xi, beta, nu=1e-3, hess=True):
        
        """
        fast_outer_loop_slope_coeff:
            run STEP 3 of FPMLE++.
            
        INPUTS:
        ------
        `alpha'       : (N-1) x 1 (Numpy array), current guess for alpha;
        `xi'          : T x 1 (Numpy array), current guess for xi;
        `beta'        : K x 1 (Numpy) array or N x K array, current guess for 
                        beta;
        `nu'          : learning rate (positive small number).

        OUTPUTS:
        -------
        `res'   : updated (homogeneous) slopes.
        
        """
        A = self.A
        b = self.b
        if not self.het: # homogeneous model (update full vector)
            idx = A.dot(beta) + np.insert(alpha, self.indiv_ref, 0)[:,None] \
                + xi[None,:]
            if hess:
                try:
                    beta_hat = beta - (np.linalg.inv(self.hess_loglik_beta(A, 
                            idx)).dot(self.grad_loglik_beta(b, A, idx)))
                except:
                    print("Numerical instability of beta's hessian.")
                    beta_hat = beta + nu*self.grad_loglik_beta(b, A, idx)                    
            else:
                beta_hat = beta + nu*self.grad_loglik_beta(b, A, idx)
        elif self.K_het < self.K: # mixed model (update only homogeneous part)
            idx = np.einsum('ijk,ik->ij', A, beta) + np.insert(alpha,
                            self.indiv_ref, 0)[:,None] + xi[None,:]
            if hess:
                try:
                    theta_hat = beta[0, self.hom_exog] \
                        - (np.linalg.inv(self.hess_loglik_beta(
                        A[:,:,self.hom_exog], idx)).dot(
                        self.grad_loglik_beta(b, A[:,:,self.hom_exog], idx)))
                except:
                    print("Numerical instability of beta's' hessian.")
                    theta_hat = beta[0, self.hom_exog] \
                            + nu*self.grad_loglik_beta(b, 
                            A[:,:,self.hom_exog], idx)
            else: 
                theta_hat = beta[0, self.hom_exog] \
                            + nu*self.grad_loglik_beta(b, 
                            A[:,:,self.hom_exog], idx)
            beta_hat = np.zeros((self.N, self.K))
            beta_hat[:,self.het_exog] = beta[:,self.het_exog]
            beta_hat[:, self.hom_exog] = theta_hat
        else: # fully heterogeneous model (do nothing)
            beta_hat = beta
        return beta_hat
            
    def fit(self, alpha_init=None, xi_init=None, beta_init=None, nu=1e-3, 
            hess=True, fast=True, eps=1e-6, iter_max=500):
        
        """
        fit:
            estimate a semiparametric nonlinear model with two-way FEs.
    
        INPUTS:
        ------
        `alpha_init'    : (N-1) x 1 (Numpy) array, initial guess for alpha;
        `xi_init'       : T x 1 (Numpy) array, initial guess for xi;
        `beta_init'     : K x 1 or N x K (Numpy) array, initial guess for beta;
        `nu'            : learning rate (positive small number);
        `hess'          : 0/1, not use/use hessian steps;
        `fast'          : 0/1 not use/use FPMLE++;
        `eps'           : stopping convergence criterion;
        `iter_max'      : maximum number of iterations.       
        
        OUTPUT:
        ------
        res             : list of point estimates.  
        
        """
        if alpha_init is None:
            alpha_init = np.zeros(self.N-1) 
        if xi_init is None:
            xi_init = np.zeros(self.T) #+ 1e-5
        if beta_init is None:
            if self.het:
                beta_init = np.zeros((self.N, self.K))
            else:
                beta_init = np.zeros(self.K) 
        
        obj_list = [1e35]
        obj_diff_list = [2*eps]
        alpha_list = [[alpha_init, beta_init]]
        xi_list = [xi_init]
        beta_list = [beta_init]
        niter = 0
        if fast: # run FPMLE++
            while (obj_diff_list[-1]>eps) & (niter<iter_max):
                # step 1
                alpha_list.append(self.fast_inner_loop_indiv(alpha_list[-1][0],
                           xi_list[-1], beta_list[-1], nu, hess))
                # step 2 
                xi_list.append(self.fast_inner_loop_time(alpha_list[-1][0],
                           xi_list[-1], alpha_list[-1][1], nu, hess))
                # step 3
                beta_list.append(self.fast_outer_loop_slope_coeff(
                    alpha_list[-1][0], xi_list[-1], alpha_list[-1][1], nu, 
                    hess)) 
                obj_list.append(self.objective(alpha_list[-1][0], xi_list[-1], 
                                                   beta_list[-1]))
                obj_diff_list.append(obj_list[-2]-obj_list[-1])
                niter += 1
                
        else: # run FPMLE
           while (obj_diff_list[-1]>eps) & (niter<iter_max):
                # step 1
                alpha_list.append(self.inner_loop_indiv(np.insert(
                    alpha_list[-1][0], self.indiv_ref,0),
                           xi_list[-1], beta_list[-1]))
                # step 2 
                xi_list.append(self.inner_loop_time(alpha_list[-1][0],
                           xi_list[-1], alpha_list[-1][1]))
                # step 3
                beta_list.append(self.outer_loop_slope_coeff(alpha_list[-1][0],
                                        xi_list[-1], alpha_list[-1][1]))
                obj_list.append(self.objective(alpha_list[-1][0], xi_list[-1], 
                                                   beta_list[-1]))
                obj_diff_list.append(obj_list[-2] - obj_list[-1])
                niter += 1
        return beta_list[-1], alpha_list[-1][0], xi_list[-1]
            
            
            
        
        
        
        