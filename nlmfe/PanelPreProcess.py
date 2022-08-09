# -*- coding: utf-8 -*-
# Ensure "normal" division
from __future__ import division

# Load library dependencies
import numpy as np

class PanelPreProcess():
    
    """
    PanelPreProcess:
        This class contains useful functions for pre-processing panel data.
    
    FUNCTIONS:
    ---------
    `sort_panel'
    `reshape_MaR'
    
    """
    
    def __init__(self):
        self.init = 'init'
    
    def sort_panel(self, A, b, unit_id, time_id):
        
        """
        sort_panel:
            sort pooled panel data by (unit_id, time_id).
        
        INPUTS:
        ------
        `A'            : n x K (Numpy) array of exogeneous variables;
        `b'            : n x 1 (Numpy) array of a scalar endogeneous variable;
        `unit_id'      : index of the (numeric) variable indexing units;
        `time_id'      : index of the (numeric) variable indexing time periods.
        
        OUTPUTS:
        -------
        `X'            : n x K (Numpy) array of covariates;
        `y'            : n x 1 (Numpy) array of scalar oucomes.
        
        """
        
        ind = np.lexsort((A[:,time_id], A[:,unit_id]))
        X = A[ind]
        y = b[ind]
        return X, y
              
    def reshape_MaR(self, A, b, unit_id, time_id):
        
        """
        reshape_MaR:
            reshape sorted pooled panel n x K data to N x T x K by imputing
            missing values with NaN.
            
        INPUTS:
        ------
        `A'            : n x K (Numpy) array of exogeneous variables sorted 
                         by unit_id and time periods;
        `b'            : n x 1 (Numpy) array of a scalar endogeneous variable
                         sorted by unit_id and time periods;
        `unit_id'      : index of a 0,...,N variable indexing units;
        `time_id'      : index of a 0,...,max_i(T_i) variable indexing time 
                         periods.
        
        OUTPUTS:
        -------
        `X'            : N X T x K (Numpy) array of exogeneous variables;
        `y'            : N x T (Numpy) array of a scalar endogeneous variable.
        
        """
        n, k = A.shape
        assert n==b.shape[0]
        cov_list = np.delete(np.arange(k), [unit_id, time_id]).astype(int)
        unit_list, T_i = np.unique(A[:,unit_id], return_counts=True)
        time_list = np.unique(A[:, time_id])
        N, T = len(unit_list), len(time_list)
        X = np.empty((N, T, len(cov_list)))
        y = np.empty((N, T))
        units_list = np.empty(N)
        X[:] = np.nan
        y[:] = np.nan
        for idx, unit in enumerate(unit_list):
            mat_unit = A[A[:,unit_id]==unit]
            time_indices = mat_unit[:,time_id].astype(int)
            idx_time = [j for j in range(T) if time_list[j] in time_indices]
            X[idx, idx_time,:] = mat_unit[:, cov_list]
            y[idx, idx_time] = np.reshape(b[A[:,unit_id]==unit], len(idx_time))
            units_list[idx] = unit
        return X, y, unit_list, time_list


