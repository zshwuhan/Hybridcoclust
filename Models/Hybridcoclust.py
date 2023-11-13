#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
Hybrid Exponential Family 

"""

# Author: Hoseinipour Saeid <saeidhoseinipour@aut.ac.ir>  
#			    <saeidhoseinipour9@gmail.com>     

# License: ??????????

import itertools
from math import *
from scipy.io import loadmat, savemat
import sys
import numpy as np
import scipy.sparse as sp
from sklearn.utils import check_random_state
from sklearn.preprocessing import normalize
from sklearn.utils import check_random_state, check_array
#from coclust.utils.initialization import (random_init, check_numbers,check_array)
# use sklearn instead FR 08-05-19
#from initialization import random_init
from ..initialization import random_init
from ..io.input_checking import check_positive
#from input_checking import check_positive
from numpy.random import rand
from numpy import nan_to_num
from numpy import linalg
from datetime import datetime
import timeit

# from pylab import *



class Hybridcoclust:
    """Clustering and  Co-clustering.
    Parameters
    ----------
    n_row_clusters : int, optional, default: 2
        Number of row clusters
    n_col_clusters : int, optional, default: 2
        Number of column clusters
    init : numpy array or scipy sparse matrix, \
        shape (n_features, n_clusters), optional, default: None
        Initial column or row labels
    max_iter : int, optional, default: 100
        Maximum number of iterations
    n_init : int, optional, default: 1
        Number of time the algorithm will be run with different
        initializations. The final results will be the best output of `n_init`
        consecutive runs.
    random_state : integer or numpy.RandomState, optional
        The generator used to initialize the centers. If an integer is
        given, it fixes the seed. Defaults to the global numpy random
        number generator.
    tol : float, default: 1e-9
        Relative tolerance with regards to criterion to declare convergence
    model : str, default: "Poisson"     
        The name of distubtion based on (Sparse)Exponential Family Latent Block Model such  as:
        "Poisson", "Bernoulli", "Normal", "Gamma", "Beta".
        

    Attributes
    ----------
    row_labels_ : array-like, shape (n_rows,)
        cluster label of each row
    column_labels_ : array-like, shape (n_cols,)
        cluster label of each column
    row_labels_bayes : array-like, shape (n_rows,)
        cluster bayes label of each row
    row_labels_bayes : array-like, shape (n_rows,)
        cluster bayes label of each row
    criterion : float
        criterion obtained from the best run
    criterions_R : list of floats
        sequence of convergence_R (Complete log-likelihood function Conditional R) values during the best run
    criterions_C : list of floats
        sequence of convergence_C (Complete log-likelihood function Conditional C) values during the best run
    criterions_R_C : list of floats
        sequence of convergence_R_C (Complete log-likelihood function) values during the best run            
    """

	def __init__(self,
		 n_row_clusters = 2 , n_col_clusters = 2, model = "Poisson",
		 R_init = None, B_init = None, A_init = None, C_init = None,
		 max_iter = 100, n_init = 1, tol = 1e-9, 
		 random_state = None):
		self.n_row_clusters = n_row_clusters
		self.n_col_clusters = n_col_clusters
		self.model = model                                      # model = ("Bernoulli", "Poisson", "Normal", "Beta") 
		self.R_init = R_init                              
		self.B_init = B_init
		self.A_init = A_init
		self.C_init = C_init
		self.max_iter = max_iter
		self.n_init = n_init
		self.tol = tol
		self.random_state = check_random_state(random_state)
		self.R = None
		self.C = None
		self.B = None
		self.row_labels_ = None
		self.column_labels_= None
		self.rowcluster_matrix = None
		self.columncluster_matrix = None
		self.reorganized_matrix = None		
		self.row_labels_bayes = None
		self.column_labels_bayes= None
		self.rowcluster_matrix_bayes = None
		self.columncluster_matrix_bayes = None
		self.reorganized_matrix_bayes = None		
		self.soft_matrix = None
		self.hard_matrix = None
		self.orthogonality_F = None
		self.orthogonality_G = None
		self.MSE_1 = None
		self.MSE_2 = None
		self.criterions_R = []
		self.criterions_C = []
		self.criterions_R_C = []
		self.criterion = -np.inf
		self.runtime = None


	def fit(self, X, y=None):

		check_array(X, accept_sparse=True, dtype="numeric", order=None,
				copy=False, force_all_finite=True, ensure_2d=True,
				allow_nd=False, ensure_min_samples=self.n_row_clusters,
				ensure_min_features=self.n_col_clusters, estimator=None)
		
		criterion = self.criterion
		criterions_R_C = self.criterions_R_C
		criterions_R = self.criterions_R
		criterions_C = self.criterions_C
		row_labels_ = self.row_labels_
		column_labels_ = self.column_labels_
		row_labels_bayes = self.row_labels_bayes
		column_labels_bayes = self.column_labels_bayes

		X = X.astype(float)

		random_state = check_random_state(self.random_state) 
		seeds = random_state.randint(np.iinfo(np.int32).max, size=self.n_init)

		for seed in seeds:
			self._fit_single(X, seed, y)
			if np.isnan(self.criterion):   # c --> self.criterion
				raise ValueError("matrix may contain negative or unexpected NaN values")
			# remember attributes corresponding to the best criterion
			if (self.criterion_R_C > criterion): 
				criterions_R = self.criterions_R
				criterions_C = self.criterions_C
				criterions_R_C = self.criterions_R_C
				criterion = self.criterions_R_C
				row_labels_ = self.row_labels_
				column_labels_ = self.column_labels_
				row_labels_bayes = self.row_labels_bayes
				column_labels_bayes = self.column_labels_bayes
		self.random_state = random_state

		# update attributes
		self.criterion = criterion
		self.criterions_R = criterions_R
		self.criterions_C = criterions_C
		self.criterions_R_C = criterions_R_C
		self.row_labels_ = row_labels_ 
		self.column_labels_ = column_labels_ 
		self.row_labels_bayes = row_labels_bayes
		self.column_labels_bayes = column_labels_bayes	



	def _fit_single(self, X, random_state = None, y=None):


		m, n = X.shape
		N = n*m
		g = self.n_row_clusters
		s = self.n_col_clusters
		R = rand(m, g) if isinstance(self.R_init, type(None)) else self.R_init
		A = rand(g , s) if isinstance(self.A_init, type(None)) else self.B_init
		C = rand(n, s) if isinstance(self.C_init, type(None)) else self.C_init
		I_g = np.identity(g, dtype = None)
		I_s = np.identity(s, dtype = None)
		E_mn = np.ones((m, n))
		E_gs = np.ones((g,s))




########################################## Define Functions 

        if (self.model == "Poisson"):
           	beta = X@E_mn.T@X
           	S_X = X
        elif (self.model == "Normal"):
           	beta = E_mn
           	S_X = X
        elif (self.model == "Bernoulli"):
           	beta = E_mn
           	S_X = X
        elif (self.model == "Beta"):
           	beta = E_mn
           	S_X = np.log(X)
        else:
            print("Model name not found")


		################   Hybrib Multiplactive Update Rules   ############# 3 loop with 3 convergence_R, convergence_C, convergence_R_C
		change = True
		c_init = float(-np.inf)
		c_list_R_C = []
		runtime = []
		Orthogonal_R_list = []
		Orthogonal_C_list = []
		iteration = 0


		start = timeit.default_timer()


		while change :
			change = False


			###########################################	A 	
			for itr in range(self.max_iter):
				if isinstance(self.A_init, type(None)):
					enum = R.T@S_X.multiply(beta)@C 
					denom = R.T@E_mn.multiply(beta)@C 
					A = np.nan_to_num(enum/denom)



	        if (self.model == "Poisson"):
	        	B = np.log(A)
	        	F_X = X
	        	F_RBC = R@B@C.T
	        elif (self.model == "Normal"):
	        	sigma2 = 1
	        	B = (1/2*sigma2)*A
	        	F_X = (1/2*sigma2)*(X**2)
	        	F_RBC = (1/2*sigma2)*((R@B@C.T)**2)
	        elif (self.model == "Bernoulli"):
	        	B = np.log(A/(E_gs-A))
	        	F_X = -np.log(E_mn-X)
	        	F_RBC = -np.log(E_mn-(R@B@C.T))
	        elif (self.model == "Beta"):
	        	delta = 1
	        	B = A
	        	F_X = X
	        	F_RBC = R@B@C.T
	        else:
	            print("Model name not found")



			change = True
			c_list_R = []
			c_init_R = float(-np.inf)
			iteration = 0

			while change :
				change = False


			###########################################	R 	

				for itr in range(self.max_iter):
					if isinstance(self.R_init, type(None)):
						enum = F_X.T@C@B.T  
						denom = F_RBC.T@C@B.T

						if (self.model=="Poisson"):
							DDR = np.log(enum/denom)
						elif (self.model=="Normal"):
							sigma2 = 1
							DDR = ((2/sigma2)*(enum/denom))**0.5
						elif (self.model=="Bernoulli"):
							DDR = np.log(math.exp(enum/denom)-1)
						elif (self.model == "Gamma"):
							delta = 1
							DDR = -math.exp((-1/delta)*(enum/denom))
						else:
							print("Model name not found")
						
						R = np.nan_to_num(np.multiply(R, DDR))		

        		####################      Objective function     ###########################


        		#initial pi_k row proportion 
        		n_k = R.sum()       
        		pi_k = R.sum(axis=0)                               # r_{.k}
        		pi_k = pi_k/n_k                            #n_k or m
        		pi_k = np.asarray(pi_k)                 # (1,g)                             
        		pi_k = np.log(pi_k) 

				Statistics_term  =  np.sum(R.T@S_X.multiply(R@B@C.T)@C)
				Row_term = np.sum(pi_k@R.T) 

				convergence_R = Row_term + Statistics_term

			
				iteration += 1
				if (np.abs(convergence_R - c_init_R)  > self.tol and iteration < self.max_iter): 
					c_init_R = convergence_R
					change = True
					c_list_R.append(convergence_R)

			self.criterions_R = c_list_R


			###########################################	A 	
			for itr in range(self.max_iter):
				if isinstance(self.A_init, type(None)):
					enum = R.T@S_X.multiply(beta)@C 
					denom = R.T@E_mn.multiply(beta)@C 
					A = np.nan_to_num(enum/denom)

	        if (self.model == "Poisson"):
	        	B = np.log(A)
	        	F_X = X
	        	F_RBC = R@B@C.T
	        elif (self.model == "Normal"):
	        	sigma2 = 1
	        	B = (1/2*sigma2)*A
	        	F_X = (1/2*sigma2)*(X**2)
	        	F_RBC = (1/2*sigma2)*((R@B@C.T)**2)
	        elif (self.model == "Bernoulli"):
	        	B = np.log(A/(E_gs-A))
	        	F_X = -np.log(E_mn-X)
	        	F_RBC = -np.log(E_mn-(R@B@C.T))
	        elif (self.model == "Beta"):
	        	delta = 1
	        	B = A
	        	F_X = X
	        	F_RBC = R@B@C.T
	        else:
	            print("Model name not found")




			change = True
			c_list_C = []
			c_init_C = float(-np.inf)
			iteration = 0

			while change :
				change = False

				###########################################	C	
				for itr in range(self.max_iter):
					if isinstance(self.C_init, type(None)):
						enum = B.T@R.T@F_X 
						denom = B.T@R.T@F__RBC
						
						if (self.model=="Poisson"):
							DDC = np.log(enum/denom)
						elif (self.model=="Normal"):
							sigma2 = 1
							DDC = ((2/sigma2)*(enum/denom))**0.5
						elif (self.model=="Bernoulli"):
							DDC = np.log(math.exp(enum/denom)-1)
						elif (self.model == "Gamma"):
							delta = 1
							DDC = -math.exp((-1/delta)*(enum/denom))
						else:
							print("Model name not found")

						C = np.nan_to_num(np.multiply(C, DDC))				


				#########################  Objective function = convergence_C #########################

		        #initial rho_h column proportions 
		        n_h = C.sum()
		        #print(n_h)                                      
		        rho_h = C.sum(axis=0)                               #  c_{.h}
		        rho_h = rho_h/n_h
		        rho_h = np.asarray(rho_h)             # (1,s)
		        rho_h = np.log(rho_h)


				Statistics_term  =  np.sum(R.T@S_X.multiply(R@B@C.T)@C)
				Col_term = np.sum(rho_h@C.T)

				convergence_C = Col_term + Statistics_term

			
				iteration += 1
				if (np.abs(convergence_C - c_init_C)  > self.tol and iteration < self.max_iter): 
					c_init_C = convergence_C
					change = True
					c_list_C.append(convergence_C)

			self.criterions_C = c_list_C


			##########################   Normalization   #####################################

			DR = np.diagflat(R.sum(axis = 0))
			DC = np.diagflat(C.sum(axis = 0))

			###### Version 1
			R_bayes = R@np.diagflat((B@DC).sum(axis = 0))
			R_bayes_cluster = np.zeros_like(R)
			R_bayes_cluster[np.arange(len(R)),np.sort(np.argmax(R_bayes,axis=1))] = 1

			C_bayes = C@np.diagflat((DR@B).sum(axis = 0))
			C_bayes_cluster = np.zeros_like(C)
			C_bayes_cluster[np.arange(len(C)),np.sort(np.argmax(C_bayes,axis=1))] = 1

			###### Version 2
			R = R@np.diagflat(np.power(R.sum(axis = 0), -1))
			B = DR@B@DC
			C = (np.diagflat(np.power(C.sum(axis = 0), -1))@C.T).T   

			R_cluster = np.zeros_like(R)
			R_cluster[np.arange(len(R)),np.sort(np.argmax(R,axis=1))] = 1
			C_cluster = np.zeros_like(C)
			C_cluster[np.arange(len(C)),np.sort(np.argmax(C,axis=1))] = 1

			######################### Orthogonality #################################

			Orthogonal_R = linalg.norm(R.T@R - I_g, 'fro')              # ||sum(F^TF - I)^2||^0.5
			Orthogonal_C = linalg.norm(C.T@C - I_s, 'fro')

			Orthogonal_R_list.append(Orthogonal_F)
			Orthogonal_C_list.append(Orthogonal_G)

			################################## Objective function = convergence_R_C #########################

        	#initial pi_k row proportion 
        	n_k = R.sum()       
        	pi_k = R.sum(axis=0)                               # r_{.k}
        	pi_k = pi_k/n_k                            #n_k or m
        	pi_k = np.asarray(pi_k)                 # (1,g)                             
        	pi_k = np.log(pi_k) 
			
	        #initial rho_h column proportions 
	        n_h = C.sum()
	        #print(n_h)                                      
	        rho_h = C.sum(axis=0)                               #  c_{.h}
	        rho_h = rho_h/n_h
	        rho_h = np.asarray(rho_h)             # (1,s)
	        rho_h = np.log(rho_h)


			Statistics_term  =  np.sum(R.T@S_X.multiply(R@B@C.T)@C)
			Row_term = np.sum(pi_k@R.T) 
			Col_term = np.sum(rho_h@C.T)

			convergence_R_C = Row_term + Col_term + Statistics_term

			iteration += 1
			if (np.abs(convergence_R_C - c_init)  > self.tol and iteration < self.max_iter): 
				c_init = convergence_R_C
				change = True
				c_list.append(convergence_R_C)


		stop = timeit.default_timer()
		runtime.append(stop - start)


		self.max_iter = iteration
		self.runtime = runtime
		self.criterion_R_C = convergence_R_C
		self.criterions_R_C = c_list_R_C
		self.R = R_cluster
		self.B = B
		self.A = A 
		self.C = C_cluster
		self.soft_matrix = R@B@C.T
		self.hard_matrix = R_cluster.T@X@C_cluster
		self.rowcluster_matrix = R_cluster@R_cluster.T@X
		self.columncluster_matrix = X@C_cluster@C_cluster.T
		self.reorganized_matrix = R_cluster@R_cluster.T@X@C_cluster@C_cluster.T
		self.rowcluster_matrix_bayes = R_bayes_cluster@R_bayes_cluster.T@X
		self.columncluster_matrix_bayes = X@C_bayes_cluster@C_bayes_cluster.T
		self.reorganized_matrix_bayes = R_bayes_cluster@R_bayes_cluster.T@X@C_bayes_cluster@C_bayes_cluster.T
		self.row_labels_ = [x+1 for x in np.argmax(R, axis =1).tolist()]
		self.column_labels_ = [x+1 for x in np.argmax(C, axis =1).tolist()]
		self.row_labels_bayes = [x+1 for x in np.argmax(R_bayes, axis =1).tolist()]
		self.column_labels_bayes = [x+1 for x in np.argmax(C_bayes, axis =1).tolist()]
		self.orthogonality_R = Orthogonal_R_list
		self.orthogonality_C = Orthogonal_C_list
		self.MSE_1 =  linalg.norm( X - (R_cluster@R_cluster.T@X@C_cluster@C_cluster.T), 'fro')**2/N
		self.MSE_2 =  linalg.norm( X - (R_cluster@B@C_cluster.T), 'fro')**2/N
