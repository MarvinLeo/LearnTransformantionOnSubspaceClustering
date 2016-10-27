from __future__ import division
import numpy as np
import math
from scipy import linalg

def subgradient_matrix(A, theta=1):
    m = A.shape[0];    #nrow
    n = A.shape[1];    #ncol
    u, s, v = linalg.svd(A)  #implement svd
    bad = s < theta         #check for num singular values                  
    num_s = len(s[bad])    #smaller than theta
    num_s = max(n-m, num_s) #n-s cannot larger than m
    u1 = u[:, :n - num_s]  #partition U, u1 with n-s col
    u2 = u[:, n-num_s:]    #u2
    v1 = v[:, :n-num_s]    #partition V, v1 with n-s col
    v2 = v[:, n-num_s:]    #v2
    B = np.random.rand(m-n+num_s, num_s)  #generate a random matrix B 
    #B = np.random.rand(m-num_s, n-num_s)
    B = B/linalg.norm(B)   #normalize B
    patial_A = np.dot(u1, v1.transpose()) + np.dot(np.dot(u2,B), v2.transpose())     
    return patial_A
    
