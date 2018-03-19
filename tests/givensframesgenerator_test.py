
# coding: utf-8

# In[1]:

import numpy as np
from itertools import combinations
from scipy.stats import binom
import scipy.special
import matplotlib.pyplot as plt


# ## Givens Rotations generators

# In[2]:

def t_func(q_i,q_j,q_ij,l_i,l_j): 
    # t in section 3.1 Dhillon (2005) 
    delta = np.power(q_ij,2)-(q_i-l_i)*(q_j-l_i)
    if delta<0:
        print(delta)
        print("error sqrt")
    t = q_ij - np.sqrt(delta) 
    t = t/(q_j-l_i)
    return t
     
def G_func(i,j,q_i,q_j,q_ij,l_i,l_j,N): 
    # Gitens Rotation 
    G=np.eye(N) #identité 
    t = t_func(q_i,q_j,q_ij,l_i,l_j)
    c = 1/(np.sqrt(np.power(t,2)+1))
    s = t*c
    G[i,i]=c
    G[i,j]=s 
    G[j,i]= -s
    G[j,j]= c
    return G


# ## Initialisation by the identity matrix

# In[3]:

N = 979
d = 2
Q = np.zeros((N,d))
for _ in range(0,d):
    Q[_,_] = 1
lv_scores_vector = d/N*np.ones(N)
lv_scores_vector = np.linspace(1, 1000, num=N)
lv_scores_vector = d*lv_scores_vector/(np.sum(lv_scores_vector))


I_sorting =  list(reversed(np.argsort(lv_scores_vector)))
lv_scores_vector = np.asarray(list(reversed(np.sort(lv_scores_vector))))



# ## Transforming an idendity matrix to an orthogonal matrix with prescribed lengths

# In[4]:


i = d-1
j = d
for t in range(N-1):
    delta_i = np.abs(lv_scores_vector[i] - np.power(np.linalg.norm(Q[i,:]),2))
    delta_j = np.abs(lv_scores_vector[j] - np.power(np.linalg.norm(Q[j,:]),2))
    q_i = np.power(np.linalg.norm(Q[i,:]),2)
    q_j = np.power(np.linalg.norm(Q[j,:]),2)
    q_ij = np.dot(Q[i,:],Q[j,:].T)
    l_i = lv_scores_vector[i]
    l_j = lv_scores_vector[j]
    G = np.eye(N)
    if delta_i <= delta_j:
        l_k = q_i + q_j -l_i
        G = G_func(i,j,q_i,q_j,q_ij,l_i,l_k,N)
        Q = np.dot(G,Q)
        i = i-1
    else:
        l_k = q_i + q_j -l_j
        G = G_func(i,j,q_j,q_i,q_ij,l_j,l_k,N)
        Q = np.dot(G,Q)
        j = j+1


# ## Scatter the cloud of points

# In[5]:

plt.scatter(Q[:,0], Q[:,1])
plt.show()


# #### We observe that almost all the points are aligned with the canonical axis
