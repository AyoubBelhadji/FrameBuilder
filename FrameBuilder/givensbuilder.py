from FrameBuilder.givenshelpers import *
import numpy as np

def get_orthogonal_matrix_using_givens(N,d,lv_scores_vector):
  # ## Transforming an idendity matrix to an orthogonal matrix with prescribed lengths

  # In[4]:
# ## Initialisation by the identity matrix

  Q = np.zeros((N,d))
  for _ in range(0,d):
      Q[_,_] = 1


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
  return Q

