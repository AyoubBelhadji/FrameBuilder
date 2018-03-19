
# coding: utf-8

# In[1]:
from FrameBuilder.givensbuilder import *
import numpy as np
import matplotlib.pyplot as plt

N = 50
d = 2

#lv_scores_vector = d/N*np.ones(N)
lv_scores_vector = np.linspace(1, 50, num=N)
lv_scores_vector = d*lv_scores_vector/(np.sum(lv_scores_vector))


I_sorting =  list(reversed(np.argsort(lv_scores_vector)))
lv_scores_vector = np.asarray(list(reversed(np.sort(lv_scores_vector))))


Q = get_orthogonal_matrix_using_givens(N,d,lv_scores_vector)

# ## Scatter the cloud of points


#plt.scatter(Q[:,0], Q[:,1])
#plt.show()

print(Q)

# #### We observe that almost all the points are aligned with the canonical axis
