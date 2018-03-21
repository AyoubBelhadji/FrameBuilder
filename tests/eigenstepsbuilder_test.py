# coding: utf-8

# In[1]:
from FrameBuilder.eigenstepsbuilder import *
from FrameBuilder.eigensteps import *
import numpy as np
import matplotlib.pyplot as plt

N= 50
d = 2
E = np.zeros((N,N)) #(d,N)
mu_vector = d/N*np.ones((N,1))
lambda_vector = np.zeros((N))
lambda_vector[0:d] = np.ones((d))

mu_vector = np.linspace(1, 0.1, num=N)
sum_mu_vector = np.sum(mu_vector)
mu_vector = d/sum_mu_vector*mu_vector

#mu_vector = d/N*np.ones((N,1))

E_test = get_eigensteps_random(mu_vector,lambda_vector,N,d)
E_ = np.zeros((d,N+1))
for i in range(d):
    E_[i,1:N+1] = E_test[i,:] 
F_test = get_F(d,N,np.asmatrix(E_),mu_vector)

print("The covariance matrix")
print(np.dot(F_test[:,0:N],F_test[:,0:N].T))
print("The input vector of the spectrum")
print(lambda_vector)
print("The diagonal of the gram matrix")
print(np.diag(np.dot(F_test[:,0:N].T,F_test[:,0:N])))
print("The input vector of the lengths")
print(mu_vector)
