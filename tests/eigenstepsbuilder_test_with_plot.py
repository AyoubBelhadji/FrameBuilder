import sys
#sys.path.append('../')
sys.path.insert(0, '..')
from FrameBuilder.eigenstepsbuilder import *
import numpy as np
import matplotlib.pyplot as plt


N = 10
d = 2
Q = np.zeros((N,d))
for _ in range(0,d):
    Q[_,_] = 1
lv_scores_vector = d/N*np.ones(N)
lv_scores_vector = np.linspace(1, 1000, num=N)
lv_scores_vector = d*lv_scores_vector/(np.sum(lv_scores_vector))
I_sorting =  list(reversed(np.argsort(lv_scores_vector)))
lv_scores_vector = np.asarray(list(reversed(np.sort(lv_scores_vector))))

#N= 50
#d = 2
E = np.zeros((N,N)) #(d,N)
mu_vector = lv_scores_vector
lambda_vector = np.zeros((N))
lambda_vector[0:d] = np.ones((d))

#mu_vector = np.linspace(1, 0.1, num=N)
sum_mu_vector = np.sum(mu_vector)
mu_vector = d/sum_mu_vector*mu_vector

#mu_vector = d/N*np.ones((N,1))

E_test = get_eigensteps_random(mu_vector,lambda_vector,N,d)
E_ = np.zeros((d,N+1))
for i in range(d):
    E_[i,1:N+1] = E_test[i,:] 
F_test = get_F(d,N,np.asmatrix(E_),mu_vector)
Q = np.transpose(F_test)


print("The covariance matrix")
print(np.dot(F_test[:,0:N],F_test[:,0:N].T))
print("The input vector of the spectrum")
print(lambda_vector)
print("The diagonal of the gram matrix")
print(np.diag(np.dot(F_test[:,0:N].T,F_test[:,0:N])))
print("The input vector of the lengths")
print(mu_vector)

fig=plt.figure(1)
plt.gca().set_aspect('equal')
plt.axis([-0.7,0.7,-0.7,0.7])
ax=fig.add_subplot(1,1,1)
plt.scatter(Q[:,0], Q[:,1],marker = 'o', s=10, color = 'red')
circ_list = []
for i in range(N):
    circ_i=plt.Circle((0,0), radius=np.sqrt(lv_scores_vector[i]), color="#3F5D7D", fill=False)
    ax.add_patch(circ_i)
    #circ_list.append(circ_list)
#circ_1=plt.Circle((0,0), radius=0.1, color='g', fill=False)
#circ_2=plt.Circle((0,0), radius=0.05, color='r', fill=False)
#ax.add_patch(circ_1)
#ax.add_patch(circ_2)
plt.show()
