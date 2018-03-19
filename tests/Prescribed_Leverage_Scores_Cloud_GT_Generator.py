
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

# ## Build eigensteps

# In[ ]:




# ## Build a frame given an eigensteps

# In[1874]:

#E_test_1 = np.matrix('0 0 0 0 0.666 1.666;0 0 0.333 1.333 1.666 1.666;0 1.0 1.666 1.666 1.666 1.666')


# In[2024]:

E_test_1 = np.matrix('0 1.0 1.666 1.666 1.666 1.666;0 0 0.333 1.333 1.666 1.666;0 0 0 0 0.666 1.666')


# In[2025]:

E_test_1


# In[1876]:

E_2 = 3/5*E_test_1


# In[1880]:

[d,N] = np.shape(E_test_1) ## The dimensions of the eigensteps matrix


# In[1849]:




# In[2306]:

def get_index_lists_I_and_J(E,n,N,d):
    I_n = list(range(d))
    J_n = list(range(d))
    n_ = n+1
    for m in reversed(range(d)):
        if E[m,n_-1] in E[J_n,n_]:
            del I_n[m]
            t_J_n = [i for i in J_n if E[i,n_] == E[m,n_-1] ]
            m_max = max(t_J_n)
            del J_n[m_max]
    return I_n,J_n


# In[2027]:

def diff_of_lists(first, second):
        second = set(second)
        return [item for item in first if item not in second]


# In[2028]:

def get_permutation_I(I_n,d):
    permutation = [0]*d
    r_n = np.shape(I_n)[0]
    complementary_I_n = diff_of_lists(list(range(d)),I_n)
    c_r_n = d-r_n
    for i in range(r_n):
        permutation[I_n[i]] = i
    for i in range(c_r_n):
        permutation[complementary_I_n[i]] = i + r_n
    return permutation


# In[2029]:

def d_minus_x(x):
    return d-x-1


# In[2318]:

def get_v_n_w_n(E,I_n,J_n,d,n):
    r_n = np.shape(I_n)[0]
    v_n = np.zeros((r_n))
    w_n = np.zeros((r_n))
    permutation_I_n = get_permutation_I(I_n,d)
    permutation_J_n = get_permutation_I(J_n,d)
    #print(n)
    for m in I_n:
        #print(m)
        #print(d_minus_x(m))
        v_n_index = permutation_I_n[m]
        nom_v_n = E[m,n]*np.ones((r_n,1)) - E[list(J_n),n+1]
        I_n_without_m = diff_of_lists(I_n,[m])
        cardinal_I_n_without_m = np.shape(I_n_without_m)[0]
        denom_v_n = E[m,n]*np.ones((cardinal_I_n_without_m,1)) - E[list(I_n_without_m),n]
        v_n[v_n_index] = np.sqrt(-np.prod(nom_v_n)/np.prod(denom_v_n))
    for m in J_n:
        w_n_index = permutation_J_n[m]
        nom_w_n = E[m,n+1]*np.ones((r_n,1)) - E[list(I_n),n]
        J_n_without_m = diff_of_lists(J_n,[m])
        cardinal_J_n_without_m = np.shape(J_n_without_m)[0]
        denom_w_n = E[m,n+1]*np.ones((cardinal_J_n_without_m,1)) - E[list(J_n_without_m),n+1]
        w_n[w_n_index] = np.sqrt(np.prod(nom_w_n)/np.prod(denom_w_n))
        
    return v_n,w_n


# In[2031]:

def get_permutation_matrix(permutation,d):
    permutation_matrix = np.zeros((d,d))
    for i in range(d):
        for j in range(d):
            if i == permutation[j]:
                permutation_matrix[i,j] = 1
    return permutation_matrix


# In[2321]:

def get_W_n_matrix(E,I_n,J_n,d,n):
    r_n = np.shape(I_n)[0]
    v_n,w_n = get_v_n_w_n(E,I_n,J_n,d,n)
    W_n_matrix = np.zeros((r_n,r_n))
    permutation_I_n = get_permutation_I(I_n,d)
    permutation_J_n = get_permutation_I(J_n,d)
    for m in I_n:
        for m_ in J_n:
            v_n_index = permutation_I_n[m]
            w_n_index = permutation_J_n[m_]
            W_n_matrix[v_n_index,w_n_index] = 1/(E[m_,n+1]-E[m,n])*v_n[v_n_index]*w_n[w_n_index]
    return W_n_matrix


# In[1861]:

def get_padded_vector(v,d):
    r_n = np.shape(v)[0]
    v_padded = np.zeros((d,))
    v_padded[0:r_n] = v
    return v_padded


# In[2183]:

def get_extended_matrix_W(W_n_matrix,d):
    r_n = np.shape(W_n_matrix)[0]
    W_extended = np.eye(d)
    W_extended[0:r_n,0:r_n] = W_n_matrix
    return W_extended


# In[2396]:

def get_F(d,N,E):
    F_test = np.zeros((d,N))
    for n in range(N):
        #print(n)
        F_test[:,n] = get_F_n(n+1,d,N,E)
    return F_test


# In[2271]:

n = 4  ## The "level" 
I_n,J_n = get_index_lists_I_and_J(E_test_1,n,N,d)


# In[2272]:

permutation_I_n = get_permutation_I(I_n,d)

permutation_matrix_I_n = get_permutation_matrix(permutation_I_n,d)

permutation_J_n = get_permutation_I(J_n,d)

permutation_matrix_J_n = get_permutation_matrix(permutation_J_n,d)


# In[2273]:

I_n


# In[2274]:

J_n


# In[2275]:

permutation_I_n


# In[2276]:

permutation_J_n


# In[2277]:

permutation_matrix_I_n


# In[2278]:

permutation_matrix_J_n


# In[2279]:

v_n,w_n = get_v_n_w_n(E_test_1,I_n,J_n,d)


# In[2280]:

print(v_n)


# In[2281]:

print(w_n)


# In[2282]:

W_n_matrix = get_W_n_matrix(E_test_1,I_n,J_n,d,n)


# In[2283]:

W_n_matrix 


# In[1868]:

W_extended_test = get_extended_matrix_W(W_n_matrix,d)


# In[2336]:

def get_U_n(n,d,N,E):    
    if n==1:
        return np.eye(d)
    else:

        I_n,J_n = get_index_lists_I_and_J(E,n-1,N,d)
        #print(I_n)
        #print(J_n)
        r_n = np.shape(I_n)[0]
        permutation_matrix_I_n = get_permutation_matrix(get_permutation_I(I_n,d),d)
        #print(permutation_matrix_I_n)
        permutation_matrix_J_n = get_permutation_matrix(get_permutation_I(J_n,d),d)
        #print(permutation_matrix_J_n)
        v_n,w_n = get_v_n_w_n(E,I_n,J_n,d,n-1)
        #print(v_n)
        #print(w_n)
        W_n_matrix = get_W_n_matrix(E,I_n,J_n,d,n-1)
        W_extended = get_extended_matrix_W(get_W_n_matrix(E,I_n,J_n,d,n-1),d)
        U_n = np.zeros((d,d))
        U_n_1 = get_U_n(n-1,d,N,E)
        #print(U_n_1)
        #print(W_extended)
        #print(W_n_matrix)

        U_n_plus_one = np.dot(np.dot(np.dot(U_n_1,np.transpose(permutation_matrix_I_n)),W_extended),permutation_matrix_J_n)
        #print(n)
        return U_n_plus_one


# In[2337]:

U_n_test = get_U_n(4,d,N,E_test_1)


# In[2338]:

U_n_test


# In[2339]:

np.dot(U_n_test,U_n_test.T)


# In[2377]:

def get_F_n(n,d,N,E):
    #print(n)
    if n==1:
        M = np.eye(d)
        v = M[:,0]
        return v
    I_n,J_n = get_index_lists_I_and_J(E,n-1,N,d)
    r_n = np.shape(I_n)[0]
    permutation_matrix_I_n = get_permutation_matrix(get_permutation_I(I_n,d),d)
    permutation_matrix_J_n = get_permutation_matrix(get_permutation_I(J_n,d),d)
    v_n,w_n = get_v_n_w_n(E,I_n,J_n,d,n-1)
    W_extended = get_extended_matrix_W(get_W_n_matrix(E,I_n,J_n,d,n-1),d)
    v_padded = get_padded_vector(v_n,d)
    U_n_1 = get_U_n(n-1,d,N,E)
    f_n = np.dot(np.dot(U_n_1,np.transpose(permutation_matrix_I_n)),v_padded)
    return f_n


# In[2395]:

F_test_1 = get_F_n(1,d,N,E_test_1)


# In[2388]:

print(F_test_1)


# In[2390]:

print(np.sqrt(5)/np.sqrt(6))


# In[2398]:

F_test = get_F(d,N-1,E_test_1)


# In[2347]:

print(d)


# In[2348]:

print(N)


# In[2399]:

F_test.T


# In[2400]:

plt.scatter(F_test[0,:], F_test[1,:])
plt.show()


# ## Create an eigensteps

# In[1818]:

N = 5
d = 3

E = np.zeros((N,N)) #(d,N)
mu_vector = np.asarray([0.6,0.6,0.6,0.6,0.6])
lambda_vector = np.asarray([1,1,1,0,0])


# In[1819]:

# Initialisation
E[:,N-1] = lambda_vector


# In[1820]:

E


# In[1821]:

for n in range(N-2,-1,-1):#iter([N-2,N-3,N-4,N-5,N-6]):
    #print(n)
    if n > -1:
        #print(list(range(n, 0, -1)))
        for k in range(n, -1, -1):
            #print("k is")
            #print(k)
            #n -> N-2
            #n+1 -> N-1
            #k_aux = k+1
            #print(n)
            A_n_1_k = max(E[k+1,n+1],np.sum(E[k:n+2,n+1])-np.sum(E[k+1:n+1,n])-mu_vector[n+1])
            B_array = np.zeros(k+1)
            for l in range(k+1):
                B_array[l] = np.sum(mu_vector[l:n+1])-np.sum(E[l+1:k+1,n+1])-np.sum(E[k+1:n+1,n])
                #print(l)
            #print(B_array)
            B_n_1_k = min(E[k,n+1],min(B_array))
            #print("n is:")
            #print(n)
            #print("k is:")
            #print(k)
            #print("E[k,n+1] is")
            #print(E[k,n+1])
            #print("B_n_1_k is:")
            #print(B_n_1_k)
            
            E[k,n] = (A_n_1_k+B_n_1_k)/2
            #print(A_n_1_k)
            #print(B_n_1_k)


# In[1822]:


E_column_sum = np.zeros((N))
for n in range(N):
    E_column_sum[n] = np.sum(E[:,n])


# In[1823]:

E_column_sum


# In[1824]:

E


# In[1825]:

def get_eigensteps_random(mu_vector,lambda_vector,N,d):
    E = np.zeros((N,N)) #(d,N)
    E[:,N-1] = lambda_vector
    for n in range(N-2,-1,-1):
        #print(n)
        for k in range(n, -1, -1):
            A_n_1_k = max(E[k+1,n+1],np.sum(E[k:n+2,n+1])-np.sum(E[k+1:n+1,n])-mu_vector[n+1])
            B_array = np.zeros(k+1)
            for l in range(k+1):
                B_array[l] = np.sum(mu_vector[l:n+1])-np.sum(E[l+1:k+1,n+1])-np.sum(E[k+1:n+1,n])
            B_n_1_k = min(E[k,n+1],min(B_array))
            u = np.random.uniform(0,1)
            delta_n_1_k = B_n_1_k - A_n_1_k
            E[k,n] = A_n_1_k + u*delta_n_1_k
    return E


# In[1826]:

def get_eigensteps_mean(mu_vector,lambda_vector,N,d):
    E = np.zeros((N,N)) #(d,N)
    E[:,N-1] = lambda_vector
    for n in range(N-2,-1,-1):
        #print(n)
        for k in range(n, -1, -1):
            A_n_1_k = max(E[k+1,n+1],np.sum(E[k:n+2,n+1])-np.sum(E[k+1:n+1,n])-mu_vector[n+1])
            B_array = np.zeros(k+1)
            for l in range(k+1):
                B_array[l] = np.sum(mu_vector[l:n+1])-np.sum(E[l+1:k+1,n+1])-np.sum(E[k+1:n,n])
            B_n_1_k = min(E[k,n+1],min(B_array))
            u = np.random.uniform(0,1)
            E[k,n] = A_n_1_k
    return E


# In[2658]:

N= 50
d = 2
E = np.zeros((N,N)) #(d,N)
mu_vector = d/N*np.ones((N,1))
lambda_vector = np.zeros((N))
lambda_vector[0:d] = np.ones((d))


# In[2659]:

E_test = get_eigensteps_random(mu_vector,lambda_vector,N,d)


# In[2660]:

#E_test


# In[2661]:

E_column_sum = np.zeros((N))
for n in range(N):
    E_column_sum[n] = np.sum(E_test[:,n])


# In[2662]:

#E_column_sum


# In[2663]:

E_ = np.zeros((d,N+1))


# In[2664]:

#E_[d:0,1:N+1] = E[0:d,:]
for i in range(d):
    print(i)
    E_[i,1:N+1] = E_test[i,:] 


# In[2665]:

np.shape(E_)


# In[2666]:

#E_test


# In[2667]:

#E_


# In[2668]:

np.shape(E_)


# In[2669]:

np.shape(E_test)


# In[2670]:

#F_test = get_F(d,N+1,np.asmatrix(E_))
F_test = get_F(d,N-1,np.asmatrix(E_))


# In[2671]:

#F_test


# In[2672]:

plt.scatter(F_test[0,1:N], F_test[1,1:N])
plt.show()


# In[2611]:

type(E__)


# In[2612]:

E__


# In[2712]:

mu_vector = np.linspace(1, 0.1, num=N)
sum_mu_vector = np.sum(mu_vector)
mu_vector = d/sum_mu_vector*mu_vector


# In[2725]:


E_test = get_eigensteps_random(mu_vector,lambda_vector,N,d)
E_ = np.zeros((d,N+1))
for i in range(d):
    E_[i,1:N+1] = E_test[i,:] 
F_test = get_F(d,N-1,np.asmatrix(E_))


# In[2726]:

plt.scatter(F_test[0,1:N], F_test[1,1:N])
plt.show()


# In[ ]:



