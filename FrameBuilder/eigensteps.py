import numpy as np

def get_eigensteps_random(mu_vector,lambda_vector,N,d):
    E = np.zeros((N,N))
    E[:,N-1] = lambda_vector
    for n in range(N-2,-1,-1):
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

def get_eigensteps_mean(mu_vector,lambda_vector,N,d):
    E = np.zeros((N,N))
    E[:,N-1] = lambda_vector
    for n in range(N-2,-1,-1):
        for k in range(n, -1, -1):
            A_n_1_k = max(E[k+1,n+1],np.sum(E[k:n+2,n+1])-np.sum(E[k+1:n+1,n])-mu_vector[n+1])
            B_array = np.zeros(k+1)
            for l in range(k+1):
                B_array[l] = np.sum(mu_vector[l:n+1])-np.sum(E[l+1:k+1,n+1])-np.sum(E[k+1:n,n])
            B_n_1_k = min(E[k,n+1],min(B_array))
            u = np.random.uniform(0,1)
            E[k,n] = A_n_1_k
    return E
