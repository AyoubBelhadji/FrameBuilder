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



def diff_of_lists(first, second):
        second = set(second)
        return [item for item in first if item not in second]



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


def d_minus_x(x):
    return d-x-1



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



def get_permutation_matrix(permutation,d):
    permutation_matrix = np.zeros((d,d))
    for i in range(d):
        for j in range(d):
            if i == permutation[j]:
                permutation_matrix[i,j] = 1
    return permutation_matrix



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


def get_padded_vector(v,d):
    r_n = np.shape(v)[0]
    v_padded = np.zeros((d,))
    v_padded[0:r_n] = v
    return v_padded



def get_extended_matrix_W(W_n_matrix,d):
    r_n = np.shape(W_n_matrix)[0]
    W_extended = np.eye(d)
    W_extended[0:r_n,0:r_n] = W_n_matrix
    return W_extended


def get_F(d,N,E):
    F_test = np.zeros((d,N))
    for n in range(N):
        #print(n)
        F_test[:,n] = get_F_n(n+1,d,N,E)
    return F_test


  
  

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