# ## Givens Rotations generators

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
