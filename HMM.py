#!/usr/bin/env python
# coding: utf-8


#%%
import numpy as np

#%%
# 随机生成观测序列和状态序列    
def simulate(T, A, B, pi):
    print("simulat model:A:\n", A, "\nB\n", B, "\npi\n", pi)

    def draw_from(probs):
        """
        1.np.random.multinomial:
        按照多项式分布，生成数据
        >>> np.random.multinomial(20, [1/6.]*6, size=2)
                array([[3, 4, 3, 3, 4, 3],
                       [2, 4, 3, 4, 0, 7]])
         For the first run, we threw 3 times 1, 4 times 2, etc.  
         For the second, we threw 2 times 1, 4 times 2, etc.
        2.np.where:
        >>> x = np.arange(9.).reshape(3, 3)
        >>> np.where( x > 5 )
        (array([2, 2, 2]), array([0, 1, 2]))
        """
        return np.where(np.random.multinomial(1,probs) == 1)[0][0]

    observations = np.zeros(T, dtype=int)
    states = np.zeros(T, dtype=int)
    states[0] = draw_from(pi)
    observations[0] = draw_from(B[states[0],:])
    for t in range(1, T):
        states[t] = draw_from(A[states[t-1],:])
        observations[t] = draw_from(B[states[t],:])
    return observations, states


# ## offline 
#%%
def forward(obs_seq, A, B, pi):
    """前向算法"""
    N = A.shape[0]
    T = len(obs_seq)
    
    # F保存前向概率矩阵
    F = np.zeros((N,T))
    F[:,0] = pi * B[:, obs_seq[0]]

    for t in range(1, T):
        for n in range(N):
            F[n,t] = np.dot(F[:,t-1], (A[:,n])) * B[n, obs_seq[t]]

    return F

def backward(obs_seq, A, B):
    """后向算法"""
    N = A.shape[0]
    T = len(obs_seq)
    # X保存后向概率矩阵
    X = np.zeros((N,T))
    X[:,-1:] = 1
#     print("x: ", X)

    for t in reversed(range(T-1)):
        for n in range(N):
#             dd = X[:,t+1] * A[n,:] * B[:, obs_seq[t+1]]
#             print(dd)
            X[n,t] = np.sum(X[:,t+1] * A[n,:] * B[:, obs_seq[t+1]])

    return X


# In[11]:


def baum_welch_train(observations, A, B, pi, criterion=0.05):
    """无监督学习算法——Baum-Weich算法"""
    n_states = A.shape[0]
    n_samples = len(observations)

    done = False
    while not done:
        # alpha_t(i) = P(O_1 O_2 ... O_t, q_t = S_i | hmm)
        # Initialize alpha
        alpha = forward(observations,A,B,pi)

        # beta_t(i) = P(O_t+1 O_t+2 ... O_T | q_t = S_i , hmm)
        # Initialize beta
        beta = backward(observations, A, B)
        # ξ_t(i,j)=P(i_t=q_i,i_{i+1}=q_j|O,λ)
        xi = np.zeros((n_states,n_states,n_samples-1))
        for t in range(n_samples-1):
            denom = np.dot(np.dot(alpha[:,t].T, A) * B[:,observations[t+1]].T, beta[:,t+1])
            for i in range(n_states):
                numer = alpha[i,t] * A[i,:] * B[:,observations[t+1]].T * beta[:,t+1].T
                xi[i,:,t] = numer / denom

        # γ_t(i)：gamma_t(i) = P(q_t = S_i | O, hmm)
        gamma = np.sum(xi,axis=1)
        # Need final gamma element for new B
        # xi的第三维长度n_samples-1，少一个，所以gamma要计算最后一个
        prod =  (alpha[:,n_samples-1] * beta[:,n_samples-1]).reshape((-1,1))
        gamma = np.hstack((gamma,  prod / np.sum(prod))) #append one more to gamma!!!
        
        # 更新模型参数
        newpi = gamma[:,0]
        newA = np.sum(xi,2) / np.sum(gamma[:,:-1],axis=1).reshape((-1,1))
        newB = np.copy(B)
        num_levels = B.shape[1]
        sumgamma = np.sum(gamma,axis=1)
        for lev in range(num_levels):
            mask = observations == lev
            newB[:,lev] = np.sum(gamma[:,mask],axis=1) / sumgamma
        
        # 检查是否满足阈值
        if np.max(abs(pi - newpi)) < criterion and \
           np.max(abs(A - newA)) < criterion and \
           np.max(abs(B - newB)) < criterion:
            done = 1
        A[:], B[:], pi[:] = newA, newB, newpi
    return newA, newB, newpi



if __name__ == "__main__":
        
    #%%
    A_ = np.array([[0.3, 0.7],[0.4, 0.6]])
    B_ = np.array([[0.1, 0.3, 0.6],[0.4, 0.5, 0.1]])
    pi_ = np.array([0.5, 0.5])
    # print("pi: ", pi)
    # pic = pi.copy()
    #%%
    observations_data, states_data = simulate(500,A_, B_, pi_)
    print("observations_data:\n",observations_data)
    print("states_data: \n",states_data)
    #%%
    # pa = A_.copy() #np.zeros_like(A_)
    # pb = B_.copy() #np.zeros_like(B_)
    # pc = np.ones_like(pi_)

    # pa = np.zeros_like(A_)
    # pb = np.zeros_like(B_)
    # pc = np.ones_like(pi_)

    pa = np.array([[0.2, 0.8],[0.5, 0.5]])
    pb = np.array([[0.3, 0.3, 0.4],[0.4, 0.3, 0.3]])
    pc = pi_.copy() #np.ones_like(pi_)
    newA, newB, newpi = baum_welch_train(observations_data, pa, pb, pc,0.05)
    print("newA: ", newA)
    print("newB: ", newB)
    print("newpi: ", newpi)

    #%%
    for i in [0,1,2]:
        print(i, " observations_data-cnt", (observations_data == i).sum())

    for i in [0,1]:
        print(i, " states_data-cnt", (states_data == i).sum())




    #%%
