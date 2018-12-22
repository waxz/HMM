#%%
from HMM import baum_welch_train, simulate
import numpy as np
# ## online
# Online Learning with Hidden Markov Models[ Gianluigi Mongillo ,Sophie Deneve]

# Start with: 
# - initial guess for the model parameters, θ(0) = [A,B], 
# - initial state probabilities, ˆql (0) ≡ P(x0 = l), 
# - sufficient statistics, ˆφh ijk(0)

# specific data struct
# 


#%%
def delta(i, j):
    if i == j:
        return 1
    else :
        return 0
def g_delta(i, j, l, h):
    return delta(i, l )*delta(j, h)


class Q_data:
    def __init__(self,Q_):
        self.Q = Q_
    
    def updateQ(self, new_obs_k, Gama_):
        Q0 = self.Q.copy()
        # print("update Q with K: ", new_obs_k, "\nGama: \n", Gama_, "\nQ0: \n", Q0)
        self.Q[0] = (Gama_[0][0][new_obs_k] * Q0[0]) + (Gama_[1][0][new_obs_k] * Q0[1])  

        self.Q[1] = (Gama_[0][1][new_obs_k] * Q0[0]) + (Gama_[1][1][new_obs_k] * Q0[1])  
        
        # print("updated Q = \n", self.Q, "normalise : ", self.Q.sum())


    def getQ(self):
        return self.Q


# In[32]:


class Gama_data:
    def __init__(self, Gama_):
        self.Gama = Gama_
        self.prob_sum = None
        self.A = None
        self.B = None
    
    def updateModel(self, A_, B_):
        self.A = A_
        self.B = B_
        
    def updateGama(self, new_obs_k, Q_old):
        A_ = self.A
        B_ = self.B
        Q_ = Q_old
        # print("update Gama with k = ", new_obs_k, "\n Q = ", Q_, "\n A : \n", A_, "\n B: \n", B_)
        prob_sum = (A_[0][0]*B_[0][new_obs_k]*Q_[0] + A_[0][1]*B_[1][new_obs_k]*Q_[0]\
                  + A_[1][0]*B_[0][new_obs_k]*Q_[1] + A_[1][1]*B_[1][new_obs_k]*Q_[1]  )
        self.prob_sum = prob_sum
        self.Gama[0][0][new_obs_k] = (A_[0][0]*B_[0][new_obs_k]) / prob_sum

        self.Gama[0][1][new_obs_k] = (A_[0][1]*B_[1][new_obs_k]) / prob_sum

        self.Gama[1][0][new_obs_k] = (A_[1][0]*B_[0][new_obs_k]) / prob_sum

        self.Gama[1][1][new_obs_k] = (A_[1][1]*B_[1][new_obs_k]) / prob_sum
        # print("updated Gama = \n", self.Gama)
        
    
    
    def get(self, i_, j_, k_):
        return self.Gama[i_][j_][k_]
        
    def getGama(self):
        return self.Gama
    
# how to define a datastruct
# member data
# method

# for Gama
# contatin latest Gama
# updata Gama from Q and yT, A, B


# In[42]:


class Fi_data:
    def __init__(self, Fi_, time_factor_):
        self.Fi = Fi_
        self.time_factor = time_factor_

    def updateTimeFactor(self, time_factor_):
        self.time_factor = time_factor_

        
    def updateMode(self,new_obs_k ,Gama_, Q_):
        Fi = self.Fi.copy()
        for i in range(self.Fi.shape[0]):
            for j in range(self.Fi.shape[1]):
                for h in range(self.Fi.shape[2]):
                    for k in range(self.Fi.shape[3]):
                        # print("updata Fi with: Gama_[0][h][k]: ",Gama_[0][h][k], "\nself.Fi[i][j][0][k]: ", self.Fi[i][j][0][k])
                        # print(" self.time_factor: ",  self.time_factor)
                        # print("delta(new_obs_k, k)*g_delta(i, j ,0, h)*Q_[0]",delta(new_obs_k, k), ", ", g_delta(i, j ,0, h), ", ",Q_[0])
                        self.Fi[i][j][h][k]  = (Gama_[0][h][new_obs_k] * (Fi[i][j][0][k] + self.time_factor*(delta(new_obs_k, k)*g_delta(i, j ,0, h)*Q_[0] - Fi[i][j][0][k] ) ) ) +\
                                               (Gama_[1][h][new_obs_k] * (Fi[i][j][1][k] + self.time_factor*(delta(new_obs_k, k)*g_delta(i, j ,1, h)*Q_[1] - Fi[i][j][1][k] ) ) )
                
                        # print("updated Fi[i][j][h][k]: ", self.Fi[i][j][h][k])
        # print("updated Fi:\n", self.Fi)
    def getFi(self):
        return self.Fi


# In[43]:

# In[62]:

#%%
class HmmTrainer:
    def __init__(self, state_dim_, obs_dim_):
        transition_probability = {
            'Healthy': {'Healthy': 0.7, 'Fever': 0.3},
            'Fever': {'Healthy': 0.4, 'Fever': 0.6},
        }
        # 观测概率矩阵B
        emission_probability = {
            'Healthy': {'normal': 0.5, 'cold': 0.4, 'dizzy': 0.1},
            'Fever': {'normal': 0.1, 'cold': 0.3, 'dizzy': 0.6},
        }
        # print("transition_probability:\n", transition_probability)
        # print("emission_probability:\n", emission_probability)
        A = np.array([[0.15, 0.85],
                      [0.65, 0.35]])
        B = np.array([[0.2, 0.5, 0.3],
                    [0.4, 0.2, 0.4]])
        pi = np.array([0.5, 0.5])
        self.A = np.array([[0.7, 0.3],
                           [0.4, 0.6]])
        self.B = np.array([[0.5, 0.4, 0.1 ],
                           [0.1, 0.3, 0.6]])
        
        self.A = np.array([[0.5, 0.5],
                           [0.5, 0.5]])
        # self.B = np.array([[0.4, 0.3, 0.3 ],
        #                    [0.4, 0.3, 0.3]])

        self.pi = np.ones([state_dim_])/ float(state_dim_)
        
        self.A = A


        self.B = B
        
        self.pi = pi

        print("A:\n", self.A, "\nB:\n", self.B, "\npi:\n", self.pi)
        
        Gama_ = 0.0*np.ones([2,2,3])

        print("Gama:\n", Gama_)
        
        self.gama = Gama_data(Gama_)
        
        Q_ = self.pi
        
        self.q = Q_data(Q_)
        
        Fi_ = np.zeros([state_dim_,state_dim_,state_dim_,obs_dim_])/float(obs_dim_)
        print("信息概率:\n ", Fi_)
        
        time_factor_ = 0.001

        # print("time_factor: ", time_factor_)
        
        self.fi = Fi_data(Fi_, time_factor_)
        
        # print("Fuck===========")
#         self.Q 
#         self.Gama

    def online_train(self, data_):
        i = 0

        for ob in data_:
#             updata gama
            self.gama.updateModel(self.A,self.B)
    
#             print("debug gama:\n", self.gama.getGama(),self.q.getQ())
            self.gama.updateGama(new_obs_k=ob,Q_old=self.q.getQ())


#             update fi
            self.fi.updateMode(ob, self.gama.getGama(), self.q.getQ())

#             updata q
            self.q.updateQ(ob, self.gama.getGama())

            i += 1
            if i > 100:
                self.fi.updateTimeFactor(0.0001)
                self.updateParam()


        print("A:\n",self.A)
        print("B:\n",self.B)


        self.updateParam()


    def learn(self, data_, online = False):
        
        # print("get data:\n", data_, "train_mode online = ", online)

        if not online:
            
            self.A, self.B, self.pi = baum_welch_train(data_, self.A, self.B, self.pi, criterion=0.05)
        
        else :
            self.online_train(data_)

    
    def updateParam(self):


        # print("Fuck A:\n",self.A)
        # print("Fuck B:\n",self.B)

        f = self.fi.getFi()
        A_ = self.A.copy()

        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                A_[i][j] = f[i,j,:,:].sum()/ f[i,:,:,:].sum()

        B_ = self.B.copy()

        for j in range(B.shape[0]):
            for k in range(B.shape[1]):
                B_[j][k] = f[:,j,:,k].sum()/ f[:,j,:,:].sum()

        self.A = A_
        # self.B = B

        # print("update A", A_)
        # print("update B", B_)

        return A_, B_

        
#         only when all possible observation is detected
            




#%%
h = HmmTrainer(2,3)

#%%
y = np.array([0,1,2,0,1,2,0,1,2,2,2,2,2,1,1,1])
# y = np.array([0,0,0,0,0,0,0,0,0,0,0])
h.learn(y, True)


#%%



#%%
A = np.array([[0.1,0.9],
             [0.7, 0.3]])
B = np.array([[0.2, 0.5, 0.3],
             [0.4, 0.2, 0.4]])
pi = np.array([0.5, 0.5])
num = int(5e4)
data , _ = simulate(num, A, B, pi)

#%%
h = HmmTrainer(2,3)

h.learn(data, True)

#%%
h.learn(data, True)
newA,newB = h.updateParam()
print("newA:\n", newA)
print("newB:\n", newB)


#%%
f = h.fi.getFi()
A = h.A
# i j h k 
for i in range(A.shape[0]):
    for j in range(A.shape[1]):
        A[i][j] = f[i,j,:,:].sum()/ f[i,:,:,:].sum()

B = h.B

for j in range(B.shape[0]):
    for k in range(B.shape[1]):
        B[j][k] = f[:,j,:,k].sum()/ f[:,j,:,:].sum()

print("A:\n", A, "\nB:\n", B)

#%%
f

#%%
h.q.getQ()

#%%
h.gama.getGama()

#%%
