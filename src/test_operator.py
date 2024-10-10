#%%
from model_sir import solve_SIR, solve_SIR_test
from network import DeepONet
import matplotlib.pyplot as plt
import pickle
import numpy as np
import time

from jax import random, jit, device_put
from jax.config import config
from functools import partial
from torch.utils import data


#%%
class DataGenerator(data.Dataset):
    def __init__(self, u, y, s, 
                 batch_size=64, rng_key=random.PRNGKey(1234)):
        'Initialization'
        self.u = u # input sample
        self.y = y # location
        self.s = s # labeled data evulated at y (solution measurements, BC/IC conditions, etc.)
        
        self.N = u.shape[0]
    
        self.batch_size = batch_size
        self.key = rng_key

    def __getitem__(self, index):
        'Generate one batch of data'
        self.key, subkey = random.split(self.key)
        inputs, outputs = self.__data_generation(subkey)
        return inputs, outputs

    @partial(jit, static_argnums=(0,))
    def __data_generation(self, key):
        'Generates data containing batch_size samples'
        idx = random.choice(key, self.N, (self.batch_size,), replace=True)
        s = self.s[idx,:]
        y = self.y[idx,:]
        u = self.u[idx,:]
        # Construct batch
        inputs = (u, y)
        outputs = s
        return inputs, outputs
    
# Geneate training data corresponding to one input sample
def generate_one_training_data(key, P, nu_max, sf):
    (t, SIR), (u, y, s) = solve_SIR(key, P, length_scale = 2, nu_max = nu_max, sf=sf)
    u = np.tile(u, (P, 1))
    return u, y, s

# Geneate test data corresponding to one input sample
def generate_one_test_data(key, P, nu_max, sf):
    (t, SIR), (u_test, y, s) = solve_SIR(key, P, length_scale = 2, nu_max = nu_max, sf = sf)

    u_test = np.tile(u_test, (P, 1))
    y_test = t.flatten()[:,None]
    s_test = SIR.T.flatten()

    return u_test, y_test, s_test


def generate_one_test_data_real(key, P, nu_max, sf):
    (t, SIR), (u_test, y, s) = solve_SIR_test(key, P, constants=[0.1, 0.2, 0.3, 0.4, 0.3, 0.2, 0.1])

    u_test = np.tile(u_test, (P, 1))
    y_test = t.flatten()[:,None]
    s_test = SIR.T.flatten()

    return u_test, y_test, s_test

# Geneate training data corresponding to N input sample
def generate_training_data(key, N, P, nu_max, sf):
    config.update("jax_enable_x64", True)
    keys = random.split(key, N)
    # u_train, y_train, s_train= vmap(generate_one_training_data, (0, None))(keys, P)
    u_train_list, y_train_list, s_train_list = [], [], []

    for k in keys:
        u, y, s = generate_one_training_data(k, P, nu_max, sf)
        u_train_list.append(u)
        y_train_list.append(y)
        s_train_list.append(s)

    u_train = np.stack(u_train_list)
    y_train = np.stack(y_train_list)
    s_train = np.stack(s_train_list)

    u_train = np.float32(u_train.reshape(N * P, -1))
    y_train = np.float32(y_train.reshape(N * P, -1))
    s_train = np.float32(s_train.reshape(N * P, -1))

    config.update("jax_enable_x64", False)
    return u_train, y_train, s_train

# Geneate test data corresponding to N input sample
def generate_test_data(key, N, P, nu_max, sf):
    config.update("jax_enable_x64", True)
    keys = random.split(key, N)
    # u_test, y_test, s_test = vmap(generate_one_test_data, (0, None))(keys, P)
    u_test_list, y_test_list, s_test_list = [], [], []

    for k in keys:
        u, y, s = generate_one_test_data(k, P, nu_max, sf)

        u_test_list.append(u)
        y_test_list.append(y)
        s_test_list.append(s)

    u_test = np.stack(u_test_list)
    y_test = np.stack(y_test_list)
    s_test = np.stack(s_test_list)

    u_test = np.float32(u_test.reshape(N * P, -1))
    y_test = np.float32(y_test.reshape(N * P, -1))
    s_test = np.float32(s_test.reshape(-1, N * P)).T

    config.update("jax_enable_x64", False)
    return u_test, y_test, s_test

# Geneate test data corresponding to N input sample
def generate_test_data_real(key, N, P, nu_max, sf):
    config.update("jax_enable_x64", True)
    keys = random.split(key, N)
    # u_test, y_test, s_test = vmap(generate_one_test_data, (0, None))(keys, P)
    u_test_list, y_test_list, s_test_list = [], [], []

    for k in keys:
        u, y, s = generate_one_test_data_real(k, P, nu_max, sf)

        u_test_list.append(u)
        y_test_list.append(y)
        s_test_list.append(s)

    u_test = np.stack(u_test_list)
    y_test = np.stack(y_test_list)
    s_test = np.stack(s_test_list)

    u_test = np.float32(u_test.reshape(N * P, -1))
    y_test = np.float32(y_test.reshape(N * P, -1))
    s_test = np.float32(s_test.reshape(-1, N * P)).T

    config.update("jax_enable_x64", False)
    return u_test, y_test, s_test


#%% parameter setting
nu_max = 0
sampling_field = 'unif'
n_colloquation_pt = 100
experimental_suffix = '_unif_001_1_8k'
key = random.PRNGKey(0)

# GRF length scale
length_scale = 2

# Resolution of the solution
Nt = 100 # u의 사이즈랑 일치
N = 1000 # number of input samples
m = Nt   # number of input sensors
P_train = 500 # number of output sensors
file_path = f'../result/parmas{experimental_suffix}.pkl'

branch_layers = [m, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100*3]
trunk_layers = [1, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100*3]

model = DeepONet(branch_layers, trunk_layers)

#%%
# Generate one test sample
key = random.PRNGKey(1234)
P_test = 100
Nt = 100
u_test, y_test, s_test = generate_test_data(key, 1, P_test, 0.1, 'unif')

with open(f'../result/parmas{experimental_suffix}.pkl', 'rb') as file:
    params = pickle.load(file)
start_time = time.time()
s_pred = model.predict_s(params, u_test, y_test)
end_time = time.time()
sir_pred = np.squeeze(s_pred)
loss = np.mean((s_test - sir_pred)**2)
print("loss : ",loss)

plt.figure(figsize=(10, 8))
plt.subplot(211)
plt.plot(s_test[:,0], label='Susceptible')      
plt.plot(s_test[:,1], label='Infectious')
plt.plot(s_test[:,2], label='Recovered')
plt.plot(sir_pred[:,0] , linestyle=':', label='Predict S')
plt.plot(sir_pred[:,1] , linestyle=':', label='Predict I')
plt.plot(sir_pred[:,2] , linestyle=':', label='Predict R')
plt.title(f"loss : {loss:.6f}")
plt.legend()
plt.subplot(212)
plt.plot(s_test[:,1], label='Predict I')
plt.plot(sir_pred[:,1], linestyle='--', label='Infectious')
plt.legend()
plt.title('Enlarge Infectious')
plt.show()

elapsed_time = end_time - start_time
print(f"DNN time: {elapsed_time} seconds")

#%%
from scipy.integrate import odeint
import numpy as np

S0 = 0.999
I0 = 0.001
R0 = 0
y0 = [S0, I0, R0]
beta = 0.3
gamma = 1/14

# 시간 배열 설정
tmin, tmax = 0, 100 # days
dt = 1
t_span = np.arange(tmin, tmax, dt)

def SIR_model(y, t, beta, gamma, nu_func):
    S, I, R = y
    nu = nu_func(t)  # 시간에 따른 nu 값을 가져옵니다.
    dSdt = -beta * S * I - nu * S
    dIdt = beta * S * I - gamma * I
    dRdt = gamma * I + nu * S
    return [dSdt, dIdt, dRdt]

# customizing nu function
def nu_function(t):
    return u_test[0][int(t)-1]

start_time = time.time()
SIR = odeint(SIR_model, y0, t_span, args=(beta, gamma, nu_function))
end_time = time.time()

elapsed_time = end_time - start_time
print(f"ODE time: {elapsed_time} seconds")
#%% load test
(t, SIR), (u, y, s) = solve_SIR_test(key, n_colloquation_pt, 2, nu_max, sampling_field)
plt.figure(figsize=(10, 8))
plt.subplot(211)
plt.plot(SIR[:,0], label='Susceptible')
plt.plot(SIR[:,1], label='Infectious')
plt.plot(SIR[:,2], label='Recovered')
plt.title(f"sample{experimental_suffix}")
sir = s.T
for i, s_value in enumerate(sir):
    plt.scatter(y, s_value, marker='*', color = 'red')
plt.legend()  # 범례 표시
plt.grid(True)

plt.subplot(212)
plt.plot(u, label='Vaccine Proportion')
plt.hlines(nu_max, 0, 100, color = 'red', linestyles="-.")
plt.text(1, nu_max + 0.03, fr'$\nu_{{\mathrm{{max}}}} = {nu_max}$', fontsize=14)
plt.xlabel('t')
plt.ylim([0,1])
plt.ylabel('nu(t)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
# %%
