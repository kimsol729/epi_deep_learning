#%%
import numpy as np
from jax import random, jit
from jax.config import config
from functools import partial
from torch.utils import data
from model_sir import solve_SIR, solve_SIR_test
import pickle

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
    
#%%
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


#%% Data generator
key = random.PRNGKey(0)

# GRF length scale
length_scale = 2

# Resolution of the solution
Nt = 100 # u의 사이즈랑 일치..
N = 3000 # number of input samples
m = Nt   # number of input sensors
P_train = 500 # number of output sensors

nu_max = 0.01
sf = 'unif' # sampling field ('unif', 'gauss')

#%% training data set
u_train, y_train, s_train = generate_training_data(key, N, P_train, nu_max, sf)

# training data set 를 파일에 저장
with open('../data/data_u_train_unif_0_01_2.pkl', 'wb') as file:
    pickle.dump(u_train, file)

with open('../data/data_y_train_unif_0_01_2.pkl', 'wb') as file:
    pickle.dump(y_train, file)

with open('../data/data_s_train_unif_0_01_2.pkl', 'wb') as file:
    pickle.dump(s_train, file)

#%%
