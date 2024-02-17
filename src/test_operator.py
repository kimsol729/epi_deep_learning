#%%
from jax import random
from jax import device_put
from model_sir import solve_SIR
from network import DeepONet
from data_generator import DataGenerator, generate_test_data
import matplotlib.pyplot as plt
import pickle
import numpy as np

#%% parameter setting
nu_max = 0.6
sampling_field = 'unif'
n_colloquation_pt = 100
experimental_suffix = '_unif_001_1'
key = random.PRNGKey(0)

# GRF length scale
length_scale = 2

# Resolution of the solution
Nt = 100 # u의 사이즈랑 일치..
N = 1000 # number of input samples
m = Nt   # number of input sensors
P_train = 500 # number of output sensors
file_path = f'../result/parmas{experimental_suffix}.pkl'

branch_layers = [m, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100*3]
trunk_layers = [1, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100*3]
# branch_layers = [m, 32, 64, 128, 128, 128, 128, 128*3]
# trunk_layers =  [1, 32, 64, 128, 128, 128, 128, 128*3]
model = DeepONet(branch_layers, trunk_layers)

#%%
# Generate one test sample
key = random.PRNGKey(1234)
P_test = 100
Nt = 100
u_test, y_test, s_test = generate_test_data(key, 1, P_test, 0.4, 'unif')

with open(f'../result/parmas{experimental_suffix}.pkl', 'rb') as file:
    params = pickle.load(file)
s_pred = model.predict_s(params, u_test, y_test)
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
plt.plot(sir_pred[:,1], label='Predict I')
plt.plot(s_test[:,1], linestyle='--', label='Infectious')
plt.legend()
plt.title('Enlarge Infectious')
plt.show()
#%%
plt.plot(u_test[0])