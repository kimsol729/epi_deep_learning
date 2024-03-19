#%%
from jax import random
from jax import device_put
from model_sir import solve_SIR_test, solve_SIR
from network import DeepONet
from data_generator import DataGenerator, generate_test_data
import matplotlib.pyplot as plt
import pickle
import numpy as np

#%% parameter setting
nu_max = 0.6
sampling_field = 'unif'
n_colloquation_pt = 100
experimental_suffix = '_unif_001_1_8k'
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

#%% load test
(t, SIR), (u, y, s) = solve_SIR(key, n_colloquation_pt, 2, nu_max, sampling_field)
plt.figure(figsize=(10, 8))
plt.subplot(211)
plt.plot(SIR[:,0], label='Susceptible')
plt.plot(SIR[:,1], label='Infectious')
plt.plot(SIR[:,2], label='Recovered')
plt.title(f"sample{experimental_suffix}")
sir = s.T
for i, s_value in enumerate(sir):
    plt.scatter(y, s_value, marker='*', color = 'red')
plt.legend()
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

#%%
#%% load test
(t, SIR), (u, y, s) = solve_SIR_test(key, n_colloquation_pt, [0.2,0.2,0.2,0.2,0.2,0.2,0.2])
plt.figure(figsize=(10, 8))
plt.subplot(211)
plt.plot(SIR[:,0], label='Susceptible')
plt.plot(SIR[:,1], label='Infectious')
plt.plot(SIR[:,2], label='Recovered')
plt.title(f"sample{experimental_suffix}")
sir = s.T
for i, s_value in enumerate(sir):
    plt.scatter(y, s_value, marker='*', color = 'red')
plt.legend()
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
# %% load data
with open(f'../data/u_mix{experimental_suffix}.pkl', 'rb') as file:
    u_train = device_put(pickle.load(file))

with open(f'../data/y_mix{experimental_suffix}.pkl', 'rb') as file:
    y_train = device_put(pickle.load(file))

with open(f'../data/s_mix{experimental_suffix}.pkl', 'rb') as file:
    s_train = device_put(pickle.load(file))
# 파일 저장
batch_size = 12345
dataset = DataGenerator(u_train, y_train, s_train, batch_size)

# model training
model.train(dataset, nIter=300000)
params = model.get_params(model.opt_state)
with open(file_path, 'wb') as file:
    pickle.dump(params, file)

# Plot for loss function
plt.figure(figsize = (6,5))
plt.plot(model.loss_log, lw=2)

plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.yscale('log')
plt.legend()
plt.tight_layout()
plt.show()
plt.savefig(f'loss_plot{experimental_suffix}.pdf', format='pdf')

# %%
