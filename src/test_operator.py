#%%
from jax import random
from jax import device_put
from model_sir import solve_SIR, solve_SIR_test
from network import DeepONet
from data_generator import DataGenerator, generate_test_data
import matplotlib.pyplot as plt
import pickle
import numpy as np
import time

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
u_test, y_test, s_test = generate_test_data(key, 1, P_test, 0, 'unif')

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
