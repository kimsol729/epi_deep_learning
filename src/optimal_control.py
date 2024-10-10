# %%
import matplotlib.pyplot as plt
from network import DeepONet
import pickle
import jax.numpy as jnp
from jax import grad
from jax import random
from tqdm import tqdm
#%%
# Load the Operator
experimental_suffix = '_unif_001_1_8k'
with open(f'../result/parmas{experimental_suffix}.pkl', 'rb') as file:
    params = pickle.load(file)
m =100
day = 100
t_span = jnp.array([[i] for i in range(day)], dtype=jnp.float32)
branch_layers = [m, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100*3]
trunk_layers = [1, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100*3]
model = DeepONet(branch_layers, trunk_layers)

def OP(nu):
    nu_stacked = jnp.tile(nu, (100, 1))
    s_pred = model.predict_s(params, nu_stacked, t_span)
    sir_pred = jnp.squeeze(s_pred)
    susceptible = sir_pred[:,0]
    infected = sir_pred[:,1]
    recovered = sir_pred[:,2]
    return nu, susceptible, infected, recovered
#%%
import jax.numpy as jnp
from jax import vmap
from jax.experimental.ode import odeint
import jax

def OP_classic(nu):
    S0 = 0.999
    I0 = 0.001
    R0 = 0
    y0 = jnp.array([S0, I0, R0])
    beta = 0.3
    gamma = 1/14

    # 시간 배열 설정
    tmin, tmax = 0.0, 100.0  # days
    dt = 1.0
    t_span = jnp.arange(tmin, tmax, dt)

    def SIR_model(y, t, beta, gamma, nu_array):
        S, I, R = y
        nu = nu_array[jnp.floor(t).astype(int)]  # 시간에 따른 nu 값을 가져옵니다.
        dSdt = -beta * S * I - nu * S
        dIdt = beta * S * I - gamma * I
        dRdt = gamma * I + nu * S
        return jnp.array([dSdt, dIdt, dRdt])

    # odeint를 사용하여 해결
    SIR = odeint(SIR_model, y0, t_span, beta, gamma, nu)

    return nu, SIR[:,0], SIR[:,1], SIR[:,2]
    
#%%
def loss_J_classic(nu):
    v,S,I,R = OP_classic(nu)
    return jnp.sum(I + v*S)


def loss_J(nu):
    v,S,I,R = OP(nu)
    return jnp.sum(I + v*S)

dI_du = grad(OP)
dS_du = grad(OP)
grad_loss = grad(loss_J) # Autogradient는 loss_J를 연산하는 방법에 역전파를 수행하여 gradient값을 얻는다.

# intial nu 설정
key = random.PRNGKey(0)  # 난수 생성기 키 생성
initial_guess = random.uniform(key, (100,), minval=0.0, maxval=1.0)  # 0과 1 사이의 랜덤한 초기값 설정


#%%
learning_rate = 0.1
num_iterations = 5000
current_strategy = initial_guess.copy()
losses = []  # 손실을 저장할 리스트

for i in tqdm(range(num_iterations)):
    # 경사 하강법을 이용하여 최적의 전략 찾기
    current_strategy -= learning_rate * grad_loss(current_strategy)
    current_strategy = jnp.clip(current_strategy, 0, 1)
    # 현재 전략의 손실 계산
    current_loss = loss_J(current_strategy)
    if i%1000==0:
        print(f"i = {i}, loss = {current_loss}")

    losses.append(current_loss)

optimal_strategy = current_strategy

# 손실 그래프 그리기
plt.plot(losses)
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Loss vs. Iterations')
plt.grid(True)
plt.show()

# 최적의 백신 전략으로 예측 수행
op_v, op_s, op_i, op_r = OP_classic(optimal_strategy)

# 시각화
plt.figure(figsize=(10, 8))
plt.subplot(211)
plt.plot(op_s, label='Susceptible')
plt.plot(op_i, label='Infectious')
plt.plot(op_r, label='Recovered')
plt.title('SIR Dynamics')
plt.legend()

plt.subplot(212)
plt.plot(op_v, label='Optimal vaccine strategy')
plt.title('Optimal Vaccine Strategy')
plt.legend()

plt.show()

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
tmin, tmax = 0.0, 100.0 # days
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
    return op_v[int(t)]

# start_time = time.time()
SIR = odeint(SIR_model, y0, t_span, args=(beta, gamma, nu_function))
plt.subplot(211)
plt.plot(SIR[:,0], label='Susceptible')
plt.plot(SIR[:,1], label='Infectious')
plt.plot(SIR[:,2], label='Recovered')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
#%%
# # 최적화 함수 정의 (예: 경사 하강법)
# def optimize(initial_strategy, iterations=100, learning_rate=0.1):
#     strategy = initial_strategy
#     losses = []  # iteration에 따른 loss 값을 저장할 리스트
#     for i in range(iterations):
#         gradient = grad_loss(strategy)
#         strategy = strategy - learning_rate * gradient
#         losses.append(loss_J(strategy))  # loss 값을 리스트에 추가
#         # print(f"Iteration {i+1}, Loss: {losses[-1]}")
#     return strategy, losses


# t_span = jnp.arange(100).reshape(-1, 1).astype(jnp.float32)

# # # 경사 하강법 수행
# # learning_rate = 0.1
# # num_iterations = 100
# # current_strategy = initial_guess.copy()
# # for i in range(num_iterations):
# #     current_strategy -= learning_rate * grad_loss(current_strategy)

# # optimal_strategy = current_strategy


# # 최적화 실행
# optimal_strategy, losses = optimize(initial_strategy)

# # Loss 그래프 그리기
# plt.plot(range(1, len(losses) + 1), losses)
# plt.xlabel('Iterations')
# plt.ylabel('Loss')
# plt.title('Loss vs. Iterations')
# plt.grid(True)
# plt.show()

# print("Optimal Vaccine Strategy:", optimal_strategy)

# plt.plot(optimal_strategy)
# op_v, op_s, op_i, op_r = OP(optimal_strategy)
# plt.figure(figsize=(10, 8))
# plt.subplot(211)
# plt.plot(op_s, label='Susceptible')
# plt.plot(op_i, label='Infectious')
# plt.plot(op_r, label='Recovered')
# plt.legend()
# plt.subplot(212)
# plt.plot(op_v, label='Optimal vaccine strategy')
# plt.legend()
# plt.title('Enlarge Infectious')
# plt.show()

# # %%

# #%% operator test
# initial_guess = 0.2*jnp.random.rand(100)  # 0과 1 사이의 랜덤한 초기값 설정
# op_v, op_s, op_i, op_r = OP(initial_guess)
# plt.subplot(212)
# plt.plot(op_s , linestyle='-.', label='Predict S')
# plt.plot(op_i, linestyle='-.', label='Predict I')
# plt.plot(op_r, linestyle='-.', label='Predict R')
# plt.subplot(211)
# plt.plot(op_v, label='vaccine')
# plt.legend()
# plt.title('vaccine')
# plt.legend()
# plt.show()
# print(loss_J(initial_guess))