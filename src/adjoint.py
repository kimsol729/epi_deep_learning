#%%
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.integrate import simps
from scipy.integrate import solve_bvp
from scipy.integrate import odeint
from scipy.interpolate import interp1d
from tqdm import tqdm

#%%
# SIR 모델의 미분방정식 정의
def sir_vaccine_model(y, t, beta, gamma, nu_values):
    S, I, R = y
    nu_t = np.interp(t, np.linspace(0, 100, len(nu_values)), nu_values)  # 시간에 따라 nu 값을 보간
    dSdt = -beta * S * I - nu_t * S
    dIdt = beta * S * I - gamma * I
    dRdt = gamma * I + nu_t * S
    return [dSdt, dIdt, dRdt]

# 초기 조건 설정
S0 = 0.999  # 초기 감염 가능한 사람 비율
I0 = 0.001  # 초기 감염된 사람 비율
R0 = 0.00  # 초기 회복된 사람 비율
y0 = [S0, I0, R0]

# 파라미터 설정
beta = 0.3  # 전염율
gamma = 1/14  # 회복

# 시간 설정
t = np.linspace(0, 100, 1000)
t_int = np.linspace(0, 100, 101)  # 101개의 점 생성 (0부터 100까지)

# 백신 함수 정의 및 값 생성
nu_values = [0 * random.uniform(0, 1) for _ in t_int]
losses = []
# nu_values = [1 if t<2 else 0 for t in t_int]
# nu_values = [1 if t < 2 else np.max([(t-2)/(2-N) + 1,0]) for t in t_int]

#%%
for simulation in tqdm(range(100000)):
    # ODE 풀기
    solution = odeint(sir_vaccine_model, y0, t, args=(beta, gamma, nu_values))
    S, I, R = solution.T

    if simulation % 10000 ==0:
        # 결과 시각화
        plt.figure(figsize=(10, 6))

        plt.subplot(221)
        plt.plot(t, S, label='Susceptible')
        plt.plot(t, I, label='Infectious')
        plt.plot(t, R, label='Recovered')
        plt.xlabel('Time')
        plt.ylabel('Fraction of population')
        plt.title('SIR Model with Vaccination')
        plt.legend()

        plt.subplot(222)
        plt.plot(t_int, nu_values, label='nu(t)', color='orange')
        plt.xlabel('Time')
        plt.ylim([-0.05,1+0.05])
        plt.ylabel('nu(t) value')
        plt.title('Random Vaccination Rate nu(t)')
        plt.legend()

        plt.subplot(223)
        nu_t = np.interp(t, np.linspace(0, 100, len(nu_values)), nu_values)
        S_nu = S * nu_t
        plt.plot(t, I, label='Infectious')
        plt.plot(t, S_nu, label='S(t) * nu(t)', color='green')
        plt.xlabel('Time')
        plt.ylabel('Fraction of population')
        # plt.title('SIR Model with Vaccination')
        plt.legend()

        plt.subplot(224)
        plt.plot(t, S_nu + I, label=f"LOSS J = {simps(I + S_nu, t):.7f}", color='black')
        plt.xlabel('Time')
        plt.ylabel('Fraction of population')
        # plt.title((f"LOSS J = {simps(I + S_nu, t):.7f}"))
        plt.legend()
        plt.tight_layout()
        plt.show()
        plt.savefig(f'../result/more_z_classic_optim_{simulation+80000}.png')

    loss = simps(I + S_nu, t)
    losses.append(loss)

    # Find lambda1, lambda2
    S_fnt = interp1d(t, S, kind='cubic')
    I_fnt = interp1d(t, I, kind='cubic')
    nu_fnt = interp1d(t, nu_t, kind='cubic')

    # Define the system of ODEs
    def system(t, y, A, B, C, D, nu):
        l1, l2 = y
        l1_prime = A(t) * l1 + B(t) * l2 - nu(t)
        l2_prime = C(t) * l1 + D(t) * l2 - 1
        return [l1_prime, l2_prime]

    # Example functions for A(t), B(t), C(t), D(t), and nu(t)
    def A(t):
        return beta * I_fnt(t) + nu_fnt(t)

    def B(t):
        return - beta * I_fnt(t)

    def C(t):
        return beta * S_fnt(t)

    def D(t):
        return - beta * S_fnt(t) + gamma

    def nu(t):
        return nu_fnt(t)

    # Boundary conditions
    def boundary_conditions(ya, yb):
        l1_end = 0.0  # Terminal condition for l1 at t = 10
        l2_end = 0.0  # Terminal condition for l2 at t = 10
        return [yb[0] - l1_end, yb[1] - l2_end]

    # Initial guess for the solution
    t_guess = np.linspace(0, 100, 100)
    y_guess = np.zeros((2, t_guess.size))

    # Solve the boundary value problem
    sol = solve_bvp(lambda t, y: system(t, y, A, B, C, D, nu), 
                    boundary_conditions, t_guess, y_guess)

    # Plot the results
    plt.plot(sol.x, sol.y[0], label='lambda_1(t)')
    plt.plot(sol.x, sol.y[1], label='lambda_2(t)')
    plt.xlabel('Time t')
    plt.ylabel('Functions l1(t) and l2(t)')
    plt.legend()
    plt.title('Solution of the Boundary Value Problem')
    plt.grid()
    plt.show()

    lam1 = interp1d(sol.x, sol.y[0], kind='cubic')(t_int)
    lam2 = interp1d(sol.x, sol.y[1], kind='cubic')(t_int)
    S = interp1d(t, S, kind='cubic')(t_int)
    I = interp1d(t, I, kind='cubic')(t_int)
    nu_old = interp1d(t, nu_t, kind='cubic')(t_int)

    # dH/dnu 계산

    dH = S * ( 1 - lam1 )
    alpha = search()
    nu_values = np.clip(nu_values-alpha * dH, 0, 1)

# %%
