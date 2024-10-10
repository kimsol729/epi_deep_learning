import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.integrate import simps
from scipy.integrate import solve_bvp
from scipy.integrate import odeint
from scipy.interpolate import interp1d


#%%
# SIR 모델의 미분방정식 정의
bound_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
# bound_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
for bound in bound_list:
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
    gamma = 1/14  # 회복율

    # 시간 설정
    t = np.linspace(0, 100, 1000)
    t_int = np.linspace(0, 100, 101)  # 101개의 점 생성 (0부터 100까지)

    # 백신 함수 정의 및 값 생성
    # nu_values = [0.5 * random.uniform(0, 1) for _ in t_int]
    nu_values  = [bound for _ in t_int]
    # nu_values = [1 if t < bound else 0 for t in t_int]

    # ODE 풀기
    solution = odeint(sir_vaccine_model, y0, t, args=(beta, gamma, nu_values))
    S, I, R = solution.T


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
    plt.ylim([-0.05, 1.05])
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
    plt.plot(t, S_nu + I, label=f"LOSS J = {simps(I + S_nu, t):.5f}", color='black')
    plt.xlabel('Time')
    plt.ylabel('Fraction of population')

    # plt.title((f"LOSS J = {simps(I + S_nu, t):.5f}"))
    plt.legend()
    plt.savefig(f'../result/manual_const_{bound}.png')
    plt.tight_layout()
    plt.show()
    

# %%
