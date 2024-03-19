#%%
import numpy as np
from scipy.integrate import odeint
from scipy.interpolate import interp1d
from jax import random

#%%
def solve_SIR(key, P, length_scale, nu_max = 1, sf = 'unif'):
    """Solve SIR
    St/dt = -beta*S*I - nu*S
    It/dt = beta*S*I - gamma*I
    Rt/dt = gamma*I + nu*S

    P = location(y) 의 개수
    """
    # 초기 조건 설정
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
    
    # SIR 모델을 풀기 위한 함수
    def SIR_model(y, t, beta, gamma, nu_func):
        S, I, R = y
        nu = nu_func(t)  # 시간에 따른 nu 값을 가져옵니다.
        dSdt = -beta * S * I - nu * S
        dIdt = beta * S * I - gamma * I
        dRdt = gamma * I + nu * S
        return [dSdt, dIdt, dRdt]

    
    # Generate subkeys
    subkeys = random.split(key, 1)

    # Generate a vaccine strategy sample
    def gen_GRF(size, correlation_length, amplitude=1):
        x = np.arange(size)
        covariance_matrix = amplitude * np.exp(-0.5 * ((x[:, None] - x) / correlation_length)**2)
        grf_sample = random.multivariate_normal(subkeys[0], np.zeros(size), covariance_matrix)

        # Normalize to [0, 1]
        grf_sample = nu_max *(grf_sample - grf_sample.min()) / (grf_sample.max() - grf_sample.min())
        return grf_sample
    
    def gen_UFF(size,correlation_length):
        import numpy as np
        uni_sample = np.random.rand(size)
        # Normalize to [0, 1]---------------------------------------------------------------------------------------
        uni_sample = nu_max *(uni_sample - uni_sample.min()) / (uni_sample.max() - uni_sample.min())
        return uni_sample

    # Usage example:
    sensor = tmax  # Size of the GRF
    if sf == 'unif':
        nu = gen_UFF(sensor, length_scale)
    else:
        nu = gen_GRF(sensor, length_scale)

    def nu_function(t, nu):
        t_min = 0
        t_max = len(nu) - 1
        t_scaled = (t - t_min) / (t_max - t_min)  # Scale t to the range [0, 1]
        nu_interp = interp1d(np.linspace(0, 1, len(nu)), nu, kind='linear', fill_value='extrapolate')
        return nu_interp(t_scaled)

    f_fn = lambda t: nu_function(t, nu)
    # 미분방정식 풀이
    SIR = odeint(SIR_model, y0, t_span, args=(beta, gamma, f_fn))
  
    u = f_fn(t_span)
    idx = random.randint(subkeys[1], (1, P), 0, len(t_span))
    y = t_span[idx]
    s = SIR[idx, :]

    return (t_span, SIR), (u, y, s)

# %%
def solve_SIR_test(key, P, constants):
    """Solve SIR
    St/dt = -beta*S*I - nu*S
    It/dt = beta*S*I - gamma*I
    Rt/dt = gamma*I + nu*S

    P = location(y) 의 개수
    """
    # 초기 조건 설정
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
    
    # SIR 모델을 풀기 위한 함수
    def SIR_model(y, t, beta, gamma, nu_func):
        S, I, R = y
        nu = nu_func(t)  # 시간에 따른 nu 값을 가져옵니다.
        dSdt = -beta * S * I - nu * S
        dIdt = beta * S * I - gamma * I
        dRdt = gamma * I + nu * S
        return [dSdt, dIdt, dRdt]

    
    # Generate subkeys
    subkeys = random.split(key, 1)

    # customizing nu function
    def nu_function(t, constants):
        if t < 2:
            return 0
        else:
            index = int((t - 2) // 14)  # 해당 인덱스 계산
            return constants[index]

    f_fn = lambda t: nu_function(t,constants)

    # 미분방정식 풀이
    SIR = odeint(SIR_model, y0, t_span, args=(beta, gamma, f_fn))
  
    u = f_fn(t_span)
    idx = random.randint(subkeys[1], (1, P), 0, len(t_span))
    y = t_span[idx]
    s = SIR[idx, :]

    return (t_span, SIR), (u, y, s)

# %%
