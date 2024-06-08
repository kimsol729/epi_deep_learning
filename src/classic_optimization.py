# %%
import numpy as np
import matplotlib.pyplot as plt

# System SIR_dynamics
beta = 0.3
gamma = 1/14

def SIR_dynamics(x1, x2, u):
    dx1dt = -beta * x1 * x2 - u * x1
    dx2dt = beta * x1 * x2 - gamma * x2
    return dx1dt, dx2dt

# Forward integration of state equations
def forward_integration(x1_0, x2_0, u, dt, N):
    x1 = np.zeros(N+1)
    x2 = np.zeros(N+1)
    x1[0], x2[0] = x1_0, x2_0
    for i in range(N):
        dx1dt, dx2dt = SIR_dynamics(x1[i], x2[i], u[i])
        x1[i+1] = x1[i] + dt * dx1dt
        x2[i+1] = x2[i] + dt * dx2dt
    return x1, x2

# Backward integration of adjoint equations
def backward_integration(x1, x2, lambda1, lambda2, u, dt, N):
    for i in range(N-1, -1, -1):
        dlambda1dt = - (u[i] + lambda1[i+1] * (-beta * x2[i] - u[i]) + lambda2[i+1] * beta * x2[i])
        dlambda2dt = - (1 + lambda1[i+1] * (-beta * x1[i]) + lambda2[i+1] * (beta * x1[i] - gamma))
        lambda1[i] = lambda1[i+1] - dt * dlambda1dt
        lambda2[i] = lambda2[i+1] - dt * dlambda2dt
    return lambda1, lambda2

# Update control using gradient descent
def update_control(x1, lambda1, u, learning_rate):
    N = len(x1) - 1
    for i in range(N+1):
        u_grad = x1[i] - lambda1[i] * x1[i]
        u[i] -= learning_rate * u_grad
        u[i] = max(0, min(1, u[i]))
    return u

# Forward-Backward Method
def forward_backward(x1_0, x2_0, u0, T, N, max_iter, learning_rate):
    dt = T / N
    u = u0 * np.ones(N+1)
    lambda1 = np.zeros(N+1)
    lambda2 = np.zeros(N+1)

    J_values = []  # Objective function values
    H_values = []  # Hamiltonian values

    for iteration in range(max_iter):
        # Forward integration of state equations
        x1, x2 = forward_integration(x1_0, x2_0, u, dt, N)
        if iteration == 0:
            t = np.linspace(0, T, N+1)
            plt.figure(figsize=(12,12))
            plt.subplot(311)
            plt.plot(t, x1, label='State x1(t)')
            plt.plot(t, x2, label='State x2(t)')
            plt.grid(True)
            plt.title('States')
            plt.subplot(312)
            plt.plot(t, u, label='Control u(t)')
            plt.grid(True)
            plt.show()

        # Calculate objective value
        J = np.trapz(x2 + u * x1, dx=dt)
        J_values.append(J)

        # Calculate Hamiltonian value
        H = x2 + u * x1 + lambda1 * (-beta * x1 * x2 - u * x1) + lambda2 * (beta * x1 * x2 - gamma * x2)
        H_values.append(H[-1])

        # Backward integration of costate equations
        lambda1, lambda2 = backward_integration(x1, x2, lambda1, lambda2, u, dt, N)

        # Update control using gradient descent
        u = update_control(x1, lambda1, u, learning_rate)

    return x1, x2, u, lambda1, lambda2, J_values, H_values

#%%
# Initial conditions
x1_0 = 0.999
x2_0 = 0.001
u0 = 0
T = 100
N = 1000
max_iter = 10000
learning_rate = 0.001

# Solve optimal control problem
x1, x2, u, lambda1, lambda2, J_values, H_values = forward_backward(x1_0, x2_0, u0, T, N, max_iter, learning_rate)

# Results output
plt.figure(figsize=(12,6))

plt.subplot(121)
plt.plot(np.arange(max_iter), J_values)
plt.title('Objective function J')
plt.xlabel('Iteration')
plt.ylabel('J')
plt.grid(True)

plt.subplot(122)
plt.plot(np.arange(max_iter), H_values)
plt.title('Hamiltonian H')
plt.xlabel('Iteration')
plt.ylabel('H')
plt.grid(True)

plt.tight_layout()
plt.show()

print(f"Objective value J: {J_values[-1]}")
print(f"Hamiltonian H: {H_values[-1]}")
#%%
# # 결 결과 출력
t = np.linspace(0, T, N+1)
plt.figure(figsize=(12,12))
plt.subplot(311)
# plt.plot(t, x1, label='State x1(t)')
plt.plot(t, x2, label='State x2(t)')
plt.grid(True)
plt.title('States')
plt.subplot(312)
plt.plot(t, u, label='Control u(t)')
plt.grid(True)
plt.title('Optimal Control u(t)')
plt.subplot(313)
plt.plot(t, lambda1, label='Costate λ1(t)')
plt.plot(t, lambda2, label='Costate λ2(t)')
plt.xlabel('Time t')
plt.legend()
plt.grid(True)
plt.title('Multiplier lambda')
plt.show()
# %%
