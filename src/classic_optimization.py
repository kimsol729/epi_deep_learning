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
        k1_x1, k1_x2 = SIR_dynamics(x1[i], x2[i], u[i])
        k2_x1, k2_x2 = SIR_dynamics(x1[i] + 0.5 * dt * k1_x1, x2[i] + 0.5 * dt * k1_x2, u[i])
        k3_x1, k3_x2 = SIR_dynamics(x1[i] + 0.5 * dt * k2_x1, x2[i] + 0.5 * dt * k2_x2, u[i])
        k4_x1, k4_x2 = SIR_dynamics(x1[i] + dt * k3_x1, x2[i] + dt * k3_x2, u[i])
        
        x1[i+1] = x1[i] + (dt / 6) * (k1_x1 + 2 * k2_x1 + 2 * k3_x1 + k4_x1)
        x2[i+1] = x2[i] + (dt / 6) * (k1_x2 + 2 * k2_x2 + 2 * k3_x2 + k4_x2)
    return x1, x2

# Backward integration of adjoint equations
def backward_integration(x1, x2, lambda1, lambda2, u, dt, N):
    for i in range(N-1, -1, -1):
        k1_lambda1 = - (u[i] + lambda1[i+1] * (-beta * x2[i] - u[i]) + lambda2[i+1] * beta * x2[i])
        k1_lambda2 = - (1 + lambda1[i+1] * (-beta * x1[i]) + lambda2[i+1] * (beta * x1[i] - gamma))
        
        k2_lambda1 = - (u[i] + (lambda1[i+1] - 0.5 * dt * k1_lambda1) * (-beta * x2[i] - u[i]) + (lambda2[i+1] - 0.5 * dt * k1_lambda2) * beta * x2[i])
        k2_lambda2 = - (1 + (lambda1[i+1] - 0.5 * dt * k1_lambda1) * (-beta * x1[i]) + (lambda2[i+1] - 0.5 * dt * k1_lambda2) * (beta * x1[i] - gamma))
        
        k3_lambda1 = - (u[i] + (lambda1[i+1] - 0.5 * dt * k2_lambda1) * (-beta * x2[i] - u[i]) + (lambda2[i+1] - 0.5 * dt * k2_lambda2) * beta * x2[i])
        k3_lambda2 = - (1 + (lambda1[i+1] - 0.5 * dt * k2_lambda1) * (-beta * x1[i]) + (lambda2[i+1] - 0.5 * dt * k2_lambda2) * (beta * x1[i] - gamma))
        
        k4_lambda1 = - (u[i] + (lambda1[i+1] - dt * k3_lambda1) * (-beta * x2[i] - u[i]) + (lambda2[i+1] - dt * k3_lambda2) * beta * x2[i])
        k4_lambda2 = - (1 + (lambda1[i+1] - dt * k3_lambda1) * (-beta * x1[i]) + (lambda2[i+1] - dt * k3_lambda2) * (beta * x1[i] - gamma))
        
        lambda1[i] = lambda1[i+1] - (dt / 6) * (k1_lambda1 + 2 * k2_lambda1 + 2 * k3_lambda1 + k4_lambda1)
        lambda2[i] = lambda2[i+1] - (dt / 6) * (k1_lambda2 + 2 * k2_lambda2 + 2 * k3_lambda2 + k4_lambda2)

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
    # u = u0 * np.ones(N+1)
    u = u0
    lambda1 = np.zeros(N+1)
    lambda2 = np.zeros(N+1)

    J_values = []  # Objective function values
    H_values = []  # Hamiltonian values

    for iteration in range(max_iter):
        # Forward integration of state equations
        x1, x2 = forward_integration(x1_0, x2_0, u, dt, N)
        if iteration == 0:
            t = np.linspace(0, T, N+1)
            # plt.figure(figsize=(12,12))
            fig, axs = plt.subplots(2, 1, figsize=(10, 10))
            ax1 = axs[0]
            ax1.set_xlabel('Time')
            ax1.set_ylabel('Infectious', color='tab:red')
            ax1.plot(t, x2, color='tab:red', label='Infectious')
            ax1.tick_params(axis='y', labelcolor='tab:red')

            ax2 = ax1.twinx()
            ax2.set_ylabel('Susceptible', color='tab:blue')
            ax2.plot(t, x1, color='tab:blue', label='Susceptible')
            ax2.tick_params(axis='y', labelcolor='tab:blue')

            ax1.grid(True)
            ax1.set_title('Initial States')

            axs[1].plot(t, u, label='Control u(t)')
            axs[1].grid(True)
            axs[1].set_title(f'Initial Control u0(t)')
            axs[1].set_xlabel('Time t')
            axs[1].set_ylim([-0.01,1.01])
            axs[1].legend()
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
u = 0 * np.ones(1001)
# u[:20] = np.ones(20)
# Initial conditions
x1_0 = 0.999
x2_0 = 0.001
# u0 = 0.1
T = 100
N = 1000

max_iter = 25000
learning_rate = 0.01

# Solve optimal control problem
x1, x2, u, lambda1, lambda2, J_values, H_values = forward_backward(x1_0, x2_0, u, T, N, max_iter, learning_rate)

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

# Results output
plt.figure(figsize=(12,6))

plt.subplot(121)
plt.plot(np.arange(max_iter)[5000:], J_values[5000:])
plt.title('Objective function J after 5000it')
plt.xlabel('Iteration')
plt.ylabel('J')
plt.grid(True)

plt.subplot(122)
plt.plot(np.arange(max_iter)[5000:], H_values[5000:])
plt.title('Hamiltonian H after 5000it')
plt.xlabel('Iteration')
plt.ylabel('H')
plt.grid(True)

plt.tight_layout()
plt.show()



t = np.linspace(0, T, N+1)
fig, axs = plt.subplots(3, 1, figsize=(10, 10))
ax1 = axs[0]
ax1.set_xlabel('Time')
ax1.set_ylabel('Infectious', color='tab:red')
ax1.plot(t, x2, color='tab:red', label='Infectious')
ax1.tick_params(axis='y', labelcolor='tab:red')

ax2 = ax1.twinx()
ax2.set_ylabel('Susceptible', color='tab:blue')
ax2.plot(t, x1, color='tab:blue', label='Susceptible')
ax2.tick_params(axis='y', labelcolor='tab:blue')

ax1.grid(True)
ax1.set_title('Optimal States')

axs[1].plot(t, u, label='Control u(t)')
axs[1].grid(True)
axs[1].set_title(f'Optimal Control u0(t) (iter : {max_iter}, lr : {learning_rate})')
axs[1].set_xlabel('Time t')
axs[1].set_ylim([-0.01,1.01])
axs[1].legend()

axs[2].plot(t, lambda1, label='Costate λ1(t)')
axs[2].plot(t, lambda2, label='Costate λ2(t)')
axs[2].set_xlabel('Time t')
axs[2].legend()
axs[2].grid(True)
axs[2].set_title('Multiplier lambda')

plt.tight_layout()
plt.show()
print("1day ",u[:10])
print("2day ",u[10:20])
print("3day ",u[20:30])
print("4day ",u[30:40])
#%%
u = 0 * np.ones(1001)
u[:20] = np.ones(20)
# Initial conditions
x1_0 = 0.999
x2_0 = 0.001
# u0 = 0.1
T = 100
N = 1000

max_iter = 25000
learning_rate = 0.01

# Solve optimal control problem
x1, x2, u, lambda1, lambda2, J_values, H_values = forward_backward(x1_0, x2_0, u, T, N, max_iter, learning_rate)

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

# Results output
plt.figure(figsize=(12,6))

plt.subplot(121)
plt.plot(np.arange(max_iter)[5000:], J_values[5000:])
plt.title('Objective function J after 5000it')
plt.xlabel('Iteration')
plt.ylabel('J')
plt.grid(True)

plt.subplot(122)
plt.plot(np.arange(max_iter)[5000:], H_values[5000:])
plt.title('Hamiltonian H after 5000it')
plt.xlabel('Iteration')
plt.ylabel('H')
plt.grid(True)

plt.tight_layout()
plt.show()



t = np.linspace(0, T, N+1)
fig, axs = plt.subplots(3, 1, figsize=(10, 10))
ax1 = axs[0]
ax1.set_xlabel('Time')
ax1.set_ylabel('Infectious', color='tab:red')
ax1.plot(t, x2, color='tab:red', label='Infectious')
ax1.tick_params(axis='y', labelcolor='tab:red')

ax2 = ax1.twinx()
ax2.set_ylabel('Susceptible', color='tab:blue')
ax2.plot(t, x1, color='tab:blue', label='Susceptible')
ax2.tick_params(axis='y', labelcolor='tab:blue')

ax1.grid(True)
ax1.set_title('Optimal States')

axs[1].plot(t, u, label='Control u(t)')
axs[1].grid(True)
axs[1].set_title(f'Optimal Control u0(t) (iter : {max_iter}, lr : {learning_rate})')
axs[1].set_xlabel('Time t')
axs[1].set_ylim([-0.01,1.01])
axs[1].legend()

axs[2].plot(t, lambda1, label='Costate λ1(t)')
axs[2].plot(t, lambda2, label='Costate λ2(t)')
axs[2].set_xlabel('Time t')
axs[2].legend()
axs[2].grid(True)
axs[2].set_title('Multiplier lambda')

plt.tight_layout()
plt.show()
print("1day ",u[:10])
print("2day ",u[10:20])
print("3day ",u[20:30])
print("4day ",u[30:40])
#%%
u = 0.2 * np.ones(1001)
# u[:20] = np.ones(20)
# Initial conditions
x1_0 = 0.999
x2_0 = 0.001
# u0 = 0.1
T = 100
N = 1000

max_iter = 60000
learning_rate = 0.001

# Solve optimal control problem
x1, x2, u, lambda1, lambda2, J_values, H_values = forward_backward(x1_0, x2_0, u, T, N, max_iter, learning_rate)

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

# Results output
plt.figure(figsize=(12,6))

plt.subplot(121)
plt.plot(np.arange(max_iter)[5000:], J_values[5000:])
plt.title('Objective function J after 5000it')
plt.xlabel('Iteration')
plt.ylabel('J')
plt.grid(True)

plt.subplot(122)
plt.plot(np.arange(max_iter)[5000:], H_values[5000:])
plt.title('Hamiltonian H after 5000it')
plt.xlabel('Iteration')
plt.ylabel('H')
plt.grid(True)

plt.tight_layout()
plt.show()



t = np.linspace(0, T, N+1)
fig, axs = plt.subplots(3, 1, figsize=(10, 10))
ax1 = axs[0]
ax1.set_xlabel('Time')
ax1.set_ylabel('Infectious', color='tab:red')
ax1.plot(t, x2, color='tab:red', label='Infectious')
ax1.tick_params(axis='y', labelcolor='tab:red')

ax2 = ax1.twinx()
ax2.set_ylabel('Susceptible', color='tab:blue')
ax2.plot(t, x1, color='tab:blue', label='Susceptible')
ax2.tick_params(axis='y', labelcolor='tab:blue')

ax1.grid(True)
ax1.set_title('Optimal States')

axs[1].plot(t, u, label='Control u(t)')
axs[1].grid(True)
axs[1].set_title(f'Optimal Control u0(t) (iter : {max_iter}, lr : {learning_rate})')
axs[1].set_xlabel('Time t')
axs[1].set_ylim([-0.01,1.01])
axs[1].legend()

axs[2].plot(t, lambda1, label='Costate λ1(t)')
axs[2].plot(t, lambda2, label='Costate λ2(t)')
axs[2].set_xlabel('Time t')
axs[2].legend()
axs[2].grid(True)
axs[2].set_title('Multiplier lambda')

plt.tight_layout()
plt.show()
print("1day ",u[:10])
print("2day ",u[10:20])
print("3day ",u[20:30])
print("4day ",u[30:40])
#%%
u = 0.01 * np.ones(1001)
# u[:20] = np.ones(20)
# Initial conditions
x1_0 = 0.999
x2_0 = 0.001
# u0 = 0.1
T = 100
N = 1000

max_iter = 25000
learning_rate = 0.01

# Solve optimal control problem
x1, x2, u, lambda1, lambda2, J_values, H_values = forward_backward(x1_0, x2_0, u, T, N, max_iter, learning_rate)

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

# Results output
plt.figure(figsize=(12,6))

plt.subplot(121)
plt.plot(np.arange(max_iter)[5000:], J_values[5000:])
plt.title('Objective function J after 5000it')
plt.xlabel('Iteration')
plt.ylabel('J')
plt.grid(True)

plt.subplot(122)
plt.plot(np.arange(max_iter)[5000:], H_values[5000:])
plt.title('Hamiltonian H after 5000it')
plt.xlabel('Iteration')
plt.ylabel('H')
plt.grid(True)

plt.tight_layout()
plt.show()



t = np.linspace(0, T, N+1)
fig, axs = plt.subplots(3, 1, figsize=(10, 10))
ax1 = axs[0]
ax1.set_xlabel('Time')
ax1.set_ylabel('Infectious', color='tab:red')
ax1.plot(t, x2, color='tab:red', label='Infectious')
ax1.tick_params(axis='y', labelcolor='tab:red')

ax2 = ax1.twinx()
ax2.set_ylabel('Susceptible', color='tab:blue')
ax2.plot(t, x1, color='tab:blue', label='Susceptible')
ax2.tick_params(axis='y', labelcolor='tab:blue')

ax1.grid(True)
ax1.set_title('Optimal States')

axs[1].plot(t, u, label='Control u(t)')
axs[1].grid(True)
axs[1].set_title(f'Optimal Control u0(t) (iter : {max_iter}, lr : {learning_rate})')
axs[1].set_xlabel('Time t')
axs[1].set_ylim([-0.01,1.01])
axs[1].legend()

axs[2].plot(t, lambda1, label='Costate λ1(t)')
axs[2].plot(t, lambda2, label='Costate λ2(t)')
axs[2].set_xlabel('Time t')
axs[2].legend()
axs[2].grid(True)
axs[2].set_title('Multiplier lambda')

plt.tight_layout()
plt.show()
print("1day ",u[:10])
print("2day ",u[10:20])
print("3day ",u[20:30])
print("4day ",u[30:40])

# %%
