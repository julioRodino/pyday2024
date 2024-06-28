# %%
import numpy as np
import matplotlib.pyplot as plt

# %% Find values for cuadratic function


def cuadratic(x, a, b, c):
    return a*(x**2) + b*x + c


x = np.arange(0, 1+1e-5, 0.1)

a = 1
b = 1
c = 0

y = cuadratic(x, a, b, c)

plt.plot(x, y)

# %% Find the solution using the diff. eq.


def dydt(x, y):
    return 2*a*x + b


def eulerIntegrator(x, y, dydt, dt):
    y_next = y + dydt(x, y)*dt
    return y_next


def simulate(y0, tX, dt):
    # Allocate time and state variable vectors
    vX = np.arange(0, tX+dt, dt)
    Y = np.zeros_like(vX)

    # Initial State
    Y[0] = y0

    # Simulation loop
    for i, t in enumerate(vX[1:]):
        Y[i+1] = eulerIntegrator(t, Y[i], dydt, dt)

    return Y, vX


# %% Simulate
a = 1
b = 1
y0 = 0
dt = 0.1
stateVar, time = simulate(y0, 1, dt)

plt.scatter(time, stateVar, label='Simulation')
plt.scatter(x, y, label='Solution')
plt.plot(time, stateVar)
plt.plot(x, y)

plt.legend()
