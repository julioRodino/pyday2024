"""Script that simulates dynamics in
a bistable visual perception paradigm."""
# %% Import libraries and custom functions
import numpy as np
import matplotlib.pyplot as plt


def eulerIntegrator(x, y, dydt, dt):
    y_next = y + dydt(x, y)*dt
    return y_next


def simulate(y0, tTime, dt):
    nStateVar = len(y0)
    # Allocate time and state variable vectors
    vtime = np.arange(0, tTime+dt, dt)
    Y = np.zeros([len(vtime), nStateVar])

    # Initial State
    Y[0, :] = y0
    y = y0

    # Simulation loop
    for i, t in enumerate(vtime[1:]):
        y = eulerIntegrator(t, y, dydt, dt)
        Y[i+1, :] = y

    return Y, vtime


def sigmoid(x):
    if x <= 0:
        return 0
    else:
        return x**2 / (1+x**2)


def dydt(t, X):

    X1, X2, A1, A2 = X

    dX1dt = (I1 - (1 + A1)*X1 - gamma*sigmoid(X2) +
             np.random.normal(0, sigma1/np.sqrt(dt))) / tau
    dX2dt = (I2 - (1 + A2)*X2 - gamma*sigmoid(X1) +
             np.random.normal(0, sigma2/np.sqrt(dt))) / tau
    dA1dt = (-A1 + alpha*sigmoid(X1)) / tau_A
    dA2dt = (-A2 + alpha*sigmoid(X2)) / tau_A

    return np.array([dX1dt, dX2dt, dA1dt, dA2dt])


def obtainDurations(stateseq):
    diff = np.diff(stateseq)
    durations = np.diff(np.where(np.abs(diff))[0])
    return durations


# %% Parameters and initial conditions

tau = 1
tau_A = 120
alpha = 4
gamma = 2.7
sigma1 = 0.008
sigma2 = 0.008
I1 = 1
I2 = 1

X1 = 0
X2 = 0
A1 = 0
A2 = 0
y0 = np.array([X1, X2, A1, A2])

dt = 1e-3
stateVar, time = simulate(y0, 5000, dt)

start = int(50*dt)
stateseq1 = np.argmax(stateVar[start:, [0, 1]], axis=1)

# Second simulation with more inhibition
gamma = 3.6
stateVar, time = simulate(y0, 5000, dt)

stateseq2 = np.argmax(stateVar[start:, [0, 1]], axis=1)

# %% Plot State Sequence
plt.figure()
plt.plot(time[start:], stateVar[start:, 0], color='green')
plt.plot(time[start:], stateVar[start:, 1], color='red')

plt.twinx()
plt.plot(time[start:], stateseq2, color='black')

plt.xlim([100, 1000])

plt.xlabel('Time [s]')
plt.show()

# %% Precept duration analysis
fig, axs = plt.subplots(2, 1, sharey=True, sharex=True)
# Obtain samples where the precept changes
durations1 = obtainDurations(stateseq1)*dt
# Filter changes due to noise
durations1 = durations1[durations1 > 20]

# Plot Durations histogram
axs[0].hist(durations1, bins=20, color='purple')

axs[0].set_ylabel('Frequencia')

# Obtain samples where the precept changes
durations2 = obtainDurations(stateseq2)*dt

# Filter changes due to noise
durations2 = durations2[durations2 > 20]

# Plot Durations histogram
axs[1].hist(durations2, bins=20, color='purple')

axs[1].set_xlabel('Duración')
axs[1].set_ylabel('Frequencia')

plt.tight_layout()
plt.show()
