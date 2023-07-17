import numpy as np
import matplotlib.pyplot as plt
from BO_ODE import ODE
import warnings
import math
from skopt import gp_minimize
from skopt import Optimizer
from skopt import plots
from skopt.plots import plot_convergence

warnings.simplefilter("ignore")

filename = input('Insert name of the file:')

# Setting the enviroment
turbine = {
        'rho'       : 1.225, #[kg/m^3]
        'R'         : 63,  #[m]
        'J'         : 38677040.613, #[kg*m^2]
        'w_rated'   : 1.26711, #[rad/s]
        'v_rated'   : 11.4, #[m/s]
        'beta_rated': 0, #[rad]
        'max_pitch_rate': 0.1745,  #[rad/s]
        'max_torque_rate': 1500000 #[Nm/s] 
}

# TurbSim files
wind_profiles = ['TurbSim_13_5_4433456.dat','TurbSim_15_8_54433456.dat','TurbSim_17_9_64433456.dat','TurbSim_20_11_133456.dat','TurbSim_23_12_14433456.dat']
w_profiles = []

for i in range(len(wind_profiles)):
    with open('Wind profiles/' + wind_profiles[i], 'r') as file:
        file = file.readlines()
        tim = []
        wind_speed = []
        for row in file:
            if row.startswith('   Time'):
                continue
            value = row.split()
            t = float(value[0])
            v = float(value[1])
            if v<11.4:
                v=11.4
            tim.append(t)
            wind_speed.append(v)
    w_profiles.append(wind_speed)

# Defining the cost function
def costFun(weights):
    fitness = 0
    for i in range(len(w_profiles)):
        f = np.zeros_like(tim)
        # Here I pass the simulation with the individual
        ode = ODE(turbine, tim, turbine['w_rated'] , w_profiles[i],w_profiles[i])
        w,beta,tau = ode.sim(weights)
        for i, ti in enumerate(tim):
            f[i] = (w[i] - turbine['w_rated']) ** 2
        fitness_i = math.fsum(f) / len(tim)
        fitness = fitness + fitness_i
    print(fitness)
    return fitness

bounds = [(0.,100.),(0.,100.)]
result = gp_minimize(costFun , bounds , n_calls = 200)

# --- Save stuff

import pickle
import os

os.makedirs('./RESULTS/'+filename+'/', exist_ok=True)

with open(f'./RESULTS/'+filename+'/'+filename+'_weights.pickle', 'wb') as handle:
    pickle.dump(result.x, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Plots - Post-processing

ode = ODE(turbine, tim, turbine['w_rated'] , w_profiles[0], w_profiles[0])
w,beta,tau = ode.sim(result.x)

plt.plot(tim, beta)
plt.savefig('./RESULTS/'+filename + "/beta.png")
plt.close()

plt.figure()
plt.plot(tim,w)
plt.savefig('./RESULTS/'+filename + "/w.png")
plt.close()

plot_convergence(result)
plt.savefig('./RESULTS/'+filename + "/conv.png")
plt.close()





