import numpy as np
from deap import tools, gp, algorithms, base, creator
import operator
import math
import random
import matplotlib.pyplot as plt
from GP_ODE import ODE
import warnings
import cloudpickle
import base64
from scipy.signal import butter,lfilter,lfilter_zi

def lambda2str(expr):
    b = cloudpickle.dumps(expr)
    s = base64.b64encode(b).decode()
    return s

warnings.simplefilter("ignore")

def protectedDiv(left, right):
    try:
        return left / right
    except (ZeroDivisionError, ValueError, OverflowError):
        return 1
    except math.DomainError:
        return 1

def protectedSquare(x):
    try:
        return np.square(x)
    except (ValueError, OverflowError):
        return 1

def protectedLog(x):
    try:
       return math.log(x)
    except (ValueError, OverflowError):
       return 1

def protectedSqrt(x):
    try:
       return math.sqrt(x)
    except (ValueError, OverflowError):
       return 1
   
def protectedExp(a):
    try:
        return math.exp(a)
    except (ValueError, OverflowError):
        return 1
    
def Filter(wind):
    # Sampling frequency
    fs = 1/(tim[1]-tim[0]) # Hz
    # Cut-off frequency - Frequency of -3dB attenuation
    fc = 0.25 # Hz
    # Butterworth filter of the 4° order
    b, a = butter(4, 2*np.pi*fc/fs, 'low', analog=False)
    # Calculate initial state of filter
    zi = lfilter_zi(b, a) * wind[0]
    w_filt, _  = lfilter(b, a, wind, zi=zi)
    return w_filt

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
w_filt = []

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
    w_filt.append(Filter(wind_speed))

# Initialize the primitive and terminal set 
pset = gp.PrimitiveSet("MAIN", 3)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(protectedSquare, 1)
pset.addPrimitive(operator.abs, 1)
pset.addPrimitive(protectedSqrt, 1)
pset.addPrimitive(protectedDiv, 2)
pset.addPrimitive(protectedLog, 1)
pset.addPrimitive(protectedExp, 1)
pset.addPrimitive(math.sin, 1)
pset.addPrimitive(math.tanh, 1)
pset.addTerminal(turbine['w_rated'])
pset.addEphemeralConstant("rand101", lambda: random.randint(-1,1))
pset.renameArguments(ARG0='v_i', ARG1='w_i',ARG2='int_e_i')

'''
The following block you can left unchanged. It is used to create the individual and the population. You find
the description inside the presentation if interested in exploring different possibilities.
'''

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin,pset=pset)
toolbox = base.Toolbox()
#toolbox.register("expr", gp.genFull, pset=pset, min_=1, max_=3)           # --- full method
# toolbox.register("expr", gp.genGrow, pset=pset, min_=1, max_=3)         # --- grow
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)  # --- half and half
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

# Defining the cost function
def evalVDPReg(individual):
    func = toolbox.compile(expr=individual) # --- the individual is "lambdified" -i.e. transformed into a lambda function
    fitness = 0
    for i in range(len(w_profiles)):
        f = np.zeros_like(tim)
        # Here I pass the simulation with the individual
        ode = ODE(turbine, tim, turbine['w_rated'] , w_profiles[i], w_filt[i])
        w,beta,tau,flag = ode.sim(func)
        for i, ti in enumerate(tim):
            f[i] = (w[i] - turbine['w_rated']) ** 2
        fitness_i = math.fsum(f) / len(tim)
        if flag==True:
            fitness_i = fitness_i + 10
            print('Error somewhere in this simulation')
        fitness = fitness + fitness_i
    return fitness,

'''
Also here you can ignore the following block. It is used to define the genetic operators. You find
the description inside the presentation if interested in exploring different possibilities.
'''

toolbox.register("select", tools.selTournament, tournsize=7)
toolbox.register("evaluate", evalVDPReg) # --- the evaluation function is the one defined above, must have the same name!
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=3) #
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
stats_size = tools.Statistics(len)
mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
mstats.register("avg", np.mean)
#mstats.register("std", np.std)
mstats.register("min", np.min)
mstats.register("max", np.max)
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

pop = toolbox.population(n=50)
hof = tools.HallOfFame(1)
# pop, log = algorithms.eaSimple(pop, toolbox, 0.5, 0.25, 30, stats=mstats,
#                                halloffame=hof, verbose=True)
# pop, log = algorithms.eaMuCommaLambda(pop, toolbox, 300, 350, 0.65, 0.25, 300, stats=mstats,
#                                halloffame=hof, verbose=True)
pop, log = algorithms.eaMuPlusLambda(pop, toolbox, 50, 65 , 0.65, 0.25, 43, stats=mstats,
                               halloffame=hof, verbose=True)
hof_ = toolbox.compile(hof[0])

# --- Save stuff

import pickle
import os

os.makedirs('./RESULTS/'+filename+'/', exist_ok=True)

with open('./RESULTS/'+filename+'/'+filename+'_sim_logs.pickle', 'wb') as handle:
    pickle.dump(log, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('./RESULTS/'+filename+'/'+filename+'_best_individual.txt', 'w') as file_handle:
    file_handle.write('{}'.format(hof[0]))

s = lambda2str(hof_)

with open('./RESULTS/'+filename+'/'+filename+'_sim_hof.pickle', 'wb') as handle:
    pickle.dump(s, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Plots - Post-processing

ode = ODE(turbine, tim, turbine['w_rated'] , w_profiles[0], w_filt[0])
w,beta,tau,f = ode.sim(hof_)

fig, ax = plt.subplots()
y_pred = []
for i, ti in enumerate(tim):
    e_i = np.trapz(w[0:i]-ode.w_rated,None,tim[1]-tim[0])
    y_pred.append(hof_(w_profiles[0][i], w[i],e_i))

ax.plot(tim, y_pred, label='GP')
plt.savefig('./RESULTS/'+filename + "/beta.png")
plt.close()

plt.figure()
plt.plot(tim,w)
plt.savefig('./RESULTS/'+filename + "/w.png")
plt.close()

import networkx as nx
import matplotlib.pyplot as plt

# Individuo da visualizzare
ind = hof[0]

# Crea un grafo diretto aciclico per l'individuo
expr = gp.PrimitiveTree(ind)
nodes, edges, labels = gp.graph(expr)

# Crea il grafo NetworkX
g = nx.DiGraph()
g.add_nodes_from(nodes)
g.add_edges_from(edges)

# Visualizza il grafo
pos = nx.nx_pydot.graphviz_layout(g, prog="dot")
nx.draw_networkx_nodes(g, pos)
nx.draw_networkx_edges(g, pos)
nx.draw_networkx_labels(g, pos, labels)
plt.savefig('./RESULTS/'+filename + "/tree.png")
plt.close()





