import numpy as np
import matplotlib.pyplot as plt
from numpy.random import *
import time
import gc

def delta(v_t_fire):
    if v_t_fire == 1.:
        return 1
    else:
        return 0

def depressing_synapses_model(x_t, y_t, z_t, t_rec, t_in, U, dt, v_t_fire):    
    dx = z_t / t_rec - U * x_t * delta(v_t_fire) * (1/dt)
    dy = - y_t / t_in + U * x_t * delta(v_t_fire) * (1/dt)
    dz = - z_t / t_rec + y_t / t_in
    
    x_t_ = x_t + dx * dt
    y_t_ = y_t + dy * dt
    z_t_ = z_t + dz * dt
    return x_t_, y_t_, z_t_

def IF_neuron_wn(v_t, t_mem, I_syn, I_b, dt, v_t_fire, current_time, i_th_neuron, refrac_flag, intT, theta_i, wn):
    if refrac_flag[i_th_neuron, current_time] == 1:
        v_t_ = 13.5
    else:
        dv = (- v_t + I_syn + I_b) / t_mem
        v_t_ = v_t + dv * dt + wn * (dt ** 0.5)
        if v_t_ >= theta_i:
            for i in range(30):
                if current_time + i >= intT:
                    pass
                else:
                    refrac_flag[i_th_neuron, current_time + i] = 1
            v_t_ = 13.5
            v_t_fire[i_th_neuron, current_time + 1] = 1.
            #print("neuron fired !")

    return v_t_

def raster(event_times_list, **kwargs):
    """
    Creates a raster plot
    Parameters
    ----------
    event_times_list : iterable
                       a list of event time iterables
    color : string
            color of vlines
    Returns
    -------
    ax : an axis containing the raster plot
    """
    ax = plt.gca()
    for ith, trial in enumerate(event_times_list):
        plt.vlines(trial, ith + .5, ith + 1.5, **kwargs)
    plt.ylim(.5, len(event_times_list) + .5)
    return ax

def dynamic_theta_i(t_th, S_i, theta_i_t, theta_i_0, v_t_fire, dt):
    dtheta_i = - (theta_i_t - theta_i_0) / t_th + S_i * delta(v_t_fire) * (1/dt)
    theta_t_ = theta_i_t + dtheta_i * dt
    return theta_t_

def plot(v_t_fire, intT, times, N):
    spikes = []

    for j in range(N):
        spike = []
        for i in range(intT):
            if v_t_fire[j,i] == 1:
                spike.append(i)
                #print("{}th neuron fired spike appended".format(j))
        spikes.append(spike)
    
    fig = plt.figure(figsize = (40, 10))        
    ax = raster(spikes)
    plt.title('small network raster plot')
    plt.xlabel('time [msec]')
    plt.ylabel('neuron')

    plt.savefig("raster_40_allext_dynamicthr_wn.png")


    plt.clf()
    fractions = np.zeros(intT)

    for t in range(intT):
        times[t] = t *dt
        fraction = 0
        for i in range(N):
            if v_t_fire[i, t] == 1:
                fraction +=1
        fraction = fraction / N
        fractions[t] = fraction

    fig = plt.figure(figsize = (40, 10))
    plt.plot(times, fractions, label='fraction')
    plt.legend(loc="lower right")
    plt.savefig("fraction_40_allext_dynamicthr_wn.png")
    plt.clf()

N = 40

J_ij = np.empty((0, N))
for i in range(N):
    tmp = np.zeros((1,N))
    for j in range(N):
        if randint(10) % 10 == 1:
            tmp[0, j] = 15 * np.random.normal(1, 0.5, 1)
    J_ij = np.append(J_ij, tmp, axis = 0)

x = 0.5
y = 0.5
z = 1 - x - y

#mili second
t_in= 3
t_mem = 30
t_th = 100

U = np.zeros(N)
for i in range(N):
    U[i] = np.clip(np.random.normal(loc = 0.5, scale = 0.25, size = 1), 0.1, 0.9)

I_b = np.zeros(N)
for i in range(N):
    I_b[i] = np.random.uniform(low = 14.7 - 0.4, high = 14.7 + 0.4, size = 1)

t_rec = np.zeros(N)
for i in range(N):
    t_rec[i] = np.clip(np.random.normal(loc = 800, scale = 400, size = 1), 5, None)

S_i = np.zeros(N)
for i in range(N):
    S_i[i] = np.random.uniform(low = -0.06 * 1.5, high = 0.04 * 1.5, size = 1)
 
dt = 0.1

Total = 100000
T = Total * (1/dt)
intT = int(T)

times = np.zeros(intT)

v_t = np.zeros((N, intT))
v_t_fire = np.zeros((N, intT))
refrac_flag = np.zeros((N, intT))
theta_i_t = np.zeros((N, intT))
x_t = np.zeros((N, N))
y_t = np.zeros((N, N))
z_t = np.zeros((N, N))

x_t_1 = np.zeros((N, N))
y_t_1 = np.zeros((N, N))
z_t_1 = np.zeros((N, N))

wn = np.random.normal(0, 0.5, 1)

# for debugging
I_syn = np.zeros(intT)

v_t[:,0] = 14.
x_t[:, :] = x
y_t[:, :] = y
z_t[:, :] = z
theta_i_0 = 15.
theta_i_t[:, 0] = theta_i_0

start = time.time()

for t in range(intT - 1):
    # this is i * dt mili second
    times[t+1] = t * dt

    if t % 10000 == 0:
        print("this is time {} ms/{} --- {} from previous stamp".format(t * dt, Total, time.time() - start))
        plot(v_t_fire, intT, times, N)
        start = time.time()
    for i in range(N):
        # calculate the variable of i_th neuron
        I_syn_i = 0.
        #if (t +1) % 10 == 0:
        #    I_b[i] = np.random.normal(loc = 14.7 - 0.4, scale = 14.7 + 0.4, size = 1)
        for j in range(N):
            # j_th synapse of i_th neuron
            # calculate I_syn = sum_of J_ij * e_t_ij
            if j == i:
                pass
            else:
                x_t_1[i, j], y_t_1[i, j], z_t_1[i, j] = depressing_synapses_model(x_t[i, j], y_t[i, j], z_t[i, j], t_rec[i], t_in, U[i], dt, v_t_fire[i, t])

            I_syn_i += J_ij[j, i] * y_t[j, i]
            x_t[i, j] = x_t_1[i, j]
            y_t[i, j] = y_t_1[i, j]
            z_t[i, j] = z_t_1[i, j]
        wn = np.random.normal(0, 0.5, 1)
        v_t[i, t+1] = IF_neuron_wn(v_t[i, t], t_mem, I_syn_i, I_b[i], dt, v_t_fire, t, i, refrac_flag, intT, theta_i_t[i, t], wn)  
        theta_i_t[i, t+1] = dynamic_theta_i(t_th, S_i[i], theta_i_t[i, t], theta_i_0, v_t_fire[i, t], dt)
    if t == intT - 2:
        print("this is time {}\ndone !".format((t + 2)* dt))



spikes = []

for j in range(N):
    spike = []
    for i in range(intT):
        if v_t_fire[j,i] == 1:
            spike.append(i)
            #print("{}th neuron fired spike appended".format(j))
    spikes.append(spike)
            
fig = plt.figure(figsize = (40, 10), dpi = 300)
ax = raster(spikes)
plt.title('small network raster plot')
plt.xlabel('time [msec]')
plt.ylabel('neuron')

plt.savefig("raster_40_allext_dynamicthr_wn.png")


plt.clf()
fractions = np.zeros(intT)

for t in range(intT):
    fraction = 0
    for i in range(N):
        if v_t_fire[i, t] == 1:
            fraction +=1
    fraction = fraction / N
    #print(fraction)
    fractions[t] = fraction

fig = plt.figure(figsize = (40, 10), dpi = 300)
plt.plot(times, fractions, label='fraction')

plt.legend(loc="lower right")
plt.savefig("fraction_40_allext_dynamicthr_wn.png")
plt.clf()
