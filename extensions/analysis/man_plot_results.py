import os
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

#---------Tomato Env-------------------
# env = "tomato"
# frh_extra_reg = [0.44000, 6.98399, 7.7040, 8.76399, 8.425999, 8.547999, 6.7789, 9.00299]
# frh = [0.672, 2.842000000000002, 5.625999999999, 7.346000, 7.434000, 8.34999, 8.3239, 8.64799999]

# pref_against_ref = [6.7259999, 6.74199999, 7.6359999999, 8.13599999999, 8.311999999, 9.34599999999]
# pref_against_uniform = [6.72599, 7.6719, 7.0519, 7.611999, 8.0299, 8.0059, 8.181999, 8.327999]
# constrained_mean_rew =  7.07 #proxy reward is the same
# best_constrained_mean_rew = 9.17

# training_w_true_rew_mean_rew = 8.54

# moving_ref_policy = [6.1559999999999, 5.127999999, 6.619999999, 6.31999999, 5.611999999, 6.50399999999,  6.9919999999, 6.70799999]
# pref_against_fixed_ref = [6.7259999999, 6.63799999,  6.7159999999,  6.61199999, 6.881999999, 6.7959999999,  6.6859999, 7.12799999999]
#---------------------------------------
#---------Pandemic Env-------------------
env = "pandemic"
frh_extra_reg=[-114.9872,  -8.45328, -7.317, -7.9821, -11.831, -5.2568,  -8.261, -5.47270]
frh=[-114.9872, -9.10777,-29.710, -8.154680, -5.1419, -5.2109, -4.54066]

pref_against_ref = [-12.11442, -11.8717, -12.0077, -10.514, -10.9109, -11.730,  -11.34714]#TODO: why is the initial performance so high and != -12.64?
pref_against_uniform =[-12.64294, -11.9738, -11.52787, -11.523695, -10.66271, -10.9774,  -10.624, -11.280]
pref_against_fixed_ref = [-12.642, -12.0304, -11.44133, -11.792017, -11.259]

constrained_mean_rew = -10.24
training_w_true_rew_mean_rew= -2.65

#TODO: replace with updated seed later
moving_ref_policy = [-11.8827,  -12.0061, -12.2583, -11.979, -11.5827, -12.2974,  -11.6393, -11.9212]



if env == "tomato":
    n_pref_per_iter = 19*19
elif env == "pandemic":
    n_pref_per_iter = 79*79

def plot_line(x, color, label, n_pref_per_iter):
    num_prefs = np.arange(0, len(x))*n_pref_per_iter
    plt.plot(num_prefs,x, label=label, color=color)

plot_line(frh_extra_reg, color="blue", label="Fix Reward Hacking Alg.", n_pref_per_iter=n_pref_per_iter)
plot_line(frh, color="navy", label="Fix Reward Hacking Alg.", n_pref_per_iter=n_pref_per_iter)
plot_line(pref_against_ref, color="red", label="Pref. Against Ref.", n_pref_per_iter=n_pref_per_iter)
plot_line(pref_against_fixed_ref, color="darkred", label="Pref. Against Fixed Ref.", n_pref_per_iter=n_pref_per_iter)
plot_line(pref_against_uniform, color="orange", label="Pref. Against Unif.", n_pref_per_iter=n_pref_per_iter)
# plot_line(moving_ref_policy, color="green", label="Moving Ref. Policy", n_pref_per_iter=n_pref_per_iter)


#plot constrained_mean_rew as a horizontal line
plt.axhline(y=constrained_mean_rew, color='black', linestyle='--', label='KL-Constrained RL with Proxy Reward')
plt.axhline(y=training_w_true_rew_mean_rew, color='silver', linestyle='--', label='RL with True Reward')

# plt.axhline(y=best_constrained_mean_rew, color='gray', linestyle='--', label='RLHF with Proxy Reward, Best Regularization')
#set y limit
if env == "pandemic":
    plt.ylim(-20, -2)
else:
    plt.ylim(0, 10)
# plt.xlim(0, 2600)
plt.xlabel('Number of Preferences')
plt.ylabel('True Reward')
plt.legend()

plt.savefig('frh_vs_up.png')