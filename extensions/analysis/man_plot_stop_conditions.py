import os
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

#---------Tomato Env-------------------
env = "tomato"
win_number = [0.04155124653739612, 0.5069252077562327, 0.667590027700831, 0.7645429362880887, 0.6869806094182825, 0.6925207756232687, 0.5346260387811634, 0.7340720221606648]
#---------Pandemic Env-------------------
# env = "pandemic"
# win_number=[0.0001602307322544464, 0.6207338567537254, 0.5417401057522833, 0.8139721198525878, 0.8846338727767986, 0.8945681781765743]


if env == "tomato":
    n_pref_per_iter = 19*19
elif env == "pandemic":
    n_pref_per_iter = 79*79

def plot_line(x, color, label, n_pref_per_iter):
    num_prefs = np.arange(0, len(x))*n_pref_per_iter
    plt.plot(num_prefs,x, label=label, color=color)

plot_line(win_number, color="navy", label="Fix Reward Hacking Alg.", n_pref_per_iter=n_pref_per_iter)

# plt.axhline(y=best_constrained_mean_rew, color='gray', linestyle='--', label='RLHF with Proxy Reward, Best Regularization')
#set y limit
# if env == "pandemic":
#     plt.ylim(-20, -2)
# else:
plt.ylim(0, 1)
# plt.xlim(0, 2600)
plt.xlabel('# of Preferences')
plt.ylabel('Fraction of wins against\nconstrained PPO at iteration 0')
plt.legend()

plt.savefig('frh_vs_up.png')