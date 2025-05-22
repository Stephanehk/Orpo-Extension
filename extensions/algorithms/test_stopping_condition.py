import os
from typing import Callable, Optional, List, Union
import ray
from ray.rllib.algorithms import Algorithm
from ray.rllib.evaluation.rollout_worker import RolloutWorker
from ray.rllib.policy.policy import Policy
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.env.multi_agent_env import make_multi_agent
from ray.tune.registry import register_env
from ray.rllib.policy.sample_batch import MultiAgentBatch, SampleBatch, concat_samples
import torch
from sacred import Experiment
from occupancy_measures.agents.orpo import ORPO, ORPOPolicy

# from extensions.reward_modeling.reward_wrapper import RewardWrapper,RewardModel,ReplayBuffer
# from extensions.reward_modeling.reward_wrapper_reg import RewardWrapper,RewardModel,ReplayBuffer
from extensions.reward_modeling.reward_wrapper_reg_extra import RewardWrapper,RewardModel,ReplayBuffer
# from extensions.reward_modeling.reward_wrapper import RewardModel

from pandemic_simulator.environment.pandemic_env import PandemicPolicyGymEnv

from extensions.algorithms.train_policy import ex
from extensions.environments.pandemic_configs import get_pandemic_env_gt_rew
import extensions.algorithms.train_policy
import extensions.algorithms.unique_id_state as unique_id_state
import time

import warnings
warnings.filterwarnings("ignore")


reward_model = RewardModel(
    obs_dim=2*24*13, # Assuming the observation space is a 1D array of size 24*13
    action_dim=3,
    sequence_lens=193,
    discrete_actions = True,
    env_name="pandemic_sas",
    unique_id=-106,
    lr=0.0001,
    n_epochs=200,
)    

#load replay buffer:
torch.serialization.add_safe_globals([ReplayBuffer])
with open(f"active_models/replay_buffer_{575}.pkl", "rb") as f:
    reward_model.replay_buffer = torch.load(f)
# reward_model.replay_buffer.buffer = reward_model.replay_buffer.buffer[:2*79*79]
print ("replay buffer size: ", len(reward_model.replay_buffer))

good_epoch_trajs = reward_model.replay_buffer.buffer[2*79*79:3*79*79]
print (len(good_epoch_trajs))
over_opt_pref=0
under_opt_pref=0
for pair in reward_model.replay_buffer.buffer:
    if pair["true_label"] == 0:
        over_opt_pref +=1
    else:
        under_opt_pref +=1

print ("# of times over-optimized policy is preferred: ", over_opt_pref)
print ("# of times under-optimized policy is preferred: ", under_opt_pref)

#traj1 = over-opt, traj2 = under-opt

# reward_model.update_params(None, None,0, debug_mode=True,use_minibatch=True)

# reward_model.unique_id = 3
# reward_model.load_params()
# reward_model.unique_id=-1