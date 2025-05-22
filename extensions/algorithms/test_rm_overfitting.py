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
from extensions.reward_modeling.reward_wrapper_reg import RewardWrapper,RewardModel,ReplayBuffer
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
    env_name="pandemic_testing_sas",
    unique_id=-104,
    lr=0.0001
)    


#load replay buffer sized via
# with open(f"active_models/replay_buffer_{self.unique_id}.pkl", "wb") as f:
#    torch.save(self.replay_buffer, f)

#load replay buffer:
torch.serialization.add_safe_globals([ReplayBuffer])
with open(f"active_models/replay_buffer_{21}.pkl", "rb") as f:
    reward_model.replay_buffer = torch.load(f)
reward_model.replay_buffer.buffer = reward_model.replay_buffer.buffer[:18723]
print ("replay buffer size: ", len(reward_model.replay_buffer))
reward_model.update_params_gt(None, None,0, debug_mode=True,use_minibatch=True,force_n_epochs=20,psuh2zero=10)

# reward_model.unique_id = 3
# reward_model.load_params()
# reward_model.unique_id=-1