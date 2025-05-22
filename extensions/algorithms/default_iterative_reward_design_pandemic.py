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

from sacred import Experiment
from occupancy_measures.agents.orpo import ORPO, ORPOPolicy

from extensions.reward_modeling.reward_wrapper_reg import RewardWrapper,RewardModel
from pandemic_simulator.environment.pandemic_env import PandemicPolicyGymEnv

from extensions.algorithms.default_train_policy_pandemic import ex
from extensions.environments.pandemic_configs import get_pandemic_env_gt_rew
import extensions.algorithms.default_train_policy_pandemic
import extensions.algorithms.default_unique_id_state_pandemic as unique_id_state
import time

import warnings
warnings.filterwarnings("ignore")


# Create a new experiment for iterative reward design
iterative_ex = Experiment("iterative_reward_design_pandemic", save_git_info=False)

# def create_custom_env(config, reward_wrapper_class: Optional[Callable] = None, reward_net= None):
#     """
#     Creates an environment with a configurable reward wrapper.
    
#     Args:
#         config: The environment configuration
#         reward_wrapper_class: Optional custom reward wrapper class to use instead of RewardWrapper
#     """
#     base_env = PandemicPolicyGymEnv(config)
#     print (base_env.observation_space.shape[0])
#     print (base_env.action_space.shape[0])
#     wrapper_class = reward_wrapper_class if reward_wrapper_class is not None else RewardWrapper
#     #pass in reward_net as an input to the wrapper class
#     return wrapper_class(base_env, reward_model=config.get("reward_model", "default"), reward_net=reward_net)

@iterative_ex.config
def config():
    # Default configuration matching your typical train_policy arguments
    env_to_run = "pandemic"
    level=4
    reward_fun = "proxy"
    exp_algo = "ORPO"
    om_divergence_coeffs_1 = ["0.06"]
    om_divergence_coeffs_2 = ["0.0"]
    checkpoint_to_load_policies = None
    checkpoint_to_load_current_policy = None
    seed = 0
    experiment_tag = "state"
    om_divergence_type = ["kl"]
    num_rollout_workers = 2
    num_gpus = 1
    experiment_parts = [env_to_run]
    reward_wrapper_class = None  # Use default RewardWrapper if None
    num_training_iters_1 = 260,
    num_training_iters_2 = 260,

    # num_rollouts = 10  # Number of rollouts to collect
    # rollout_length = 192  # Length of each rollout

# @iterative_ex.automain
@iterative_ex.automain

def main(
    env_to_run,
    level,
    reward_fun,
    exp_algo,
    om_divergence_coeffs_1,
    om_divergence_coeffs_2,
    checkpoint_to_load_policies,
    checkpoint_to_load_current_policy,
    seed,
    experiment_tag,
    om_divergence_type,
    num_rollout_workers,
    num_gpus,
    experiment_parts,
    reward_wrapper_class,
    num_training_iters_1,
    num_training_iters_2,
    _log
):
    """
    Main function that runs the training with a configurable reward wrapper.
    """

    # unique_id_state.state["unique_id"] = f"{reward_fun}_{seed}_{int(time.time())}"

    #all these args must be manual set per environment (annoying but we can't init gym env here) 
    if "pandemic" in env_to_run:
        reward_model = RewardModel(
            obs_dim=24*13, # Assuming the observation space is a 1D array of size 24*13
            action_dim=3,
            sequence_lens=193,
            discrete_actions = True,
            env_name="pandemic",
            unique_id=unique_id_state.state["unique_id"]
        )    
    elif "tomato" in env_to_run:
        reward_model = RewardModel(
            obs_dim=2*36, # Assuming the observation space is a 1D array of size 24*13
            action_dim=4,
            sequence_lens=100,
            discrete_actions = True,
            env_name="tomato",
            unique_id=unique_id_state.state["unique_id"],
            n_epochs=100
        )
    else:
        raise ValueError("Unsupported environment type")
    reward_model.zero_model_params()
    reward_model.save_params()
    
    for i in range(10):
        print ("(iterative_reward_design.py) UNIQUE ID (WHICH SHOULD BE THE SAME FOR ALL ITERATIONS):")
        print (unique_id_state.state["unique_id"])
        print ("======================")
        print ("checkpoint_to_load_current_policy", checkpoint_to_load_current_policy)
        print ("checkpoint_to_load_policies", checkpoint_to_load_policies)
        print ("======================")
        # if i == 0:
        # if (int(om_divergence_coeffs_1[0]) == 0 and i == 0) or int(om_divergence_coeffs_1[0]) != 0:

        # #TODO: remember to delete after debugging
        # if i == 0:
        #     checkpoint_to_load_current_policy = "/next/u/stephhk/orpo/data/logs/tomato/2025-05-07_13-09-29/checkpoint_000300"
        #     temp_num_training_iters_1=num_training_iters_1
        #     num_training_iters_1 = 1
        # else:
        #     checkpoint_to_load_current_policy = None
        #     num_training_iters_1 = temp_num_training_iters_1

        
        reference_result = ex.run(
            config_updates={
                "env_to_run": env_to_run,
                "level": level,
                "reward_fun": reward_fun,
                "exp_algo": exp_algo,
                "om_divergence_coeffs": om_divergence_coeffs_1,
                "checkpoint_to_load_policies": checkpoint_to_load_policies,
                "checkpoint_to_load_current_policy": checkpoint_to_load_current_policy,
                "seed": seed,
                "experiment_tag": experiment_tag,
                "om_divergence_type": om_divergence_type,
                "num_rollout_workers": num_rollout_workers,
                "num_gpus": num_gpus,
                "experiment_parts": experiment_parts,
                "num_training_iters": num_training_iters_1,      
            }
        )
        eval_batch_reference = reference_result.result[2]

        
        over_opt_result = ex.run(
            config_updates={
                "env_to_run": env_to_run,
                "level":level,
                "reward_fun": reward_fun,
                "exp_algo": exp_algo,
                "om_divergence_coeffs": om_divergence_coeffs_2,
                "checkpoint_to_load_policies": None,
                "checkpoint_to_load_current_policy": checkpoint_to_load_current_policy,
                "seed": seed,
                "experiment_tag": experiment_tag,
                "om_divergence_type": om_divergence_type,
                "num_rollout_workers": num_rollout_workers,
                "num_gpus": num_gpus,
                "experiment_parts": experiment_parts,
                "num_training_iters": num_training_iters_2,
            }
        )

        
        eval_batch_over_opt = over_opt_result.result[2]
        reward_model.update_params(eval_batch_over_opt["current"][:len(eval_batch_reference["current"])],eval_batch_reference["current"], iteration=i,use_minibatch=True)

        checkpoint_to_load_policies = ["/next/u/stephhk/orpo/"+reference_result.result[1]]
        # checkpoint_to_load_current_policy = "/next/u/stephhk/orpo/"+reference_result.result[1]

        print (checkpoint_to_load_policies)
        print (checkpoint_to_load_current_policy)

        print ("======================")
        print ("saving over opt checkpoint to:", over_opt_result.result[1])
        print ("saving reference checkpoint to:", reference_result.result[1])
        print ("======================")

        time.sleep(5)

        