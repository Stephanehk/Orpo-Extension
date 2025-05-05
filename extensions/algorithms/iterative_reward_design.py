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

from extensions.reward_modeling.reward_wrapper import RewardWrapper,RewardModel
from pandemic_simulator.environment.pandemic_env import PandemicPolicyGymEnv

from extensions.algorithms.train_policy import ex
from extensions.environments.pandemic_configs import get_pandemic_env_gt_rew
import extensions.algorithms.train_policy

# Create a new experiment for iterative reward design
iterative_ex = Experiment("iterative_reward_design", save_git_info=False)

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
    reward_fun = "proxy"
    exp_algo = "ORPO"
    om_divergence_coeffs_1 = ["0.06"]
    om_divergence_coeffs_2 = ["0.0"]
    checkpoint_to_load_policies = ["data/logs/pandemic/BC/true/model_128-128/seed_0/2025-04-28_09-41-36/checkpoint_000260"]
    checkpoint_to_load_current_policy = "data/logs/pandemic/BC/true/model_128-128/seed_0/2025-04-28_09-41-36/checkpoint_000260"
    seed = 0
    experiment_tag = "state"
    om_divergence_type = ["kl"]
    num_rollout_workers = 2
    num_gpus = 1
    experiment_parts = [env_to_run]
    reward_wrapper_class = None  # Use default RewardWrapper if None
    num_training_iters = 260,
    unique_id = 1
    # num_rollouts = 10  # Number of rollouts to collect
    # rollout_length = 192  # Length of each rollout

# @iterative_ex.automain
@iterative_ex.automain

def main(
    env_to_run,
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
    num_training_iters,
    unique_id,
    _log
):
    """
    Main function that runs the training with a configurable reward wrapper.
    """

    #all these args must be manual set per environment (annoying but we can't init gym env here) 
    reward_model = RewardModel(
        obs_dim=24*13, # Assuming the observation space is a 1D array of size 24*13
        action_dim=3,
        sequence_lens=193,
        discrete_actions = True,
        unique_id=unique_id
    )    

    for i in range(3):
        
        print ("======================")
        print ("checkpoint_to_load_current_policy", checkpoint_to_load_current_policy)
        print ("checkpoint_to_load_policies", checkpoint_to_load_policies)
        print ("======================")
        # if i == 0:
        # if (int(om_divergence_coeffs_1[0]) == 0 and i == 0) or int(om_divergence_coeffs_1[0]) != 0:
    
        # Run the original experiment with all the passed parameters
        #TODO: update the reference policy to be the one we trained previously
        #TODO: also update the initialization policy
        reference_result = ex.run(
            config_updates={
                "env_to_run": env_to_run,
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
                "num_training_iters": num_training_iters,
                "unique_id":unique_id
            }
        )
        eval_batch_reference = reference_result.result[2]

        over_opt_result = ex.run(
            config_updates={
                "env_to_run": env_to_run,
                "reward_fun": reward_fun,
                "exp_algo": exp_algo,
                "om_divergence_coeffs": om_divergence_coeffs_2,
                "checkpoint_to_load_policies": checkpoint_to_load_policies,
                "checkpoint_to_load_current_policy": checkpoint_to_load_current_policy,
                "seed": seed,
                "experiment_tag": experiment_tag,
                "om_divergence_type": om_divergence_type,
                "num_rollout_workers": num_rollout_workers,
                "num_gpus": num_gpus,
                "experiment_parts": experiment_parts,
                "num_training_iters": num_training_iters,
                "unique_id":unique_id
            }
        )

        
        eval_batch_over_opt = over_opt_result.result[2]
        
        #TODO: our reward model might need to input a history of obs (possibly just last obs) instead of just the current obs
        reward_model.update_params(eval_batch_over_opt["current"],eval_batch_reference["current"], iteration=i)

        checkpoint_to_load_policies = reference_result.result[1]
        checkpoint_to_load_current_policy = reference_result.result[1]

        