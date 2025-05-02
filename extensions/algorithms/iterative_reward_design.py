import os
from typing import Callable, Optional, List, Union

from sacred import Experiment
from extensions.reward_modeling.reward_wrapper import RewardWrapper
from pandemic_simulator.environment.pandemic_env import PandemicPolicyGymEnv

from extensions.algorithms.train_policy import create_env as original_create_env, ex

# Create a new experiment for iterative reward design
iterative_ex = Experiment("iterative_reward_design", save_git_info=False)

def create_custom_env(config, reward_wrapper_class: Optional[Callable] = None):
    """
    Creates an environment with a configurable reward wrapper.
    
    Args:
        config: The environment configuration
        reward_wrapper_class: Optional custom reward wrapper class to use instead of RewardWrapper
    """
    base_env = PandemicPolicyGymEnv(config)
    wrapper_class = reward_wrapper_class if reward_wrapper_class is not None else RewardWrapper
    return wrapper_class(base_env, reward_model=config.get("reward_model", "default"))

@iterative_ex.config
def config():
    # Default configuration matching your typical train_policy arguments
    env_to_run = "pandemic"
    reward_fun = "proxy"
    exp_algo = "ORPO"
    om_divergence_coeffs = ["0.06"]
    checkpoint_to_load_policies = ["data/logs/pandemic/BC/true/model_128-128/seed_0/2025-04-28_09-41-36/checkpoint_000260"]
    checkpoint_to_load_current_policy = "data/logs/pandemic/BC/true/model_128-128/seed_0/2025-04-28_09-41-36/checkpoint_000260"
    seed = 0
    experiment_tag = "state"
    om_divergence_type = ["kl"]
    num_rollout_workers = 2
    num_gpus = 1
    experiment_parts = [env_to_run]
    reward_wrapper_class = None  # Use default RewardWrapper if None
    num_training_iters = 260

@iterative_ex.automain
def main(
    env_to_run,
    reward_fun,
    exp_algo,
    om_divergence_coeffs,
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
    _log
):
    """
    Main function that runs the training with a configurable reward wrapper.
    """
    # Override the create_env function in train_policy with our custom version
    global create_env
    create_env = lambda config: create_custom_env(config, reward_wrapper_class)
    
    # Run the original experiment with all the passed parameters
    ex.run(
        config_updates={
            "env_to_run": env_to_run,
            "reward_fun": reward_fun,
            "exp_algo": exp_algo,
            "om_divergence_coeffs": om_divergence_coeffs,
            "checkpoint_to_load_policies": checkpoint_to_load_policies,
            "checkpoint_to_load_current_policy": checkpoint_to_load_current_policy,
            "seed": seed,
            "experiment_tag": experiment_tag,
            "om_divergence_type": om_divergence_type,
            "num_rollout_workers": num_rollout_workers,
            "num_gpus": num_gpus,
            "experiment_parts": experiment_parts,
            "num_training_iters": num_training_iters,
        }
    ) 


    #Next steps:
    #(1) get the model that was just trained
    
    #(2) rollout out the model
    #(3) get preferences over rollouts

