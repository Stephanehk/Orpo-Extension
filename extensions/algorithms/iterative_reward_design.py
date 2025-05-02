import os
from typing import Callable, Optional, List, Union
import ray
from ray.rllib.algorithms import Algorithm
from ray.rllib.evaluation.rollout_worker import RolloutWorker
from ray.rllib.policy.policy import Policy
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
    num_rollouts = 10  # Number of rollouts to collect
    rollout_length = 192  # Length of each rollout

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
    num_rollouts,
    rollout_length,
    _log
):
    """
    Main function that runs the training with a configurable reward wrapper.
    """
    # Override the create_env function in train_policy with our custom version
    global create_env
    create_env = lambda config: create_custom_env(config, reward_wrapper_class)
    
    # Run the original experiment with all the passed parameters
    result = ex.run(
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

    # Get the checkpoint path from the result
    checkpoint_path = result.info.get("checkpoint")
    if checkpoint_path is None:
        raise ValueError("No checkpoint was saved during training")

    # Load the trained algorithm
    _log.info(f"Loading trained model from {checkpoint_path}")
    algorithm = Algorithm.from_checkpoint(checkpoint_path)

    # Collect rollouts
    _log.info(f"Collecting {num_rollouts} rollouts of length {rollout_length}")
    rollouts = []
    for i in range(num_rollouts):
        _log.info(f"Collecting rollout {i+1}/{num_rollouts}")
        rollout = algorithm.compute_single_episode(
            policy_id="current" if exp_algo == "ORPO" else "default",
            episode_length=rollout_length,
            explore=False  # Set to True if you want to use exploration
        )
        rollouts.append(rollout)
        print (rollout)
        print ("\n")

    # Clean up
    algorithm.stop()
    
    # return rollouts

