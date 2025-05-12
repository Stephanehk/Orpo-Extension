import argparse
import logging
from typing import Dict, Optional

import ray
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.rllib.env.multi_agent_env import make_multi_agent
from ray.tune.registry import register_env
from pandemic_simulator.environment.pandemic_env import PandemicPolicyGymEnv
from sacred import Experiment
from sacred import SETTINGS as sacred_settings

from reward_wrapper import RewardWrapper

def create_env(config):
    base_env = PandemicPolicyGymEnv(config)
    return RewardWrapper(base_env, reward_model=config.get("reward_model", "custom"))

def train_policy(
    env_to_run: str,
    om_divergence_coeffs: Dict[str, float],
    checkpoint_to_load_policies: Optional[str] = None,
    checkpoint_to_load_current_policy: Optional[str] = None,
    seed: int = 42,
    om_divergence_type: str = "kl",
    num_workers: int = 2,
    num_gpus: int = 0,
    num_envs_per_worker: int = 1,
    train_batch_size: int = 4000,
    sgd_minibatch_size: int = 128,
    num_sgd_iter: int = 10,
    gamma: float = 0.99,
    lr: float = 5e-5,
    entropy_coeff: float = 0.01,
    clip_param: float = 0.3,
    lambda_: float = 0.95,
    num_training_iterations: int = 1000,
    checkpoint_freq: int = 10,
    checkpoint_at_end: bool = True,
    reward_model: str = "custom",
):
    """
    Train a policy with the specified parameters and reward function.
    
    Args:
        env_to_run: Name of the environment to run
        om_divergence_coeffs: Dictionary mapping policy IDs to divergence coefficients
        checkpoint_to_load_policies: Path to checkpoint for loading policies
        checkpoint_to_load_current_policy: Path to checkpoint for loading current policy
        seed: Random seed
        om_divergence_type: Type of divergence measure to use
        num_workers: Number of parallel workers
        num_gpus: Number of GPUs to use
        num_envs_per_worker: Number of environments per worker
        train_batch_size: Training batch size
        sgd_minibatch_size: SGD minibatch size
        num_sgd_iter: Number of SGD iterations per update
        gamma: Discount factor
        lr: Learning rate
        entropy_coeff: Entropy coefficient
        clip_param: PPO clip parameter
        lambda_: GAE lambda parameter
        num_training_iterations: Number of training iterations
        checkpoint_freq: Frequency of checkpoints
        checkpoint_at_end: Whether to save checkpoint at end
        reward_model: Which reward model to use ("custom" or "pandemic")
    """
    # Initialize Ray
    ray.init(ignore_reinit_error=True)

    ex = Experiment("orpo_experiments", save_git_info=False)
    sacred_settings.CONFIG.READ_ONLY_CONFIG = False
    
    # Set up environment configuration
    if env_to_run == "pandemic":
        # Import pandemic configuration
        from extensions.utils.custom_configs import pandemic_configs
        env_config = pandemic_configs()
        
        # Update environment name
        env_name = "pandemic_env_multiagent"
        register_env(
            env_name,
            make_multi_agent(lambda config: create_env(config)),
        )
    else:
        raise NotImplementedError(f"Environment {env_to_run} is not implemented yet")
    
    # Configure the algorithm
    config = PPOConfig().environment(
        env=env_name,
        env_config=env_config
    ).framework(
        framework="torch"
    ).rollouts(
        num_rollout_workers=num_workers,
        num_envs_per_worker=num_envs_per_worker,
        rollout_fragment_length=200,
    ).training(
        train_batch_size=train_batch_size,
        sgd_minibatch_size=sgd_minibatch_size,
        num_sgd_iter=num_sgd_iter,
        gamma=gamma,
        lr=lr,
        entropy_coeff=entropy_coeff,
        clip_param=clip_param,
        lambda_=lambda_,
    ).resources(
        num_gpus=num_gpus,
    )
    
    # Add reward model to env config
    config.env_config["reward_model"] = reward_model
    
    # Create the algorithm
    algo = PPO(config=config)
    
    # Load checkpoints if provided
    if checkpoint_to_load_policies:
        algo.load_checkpoint(checkpoint_to_load_policies)
    if checkpoint_to_load_current_policy:
        algo.load_checkpoint(checkpoint_to_load_current_policy)
    
    # Training loop
    for i in range(num_training_iterations):
        result = algo.train()
        
        # Print training progress
        print(f"Iteration {i+1}/{num_training_iterations}")
        print(f"Episode reward mean: {result['episode_reward_mean']}")
        print(f"Episode length mean: {result['episode_len_mean']}")
        
        # Save checkpoint if needed
        if (i + 1) % checkpoint_freq == 0:
            checkpoint = algo.save()
            print(f"Checkpoint saved at {checkpoint}")
    
    # Save final checkpoint if requested
    if checkpoint_at_end:
        final_checkpoint = algo.save()
        print(f"Final checkpoint saved at {final_checkpoint}")
    
    # Clean up
    ray.shutdown()
    return algo

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, required=True, help="Environment to run")
    parser.add_argument("--om_divergence_coeffs", type=str, required=True, 
                       help="Dictionary of divergence coefficients")
    parser.add_argument("--checkpoint_to_load_policies", type=str, default=None,
                       help="Path to checkpoint for loading policies")
    parser.add_argument("--checkpoint_to_load_current_policy", type=str, default=None,
                       help="Path to checkpoint for loading current policy")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--om_divergence_type", type=str, default="kl",
                       help="Type of divergence measure")
    parser.add_argument("--num_workers", type=int, default=2,
                       help="Number of parallel workers")
    parser.add_argument("--num_gpus", type=int, default=0,
                       help="Number of GPUs to use")
    parser.add_argument("--num_envs_per_worker", type=int, default=1,
                       help="Number of environments per worker")
    parser.add_argument("--train_batch_size", type=int, default=4000,
                       help="Training batch size")
    parser.add_argument("--sgd_minibatch_size", type=int, default=128,
                       help="SGD minibatch size")
    parser.add_argument("--num_sgd_iter", type=int, default=10,
                       help="Number of SGD iterations per update")
    parser.add_argument("--gamma", type=float, default=0.99,
                       help="Discount factor")
    parser.add_argument("--lr", type=float, default=5e-5,
                       help="Learning rate")
    parser.add_argument("--entropy_coeff", type=float, default=0.01,
                       help="Entropy coefficient")
    parser.add_argument("--clip_param", type=float, default=0.3,
                       help="PPO clip parameter")
    parser.add_argument("--lambda_", type=float, default=0.95,
                       help="GAE lambda parameter")
    parser.add_argument("--num_training_iterations", type=int, default=1000,
                       help="Number of training iterations")
    parser.add_argument("--checkpoint_freq", type=int, default=10,
                       help="Frequency of checkpoints")
    parser.add_argument("--checkpoint_at_end", action="store_true",
                       help="Save checkpoint at end")
    parser.add_argument("--reward_model", type=str, default="custom",
                       help="Which reward model to use (custom or pandemic)")
    
    args = parser.parse_args()
    
    # Convert divergence coefficients string to dict
    om_divergence_coeffs = eval(args.om_divergence_coeffs)
    
    train_policy(
        env_to_run=args.env,
        om_divergence_coeffs=om_divergence_coeffs,
        checkpoint_to_load_policies=args.checkpoint_to_load_policies,
        checkpoint_to_load_current_policy=args.checkpoint_to_load_current_policy,
        seed=args.seed,
        om_divergence_type=args.om_divergence_type,
        num_workers=args.num_workers,
        num_gpus=args.num_gpus,
        num_envs_per_worker=args.num_envs_per_worker,
        train_batch_size=args.train_batch_size,
        sgd_minibatch_size=args.sgd_minibatch_size,
        num_sgd_iter=args.num_sgd_iter,
        gamma=args.gamma,
        lr=args.lr,
        entropy_coeff=args.entropy_coeff,
        clip_param=args.clip_param,
        lambda_=args.lambda_,
        num_training_iterations=args.num_training_iterations,
        checkpoint_freq=args.checkpoint_freq,
        checkpoint_at_end=args.checkpoint_at_end,
        reward_model=args.reward_model,
    ) 