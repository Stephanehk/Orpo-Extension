#!/bin/bash
cd /next/u/stephhk/orpo/
conda activate orpo


# python -m extensions.algorithms.iterative_reward_design with env_to_run=tomato level=4 reward_fun=proxy exp_algo=ORPO 'om_divergence_coeffs_1=['0.8']'  'om_divergence_coeffs_2=['0.0']' seed=0 experiment_tag=state 'om_divergence_type=["'kl'"]' num_rollout_workers=10 num_gpus=1 num_training_iters_1=300 num_training_iters_2=300
python -m extensions.algorithms.iterative_reward_design with env_to_run=tomato level=4 reward_fun=proxy exp_algo=ORPO 'om_divergence_coeffs_1=['0.8']'  'om_divergence_coeffs_2=['0.0']' 'checkpoint_to_load_policies=["'/next/u/stephhk/orpo/data/base_policy_checkpoints/tomato_base_policy/checkpoint_000003'"]' seed=0 experiment_tag=state 'om_divergence_type=["'kl'"]' num_rollout_workers=10 num_gpus=1 num_training_iters_1=300 num_training_iters_2=300
