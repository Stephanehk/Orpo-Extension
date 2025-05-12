#!/bin/bash
cd /next/u/stephhk/orpo/
conda activate orpo

# python -m occupancy_measures.experiments.orpo_experiments with env_to_run=pandemic exp_algo=SafePolicyGenerationAlgorithm num_rollout_workers=20 safe_policy_action_dist_input_info_key=S0-4-0 num_training_iters=0

python -m occupancy_measures.experiments.evaluate with run=SafePolicyGenerationAlgorithm episodes=1000 "policy_ids=['safe_policy0']" num_workers=20 experiment_name=1000-episodes checkpoint=data/logs/pandemic/SafePolicyGenerationAlgorithm/true/model_128-128/seed_0/2025-05-04_16-03-31/checkpoint_000000

