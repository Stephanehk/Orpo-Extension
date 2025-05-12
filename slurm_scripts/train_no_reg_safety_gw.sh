#!/bin/bash
cd /next/u/stephhk/orpo/
conda activate orpo

#checkpoint_to_load_current_policy=/next/u/stephhk/orpo/data/base_policy_checkpoints/tomato_base_policy/checkpoint_000003

python -m occupancy_measures.experiments.orpo_experiments with env_to_run=tomato level=4 reward_fun=proxy exp_algo=ORPO om_divergence_coeffs=[0.0] seed=0 num_rollout_workers=10 num_gpus=1