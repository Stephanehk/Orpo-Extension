#!/bin/bash
cd /next/u/stephhk/orpo/
conda activate orpo


python -m occupancy_measures.experiments.orpo_experiments with env_to_run=glucose checkpoint_to_load_current_policy=data/base_policy_checkpoints/glucose_base_policy/checkpoint_000300 reward_fun=proxy exp_algo=ORPO om_divergence_coeffs=[0.0] seed=0 num_rollout_workers=10 num_gpus=1