#!/bin/bash
cd /next/u/stephhk/orpo/
conda activate orpo


python -m occupancy_measures.experiments.orpo_experiments with env_to_run=glucose reward_fun=proxy exp_algo=ORPO 'om_divergence_coeffs=['0.05']' 'checkpoint_to_load_policies=["'data/base_policy_checkpoints/glucose_base_policy/checkpoint_000300'"]' checkpoint_to_load_current_policy=data/base_policy_checkpoints/glucose_base_policy/checkpoint_000300 seed=0 experiment_tag=state-action 'om_divergence_type=["'kl'"]' num_rollout_workers=20 num_gpus=1