#!/bin/bash
cd /next/u/stephhk/orpo/
conda activate orpo


python -m occupancy_measures.experiments.orpo_experiments with env_to_run=pandemic reward_fun=proxy exp_algo=ORPO 'om_divergence_coeffs=['0.06']' 'checkpoint_to_load_policies=["'data/base_policy_checkpoints/pandemic_base_policy/checkpoint_000100'"]' checkpoint_to_load_current_policy=data/base_policy_checkpoints/pandemic_base_policy/checkpoint_000100 seed=0 experiment_tag=state 'om_divergence_type=["'kl'"]' num_rollout_workers=2 num_gpus=1