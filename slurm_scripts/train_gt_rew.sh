#!/bin/bash
cd /next/u/stephhk/orpo/
conda activate orpo


python -m occupancy_measures.experiments.orpo_experiments with env_to_run=pandemic checkpoint_to_load_current_policy=data/logs/pandemic/BC/true/model_128-128/seed_0/2025-04-28_09-41-36/checkpoint_000260 reward_fun=true exp_algo=ORPO 'om_divergence_coeffs=['0']' seed=0 experiment_tag=state-action num_rollout_workers=2 num_gpus=1