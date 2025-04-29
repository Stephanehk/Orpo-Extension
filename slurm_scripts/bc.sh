#!/bin/bash
cd /next/u/stephhk/orpo/
conda activate orpo

python -m occupancy_measures.experiments.orpo_experiments with env_to_run=pandemic exp_algo=BC save_freq=5 evaluation_num_workers=2 evaluation_interval=5 evaluation_duration=20 input=/next/u/stephhk/orpo/data/logs/pandemic/SafePolicyGenerationAlgorithm/true/model_128-128/seed_0/2025-04-26_20-05-43/rollouts_1000-episodes_2025-04-26_20-08-03_safe_policy0 entropy_coeff=0.1
