#!/bin/bash
cd /next/u/stephhk/orpo/
conda activate orpo


python -m occupancy_measures.experiments.orpo_experiments with env_to_run=pandemic reward_fun=proxy exp_algo=ORPO om_divergence_coeffs=[0.0] seed=0 num_rollout_workers=2 num_gpus=1