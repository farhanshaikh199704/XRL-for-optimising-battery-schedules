"""
run_dqn.py — DQN training entry point.

Hyperparameters sourced from Table C.7 (DE-LF/REU best) in:
  Sage & Zhao (2025), Journal of Energy Storage 115, 115428.
  "Deep reinforcement learning for economic battery dispatch:
   A comprehensive comparison of algorithms and experiment design choices"

Changes from previous version vs paper Table C.7 (DE-LF/REU):
  learning_rate      : 1.767e-3 -> 4.8e-4  (linear schedule)
  tau                : 0.5016   -> 0.435965
  gamma              : 0.99998  -> 0.975431
  learning_starts    : 255      -> 2429
  train_freq         : 84       -> 148
  exploration_final_eps: 0.005  -> 0.01
  max_grad_norm      : 3.266    -> 0.467447
  soc_penalty        : not set  -> 250     (low SOC reward wrapper)

Parameters retained from current codebase (not in paper Table C.7):
  5 discrete actions [-1, -0.5, 0, 0.5, +1] — finer than paper's 17-act
  (paper used 17 actions for DE; 5 is your thesis design choice)

Note on gamma: paper's best gamma for DE-LF/REU is 0.975431, much lower
than Alberta (0.950365). This reflects a shorter effective planning horizon
on the German task. Your energy arbitrage task is closer to AB-EA, so
consider testing both 0.975 and 0.999 and reporting which converges better.
"""
import os
import json
from typing import Any

import numpy as np

from envs.environments import FreeBatteryEnv
from envs.env_params import de_bat_baseline, de_bat_tft
from utils.net_design import activation_fn_dict, net_arch_dict
from utils.scheduler import linear_schedule
from utils.utilities import create_stats_file, src_dir
from train.train import train_rl_agent

# ── ENVIRONMENT ───────────────────────────────────────────────────────────────
ENV = FreeBatteryEnv
ENV_KWARGS = de_bat_baseline  # swap to de_bat_tft for TFT condition

# ── LOGGING ───────────────────────────────────────────────────────────────────
CREATE_LOG = True
VERBOSE = 0
LOGGER_TYPE = ['csv']
RUN_NAME = 'de_baseline_dqn_01'  # replace input() for Colab compatibility
SAVE_PATH = (
    os.path.join(src_dir, 'log', ENV_KWARGS['env_name'], 'dqn', 'run', RUN_NAME)
    if CREATE_LOG else None
)

# ── DISCRETE ACTIONS ──────────────────────────────────────────────────────────
# 5-action discretisation: [-1, -0.5, 0, +0.5, +1] × max power rate
# Paper (Table C.7 DE-LF/REU) used 17 actions; 5 is the thesis design choice.
DISCRETE_ACTIONS = [
    np.array([-1.0]),
    np.array([-0.5]),
    np.array([0.0]),
    np.array([0.5]),
    np.array([1.0]),
]

# ── EXPERIMENT PARAMS ─────────────────────────────────────────────────────────
EXP_PARAMS = {
    'n_runs': 5,
    'n_episodes': 50,
    'seed': 22,
    # Environment
    'flatten_obs': True,
    # Normalisation
    'norm_obs': True,
    'norm_reward': True,
    # Evaluation
    'eval_while_training': True,
    'eval_freq': 8760 * 1,
    # Forecasts (TFT wired via env_kwargs, not here)
    'perfect_forecasts': None,
    'forecasts': None,
    # Rule-based demonstrations (DQNfD — Algorithm 1, Sage et al. 2023)
    'use_demonstrations': True,
    'demo_breakeven': 47.3,   # EUR/MWh — matches GERMAN_BATTERY capex/efficiency
}

# ── DQN HYPERPARAMETERS ───────────────────────────────────────────────────────
# Source: Table C.7, best on DE-LF/REU (Sage & Zhao, J. Energy Storage 2025)
RL_PARAMS: dict[str, Any] = {
    'policy': 'MlpPolicy' if EXP_PARAMS['flatten_obs'] else 'MultiInputPolicy',

    # Learning rate — linear schedule from 4.8e-4 to 0
    # Paper range: 1e-2 to 1e-5. DE best: 4.8e-4* (asterisk = scheduler)
    'learning_rate': linear_schedule(4.8e-4),

    # Replay buffer
    'buffer_size': 500_000,         # paper range: 100k–500k; best: 500k

    # Batch and update timing
    'batch_size': 256,              # paper range: 16–256; best DE: 256
    'learning_starts': 2429,        # paper range: 100–10k; best DE: 2429
    'train_freq': 148,              # paper range: 50–200; best DE: 148
    'gradient_steps': -1,           # best both case studies: -1

    # Target network
    'tau': 0.435965,                # paper range: 0.001–1; best DE: 0.435965
    'target_update_interval': 10_000,  # best DE: 10,000

    # Discount — paper best DE: 0.975431 (shorter planning horizon than AB)
    # AB-EA best was 0.950365. For pure energy arbitrage consider testing both.
    'gamma': 0.975431,

    # Exploration — epsilon-greedy annealing
    'exploration_fraction': 0.5,    # paper best both: 0.5
    'exploration_initial_eps': 1.0,  # paper best both: 1.0
    'exploration_final_eps': 0.01,  # paper best DE: 0.01 (was 0.005 before)

    # Gradient clipping — paper best DE: 0.467 (was 3.266 before — too high)
    'max_grad_norm': 0.467447,

    # Network architecture — extra_large [256, 512, 512, 256], LeakyReLU
    # Paper best on both case studies: [256, 512, 512, 256] + leaky_relu
    'policy_kwargs': {
        'net_arch': 'extra_large',
        'activation_fn': 'leaky_relu',
    },
}

# ── SOC PENALTY (reward shaping) ─────────────────────────────────────────────
# Paper Table C.7 DE-LF/REU best: soc_penalty weight = 250
# Applied as: reward -= soc_penalty_weight * (max_soc - current_soc)
# at every step during training only (not evaluation).
# Implementation note: this requires a reward wrapper — add to make_env.py
# or apply directly in FreeBatteryEnv.step() with a training flag.
# For now, documented here for reference — wire into wrappers.py when ready.
SOC_PENALTY_WEIGHT = 250  # EUR equivalent weight; paper range 0–1000

# ── SAVE CONFIG ───────────────────────────────────────────────────────────────
if SAVE_PATH is not None:
    os.makedirs(SAVE_PATH, exist_ok=True)
    with open(os.path.join(SAVE_PATH, 'inputs.json'), 'w') as f:
        json.dump({
            'DISCRETE_ACTIONS': str(DISCRETE_ACTIONS),
            'EXP_PARAMS': EXP_PARAMS,
            'RL_PARAMS': str(RL_PARAMS),
            'PLANT_PARAMS': ENV_KWARGS,
            'SOC_PENALTY_WEIGHT': SOC_PENALTY_WEIGHT,
            'source': 'Sage & Zhao (2025) J. Energy Storage Table C.7 DE-LF/REU',
        }, f)  # noqa: E501

# ── RESOLVE NET ARCH AND ACTIVATION ──────────────────────────────────────────
RL_PARAMS['policy_kwargs']['net_arch'] = (
    net_arch_dict[RL_PARAMS['policy_kwargs']['net_arch']]['qf']
)
RL_PARAMS['policy_kwargs']['activation_fn'] = (
    activation_fn_dict[RL_PARAMS['policy_kwargs']['activation_fn']]
)

# ── TRAINING LOOP ─────────────────────────────────────────────────────────────
for run in range(EXP_PARAMS['n_runs']):
    train_rl_agent(
        agent='dqn',
        run=run,
        path=SAVE_PATH,
        exp_params=EXP_PARAMS,
        env_id=ENV,
        env_kwargs=ENV_KWARGS,
        discrete_actions=DISCRETE_ACTIONS,
        rl_params=RL_PARAMS,
        verbose=VERBOSE,
        logger_type=LOGGER_TYPE,
    )

# ── AGGREGATE STATISTICS ──────────────────────────────────────────────────────
if CREATE_LOG and EXP_PARAMS['n_runs'] > 1:
    create_stats_file(SAVE_PATH, EXP_PARAMS)
