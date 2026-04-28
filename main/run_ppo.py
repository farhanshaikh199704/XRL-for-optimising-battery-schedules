"""
run_ppo.py — PPO training entry point.

Hyperparameters sourced from Table C.6 (DE-LF/REU best) in:
  Sage & Zhao (2025), Journal of Energy Storage 115, 115428.
  "Deep reinforcement learning for economic battery dispatch:
   A comprehensive comparison of algorithms and experiment design choices"

Changes from previous version vs paper Table C.6 (DE-LF/REU):
  ENV_KWARGS         : al4_bat_ea    -> de_bat_ppo   (FIXED — was Alberta)
  forecasts          : tft_checkpoint -> None         (FIXED — use env_kwargs)
  learning_rate      : 1.837e-4      -> 1.1e-5  (linear schedule)
  n_epochs           : 20            -> 5
  gamma              : 0.99998       -> 0.900887
  gae_lambda         : 0.9361        -> 0.985039
  clip_range         : 0.0918        -> 0.176607 (linear schedule)
  normalize_advantage: True          -> False
  ent_coef           : 0.2869        -> 0.969954
  vf_coef            : 0.7889        -> 0.696368
  max_grad_norm      : 0.8844        -> 0.411048
  net_arch           : extra_large   -> small [128, 128]
  ortho_init         : True          -> False
  squash_output      : False         -> True
  Reward penalties   : not set       -> inactivity(patience=143, w=28),
                                        action_correction(w=255), soc(w=476)

Note on network size: paper found [128, 128] beat extra_large for PPO on
the German task. PPO as an on-policy algorithm is more sample-limited than
DQN — smaller networks generalise better under limited data per update.

Note on gamma: 0.900887 is much lower than the DQN value (0.975431).
On-policy PPO discounts more aggressively; this reflects the shorter
effective rollout used per update (n_steps=1024) vs DQN's replay horizon.

Note on reward penalties: the paper implemented these inside wrappers.
For now they are documented here — wire into make_env.py when needed.
"""
import os
import json
from typing import Any

from envs.environments import FreeBatteryEnv
from envs.env_params import de_bat_ppo, de_bat_ppo_tft
from utils.net_design import activation_fn_dict, net_arch_dict
from utils.scheduler import linear_schedule
from utils.utilities import create_stats_file, src_dir
from train.train import train_rl_agent

# ── ENVIRONMENT ───────────────────────────────────────────────────────────────
ENV = FreeBatteryEnv
ENV_KWARGS = de_bat_ppo  # swap to de_bat_ppo_tft for TFT condition
# FIXED: was al4_bat_ea (Alberta) — now correctly points to German env

# ── LOGGING ───────────────────────────────────────────────────────────────────
CREATE_LOG = True
VERBOSE = 0
LOGGER_TYPE = ['csv']
RUN_NAME = 'de_baseline_ppo_01'  # replace input() for Colab compatibility
SAVE_PATH = (
    os.path.join(src_dir, 'log', ENV_KWARGS['env_name'], 'ppo', 'run', RUN_NAME)
    if CREATE_LOG else None
)

# ── ACTIONS ───────────────────────────────────────────────────────────────────
# PPO uses continuous action space — no discretisation needed
DISCRETE_ACTIONS = None

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
    # Forecasts — TFT wired via ENV_KWARGS['tft_24h'], never via forecasts=
    'perfect_forecasts': None,
    'forecasts': None,
    # Demonstrations — PPO is on-policy, no replay buffer, not applicable
    'use_demonstrations': False,
}

# ── PPO HYPERPARAMETERS ───────────────────────────────────────────────────────
# Source: Table C.6, best on DE-LF/REU (Sage & Zhao, J. Energy Storage 2025)
RL_PARAMS: dict[str, Any] = {
    'policy': 'MlpPolicy' if EXP_PARAMS['flatten_obs'] else 'MultiInputPolicy',

    # Learning rate — linear schedule from 1.1e-5 to 0
    # Paper range: 1e-2 to 1e-5. DE best: 1.1e-5* (asterisk = scheduler)
    'learning_rate': linear_schedule(1.1e-5),

    # Rollout and update
    'n_steps': 1024,            # paper range: 256–1024; best DE: 1024
    'batch_size': 128,          # paper range: 64–512; best DE: 128
    'n_epochs': 5,              # paper range: 5–20; best DE: 5 (was 20)

    # Discount and advantage
    'gamma': 0.900887,          # paper range: 0.9–0.99999; best DE: 0.900887
    'gae_lambda': 0.985039,     # paper range: 0.9–0.9999; best DE: 0.985039

    # Clipping — linear schedule from 0.176607 to 0
    # Paper range: 0.05–0.3. DE best: 0.176607* (asterisk = scheduler)
    'clip_range': linear_schedule(0.176607),
    'clip_range_vf': None,      # paper best DE: None

    # Advantage normalisation — paper best DE: False (was True before)
    'normalize_advantage': False,

    # Entropy and value function coefficients
    'ent_coef': 0.969954,       # paper range: 0–1; best DE: 0.969954 (high)
    'vf_coef': 0.696368,        # paper range: 0.1–1; best DE: 0.696368

    # Gradient clipping — paper best DE: 0.411 (tighter than current 0.884)
    'max_grad_norm': 0.411048,

    # State-dependent exploration — not used in paper
    'use_sde': False,
    'target_kl': None,

    # Network architecture — paper best DE: [128, 128] tanh
    # Smaller than DQN — PPO is on-policy and benefits from smaller networks
    'policy_kwargs': {
        'net_arch': 'small',            # [128, 128] — paper best DE
        'activation_fn': 'tanh',        # paper best DE: tanh
        'ortho_init': False,            # paper best DE: False (was True)
        'squash_output': True,          # paper best DE: True (was False)
        'share_features_extractor': True,  # paper best DE: True
    },
}

# ── REWARD PENALTIES (paper Table C.6 DE-LF/REU best) ────────────────────────
# Paper found all three penalties improved PPO on the German task:
#   inactivity penalty  : patience=143 steps, weight=28
#   action correction   : weight=255
#   low SOC             : weight=476
# Implementation: requires reward wrappers in make_env.py or wrappers.py.
# Documented here for reference — wire into reward shaping when ready.
REWARD_PENALTIES = {
    'inactivity_patience': 143,
    'inactivity_weight': 28,
    'action_correction_weight': 255,
    'soc_penalty_weight': 476,
}

# ── SAVE CONFIG ───────────────────────────────────────────────────────────────
if SAVE_PATH is not None:
    os.makedirs(SAVE_PATH, exist_ok=True)
    with open(os.path.join(SAVE_PATH, 'inputs.json'), 'w') as f:
        json.dump({
            'DISCRETE_ACTIONS': str(DISCRETE_ACTIONS),
            'EXP_PARAMS': EXP_PARAMS,
            'RL_PARAMS': str(RL_PARAMS),
            'PLANT_PARAMS': ENV_KWARGS,
            'REWARD_PENALTIES': REWARD_PENALTIES,
            'source': 'Sage & Zhao (2025) J. Energy Storage Table C.6 DE-LF/REU',
        }, f)

# ── RESOLVE NET ARCH AND ACTIVATION ──────────────────────────────────────────
RL_PARAMS['policy_kwargs']['net_arch'] = (
    net_arch_dict[RL_PARAMS['policy_kwargs']['net_arch']]
)
RL_PARAMS['policy_kwargs']['activation_fn'] = (
    activation_fn_dict[RL_PARAMS['policy_kwargs']['activation_fn']]
)

# ── TRAINING LOOP ─────────────────────────────────────────────────────────────
for run in range(EXP_PARAMS['n_runs']):
    train_rl_agent(
        agent='ppo',
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
