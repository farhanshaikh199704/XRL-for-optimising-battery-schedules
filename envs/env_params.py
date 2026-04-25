"""
env_params.py — German EPEX DAM + IDM Energy Arbitrage
Shaikh 2025 — Master's Thesis

Experimental conditions:
  de_bat_baseline  → DQN Baseline    (no TFT, pool_price reward only)
  de_bat_tft       → DQN + TFT       (with quantile forecasts)
  de_bat_ppo       → PPO Baseline    (same env as DQN baseline)
  de_bat_ppo_tft   → PPO + TFT       (same env as DQN + TFT)

Note: PPO and DQN use identical environment configs.
      Algorithm differences are handled in train/train.py.
      Alberta config retained for reproducibility of Sage et al. 2024.
"""
import os
from utils.utilities import src_dir

# ── BATTERY DEGRADATION ───────────────────────────────────────────
dod_degr = {
    'type':          'DOD',
    'battery_capex': 250_000,   # EUR/MWh
    'k_p':           1.14,
    'N_fail_100':    3_000,
    'add_cal_age':   False,
    'battery_life':  15,
}

# ── SHARED BATTERY CONFIG ─────────────────────────────────────────
# 10 MWh / 5 MW German utility-scale BESS
# Round-trip efficiency: 0.92 × 0.92 = 84.6%
# Breakeven: 40 / 0.846 = 47.3 EUR/MWh
GERMAN_BATTERY = dict(
    total_cap=10,
    max_soc=0.9,
    min_soc=0.1,
    max_charge_rate=5.0,
    max_discharge_rate=5.0,
    charge_eff=0.92,
    discharge_eff=0.92,
    aux_equip_eff=1.0,
    self_discharge=0.0,
    init_strategy='half',
    degradation=dod_degr,
)

# ── STATE VARIABLES ───────────────────────────────────────────────
# pool_price removed from observation — used for reward only
# Requires base_envs.py _init_data() fix to preserve pool_price in data
# Requires environments.py step() fix to read pool_price from data not obs
GERMAN_STATE_VARS = [
    # DAM price lags (24h, 48h, 72h, 96h, 120h)
    'DAM_t1', 'DAM_t2', 'DAM_t3', 'DAM_t4', 'DAM_t5',
    # Neighbouring zone price lag
    'AvgNeighbour_t1',
    # IDM intraday features — lagged, no lookahead
    'IDM_t1',               # IDM price 1h ago
    'IDM_t24',              # IDM price 24h ago
    'DAM_IDM_spread_t1',    # IDM minus DAM spread 1h ago
    # Renewable generation lags
    'Wind_offshore_t1',
    'Wind_onshore_t1',
    'Solar_t1',
    # Load lags
    'Net_load_t1',
    'Residual_load_t1',
    # Cyclic time encodings
    'sin_hour',   'cos_hour',
    'sin_weekday','cos_weekday',
    'sin_month',  'cos_month',
]
# 20 market features + soc = 21 baseline state dims
# With TFT (p10/p50/p90/uncertainty × 24h): 21 + 96 = 117 dims

GERMAN_DATA = os.path.join(
    src_dir, 'data', 'German data',
    'german_epex_dam_idm_2022_2025.csv'
)

TFT_CKPT = os.path.join(
    src_dir, 'forecasters', 'trained_models', 'tft_24h',
    'tft_best.ckpt'
)

# ── CONDITION 1: DQN BASELINE ─────────────────────────────────────
# No TFT. Agent uses DAM lags + IDM signals + cyclic time features.
# pool_price drives reward but is not in observation.
# State: 21 dims
de_bat_baseline = {
    'env_name':          'de_bat_baseline',
    'data_file':         GERMAN_DATA,
    'state_vars':        GERMAN_STATE_VARS,
    'grid':              dict(demand_profile=None),
    'storage':           dict(**GERMAN_BATTERY),
    'resolution_h':      1.0,
    'modeling_period_h': 8760,
    'tft_24h':    None,
}

# ── CONDITION 2: DQN + TFT ────────────────────────────────────────
# TFT provides p10/p50/p90/uncertainty for 24 delivery hours.
# 96 TFT features injected into state at runtime via tft_forecaster.py.
# TFT p50 used as implicit bid price in bid acceptance check.
# State: 21 + 96 = 117 dims
de_bat_tft = {
    'env_name':          'de_bat_tft',
    'data_file':         GERMAN_DATA,
    'state_vars':        GERMAN_STATE_VARS,
    'grid':              dict(demand_profile=None),
    'storage':           dict(**GERMAN_BATTERY),
    'resolution_h':      1.0,
    'modeling_period_h': 8760,
    'tft_24h':    TFT_CKPT,
}

# ── CONDITION 3: PPO BASELINE ─────────────────────────────────────
# Identical environment to DQN baseline.
# Algorithm difference handled in train/train.py.
# PPO uses continuous action space — environments.py clips to [-1, 1].
# State: 21 dims
de_bat_ppo = {
    'env_name':          'de_bat_ppo',
    'data_file':         GERMAN_DATA,
    'state_vars':        GERMAN_STATE_VARS,
    'grid':              dict(demand_profile=None),
    'storage':           dict(**GERMAN_BATTERY),
    'resolution_h':      1.0,
    'modeling_period_h': 8760,
    'tft_24h':    None,
}

# ── CONDITION 4: PPO + TFT ────────────────────────────────────────
# Identical environment to DQN + TFT.
# Algorithm difference handled in train/train.py.
# State: 21 + 96 = 117 dims
de_bat_ppo_tft = {
    'env_name':          'de_bat_ppo_tft',
    'data_file':         GERMAN_DATA,
    'state_vars':        GERMAN_STATE_VARS,
    'grid':              dict(demand_profile=None),
    'storage':           dict(**GERMAN_BATTERY),
    'resolution_h':      1.0,
    'modeling_period_h': 8760,
    'tft_24h':    TFT_CKPT,
}

# ── ALBERTA BASELINE (Sage et al. 2024 — reference only) ──────────
# Original environment from arXiv:2410.20005.
# Retained for reproducibility. Not used in Shaikh 2025 experiments.
al4_bat_ea = {
    'env_name':   'al4_bat_ea',
    'data_file':  os.path.join(src_dir, 'data', 'alberta3',
                               'alberta_2022_electricity_final.csv'),
    'state_vars': ['pool_price'],
    'grid':       dict(demand_profile=None),
    'storage':    dict(
        total_cap=10, max_soc=0.8, min_soc=0.2,
        max_charge_rate=2.5, max_discharge_rate=2.5,
        charge_eff=0.92, discharge_eff=0.92,
        aux_equip_eff=1.0, self_discharge=0.0,
        init_strategy='half',
        degradation={
            'type': 'DOD', 'battery_capex': 300_000,
            'k_p': 1.14, 'N_fail_100': 6_000,
            'add_cal_age': False, 'battery_life': 20,
        },
    ),
    'resolution_h':      1.0,
    'modeling_period_h': 8760,
}