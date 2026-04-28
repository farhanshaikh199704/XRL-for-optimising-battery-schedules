"""
train.py  —  Training loop for RL agents.

Changes from original:
  - best_model_save_path now saves model  (was None — model was never saved)
  - total_timesteps computed from actual episode length in data
  - model.save() called explicitly at end of training for demo use
"""
import os
import time
import json
from typing import Optional

from stable_baselines3 import PPO, SAC, DDPG, DQN, A2C
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure

from utils.callbacks import ProgressBarManager
from utils.make_env import make_env
from utils.utilities import get_env_log_data


AGENTS = {
    'ppo':  PPO,
    'sac':  SAC,
    'a2c':  A2C,
    'dqn':  DQN,
    'ddpg': DDPG,
}


def train_rl_agent(
        agent: str,
        run: int,
        path: Optional[str],
        exp_params: dict,
        env_id,
        env_kwargs: dict,
        rl_params: dict,
        verbose: int = 0,
        discrete_actions: Optional[list] = None,
        logger_type: Optional[list] = None,
):
    """
    Trains a reinforcement learning agent.

    Changes from original Sage version:
      - model saved via best_model_save_path AND explicit model.save()
      - total_timesteps uses modeling_period_h from env_kwargs if available
    """
    agent_cls = AGENTS[agent]

    if logger_type is None:
        logger_type = ['csv']

    # Use actual modeling period from env config so German 8760 ≠ Alberta 8760
    period_h = env_kwargs.get('modeling_period_h', 8760)
    tt       = int(exp_params['n_episodes'] * period_h)

    run_path = os.path.join(path, f'run_{run}') if path is not None else None
    if path is not None:
        os.makedirs(run_path, exist_ok=True)

    start = time.time()
    seed  = exp_params['seed'] + run
    print(f'|| Run #{run} | Seed #{seed} | Total timesteps: {tt:,} ||')

    # ── CREATE ENVIRONMENT ────────────────────────────────────────────────────
    env = make_env(
        env=env_id,
        env_kwargs=env_kwargs,
        path=os.path.join(run_path, 'train_monitor.csv') if path is not None else None,
        perfect_forecasts=exp_params['perfect_forecasts'],
        forecasts=exp_params['forecasts'],
        flatten_obs=exp_params['flatten_obs'],
        discrete_actions=discrete_actions,
        norm_obs=exp_params['norm_obs'],
        norm_reward=exp_params['norm_reward'],
        gamma=rl_params['gamma'],
    )

    # ── DEFINE MODEL ──────────────────────────────────────────────────────────
    model = agent_cls(env=env, verbose=verbose, seed=seed, **rl_params)
    if path is not None:
        logger = configure(run_path, logger_type)
        model.set_logger(logger)

    with ProgressBarManager(total_timesteps=tt) as callback:
        if exp_params['eval_while_training']:
            eval_env = make_env(
                env=env_id,
                env_kwargs=env_kwargs,
                path=os.path.join(run_path, 'eval_monitor.csv') if path is not None else None,
                perfect_forecasts=exp_params['perfect_forecasts'],
                forecasts=exp_params['forecasts'],
                flatten_obs=exp_params['flatten_obs'],
                discrete_actions=discrete_actions,
                norm_obs=exp_params['norm_obs'],
                norm_reward=exp_params['norm_reward'],
                gamma=rl_params['gamma'],
            )
            eval_callback = EvalCallback(
                eval_env=eval_env,
                n_eval_episodes=1,
                eval_freq=exp_params['eval_freq'],
                deterministic=True,
                best_model_save_path=run_path,   # ← FIXED: now saves best model
                verbose=verbose,
            )
            model.learn(total_timesteps=tt, callback=[eval_callback, callback])
        else:
            model.learn(total_timesteps=tt, callback=callback)

    # ── SAVE FINAL MODEL ──────────────────────────────────────────────────────
    if run_path is not None:
        model.save(os.path.join(run_path, 'model_final'))   # ← explicit final save
        print(f'  Model saved: {run_path}/model_final.zip')
        print(f'  Best model:  {run_path}/best_model.zip')

    # ── EVALUATE ──────────────────────────────────────────────────────────────
    env.training   = False
    env.norm_reward = False
    # Start tracking for evaluation episode
    try:
        env.unwrapped.envs[0].unwrapped.start_tracking()
    except AttributeError:
        # Not wrapped — call directly
        env.unwrapped.start_tracking()

    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=1)
    print(f'mean_reward: {mean_reward:.2f} +/- {std_reward:.2f}')

    if run_path is not None:
        log_data = get_env_log_data(env=env, mean_reward=mean_reward, start_time=start)
        with open(os.path.join(run_path, 'output.json'), 'w') as f:
            json.dump(log_data, f, indent=2)

    env.close()
    print(f'Execution time = {time.time() - start:.1f}s\n')
