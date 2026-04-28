"""
forecasters/tft_forecaster.py
─────────────────────────────────────────────────────────────────────────────
Temporal Fusion Transformer quantile forecaster for German EPEX DAM prices.

This file does three things:
  1. train_tft()   — builds TimeSeriesDataSet, trains TFT, saves model
  2. TFTForecaster — inference wrapper that plugs into Sage's make_env /
                     forecast injection mechanism
  3. extract_quantile_state() — converts TFT output to the 103-dim DQN
                                state vector extension

Architecture
────────────
  Input  : 35,040 rows × 27 features ('german_epex_dam_idm_2022_2025.csv')
  Target : pool_price (DAM_yt renamed)
  Output : p10, p50, p90 for each of next 24 hours  →  72 values
           + uncertainty band (p90-p10) per hour      →  24 values
           Total forecast contribution to state: 96 values

DQN state vector (103-dim):
  [0]     soc
  [1-24]  p10_h1 .. p10_h24
  [25-48] p50_h1 .. p50_h24
  [49-72] p90_h1 .. p90_h24
  [73-96] uncertainty_h1 .. uncertainty_h24
  [97-102] sin_hour, cos_hour, sin_weekday, cos_weekday, sin_month, cos_month

Compatibility
─────────────
  PyTorch Forecasting >= 0.9.0
  torch >= 1.10
  pandas >= 1.3
  lightning >= 2.0  (or pytorch_lightning >= 1.8)

Install:
  pip install pytorch-forecasting pytorch-lightning torch pandas openpyxl
"""

import os
import warnings
from pathlib import Path
from typing import List, Optional, Dict, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

QUANTILES       = [0.1, 0.5, 0.9]          # p10, p50, p90
FORECAST_HORIZON = 24                       # hours ahead
ENCODER_LENGTH   = 168                      # 1 week of context (168h)
TARGET_COL       = 'pool_price'             # renamed from DAM_yt
GROUP_COL        = 'group_id'
TIME_IDX_COL     = 'time_idx'

# Columns available at prediction time (future-known: calendar features)
TIME_VARYING_KNOWN = [
    'sin_hour', 'cos_hour',
    'sin_weekday', 'cos_weekday',
    'sin_month', 'cos_month',
]

# Columns that are only known up to the current time (lags, renewables, load)
TIME_VARYING_UNKNOWN = [
    'DAM_t1', 'DAM_t2', 'DAM_t3', 'DAM_t4', 'DAM_t5',
    'AvgNeighbour_t1',
    'Wind_offshore_t1', 'Wind_onshore_t1',
    'Solar_t1',
    'Net_load_t1', 'Residual_load_t1',
]


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_german_data(csv_path: str) -> pd.DataFrame:
    """
    Load and prepare German EPEX data for PyTorch Forecasting.

    Adds time_idx and group_id if not already present.
    Verifies all required columns exist.

    :param csv_path: Path to german_epex_dam_idm_2022_2025.csv
    :return: DataFrame ready for TimeSeriesDataSet
    """
    df = pd.read_csv(csv_path, index_col=0)

    # Add time_idx if missing (required by PTF)
    if TIME_IDX_COL not in df.columns:
        df[TIME_IDX_COL] = range(len(df))

    # Add group_id if missing
    if GROUP_COL not in df.columns:
        df[GROUP_COL] = 'DE_LU'

    # Verify target column
    assert TARGET_COL in df.columns, (
        f"Target column '{TARGET_COL}' not found. "
        f"Available: {df.columns.tolist()}"
    )

    # Verify feature columns
    missing = [c for c in TIME_VARYING_KNOWN + TIME_VARYING_UNKNOWN
               if c not in df.columns]
    if missing:
        print(f"WARNING: Missing feature columns: {missing}")

    # Ensure numeric dtypes
    for col in [TARGET_COL] + TIME_VARYING_KNOWN + TIME_VARYING_UNKNOWN:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df[TIME_IDX_COL] = df[TIME_IDX_COL].astype(int)

    print(f"Loaded {len(df):,} rows | "
          f"{df[TIME_IDX_COL].min()} to {df[TIME_IDX_COL].max()} | "
          f"price: [{df[TARGET_COL].min():.1f}, {df[TARGET_COL].max():.1f}] EUR/MWh")

    return df


def make_train_val_dataloaders(
    df: pd.DataFrame,
    batch_size: int = 64,
    num_workers: int = 0,
) -> Tuple:
    """
    Build PyTorch Forecasting TimeSeriesDataSet objects and dataloaders.

    Train : 2022-01-02 to 2024-12-31  (26,280 rows, time_idx 0-26279)
    Val   : 2025-01-01 to 2025-06-30  (4,343 rows,  time_idx 26280-30622)

    :return: (training_dataset, val_dataset, train_loader, val_loader)
    """
    try:
        from pytorch_forecasting import TimeSeriesDataSet
        from pytorch_forecasting.data import GroupNormalizer, EncoderNormalizer
    except ImportError:
        raise ImportError(
            "Install PyTorch Forecasting: pip install pytorch-forecasting"
        )

    # Split indices
    # Train covers first 3 years: time_idx 0 to 26279
    # Val covers 2025 H1: time_idx 26280 to 30622
    train_cutoff = 26279   # last train index
    val_cutoff   = 30622   # last val index

    training = TimeSeriesDataSet(
        df[df[TIME_IDX_COL] <= train_cutoff],
        time_idx=TIME_IDX_COL,
        target=TARGET_COL,
        group_ids=[GROUP_COL],
        # Encoder sees 168h of history (1 week)
        min_encoder_length=ENCODER_LENGTH // 2,
        max_encoder_length=ENCODER_LENGTH,
        # Decoder predicts 24h ahead
        min_prediction_length=FORECAST_HORIZON,
        max_prediction_length=FORECAST_HORIZON,
        # Calendar features — known for future hours
        time_varying_known_reals=TIME_VARYING_KNOWN,
        # Lag features — only known up to now
        time_varying_unknown_reals=TIME_VARYING_UNKNOWN + [TARGET_COL],
        # Normalise target using encoder statistics (handles regime shifts)
        target_normalizer=EncoderNormalizer(
            method='robust',            # median/IQR robust to outliers
            center=True,
        ),
        # Normalise continuous inputs
        scalers={
            col: GroupNormalizer(groups=[GROUP_COL])
            for col in TIME_VARYING_UNKNOWN
        },
        add_relative_time_idx=True,     # adds t/max_t feature automatically
        add_target_scales=True,         # adds target mean/std to encoder
        add_encoder_length=True,        # adds encoder length feature
        allow_missing_timesteps=True,   # handles DST gaps gracefully
    )

    # Validation dataset shares normalisation parameters from training
    validation = TimeSeriesDataSet.from_dataset(
        training,
        df[(df[TIME_IDX_COL] > train_cutoff) & (df[TIME_IDX_COL] <= val_cutoff)],
        predict=True,
        stop_randomization=True,
    )

    train_loader = training.to_dataloader(
        train=True,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
    )
    val_loader = validation.to_dataloader(
        train=False,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )

    print(f"Training samples:   {len(training):,}")
    print(f"Validation samples: {len(validation):,}")

    return training, validation, train_loader, val_loader


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def train_tft(
    csv_path: str,
    save_dir: str,
    max_epochs: int = 30,
    batch_size: int = 64,
    learning_rate: float = 3e-3,
    hidden_size: int = 64,
    attention_head_size: int = 4,
    dropout: float = 0.1,
    hidden_continuous_size: int = 16,
    gradient_clip_val: float = 0.1,
    num_workers: int = 0,
    accelerator: str = 'auto',     # 'gpu' on cluster, 'cpu' locally
) -> str:
    """
    Train TFT on German EPEX data and save model to disk.

    :param csv_path: Path to german_epex_dam_idm_2022_2025.csv
    :param save_dir: Directory to save trained model
    :param max_epochs: Training epochs (30 sufficient for demo, 50+ for thesis)
    :param accelerator: 'gpu' for cluster, 'cpu' for local testing
    :return: Path to saved model checkpoint
    """
    try:
        import torch
        import lightning as L
        from pytorch_forecasting import TemporalFusionTransformer
        from pytorch_forecasting.metrics import QuantileLoss
        from lightning.pytorch.callbacks import (
            EarlyStopping, LearningRateMonitor, ModelCheckpoint
        )
        from lightning.pytorch.loggers import CSVLogger
    except ImportError:
        raise ImportError(
            "Install: pip install pytorch-forecasting lightning torch"
        )

    Path(save_dir).mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("TFT TRAINING — German EPEX DAM Price Forecasting")
    print("="*60)

    # Load data
    df = load_german_data(csv_path)

    # Build dataloaders
    training, validation, train_loader, val_loader = make_train_val_dataloaders(
        df, batch_size=batch_size, num_workers=num_workers
    )

    # Build TFT model
    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=learning_rate,
        hidden_size=hidden_size,
        attention_head_size=attention_head_size,
        dropout=dropout,
        hidden_continuous_size=hidden_continuous_size,
        loss=QuantileLoss(quantiles=QUANTILES),  # p10, p50, p90
        log_interval=10,
        reduce_on_plateau_patience=4,
        optimizer='adam',
    )

    n_params = sum(p.numel() for p in tft.parameters() if p.requires_grad)
    print(f"TFT parameters: {n_params:,}")
    print(f"Forecast horizon: {FORECAST_HORIZON}h")
    print(f"Encoder length:   {ENCODER_LENGTH}h")
    print(f"Quantiles:        {QUANTILES}")
    print()

    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            mode='min',
            verbose=True,
        ),
        LearningRateMonitor(),
        ModelCheckpoint(
            dirpath=save_dir,
            filename='tft_best_{epoch:02d}_{val_loss:.4f}',
            monitor='val_loss',
            mode='min',
            save_top_k=1,
        ),
    ]

    logger = CSVLogger(save_dir=save_dir, name='tft_training')

    # Trainer
    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        gradient_clip_val=gradient_clip_val,
        callbacks=callbacks,
        logger=logger,
        enable_progress_bar=True,
        log_every_n_steps=10,
    )

    print("Training...")
    trainer.fit(
        tft,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )

    # Save best model path for later loading
    best_ckpt = trainer.checkpoint_callback.best_model_path
    print(f"\nBest checkpoint: {best_ckpt}")
    print(f"Best val_loss:   {trainer.checkpoint_callback.best_model_score:.4f}")

    # Also save as simple torch state dict for compatibility
    simple_save = os.path.join(save_dir, 'tft_model.pt')
    torch.save({
        'state_dict': tft.state_dict(),
        'hparams':    tft.hparams,
        'dataset_params': training.get_parameters(),
        'quantiles':  QUANTILES,
        'horizon':    FORECAST_HORIZON,
        'encoder_length': ENCODER_LENGTH,
    }, simple_save)
    print(f"Simple save:     {simple_save}")

    return best_ckpt


# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE WRAPPER
# ─────────────────────────────────────────────────────────────────────────────

class TFTForecaster:
    """
    Inference wrapper that plugs into Sage's forecasting pipeline.

    This class mirrors the interface of the existing CNN/LSTM forecasters
    in forecasters/trained_models/ so it can be passed via the 'forecasts'
    dict in run_dqn.py without changing make_env.

    The key method is predict(window) which returns a dict:
      {'p10': array(24,), 'p50': array(24,), 'p90': array(24,),
       'uncertainty': array(24,)}

    Usage in make_env (via forecasts dict):
      forecasts = {
          'log_folder_paths': ['forecasters/trained_models/tft_24h'],
          'path_datafile': 'data/German data/german_epex_dam_idm_2022_2025.csv',
          'forecaster_type': 'tft',
      }
    """

    def __init__(
        self,
        checkpoint_path: str,
        csv_path: str,
        device: str = 'cpu',
    ):
        """
        Load trained TFT from checkpoint.

        :param checkpoint_path: Path to .ckpt file from train_tft()
        :param csv_path: Path to german_epex_dam_idm_2022_2025.csv (for normalisation)
        :param device: 'cpu' or 'cuda'
        """
        try:
            from pytorch_forecasting import TemporalFusionTransformer
        except ImportError:
            raise ImportError(
                "Install: pip install pytorch-forecasting"
            )

        self.device = device
        self.horizon = FORECAST_HORIZON
        self.quantiles = QUANTILES

        print(f"Loading TFT from: {checkpoint_path}")
        self.model = TemporalFusionTransformer.load_from_checkpoint(
            checkpoint_path
        ).to(device).eval()

        # Load full dataset for normalisation reference
        self.df_full = load_german_data(csv_path)
        print("TFTForecaster ready.")

    def predict_from_index(
        self,
        current_time_idx: int,
    ) -> Dict[str, np.ndarray]:
        """
        Generate 24-hour quantile forecasts from current position.

        :param current_time_idx: Current time_idx value (0-based position in data)
        :return: Dict with keys p10, p50, p90, uncertainty — each array(24,)
        """
        try:
            from pytorch_forecasting import TimeSeriesDataSet
            from pytorch_forecasting.data import EncoderNormalizer, GroupNormalizer
            import torch
        except ImportError:
            raise ImportError("Install: pip install pytorch-forecasting torch")

        # Build a small window around current position for inference
        start_idx = max(0, current_time_idx - ENCODER_LENGTH)
        end_idx   = current_time_idx + FORECAST_HORIZON

        window_df = self.df_full[
            (self.df_full[TIME_IDX_COL] >= start_idx) &
            (self.df_full[TIME_IDX_COL] <= end_idx)
        ].copy()

        if len(window_df) < ENCODER_LENGTH + FORECAST_HORIZON:
            # Not enough history — return flat forecast at current price
            current_price = float(
                self.df_full.loc[
                    self.df_full[TIME_IDX_COL] == current_time_idx,
                    TARGET_COL
                ].values[0]
                if current_time_idx in self.df_full[TIME_IDX_COL].values
                else self.df_full[TARGET_COL].mean()
            )
            return self._flat_forecast(current_price)

        # Build minimal TimeSeriesDataSet for prediction
        pred_dataset = TimeSeriesDataSet(
            window_df,
            time_idx=TIME_IDX_COL,
            target=TARGET_COL,
            group_ids=[GROUP_COL],
            min_encoder_length=ENCODER_LENGTH // 2,
            max_encoder_length=ENCODER_LENGTH,
            min_prediction_length=FORECAST_HORIZON,
            max_prediction_length=FORECAST_HORIZON,
            time_varying_known_reals=TIME_VARYING_KNOWN,
            time_varying_unknown_reals=TIME_VARYING_UNKNOWN + [TARGET_COL],
            target_normalizer=EncoderNormalizer(method='robust', center=True),
            allow_missing_timesteps=True,
        )

        pred_loader = pred_dataset.to_dataloader(
            train=False, batch_size=1, num_workers=0, shuffle=False
        )

        with torch.no_grad():
            preds = self.model.predict(
                pred_loader,
                mode='quantiles',
                return_x=False,
            )

        # preds shape: (n_samples, horizon, n_quantiles)
        # Take the last prediction (most recent)
        last_pred = preds[-1].cpu().numpy()  # (horizon, 3)

        p10 = last_pred[:, 0]   # quantile 0.1
        p50 = last_pred[:, 1]   # quantile 0.5 (median)
        p90 = last_pred[:, 2]   # quantile 0.9
        unc = p90 - p10         # uncertainty band

        return {
            'p10':         p10,
            'p50':         p50,
            'p90':         p90,
            'uncertainty': unc,
        }

    def _flat_forecast(self, price: float) -> Dict[str, np.ndarray]:
        """Fallback when insufficient history is available."""
        flat = np.full(self.horizon, price, dtype=np.float32)
        return {
            'p10':         flat * 0.9,
            'p50':         flat,
            'p90':         flat * 1.1,
            'uncertainty': flat * 0.2,
        }

    def build_state_extension(
        self,
        current_time_idx: int,
    ) -> np.ndarray:
        """
        Build the 96-dimensional TFT extension to the DQN state vector.

        Returns flattened array:
          p10_h1..p10_h24 (24) +
          p50_h1..p50_h24 (24) +
          p90_h1..p90_h24 (24) +
          uncertainty_h1..uncertainty_h24 (24)
          = 96 values

        The full 103-dim state is built in the environment's _get_obs():
          [soc(1)] + [this output(96)] + [cyclic(6)] = 103

        :return: numpy array shape (96,)
        """
        forecasts = self.predict_from_index(current_time_idx)
        return np.concatenate([
            forecasts['p10'],
            forecasts['p50'],
            forecasts['p90'],
            forecasts['uncertainty'],
        ]).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# STANDALONE TRAINING SCRIPT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    """
    Run this script directly to train the TFT:
        python forecasters/tft_forecaster.py
    """
    import sys
    import os

    # Determine paths relative to repo root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root  = os.path.dirname(script_dir)

    csv_path = os.path.join(
        repo_root, 'data', 'German data', 'german_epex_dam_idm_2022_2025.csv'
    )
    save_dir = os.path.join(
        repo_root, 'forecasters', 'trained_models', 'tft_24h'
    )

    if not os.path.exists(csv_path):
        print(f"ERROR: Data file not found: {csv_path}")
        print("Place 'german_epex_dam_idm_2022_2025.csv' in data/German data/")
        sys.exit(1)

    print(f"Data:    {csv_path}")
    print(f"Output:  {save_dir}")
    print()

    # Quick demo run: 2 epochs to verify pipeline works
    # For full training: set max_epochs=50
    quick_demo = '--demo' in sys.argv

    best_ckpt = train_tft(
        csv_path=csv_path,
        save_dir=save_dir,
        max_epochs=2 if quick_demo else 30,
        batch_size=32 if quick_demo else 64,
        hidden_size=32 if quick_demo else 64,
        num_workers=0,           # 0 on Windows to avoid multiprocessing issues
        accelerator='auto',      # uses GPU if available
    )

    print(f"\nTraining complete. Best checkpoint: {best_ckpt}")

    # Quick inference test
    print("\nTesting inference...")
    forecaster = TFTForecaster(
        checkpoint_path=best_ckpt,
        csv_path=csv_path,
    )
    result = forecaster.predict_from_index(current_time_idx=500)
    print(f"p50 forecast (first 6h): {result['p50'][:6].round(2)}")
    print(f"Uncertainty  (first 6h): {result['uncertainty'][:6].round(2)}")
    state_ext = forecaster.build_state_extension(current_time_idx=500)
    print(f"State extension shape: {state_ext.shape}  (expect 96)")
    print("\nAll checks passed.")


if __name__ == '__main__':
    main()


# ─────────────────────────────────────────────────────────────────────────────
# ENVIRONMENT INTEGRATION GUIDE
# ─────────────────────────────────────────────────────────────────────────────
"""
HOW TO WIRE TFT INTO FreeBatteryEnv._get_obs()
───────────────────────────────────────────────

In environments.py, modify FreeBatteryEnv.__init__() to optionally accept
a TFTForecaster instance, and modify _get_obs() to extend the state vector.

Step 1 — Add to FreeBatteryEnv.__init__():

    def __init__(self, ..., tft_forecaster=None):
        ...
        self.tft_forecaster = tft_forecaster
        
        # Update observation space if TFT is active
        if self.tft_forecaster is not None:
            # Add 96 TFT forecast dimensions
            self.observation_space['tft_forecasts'] = spaces.Box(
                low=-1000, high=2000, shape=(96,), dtype=np.float32
            )

Step 2 — Add to FreeBatteryEnv._get_obs():

    def _get_obs(self):
        ...
        obs = {
            'soc': np.array([self.storage.soc], dtype=self.precision['float'])
        }
        for i in self.state_vars:
            obs[i] = np.array([row[i]], dtype=self.precision['float'])
        
        # Add TFT quantile forecasts if available
        if self.tft_forecaster is not None:
            tft_state = self.tft_forecaster.build_state_extension(
                current_time_idx=self.count
            )
            obs['tft_forecasts'] = tft_state
        
        self.obs = obs
        return obs, False

Step 3 — In run_dqn.py, instantiate and pass the forecaster:

    from forecasters.tft_forecaster import TFTForecaster
    
    tft = TFTForecaster(
        checkpoint_path='forecasters/trained_models/tft_24h/tft_best_*.ckpt',
        csv_path='data/German data/'german_epex_dam_idm_2022_2025.csv',
    )
    ENV_KWARGS['tft_forecaster'] = tft
"""
