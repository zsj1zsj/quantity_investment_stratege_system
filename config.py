from pathlib import Path

# Paths
PROJECT_DIR = Path(__file__).parent
DATA_CACHE_DIR = PROJECT_DIR / "data" / "cache"
MODEL_SAVE_DIR = PROJECT_DIR / "model" / "saved"

# Symbols — indices (original)
SYMBOLS = {
    "SP500": "^GSPC",
    "NASDAQ": "^IXIC",
}

# Sector ETFs for multi-asset portfolio
ETF_SYMBOLS = {
    "XLK": "XLK",   # Technology
    "XLF": "XLF",   # Financials
    "XLV": "XLV",   # Healthcare
    "XLI": "XLI",   # Industrials
    "XLC": "XLC",   # Communication Services (started 2018-06)
    "XLY": "XLY",   # Consumer Discretionary
    "XLP": "XLP",   # Consumer Staples
}
# Note: XLE (Energy) excluded — DD=33%, SR=-0.30, degrades portfolio

# All tradeable symbols (indices + ETFs)
ALL_SYMBOLS = {**SYMBOLS, **ETF_SYMBOLS}

# Data
DATA_START_DATE = "2010-01-01"

# Cross-market symbols (Phase 2)
CROSS_MARKET_SYMBOLS = {
    "VIX": "^VIX",
    "TNX": "^TNX",  # 10-year Treasury yield
}

# Feature parameters
MA_WINDOWS = [5, 20]
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
VOLATILITY_WINDOWS = [5, 20]
VOLUME_AVG_WINDOW = 5
RETURN_WINDOWS = [5, 10]

# Label
FORWARD_DAYS = 20
LABEL_THRESHOLD = 0.03  # 3% - predict if 20-day return > 3%

# Model - sliding window (trading days)
TRAIN_WINDOW = 500
TEST_WINDOW = 60
STEP_SIZE = 60

# LightGBM hyperparameters
LGBM_PARAMS = {
    "max_depth": 4,
    "n_estimators": 200,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "verbose": -1,
}

# Transaction cost model
COMMISSION_RATE = 0.0005   # 0.05% per side
SLIPPAGE_RATE = 0.0005     # 0.05% per side
COST_PER_SIDE = COMMISSION_RATE + SLIPPAGE_RATE  # 0.1%
COST_ROUND_TRIP = 2 * COST_PER_SIDE              # 0.2%

# Strategy thresholds
PROB_BUY_THRESHOLD = 0.5
PROB_SELL_THRESHOLD = 0.3
STOP_LOSS_PCT = 0.10       # -10% stop loss
HOLD_PERIOD = 25           # fixed holding period (slightly > FORWARD_DAYS for momentum)
REQUIRE_TREND_UP = False   # Regime detection handles downside protection instead

# Position sizing
POSITION_HIGH_CONF = 1.0   # 100% position when prob > 0.8
POSITION_MED_CONF = 0.8    # 80% position when prob > 0.5

# Multi-asset portfolio
MULTI_ASSET_MODE = True        # combine all available assets into one portfolio
PER_ASSET_MAX_POSITION = 0.3   # max 30% of portfolio per asset
MAX_TOTAL_EXPOSURE = 0.9       # max 90% total portfolio exposure across all assets

# Gradual position building (Phase 2)
GRADUAL_ENTRY = False       # disabled: full entry performs better with regime protection
GRADUAL_INITIAL_SIZE = 0.5
GRADUAL_ADD_DAY = 5
GRADUAL_ADD_THRESHOLD = 0.4

# Signal calibration
SIGNAL_SMOOTH_SPAN = 3     # EWM smoothing span (days)

# Model retraining
RETRAIN_INTERVAL = 60      # retrain every 60 trading days
DEGRADATION_WINDOW = 20    # check last 20 signals for degradation
DEGRADATION_MIN_WINRATE = 0.55  # min winrate for prob>0.7 signals

# Risk-free rate for Sharpe calculation
RISK_FREE_RATE = 0.04      # 4% annual
