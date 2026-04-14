import sys


def cmd_fetch():
    from data.fetcher import fetch_all
    from data.store import save

    data = fetch_all()
    for key, df in data.items():
        save(key, df)
    print("\nDone. Data fetched and cached.")


def cmd_train():
    from model.train import train_all
    from model.evaluate import print_summary

    results = train_all()
    print_summary(results)


def cmd_predict():
    from model.predict import predict_all
    from output.report import format_report

    predictions = predict_all()
    report = format_report(predictions)
    print(report)


def cmd_evaluate():
    """Re-run training with evaluation output (same as train)."""
    cmd_train()


def cmd_backtest():
    """Run walk-forward backtest with transaction costs."""
    from model.train import prepare_data
    from backtest.engine import run_backtest, print_backtest_report
    from backtest.multi_asset import run_multi_asset_backtest
    from config import SYMBOLS, MULTI_ASSET_MODE

    # Per-asset backtests
    prepared = {}
    for key in SYMBOLS:
        print(f"\nPreparing data for {key}...")
        df = prepare_data(key)
        prepared[key] = df
        result = run_backtest(df, key)
        print_backtest_report(result)

    # Add ETFs if data is available
    from data.store import has_cache
    from config import ETF_SYMBOLS
    for key in ETF_SYMBOLS:
        if has_cache(key):
            print(f"\nPreparing data for {key} (ETF)...")
            try:
                df = prepare_data(key)
                prepared[key] = df
                result = run_backtest(df, key)
                print_backtest_report(result)
            except Exception as e:
                print(f"  Skipping {key}: {e}")

    # Multi-asset combined backtest
    if MULTI_ASSET_MODE and len(prepared) > 1:
        print(f"\n{'='*60}")
        print(f"  Multi-Asset Portfolio Backtest ({len(prepared)} assets)")
        print(f"{'='*60}")
        multi_result = run_multi_asset_backtest(prepared)
        print_backtest_report(multi_result)


def cmd_validate_signal():
    """Run signal quality validation (bucket analysis)."""
    from model.train import prepare_data
    from backtest.signal_validation import validate_signals, print_signal_validation
    from config import SYMBOLS

    for key in SYMBOLS:
        print(f"\nValidating signals for {key}...")
        df = prepare_data(key)
        result = validate_signals(df, key)
        print_signal_validation(result)


def cmd_sector_analysis():
    """Run per-asset performance analysis across all available assets."""
    from model.train import prepare_data
    from backtest.sector_analysis import run_sector_analysis, print_sector_analysis
    from config import SYMBOLS, ETF_SYMBOLS
    from data.store import has_cache

    print("Preparing data...")
    prepared = {}
    for key in SYMBOLS:
        prepared[key] = prepare_data(key)
    for key in ETF_SYMBOLS:
        if has_cache(key):
            try:
                prepared[key] = prepare_data(key)
                print(f"  Loaded {key}")
            except Exception:
                pass
    print(f"  Total assets: {len(prepared)}")

    print("\nRunning per-asset walk-forward backtests...")
    results = run_sector_analysis(prepared)
    print_sector_analysis(results)


def cmd_holdout():
    """Run hold-out validation: compare in-sample vs post-2024 performance."""
    from model.train import prepare_data
    from backtest.signals import collect_all_signals
    from backtest.stability import holdout_analysis, print_holdout_analysis
    from config import SYMBOLS, ETF_SYMBOLS
    from data.store import has_cache

    print("Preparing data...")
    prepared = {}
    for key in SYMBOLS:
        prepared[key] = prepare_data(key)
    for key in ETF_SYMBOLS:
        if has_cache(key):
            try:
                prepared[key] = prepare_data(key)
            except Exception:
                pass
    print(f"  Total assets: {len(prepared)}")

    print("Collecting walk-forward signals...")
    asset_signals = collect_all_signals(prepared)

    result = holdout_analysis(asset_signals)
    print_holdout_analysis(result)


def cmd_stability():
    """Run rolling stability analysis and parameter sensitivity test."""
    from model.train import prepare_data
    from backtest.stability import (
        _collect_all_signals, _run_multi_backtest,
        print_rolling_analysis, parameter_sensitivity,
    )
    from config import SYMBOLS

    from data.store import has_cache
    from config import ETF_SYMBOLS

    print("Preparing data...")
    prepared = {}
    for key in SYMBOLS:
        prepared[key] = prepare_data(key)
    for key in ETF_SYMBOLS:
        if has_cache(key):
            try:
                prepared[key] = prepare_data(key)
                print(f"  Loaded {key} (ETF)")
            except Exception:
                pass
    print(f"  Total assets: {len(prepared)}")

    print("Collecting walk-forward signals...")
    asset_signals = _collect_all_signals(prepared)

    from config import (
        PROB_BUY_THRESHOLD, HOLD_PERIOD, POSITION_MED_CONF,
        POSITION_HIGH_CONF, STOP_LOSS_PCT, PER_ASSET_MAX_POSITION,
    )
    from strategy.regime import VIX_CAUTION, VIX_STRESS

    # Run baseline backtest for rolling analysis
    dates, values = _run_multi_backtest(
        asset_signals,
        buy_thresh=PROB_BUY_THRESHOLD, hold_days=HOLD_PERIOD,
        pos_med=POSITION_MED_CONF, pos_high=POSITION_HIGH_CONF,
        stop_loss=STOP_LOSS_PCT, per_asset_max=PER_ASSET_MAX_POSITION,
        vix_caution=VIX_CAUTION, vix_stress=VIX_STRESS,
    )

    print_rolling_analysis(dates, values)
    parameter_sensitivity(asset_signals)


COMMANDS = {
    "fetch": cmd_fetch,
    "train": cmd_train,
    "predict": cmd_predict,
    "evaluate": cmd_evaluate,
    "backtest": cmd_backtest,
    "validate-signal": cmd_validate_signal,
    "sector-analysis": cmd_sector_analysis,
    "holdout": cmd_holdout,
    "stability": cmd_stability,
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print("Usage: python main.py <command>")
        print(f"Commands: {', '.join(COMMANDS.keys())}")
        sys.exit(1)

    command = sys.argv[1]
    COMMANDS[command]()


if __name__ == "__main__":
    main()
