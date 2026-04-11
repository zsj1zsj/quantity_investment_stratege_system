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
    from config import SYMBOLS

    for key in SYMBOLS:
        print(f"\nPreparing data for {key}...")
        df = prepare_data(key)
        result = run_backtest(df, key)
        print_backtest_report(result)


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


COMMANDS = {
    "fetch": cmd_fetch,
    "train": cmd_train,
    "predict": cmd_predict,
    "evaluate": cmd_evaluate,
    "backtest": cmd_backtest,
    "validate-signal": cmd_validate_signal,
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
