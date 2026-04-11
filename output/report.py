import json
from datetime import date


SYMBOL_NAMES = {
    "SP500": "S&P 500",
    "NASDAQ": "NASDAQ Composite",
}


def format_json_report(predictions: list[dict]) -> str:
    """Format predictions as JSON output matching v2 spec."""
    return json.dumps(predictions, indent=2, ensure_ascii=False)


def format_report(predictions: list[dict]) -> str:
    """Format predictions into a human-readable report with JSON details."""
    lines = []
    lines.append("=" * 60)
    lines.append(f"  Quantitative Investment Decision Report - {date.today()}")
    lines.append("=" * 60)

    for pred in predictions:
        symbol = pred["symbol"]
        name = SYMBOL_NAMES.get(symbol, symbol)
        details = pred.get("details", {})

        lines.append("")
        lines.append(f"  {name} ({details.get('ticker', '')})")
        lines.append(f"  Last Close: {details.get('close', 'N/A')}  ({pred['date']})")
        lines.append(f"  Signal Strength: {pred['signal_strength']}")
        lines.append(f"  Probability (5-day >1%): {pred['probability']:.2%}")
        lines.append(f"  Suggestion: {pred['suggestion']}")
        lines.append(f"  Position Size: {pred['position_size']:.0%}")
        lines.append(f"  Regime: {pred['regime']}")
        lines.append(f"  Risk Note: {pred['risk_note']}")
        lines.append(f"  Model: {pred['model_version']}")
        lines.append("")
        lines.append("  Signal Sources:")
        lines.append(f"    - MA Trend: {details.get('ma_trend', 'N/A')}")
        lines.append(f"    - RSI (14): {details.get('rsi', 'N/A')}")
        lines.append(f"    - MACD Hist: {details.get('macd_hist', 'N/A')}")
        lines.append(f"    - Reason: {details.get('decision_reason', 'N/A')}")
        lines.append("-" * 60)

    lines.append("")
    lines.append("  JSON Output:")
    # Strip details for clean JSON output per spec
    clean = []
    for p in predictions:
        clean.append({k: v for k, v in p.items() if k != "details"})
    lines.append(json.dumps(clean, indent=2, ensure_ascii=False))

    return "\n".join(lines)
