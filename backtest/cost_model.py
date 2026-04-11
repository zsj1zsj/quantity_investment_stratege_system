"""Transaction cost model: simulates commission and slippage."""
from config import COST_PER_SIDE


def apply_buy_cost(price: float) -> float:
    """Return effective buy price after slippage and commission.

    Buying costs more than the quoted price.
    """
    return price * (1 + COST_PER_SIDE)


def apply_sell_cost(price: float) -> float:
    """Return effective sell price after slippage and commission.

    Selling yields less than the quoted price.
    """
    return price * (1 - COST_PER_SIDE)


def net_return(gross_return: float) -> float:
    """Compute net return after round-trip transaction costs.

    gross_return: fractional return (e.g., 0.05 for 5%)
    """
    # Buy at (1 + cost), sell at (1 - cost)
    # net = (1 + gross) * (1 - cost) / (1 + cost) - 1
    effective = (1 + gross_return) * (1 - COST_PER_SIDE) / (1 + COST_PER_SIDE) - 1
    return effective
