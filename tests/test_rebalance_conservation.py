import numpy as np
import pytest
from src.inference import PortfolioConstructor


def test_rebalance_conservation_invariants():
    """
    Verifies fundamental accounting and conservation invariants:
    1. sum(target_weights) == 1.0
    2. User supplied current weights are NOT normalized (e.g. 50% invested portfolio is preserved)
    3. current_weight + delta_weight == target_weight for every ticker in Target ∪ Current
    4. sum(current_weight) + sum(delta_weight) == sum(target_weight)
    5. trade_value == abs(delta_weight) * portfolio_value
    """
    portfolio_value = 100000.0

    target = {
        "AAPL": 0.20,
        "MSFT": 0.20,
        "GOOG": 0.20,
        "AMZN": 0.20,
        "META": 0.20,
    }

    current = {
        "AAPL": 0.10,
        "MSFT": 0.30,
        "GOOG": 0.20,
        "TSLA": 0.10,  # 70% total invested
    }

    orders = PortfolioConstructor.compute_rebalance_orders(
        target_weights=target,
        current_weights=current,
        portfolio_value=portfolio_value,
        include_holds=True,
        detailed_action=True,
    )

    all_tickers = set(target.keys()) | set(current.keys())
    assert len(orders) == len(all_tickers)

    # Invariant 1: Target sum is 1.0
    assert np.isclose(sum(target.values()), 1.0)

    # Invariant 2: Current weights preserved as supplied
    assert np.isclose(sum(current.values()), 0.70)

    # Invariant 3: current_weight + delta_weight == target_weight for every ticker
    for _, row in orders.iterrows():
        t = row["Ticker"]
        cur = row["CurrentWeight"]
        diff = row["WeightDelta"]
        tgt = row["TargetWeight"]
        val = row["TradeValue"]

        assert np.isclose(cur + diff, tgt, atol=1e-7), f"Conservation failed for {t}"
        assert np.isclose(val, abs(diff) * portfolio_value, atol=1e-5), f"Trade value mismatch for {t}"

    # Invariant 4: sum(current) + sum(delta) == sum(target)
    total_cur = orders["CurrentWeight"].sum()
    total_delta = orders["WeightDelta"].sum()
    total_tgt = orders["TargetWeight"].sum()
    assert np.isclose(total_cur + total_delta, total_tgt, atol=1e-7)

    # Verify specific order classifications
    order_map = {row["Ticker"]: row for _, row in orders.iterrows()}

    assert order_map["TSLA"]["Action"] == "SELL (EXIT)"
    assert order_map["TSLA"]["TargetWeight"] == 0.0
    assert np.isclose(order_map["TSLA"]["TradeValue"], 10000.0)

    assert order_map["AMZN"]["Action"] == "BUY (NEW)"
    assert order_map["AMZN"]["CurrentWeight"] == 0.0
    assert np.isclose(order_map["AMZN"]["TradeValue"], 20000.0)

    assert order_map["AAPL"]["Action"] == "BUY (ADD)"
    assert np.isclose(order_map["AAPL"]["TradeValue"], 10000.0)

    assert order_map["MSFT"]["Action"] == "SELL (TRIM)"
    assert np.isclose(order_map["MSFT"]["TradeValue"], 10000.0)

    assert order_map["GOOG"]["Action"] == "HOLD (NO CHANGE)"
    assert np.isclose(order_map["GOOG"]["TradeValue"], 0.0)


def test_min_trade_boundary_conditions():
    """
    Verifies exact boundary conditions for minTrade filter:
    1. tradeVal == minTrade - 0.01 -> HOLD (BELOW MIN)
    2. tradeVal == minTrade        -> active trade (BUY / SELL)
    3. tradeVal == minTrade + 0.01 -> active trade (BUY / SELL)
    4. abs(delta) < 1e-5           -> HOLD (NO CHANGE)
    """
    portfolio_value = 100000.0
    min_trade = 100.0

    target = {
        "STOCK_BELOW": 0.200999,
        "STOCK_EXACT": 0.201000,
        "STOCK_ABOVE": 0.201001,
        "STOCK_EQUAL": 0.200000,
    }
    current = {
        "STOCK_BELOW": 0.200000,  # Delta: +0.000999 ($99.90) -> BELOW MIN
        "STOCK_EXACT": 0.200000,  # Delta: +0.001000 ($100.00) -> EXACT THRESHOLD
        "STOCK_ABOVE": 0.200000,  # Delta: +0.001001 ($100.10) -> ABOVE MIN
        "STOCK_EQUAL": 0.200000,  # Delta: 0.000000 ($0.00)   -> NO CHANGE
    }

    orders = PortfolioConstructor.compute_rebalance_orders(
        target_weights=target,
        current_weights=current,
        portfolio_value=portfolio_value,
        min_trade=min_trade,
        include_holds=True,
        detailed_action=True,
    )

    order_map = {row["Ticker"]: row for _, row in orders.iterrows()}

    assert order_map["STOCK_BELOW"]["Action"] == "HOLD (BELOW MIN)"
    assert order_map["STOCK_EXACT"]["Action"] == "BUY (ADD)"
    assert order_map["STOCK_ABOVE"]["Action"] == "BUY (ADD)"
    assert order_map["STOCK_EQUAL"]["Action"] == "HOLD (NO CHANGE)"

    active_orders = orders[~orders["Action"].str.startswith("HOLD")]
    assert len(active_orders) == 2
    assert set(active_orders["Ticker"]) == {"STOCK_EXACT", "STOCK_ABOVE"}

    gross_turnover = active_orders["TradeValue"].sum()
    assert np.isclose(gross_turnover, 100.0 + 100.10, atol=1e-2)
