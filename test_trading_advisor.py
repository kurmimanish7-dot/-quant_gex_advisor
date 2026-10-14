# test_trading_advisor.py
# -*- coding: utf-8 -*-
import math
import os
import pytest
import numpy as np
import pandas as pd
from datetime import datetime as dt

from quant_math import (
    norm_cdf,
    norm_pdf,
    calculate_advanced_greeks,
    compute_gex_and_max_pain,
    add_indicators
)
from risk_engine import CentralizedRiskEngine, PaperRiskLimits
from market_data import MarketDataEnvelope, DataStatus, CONTRACT_MASTER, get_contract_lot_size, IST

def test_norm_distribution_accuracy():
    assert math.isclose(norm_cdf(0.0), 0.5, abs_tol=1e-7)
    assert math.isclose(norm_cdf(1.95996398454), 0.975, abs_tol=1e-4)
    assert math.isclose(norm_pdf(0.0), 0.39894228, abs_tol=1e-6)

def test_black_scholes_call_put_parity():
    spot = 22700.0
    strike = 22700.0
    dte = 30.0
    r = 0.07
    sigma = 0.15

    call = calculate_advanced_greeks(spot, strike, dte, r, sigma, "CE")
    put = calculate_advanced_greeks(spot, strike, dte, r, sigma, "PE")

    # Put-Call Parity: C - P = S - K * exp(-r*T)
    t = dte / 365.0
    lhs = call["price"] - put["price"]
    rhs = spot - strike * math.exp(-r * t)
    assert math.isclose(lhs, rhs, abs_tol=1e-2)

    # Greeks Symmetry
    assert math.isclose(call["gamma"], put["gamma"], rel_tol=1e-4)
    assert math.isclose(call["vega"], put["vega"], rel_tol=1e-4)
    assert math.isclose(call["charm"], put["charm"], rel_tol=1e-4)

def test_greeks_zero_volatility_edge_case():
    res = calculate_advanced_greeks(spot=22700, strike=22600, dte_days=10, rate=0.07, sigma=0.0, option_type="CE")
    # Must fall back to intrinsic lower bound without zero-division runtime crash
    assert res["price"] >= 100.0
    assert np.isfinite(res["gamma"])

def test_wilder_rsi_calculation():
    prices = [100.0 + i * 2.0 for i in range(30)]
    df = pd.DataFrame({
        "open": prices, "high": [p + 1 for p in prices],
        "low": [p - 1 for p in prices], "close": prices,
        "volume": [1000] * 30, "datetime": pd.date_range("2026-01-01", periods=30, freq="5min")
    })
    res = add_indicators(df)
    assert res["RSI"].iloc[-1] > 95.0

def test_session_reset_vwap():
    times = [
        dt(2026, 1, 1, 9, 15, tzinfo=IST), dt(2026, 1, 1, 9, 20, tzinfo=IST),
        dt(2026, 1, 2, 9, 15, tzinfo=IST), dt(2026, 1, 2, 9, 20, tzinfo=IST)
    ]
    df = pd.DataFrame({
        "open": [100, 102, 200, 202], "high": [101, 103, 201, 203],
        "low": [99, 101, 199, 201], "close": [100, 102, 200, 202],
        "volume": [100, 100, 100, 100], "datetime": times
    })
    res = add_indicators(df)
    expected_day2_open_vwap = (200 + 201 + 199) / 3.0
    assert math.isclose(res["VWAP"].iloc[2], expected_day2_open_vwap, rel_tol=1e-3)

def test_contract_lot_sizes():
    assert get_contract_lot_size("NIFTY") == 65
    assert get_contract_lot_size("BANKNIFTY") == 30
    assert get_contract_lot_size("FINNIFTY") == 60
    assert get_contract_lot_size("SENSEX") == 20

def test_gex_root_finding_and_unavailable():
    # Empty chain must produce UNAVAILABLE
    res_empty = compute_gex_and_max_pain(pd.DataFrame(), 22700.0)
    assert res_empty["status"] == "UNAVAILABLE"
    assert res_empty["gamma_flip"] is None

    # Synthetic balanced chain
    data = []
    for strike in [22500, 22600, 22700, 22800, 22900]:
        data.append({"strikePrice": strike, "option_type": "CE", "openInterest": 50000, "lastPrice": 100, "impliedVolatility": 13.0})
        data.append({"strikePrice": strike, "option_type": "PE", "openInterest": 50000, "lastPrice": 100, "impliedVolatility": 13.0})
    chain = pd.DataFrame(data)
    res = compute_gex_and_max_pain(chain, 22700.0, lot_size=65)
    assert res["status"] == "VALID"
    assert res["max_pain"] == 22700.0
    assert math.isclose(res["net_gex_cr"], 0.0, abs_tol=1e-1)

def test_risk_engine_geometry_rejection():
    engine = CentralizedRiskEngine(state_file="test_state.json")
    try:
        # Inverted BUY: SL > Entry
        ok, msg, _ = engine.validate_and_size_order("NIFTY", "INDEX", "BUY", entry=22700, sl=22800, t1=22900, lot_size=65, confidence=80)
        assert not ok
        assert "GEOMETRY ERROR" in msg

        # Inverted SELL: SL < Entry
        ok, msg, _ = engine.validate_and_size_order("NIFTY", "INDEX", "SELL", entry=22700, sl=22600, t1=22500, lot_size=65, confidence=80)
        assert not ok
        assert "GEOMETRY ERROR" in msg
    finally:
        if os.path.exists("test_state.json"):
            os.remove("test_state.json")

def test_risk_engine_safety_lock():
    limits = PaperRiskLimits(max_daily_loss_inr=10000.0)
    engine = CentralizedRiskEngine(limits=limits, state_file="test_state_lock.json")
    try:
        engine.record_pnl("NIFTY", -11000.0)
        assert engine.safety_lock is True

        ok, msg, _ = engine.validate_and_size_order("NIFTY", "INDEX", "BUY", entry=22700, sl=22650, t1=22800, lot_size=65, confidence=80)
        assert not ok
        assert "SAFETY LOCK ENGAGED" in msg
    finally:
        if os.path.exists("test_state_lock.json"):
            os.remove("test_state_lock.json")

def test_market_data_envelope_actionable():
    now = dt.now(IST)
    valid_env = MarketDataEnvelope("NIFTY", "99926000", "NSE", 22700.0, "NSE_PRIMARY", DataStatus.LIVE, now, now)
    assert valid_env.is_actionable() is True

    delayed_env = MarketDataEnvelope("NIFTY", "99926000", "NSE", 22700.0, "YAHOO", DataStatus.DELAYED, now, now)
    assert delayed_env.is_actionable() is False

    stale_now = dt.fromtimestamp(now.timestamp() - 60, tz=IST)
    stale_env = MarketDataEnvelope("NIFTY", "99926000", "NSE", 22700.0, "NSE_PRIMARY", DataStatus.LIVE, stale_now, stale_now)
    assert stale_env.is_actionable() is False
