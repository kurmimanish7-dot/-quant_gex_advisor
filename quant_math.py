# -*- coding: utf-8 -*-
"""
Derivative Mathematics, Advanced Greeks, and Statistical Indicator Engines.
"""

import math
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")

def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / 1.4142135623730951))

def norm_pdf(x: float) -> float:
    return 0.3989422804014327 * math.exp(-0.5 * x * x)

def calculate_advanced_greeks(
    spot: float,
    strike: float,
    dte_days: float,
    rate: float = 0.07,
    sigma: float = 0.128,
    option_type: str = "CE"
) -> Dict[str, float]:
    t = max(dte_days / 365.0, 1e-5)
    v = max(sigma, 1e-4)

    d1 = (math.log(spot / strike) + (rate + 0.5 * v**2) * t) / (v * math.sqrt(t))
    d2 = d1 - v * math.sqrt(t)

    pdf_d1 = norm_pdf(d1)
    gamma = pdf_d1 / (spot * v * math.sqrt(t))
    vega = (spot * pdf_d1 * math.sqrt(t)) / 100.0
    vanna = -pdf_d1 * d2 / v
    # Exact Charm under zero continuous dividend yield
    charm = (pdf_d1 * (2.0 * rate * t - d2 * v * math.sqrt(t)) / (2.0 * t * v * math.sqrt(t))) / 365.0

    if option_type == "CE":
        price = spot * norm_cdf(d1) - strike * math.exp(-rate * t) * norm_cdf(d2)
        delta = norm_cdf(d1)
        theta = (-(spot * pdf_d1 * v) / (2.0 * math.sqrt(t)) - rate * strike * math.exp(-rate * t) * norm_cdf(d2)) / 365.0
    else:
        price = strike * math.exp(-rate * t) * norm_cdf(-d2) - spot * norm_cdf(-d1)
        delta = norm_cdf(d1) - 1.0
        theta = (-(spot * pdf_d1 * v) / (2.0 * math.sqrt(t)) + rate * strike * math.exp(-rate * t) * norm_cdf(-d2)) / 365.0

    return {
        "price": price, "delta": delta, "gamma": gamma,
        "theta": theta, "vega": vega, "vanna": vanna, "charm": charm
    }

def compute_gex_and_max_pain(
    chain: Optional[pd.DataFrame],
    spot: Optional[float],
    lot_size: int = 65,
    dte_days: float = 5.0
) -> Dict[str, Any]:
    default_out = {
        "net_gex_cr": 0.0, "gex_regime": "UNAVAILABLE", "gamma_flip": None,
        "max_pain": None, "call_wall": None, "put_wall": None,
        "expected_move": 0.0, "atm_iv": 0.0, "status": "UNAVAILABLE"
    }
    if chain is None or chain.empty or spot is None or spot <= 0:
        return default_out

    work = chain.copy()
    work["_s"] = pd.to_numeric(work["strikePrice"], errors="coerce")
    work["_t"] = work["option_type"].astype(str).str.upper()
    work["_oi"] = pd.to_numeric(work["openInterest"], errors="coerce").fillna(0)
    work["_p"] = pd.to_numeric(work["lastPrice"], errors="coerce").fillna(0.0)
    work["_iv"] = pd.to_numeric(work["impliedVolatility"], errors="coerce").fillna(12.8)

    work = work.dropna(subset=["_s"]).copy()
    work = work[(work["_s"] >= spot * 0.88) & (work["_s"] <= spot * 1.12)]
    if work.empty:
        return default_out

    calls = work[work["_t"] == "CE"]
    puts = work[work["_t"] == "PE"]

    call_wall = float(calls.loc[calls["_oi"].idxmax()]["_s"]) if not calls.empty and calls["_oi"].max() > 0 else None
    put_wall = float(puts.loc[puts["_oi"].idxmax()]["_s"]) if not puts.empty and puts["_oi"].max() > 0 else None

    # Max Pain Minimization
    unique_strikes = sorted(work["_s"].unique())
    strike_payouts = {}
    for test_s in unique_strikes:
        c_loss = (calls["_s"].apply(lambda k: max(0.0, test_s - k)) * calls["_oi"]).sum()
        p_loss = (puts["_s"].apply(lambda k: max(0.0, k - test_s)) * puts["_oi"]).sum()
        strike_payouts[test_s] = c_loss + p_loss
    max_pain = min(strike_payouts, key=strike_payouts.get) if strike_payouts else None

    # Exact Net GEX Calculation
    total_gex = 0.0
    for s_val in unique_strikes:
        c_sub = calls[calls["_s"] == s_val]
        p_sub = puts[puts["_s"] == s_val]
        c_oi = float(c_sub["_oi"].iloc[0]) if not c_sub.empty else 0.0
        p_oi = float(p_sub["_oi"].iloc[0]) if not p_sub.empty else 0.0
        c_iv = float(c_sub["_iv"].iloc[0]) if not c_sub.empty and c_sub["_iv"].iloc[0] > 0 else 12.8
        p_iv = float(p_sub["_iv"].iloc[0]) if not p_sub.empty and p_sub["_iv"].iloc[0] > 0 else 12.8

        g_c = calculate_advanced_greeks(spot, s_val, dte_days, 0.07, c_iv / 100.0, "CE")["gamma"]
        g_p = calculate_advanced_greeks(spot, s_val, dte_days, 0.07, p_iv / 100.0, "PE")["gamma"]

        ce_gex = (spot**2 * g_c * c_oi * lot_size * 0.01) / 1e7
        pe_gex = (-spot**2 * g_p * p_oi * lot_size * 0.01) / 1e7
        total_gex += (ce_gex + pe_gex)

    # Root Solving for Gamma Flip
    def eval_gex(s_test: float) -> float:
        net_val = 0.0
        for s_val in unique_strikes:
            c_sub = calls[calls["_s"] == s_val]
            p_sub = puts[puts["_s"] == s_val]
            c_oi = float(c_sub["_oi"].iloc[0]) if not c_sub.empty else 0.0
            p_oi = float(p_sub["_oi"].iloc[0]) if not p_sub.empty else 0.0
            gc = calculate_advanced_greeks(s_test, s_val, dte_days, 0.07, 0.13, "CE")["gamma"]
            gp = calculate_advanced_greeks(s_test, s_val, dte_days, 0.07, 0.13, "PE")["gamma"]
            net_val += (s_test**2 * (gc * c_oi - gp * p_oi) * lot_size * 0.01) / 1e7
        return net_val

    gamma_flip = None
    search_space = np.linspace(spot * 0.88, spot * 1.12, 50)
    gex_evals = [eval_gex(x) for x in search_space]
    for idx in range(len(gex_evals) - 1):
        if (gex_evals[idx] <= 0 and gex_evals[idx+1] >= 0) or (gex_evals[idx] >= 0 and gex_evals[idx+1] <= 0):
            # Bisection refinement
            low, high = search_space[idx], search_space[idx+1]
            for _ in range(8):
                mid = (low + high) / 2.0
                if eval_gex(mid) * gex_evals[idx] <= 0:
                    high = mid
                else:
                    low = mid
            gamma_flip = round(float((low + high) / 2.0), 1)
            break

    # ATM Expected Move
    atm_calls = calls.iloc[(calls["_s"] - spot).abs().argsort()[:1]] if not calls.empty else pd.DataFrame()
    atm_puts = puts.iloc[(puts["_s"] - spot).abs().argsort()[:1]] if not puts.empty else pd.DataFrame()
    c_p = float(atm_calls["_p"].iloc[0]) if not atm_calls.empty else 0.0
    p_p = float(atm_puts["_p"].iloc[0]) if not atm_puts.empty else 0.0
    straddle = c_p + p_p
    expected_move = round(0.84 * straddle, 2) if straddle > 0 else 0.0
    atm_iv = float(atm_calls["_iv"].iloc[0]) if not atm_calls.empty else 12.8

    regime = "POSITIVE (MEAN-REVERTING)" if total_gex >= 0 else "NEGATIVE (VOLATILITY AMPLIFIER)"

    return {
        "net_gex_cr": round(total_gex, 2), "gex_regime": regime,
        "gamma_flip": gamma_flip, "max_pain": max_pain,
        "call_wall": call_wall, "put_wall": put_wall,
        "expected_move": expected_move, "atm_iv": round(atm_iv, 1),
        "status": "VALID"
    }

def calculate_quarter_kelly(p_win: float, reward_risk: float) -> float:
    if reward_risk <= 0:
        return 0.0
    q = 1.0 - p_win
    kelly = (p_win * reward_risk - q) / reward_risk
    quarter = max(0.0, kelly / 4.0)
    return round(quarter * 100.0, 1)

def classify_oi_buildup(price_change: float, oi_change: float) -> str:
    if price_change >= 0 and oi_change > 0:
        return "LONG BUILDUP (Bullish Expansion)"
    elif price_change >= 0 and oi_change <= 0:
        return "SHORT COVERING (Shorts Liquidating)"
    elif price_change < 0 and oi_change > 0:
        return "SHORT BUILDUP (Writing Pressure)"
    elif price_change < 0 and oi_change <= 0:
        return "LONG UNWINDING (Bulls Liquidating)"
    return "CONSOLIDATION"

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or len(df) < 5:
        return pd.DataFrame()
    work = df.copy()

    for col in ["open", "high", "low", "close"]:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work = work.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)

    close, high, low = work["close"], work["high"], work["low"]

    work["EMA20"] = close.ewm(span=20, adjust=False).mean()
    work["EMA50"] = close.ewm(span=50, adjust=False).mean()

    # Wilder's Smoothing for RSI
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0/14.0, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0/14.0, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    work["RSI"] = 100.0 - (100.0 / (1.0 + rs))

    # Wilder's ATR
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    work["ATR"] = tr.ewm(alpha=1.0/14.0, adjust=False).mean()

    # Wilder's ADX
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    s_plus = pd.Series(plus_dm, index=work.index).ewm(alpha=1.0/14.0, adjust=False).mean()
    s_minus = pd.Series(minus_dm, index=work.index).ewm(alpha=1.0/14.0, adjust=False).mean()
    plus_di = 100.0 * (s_plus / work["ATR"].replace(0, np.nan))
    minus_di = 100.0 * (s_minus / work["ATR"].replace(0, np.nan))
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    work["ADX"] = dx.ewm(alpha=1.0/14.0, adjust=False).mean()

    # Session-Reset VWAP at 09:15 IST
    typical = (high + low + close) / 3.0
    vol = pd.to_numeric(work["volume"], errors="coerce").fillna(1.0) if "volume" in work.columns else pd.Series(1.0, index=work.index)

    if "datetime" in work.columns:
        work["_date"] = pd.to_datetime(work["datetime"]).dt.date
    else:
        work["_date"] = dt.now(IST).date()

    pv = typical * vol
    work["VWAP"] = pv.groupby(work["_date"]).cumsum() / vol.groupby(work["_date"]).cumsum().replace(0, np.nan)
    work.drop(columns=["_date"], errors="ignore", inplace=True)

    return work

def resample_ohlcv(df: pd.DataFrame, target_rule: str) -> pd.DataFrame:
    if df.empty or len(df) < 2:
        return df
    work = df.copy()
    if "datetime" in work.columns:
        try:
            work["datetime"] = pd.to_datetime(work["datetime"])
            if work["datetime"].dt.tz is None:
                work["datetime"] = work["datetime"].dt.tz_localize(IST)
            work = work.set_index("datetime")
            rule_map = {"10m": "10min", "2h": "2h", "4h": "4h", "1w": "W-FRI", "1mo": "ME", "3mo": "3ME"}
            mapped = rule_map.get(target_rule, target_rule)
            return work.resample(mapped, closed="left", label="left").agg({
                "open": "first", "high": "max", "low": "min", "close": "last",
                "volume": "sum" if "volume" in work.columns else "count"
            }).dropna(subset=["close"]).reset_index()
        except Exception:
            pass
    return df
