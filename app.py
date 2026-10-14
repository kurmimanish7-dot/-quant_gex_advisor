# -*- coding: utf-8 -*-
import datetime
from datetime import datetime as dt, date, time as dtime, timedelta
import json
import logging
import math
import os
import time
import xml.etree.ElementTree as ET
from zoneinfo import ZoneInfo
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st

from market_data import (
    DataStatus,
    MarketDataEnvelope,
    CONTRACT_MASTER,
    NSEExchangeSession,
    get_contract_lot_size,
    is_trading_holiday,
    get_preceding_trading_day,
    market_open,
    IST
)
from quant_math import (
    calculate_advanced_greeks,
    compute_gex_and_max_pain,
    calculate_quarter_kelly,
    classify_oi_buildup,
    add_indicators,
    resample_ohlcv,
    norm_cdf,
    norm_pdf
)
from risk_engine import CentralizedRiskEngine, PaperRiskLimits

logger = logging.getLogger("QuantitativeAdvisor")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# =========================================================
# SAFE UNICODE ESCAPE CONSTANTS
# =========================================================
RUPEE = "\u20B9"                # ₹
BOLT = "\u26A1"                 # ⚡
FLAG_IN = "\U0001F1EE\U0001F1F3"# 🇮🇳
CHART_BAR = "\U0001F4CA"        # 📊
CHART_UP = "\U0001F4C8"         # 📈
TARGET = "\U0001F3AF"           # 🎯
TEMPLE = "\U0001F3DB"           # 🏛️
COMPASS = "\U0001F9ED"          # 🧭
GLOBE = "\U0001F310"            # 🌐
TRAFFIC = "\U0001F6A6"          # 🚦
GREEN_CIRCLE = "\U0001F7E2"     # 🟢
RED_CIRCLE = "\U0001F534"       # 🔴
YELLOW_CIRCLE = "\U0001F7E1"    # 🟡
SIREN = "\U0001F6A8"            # 🚨
FIRE = "\U0001F525"             # 🔥
SWORDS = "\u2694\uFE0F"         # ⚔️
PAUSE = "\u23F8\uFE0F"          # ⏸️
ROCKET = "\U0001F680"           # 🚀
DOWN_ARROW = "\U0001F53D"       # 🔻
SCALE = "\u2696\uFE0F"          # ⚖️
SHIELD = "\U0001F6E1\uFE0F"     # 🛡️
PIN = "\U0001F4CC"              # 📌
WARNING = "\u26A0\uFE0F"        # ⚠️
CHECK = "\u2705"                # ✅
STOP = "\U0001F6D1"             # 🛑
REFRESH = "\U0001F504"          # 🔄
STOPWATCH = "\u23F1\uFE0F"      # ⏱️
GEAR = "\u2699\uFE0F"           # ⚙️
SEARCH = "\U0001F50D"           # 🔍
BUILDING = "\U0001F3E2"         # 🏢
RULER = "\U0001F4D0"            # 📐
NEWS = "\U0001F4F0"             # 📰
ROBOT = "\U0001F916"            # 🤖
DELTA_SYM = "\u0394"            # Δ
THETA_SYM = "\u0398"            # Θ
GAMMA_SYM = "\u0393"            # Γ
BULLET = " - "

# =========================================================
# PAGE CONFIGURATION & INSTITUTIONAL THEME
# =========================================================
st.set_page_config(
    page_title="AI Institutional Live Trading Advisor (Quantitative GEX Engine)",
    page_icon=BOLT,
    layout="wide",
)

PAPER_TRADING = True

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 0.8rem !important;
        padding-left: 0.5rem !important;
        padding-right: 0.5rem !important;
        max-width: 100% !important;
    }
    .trade-card {
        width: 100%;
        box-sizing: border-box;
        border: 1px solid rgba(128, 128, 128, 0.30);
        border-radius: 12px;
        padding: 1rem;
        margin: 0.7rem 0;
        background-color: rgba(255, 255, 255, 0.02);
    }
    .light-box {
        border: 1px solid rgba(128, 128, 128, 0.25);
        border-radius: 12px;
        padding: 0.8rem;
        text-align: center;
        background-color: rgba(255, 255, 255, 0.02);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .reason-box {
        background-color: rgba(128, 128, 128, 0.08);
        border-left: 4px solid #4CAF50;
        padding: 0.75rem;
        border-radius: 4px;
        margin: 0.5rem 0;
        line-height: 1.45;
    }
    .prediction-card-green {
        background: linear-gradient(135deg, rgba(76, 175, 80, 0.15), rgba(76, 175, 80, 0.05));
        border: 1.5px solid #4CAF50;
        border-radius: 12px;
        padding: 1rem 1.1rem;
        margin: 0.5rem 0 0.8rem 0;
    }
    .prediction-card-red {
        background: linear-gradient(135deg, rgba(244, 67, 54, 0.15), rgba(244, 67, 54, 0.05));
        border: 1.5px solid #F44336;
        border-radius: 12px;
        padding: 1rem 1.1rem;
        margin: 0.5rem 0 0.8rem 0;
    }
    .prediction-card-gold {
        background: linear-gradient(135deg, rgba(255, 193, 7, 0.15), rgba(255, 193, 7, 0.05));
        border: 1.5px solid #FFC107;
        border-radius: 12px;
        padding: 1rem 1.1rem;
        margin: 0.5rem 0 0.8rem 0;
    }
    [data-testid="stMetric"] {
        background-color: rgba(255, 255, 255, 0.02) !important;
        border: 1px solid rgba(128, 128, 128, 0.25) !important;
        border-radius: 10px !important;
        padding: 0.5rem 0.55rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if hasattr(st, "fragment"):
    live_fragment = st.fragment
elif hasattr(st, "experimental_fragment"):
    live_fragment = st.experimental_fragment
else:
    def live_fragment(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

# =========================================================
# 12-TIER TIMEFRAME CONFIGURATION
# =========================================================
TIMEFRAME_CONFIG = {
    "1m": {"label": f"{BOLT} 1 Minute (Scalping / High-Speed)", "smartapi": "ONE_MINUTE", "yfinance": "1m", "yf_range": "2d", "days": 2, "holding": "Quick Scalp (1-5 Mins)", "atr_mult": 0.8, "desc": "Ultra-fast scalp setup. High sensitivity, tight SL."},
    "5m": {"label": f"{CHART_BAR} 5 Minutes (Standard Intraday)", "smartapi": "FIVE_MINUTE", "yfinance": "5m", "yf_range": "5d", "days": 5, "holding": "Intraday Momentum (15-45 Mins)", "atr_mult": 1.0, "desc": "Standard institutional intraday timeframe."},
    "10m": {"label": f"{TARGET} 10 Minutes (Noise-Filtered Scalp)", "smartapi": "TEN_MINUTE", "yfinance": "10m", "yf_range": "5d", "days": 5, "holding": "Intraday Trend (30-60 Mins)", "atr_mult": 1.2, "desc": "Smooth momentum timeframe filtering out minor whipsaws."},
    "15m": {"label": f"{TEMPLE} 15 Minutes (Institutional Intraday)", "smartapi": "FIFTEEN_MINUTE", "yfinance": "15m", "yf_range": "7d", "days": 7, "holding": "Multi-Hour Trend (1-3 Hours)", "atr_mult": 1.5, "desc": "Key institutional breakout timeframe for major intraday moves."},
    "30m": {"label": f"{CHART_UP} 30 Minutes (Positional / BTST)", "smartapi": "THIRTY_MINUTE", "yfinance": "30m", "yf_range": "1mo", "days": 15, "holding": "BTST / Positional (1-3 Days)", "atr_mult": 2.0, "desc": "Strong trend conviction for overnight holding and multi-day swings."},
    "60m": {"label": f"{COMPASS} 60 Minutes (1 Hour Macro Swing)", "smartapi": "ONE_HOUR", "yfinance": "60m", "yf_range": "1mo", "days": 30, "holding": "Swing Trend (3-7 Days)", "atr_mult": 2.8, "desc": "Macro hourly trend framework for swing traders."},
    "2h": {"label": "\u23F0 2 Hours (120 Mins Multi-Day Swing)", "smartapi": "ONE_HOUR", "yfinance": "60m", "yf_range": "2mo", "days": 60, "holding": "Multi-Day Swing (3-7 Days)", "atr_mult": 3.0, "desc": "Aggregated 120-minute candles filtering out intraday noise."},
    "4h": {"label": "\U0001F553 4 Hours (Half-Day Institutional Matrix)", "smartapi": "ONE_HOUR", "yfinance": "60m", "yf_range": "3mo", "days": 90, "holding": "Weekly Swing (1-2 Weeks)", "atr_mult": 3.5, "desc": "Half-day market structure utilized by institutional trend followers."},
    "1d": {"label": "\U0001F4C5 1 Day (Daily EOD Positional)", "smartapi": "ONE_DAY", "yfinance": "1d", "yf_range": "1y", "days": 365, "holding": "Positional Trend (2-4 Weeks)", "atr_mult": 4.0, "desc": "Daily closing resolution reflecting broad accumulation/distribution."},
    "1w": {"label": "\U0001F5D3\uFE0F 1 Week (Weekly Macro Swing)", "smartapi": "ONE_DAY", "yfinance": "1wk", "yf_range": "2y", "days": 730, "holding": "Macro Positional (1-3 Months)", "atr_mult": 5.0, "desc": "Weekly secular trends removing short-term market corrections."},
    "1mo": {"label": f"{GLOBE} 1 Month (Monthly Positional Cycle)", "smartapi": "ONE_DAY", "yfinance": "1mo", "yf_range": "5y", "days": 1825, "holding": "Secular Long (3-6 Months)", "atr_mult": 6.0, "desc": "Macroeconomic cycle resolution for structural equity trends."},
    "3mo": {"label": f"{TEMPLE} 3 Months (Quarterly Strategic Trend)", "smartapi": "ONE_DAY", "yfinance": "3mo", "yf_range": "10y", "days": 3650, "holding": "Strategic Allocation (6-12 Months)", "atr_mult": 7.5, "desc": "Quarterly earnings cycle resolution for strategic positioning."},
}

# =========================================================
# SECRETS & CLIENT INITIALIZATION
# =========================================================
def secret(name: str) -> str:
    try:
        val = st.secrets.get(name)
        return str(val).strip() if val else ""
    except Exception:
        return ""

ANGEL_API_KEY = secret("ANGEL_API_KEY")
ANGEL_CLIENT_CODE = secret("ANGEL_CLIENT_CODE")
ANGEL_PIN = secret("ANGEL_PIN")
ANGEL_TOTP_SECRET = secret("ANGEL_TOTP_SECRET")
ANGEL_JWT_TOKEN = secret("ANGEL_JWT_TOKEN")
GEMINI_API_KEY = secret("GEMINI_API_KEY") or secret("GOOGLE_API_KEY")
GEMINI_MODEL = secret("GEMINI_MODEL") or "gemini-1.5-flash"

@st.cache_resource(show_spinner=False)
def get_nse_session() -> NSEExchangeSession:
    return NSEExchangeSession()

nse_client = get_nse_session()

@st.cache_resource(show_spinner=False)
def get_risk_engine() -> CentralizedRiskEngine:
    return CentralizedRiskEngine()

risk_engine = get_risk_engine()

# =========================================================
# UTILITIES & NORMALIZATION
# =========================================================
def num(x: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if x is None:
            return default
        if isinstance(x, str):
            x = x.replace(",", "").strip()
        val = float(x)
        return val if np.isfinite(val) else default
    except Exception:
        return default

def fmt(x: Any, digits: int = 2) -> str:
    val = num(x)
    return "UNAVAILABLE" if val is None else f"{val:,.{digits}f}"

def expiry_norm(x: Any) -> Optional[str]:
    if x is None:
        return None
    if isinstance(x, (dt, date)):
        return x.strftime("%Y-%m-%d")
    s = str(x).strip().upper()
    formats = ["%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y", "%d/%m/%Y", "%d%b%Y", "%d-%b-%Y"]
    for f in formats:
        try:
            return dt.strptime(s, f).strftime("%Y-%m-%d")
        except Exception:
            pass
    return s

def expiry_label(x: Any) -> str:
    n = expiry_norm(x)
    if not n:
        return "-"
    try:
        return dt.strptime(n, "%Y-%m-%d").strftime("%d %b %Y")
    except Exception:
        return str(x)

def normalize_option_type(value: Any) -> str:
    if not value:
        return ""
    x = str(value).strip().upper()
    if "CE" in x or "CALL" in x or x.endswith("C"):
        return "CE"
    if "PE" in x or "PUT" in x or x.endswith("P"):
        return "PE"
    return ""

# =========================================================
# REAL-TIME MARKET DATA FETCHING (FAIL-SAFE, NO DEFAULTS)
# =========================================================
@st.cache_data(ttl=5, show_spinner=False)
def get_spot(symbol: str) -> MarketDataEnvelope:
    now_ist = dt.now(IST)
    meta = CONTRACT_MASTER.get(symbol)
    if not meta:
        return MarketDataEnvelope(symbol, "", "NSE", None, "UNKNOWN", DataStatus.UNAVAILABLE, None, now_ist, "Symbol not recognized")

    # 1. Primary Source: Official NSE API
    try:
        url = "https://www.nseindia.com/api/allIndices"
        data = nse_client.get_json(url)
        if data and "data" in data:
            name_check = "NIFTY BANK" if symbol == "BANKNIFTY" else ("NIFTY 50" if symbol == "NIFTY" else symbol)
            for idx in data["data"]:
                idx_sym = str(idx.get("indexSymbol", "")).strip().upper()
                idx_name = str(idx.get("index", "")).strip().upper()
                if idx_sym == name_check or idx_name == name_check:
                    val = num(idx.get("last"))
                    if val and val > 0:
                        status = DataStatus.LIVE if market_open() else DataStatus.HISTORICAL
                        return MarketDataEnvelope(symbol, meta["candle_token"], meta["exchange"], val, "NSE_OFFICIAL_PRIMARY", status, now_ist, now_ist)
    except Exception as exc:
        logger.debug("NSE spot fetch failed: %s", exc)

    # 2. Fallback: Yahoo Finance (Explicitly labeled)
    yf_symbol = meta.get("yfinance")
    if yf_symbol:
        try:
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{yf_symbol}?interval=1d&range=2d"
            resp = requests.get(url, timeout=3, headers={"User-Agent": "Mozilla/5.0"})
            if resp.ok:
                data = resp.json()
                meta_data = data["chart"]["result"][0]["meta"]
                price = float(meta_data.get("regularMarketPrice", 0.0))
                if price > 0:
                    status = DataStatus.DELAYED if market_open() else DataStatus.HISTORICAL
                    return MarketDataEnvelope(symbol, meta["candle_token"], meta["exchange"], price, "YAHOO_FALLBACK_DELAYED", status, None, now_ist)
        except Exception as exc:
            logger.debug("Yahoo spot fallback failed: %s", exc)

    return MarketDataEnvelope(symbol, meta["candle_token"], meta["exchange"], None, "NONE", DataStatus.UNAVAILABLE, None, now_ist, "All data feeds unavailable")

@st.cache_data(ttl=60, show_spinner=False)
def fetch_ohlcv(symbol: str, interval_str: str = "5m", days: int = 5) -> pd.DataFrame:
    meta = CONTRACT_MASTER.get(symbol)
    if not meta:
        return pd.DataFrame()

    tf_entry = None
    for k, v in TIMEFRAME_CONFIG.items():
        if v["smartapi"] == interval_str or k == interval_str:
            tf_entry = v
            break
    if not tf_entry:
        tf_entry = TIMEFRAME_CONFIG.get("5m")

    resample_target = None
    if interval_str in ["2h", "TWO_HOUR"]:
        resample_target = "2h"
    elif interval_str in ["4h", "FOUR_HOUR"]:
        resample_target = "4h"
    elif interval_str in ["1w", "ONE_WEEK"]:
        resample_target = "1w"
    elif interval_str in ["1mo", "ONE_MONTH"]:
        resample_target = "1mo"
    elif interval_str in ["3mo", "THREE_MONTH"]:
        resample_target = "3mo"
    elif interval_str in ["10m", "TEN_MINUTE"]:
        resample_target = "10m"

    yf_symbol = meta.get("yfinance")
    if yf_symbol:
        yf_int = tf_entry["yfinance"]
        yf_range = tf_entry["yf_range"]
        try:
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{yf_symbol}?interval={yf_int}&range={yf_range}"
            resp = requests.get(url, timeout=5, headers={"User-Agent": "Mozilla/5.0"})
            if resp.ok:
                data = resp.json()["chart"]["result"][0]
                quotes = data["indicators"]["quote"][0]
                timestamps = data.get("timestamp", [])
                df = pd.DataFrame({
                    "datetime": [dt.fromtimestamp(ts, tz=IST) for ts in timestamps] if timestamps else None,
                    "open": quotes.get("open", []),
                    "high": quotes.get("high", []),
                    "low": quotes.get("low", []),
                    "close": quotes.get("close", []),
                    "volume": quotes.get("volume", []),
                }).dropna(subset=["close"]).reset_index(drop=True)

                if resample_target:
                    df = resample_ohlcv(df, resample_target)

                if not df.empty and len(df) >= 5:
                    return df
        except Exception as exc:
            logger.warning("Candle fetch failed for %s: %s", symbol, exc)

    return pd.DataFrame()

@st.cache_data(ttl=120, show_spinner=False)
def load_expiries(symbol: str) -> List[str]:
    try:
        url = f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol.upper()}"
        data = nse_client.get_json(url)
        if data:
            exp_dates = data.get("records", {}).get("expiryDates", [])
            vals = [expiry_norm(x) for x in exp_dates if expiry_norm(x)]
            if vals:
                return sorted(set(vals))
    except Exception:
        pass

    now = dt.now(IST)
    target_weekday = 3 if ("SENSEX" in symbol or "BSE" in symbol) else 1
    expiries = []
    cur = now.date()
    for _ in range(5):
        days_ahead = (target_weekday - cur.weekday()) % 7
        if days_ahead == 0 and now.time() > dtime(15, 30) and cur == now.date():
            days_ahead = 7
        raw_expiry = cur + timedelta(days=days_ahead)
        valid_expiry = get_preceding_trading_day(raw_expiry)
        expiries.append(valid_expiry.strftime("%Y-%m-%d"))
        cur = raw_expiry + timedelta(days=7)

    return sorted(list(set(expiries)))

@st.cache_data(ttl=15, show_spinner=False)
def get_chain(symbol: str, expiry: Optional[str]) -> Tuple[pd.DataFrame, DataStatus]:
    if not expiry:
        return pd.DataFrame(), DataStatus.UNAVAILABLE
    try:
        url = f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol.upper()}"
        data = nse_client.get_json(url)
        if not data:
            return pd.DataFrame(), DataStatus.UNAVAILABLE

        records = data.get("records", {})
        data_items = records.get("data", [])
        target_exp_norm = expiry_norm(expiry)

        matched_rows = []
        for item in data_items:
            item_exp = expiry_norm(item.get("expiryDate"))
            if item_exp == target_exp_norm:
                for side in ["CE", "PE"]:
                    if side in item:
                        c_side = item[side]
                        matched_rows.append({
                            "strikePrice": c_side.get("strikePrice"),
                            "option_type": side,
                            "lastPrice": c_side.get("lastPrice"),
                            "openInterest": c_side.get("openInterest"),
                            "changeinOpenInterest": c_side.get("changeinOpenInterest"),
                            "change": c_side.get("change", 0.0),
                            "impliedVolatility": c_side.get("impliedVolatility"),
                            "expiryDate": c_side.get("expiryDate"),
                        })
        if matched_rows:
            return pd.DataFrame(matched_rows), DataStatus.LIVE if market_open() else DataStatus.HISTORICAL
    except Exception as exc:
        logger.warning("Chain lookup failed: %s", exc)

    return pd.DataFrame(), DataStatus.UNAVAILABLE

@st.cache_data(ttl=600, show_spinner=False)
def fetch_fii_dii() -> Dict[str, Any]:
    url = "https://www.nseindia.com/api/fiidiiTradeReact"
    data = nse_client.get_json(url)
    if data:
        fii_net, dii_net = None, None
        for row in data:
            cat = str(row.get("category", "")).upper()
            raw_val = str(row.get("netValue", "0")).replace(",", "").strip()
            val = num(raw_val)
            if "FII" in cat or "FPI" in cat:
                fii_net = val
            elif "DII" in cat:
                dii_net = val

        if fii_net is not None and dii_net is not None:
            combined = round(fii_net + dii_net, 2)
            score = 0
            score += 2 if fii_net > 1000 else (1 if fii_net > 300 else (-2 if fii_net < -1000 else (-1 if fii_net < -300 else 0)))
            score += 1 if dii_net > 500 else (-1 if dii_net < -500 else 0)
            bias = "STRONG BULLISH" if score >= 2 else ("MODERATE BULLISH" if score == 1 else ("STRONG BEARISH" if score <= -2 else ("MODERATE BEARISH" if score == -1 else "NEUTRAL")))
            return {"available": True, "fii_net": fii_net, "dii_net": dii_net, "combined": combined, "score": score, "bias": bias, "source": "NSE_PRIMARY"}

    return {"available": False, "fii_net": None, "dii_net": None, "combined": None, "score": 0, "bias": "UNAVAILABLE", "source": "NONE"}

# =========================================================
# DERIVATIVES & OPTIONS LEVEL FACTORY
# =========================================================
def calculate_pcr(chain: pd.DataFrame) -> Optional[float]:
    if chain is None or chain.empty:
        return None
    c_df = chain[chain["option_type"] == "CE"]
    p_df = chain[chain["option_type"] == "PE"]
    ce_oi = pd.to_numeric(c_df["openInterest"], errors="coerce").fillna(0).sum()
    pe_oi = pd.to_numeric(p_df["openInterest"], errors="coerce").fillna(0).sum()
    if ce_oi > 0:
        val = pe_oi / ce_oi
        return round(float(val), 2)
    return None

def make_trade_idea(
    symbol: str,
    spot_envelope: MarketDataEnvelope,
    instrument: str = "INDEX",
    option_side: Optional[str] = None,
    expiry: Optional[str] = None,
    chain: Optional[pd.DataFrame] = None,
    tf_cfg: Optional[Dict[str, Any]] = None
) -> Optional[Dict[str, Any]]:
    if tf_cfg is None:
        tf_cfg = TIMEFRAME_CONFIG["5m"]

    if not spot_envelope.is_actionable():
        return None

    spot = spot_envelope.value
    if spot is None or spot <= 0:
        return None

    raw_df = fetch_ohlcv(symbol, tf_cfg["smartapi"], tf_cfg["days"])
    if raw_df.empty or len(raw_df) < 20:
        return None

    df = add_indicators(raw_df)
    last_row = df.iloc[-1]

    rsi = float(last_row.get("RSI", 50.0))
    vwap = float(last_row.get("VWAP", spot))
    ema20 = float(last_row.get("EMA20", spot))
    atr = float(last_row.get("ATR", spot * 0.005))

    # Analytical GEX Integration
    lot_size = get_contract_lot_size(symbol)
    gex_data = compute_gex_and_max_pain(chain, spot, lot_size=lot_size, dte_days=5.0)
    expected_move = gex_data["expected_move"] if gex_data["status"] == "VALID" and gex_data["expected_move"] > 0 else (atr * 2.0)

    bullish = (spot > vwap) and (spot > ema20) and (rsi >= 50.0)
    bearish = (spot < vwap) and (spot < ema20) and (rsi < 50.0)

    if not (bullish or bearish):
        return None

    action = "BUY" if bullish else "SELL"
    if instrument == "INDEX":
        entry = spot
        if action == "BUY":
            sl = round(entry - (atr * 1.5), 2)
            t1 = round(entry + (expected_move * 0.618), 2)
            t2 = round(entry + expected_move, 2)
        else:
            sl = round(entry + (atr * 1.5), 2)
            t1 = round(entry - (expected_move * 0.618), 2)
            t2 = round(entry - expected_move, 2)

        valid, reason, sized_qty = risk_engine.validate_and_size_order(
            symbol=symbol, instrument="INDEX", action=action,
            entry=entry, sl=sl, t1=t1, lot_size=lot_size, confidence=80.0
        )
        if not valid:
            logger.info("Spot trade rejected by risk engine: %s", reason)
            return None

        rr = abs(t1 - entry) / max(abs(entry - sl), 1e-4)
        return {
            "segment": "INDEX SPOT",
            "symbol": symbol,
            "action": action,
            "entry": entry,
            "sl": sl,
            "target1": t1,
            "target2": t2,
            "quantity": sized_qty,
            "risk_reward": round(rr, 2),
            "holding": tf_cfg["holding"],
            "timeframe": tf_cfg["label"],
            "gex_regime": gex_data["gex_regime"],
            "net_gex": gex_data["net_gex_cr"],
            "gamma_flip": gex_data["gamma_flip"],
            "max_pain": gex_data["max_pain"],
            "expected_move": expected_move,
            "confidence": 80,
            "why": f"Price aligned with Session VWAP ({RUPEE}{vwap:,.2f}), RSI {rsi:.1f}, and GEX: {gex_data['gex_regime']}.",
            "invalidation": f"Bar close beyond {RUPEE}{sl:,.2f} invalidates setup.",
            "trailing": "Book 50% at Target 1 and trail Stop Loss to Entry.",
        }

    elif instrument == "INDEX OPTION":
        if chain is None or chain.empty:
            return None

        side = option_side if option_side in ["CE", "PE"] else ("CE" if bullish else "PE")
        step = 50 if symbol in ["NIFTY", "FINNIFTY"] else 100
        target_strike = int(round(spot / step) * step)

        c_sub = chain[(chain["option_type"] == side) & (chain["strikePrice"] == target_strike)]
        if c_sub.empty:
            return None

        row = c_sub.iloc[0]
        ltp = num(row.get("lastPrice"))
        if ltp is None or ltp <= 0:
            return None  # Strict gate: No execution without traded LTP

        oi = int(num(row.get("openInterest"), 0) or 0)
        chg_oi = int(num(row.get("changeinOpenInterest"), 0) or 0)
        p_chg = float(num(row.get("change"), 0.0) or 0.0)
        iv = float(num(row.get("impliedVolatility"), 12.8) or 12.8)

        g_info = calculate_advanced_greeks(spot, target_strike, dte_days=5.0, rate=0.07, sigma=iv / 100.0, option_type=side)

        opt_entry = ltp
        opt_risk = max(round(opt_entry * 0.18, 2), 5.0)
        opt_sl = round(opt_entry - opt_risk, 2)
        opt_t1 = round(opt_entry + (opt_risk * 1.5), 2)
        opt_t2 = round(opt_entry + (opt_risk * 2.5), 2)

        valid, reason, sized_qty = risk_engine.validate_and_size_order(
            symbol=f"{symbol}_{target_strike}_{side}", instrument="INDEX OPTION", action="BUY",
            entry=opt_entry, sl=opt_sl, t1=opt_t1, lot_size=lot_size, confidence=80.0
        )
        if not valid:
            logger.info("Option trade rejected by risk engine: %s", reason)
            return None

        opt_rr = (opt_t1 - opt_entry) / opt_risk
        buildup = classify_oi_buildup(p_chg, chg_oi)

        return {
            "segment": "INDEX OPTION",
            "symbol": symbol,
            "action": "BUY",
            "option": side,
            "strike": target_strike,
            "entry": opt_entry,
            "sl": opt_sl,
            "target1": opt_t1,
            "target2": opt_t2,
            "quantity": sized_qty,
            "risk_reward": round(opt_rr, 2),
            "delta": round(g_info["delta"], 2),
            "gamma": round(g_info["gamma"], 5),
            "theta": round(g_info["theta"], 1),
            "vega": round(g_info["vega"], 1),
            "vanna": round(g_info["vanna"], 3),
            "iv": round(iv, 1),
            "oi": oi,
            "buildup": buildup,
            "holding": tf_cfg["holding"],
            "timeframe": tf_cfg["label"],
            "expiry": expiry_label(expiry),
            "confidence": 80,
            "why": f"{symbol} {target_strike} {side} selected. Delta: {g_info['delta']:.2f}, IV: {iv:.1f}%, Flow: {buildup}.",
            "invalidation": f"Premium close below {RUPEE}{opt_sl:,.2f} triggers risk liquidation.",
            "trailing": "Lock 50% profits at Target 1 and trail stop to breakeven.",
        }

    return None

# =========================================================
# APPLICATION DASHBOARD INTERFACE
# =========================================================
st.title(f"{BOLT} AI Institutional Live Trading Advisor (Quant GEX Engine)")
st.caption(f"Quantitative Market-Maker Architecture {BULLET} Verified Exchange Math {BULLET} Centralized Paper Risk Engine")

st.sidebar.header(f"{STOPWATCH} Strategy Timeframe")
selected_tf_key = st.sidebar.selectbox(
    "Candle Resolution / Timeframe",
    options=list(TIMEFRAME_CONFIG.keys()),
    index=1,
    format_func=lambda k: TIMEFRAME_CONFIG[k]["label"],
    help="Controls the exact candle aggregation and indicator evaluation horizon."
)
active_tf = TIMEFRAME_CONFIG[selected_tf_key]

st.sidebar.header(f"{GEAR} Trading Environment")
segment_mode = st.sidebar.radio("Active Market Segment", [f"{CHART_BAR} Index & Options Advisor", f"{CHART_UP} Equity / Share Research (NSE & BSE)"])

if st.sidebar.button(f"{REFRESH} Force Refresh Data Cache", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

col_t1, col_t2, col_t3, col_t4 = st.columns(4)
with col_t1:
    st.metric("System Mode", "PAPER SIMULATION ONLY", delta="Real Routing Disabled", delta_color="off")
with col_t2:
    st.metric("Market Status", "LIVE SESSION" if market_open() else "MARKET CLOSED", delta="Active 09:15-15:30" if market_open() else "Off-Hours", delta_color="normal" if market_open() else "off")
with col_t3:
    st.metric("Selected Timeframe", active_tf["smartapi"], delta=selected_tf_key.upper())
with col_t4:
    st.metric("Daily Paper P&L", f"{RUPEE}{risk_engine.daily_realized_pnl:+,.2f}", delta="Safety Locked" if risk_engine.safety_lock else "Active", delta_color="inverse" if risk_engine.safety_lock else "normal")

# ==============================================================================
# SEGMENT 1: EQUITY / SHARE RESEARCH
# ==============================================================================
if "Equity / Share Research" in segment_mode:
    st.markdown(f"## {CHART_UP} Universal Equity Stock Intelligence - [{active_tf['label']}]")
    st.caption("Exchange-Validated Pricing | Strict Volatility-Normalized Levels")

    eq_symbol = st.selectbox("Select Equity Benchmark", ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "SBIN"])
    spot_env = get_spot(eq_symbol)

    p1, p2, p3 = st.columns(3)
    with p1:
        st.metric(f"{eq_symbol} Traded Quote", fmt(spot_env.value), delta=spot_env.status.value, delta_color="normal" if spot_env.status == DataStatus.LIVE else "off")
    with p2:
        st.metric("Data Feed Source", spot_env.source)
    with p3:
        st.metric("Actionable Freshness", "VALID TICK" if spot_env.is_actionable() else "STALE / UNAVAILABLE", delta_color="normal" if spot_env.is_actionable() else "inverse")

    if not spot_env.is_actionable():
        st.warning(f"Live tick feed for {eq_symbol} is unavailable. Trade idea formulation disabled.")
    else:
        df_equity = add_indicators(fetch_ohlcv(eq_symbol, active_tf["smartapi"], active_tf["days"]))
        if not df_equity.empty:
            l_row = df_equity.iloc[-1]
            st.info(f"**Technical Structure:** RSI: {l_row.get('RSI', 50):.1f} | EMA20: {l_row.get('EMA20', 0):,.2f} | Session VWAP: {l_row.get('VWAP', 0):,.2f}")

# ==============================================================================
# SEGMENT 2: INDEX & OPTIONS ADVISOR
# ==============================================================================
else:
    underlying = st.sidebar.selectbox("Active Underlying Index", ["NIFTY", "BANKNIFTY", "FINNIFTY", "SENSEX"])
    expiries = load_expiries(underlying)
    selected_expiry = st.sidebar.selectbox("Target Expiry", options=expiries, format_func=expiry_label, index=0) if expiries else None

    spot_envelope = get_spot(underlying)
    chain_df, chain_status = get_chain(underlying, selected_expiry)
    lot_size = get_contract_lot_size(underlying)

    gex_data = compute_gex_and_max_pain(chain_df, spot_envelope.value, lot_size=lot_size, dte_days=5.0)
    pcr_val = calculate_pcr(chain_df)
    fii_dii_info = fetch_fii_dii()

    # Top Metric Bar
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric(f"{underlying} Verified Spot", fmt(spot_envelope.value), delta=spot_envelope.status.value, delta_color="normal" if spot_envelope.status == DataStatus.LIVE else "off")
    with m2:
        st.metric("Put-Call Ratio (PCR)", fmt(pcr_val) if pcr_val else "UNAVAILABLE", delta="Measured OI Ratio" if pcr_val else "No Feed", delta_color="off")
    with m3:
        st.metric("Dealer GEX Regime", gex_data["gex_regime"], delta=f"Net: {gex_data['net_gex_cr']:+,.0f} Cr" if gex_data["status"] == "VALID" else "Model Inactive")
    with m4:
        st.metric("Max Pain Pivot", fmt(gex_data["max_pain"]), delta="Expiry Gravitational Center", delta_color="off")

    st.divider()

    # Institutional Marker Radar
    st.info(f"""
    ⚖️ **Institutional Market Maker Radar:**
    * **Dealer GEX Regime:** `{gex_data['gex_regime']}` (Net GEX: `{gex_data['net_gex_cr']:+,.0f} Cr / 1% move`)
    * **Zero-Gamma Flip Level:** `{fmt(gex_data['gamma_flip'])}` | **Max Pain Target:** `{fmt(gex_data['max_pain'])}`
    * **Institutional Barriers:** Call Wall `{fmt(gex_data['call_wall'])}` | Put Wall `{fmt(gex_data['put_wall'])}` | $1\sigma$ Expected Move: `\u00B1{fmt(gex_data['expected_move'], 1)} Pts`
    * *Notice: Dealer positioning is a quantitative model estimate derived under standard open interest sign assumptions.*
    """)

    # Interactive Generation Trigger
    if st.button(f"{ROCKET} GENERATE RISK-VALIDATED SETUP", type="primary", use_container_width=True):
        if not spot_envelope.is_actionable():
            st.error(f"{STOP} Cannot formulate setups: Underlying market feed is {spot_envelope.status.value}.")
        else:
            with st.spinner("Executing analytical Greek models and centralized risk validation..."):
                spot_setup = make_trade_idea(underlying, spot_envelope, instrument="INDEX", expiry=selected_expiry, chain=chain_df, tf_cfg=active_tf)
                option_setup = make_trade_idea(underlying, spot_envelope, instrument="INDEX OPTION", expiry=selected_expiry, chain=chain_df, tf_cfg=active_tf)

                valid_setups = [s for s in [spot_setup, option_setup] if s is not None]
                if not valid_setups:
                    st.warning("No setup meets quantitative confluence and centralized risk engine criteria.")
                else:
                    for i, idea in enumerate(valid_setups, 1):
                        is_opt = (idea["segment"] == "INDEX OPTION")
                        title = f"{idea['symbol']} {idea.get('strike', '')} {idea.get('option', '')}" if is_opt else f"{idea['symbol']} SPOT"
                        st.markdown(
                            f"""
                            <div class="trade-card">
                                <h3>Setup {i}: {title} ({idea['action']})</h3>
                                <b>Risk-Approved Quantity:</b> {idea['quantity']} units ({idea['quantity'] // lot_size} lots) | 
                                <b>Risk/Reward:</b> 1:{idea['risk_reward']} | <b>Conviction:</b> {idea['confidence']}%
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        c1, c2, c3, c4 = st.columns(4)
                        with c1: st.metric("Entry Level", f"{RUPEE}{fmt(idea['entry'])}")
                        with c2: st.metric("Stop Loss (SL)", f"{RUPEE}{fmt(idea['sl'])}")
                        with c3: st.metric("Target 1", f"{RUPEE}{fmt(idea['target1'])}")
                        with c4: st.metric("Target 2", f"{RUPEE}{fmt(idea['target2'])}")

                        if is_opt:
                            g1, g2, g3, g4 = st.columns(4)
                            with g1: st.metric(f"Delta ({DELTA_SYM})", f"{idea['delta']:.2f}")
                            with g2: st.metric(f"Gamma ({GAMMA_SYM})", f"{idea['gamma']:.5f}")
                            with g3: st.metric(f"Theta ({THETA_SYM})", f"{idea['theta']:.1f}")
                            with g4: st.metric("Vega", f"{idea['vega']:.1f}")

                        st.markdown(f'<div class="reason-box"><b>Institutional Logic:</b><br>{idea["why"]}</div>', unsafe_allow_html=True)
                        st.write(f"{STOP} **Invalidation:** {idea['invalidation']}")
                        st.write(f"{CHART_UP} **Trailing Guidance:** {idea['trailing']}")
                        st.divider()

st.caption(f"Paper Trading Simulation Only {BULLET} Real Order Routing Disabled {BULLET} Strict Educational Analytics.")
