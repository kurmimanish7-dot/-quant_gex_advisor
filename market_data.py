# -*- coding: utf-8 -*-
"""
Market Data Contracts, Instrument Specifications, and Secure Session Protocols.
"""

from dataclasses import dataclass
from datetime import datetime as dt, date, time as dtime, timedelta
from enum import Enum
import logging
import time
from typing import Optional, Dict, Any
import requests
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")
logger = logging.getLogger("MarketData")

class DataStatus(str, Enum):
    LIVE = "LIVE"
    DELAYED = "DELAYED"
    HISTORICAL = "HISTORICAL"
    STALE = "STALE"
    UNAVAILABLE = "UNAVAILABLE"

@dataclass(frozen=True)
class MarketDataEnvelope:
    symbol: str
    instrument_token: str
    exchange: str
    value: Optional[float]
    source: str
    status: DataStatus
    exchange_timestamp: Optional[dt]
    fetch_timestamp: dt
    error_reason: Optional[str] = None

    def is_actionable(self, max_staleness_sec: float = 30.0) -> bool:
        if self.status != DataStatus.LIVE or self.value is None or self.value <= 0:
            return False
        age = (dt.now(self.fetch_timestamp.tzinfo) - self.fetch_timestamp).total_seconds()
        return age <= max_staleness_sec

# Official Exchange Lot Size Master
CONTRACT_MASTER = {
    "NIFTY": {"lot_size": 65, "exchange": "NSE", "candle_token": "99926000", "yfinance": "^NSEI"},
    "BANKNIFTY": {"lot_size": 30, "exchange": "NSE", "candle_token": "99926009", "yfinance": "^NSEBANK"},
    "FINNIFTY": {"lot_size": 60, "exchange": "NSE", "candle_token": "99926037", "yfinance": "NIFTY_FIN_SERVICE.NS"},
    "MIDCPNIFTY": {"lot_size": 120, "exchange": "NSE", "candle_token": "99926074", "yfinance": "NIFTY_MIDCAP_100.NS"},
    "SENSEX": {"lot_size": 20, "exchange": "BSE", "candle_token": "99919000", "yfinance": "^BSESN"},
}

def get_contract_lot_size(symbol: str) -> int:
    return CONTRACT_MASTER.get(symbol.upper(), {}).get("lot_size", 65)

NSE_TRADING_HOLIDAYS = {
    date(2025, 1, 26), date(2025, 2, 26), date(2025, 3, 14), date(2025, 3, 31),
    date(2025, 4, 10), date(2025, 4, 14), date(2025, 4, 18), date(2025, 5, 1),
    date(2025, 8, 15), date(2025, 8, 27), date(2025, 10, 2), date(2025, 10, 21),
    date(2025, 11, 1), date(2025, 11, 5), date(2025, 12, 25),
    date(2026, 1, 26), date(2026, 3, 3), date(2026, 3, 26), date(2026, 3, 31),
    date(2026, 4, 3), date(2026, 4, 14), date(2026, 5, 1), date(2026, 5, 28),
    date(2026, 6, 26), date(2026, 9, 14), date(2026, 10, 2), date(2026, 10, 20),
    date(2026, 11, 10), date(2026, 11, 24), date(2026, 12, 25),
}

def is_trading_holiday(d: date) -> bool:
    if d.weekday() >= 5:
        return True
    return d in NSE_TRADING_HOLIDAYS

def get_preceding_trading_day(d: date) -> date:
    candidate = d
    while is_trading_holiday(candidate):
        candidate -= timedelta(days=1)
    return candidate

def market_open() -> bool:
    now = dt.now(IST)
    if is_trading_holiday(now.date()):
        return False
    return dtime(9, 15) <= now.time() <= dtime(15, 30)

class NSEExchangeSession:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/126.0.0.0 Safari/537.36",
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9",
        })
        self.last_warm = 0.0

    def warm_cookies(self):
        now = time.time()
        if now - self.last_warm > 120 or len(self.session.cookies) == 0:
            try:
                self.session.get("https://www.nseindia.com", timeout=5)
                self.session.get("https://www.nseindia.com/option-chain", timeout=5)
                self.last_warm = now
            except Exception as exc:
                logger.warning("Cookie handshake failed: %s", exc)

    def get_json(self, url: str) -> Optional[Dict[str, Any]]:
        self.warm_cookies()
        try:
            resp = self.session.get(url, timeout=5)
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code in [401, 403]:
                self.last_warm = 0.0
                self.warm_cookies()
                retry = self.session.get(url, timeout=5)
                if retry.status_code == 200:
                    return retry.json()
        except Exception as exc:
            logger.warning("Query failed on %s: %s", url, exc)
        return None
