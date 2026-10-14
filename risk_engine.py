# -*- coding: utf-8 -*-
"""
Centralized Paper Trading Risk Engine with File-Backed Atomic State Persistence.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime as dt, date
import json
import logging
import os
import threading
from typing import Dict, Any, Optional, Tuple, List

logger = logging.getLogger("RiskEngine")

@dataclass
class PaperRiskLimits:
    max_loss_per_trade_inr: float = 5000.0
    max_daily_loss_inr: float = 15000.0
    max_trades_per_day: int = 6
    max_capital_allocation_pct: float = 0.25
    total_paper_capital_inr: float = 200000.0

@dataclass
class PaperTradeOrder:
    trade_id: str
    symbol: str
    instrument: str
    action: str
    entry_price: float
    stop_loss: float
    target1: float
    target2: float
    quantity: int
    lot_size: int
    timestamp: str
    pnl: float = 0.0
    status: str = "OPEN"

class CentralizedRiskEngine:
    def __init__(self, limits: Optional[PaperRiskLimits] = None, state_file: str = "paper_trading_state.json"):
        self.limits = limits or PaperRiskLimits()
        self.state_file = state_file
        self.lock = threading.Lock()
        self.daily_realized_pnl: float = 0.0
        self.safety_lock: bool = False
        self.trade_history: List[Dict[str, Any]] = []
        self.active_positions: Dict[str, Dict[str, Any]] = {}
        self.last_reset_date: str = dt.now().strftime("%Y-%m-%d")
        self._load_state()

    def _load_state(self):
        with self.lock:
            if os.path.exists(self.state_file):
                try:
                    with open(self.state_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        self.daily_realized_pnl = float(data.get("daily_realized_pnl", 0.0))
                        self.safety_lock = bool(data.get("safety_lock", False))
                        self.last_reset_date = str(data.get("last_reset_date", dt.now().strftime("%Y-%m-%d")))
                        self.trade_history = data.get("trade_history", [])
                        self.active_positions = data.get("active_positions", {})
                except Exception as exc:
                    logger.error("Failed to load paper state file: %s", exc)
            self._check_and_reset_daily()

    def _persist_state(self):
        data = {
            "daily_realized_pnl": self.daily_realized_pnl,
            "safety_lock": self.safety_lock,
            "last_reset_date": self.last_reset_date,
            "trade_history": self.trade_history,
            "active_positions": self.active_positions,
        }
        try:
            temp_file = f"{self.state_file}.tmp"
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            os.replace(temp_file, self.state_file)
        except Exception as exc:
            logger.error("Failed to persist paper state file: %s", exc)

    def _check_and_reset_daily(self):
        today_str = dt.now().strftime("%Y-%m-%d")
        if today_str > self.last_reset_date:
            self.daily_realized_pnl = 0.0
            self.active_positions.clear()
            self.safety_lock = False
            self.last_reset_date = today_str
            self._persist_state()
            logger.info("Daily risk engine limits reset for %s", today_str)

    def validate_and_size_order(
        self,
        symbol: str,
        instrument: str,
        action: str,
        entry: float,
        sl: float,
        t1: float,
        lot_size: int,
        confidence: float
    ) -> Tuple[bool, str, int]:
        with self.lock:
            self._check_and_reset_daily()

            if self.safety_lock:
                return False, "SAFETY LOCK ENGAGED: Cumulative daily loss exceeded.", 0

            today_trades = [t for t in self.trade_history if t.get("timestamp", "").startswith(self.last_reset_date)]
            if len(today_trades) >= self.limits.max_trades_per_day:
                return False, f"TRADE COUNT EXCEEDED: Max {self.limits.max_trades_per_day} trades reached.", 0

            # Mathematical Geometric Integrity
            if action == "BUY":
                if not (sl < entry < t1):
                    return False, f"GEOMETRY ERROR (BUY): SL ({sl}) must be < Entry ({entry}) < Target ({t1}).", 0
                unit_risk = entry - sl
            elif action == "SELL":
                if not (sl > entry > t1):
                    return False, f"GEOMETRY ERROR (SELL): SL ({sl}) must be > Entry ({entry}) > Target ({t1}).", 0
                unit_risk = sl - entry
            else:
                return False, f"INVALID ACTION: {action}", 0

            if unit_risk <= 0:
                return False, "INVALID RISK: Distance between Entry and Stop Loss is non-positive.", 0

            # Position Sizing
            max_risk_allowed = self.limits.max_loss_per_trade_inr
            risk_per_lot = unit_risk * lot_size
            if risk_per_lot > max_risk_allowed:
                return False, f"LOT RISK TOO HIGH: 1 lot risk (₹{risk_per_lot:,.2f}) > Max allowed (₹{max_risk_allowed:,.2f}).", 0

            num_lots = int(max_risk_allowed // risk_per_lot)
            sized_qty = num_lots * lot_size

            # Capital Allocation Cap
            total_cost = entry * sized_qty
            max_alloc = self.limits.total_paper_capital_inr * self.limits.max_capital_allocation_pct
            if total_cost > max_alloc:
                num_lots = int(max_alloc // (entry * lot_size))
                sized_qty = num_lots * lot_size

            if sized_qty < lot_size:
                return False, "CAPITAL CAP EXCEEDED: Cannot allocate minimum 1 lot safely.", 0

            # Duplicate Position Check
            if symbol in self.active_positions:
                return False, f"DUPLICATE TRADE: Open position already exists for {symbol}.", 0

            return True, "VALIDATED", sized_qty

    def record_trade_execution(self, order: PaperTradeOrder):
        with self.lock:
            ord_dict = asdict(order)
            self.active_positions[order.symbol] = ord_dict
            self.trade_history.append(ord_dict)
            self._persist_state()

    def record_pnl(self, symbol: str, realized_pnl: float):
        with self.lock:
            self.daily_realized_pnl += realized_pnl
            if symbol in self.active_positions:
                del self.active_positions[symbol]

            if self.daily_realized_pnl <= -self.limits.max_daily_loss_inr:
                self.safety_lock = True
                logger.warning("SAFETY LOCK TRIGGERED: Daily loss at ₹%.2f", self.daily_realized_pnl)

            self._persist_state()
