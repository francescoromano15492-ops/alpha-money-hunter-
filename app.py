# Alpha v0.7.1 phone-friendly single-file bootstrap.
# Reconstructs the internal `alpha` package at runtime.
from pathlib import Path as _AlphaPath
import tempfile as _alpha_tempfile
import sys as _alpha_sys

_ALPHA_SOURCES = {'__init__.py': '', 'backtest.py': 'from __future__ import annotations\n\nfrom dataclasses import dataclass, asdict\nfrom typing import List\nimport math\nimport numpy as np\nimport pandas as pd\n\nfrom .config import RiskConfig\nfrom .strategies import scan_at, precompute_indicators\nfrom .skeptic import review_at\nfrom .data_quality import observations_per_year, validate_ohlcv\n\n\n@dataclass\nclass ClosedTrade:\n    symbol: str\n    strategy: str\n    side: str\n    signal_time: str\n    entry_time: str\n    exit_time: str\n    entry: float\n    exit: float\n    stop: float\n    target: float\n    quantity: float\n    initial_risk: float\n    pnl: float\n    r_multiple: float\n    return_pct: float\n    hold_bars: int\n    exit_reason: str\n\n    def to_dict(self):\n        return asdict(self)\n\n\n@dataclass\nclass BacktestResult:\n    symbol: str\n    starting_equity: float\n    ending_equity: float\n    trades: List[ClosedTrade]\n    equity_curve: pd.DataFrame\n    benchmark_curve: pd.DataFrame\n    metrics: dict\n\n\ndef _fees(value: float, bps: float) -> float:\n    return abs(value) * bps / 10_000.0\n\n\ndef _sell_fill(base_price: float, slippage_bps: float) -> float:\n    return float(base_price) * (1 - slippage_bps / 10_000.0)\n\n\ndef _max_drawdown(series: pd.Series) -> float:\n    if len(series) == 0:\n        return 0.0\n    peak = series.cummax()\n    return float((series / peak - 1.0).min())\n\n\ndef _resolve_long_exit(bar: pd.Series, stop: float, target: float, slippage_bps: float, entered_this_bar: bool = False):\n    """Conservative OHLC execution. Stop gaps fill at the worse opening price.\n    Profit gaps do not receive optimistic price improvement. If stop and target are\n    both touched intrabar, stop is assumed first.\n    """\n    o, h, l = float(bar["open"]), float(bar["high"]), float(bar["low"])\n    if not entered_this_bar:\n        if o <= stop:\n            return _sell_fill(o, slippage_bps), "STOP_GAP"\n        if o >= target:\n            return _sell_fill(target, slippage_bps), "TARGET_GAP"\n    stop_hit = l <= stop\n    target_hit = h >= target\n    if stop_hit:\n        return _sell_fill(stop, slippage_bps), "STOP_SAME_BAR" if entered_this_bar else "STOP"\n    if target_hit:\n        return _sell_fill(target, slippage_bps), "TARGET_SAME_BAR" if entered_this_bar else "TARGET"\n    return None, None\n\n\ndef _bootstrap_r_stats(trades: list[ClosedTrade], samples: int = 3000, seed: int = 17) -> dict:\n    r = np.array([t.r_multiple for t in trades if np.isfinite(t.r_multiple)], dtype=float)\n    observed = float(r.mean()) if len(r) else 0.0\n    if samples <= 0 or len(r) < 2:\n        return {\n            "mean_r": observed,\n            "bootstrap_mean_r_ci_low": 0.0,\n            "bootstrap_mean_r_ci_high": 0.0,\n            "bootstrap_prob_mean_r_positive": 0.0,\n        }\n    rng = np.random.default_rng(seed)\n    means = rng.choice(r, size=(samples, len(r)), replace=True).mean(axis=1)\n    return {\n        "mean_r": observed,\n        "bootstrap_mean_r_ci_low": float(np.quantile(means, 0.025)),\n        "bootstrap_mean_r_ci_high": float(np.quantile(means, 0.975)),\n        "bootstrap_prob_mean_r_positive": float((means > 0).mean()),\n    }\n\n\ndef _metrics(starting_equity, ending_equity, curve, trades, benchmark_curve, bootstrap_samples: int = 3000):\n    pnls = np.array([t.pnl for t in trades], dtype=float)\n    wins = pnls[pnls > 0]\n    losses = pnls[pnls < 0]\n    gross_profit = float(wins.sum()) if len(wins) else 0.0\n    gross_loss = float(abs(losses.sum())) if len(losses) else 0.0\n    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (math.inf if gross_profit > 0 else 0.0)\n    win_rate = float((pnls > 0).mean()) if len(pnls) else 0.0\n    expectancy = float(pnls.mean()) if len(pnls) else 0.0\n\n    eq = curve["equity"]\n    returns = eq.pct_change().dropna()\n    ppy = observations_per_year(eq.index)\n    sharpe = 0.0\n    sortino = 0.0\n    if len(returns) > 5 and returns.std(ddof=0) > 0:\n        sharpe = float(np.sqrt(ppy) * returns.mean() / returns.std(ddof=0))\n        downside = returns[returns < 0]\n        if len(downside) > 2 and downside.std(ddof=0) > 0:\n            sortino = float(np.sqrt(ppy) * returns.mean() / downside.std(ddof=0))\n\n    max_dd = abs(_max_drawdown(eq))\n    elapsed_years = max((pd.DatetimeIndex(eq.index)[-1] - pd.DatetimeIndex(eq.index)[0]).total_seconds() / (365.25*86400), 1/365.25) if len(eq) > 1 else 1/365.25\n    cagr = float((ending_equity / starting_equity) ** (1 / elapsed_years) - 1) if ending_equity > 0 else -1.0\n    calmar = cagr / max_dd if max_dd > 0 else (math.inf if cagr > 0 else 0.0)\n\n    benchmark_return = float(benchmark_curve["benchmark"].iloc[-1] / benchmark_curve["benchmark"].iloc[0] - 1) if len(benchmark_curve) > 1 else 0.0\n    total_return = float(ending_equity / starting_equity - 1)\n    exposure_pct = float(curve["exposure"].mean() * 100) if "exposure" in curve else 0.0\n    turnover_x = float(sum((t.entry + t.exit) * t.quantity for t in trades) / starting_equity) if starting_equity > 0 else 0.0\n    avg_hold = float(np.mean([t.hold_bars for t in trades])) if trades else 0.0\n    boot = _bootstrap_r_stats(trades, samples=bootstrap_samples)\n\n    return {\n        "total_return_pct": total_return * 100,\n        "benchmark_return_pct": benchmark_return * 100,\n        "alpha_vs_benchmark_pct": (total_return - benchmark_return) * 100,\n        "cagr_pct": cagr * 100,\n        "max_drawdown_pct": max_dd * 100,\n        "sharpe": sharpe,\n        "sortino": sortino,\n        "calmar": calmar,\n        "observations_per_year": ppy,\n        "trades": int(len(trades)),\n        "win_rate_pct": win_rate * 100,\n        "profit_factor": profit_factor,\n        "expectancy_currency": expectancy,\n        "exposure_pct": exposure_pct,\n        "turnover_x": turnover_x,\n        "avg_hold_bars": avg_hold,\n        **boot,\n    }\n\n\ndef _trailing_adv(df: pd.DataFrame, i: int, n: int = 20) -> float | None:\n    if "volume" not in df or i <= 0:\n        return None\n    hist = df.iloc[max(0, i-n):i]\n    if len(hist) < 5:\n        return None\n    dv = (hist["close"] * hist["volume"]).replace([np.inf, -np.inf], np.nan).dropna()\n    dv = dv[dv > 0]\n    return float(dv.median()) if len(dv) else None\n\n\ndef run_backtest(\n    symbol: str,\n    df: pd.DataFrame,\n    cfg: RiskConfig,\n    warmup: int = 60,\n    strategy_names: set[str] | list[str] | tuple[str, ...] | None = None,\n    *,\n    precomputed_indicators: pd.DataFrame | None = None,\n    precomputed_review_context: pd.DataFrame | None = None,\n    prevalidated: bool = False,\n    bootstrap_samples: int = 3000,\n) -> BacktestResult:\n    """Long-only event-driven research backtest.\n\n    Signals are generated at bar close t and can only enter at bar open t+1.\n    Same-bar exits after entry are modeled; overnight stop gaps fill at the worse\n    opening price. Drawdown kill switch latches for the rest of the run.\n    """\n    if prevalidated:\n        if not df.index.is_monotonic_increasing:\n            raise ValueError("prevalidated backtest input must be sorted")\n        data_quality_passed = True\n    else:\n        df = df.sort_index().copy()\n        q = validate_ohlcv(df)\n        if not q.passed:\n            raise ValueError(f"Data quality failed: {q.notes}")\n        data_quality_passed = q.passed\n    if len(df) < warmup + 10:\n        raise ValueError("Not enough data for backtest")\n\n    realized_equity = cfg.starting_equity\n    peak_equity = realized_equity\n    last_marked_equity = realized_equity\n    current_day = None\n    daily_anchor = realized_equity\n    daily_halted = False\n    drawdown_halted = False\n    kill_flatten_pending = False\n    trades: List[ClosedTrade] = []\n    curve_rows = []\n    open_trade = None\n    pending = None\n    indicators = precomputed_indicators\n    if indicators is None:\n        indicators = precompute_indicators(df)\n    else:\n        if not indicators.index.equals(df.index):\n            indicators = indicators.reindex(df.index)\n        if len(indicators) != len(df) or indicators.isna().all(axis=None):\n            raise ValueError("precomputed indicators do not align with backtest input")\n\n    review_context = precomputed_review_context\n    if review_context is not None and not review_context.index.equals(df.index):\n        review_context = review_context.reindex(df.index)\n\n    allowed = None if strategy_names is None else set(strategy_names)\n\n    for i in range(warmup, len(df)):\n        ts = df.index[i]\n        bar = df.iloc[i]\n        day = ts.date()\n        if current_day != day:\n            current_day = day\n            daily_anchor = last_marked_equity\n            daily_halted = False\n\n        # 1) A latched risk kill liquidates existing exposure at the next executable open.\n        entered_this_bar = False\n        if kill_flatten_pending and open_trade is not None and cfg.flatten_on_risk_kill:\n            exit_px = _sell_fill(float(bar["open"]), cfg.simulated_slippage_bps)\n            gross = (exit_px - open_trade["entry"]) * open_trade["qty"]\n            exit_fee = _fees(exit_px * open_trade["qty"], cfg.simulated_fee_bps)\n            realized_equity += gross - exit_fee\n            trade_pnl = gross - exit_fee - open_trade["entry_fee"]\n            ir = max(open_trade["initial_risk"], 1e-12)\n            trades.append(ClosedTrade(\n                symbol=symbol, strategy=open_trade["strategy"], side="BUY",\n                signal_time=str(open_trade["signal_time"]), entry_time=str(open_trade["entry_time"]), exit_time=str(ts),\n                entry=open_trade["entry"], exit=exit_px, stop=open_trade["stop"], target=open_trade["target"],\n                quantity=open_trade["qty"], initial_risk=open_trade["initial_risk"], pnl=trade_pnl, r_multiple=trade_pnl/ir,\n                return_pct=(exit_px/open_trade["entry"]-1)*100, hold_bars=i-open_trade["entry_i"]+1, exit_reason="RISK_KILL",\n            ))\n            open_trade = None\n        kill_flatten_pending = False\n\n        # 2) Execute yesterday\'s frozen signal at today\'s open if the gap did not stale it.\n        if pending is not None and open_trade is None and not drawdown_halted and not daily_halted:\n            sig = pending["signal"]\n            signal_i = pending["signal_i"]\n            raw_entry = float(bar["open"])\n            intended_risk = abs(sig.entry - sig.stop)\n            gap_units = abs(raw_entry - sig.entry) / intended_risk if intended_risk > 0 else math.inf\n            if intended_risk > 0 and gap_units <= cfg.max_entry_gap_risk_units and raw_entry > sig.stop:\n                entry = raw_entry * (1 + cfg.simulated_slippage_bps / 10_000.0)\n                stop = float(sig.stop)\n                risk_distance = entry - stop\n                if risk_distance > 0:\n                    target = entry + risk_distance * max(sig.reward_risk, cfg.min_reward_risk)\n                    marked_equity_for_size = last_marked_equity\n                    risk_budget = marked_equity_for_size * cfg.max_risk_per_trade_pct\n                    qty_risk = risk_budget / risk_distance\n                    qty_value = (marked_equity_for_size * cfg.max_position_value_pct) / entry\n                    adv = _trailing_adv(df, i)\n                    qty_adv = (adv * cfg.max_adv_participation_pct / entry) if adv and adv > 0 else math.inf\n                    qty = max(0.0, min(qty_risk, qty_value, qty_adv))\n                    if qty > 0:\n                        entry_fee = _fees(entry * qty, cfg.simulated_fee_bps)\n                        realized_equity -= entry_fee\n                        open_trade = {\n                            "strategy": sig.strategy, "signal_time": pending["signal_time"],\n                            "entry_time": ts, "entry_i": i, "entry": entry, "stop": stop,\n                            "target": target, "qty": qty, "entry_fee": entry_fee,\n                            "initial_risk": risk_distance * qty,\n                        }\n                        entered_this_bar = True\n            pending = None\n\n        # 3) Manage an open trade using today\'s bar, including same-day entry exits.\n        if open_trade is not None:\n            exit_px, reason = _resolve_long_exit(bar, open_trade["stop"], open_trade["target"], cfg.simulated_slippage_bps, entered_this_bar=entered_this_bar)\n            if exit_px is not None:\n                gross = (exit_px - open_trade["entry"]) * open_trade["qty"]\n                exit_fee = _fees(exit_px * open_trade["qty"], cfg.simulated_fee_bps)\n                realized_equity += gross - exit_fee\n                trade_pnl = gross - exit_fee - open_trade["entry_fee"]\n                initial_risk = max(open_trade["initial_risk"], 1e-12)\n                trades.append(ClosedTrade(\n                    symbol=symbol, strategy=open_trade["strategy"], side="BUY",\n                    signal_time=str(open_trade["signal_time"]), entry_time=str(open_trade["entry_time"]),\n                    exit_time=str(ts), entry=open_trade["entry"], exit=exit_px,\n                    stop=open_trade["stop"], target=open_trade["target"], quantity=open_trade["qty"],\n                    initial_risk=open_trade["initial_risk"], pnl=trade_pnl,\n                    r_multiple=trade_pnl / initial_risk,\n                    return_pct=(exit_px / open_trade["entry"] - 1) * 100,\n                    hold_bars=i - open_trade["entry_i"] + 1, exit_reason=reason,\n                ))\n                open_trade = None\n\n        # 4) Mark to market and update hard latches.\n        marked = realized_equity\n        if open_trade is not None:\n            marked += (float(bar["close"]) - open_trade["entry"]) * open_trade["qty"]\n        peak_equity = max(peak_equity, marked)\n        dd = (peak_equity - marked) / peak_equity if peak_equity > 0 else 0.0\n        if dd >= cfg.max_drawdown_pct:\n            drawdown_halted = True\n            if open_trade is not None:\n                kill_flatten_pending = True\n        if marked - daily_anchor <= -daily_anchor * cfg.max_daily_loss_pct:\n            daily_halted = True\n            if open_trade is not None:\n                kill_flatten_pending = True\n        curve_rows.append((ts, marked, 1.0 if open_trade is not None else 0.0, drawdown_halted, daily_halted))\n        last_marked_equity = marked\n\n        # 5) Generate a signal only at this close, for possible entry next bar.\n        if i < len(df) - 1 and open_trade is None and pending is None and not drawdown_halted and not daily_halted:\n            candidates = scan_at(symbol, df, i, indicators, allowed_strategies=allowed)\n            approved = []\n            for sig in candidates:\n                ok, _ = review_at(sig, df, i, review_context)\n                if ok and sig.reward_risk >= cfg.min_reward_risk:\n                    approved.append(sig)\n            if approved:\n                approved.sort(key=lambda s: (s.quality_score * min(s.reward_risk, 4.0)), reverse=True)\n                pending = {"signal": approved[0], "signal_time": ts, "signal_i": i}\n\n    # Force-close at last close only for accounting comparability.\n    if open_trade is not None:\n        ts = df.index[-1]\n        px = _sell_fill(float(df.iloc[-1]["close"]), cfg.simulated_slippage_bps)\n        gross = (px - open_trade["entry"]) * open_trade["qty"]\n        exit_fee = _fees(px * open_trade["qty"], cfg.simulated_fee_bps)\n        realized_equity += gross - exit_fee\n        trade_pnl = gross - exit_fee - open_trade["entry_fee"]\n        initial_risk = max(open_trade["initial_risk"], 1e-12)\n        trades.append(ClosedTrade(\n            symbol=symbol, strategy=open_trade["strategy"], side="BUY",\n            signal_time=str(open_trade["signal_time"]), entry_time=str(open_trade["entry_time"]),\n            exit_time=str(ts), entry=open_trade["entry"], exit=px, stop=open_trade["stop"], target=open_trade["target"],\n            quantity=open_trade["qty"], initial_risk=open_trade["initial_risk"], pnl=trade_pnl,\n            r_multiple=trade_pnl / initial_risk, return_pct=(px / open_trade["entry"] - 1) * 100,\n            hold_bars=len(df) - open_trade["entry_i"], exit_reason="END_OF_TEST",\n        ))\n        if curve_rows:\n            curve_rows[-1] = (curve_rows[-1][0], realized_equity, 0.0, drawdown_halted, daily_halted)\n\n    curve = pd.DataFrame(curve_rows, columns=["timestamp", "equity", "exposure", "drawdown_halted", "daily_halted"]).drop_duplicates("timestamp").set_index("timestamp")\n    if curve.empty:\n        curve = pd.DataFrame({"equity": [cfg.starting_equity], "exposure": [0.0]}, index=[df.index[warmup]])\n\n    first_idx = curve.index[0]\n    start_close = float(df.loc[first_idx, "close"]) if first_idx in df.index else float(df["close"].iloc[warmup])\n    bh = cfg.starting_equity * (df["close"] / start_close)\n    benchmark = pd.DataFrame({"benchmark": bh.reindex(curve.index).ffill().bfill()})\n    metrics = _metrics(\n        cfg.starting_equity, realized_equity, curve, trades, benchmark,\n        bootstrap_samples=bootstrap_samples,\n    )\n    metrics["data_quality_passed"] = bool(data_quality_passed)\n    return BacktestResult(symbol, cfg.starting_equity, realized_equity, trades, curve, benchmark, metrics)\n', 'config.py': 'from dataclasses import dataclass\n\n@dataclass(frozen=True)\nclass RiskConfig:\n    # Research capital only. v0.4 has no live order routing.\n    starting_equity: float = 10_000.0\n\n    # Position / portfolio risk\n    max_risk_per_trade_pct: float = 0.005       # 0.5% of marked equity\n    max_portfolio_heat_pct: float = 0.015       # 1.5% total open stop-risk\n    max_daily_loss_pct: float = 0.02            # latch for the rest of the day\n    max_drawdown_pct: float = 0.08              # latch for the rest of the run\n    flatten_on_risk_kill: bool = True           # exit existing risk at next executable open\n    max_position_value_pct: float = 0.20        # 20% per position\n    max_open_positions: int = 4\n    max_leverage: float = 1.0                    # long-only, no leverage\n    min_reward_risk: float = 1.5\n\n    # Execution realism\n    simulated_fee_bps: float = 5.0\n    simulated_slippage_bps: float = 8.0\n    max_entry_gap_risk_units: float = 0.75       # do not chase stale signals\n    max_adv_participation_pct: float = 0.01      # <=1% of trailing $ADV\n\n    # Diversification\n    correlation_lookback: int = 60\n    max_pair_correlation: float = 0.85\n\n    # Evidence gate\n    min_oos_folds: int = 4\n    min_oos_trades: int = 30\n    min_positive_alpha_folds: float = 0.60\n    min_bootstrap_prob_positive: float = 0.90\n', 'data_provider.py': 'from __future__ import annotations\n\nfrom dataclasses import dataclass\nfrom datetime import date, timedelta, datetime, timezone\nfrom io import StringIO\nfrom typing import Dict, Iterable\n\nimport pandas as pd\nimport requests\n\n\nclass MarketDataError(RuntimeError):\n    pass\n\n\n@dataclass(frozen=True)\nclass DataRequest:\n    symbol: str\n    start: date\n    end: date\n\n\nclass StooqDailyProvider:\n    """Broker-independent daily market-data provider.\n\n    Primary source: Stooq. Automatic fallback: Yahoo Finance chart endpoint.\n    The public UI keeps accepting Stooq-style US symbols such as SPY.US; the\n    Yahoo fallback transparently maps those to SPY.\n    This provider is data-only and has no order-routing capability.\n    """\n\n    stooq_url = "https://stooq.com/q/d/l/"\n    yahoo_url = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"\n    headers = {\n        "User-Agent": "Mozilla/5.0 (compatible; AlphaMoneyHunter/0.6; research-only)"\n    }\n\n    def __init__(self):\n        self.last_errors: list[str] = []\n        self.last_source: str | None = None\n        self.source_by_symbol: dict[str, str] = {}\n        self._stooq_unavailable = False\n        self.session = requests.Session()\n\n    @staticmethod\n    def _clean(df: pd.DataFrame, symbol: str) -> pd.DataFrame:\n        required = ["open", "high", "low", "close"]\n        for col in [*required, "volume"]:\n            if col not in df.columns:\n                df[col] = 0.0 if col == "volume" else pd.NA\n            df[col] = pd.to_numeric(df[col], errors="coerce")\n        df = df.sort_index()\n        df = df[~df.index.duplicated(keep="last")]\n        df = df.dropna(subset=required)\n        if len(df) < 60:\n            raise MarketDataError(f"Too little history returned for {symbol} ({len(df)} rows)")\n        return df[["open", "high", "low", "close", "volume"]]\n\n    def _fetch_stooq(self, req: DataRequest) -> pd.DataFrame:\n        params = {\n            "s": req.symbol.strip().lower(),\n            "i": "d",\n            "d1": req.start.strftime("%Y%m%d"),\n            "d2": req.end.strftime("%Y%m%d"),\n        }\n        r = self.session.get(self.stooq_url, params=params, headers=self.headers, timeout=20)\n        r.raise_for_status()\n        text = r.text.strip()\n        if not text or "No data" in text or text.startswith("Get your apikey"):\n            raise MarketDataError(f"No Stooq data returned for {req.symbol}")\n        df = pd.read_csv(StringIO(text))\n        required = {"Date", "Open", "High", "Low", "Close"}\n        if not required.issubset(df.columns):\n            raise MarketDataError(f"Unexpected Stooq columns for {req.symbol}: {list(df.columns)}")\n        if "Volume" not in df.columns:\n            df["Volume"] = 0.0\n        df = df.rename(columns={\n            "Date": "timestamp", "Open": "open", "High": "high",\n            "Low": "low", "Close": "close", "Volume": "volume",\n        })\n        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)\n        df = df.set_index("timestamp")\n        return self._clean(df, req.symbol)\n\n    @staticmethod\n    def _to_yahoo_symbol(symbol: str) -> str:\n        s = symbol.strip().upper()\n        if s.endswith(".US"):\n            s = s[:-3]\n        return s.replace(".", "-") if s.count(".") == 1 and not s.endswith((".L", ".PA", ".DE")) else s\n\n    def _fetch_yahoo(self, req: DataRequest) -> pd.DataFrame:\n        symbol = self._to_yahoo_symbol(req.symbol)\n        start_dt = datetime(req.start.year, req.start.month, req.start.day, tzinfo=timezone.utc)\n        # Yahoo period2 is exclusive; include the requested end date.\n        end_plus = req.end + timedelta(days=1)\n        end_dt = datetime(end_plus.year, end_plus.month, end_plus.day, tzinfo=timezone.utc)\n        params = {\n            "period1": int(start_dt.timestamp()),\n            "period2": int(end_dt.timestamp()),\n            "interval": "1d",\n            "events": "div,splits",\n            "includeAdjustedClose": "true",\n        }\n        r = self.session.get(self.yahoo_url.format(symbol=symbol), params=params, headers=self.headers, timeout=20)\n        r.raise_for_status()\n        payload = r.json()\n        chart = payload.get("chart", {})\n        if chart.get("error"):\n            raise MarketDataError(f"Yahoo error for {symbol}: {chart[\'error\']}")\n        results = chart.get("result") or []\n        if not results:\n            raise MarketDataError(f"No Yahoo data returned for {symbol}")\n        result = results[0]\n        timestamps = result.get("timestamp") or []\n        quote = ((result.get("indicators") or {}).get("quote") or [{}])[0]\n        if not timestamps or not quote:\n            raise MarketDataError(f"Incomplete Yahoo data for {symbol}")\n        df = pd.DataFrame({\n            "timestamp": pd.to_datetime(timestamps, unit="s", utc=True),\n            "open": quote.get("open", []),\n            "high": quote.get("high", []),\n            "low": quote.get("low", []),\n            "close": quote.get("close", []),\n            "volume": quote.get("volume", []),\n        })\n        adj_list = ((result.get("indicators") or {}).get("adjclose") or [{}])[0].get("adjclose")\n        if adj_list and len(adj_list) == len(df):\n            raw_close = pd.to_numeric(df["close"], errors="coerce")\n            adj_close = pd.to_numeric(pd.Series(adj_list), errors="coerce")\n            factor = (adj_close / raw_close).replace([float("inf"), float("-inf")], pd.NA)\n            # Adjust OHLC consistently so splits/dividends do not create false signals.\n            for col in ["open", "high", "low", "close"]:\n                df[col] = pd.to_numeric(df[col], errors="coerce") * factor\n        df = df.set_index("timestamp")\n        return self._clean(df, req.symbol)\n\n    def fetch(self, req: DataRequest) -> pd.DataFrame:\n        errors = []\n        if not self._stooq_unavailable:\n            try:\n                df = self._fetch_stooq(req)\n                self.last_source = "Stooq"\n                self.source_by_symbol[req.symbol] = self.last_source\n                return df\n            except Exception as exc:\n                msg = str(exc)\n                errors.append(f"Stooq: {msg}")\n                # Stooq sometimes serves a browser-verification HTML page to cloud hosts.\n                # After detecting that infrastructure failure once, do not waste one failed\n                # request per symbol during a large universe screen.\n                low = msg.lower()\n                if any(x in low for x in ["unexpected stooq columns", "javascript", "doctype", "robots", "403", "forbidden"]):\n                    self._stooq_unavailable = True\n        else:\n            errors.append("Stooq: skipped after provider-level browser-verification failure")\n        try:\n            df = self._fetch_yahoo(req)\n            self.last_source = "Yahoo Finance fallback"\n            self.source_by_symbol[req.symbol] = self.last_source\n            return df\n        except Exception as exc:\n            errors.append(f"Yahoo: {exc}")\n        raise MarketDataError(f"Could not fetch {req.symbol}. " + " | ".join(errors))\n\n    def fetch_many(self, symbols: Iterable[str], years: int = 3) -> Dict[str, pd.DataFrame]:\n        end = date.today()\n        start = end - timedelta(days=int(365.25 * years))\n        out: Dict[str, pd.DataFrame] = {}\n        errors = []\n        self.last_errors = []\n        self.source_by_symbol = {}\n        # Preserve order while removing accidental duplicates.\n        seen = set()\n        ordered = []\n        for raw in symbols:\n            symbol = raw.strip().upper()\n            if symbol and symbol not in seen:\n                seen.add(symbol)\n                ordered.append(symbol)\n        for symbol in ordered:\n            try:\n                out[symbol] = self.fetch(DataRequest(symbol, start, end))\n            except Exception as exc:\n                errors.append(f"{symbol}: {exc}")\n        self.last_errors = errors\n        if not out:\n            raise MarketDataError("No symbols could be loaded. " + " | ".join(errors))\n        return out\n\n\ndef normalize_uploaded_csv(file_obj) -> pd.DataFrame:\n    df = pd.read_csv(file_obj)\n    lower = {c.lower().strip(): c for c in df.columns}\n    required = ["open", "high", "low", "close"]\n    missing = [c for c in required if c not in lower]\n    if missing:\n        raise MarketDataError(f"CSV missing columns: {\', \'.join(missing)}")\n\n    ts_col = None\n    for candidate in ["timestamp", "date", "datetime", "time"]:\n        if candidate in lower:\n            ts_col = lower[candidate]\n            break\n    if ts_col is None:\n        raise MarketDataError("CSV needs timestamp/date/datetime/time column")\n\n    rename = {lower[c]: c for c in required}\n    if "volume" in lower:\n        rename[lower["volume"]] = "volume"\n    rename[ts_col] = "timestamp"\n    df = df.rename(columns=rename)\n    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)\n    df = df.set_index("timestamp").sort_index()\n    if "volume" not in df.columns:\n        df["volume"] = 0.0\n    for c in ["open", "high", "low", "close", "volume"]:\n        df[c] = pd.to_numeric(df[c], errors="coerce")\n    return df[["open", "high", "low", "close", "volume"]].dropna(subset=required)\n', 'data_quality.py': 'from __future__ import annotations\nfrom dataclasses import dataclass, asdict\nimport pandas as pd\nimport numpy as np\n\n@dataclass\nclass DataQualityReport:\n    rows: int\n    start: str\n    end: str\n    duplicate_timestamps: int\n    invalid_ohlc_rows: int\n    nonpositive_price_rows: int\n    missing_volume_pct: float\n    median_gap_hours: float\n    observations_per_year: float\n    passed: bool\n    notes: list[str]\n\n    def to_dict(self):\n        return asdict(self)\n\ndef observations_per_year(index: pd.Index) -> float:\n    if len(index) < 3:\n        return 252.0\n    idx = pd.DatetimeIndex(index).sort_values()\n    elapsed_days = max((idx[-1] - idx[0]).total_seconds() / 86400.0, 1e-9)\n    return float((len(idx) - 1) / elapsed_days * 365.25)\n\ndef validate_ohlcv(df: pd.DataFrame) -> DataQualityReport:\n    if df is None or len(df) == 0:\n        raise ValueError("empty OHLCV data")\n    d = df.copy().sort_index()\n    dup = int(d.index.duplicated().sum())\n    max_oc = d[["open", "close"]].max(axis=1)\n    min_oc = d[["open", "close"]].min(axis=1)\n    invalid = int(((d["high"] < max_oc) | (d["low"] > min_oc) | (d["high"] < d["low"])).sum())\n    nonpositive = int((d[["open", "high", "low", "close"]] <= 0).any(axis=1).sum())\n    missing_vol = float(d["volume"].isna().mean() * 100) if "volume" in d else 100.0\n    gaps = pd.DatetimeIndex(d.index).to_series().diff().dropna().dt.total_seconds() / 3600.0\n    median_gap = float(gaps.median()) if len(gaps) else 0.0\n    notes = []\n    if dup:\n        notes.append("duplicate timestamps")\n    if invalid:\n        notes.append("OHLC consistency errors")\n    if nonpositive:\n        notes.append("non-positive prices")\n    if len(d) < 100:\n        notes.append("short history")\n    passed = dup == 0 and invalid == 0 and nonpositive == 0 and len(d) >= 60\n    return DataQualityReport(\n        rows=len(d), start=str(d.index[0]), end=str(d.index[-1]),\n        duplicate_timestamps=dup, invalid_ohlc_rows=invalid,\n        nonpositive_price_rows=nonpositive, missing_volume_pct=missing_vol,\n        median_gap_hours=median_gap, observations_per_year=observations_per_year(d.index),\n        passed=passed, notes=notes\n    )\n', 'demo_data.py': 'import numpy as np\nimport pandas as pd\n\ndef synthetic_ohlcv(seed=7, n=240, start=100.0, regime="mixed"):\n    rng = np.random.default_rng(seed)\n    rets = []\n    for i in range(n):\n        if regime == "trend":\n            mu, sigma = 0.0012, 0.012\n        elif regime == "volatile":\n            mu, sigma = 0.0001, 0.035\n        else:\n            # regime shifts: sideways -> uptrend -> volatile\n            if i < n*0.35:\n                mu, sigma = 0.0000, 0.010\n            elif i < n*0.70:\n                mu, sigma = 0.0014, 0.012\n            else:\n                mu, sigma = -0.0002, 0.025\n        rets.append(rng.normal(mu, sigma))\n    close = start * np.exp(np.cumsum(rets))\n    spread = np.maximum(close * rng.uniform(0.002, 0.012, size=n), 0.05)\n    open_ = np.r_[close[0], close[:-1]]\n    high = np.maximum(open_, close) + spread\n    low = np.maximum(np.minimum(open_, close) - spread, 0.01)\n    volume = rng.integers(1000, 100000, size=n)\n    idx = pd.date_range(end=pd.Timestamp.utcnow().floor("h"), periods=n, freq="h")\n    return pd.DataFrame(\n        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},\n        index=idx\n    )\n', 'integrity.py': 'from __future__ import annotations\n\nimport numpy as np\nimport pandas as pd\n\nfrom .backtest import _resolve_long_exit, run_backtest\nfrom .config import RiskConfig\nfrom .research import (\n    benjamini_hochberg_qvalues,\n    discovery_screen,\n    select_candidates_from_matrix,\n)\nfrom .strategies import STRATEGIES, precompute_indicators, scan_at\nfrom .skeptic import precompute_review_context\nfrom .snapshot import (\n    asset_sha256, build_snapshot_zip, dataset_manifest, load_snapshot_zip,\n    load_snapshot_ledger, matrix_result_fingerprint,\n)\nfrom .ledger import (\n    apply_cumulative_gate, cumulative_table, empty_ledger, finalize_batch,\n    ledger_fingerprint, preregister_batch,\n)\n\n\ndef _synthetic(seed=7, n=1700, drift=0.00025, vol=0.012):\n    rng = np.random.default_rng(seed)\n    idx = pd.bdate_range("2018-01-01", periods=n, tz="UTC")\n    ret = rng.normal(drift, vol, n)\n    close = 100.0 * np.exp(np.cumsum(ret))\n    open_ = close * (1 + rng.normal(0, 0.002, n))\n    spread = np.abs(rng.normal(0.006, 0.003, n))\n    high = np.maximum(open_, close) * (1 + spread)\n    low = np.minimum(open_, close) * (1 - spread)\n    volume = rng.integers(2_000_000, 8_000_000, n).astype(float)\n    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=idx)\n\n\ndef _sig_key(signals):\n    return [\n        (\n            s.strategy,\n            round(float(s.entry), 10),\n            round(float(s.stop), 10),\n            round(float(s.target), 10),\n            round(float(s.quality_score), 10),\n        )\n        for s in signals\n    ]\n\n\ndef run_integrity_checks() -> pd.DataFrame:\n    checks = []\n\n    def record(name, ok, detail):\n        checks.append({"check": name, "passed": bool(ok), "detail": str(detail)})\n\n    # 1) Future bars must not alter a signal already generated at bar t.\n    d = _synthetic(seed=11, n=700)\n    ind1 = precompute_indicators(d)\n    i = None\n    before = []\n    for candidate_i in range(220, min(650, len(d)-2)):\n        candidate = _sig_key(scan_at("SYN", d, candidate_i, ind1))\n        if candidate:\n            i = candidate_i\n            before = candidate\n            break\n    if i is None:\n        record("No future-bar leakage in signal generation", False, "synthetic series produced no signal")\n    else:\n        d2 = d.copy()\n        d2.iloc[i+1:, d2.columns.get_indexer(["open","high","low","close"])] *= 4.0\n        ind2 = precompute_indicators(d2)\n        after = _sig_key(scan_at("SYN", d2, i, ind2))\n        record("No future-bar leakage in signal generation", before == after, f"{len(before)} signal(s) at frozen bar {i}")\n\n    # 2) Same-bar ambiguity is resolved conservatively in favor of the stop.\n    bar = pd.Series({"open": 100.0, "high": 110.0, "low": 90.0})\n    px, reason = _resolve_long_exit(bar, 95.0, 105.0, 8.0, entered_this_bar=True)\n    record("Same-bar stop/target ambiguity is conservative", reason == "STOP_SAME_BAR" and px < 95.0, f"{reason} @ {px:.4f}")\n\n    # 3) Overnight stop gaps fill at the worse opening price, not at the stop.\n    bar2 = pd.Series({"open": 90.0, "high": 96.0, "low": 88.0})\n    px2, reason2 = _resolve_long_exit(bar2, 95.0, 110.0, 8.0, entered_this_bar=False)\n    record("Stop-gap execution uses worse open", reason2 == "STOP_GAP" and px2 < 90.0, f"{reason2} @ {px2:.4f}")\n\n    # 4) WATCH rows can never be frozen.\n    fake = pd.DataFrame([\n        {"symbol":"A","strategy":"watch","status":"WATCH","prob_mean_r_positive":0.99,"bh_q_value":0.01,"mean_r":1.0,"profit_factor":5.0,"return_pct":10.0},\n        {"symbol":"A","strategy":"cand","status":"CANDIDATE","prob_mean_r_positive":0.95,"bh_q_value":0.05,"mean_r":0.2,"profit_factor":1.4,"return_pct":3.0},\n        {"symbol":"B","strategy":"watch2","status":"WATCH","prob_mean_r_positive":0.999,"bh_q_value":0.001,"mean_r":2.0,"profit_factor":9.0,"return_pct":20.0},\n    ])\n    sel = select_candidates_from_matrix(fake)\n    record("Candidate-only freeze invariant", sel == {"A":"cand"}, str(sel))\n\n    # 5) BH q-values are bounded and monotone in sorted-p order.\n    p = np.array([0.001, 0.01, 0.02, 0.20, 0.9])\n    q = benjamini_hochberg_qvalues(p)\n    ord_ = np.argsort(p)\n    ok_q = np.all((q >= 0) & (q <= 1)) and np.all(np.diff(q[ord_]) >= -1e-12)\n    record("Multiple-testing correction sanity", ok_q, f"q={np.round(q,4).tolist()}")\n\n    # 6) The historical holdout must be physically irrelevant to discovery results.\n    data = {"A": _synthetic(21), "B": _synthetic(22), "C": _synthetic(23)}\n    cfg = RiskConfig()\n    m1, s1, id1 = discovery_screen(data, cfg, holdout_bars=252, burn_in_bars=504, fold_bars=126)\n    mutated = {k:v.copy() for k,v in data.items()}\n    for k,v in mutated.items():\n        cols = ["open","high","low","close"]\n        v.loc[v.index[-252:], cols] = v.loc[v.index[-252:], cols] * 7.0\n    m2, s2, id2 = discovery_screen(mutated, cfg, holdout_bars=252, burn_in_bars=504, fold_bars=126)\n    cols = ["symbol","strategy","status","folds","trades","return_pct","profit_factor","mean_r","prob_mean_r_positive","bh_q_value"]\n    a = m1[cols].sort_values(["symbol","strategy"]).reset_index(drop=True)\n    b = m2[cols].sort_values(["symbol","strategy"]).reset_index(drop=True)\n    numeric = ["return_pct","profit_factor","mean_r","prob_mean_r_positive","bh_q_value"]\n    exact_nonnum = a.drop(columns=numeric).equals(b.drop(columns=numeric))\n    numeric_equal = np.allclose(a[numeric].to_numpy(float), b[numeric].to_numpy(float), equal_nan=True, atol=1e-12, rtol=0)\n    record("Discovery is invariant to final-holdout mutation", exact_nonnum and numeric_equal and s1 == s2 and id1 == id2, f"selection_id {id1}")\n\n    # 7) Strategy registry must have unique names and the expected breadth.\n    names = [fn.__name__ for fn in STRATEGIES]\n    record("Strategy registry integrity", len(names) == 6 and len(set(names)) == 6, ", ".join(names))\n\n    # 8) The discovery fast path must be economically identical to the reference path.\n    d = _synthetic(seed=77, n=700)\n    cfg = RiskConfig()\n    ref = run_backtest(\n        "SYN", d, cfg, warmup=220, strategy_names={"trend_pullback"},\n        bootstrap_samples=0,\n    )\n    ind = precompute_indicators(d)\n    ctx = precompute_review_context(d)\n    fast = run_backtest(\n        "SYN", d, cfg, warmup=220, strategy_names={"trend_pullback"},\n        precomputed_indicators=ind,\n        precomputed_review_context=ctx,\n        prevalidated=True,\n        bootstrap_samples=0,\n    )\n    ref_trades = [\n        (t.strategy, t.signal_time, t.entry_time, t.exit_time, round(t.entry, 10),\n         round(t.exit, 10), round(t.pnl, 10), round(t.r_multiple, 10), t.exit_reason)\n        for t in ref.trades\n    ]\n    fast_trades = [\n        (t.strategy, t.signal_time, t.entry_time, t.exit_time, round(t.entry, 10),\n         round(t.exit, 10), round(t.pnl, 10), round(t.r_multiple, 10), t.exit_reason)\n        for t in fast.trades\n    ]\n    keys = ["total_return_pct", "benchmark_return_pct", "max_drawdown_pct", "profit_factor", "mean_r", "trades"]\n    ref_metrics = np.asarray([float(ref.metrics[k]) for k in keys], dtype=float)\n    fast_metrics = np.asarray([float(fast.metrics[k]) for k in keys], dtype=float)\n    same_metrics = np.allclose(ref_metrics, fast_metrics, atol=1e-12, rtol=0, equal_nan=True)\n    record(\n        "Optimized discovery path matches reference execution",\n        ref_trades == fast_trades and same_metrics,\n        f"{len(ref_trades)} trade(s); exact trade ledger + core metrics",\n    )\n\n    # 9) Disabling per-fold bootstrap must never change trading mechanics.\n    full_boot = run_backtest("SYN", d, cfg, warmup=220, strategy_names={"momentum_breakout"}, bootstrap_samples=300)\n    no_boot = run_backtest("SYN", d, cfg, warmup=220, strategy_names={"momentum_breakout"}, bootstrap_samples=0)\n    ledger_full = [(t.entry_time, t.exit_time, round(t.pnl, 10), t.exit_reason) for t in full_boot.trades]\n    ledger_none = [(t.entry_time, t.exit_time, round(t.pnl, 10), t.exit_reason) for t in no_boot.trades]\n    record(\n        "Bootstrap suppression cannot alter trades",\n        ledger_full == ledger_none and abs(full_boot.ending_equity - no_boot.ending_equity) < 1e-12,\n        f"{len(ledger_full)} trade(s)",\n    )\n\n    # 10) Dataset hashing must ignore dict insertion order but detect a one-value mutation.\n    a = _synthetic(seed=91, n=500)\n    b = _synthetic(seed=92, n=500)\n    m_ab = dataset_manifest({"A": a, "B": b})\n    m_ba = dataset_manifest({"B": b, "A": a})\n    a2 = a.copy()\n    a2.iloc[123, a2.columns.get_loc("close")] = np.nextafter(float(a2.iloc[123]["close"]), np.inf)\n    m_mut = dataset_manifest({"A": a2, "B": b})\n    record(\n        "Dataset ID is order-invariant and mutation-sensitive",\n        m_ab["dataset_id"] == m_ba["dataset_id"] and m_ab["dataset_id"] != m_mut["dataset_id"],\n        f"base={m_ab[\'dataset_id\']} mutated={m_mut[\'dataset_id\']}",\n    )\n\n    # 11) Snapshot export/import must reproduce exact per-asset hashes and dataset ID.\n    raw_zip, manifest = build_snapshot_zip({"A": a, "B": b}, {"A":"synthetic","B":"synthetic"}, spec_id="audit-spec")\n    restored, restored_manifest = load_snapshot_zip(raw_zip)\n    roundtrip_ok = (\n        restored_manifest["dataset_id"] == manifest["dataset_id"]\n        and asset_sha256(restored["A"]) == asset_sha256(a)\n        and asset_sha256(restored["B"]) == asset_sha256(b)\n    )\n    record(\n        "Snapshot ZIP round-trip is exact",\n        roundtrip_ok,\n        f"dataset {manifest[\'dataset_id\']}",\n    )\n\n    # 12) Research-result fingerprint must be stable and include the trade-ledger hash.\n    demo = pd.DataFrame([{\n        "symbol":"A", "strategy":"x", "status":"WATCH", "folds":3, "trades":12,\n        "return_pct":1.0, "benchmark_pct":2.0, "alpha_pct":-1.0, "max_drawdown_pct":1.2,\n        "profit_factor":1.1, "mean_r":0.05, "prob_mean_r_positive":0.65,\n        "mean_r_p_value":0.2, "mean_r_ci_low":-0.1, "mean_r_ci_high":0.2,\n        "positive_return_folds_pct":66.7, "bh_q_value":0.4, "trade_ledger_hash":"abc123",\n    }])\n    fp1 = matrix_result_fingerprint(demo)\n    fp2 = matrix_result_fingerprint(demo.copy())\n    demo2 = demo.copy(); demo2.loc[0, "trade_ledger_hash"] = "def456"\n    fp3 = matrix_result_fingerprint(demo2)\n    record(\n        "Research fingerprint is stable and ledger-sensitive",\n        fp1 == fp2 and fp1 != fp3,\n        f"base={fp1} changed={fp3}",\n    )\n\n\n    # 13) Exact reruns must be idempotent in the cumulative research ledger.\n    ledger = empty_ledger()\n    ledger, batch, is_new = preregister_batch(\n        ledger,\n        spec_id="spec-a",\n        dataset_id="data-a",\n        symbols=["A"],\n        strategies=["momentum_breakout"],\n        cfg=cfg,\n        scientific_protocol="0.6",\n        engine_fingerprint="audit",\n        legacy=False,\n    )\n    ledger_matrix = pd.DataFrame([{\n        "symbol":"A", "strategy":"momentum_breakout", "status":"WATCH", "why":"audit",\n        "folds":4, "trades":35, "return_pct":3.0, "benchmark_pct":2.0, "alpha_pct":1.0,\n        "max_drawdown_pct":2.0, "profit_factor":1.3, "mean_r":0.15,\n        "prob_mean_r_positive":0.92, "mean_r_p_value":0.01,\n        "mean_r_ci_low":0.01, "mean_r_ci_high":0.30, "positive_return_folds_pct":75.0,\n        "bh_q_value":0.05, "trade_ledger_hash":"audit-ledger-a",\n    }])\n    ledger = finalize_batch(ledger, batch["batch_id"], ledger_matrix, base_result_fingerprint="base-a")\n    events_before = len(ledger["events"])\n    ledger2 = finalize_batch(ledger, batch["batch_id"], ledger_matrix.copy(), base_result_fingerprint="base-a")\n    record(\n        "Exact research rerun is ledger-idempotent",\n        is_new and len(ledger2["events"]) == events_before == 1,\n        f"events={len(ledger2[\'events\'])}",\n    )\n\n    # 14) A genuinely new dataset is a new statistical look at the same hypothesis.\n    first_table = cumulative_table(ledger2)\n    first_anytime = float(first_table.iloc[0]["anytime_p"])\n    ledger3, batch2, is_new2 = preregister_batch(\n        ledger2,\n        spec_id="spec-a",\n        dataset_id="data-b",\n        symbols=["A"],\n        strategies=["momentum_breakout"],\n        cfg=cfg,\n        scientific_protocol="0.6",\n        engine_fingerprint="audit",\n        legacy=False,\n    )\n    ledger3 = finalize_batch(ledger3, batch2["batch_id"], ledger_matrix.copy(), base_result_fingerprint="base-b")\n    second_table = cumulative_table(ledger3)\n    row = second_table.iloc[0]\n    record(\n        "New dataset increments cumulative evidence look",\n        is_new2 and int(row["looks"]) == 2,\n        f"looks={int(row[\'looks\'])}",\n    )\n\n    # 15) Repeated peeking cannot improve the anytime-valid p-value with identical evidence.\n    second_anytime = float(row["anytime_p"])\n    record(\n        "Repeated identical evidence cannot game anytime p-value",\n        second_anytime >= first_anytime - 1e-15,\n        f"first={first_anytime:.6g} second={second_anytime:.6g}",\n    )\n\n    # 16) Pre-v0.7 legacy evidence can inform penalties but cannot directly unlock a candidate.\n    legacy_ledger = empty_ledger()\n    legacy_ledger, legacy_batch, _ = preregister_batch(\n        legacy_ledger,\n        spec_id="legacy-spec",\n        dataset_id="legacy-data",\n        symbols=["A"],\n        strategies=["momentum_breakout"],\n        cfg=cfg,\n        scientific_protocol="0.6",\n        engine_fingerprint="old-engine",\n        legacy=True,\n    )\n    cand = ledger_matrix.copy()\n    cand.loc[0, "status"] = "CANDIDATE"\n    cand.loc[0, "why"] = "passed batch gate"\n    cand.loc[0, "mean_r_p_value"] = 0.0001\n    cand.loc[0, "bh_q_value"] = 0.001\n    legacy_ledger = finalize_batch(\n        legacy_ledger, legacy_batch["batch_id"], cand, base_result_fingerprint="legacy-base"\n    )\n    gated = apply_cumulative_gate(\n        cand, legacy_ledger, cfg=cfg, scientific_protocol="0.6",\n        batch_id=legacy_batch["batch_id"],\n    )\n    record(\n        "Legacy evidence cannot directly unlock cumulative candidate",\n        str(gated.iloc[0]["status"]) == "WATCH" and str(gated.iloc[0]["ledger_gate"]) == "BLOCK_LEGACY",\n        str(gated.iloc[0]["ledger_gate"]),\n    )\n\n    # 17) Legacy exploratory evidence may consume research budget but must never\n    # supply affirmative anytime evidence by itself.\n    legacy_table = cumulative_table(legacy_ledger)\n    legacy_anytime = float(legacy_table.iloc[0]["anytime_p"]) if len(legacy_table) else 0.0\n    record(\n        "Legacy evidence is penalty-only in cumulative test",\n        len(legacy_table) == 1\n        and int(legacy_table.iloc[0]["legacy_looks"]) == 1\n        and int(legacy_table.iloc[0]["preregistered_looks"]) == 0\n        and abs(legacy_anytime - 1.0) < 1e-15,\n        f"anytime_p={legacy_anytime:.6g}",\n    )\n\n    # 18) Reproducibility bundle must carry the cumulative ledger bit-for-bit logically.\n    raw_with_ledger, _ = build_snapshot_zip(\n        {"A": a, "B": b}, {"A":"synthetic","B":"synthetic"},\n        spec_id="audit-ledger-spec", ledger=ledger3,\n        ledger_result_fingerprint="audit-ledger-result",\n    )\n    restored_ledger = load_snapshot_ledger(raw_with_ledger)\n    record(\n        "Snapshot ZIP preserves cumulative research ledger",\n        restored_ledger is not None and ledger_fingerprint(restored_ledger) == ledger_fingerprint(ledger3),\n        f"ledger={ledger_fingerprint(ledger3)}",\n    )\n\n    return pd.DataFrame(checks)\n', 'leaderboard.py': 'from __future__ import annotations\n\nimport math\nimport numpy as np\nimport pandas as pd\n\nfrom .backtest import run_backtest\nfrom .config import RiskConfig\nfrom .strategies import STRATEGIES\n\n\ndef _pf(x):\n    if x == math.inf:\n        return 5.0\n    return float(x) if np.isfinite(x) else 0.0\n\n\ndef strategy_leaderboard(\n    data: dict[str, pd.DataFrame],\n    cfg: RiskConfig,\n    train_fraction: float = 0.70,\n    warmup: int = 220,\n) -> tuple[pd.DataFrame, pd.DataFrame]:\n    """Chronological holdout ranking across multiple assets.\n\n    The first part of each asset is treated as development history; metrics shown in\n    the leaderboard come only from the final holdout section.\n    """\n    rows = []\n    for symbol, df in data.items():\n        if len(df) < warmup + 120:\n            continue\n        split = int(len(df) * train_fraction)\n        split = max(split, warmup + 30)\n        if split >= len(df) - 30:\n            continue\n        test = pd.concat([df.iloc[max(0, split-warmup):split], df.iloc[split:]])\n        for fn in STRATEGIES:\n            name = fn.__name__\n            try:\n                res = run_backtest(symbol, test, cfg, warmup=warmup, strategy_names={name})\n            except Exception:\n                continue\n            m = res.metrics\n            rows.append({\n                "symbol": symbol,\n                "strategy": name,\n                "oos_return_pct": m["total_return_pct"],\n                "oos_benchmark_pct": m["benchmark_return_pct"],\n                "oos_alpha_pct": m["alpha_vs_benchmark_pct"],\n                "oos_drawdown_pct": m["max_drawdown_pct"],\n                "oos_sharpe": m["sharpe"],\n                "oos_profit_factor": _pf(m["profit_factor"]),\n                "oos_trades": m["trades"],\n            })\n    detail = pd.DataFrame(rows)\n    if detail.empty:\n        return pd.DataFrame(), detail\n\n    agg = []\n    for strategy, g in detail.groupby("strategy"):\n        tested = len(g)\n        alpha_pos = float((g["oos_alpha_pct"] > 0).mean()) * 100\n        total_trades = int(g["oos_trades"].sum())\n        median_return = float(g["oos_return_pct"].median())\n        median_alpha = float(g["oos_alpha_pct"].median())\n        median_pf = float(g["oos_profit_factor"].median())\n        median_sharpe = float(g["oos_sharpe"].median())\n        worst_dd = float(g["oos_drawdown_pct"].max())\n        gate = (\n            "PASS" if tested >= 3 and total_trades >= 12 and alpha_pos >= 60\n            and median_return > 0 and median_alpha > 0 and median_pf > 1.05 and worst_dd <= cfg.max_drawdown_pct * 100\n            else "FAIL"\n        )\n        score = median_alpha + 2.0 * median_sharpe + 2.0 * (alpha_pos / 100 - 0.5) - 0.25 * worst_dd\n        agg.append({\n            "strategy": strategy,\n            "assets_tested": tested,\n            "positive_alpha_assets_pct": alpha_pos,\n            "median_oos_return_pct": median_return,\n            "median_oos_alpha_pct": median_alpha,\n            "median_oos_profit_factor": median_pf,\n            "median_oos_sharpe": median_sharpe,\n            "worst_oos_drawdown_pct": worst_dd,\n            "total_oos_trades": total_trades,\n            "research_gate": gate,\n            "robustness_score": score,\n        })\n    summary = pd.DataFrame(agg).sort_values(\n        ["research_gate", "robustness_score"], ascending=[True, False]\n    ).reset_index(drop=True)\n    # Put PASS before FAIL without relying on alphabetical order.\n    summary["_gate_order"] = summary["research_gate"].map({"PASS": 0, "FAIL": 1}).fillna(2)\n    summary = summary.sort_values(["_gate_order", "robustness_score"], ascending=[True, False]).drop(columns="_gate_order").reset_index(drop=True)\n    return summary, detail.sort_values(["strategy", "symbol"]).reset_index(drop=True)\n', 'models.py': 'from dataclasses import dataclass, asdict\nfrom datetime import datetime, timezone\nfrom typing import Optional\n\n@dataclass\nclass Signal:\n    symbol: str\n    strategy: str\n    side: str\n    entry: float\n    stop: float\n    target: float\n    quality_score: float  # heuristic ranking score, NOT a probability\n    reason: str\n\n    @property\n    def reward_risk(self) -> float:\n        risk = abs(self.entry - self.stop)\n        reward = abs(self.target - self.entry)\n        return reward / risk if risk > 0 else 0.0\n\n@dataclass\nclass RiskDecision:\n    approved: bool\n    quantity: float\n    risk_amount: float\n    reason: str\n\n@dataclass\nclass Trade:\n    symbol: str\n    strategy: str\n    side: str\n    entry: float\n    stop: float\n    target: float\n    quantity: float\n    status: str = "OPEN"\n    exit_price: Optional[float] = None\n    pnl: float = 0.0\n    opened_at: str = ""\n    closed_at: Optional[str] = None\n\n    def __post_init__(self):\n        if not self.opened_at:\n            self.opened_at = datetime.now(timezone.utc).isoformat()\n\n    def to_dict(self):\n        return asdict(self)\n', 'montecarlo.py': 'from __future__ import annotations\nimport numpy as np\n\n\ndef simulate_r_paths(r_values, risk_fraction=0.005, trades_per_path=100, paths=5000, seed=123):\n    """Bootstrap trade R-multiples into hypothetical equity paths.\n\n    This is a sequence-risk stress test, not a forecast. It assumes sampled trades\n    are exchangeable and therefore cannot model regime clustering or structural breaks.\n    """\n    r = np.asarray([x for x in r_values if np.isfinite(x)], dtype=float)\n    if len(r) < 20:\n        return {\n            "paths": 0,\n            "trades_per_path": trades_per_path,\n            "median_return_pct": 0.0,\n            "p05_return_pct": 0.0,\n            "p95_max_drawdown_pct": 0.0,\n            "prob_finish_negative": 0.0,\n            "prob_drawdown_gt_10pct": 0.0,\n            "prob_drawdown_gt_20pct": 0.0,\n        }\n    rng = np.random.default_rng(seed)\n    sampled = rng.choice(r, size=(paths, trades_per_path), replace=True)\n    trade_returns = sampled * float(risk_fraction)\n    trade_returns = np.maximum(trade_returns, -0.99)\n    equity = np.cumprod(1.0 + trade_returns, axis=1)\n    equity = np.concatenate([np.ones((paths, 1)), equity], axis=1)\n    peaks = np.maximum.accumulate(equity, axis=1)\n    drawdowns = 1.0 - equity / peaks\n    max_dd = drawdowns.max(axis=1)\n    ending = equity[:, -1] - 1.0\n    return {\n        "paths": int(paths),\n        "trades_per_path": int(trades_per_path),\n        "median_return_pct": float(np.median(ending) * 100),\n        "p05_return_pct": float(np.quantile(ending, 0.05) * 100),\n        "p95_max_drawdown_pct": float(np.quantile(max_dd, 0.95) * 100),\n        "prob_finish_negative": float((ending < 0).mean()),\n        "prob_drawdown_gt_10pct": float((max_dd > 0.10).mean()),\n        "prob_drawdown_gt_20pct": float((max_dd > 0.20).mean()),\n    }\n', 'portfolio.py': 'from __future__ import annotations\n\nfrom dataclasses import dataclass\nimport math\nimport numpy as np\nimport pandas as pd\n\nfrom .backtest import ClosedTrade, _metrics, _resolve_long_exit, _fees, _sell_fill\nfrom .config import RiskConfig\nfrom .skeptic import review\nfrom .strategies import scan_at, precompute_indicators\nfrom .data_quality import validate_ohlcv\n\n\n@dataclass\nclass PortfolioResult:\n    starting_equity: float\n    ending_equity: float\n    trades: list[ClosedTrade]\n    equity_curve: pd.DataFrame\n    benchmark_curve: pd.DataFrame\n    metrics: dict\n\n\ndef _build_benchmark(data: dict[str, pd.DataFrame], dates: pd.Index, start_equity: float) -> pd.DataFrame:\n    normalized = []\n    for _, df in data.items():\n        s = df["close"].reindex(dates).ffill()\n        first_valid = s.first_valid_index()\n        if first_valid is None:\n            continue\n        base = float(s.loc[first_valid])\n        if base <= 0:\n            continue\n        ratio = s / base\n        ratio.loc[:first_valid] = 1.0\n        normalized.append(ratio.ffill())\n    if not normalized:\n        return pd.DataFrame({"benchmark": [start_equity] * len(dates)}, index=dates)\n    ratios = pd.concat(normalized, axis=1).mean(axis=1)\n    return pd.DataFrame({"benchmark": start_equity * ratios}, index=dates)\n\n\ndef _adv(df: pd.DataFrame, i: int, n: int = 20):\n    if "volume" not in df or i <= 0:\n        return None\n    h = df.iloc[max(0, i-n):i]\n    dv = (h["close"] * h["volume"]).replace([np.inf, -np.inf], np.nan).dropna()\n    dv = dv[dv > 0]\n    return float(dv.median()) if len(dv) >= 5 else None\n\n\ndef _trailing_corr(a: pd.DataFrame, ai: int, b: pd.DataFrame, bi: int, lookback: int) -> float:\n    ar = a["close"].iloc[:ai+1].pct_change().dropna().tail(lookback).rename("a")\n    br = b["close"].iloc[:bi+1].pct_change().dropna().tail(lookback).rename("b")\n    x = pd.concat([ar, br], axis=1).dropna()\n    if len(x) < max(15, lookback // 3):\n        return 0.0\n    c = float(x["a"].corr(x["b"]))\n    return c if np.isfinite(c) else 0.0\n\n\ndef run_portfolio_backtest(data: dict[str, pd.DataFrame], cfg: RiskConfig, warmup: int = 220, strategy_by_symbol: dict[str, str] | None = None) -> PortfolioResult:\n    valid = {}\n    for k, v in data.items():\n        if len(v) <= warmup + 20:\n            continue\n        d = v.sort_index().copy()\n        q = validate_ohlcv(d)\n        if q.passed:\n            valid[k] = d\n    if not valid:\n        raise ValueError("No asset has enough clean history")\n\n    start_candidates = [df.index[min(warmup, len(df)-1)] for df in valid.values()]\n    start_date = max(start_candidates)\n    dates = pd.DatetimeIndex(sorted(set().union(*[set(df.index[df.index >= start_date]) for df in valid.values()])))\n    if len(dates) < 30:\n        raise ValueError("Too little overlapping history")\n\n    indicators = {sym: precompute_indicators(df) for sym, df in valid.items()}\n    realized = cfg.starting_equity\n    peak = realized\n    positions: dict[str, dict] = {}\n    trades: list[ClosedTrade] = []\n    rows = []\n    drawdown_halted = False\n    kill_flatten_pending = False\n    heat_rebalance_pending = False\n    heat_rebalance_count = 0\n\n    def mark_equity(ts, mode="close"):\n        marked = realized\n        gross = 0.0\n        for sym, pos in positions.items():\n            df = valid[sym]\n            if mode == "open" and ts in df.index:\n                px = float(df.loc[ts, "open"])\n            else:\n                h = df[df.index <= ts]\n                if not len(h):\n                    continue\n                px = float(h.iloc[-1]["close"])\n            marked += (px - pos["entry"]) * pos["qty"]\n            gross += px * pos["qty"]\n        return marked, gross\n\n    def close_position(sym, ts, exit_px, reason, exit_i):\n        nonlocal realized\n        pos = positions[sym]\n        gross_pnl = (exit_px - pos["entry"]) * pos["qty"]\n        exit_fee = _fees(exit_px * pos["qty"], cfg.simulated_fee_bps)\n        realized += gross_pnl - exit_fee\n        trade_pnl = gross_pnl - exit_fee - pos["entry_fee"]\n        ir = max(pos["initial_risk"], 1e-12)\n        trades.append(ClosedTrade(\n            symbol=sym, strategy=pos["strategy"], side="BUY", signal_time=str(pos["signal_time"]),\n            entry_time=str(pos["entry_time"]), exit_time=str(ts), entry=pos["entry"], exit=exit_px,\n            stop=pos["stop"], target=pos["target"], quantity=pos["qty"], initial_risk=pos["initial_risk"],\n            pnl=trade_pnl, r_multiple=trade_pnl/ir, return_pct=(exit_px/pos["entry"]-1)*100,\n            hold_bars=max(1, exit_i-pos["entry_i"]+1), exit_reason=reason,\n        ))\n        positions.pop(sym, None)\n\n    for ts in dates:\n        previous_marked = rows[-1][1] if rows else cfg.starting_equity\n        daily_anchor = previous_marked\n\n        # 1) A latched risk kill flattens exposure at the next executable open.\n        if kill_flatten_pending and cfg.flatten_on_risk_kill:\n            for sym in list(positions):\n                df = valid[sym]\n                if ts not in df.index:\n                    continue\n                i = int(df.index.get_loc(ts))\n                exit_px = _sell_fill(float(df.loc[ts, "open"]), cfg.simulated_slippage_bps)\n                close_position(sym, ts, exit_px, "RISK_KILL", i)\n            kill_flatten_pending = bool(positions)\n\n        # 2) At the open, only overnight gap exits are knowable. Do not use today\'s high/low yet.\n        for sym in list(positions):\n            df = valid[sym]\n            if ts not in df.index:\n                continue\n            i = int(df.index.get_loc(ts))\n            bar = df.loc[ts]\n            pos = positions[sym]\n            o = float(bar["open"])\n            if o <= pos["stop"]:\n                close_position(sym, ts, _sell_fill(o, cfg.simulated_slippage_bps), "STOP_GAP", i)\n            elif o >= pos["target"]:\n                close_position(sym, ts, _sell_fill(pos["target"], cfg.simulated_slippage_bps), "TARGET_GAP", i)\n\n        # 2b) If the previous close pushed heat above the cap, de-risk at this\n        # executable open by closing the largest stop-risk positions first.\n        if heat_rebalance_pending and positions:\n            marked_open, _ = mark_equity(ts, mode="open")\n            def _heat_amount():\n                return sum(max(0.0, p["entry"] - p["stop"]) * p["qty"] for p in positions.values())\n            while positions and marked_open > 0 and _heat_amount() > marked_open * cfg.max_portfolio_heat_pct:\n                closable = []\n                for sym, pos in positions.items():\n                    df = valid[sym]\n                    if ts in df.index:\n                        risk_amt = max(0.0, pos["entry"] - pos["stop"]) * pos["qty"]\n                        closable.append((risk_amt, sym, df))\n                if not closable:\n                    break\n                _, sym, df = max(closable, key=lambda x: x[0])\n                i = int(df.index.get_loc(ts))\n                exit_px = _sell_fill(float(df.loc[ts, "open"]), cfg.simulated_slippage_bps)\n                close_position(sym, ts, exit_px, "HEAT_REBALANCE", i)\n                heat_rebalance_count += 1\n                marked_open, _ = mark_equity(ts, mode="open")\n            heat_rebalance_pending = False\n\n        # Opening risk state uses only prices known at the open.\n        marked_before, gross_before = mark_equity(ts, mode="open")\n        peak = max(peak, marked_before)\n        dd = (peak - marked_before) / peak if peak > 0 else 0.0\n        if dd >= cfg.max_drawdown_pct:\n            drawdown_halted = True\n            if positions:\n                kill_flatten_pending = True\n        daily_halted = (marked_before - daily_anchor) <= -daily_anchor * cfg.max_daily_loss_pct\n        if daily_halted and positions:\n            kill_flatten_pending = True\n\n        # 3) Candidate signals are based strictly on each asset\'s prior bar.\n        candidates = []\n        if not drawdown_halted and not daily_halted and not kill_flatten_pending:\n            for sym, df in valid.items():\n                if sym in positions or ts not in df.index:\n                    continue\n                if strategy_by_symbol is not None and sym not in strategy_by_symbol:\n                    continue\n                i_today = int(df.index.get_loc(ts))\n                i = i_today - 1\n                if i < warmup:\n                    continue\n                hist = df.iloc[:i+1]\n                allowed = {strategy_by_symbol[sym]} if strategy_by_symbol is not None else None\n                for sig in scan_at(sym, df, i, indicators[sym], allowed_strategies=allowed):\n                    ok, _ = review(sig, hist)\n                    if ok and sig.reward_risk >= cfg.min_reward_risk:\n                        candidates.append((sig.quality_score * min(sig.reward_risk, 4.0), sig, df.index[i], i, i_today))\n        candidates.sort(key=lambda x: x[0], reverse=True)\n\n        # 4) Allocate one shared risk budget with heat, leverage, ADV, and correlation caps.\n        for _, sig, signal_time, signal_i, i_today in candidates:\n            if len(positions) >= cfg.max_open_positions or sig.symbol in positions:\n                continue\n            df = valid[sig.symbol]\n            bar = df.iloc[i_today]\n            raw_entry = float(bar["open"])\n            intended_risk = abs(sig.entry - sig.stop)\n            gap_units = abs(raw_entry - sig.entry) / intended_risk if intended_risk > 0 else math.inf\n            if intended_risk <= 0 or gap_units > cfg.max_entry_gap_risk_units or raw_entry <= sig.stop:\n                continue\n\n            # Correlation veto against current holdings using information available before entry.\n            too_correlated = False\n            for held_sym, held in positions.items():\n                held_df = valid[held_sym]\n                held_prior = np.flatnonzero(held_df.index < ts)\n                if not len(held_prior):\n                    continue\n                c = _trailing_corr(df, signal_i, held_df, int(held_prior[-1]), cfg.correlation_lookback)\n                if c >= cfg.max_pair_correlation:\n                    too_correlated = True\n                    break\n            if too_correlated:\n                continue\n\n            current_equity, current_gross = mark_equity(ts, mode="open")\n            if current_equity <= 0:\n                break\n            entry = raw_entry * (1 + cfg.simulated_slippage_bps / 10_000.0)\n            stop = float(sig.stop)\n            risk_distance = entry - stop\n            if risk_distance <= 0:\n                continue\n            target = entry + risk_distance * max(sig.reward_risk, cfg.min_reward_risk)\n\n            current_heat = sum(max(0.0, p["entry"] - p["stop"]) * p["qty"] for p in positions.values())\n            available_heat = max(0.0, current_equity * cfg.max_portfolio_heat_pct - current_heat)\n            risk_budget = min(current_equity * cfg.max_risk_per_trade_pct, available_heat)\n            if risk_budget <= 0:\n                continue\n            qty_risk = risk_budget / risk_distance\n            # Exact heat cap at allocation, including the entry fee reducing equity.\n            fee_rate = cfg.simulated_fee_bps / 10_000.0\n            heat_numerator = max(0.0, current_equity * cfg.max_portfolio_heat_pct - current_heat)\n            qty_heat = heat_numerator / max(risk_distance + cfg.max_portfolio_heat_pct * entry * fee_rate, 1e-12)\n            qty_value = (current_equity * cfg.max_position_value_pct) / entry\n            remaining_gross = max(0.0, current_equity * cfg.max_leverage - current_gross)\n            qty_leverage = remaining_gross / entry\n            adv = _adv(df, i_today)\n            qty_adv = (adv * cfg.max_adv_participation_pct / entry) if adv and adv > 0 else math.inf\n            qty = max(0.0, min(qty_risk, qty_heat, qty_value, qty_leverage, qty_adv))\n            if qty <= 0:\n                continue\n\n            entry_fee = _fees(entry * qty, cfg.simulated_fee_bps)\n            realized -= entry_fee\n            positions[sig.symbol] = {\n                "strategy": sig.strategy, "signal_time": signal_time, "entry_time": ts,\n                "entry_i": i_today, "entry": entry, "stop": stop, "target": target,\n                "qty": qty, "entry_fee": entry_fee, "initial_risk": risk_distance*qty,\n            }\n            # Update gross exposure before considering next candidate.\n            _, current_gross = mark_equity(ts, mode="open")\n\n        # 5) Intraday high/low now becomes available. Existing and new positions are managed alike.\n        for sym in list(positions):\n            df = valid[sym]\n            if ts not in df.index:\n                continue\n            pos = positions[sym]\n            i = int(df.index.get_loc(ts))\n            bar = df.loc[ts]\n            exit_px, reason = _resolve_long_exit(bar, pos["stop"], pos["target"], cfg.simulated_slippage_bps, entered_this_bar=True)\n            if exit_px is not None:\n                reason = "STOP_INTRADAY" if reason.startswith("STOP") else "TARGET_INTRADAY"\n                close_position(sym, ts, exit_px, reason, i)\n\n        marked, gross = mark_equity(ts, mode="close")\n        peak = max(peak, marked)\n        dd = (peak - marked) / peak if peak > 0 else 0.0\n        if dd >= cfg.max_drawdown_pct:\n            drawdown_halted = True\n            if positions:\n                kill_flatten_pending = True\n        if marked - daily_anchor <= -daily_anchor * cfg.max_daily_loss_pct and positions:\n            kill_flatten_pending = True\n        heat = sum(max(0.0, p["entry"] - p["stop"]) * p["qty"] for p in positions.values())\n        heat_ratio = heat/marked if marked > 0 else 0.0\n        if heat_ratio > cfg.max_portfolio_heat_pct and positions:\n            heat_rebalance_pending = True\n        rows.append((ts, marked, gross/marked if marked > 0 else 0.0, heat_ratio, drawdown_halted))\n\n    # Force-close remaining positions for comparable terminal accounting.\n    for sym in list(positions):\n        pos = positions[sym]\n        df = valid[sym]\n        last_ts = df.index[-1]\n        exit_px = _sell_fill(float(df.iloc[-1]["close"]), cfg.simulated_slippage_bps)\n        close_position(sym, last_ts, exit_px, "END_OF_TEST", len(df)-1)\n\n    curve = pd.DataFrame(rows, columns=["timestamp", "equity", "gross_exposure", "portfolio_heat", "drawdown_halted"]).drop_duplicates("timestamp").set_index("timestamp")\n    if len(curve):\n        curve.iloc[-1, curve.columns.get_loc("equity")] = realized\n        curve["exposure"] = (curve["gross_exposure"] > 0).astype(float)\n    benchmark = _build_benchmark(valid, curve.index, cfg.starting_equity)\n    metrics = _metrics(cfg.starting_equity, realized, curve, trades, benchmark)\n    metrics["max_portfolio_heat_pct_observed"] = float(curve["portfolio_heat"].max()*100) if len(curve) else 0.0\n    metrics["max_gross_exposure_pct_observed"] = float(curve["gross_exposure"].max()*100) if len(curve) else 0.0\n    metrics["heat_rebalance_count"] = int(heat_rebalance_count)\n    return PortfolioResult(cfg.starting_equity, realized, trades, curve, benchmark, metrics)\n', 'registry.py': 'from __future__ import annotations\nimport hashlib\nimport json\nimport sqlite3\nfrom dataclasses import asdict, is_dataclass\nfrom pathlib import Path\nfrom datetime import datetime, timezone\n\n\ndef code_fingerprint() -> str:\n    root = Path(__file__).resolve().parent\n    h = hashlib.sha256()\n    for path in sorted(root.glob("*.py")):\n        h.update(path.name.encode())\n        h.update(path.read_bytes())\n    return h.hexdigest()[:16]\n\n\ndef _jsonable(x):\n    if is_dataclass(x):\n        return asdict(x)\n    return x\n\nclass ExperimentRegistry:\n    def __init__(self, path="data/experiments.db"):\n        self.path = path\n        Path(path).parent.mkdir(parents=True, exist_ok=True)\n        with sqlite3.connect(path) as con:\n            con.execute("""\n            CREATE TABLE IF NOT EXISTS experiments(\n                id INTEGER PRIMARY KEY AUTOINCREMENT,\n                created_at TEXT NOT NULL,\n                kind TEXT NOT NULL,\n                label TEXT NOT NULL,\n                data_start TEXT, data_end TEXT,\n                code_hash TEXT NOT NULL,\n                config_json TEXT NOT NULL,\n                result_json TEXT NOT NULL,\n                notes TEXT\n            )\n            """)\n\n    def save(self, kind, label, cfg, result, data_start="", data_end="", notes=""):\n        cfg_json = json.dumps(_jsonable(cfg), sort_keys=True, default=str)\n        result_json = json.dumps(result, sort_keys=True, default=str)\n        with sqlite3.connect(self.path) as con:\n            cur = con.execute(\n                """INSERT INTO experiments(created_at,kind,label,data_start,data_end,code_hash,config_json,result_json,notes)\n                VALUES(?,?,?,?,?,?,?,?,?)""",\n                (datetime.now(timezone.utc).isoformat(), kind, label, data_start, data_end, code_fingerprint(), cfg_json, result_json, notes),\n            )\n            return int(cur.lastrowid)\n\n    def list(self, limit=200):\n        import pandas as pd\n        with sqlite3.connect(self.path) as con:\n            return pd.read_sql_query(\n                "SELECT id,created_at,kind,label,data_start,data_end,code_hash,notes FROM experiments ORDER BY id DESC LIMIT ?",\n                con, params=(int(limit),)\n            )\n', 'research.py': 'from __future__ import annotations\n\nimport hashlib\nimport json\nimport math\nimport numpy as np\nimport pandas as pd\n\nfrom .backtest import run_backtest\nfrom .config import RiskConfig\nfrom .portfolio import run_portfolio_backtest\nfrom .strategies import STRATEGIES, precompute_indicators\nfrom .skeptic import precompute_review_context\nfrom .data_quality import validate_ohlcv\n\nPROTOCOL_VERSION = "0.6"       # scientific ruleset unchanged from v0.6\nENGINE_VERSION = "0.7"          # cumulative-ledger implementation\nSTRATEGY_NAMES = [fn.__name__ for fn in STRATEGIES]\nDISCOVERY_WARMUP = 220\n\n\ndef _float_token(value: float) -> str:\n    f = float(value)\n    if math.isnan(f):\n        return "NaN"\n    if math.isinf(f):\n        return "+Inf" if f > 0 else "-Inf"\n    return f.hex()\n\n\ndef _trade_ledger_hash(trades) -> str:\n    rows = []\n    for t in trades:\n        rows.append({\n            "strategy": t.strategy,\n            "signal_time": t.signal_time,\n            "entry_time": t.entry_time,\n            "exit_time": t.exit_time,\n            "entry": _float_token(t.entry),\n            "exit": _float_token(t.exit),\n            "stop": _float_token(t.stop),\n            "target": _float_token(t.target),\n            "quantity": _float_token(t.quantity),\n            "initial_risk": _float_token(t.initial_risk),\n            "pnl": _float_token(t.pnl),\n            "r_multiple": _float_token(t.r_multiple),\n            "exit_reason": t.exit_reason,\n        })\n    raw = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")\n    return hashlib.sha256(raw).hexdigest()[:16]\n\n\ndef _finite_pf(value: float) -> float:\n    if value == math.inf:\n        return 9.99\n    return float(value) if np.isfinite(value) else 0.0\n\n\ndef _bootstrap_mean_r(r_values, samples: int = 5000, seed: int = 2026):\n    """Circular moving-block bootstrap for trade R-multiples.\n\n    Returns both an uncertainty interval/probability and a centered-null one-sided\n    bootstrap p-value suitable for the later multiple-testing correction. A short\n    block preserves some local trade-sequence dependence instead of assuming every\n    trade is fully independent.\n    """\n    r = np.asarray([x for x in r_values if np.isfinite(x)], dtype=float)\n    if len(r) < 2:\n        return {\n            "mean_r": float(r.mean()) if len(r) else 0.0,\n            "prob_mean_r_positive": 0.0,\n            "p_value_mean_r_positive": 1.0,\n            "ci_low": 0.0,\n            "ci_high": 0.0,\n        }\n    n = len(r)\n    block_len = min(3, n)\n    n_blocks = int(np.ceil(n / block_len))\n    rng = np.random.default_rng(seed)\n    starts = rng.integers(0, n, size=(samples, n_blocks))\n    offsets = np.arange(block_len, dtype=int)\n    idx = ((starts[..., None] + offsets) % n).reshape(samples, -1)[:, :n]\n\n    sampled_means = r[idx].mean(axis=1)\n    observed = float(r.mean())\n\n    # Null bootstrap: center the empirical R distribution at zero, preserving its\n    # shape and short-range dependence. Compare null means with the observed mean.\n    centered = r - observed\n    null_means = centered[idx].mean(axis=1)\n    p_one_sided = float((1 + np.sum(null_means >= observed)) / (samples + 1))\n\n    return {\n        "mean_r": observed,\n        "prob_mean_r_positive": float((sampled_means > 0).mean()),\n        "p_value_mean_r_positive": p_one_sided,\n        "ci_low": float(np.quantile(sampled_means, 0.025)),\n        "ci_high": float(np.quantile(sampled_means, 0.975)),\n    }\n\n\ndef benjamini_hochberg_qvalues(p_values) -> np.ndarray:\n    """Benjamini-Hochberg FDR-adjusted q-values; NaNs are treated as p=1."""\n    p = np.asarray(p_values, dtype=float)\n    p = np.where(np.isfinite(p), np.clip(p, 0.0, 1.0), 1.0)\n    m = len(p)\n    if m == 0:\n        return np.asarray([], dtype=float)\n    order = np.argsort(p, kind="stable")\n    ranked = p[order]\n    q_ranked = ranked * m / np.arange(1, m + 1, dtype=float)\n    q_ranked = np.minimum.accumulate(q_ranked[::-1])[::-1]\n    q_ranked = np.clip(q_ranked, 0.0, 1.0)\n    q = np.empty(m, dtype=float)\n    q[order] = q_ranked\n    return q\n\n\ndef _pair_status(trades, ret, pf, mean_r, prob, positive_folds):\n    reasons = []\n    if trades < 15:\n        reasons.append(f"only {trades} trades (<15)")\n    if ret <= 0:\n        reasons.append("non-positive discovery return")\n    if mean_r <= 0:\n        reasons.append("mean R <= 0")\n    if pf <= 1.0:\n        reasons.append("profit factor <= 1.0")\n    if prob < 0.60:\n        reasons.append("weak bootstrap confidence")\n    if positive_folds < 0.50:\n        reasons.append("positive in <50% of folds")\n\n    if trades < 15:\n        status = "INSUFFICIENT"\n    elif ret <= 0 or mean_r <= 0 or pf <= 1.0 or prob < 0.55:\n        status = "REJECT"\n    elif (\n        trades >= 30\n        and pf >= 1.20\n        and mean_r >= 0.10\n        and prob >= 0.90\n        and positive_folds >= 0.60\n    ):\n        # Final CANDIDATE status also requires FDR control across the whole screen.\n        status = "PRE_CANDIDATE"\n    else:\n        status = "WATCH"\n\n    if not reasons:\n        reasons.append("passed absolute expectancy screen")\n    return status, "; ".join(reasons)\n\n\ndef fixed_strategy_discovery(\n    symbol: str,\n    df: pd.DataFrame,\n    cfg: RiskConfig,\n    strategy: str,\n    burn_in_bars: int = 504,\n    fold_bars: int = 126,\n    warmup: int = DISCOVERY_WARMUP,\n    *,\n    precomputed_indicators: pd.DataFrame | None = None,\n    precomputed_review_context: pd.DataFrame | None = None,\n    prevalidated: bool = False,\n) -> dict:\n    """Evaluate one fixed asset/strategy pair on rolling chronological folds.\n\n    No strategy selection or parameter tuning happens inside these folds. The first\n    `burn_in_bars` are intentionally not scored. Each subsequent fold is treated as\n    unseen for the fixed strategy.\n    """\n    if strategy not in STRATEGY_NAMES:\n        raise ValueError(f"Unknown strategy: {strategy}")\n    if len(df) < burn_in_bars + fold_bars:\n        return {\n            "symbol": symbol, "strategy": strategy, "status": "INSUFFICIENT",\n            "why": "not enough discovery history", "folds": 0, "trades": 0,\n            "return_pct": 0.0, "benchmark_pct": 0.0, "alpha_pct": 0.0,\n            "max_drawdown_pct": 0.0, "profit_factor": 0.0, "mean_r": 0.0,\n            "prob_mean_r_positive": 0.0, "mean_r_p_value": 1.0, "positive_return_folds_pct": 0.0,\n            "mean_r_ci_low": 0.0, "mean_r_ci_high": 0.0,\n            "trade_ledger_hash": _trade_ledger_hash([]),\n        }\n\n    fold_returns = []\n    fold_bench = []\n    fold_dd = []\n    fold_positive = []\n    all_trades = []\n\n    start = burn_in_bars\n    while start + fold_bars <= len(df):\n        prefix_start = max(0, start - warmup)\n        inp = df.iloc[prefix_start:start + fold_bars]\n        prefix_len = start - prefix_start\n        test_len = len(inp) - prefix_len\n        if prefix_len < min(60, warmup) or test_len < 20:\n            start += fold_bars\n            continue\n\n        fold_indicators = None\n        if precomputed_indicators is not None:\n            fold_indicators = precomputed_indicators.iloc[prefix_start:start + fold_bars]\n        fold_review_context = None\n        if precomputed_review_context is not None:\n            fold_review_context = precomputed_review_context.iloc[prefix_start:start + fold_bars]\n\n        try:\n            res = run_backtest(\n                symbol, inp, cfg,\n                warmup=prefix_len,\n                strategy_names={strategy},\n                precomputed_indicators=fold_indicators,\n                precomputed_review_context=fold_review_context,\n                prevalidated=prevalidated,\n                bootstrap_samples=0,\n            )\n        except ValueError:\n            # Expected data/length validation failures may skip a fold.\n            # Programming errors are intentionally NOT swallowed.\n            start += fold_bars\n            continue\n        m = res.metrics\n        fold_returns.append(float(m["total_return_pct"]) / 100.0)\n        fold_bench.append(float(m["benchmark_return_pct"]) / 100.0)\n        fold_dd.append(float(m["max_drawdown_pct"]))\n        fold_positive.append(float(m["total_return_pct"]) > 0)\n        all_trades.extend(res.trades)\n        start += fold_bars\n\n    compounded = float(np.prod(1 + np.asarray(fold_returns)) - 1) if fold_returns else 0.0\n    bench = float(np.prod(1 + np.asarray(fold_bench)) - 1) if fold_bench else 0.0\n    pnls = np.asarray([t.pnl for t in all_trades], dtype=float)\n    gross_profit = float(pnls[pnls > 0].sum()) if len(pnls) else 0.0\n    gross_loss = float(-pnls[pnls < 0].sum()) if len(pnls) else 0.0\n    pf = gross_profit / gross_loss if gross_loss > 0 else (math.inf if gross_profit > 0 else 0.0)\n    boot = _bootstrap_mean_r([t.r_multiple for t in all_trades])\n    pos_folds = float(np.mean(fold_positive)) if fold_positive else 0.0\n    status, why = _pair_status(\n        len(all_trades), compounded * 100, _finite_pf(pf),\n        boot["mean_r"], boot["prob_mean_r_positive"], pos_folds,\n    )\n\n    return {\n        "symbol": symbol,\n        "strategy": strategy,\n        "status": status,\n        "why": why,\n        "folds": len(fold_returns),\n        "trades": len(all_trades),\n        "return_pct": compounded * 100,\n        "benchmark_pct": bench * 100,\n        "alpha_pct": (compounded - bench) * 100,\n        "max_drawdown_pct": max(fold_dd, default=0.0),\n        "profit_factor": _finite_pf(pf),\n        "mean_r": boot["mean_r"],\n        "prob_mean_r_positive": boot["prob_mean_r_positive"],\n        "mean_r_p_value": boot["p_value_mean_r_positive"],\n        "mean_r_ci_low": boot["ci_low"],\n        "mean_r_ci_high": boot["ci_high"],\n        "positive_return_folds_pct": pos_folds * 100,\n        "trade_ledger_hash": _trade_ledger_hash(all_trades),\n    }\n\n\ndef _selection_key(row: dict):\n    return (\n        row.get("prob_mean_r_positive", 0.0),\n        -row.get("bh_q_value", 1.0),\n        row.get("mean_r", 0.0),\n        row.get("profit_factor", 0.0),\n        row.get("return_pct", 0.0),\n    )\n\n\ndef select_candidates_from_matrix(matrix: pd.DataFrame) -> dict[str, str]:\n    """Freeze at most one CANDIDATE per asset. WATCH can never unlock a holdout."""\n    selections = {}\n    if matrix is None or matrix.empty:\n        return selections\n    for symbol, group in matrix.groupby("symbol"):\n        eligible = group[group["status"] == "CANDIDATE"]\n        if eligible.empty:\n            continue\n        best = max(eligible.to_dict("records"), key=_selection_key)\n        selections[symbol] = best["strategy"]\n    return selections\n\n\ndef family_summary(matrix: pd.DataFrame) -> pd.DataFrame:\n    if matrix is None or matrix.empty:\n        return pd.DataFrame()\n    rows = []\n    for strategy, g in matrix.groupby("strategy"):\n        evaluable = g[g["trades"] >= 15]\n        if evaluable.empty:\n            rows.append({\n                "strategy": strategy, "assets_evaluable": 0, "combined_trades": int(g["trades"].sum()),\n                "positive_mean_r_assets_pct": 0.0, "median_mean_r": 0.0,\n                "median_profit_factor": 0.0, "median_bootstrap_prob": 0.0,\n                "candidates": int((g["status"] == "CANDIDATE").sum()),\n                "watch": int((g["status"] == "WATCH").sum()),\n            })\n            continue\n        rows.append({\n            "strategy": strategy,\n            "assets_evaluable": int(len(evaluable)),\n            "combined_trades": int(evaluable["trades"].sum()),\n            "positive_mean_r_assets_pct": float((evaluable["mean_r"] > 0).mean() * 100),\n            "median_mean_r": float(evaluable["mean_r"].median()),\n            "median_profit_factor": float(evaluable["profit_factor"].median()),\n            "median_bootstrap_prob": float(evaluable["prob_mean_r_positive"].median()),\n            "candidates": int((g["status"] == "CANDIDATE").sum()),\n            "watch": int((g["status"] == "WATCH").sum()),\n        })\n    out = pd.DataFrame(rows)\n    return out.sort_values(\n        ["candidates", "positive_mean_r_assets_pct", "median_mean_r", "combined_trades"],\n        ascending=[False, False, False, False],\n    ).reset_index(drop=True)\n\n\ndef discovery_screen(\n    data: dict[str, pd.DataFrame],\n    cfg: RiskConfig,\n    holdout_bars: int = 504,\n    burn_in_bars: int = 504,\n    fold_bars: int = 126,\n):\n    """Screen pairs using only pre-holdout history, then control multiple testing.\n\n    The final `holdout_bars` are removed before ANY pair is scored. Candidate status\n    requires both strong absolute evidence and Benjamini-Hochberg FDR q<=0.10.\n    """\n    rows = []\n    discovery_ranges = {}\n    for symbol, df in data.items():\n        df = df.sort_index()\n        if len(df) <= holdout_bars + burn_in_bars + fold_bars:\n            for strategy in STRATEGY_NAMES:\n                rows.append({\n                    "symbol": symbol, "strategy": strategy, "status": "INSUFFICIENT",\n                    "why": "not enough rows after reserving holdout", "folds": 0, "trades": 0,\n                    "return_pct": 0.0, "benchmark_pct": 0.0, "alpha_pct": 0.0,\n                    "max_drawdown_pct": 0.0, "profit_factor": 0.0, "mean_r": 0.0,\n                    "prob_mean_r_positive": 0.0, "mean_r_p_value": 1.0, "positive_return_folds_pct": 0.0,\n                    "mean_r_ci_low": 0.0, "mean_r_ci_high": 0.0,\n                    "trade_ledger_hash": _trade_ledger_hash([]),\n                })\n            continue\n\n        discovery = df.iloc[:-holdout_bars]\n        q = validate_ohlcv(discovery)\n        if not q.passed:\n            for strategy in STRATEGY_NAMES:\n                rows.append({\n                    "symbol": symbol, "strategy": strategy, "status": "INSUFFICIENT",\n                    "why": f"discovery data quality failed: {q.notes}", "folds": 0, "trades": 0,\n                    "return_pct": 0.0, "benchmark_pct": 0.0, "alpha_pct": 0.0,\n                    "max_drawdown_pct": 0.0, "profit_factor": 0.0, "mean_r": 0.0,\n                    "prob_mean_r_positive": 0.0, "mean_r_p_value": 1.0,\n                    "positive_return_folds_pct": 0.0, "mean_r_ci_low": 0.0, "mean_r_ci_high": 0.0,\n                    "trade_ledger_hash": _trade_ledger_hash([]),\n                })\n            continue\n\n        discovery_ranges[symbol] = (str(discovery.index[0]), str(discovery.index[-1]))\n\n        # Performance-only optimization: both objects are strictly causal and are\n        # computed once per asset, then sliced into each fold. Candidate rules,\n        # execution, fold boundaries, and statistical thresholds are unchanged.\n        discovery_indicators = precompute_indicators(discovery)\n        discovery_review_context = precompute_review_context(discovery)\n\n        for strategy in STRATEGY_NAMES:\n            rows.append(fixed_strategy_discovery(\n                symbol, discovery, cfg, strategy,\n                burn_in_bars=burn_in_bars, fold_bars=fold_bars,\n                precomputed_indicators=discovery_indicators,\n                precomputed_review_context=discovery_review_context,\n                prevalidated=True,\n            ))\n\n    matrix = pd.DataFrame(rows)\n    if matrix.empty:\n        return matrix, {}, hashlib.sha256(b"empty-v06").hexdigest()[:12]\n\n    # Multiple-testing control. Use the centered-null block-bootstrap p-value,\n    # not 1-P(mean R>0), because the latter is descriptive confidence rather than\n    # a calibrated hypothesis-test p-value. Insufficient rows receive p=1.\n    pvals = []\n    for _, row in matrix.iterrows():\n        if int(row["trades"]) < 15:\n            pvals.append(1.0)\n        else:\n            pvals.append(max(0.0, min(1.0, float(row.get("mean_r_p_value", 1.0)))))\n    matrix["bh_q_value"] = benjamini_hochberg_qvalues(pvals)\n\n    # Convert PRE_CANDIDATE into CANDIDATE only after controlling the whole screen.\n    for idx in matrix.index:\n        if matrix.at[idx, "status"] == "PRE_CANDIDATE":\n            q = float(matrix.at[idx, "bh_q_value"])\n            if q <= 0.10:\n                matrix.at[idx, "status"] = "CANDIDATE"\n                matrix.at[idx, "why"] = str(matrix.at[idx, "why"]) + "; FDR q<=0.10"\n            else:\n                matrix.at[idx, "status"] = "WATCH"\n                matrix.at[idx, "why"] = str(matrix.at[idx, "why"]) + f"; multiple-testing q={q:.3f}>0.10"\n\n    selections = select_candidates_from_matrix(matrix)\n    payload = {\n        "protocol": PROTOCOL_VERSION,\n        "holdout_bars": holdout_bars,\n        "burn_in_bars": burn_in_bars,\n        "fold_bars": fold_bars,\n        "strategy_names": STRATEGY_NAMES,\n        "symbols": sorted(data.keys()),\n        "selections": selections,\n        "discovery_ranges": discovery_ranges,\n    }\n    selection_id = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]\n    return matrix, selections, selection_id\n\n\ndef evaluate_frozen_holdout(\n    data: dict[str, pd.DataFrame],\n    selections: dict[str, str],\n    cfg: RiskConfig,\n    holdout_bars: int = 504,\n    warmup: int = DISCOVERY_WARMUP,\n):\n    """Secondary historical holdout for already-frozen candidate selections.\n\n    This is a robustness check, not the final proof. Forward paper data remains the\n    only genuinely future, untouched validation after the research protocol is frozen.\n    """\n    pair_rows = []\n    portfolio_data = {}\n\n    for symbol, strategy in selections.items():\n        if symbol not in data:\n            continue\n        df = data[symbol].sort_index()\n        if len(df) <= holdout_bars + warmup:\n            continue\n        prefix = df.iloc[-(holdout_bars + warmup):-holdout_bars]\n        holdout = df.iloc[-holdout_bars:]\n        inp = pd.concat([prefix, holdout]).sort_index()\n        res = run_backtest(symbol, inp, cfg, warmup=len(prefix), strategy_names={strategy})\n        m = res.metrics\n        pair_rows.append({\n            "symbol": symbol,\n            "strategy": strategy,\n            "holdout_return_pct": m["total_return_pct"],\n            "holdout_benchmark_pct": m["benchmark_return_pct"],\n            "holdout_alpha_pct": m["alpha_vs_benchmark_pct"],\n            "holdout_trades": m["trades"],\n            "holdout_profit_factor": _finite_pf(m["profit_factor"]),\n            "holdout_mean_r": m.get("mean_r", 0.0),\n            "holdout_prob_mean_r_positive": m.get("bootstrap_prob_mean_r_positive", 0.0),\n            "holdout_max_drawdown_pct": m["max_drawdown_pct"],\n        })\n        portfolio_data[symbol] = inp\n\n    pair_df = pd.DataFrame(pair_rows)\n    if not portfolio_data:\n        return pair_df, None, "INCONCLUSIVE"\n\n    port = run_portfolio_backtest(\n        portfolio_data, cfg, warmup=warmup,\n        strategy_by_symbol=selections,\n    )\n    m = port.metrics\n    if m["trades"] < 20:\n        gate = "INCONCLUSIVE"\n    elif (\n        m["total_return_pct"] > 0\n        and m["profit_factor"] > 1.15\n        and m.get("mean_r", 0.0) > 0\n        and m.get("bootstrap_prob_mean_r_positive", 0.0) >= 0.80\n        and m["max_drawdown_pct"] <= cfg.max_drawdown_pct * 100\n    ):\n        gate = "HISTORICAL_HOLDOUT_PASS"\n    else:\n        gate = "FAIL_HISTORICAL_HOLDOUT"\n    return pair_df, port, gate\n', 'risk.py': 'from .config import RiskConfig\nfrom .models import Signal, RiskDecision\n\nclass RiskEngine:\n    def __init__(self, config: RiskConfig):\n        self.config = config\n\n    def approve(\n        self,\n        signal: Signal,\n        equity: float,\n        peak_equity: float,\n        daily_pnl: float,\n        open_positions: int,\n    ) -> RiskDecision:\n\n        if equity <= 0:\n            return RiskDecision(False, 0, 0, "equity <= 0")\n\n        drawdown = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0\n        if drawdown >= self.config.max_drawdown_pct:\n            return RiskDecision(False, 0, 0, "KILL SWITCH: max drawdown reached")\n\n        if daily_pnl <= -equity * self.config.max_daily_loss_pct:\n            return RiskDecision(False, 0, 0, "KILL SWITCH: daily loss limit reached")\n\n        if open_positions >= self.config.max_open_positions:\n            return RiskDecision(False, 0, 0, "max open positions reached")\n\n        if signal.reward_risk < self.config.min_reward_risk:\n            return RiskDecision(False, 0, 0, "reward/risk below minimum")\n\n        per_unit_risk = abs(signal.entry - signal.stop)\n        if per_unit_risk <= 0:\n            return RiskDecision(False, 0, 0, "invalid stop distance")\n\n        risk_budget = equity * self.config.max_risk_per_trade_pct\n        qty_by_risk = risk_budget / per_unit_risk\n\n        max_position_value = equity * self.config.max_position_value_pct\n        qty_by_value = max_position_value / signal.entry\n\n        qty = max(0.0, min(qty_by_risk, qty_by_value))\n        if qty <= 0:\n            return RiskDecision(False, 0, 0, "position size resolved to zero")\n\n        actual_risk = qty * per_unit_risk\n        return RiskDecision(True, qty, actual_risk, "approved by hard risk rules")\n', 'scanner.py': 'from __future__ import annotations\n\nimport pandas as pd\nfrom .strategies import scan\nfrom .skeptic import review\nfrom .data_quality import validate_ohlcv\n\n\ndef rank_watchlist(data: dict[str, pd.DataFrame]) -> pd.DataFrame:\n    rows = []\n    for symbol, df in data.items():\n        if len(df) < 60:\n            continue\n        q = validate_ohlcv(df)\n        if not q.passed:\n            continue\n        for sig in scan(symbol, df):\n            skeptic_ok, skeptic_reason = review(sig, df)\n            score = sig.quality_score * min(sig.reward_risk, 4.0)\n            rows.append({\n                "symbol": symbol,\n                "strategy": sig.strategy,\n                "last_bar": str(df.index[-1]),\n                "entry_reference": sig.entry,\n                "stop": sig.stop,\n                "target": sig.target,\n                "reward_risk": sig.reward_risk,\n                "quality_score": sig.quality_score,\n                "skeptic": "PASS" if skeptic_ok else "VETO",\n                "skeptic_reason": skeptic_reason,\n                "score": score if skeptic_ok else 0.0,\n                "reason": sig.reason,\n                "data_rows": q.rows,\n            })\n    if not rows:\n        return pd.DataFrame(columns=[\n            "symbol", "strategy", "last_bar", "entry_reference", "stop", "target",\n            "reward_risk", "quality_score", "skeptic", "skeptic_reason", "score", "reason", "data_rows"\n        ])\n    return pd.DataFrame(rows).sort_values(["score", "quality_score"], ascending=False).reset_index(drop=True)\n', 'skeptic.py': 'import numpy as np\nimport pandas as pd\nfrom .models import Signal\n\n\ndef review(signal: Signal, df):\n    """Adaptive veto layer. quality_score is a ranking heuristic, not probability."""\n    reasons = []\n    if signal.quality_score < 0.52:\n        reasons.append("heuristic quality below floor")\n    if signal.reward_risk < 1.5:\n        reasons.append(f"reward/risk too low ({signal.reward_risk:.2f})")\n\n    returns = df["close"].pct_change().dropna()\n    if len(returns) >= 30:\n        last_abs = abs(float(returns.iloc[-1]))\n        prior_abs = returns.iloc[:-1].tail(120).abs()\n        if len(prior_abs) >= 20:\n            shock_threshold = max(float(prior_abs.quantile(0.98)), 4.0 * float(prior_abs.median()))\n            if shock_threshold > 0 and last_abs > shock_threshold:\n                reasons.append("latest bar is an adaptive return shock")\n\n    if len(returns) >= 80:\n        prior = returns.iloc[:-10]\n        recent_vol = float(returns.tail(10).std(ddof=0))\n        rolling = prior.rolling(10).std(ddof=0).dropna().tail(120)\n        if len(rolling) >= 20:\n            baseline = float(rolling.median())\n            if baseline > 0 and recent_vol > 2.5 * baseline:\n                reasons.append("volatility regime >2.5x recent baseline")\n\n    approved = not reasons\n    return approved, "APPROVED" if approved else "; ".join(reasons)\n\n\ndef precompute_review_context(df: pd.DataFrame) -> pd.DataFrame:\n    """Vectorized, causal version of the signal-independent review checks.\n\n    Every value at bar t uses only returns known at or before t. This is used by\n    the discovery fast path so the same adaptive veto is not recomputed hundreds\n    of times across strategy/fold combinations.\n    """\n    close = pd.to_numeric(df["close"], errors="coerce")\n    returns = close.pct_change()\n    abs_r = returns.abs()\n\n    # Exact counterpart of returns.iloc[:-1].tail(120) at each bar.\n    prior_q98 = abs_r.shift(1).rolling(120, min_periods=20).quantile(0.98)\n    prior_med = abs_r.shift(1).rolling(120, min_periods=20).median()\n    shock_threshold = pd.concat([prior_q98, 4.0 * prior_med], axis=1).max(axis=1)\n    return_count = returns.notna().cumsum()\n    shock = (return_count >= 30) & (shock_threshold > 0) & (abs_r > shock_threshold)\n\n    # Exact counterpart of:\n    # prior = returns.iloc[:-10]\n    # rolling = prior.rolling(10).std(ddof=0).dropna().tail(120)\n    rolling10 = returns.rolling(10).std(ddof=0)\n    recent_vol = rolling10\n    prior_vol_baseline = rolling10.shift(10).rolling(120, min_periods=20).median()\n    extreme_vol = (\n        (return_count >= 80)\n        & (prior_vol_baseline > 0)\n        & (recent_vol > 2.5 * prior_vol_baseline)\n    )\n\n    return pd.DataFrame(\n        {\n            "adaptive_return_shock": shock.fillna(False).astype(bool),\n            "extreme_volatility_regime": extreme_vol.fillna(False).astype(bool),\n        },\n        index=df.index,\n    )\n\n\ndef review_at(signal: Signal, df: pd.DataFrame, i: int, context: pd.DataFrame | None = None):\n    """Review one signal at bar i, optionally using precomputed causal context."""\n    if context is None:\n        return review(signal, df.iloc[: i + 1])\n\n    reasons = []\n    if signal.quality_score < 0.52:\n        reasons.append("heuristic quality below floor")\n    if signal.reward_risk < 1.5:\n        reasons.append(f"reward/risk too low ({signal.reward_risk:.2f})")\n\n    if bool(context["adaptive_return_shock"].iloc[i]):\n        reasons.append("latest bar is an adaptive return shock")\n    if bool(context["extreme_volatility_regime"].iloc[i]):\n        reasons.append("volatility regime >2.5x recent baseline")\n\n    approved = not reasons\n    return approved, "APPROVED" if approved else "; ".join(reasons)\n', 'snapshot.py': 'from __future__ import annotations\n\nimport hashlib\nimport io\nimport json\nimport math\nimport zipfile\nfrom typing import Any\n\nimport numpy as np\nimport pandas as pd\n\nfrom .ledger import validate_ledger, ledger_fingerprint\n\nSNAPSHOT_FORMAT_VERSION = 1\n_OHLCV_ORDER = ["open", "high", "low", "close", "volume"]\n\n\ndef _safe_symbol_filename(symbol: str) -> str:\n    out = []\n    for ch in str(symbol):\n        if ch.isalnum() or ch in ("-", "_", "."):\n            out.append(ch)\n        else:\n            out.append("_")\n    return "".join(out) or "asset"\n\n\ndef canonical_frame(df: pd.DataFrame) -> pd.DataFrame:\n    """Return a stable OHLCV representation used for hashing and snapshots."""\n    if df is None or len(df) == 0:\n        raise ValueError("cannot snapshot an empty dataframe")\n    x = df.copy().sort_index()\n    idx = pd.to_datetime(x.index, utc=True, errors="raise")\n    cols = [c for c in _OHLCV_ORDER if c in x.columns]\n    required = {"open", "high", "low", "close"}\n    if not required.issubset(cols):\n        missing = sorted(required.difference(cols))\n        raise ValueError(f"snapshot missing OHLC columns: {missing}")\n    x = x[cols].astype(float)\n    x.index = idx\n    x.index.name = "timestamp"\n    if x.index.has_duplicates:\n        raise ValueError("snapshot index contains duplicate timestamps")\n    return x\n\n\ndef canonical_csv_bytes(df: pd.DataFrame) -> bytes:\n    x = canonical_frame(df)\n    out = x.copy()\n    # Fixed UTC string avoids locale/timezone formatting differences. Numeric values\n    # are stored as IEEE-754 hexadecimal literals so snapshot restore is bit-exact;\n    # decimal CSV parsers are permitted to round the final bit differently.\n    for col in out.columns:\n        out[col] = out[col].map(lambda v: float(v).hex())\n    out.insert(0, "timestamp", out.index.strftime("%Y-%m-%dT%H:%M:%S.%fZ"))\n    out = out.reset_index(drop=True)\n    text = out.to_csv(index=False, lineterminator="\\n")\n    return text.encode("utf-8")\n\n\ndef asset_sha256(df: pd.DataFrame) -> str:\n    return hashlib.sha256(canonical_csv_bytes(df)).hexdigest()\n\n\ndef _manifest_core(data: dict[str, pd.DataFrame], sources: dict[str, str] | None = None) -> dict[str, Any]:\n    sources = sources or {}\n    assets = []\n    for symbol in sorted(data):\n        x = canonical_frame(data[symbol])\n        assets.append({\n            "symbol": symbol,\n            "sha256": asset_sha256(x),\n            "rows": int(len(x)),\n            "start": x.index[0].isoformat(),\n            "end": x.index[-1].isoformat(),\n            "columns": list(x.columns),\n            "source": str(sources.get(symbol, "unknown")),\n            "file": f"assets/{_safe_symbol_filename(symbol)}.csv",\n        })\n    return {\n        "format_version": SNAPSHOT_FORMAT_VERSION,\n        "assets": assets,\n    }\n\n\ndef dataset_manifest(\n    data: dict[str, pd.DataFrame],\n    sources: dict[str, str] | None = None,\n    *,\n    spec_id: str | None = None,\n    spec_payload: dict[str, Any] | None = None,\n    result_fingerprint: str | None = None,\n    engine_fingerprint: str | None = None,\n) -> dict[str, Any]:\n    core = _manifest_core(data, sources)\n    dataset_id_payload = {\n        "format_version": core["format_version"],\n        "assets": [\n            {k: a[k] for k in ("symbol", "sha256", "rows", "start", "end", "columns")}\n            for a in core["assets"]\n        ],\n    }\n    dataset_id = hashlib.sha256(\n        json.dumps(dataset_id_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")\n    ).hexdigest()[:16]\n    manifest = {\n        **core,\n        "dataset_id": dataset_id,\n        "spec_id": spec_id,\n        "spec_payload": spec_payload or {},\n        "result_fingerprint": result_fingerprint,\n        "engine_fingerprint": engine_fingerprint,\n    }\n    return manifest\n\n\ndef build_snapshot_zip(\n    data: dict[str, pd.DataFrame],\n    sources: dict[str, str] | None = None,\n    *,\n    spec_id: str | None = None,\n    spec_payload: dict[str, Any] | None = None,\n    result_fingerprint: str | None = None,\n    engine_fingerprint: str | None = None,\n    matrix: pd.DataFrame | None = None,\n    ledger: dict[str, Any] | None = None,\n    ledger_result_fingerprint: str | None = None,\n) -> tuple[bytes, dict[str, Any]]:\n    manifest = dataset_manifest(\n        data,\n        sources,\n        spec_id=spec_id,\n        spec_payload=spec_payload,\n        result_fingerprint=result_fingerprint,\n        engine_fingerprint=engine_fingerprint,\n    )\n    if ledger is not None:\n        checked_ledger = validate_ledger(ledger)\n        manifest["ledger_fingerprint"] = ledger_fingerprint(checked_ledger)\n        manifest["ledger_result_fingerprint"] = ledger_result_fingerprint\n        manifest["ledger_file"] = "research_ledger.json"\n    else:\n        checked_ledger = None\n\n    buf = io.BytesIO()\n    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:\n        zf.writestr(\n            "manifest.json",\n            json.dumps(manifest, sort_keys=True, indent=2, ensure_ascii=False),\n        )\n        by_symbol = {a["symbol"]: a for a in manifest["assets"]}\n        for symbol in sorted(data):\n            zf.writestr(by_symbol[symbol]["file"], canonical_csv_bytes(data[symbol]))\n        if matrix is not None:\n            zf.writestr("research_matrix.csv", matrix.to_csv(index=False, lineterminator="\\n", float_format="%.17g"))\n        if checked_ledger is not None:\n            zf.writestr(\n                "research_ledger.json",\n                json.dumps(checked_ledger, sort_keys=True, indent=2, ensure_ascii=False),\n            )\n    return buf.getvalue(), manifest\n\n\ndef load_snapshot_ledger(raw: bytes) -> dict[str, Any] | None:\n    """Return an embedded v0.7 research ledger, if present.\n\n    Older v0.6.2 bundles intentionally return None and remain backward compatible.\n    """\n    if not raw:\n        raise ValueError("empty snapshot file")\n    with zipfile.ZipFile(io.BytesIO(raw), "r") as zf:\n        if "research_ledger.json" not in zf.namelist():\n            return None\n        ledger = validate_ledger(json.loads(zf.read("research_ledger.json").decode("utf-8")))\n        if "manifest.json" in zf.namelist():\n            manifest = json.loads(zf.read("manifest.json").decode("utf-8"))\n            expected = manifest.get("ledger_fingerprint")\n            if expected and str(expected) != ledger_fingerprint(ledger):\n                raise ValueError("embedded research ledger fingerprint mismatch")\n        return ledger\n\n\ndef load_snapshot_zip(raw: bytes) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:\n    if not raw:\n        raise ValueError("empty snapshot file")\n    with zipfile.ZipFile(io.BytesIO(raw), "r") as zf:\n        if "manifest.json" not in zf.namelist():\n            raise ValueError("snapshot is missing manifest.json")\n        manifest = json.loads(zf.read("manifest.json").decode("utf-8"))\n        if int(manifest.get("format_version", -1)) != SNAPSHOT_FORMAT_VERSION:\n            raise ValueError("unsupported snapshot format version")\n        data: dict[str, pd.DataFrame] = {}\n        for asset in manifest.get("assets", []):\n            symbol = str(asset["symbol"])\n            path = str(asset["file"])\n            if path not in zf.namelist():\n                raise ValueError(f"snapshot missing asset file for {symbol}")\n            raw_csv = zf.read(path)\n            observed_hash = hashlib.sha256(raw_csv).hexdigest()\n            if observed_hash != asset.get("sha256"):\n                raise ValueError(f"asset hash mismatch for {symbol}")\n            df = pd.read_csv(io.BytesIO(raw_csv), dtype=str, keep_default_na=False)\n            if "timestamp" not in df.columns:\n                raise ValueError(f"asset {symbol} is missing timestamp")\n            idx = pd.to_datetime(df.pop("timestamp"), utc=True, errors="raise")\n            for col in df.columns:\n                df[col] = df[col].map(float.fromhex)\n            df.index = idx\n            df.index.name = "timestamp"\n            df = canonical_frame(df)\n            # Re-canonicalize after parsing to prove exact round-trip.\n            if asset_sha256(df) != asset.get("sha256"):\n                raise ValueError(f"asset round-trip hash mismatch for {symbol}")\n            data[symbol] = df\n\n    rebuilt = dataset_manifest(\n        data,\n        {a["symbol"]: a.get("source", "unknown") for a in manifest.get("assets", [])},\n        spec_id=manifest.get("spec_id"),\n        spec_payload=manifest.get("spec_payload") or {},\n        result_fingerprint=manifest.get("result_fingerprint"),\n        engine_fingerprint=manifest.get("engine_fingerprint"),\n    )\n    if rebuilt["dataset_id"] != manifest.get("dataset_id"):\n        raise ValueError("dataset ID mismatch after snapshot restore")\n    return data, manifest\n\n\ndef compare_manifests(old: dict[str, Any], new: dict[str, Any]) -> pd.DataFrame:\n    old_map = {a["symbol"]: a for a in old.get("assets", [])}\n    new_map = {a["symbol"]: a for a in new.get("assets", [])}\n    rows = []\n    for symbol in sorted(set(old_map) | set(new_map)):\n        a = old_map.get(symbol)\n        b = new_map.get(symbol)\n        rows.append({\n            "symbol": symbol,\n            "changed": a is None or b is None or a.get("sha256") != b.get("sha256"),\n            "old_rows": None if a is None else a.get("rows"),\n            "new_rows": None if b is None else b.get("rows"),\n            "old_end": None if a is None else a.get("end"),\n            "new_end": None if b is None else b.get("end"),\n            "old_sha12": None if a is None else str(a.get("sha256", ""))[:12],\n            "new_sha12": None if b is None else str(b.get("sha256", ""))[:12],\n        })\n    return pd.DataFrame(rows)\n\n\ndef _encode_value(value: Any):\n    if value is None:\n        return None\n    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):\n        return int(value)\n    if isinstance(value, (np.floating, float)):\n        f = float(value)\n        if math.isnan(f):\n            return "NaN"\n        if math.isinf(f):\n            return "+Inf" if f > 0 else "-Inf"\n        return f.hex()\n    if isinstance(value, (np.bool_, bool)):\n        return bool(value)\n    return str(value)\n\n\ndef matrix_result_fingerprint(matrix: pd.DataFrame) -> str:\n    """Exact deterministic fingerprint of pair decisions, metrics, and trade ledgers."""\n    if matrix is None or matrix.empty:\n        return hashlib.sha256(b"empty-research-matrix").hexdigest()[:16]\n    x = matrix.copy().sort_values(["symbol", "strategy"]).reset_index(drop=True)\n    cols = [\n        "symbol", "strategy", "status", "folds", "trades", "return_pct",\n        "benchmark_pct", "alpha_pct", "max_drawdown_pct", "profit_factor",\n        "mean_r", "prob_mean_r_positive", "mean_r_p_value", "mean_r_ci_low",\n        "mean_r_ci_high", "positive_return_folds_pct", "bh_q_value",\n        "trade_ledger_hash",\n    ]\n    rows = []\n    for _, row in x.iterrows():\n        rows.append({c: _encode_value(row[c]) if c in x.columns else None for c in cols})\n    raw = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")\n    return hashlib.sha256(raw).hexdigest()[:16]\n', 'strategies.py': 'import numpy as np\nimport pandas as pd\nfrom .models import Signal\n\n\ndef _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:\n    prev_close = df["close"].shift(1)\n    tr = pd.concat([\n        df["high"] - df["low"],\n        (df["high"] - prev_close).abs(),\n        (df["low"] - prev_close).abs(),\n    ], axis=1).max(axis=1)\n    return tr.rolling(n).mean()\n\n\ndef precompute_indicators(df: pd.DataFrame) -> pd.DataFrame:\n    close = df["close"]\n    ma20 = close.rolling(20).mean()\n    ma50 = close.rolling(50).mean()\n    ma200 = close.rolling(200).mean()\n    sd20 = close.rolling(20).std()\n    returns = close.pct_change()\n    bb_width = (4.0 * sd20) / ma20.replace(0, np.nan)\n    # The squeeze threshold uses only widths known BEFORE the current bar.\n    bb_width_q25 = bb_width.shift(1).rolling(120).quantile(0.25)\n    return pd.DataFrame({\n        "atr14": _atr(df, 14),\n        "ma20": ma20,\n        "ma50": ma50,\n        "ma200": ma200,\n        "recent_high20": df["high"].shift(1).rolling(20).max(),\n        "recent_high55": df["high"].shift(1).rolling(55).max(),\n        "z20": (close - ma20) / sd20.replace(0, np.nan),\n        "vol5": returns.rolling(5).std(),\n        "vol30": returns.rolling(30).std(),\n        "bb_width": bb_width,\n        "bb_width_q25": bb_width_q25,\n    }, index=df.index)\n\n\ndef _clip_quality(x: float) -> float:\n    return float(np.clip(x, 0.50, 0.95))\n\n\ndef scan_at(\n    symbol: str,\n    df: pd.DataFrame,\n    i: int,\n    indicators: pd.DataFrame | None = None,\n    allowed_strategies: set[str] | frozenset[str] | None = None,\n):\n    """Generate close-of-bar signals using only data at or before bar i.\n\n    `allowed_strategies` is a performance filter only. It never changes the\n    calculations for a strategy that remains enabled.\n    """\n    if i < 0:\n        i = len(df) + i\n    if i < 1 or i >= len(df):\n        return []\n    ind = indicators if indicators is not None else precompute_indicators(df)\n    allowed = allowed_strategies\n\n    px = float(df["close"].iat[i])\n    prev_px = float(df["close"].iat[i - 1])\n    row = ind.iloc[i]\n    prev_row = ind.iloc[i - 1]\n    a_raw = row["atr14"]\n    a = float(a_raw) if np.isfinite(a_raw) else np.nan\n    if not np.isfinite(a) or a <= 0:\n        return []\n\n    out = []\n\n    # 1) 20-bar trend/momentum breakout.\n    if (allowed is None or "momentum_breakout" in allowed) and i >= 54:\n        recent_high = row["recent_high20"]\n        fast = row["ma20"]\n        slow = row["ma50"]\n        if np.isfinite(recent_high) and np.isfinite(fast) and np.isfinite(slow) and px > float(recent_high) and fast > slow:\n            breakout_strength = max(0.0, (px - float(recent_high)) / a)\n            trend_strength = max(0.0, (float(fast) - float(slow)) / a)\n            quality = _clip_quality(0.58 + 0.10 * breakout_strength + 0.035 * trend_strength)\n            out.append(Signal(\n                symbol, "momentum_breakout", "BUY",\n                px, px - 1.5*a, px + 3.0*a, quality,\n                f"20-bar breakout; trend strength={trend_strength:.2f} ATR"\n            ))\n\n    # 2) Mean reversion after an unusual downside move and first reversal bar.\n    if (allowed is None or "mean_reversion" in allowed) and i >= 29:\n        z = row["z20"]\n        ma = row["ma20"]\n        if np.isfinite(z) and np.isfinite(ma) and float(z) < -2.0 and px > prev_px:\n            severity = max(0.0, -float(z) - 2.0)\n            reversal = max(0.0, (px - prev_px) / a)\n            quality = _clip_quality(0.56 + 0.08 * severity + 0.08 * reversal)\n            out.append(Signal(\n                symbol, "mean_reversion", "BUY",\n                px, px - 1.25*a, float(ma), quality,\n                f"Oversold z={float(z):.2f}; reversal={reversal:.2f} ATR"\n            ))\n\n    # 3) Volatility expansion aligned with the medium-term trend.\n    if (allowed is None or "volatility_expansion" in allowed) and i >= 34:\n        v5 = row["vol5"]\n        v30 = row["vol30"]\n        ma = row["ma20"]\n        if (\n            np.isfinite(v5) and np.isfinite(v30) and np.isfinite(ma)\n            and float(v30) > 0\n            and float(v5) > 1.6 * float(v30)\n            and px > float(ma)\n            and px > prev_px\n        ):\n            vol_ratio = float(v5) / float(v30)\n            trend_distance = max(0.0, (px - float(ma)) / a)\n            quality = _clip_quality(0.54 + 0.07 * (vol_ratio - 1.6) + 0.04 * trend_distance)\n            out.append(Signal(\n                symbol, "volatility_expansion", "BUY",\n                px, px - 1.4*a, px + 2.8*a, quality,\n                f"Vol expansion {vol_ratio:.2f}x; trend distance={trend_distance:.2f} ATR"\n            ))\n\n    # 4) Long-trend pullback/reclaim.\n    if (allowed is None or "trend_pullback" in allowed) and i >= 204:\n        ma20 = row["ma20"]\n        ma20_prev = prev_row["ma20"]\n        ma50 = row["ma50"]\n        ma200 = row["ma200"]\n        if (\n            np.isfinite(ma20) and np.isfinite(ma20_prev) and np.isfinite(ma50) and np.isfinite(ma200)\n            and float(ma50) > float(ma200)\n            and px > float(ma200)\n            and prev_px <= float(ma20_prev)\n            and px > float(ma20)\n            and px > prev_px\n        ):\n            reclaim = max(0.0, (px - float(ma20)) / a)\n            trend_strength = max(0.0, (float(ma50) - float(ma200)) / a)\n            quality = _clip_quality(0.57 + 0.07*reclaim + 0.02*trend_strength)\n            out.append(Signal(\n                symbol, "trend_pullback", "BUY",\n                px, px - 1.5*a, px + 3.0*a, quality,\n                f"20-day reclaim inside 50>200 trend; trend={trend_strength:.2f} ATR"\n            ))\n\n    # 5) Slower 55-day Donchian-style breakout in a long-term uptrend.\n    if (allowed is None or "donchian_55_breakout" in allowed) and i >= 204:\n        high55 = row["recent_high55"]\n        ma50 = row["ma50"]\n        ma200 = row["ma200"]\n        if (\n            np.isfinite(high55) and np.isfinite(ma50) and np.isfinite(ma200)\n            and px > float(high55)\n            and float(ma50) > float(ma200)\n        ):\n            breakout_strength = max(0.0, (px - float(high55)) / a)\n            trend_strength = max(0.0, (float(ma50) - float(ma200)) / a)\n            quality = _clip_quality(0.58 + 0.08*breakout_strength + 0.02*trend_strength)\n            out.append(Signal(\n                symbol, "donchian_55_breakout", "BUY",\n                px, px - 2.0*a, px + 4.0*a, quality,\n                f"55-day breakout in 50>200 trend; strength={breakout_strength:.2f} ATR"\n            ))\n\n    # 6) Breakout after a prior-bar volatility squeeze.\n    if (allowed is None or "squeeze_breakout" in allowed) and i >= 125:\n        prev_width = prev_row["bb_width"]\n        prev_q25 = prev_row["bb_width_q25"]\n        high20 = row["recent_high20"]\n        ma50 = row["ma50"]\n        if (\n            np.isfinite(prev_width) and np.isfinite(prev_q25)\n            and np.isfinite(high20) and np.isfinite(ma50)\n            and float(prev_width) < float(prev_q25)\n            and px > float(high20)\n            and px > float(ma50)\n            and px > prev_px\n        ):\n            compression = max(0.0, (float(prev_q25) - float(prev_width)) / max(float(prev_q25), 1e-12))\n            breakout_strength = max(0.0, (px - float(high20)) / a)\n            quality = _clip_quality(0.57 + 0.10*compression + 0.06*breakout_strength)\n            out.append(Signal(\n                symbol, "squeeze_breakout", "BUY",\n                px, px - 1.4*a, px + 2.8*a, quality,\n                f"Prior-bar volatility squeeze + 20-day breakout; compression={compression:.2f}"\n            ))\n    return out\n\n\ndef momentum_breakout(symbol: str, df: pd.DataFrame):\n    return next((s for s in scan_at(symbol, df, len(df)-1) if s.strategy == "momentum_breakout"), None)\n\ndef mean_reversion(symbol: str, df: pd.DataFrame):\n    return next((s for s in scan_at(symbol, df, len(df)-1) if s.strategy == "mean_reversion"), None)\n\ndef volatility_expansion(symbol: str, df: pd.DataFrame):\n    return next((s for s in scan_at(symbol, df, len(df)-1) if s.strategy == "volatility_expansion"), None)\n\ndef trend_pullback(symbol: str, df: pd.DataFrame):\n    return next((s for s in scan_at(symbol, df, len(df)-1) if s.strategy == "trend_pullback"), None)\n\ndef donchian_55_breakout(symbol: str, df: pd.DataFrame):\n    return next((s for s in scan_at(symbol, df, len(df)-1) if s.strategy == "donchian_55_breakout"), None)\n\ndef squeeze_breakout(symbol: str, df: pd.DataFrame):\n    return next((s for s in scan_at(symbol, df, len(df)-1) if s.strategy == "squeeze_breakout"), None)\n\n\nSTRATEGIES = [\n    momentum_breakout,\n    mean_reversion,\n    volatility_expansion,\n    trend_pullback,\n    donchian_55_breakout,\n    squeeze_breakout,\n]\n\n\ndef scan(symbol: str, df: pd.DataFrame):\n    if len(df) < 2:\n        return []\n    indicators = precompute_indicators(df)\n    return scan_at(symbol, df, len(df)-1, indicators)\n', 'walkforward.py': 'from __future__ import annotations\n\nfrom dataclasses import dataclass, asdict, replace\nimport math\nfrom typing import List\nimport numpy as np\nimport pandas as pd\n\nfrom .backtest import run_backtest\nfrom .config import RiskConfig\nfrom .strategies import STRATEGIES\nfrom .montecarlo import simulate_r_paths\n\nSTRATEGY_NAMES = [fn.__name__ for fn in STRATEGIES]\n\n@dataclass\nclass WalkForwardFold:\n    fold: int\n    train_start: str\n    train_end: str\n    test_start: str\n    test_end: str\n    selected_strategy: str\n    train_score: float | None\n    train_trades: int\n    test_return_pct: float\n    benchmark_return_pct: float\n    alpha_vs_benchmark_pct: float\n    test_drawdown_pct: float\n    test_sharpe: float\n    test_profit_factor: float\n    test_trades: int\n    mean_r: float\n\n    def to_dict(self):\n        return asdict(self)\n\n@dataclass\nclass WalkForwardResult:\n    symbol: str\n    folds: List[WalkForwardFold]\n    summary: dict\n\ndef _finite_pf(value: float) -> float:\n    if value == math.inf:\n        return 5.0\n    if not np.isfinite(value):\n        return 0.0\n    return float(value)\n\ndef _train_score(metrics: dict, min_train_trades: int) -> float | None:\n    if metrics["trades"] < min_train_trades:\n        return None\n    pf = _finite_pf(metrics["profit_factor"])\n    if pf <= 1.0 or metrics["sharpe"] <= 0 or metrics["total_return_pct"] <= 0:\n        return None\n    return float(\n        metrics["alpha_vs_benchmark_pct"]\n        + 1.5 * metrics["sharpe"]\n        + 1.0 * min(pf, 3.0)\n        + 0.5 * metrics.get("mean_r", 0.0)\n        - 0.35 * metrics["max_drawdown_pct"]\n        - 0.05 * metrics.get("turnover_x", 0.0)\n    )\n\ndef _cash_metrics(test_df: pd.DataFrame) -> dict:\n    bench = float(test_df["close"].iloc[-1] / test_df["close"].iloc[0] - 1.0) * 100 if len(test_df) >= 2 else 0.0\n    return {\n        "total_return_pct": 0.0, "benchmark_return_pct": bench, "alpha_vs_benchmark_pct": -bench,\n        "max_drawdown_pct": 0.0, "sharpe": 0.0, "trades": 0, "profit_factor": 0.0,\n        "mean_r": 0.0, "bootstrap_prob_mean_r_positive": 0.0,\n    }\n\ndef _bootstrap_all_r(r_values: list[float], samples: int = 5000, seed: int = 91):\n    r = np.array([x for x in r_values if np.isfinite(x)], dtype=float)\n    if len(r) < 2:\n        return 0.0, 0.0, 0.0, 0.0\n    rng = np.random.default_rng(seed)\n    means = rng.choice(r, size=(samples, len(r)), replace=True).mean(axis=1)\n    return float(r.mean()), float(np.quantile(means, .025)), float(np.quantile(means, .975)), float((means > 0).mean())\n\ndef run_walk_forward(\n    symbol: str, df: pd.DataFrame, cfg: RiskConfig,\n    train_bars: int = 504, test_bars: int = 126, warmup: int = 220,\n    min_train_trades: int = 3, embargo_bars: int = 1,\n) -> WalkForwardResult:\n    """Rolling walk-forward with a small embargo between train and unseen test.\n    Strategy selection uses training data only. The selected strategy is frozen for\n    the next fold. A PASS is intentionally hard to achieve.\n    """\n    if train_bars < warmup + 30:\n        raise ValueError("train_bars is too short")\n    if test_bars < 20:\n        raise ValueError("test_bars is too short")\n    if embargo_bars < 0:\n        raise ValueError("embargo_bars must be >=0")\n    if len(df) < train_bars + embargo_bars + test_bars:\n        raise ValueError("Not enough history for one walk-forward fold")\n\n    folds: List[WalkForwardFold] = []\n    all_oos_r: list[float] = []\n    start = 0\n    fold_num = 1\n    while start + train_bars + embargo_bars + test_bars <= len(df):\n        train = df.iloc[start:start + train_bars]\n        test_start_i = start + train_bars + embargo_bars\n        test = df.iloc[test_start_i:test_start_i + test_bars]\n\n        best_name = "CASH"\n        best_score = None\n        best_train_trades = 0\n        for name in STRATEGY_NAMES:\n            try:\n                res = run_backtest(symbol, train, cfg, warmup=warmup, strategy_names={name})\n            except ValueError:\n                continue\n            score = _train_score(res.metrics, min_train_trades)\n            if score is not None and (best_score is None or score > best_score):\n                best_name, best_score, best_train_trades = name, score, int(res.metrics["trades"])\n\n        if best_name == "CASH":\n            tm = _cash_metrics(test)\n        else:\n            prefix_end = test_start_i\n            prefix = df.iloc[max(0, prefix_end-warmup):prefix_end]\n            test_input = pd.concat([prefix, test]).sort_index()\n            result = run_backtest(symbol, test_input, cfg, warmup=min(warmup, len(prefix)), strategy_names={best_name})\n            tm = result.metrics\n            all_oos_r.extend([t.r_multiple for t in result.trades])\n\n        folds.append(WalkForwardFold(\n            fold=fold_num, train_start=str(train.index[0]), train_end=str(train.index[-1]),\n            test_start=str(test.index[0]), test_end=str(test.index[-1]), selected_strategy=best_name,\n            train_score=None if best_score is None else float(best_score), train_trades=best_train_trades,\n            test_return_pct=float(tm["total_return_pct"]), benchmark_return_pct=float(tm["benchmark_return_pct"]),\n            alpha_vs_benchmark_pct=float(tm["alpha_vs_benchmark_pct"]), test_drawdown_pct=float(tm["max_drawdown_pct"]),\n            test_sharpe=float(tm["sharpe"]), test_profit_factor=float(tm["profit_factor"]),\n            test_trades=int(tm["trades"]), mean_r=float(tm.get("mean_r", 0.0)),\n        ))\n        fold_num += 1\n        start += test_bars\n\n    returns = np.array([f.test_return_pct / 100 for f in folds], dtype=float)\n    bench = np.array([f.benchmark_return_pct / 100 for f in folds], dtype=float)\n    compounded = float(np.prod(1 + returns) - 1) if len(returns) else 0.0\n    compounded_bench = float(np.prod(1 + bench) - 1) if len(bench) else 0.0\n    active = [f for f in folds if f.selected_strategy != "CASH"]\n    active_pfs = [_finite_pf(f.test_profit_factor) for f in active if f.test_trades > 0]\n    positive_alpha = float(np.mean([f.alpha_vs_benchmark_pct > 0 for f in folds])) if folds else 0.0\n    total_trades = int(sum(f.test_trades for f in folds))\n    worst_dd = max([f.test_drawdown_pct for f in folds], default=0.0)\n    median_pf = float(np.median(active_pfs)) if active_pfs else 0.0\n    median_sharpe = float(np.median([f.test_sharpe for f in active])) if active else 0.0\n    mean_r, ci_low, ci_high, prob_pos = _bootstrap_all_r(all_oos_r)\n    mc = simulate_r_paths(all_oos_r, risk_fraction=cfg.max_risk_per_trade_pct, trades_per_path=100, paths=5000)\n\n    if len(folds) < cfg.min_oos_folds or total_trades < cfg.min_oos_trades:\n        gate = "INSUFFICIENT_DATA"\n    elif (\n        compounded > 0\n        and compounded > compounded_bench\n        and positive_alpha >= cfg.min_positive_alpha_folds\n        and worst_dd <= cfg.max_drawdown_pct * 100\n        and median_pf > 1.05\n        and mean_r > 0\n        and prob_pos >= cfg.min_bootstrap_prob_positive\n    ):\n        gate = "PASS_RESEARCH_GATE"\n    else:\n        gate = "FAIL_RESEARCH_GATE"\n\n    summary = {\n        "folds": len(folds), "active_folds": len(active),\n        "oos_compounded_return_pct": compounded * 100,\n        "oos_benchmark_return_pct": compounded_bench * 100,\n        "oos_alpha_vs_benchmark_pct": (compounded - compounded_bench) * 100,\n        "positive_alpha_folds_pct": positive_alpha * 100, "total_oos_trades": total_trades,\n        "worst_oos_drawdown_pct": worst_dd, "median_oos_profit_factor": median_pf,\n        "median_oos_sharpe": median_sharpe, "oos_mean_r": mean_r,\n        "bootstrap_mean_r_ci_low": ci_low, "bootstrap_mean_r_ci_high": ci_high,\n        "bootstrap_prob_mean_r_positive": prob_pos, "embargo_bars": embargo_bars,\n        "mc_paths": mc["paths"],\n        "mc_100_trade_median_return_pct": mc["median_return_pct"],\n        "mc_100_trade_p05_return_pct": mc["p05_return_pct"],\n        "mc_100_trade_p95_drawdown_pct": mc["p95_max_drawdown_pct"],\n        "mc_prob_finish_negative": mc["prob_finish_negative"],\n        "research_gate": gate,\n    }\n    return WalkForwardResult(symbol=symbol, folds=folds, summary=summary)\n\ndef cost_stress_walk_forward(symbol: str, df: pd.DataFrame, cfg: RiskConfig, multipliers=(1.0, 2.0, 3.0), **kwargs) -> pd.DataFrame:\n    rows = []\n    for m in multipliers:\n        stressed = replace(cfg, simulated_fee_bps=cfg.simulated_fee_bps*m, simulated_slippage_bps=cfg.simulated_slippage_bps*m)\n        r = run_walk_forward(symbol, df, stressed, **kwargs)\n        s = r.summary\n        rows.append({\n            "cost_multiplier": m, "fee_bps": stressed.simulated_fee_bps, "slippage_bps": stressed.simulated_slippage_bps,\n            "oos_return_pct": s["oos_compounded_return_pct"], "benchmark_pct": s["oos_benchmark_return_pct"],\n            "alpha_pct": s["oos_alpha_vs_benchmark_pct"], "trades": s["total_oos_trades"],\n            "mean_r": s["oos_mean_r"], "prob_positive_r": s["bootstrap_prob_mean_r_positive"],\n            "gate": s["research_gate"],\n        })\n    out = pd.DataFrame(rows)\n    if len(out):\n        base = out.iloc[0]\n        two = out.iloc[1] if len(out) > 1 else base\n        out.attrs["cost_resilient"] = bool(\n            base["oos_return_pct"] > 0 and base["alpha_pct"] > 0\n            and two["oos_return_pct"] > 0 and two["alpha_pct"] > 0\n        )\n    return out\n', 'ledger.py': '\nfrom __future__ import annotations\n\nfrom dataclasses import asdict, is_dataclass\nimport copy\nimport hashlib\nimport inspect\nimport json\nimport math\nfrom typing import Any\n\nimport numpy as np\nimport pandas as pd\n\nfrom .strategies import STRATEGIES\n\nLEDGER_FORMAT_VERSION = 1\nLEDGER_PROTOCOL_VERSION = "0.7"\nCUMULATIVE_Q_THRESHOLD = 0.10\n\n\ndef _jsonable(value: Any):\n    if is_dataclass(value):\n        return {k: _jsonable(v) for k, v in asdict(value).items()}\n    if isinstance(value, dict):\n        return {str(k): _jsonable(v) for k, v in value.items()}\n    if isinstance(value, (list, tuple)):\n        return [_jsonable(v) for v in value]\n    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):\n        return int(value)\n    if isinstance(value, (np.floating, float)):\n        f = float(value)\n        if math.isnan(f):\n            return "NaN"\n        if math.isinf(f):\n            return "+Inf" if f > 0 else "-Inf"\n        return f\n    if isinstance(value, (np.bool_, bool)):\n        return bool(value)\n    if value is None:\n        return None\n    return str(value)\n\n\ndef _sha(payload: Any, n: int = 16) -> str:\n    raw = json.dumps(_jsonable(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")\n    return hashlib.sha256(raw).hexdigest()[:n]\n\n\ndef empty_ledger() -> dict[str, Any]:\n    return {\n        "format_version": LEDGER_FORMAT_VERSION,\n        "protocol_version": LEDGER_PROTOCOL_VERSION,\n        "next_batch_sequence": 1,\n        "next_event_sequence": 1,\n        "batches": [],\n        "events": [],\n        "notes": [\n            "Cumulative ledger started at Alpha v0.7. Pre-v0.7 evidence imported later is marked legacy.",\n        ],\n    }\n\n\ndef validate_ledger(ledger: dict[str, Any]) -> dict[str, Any]:\n    if not isinstance(ledger, dict):\n        raise ValueError("research ledger must be a JSON object")\n    if int(ledger.get("format_version", -1)) != LEDGER_FORMAT_VERSION:\n        raise ValueError("unsupported research ledger format version")\n    if str(ledger.get("protocol_version")) != LEDGER_PROTOCOL_VERSION:\n        raise ValueError("unsupported research ledger protocol version")\n    out = copy.deepcopy(ledger)\n    out.setdefault("next_batch_sequence", 1)\n    out.setdefault("next_event_sequence", 1)\n    out.setdefault("batches", [])\n    out.setdefault("events", [])\n    out.setdefault("notes", [])\n    # Structural uniqueness checks matter because duplicate evidence could\n    # accidentally reduce a multiple-testing penalty.\n    batch_ids = [str(x.get("batch_id")) for x in out["batches"]]\n    if len(batch_ids) != len(set(batch_ids)):\n        raise ValueError("research ledger contains duplicate batch IDs")\n    event_ids = [str(x.get("event_id")) for x in out["events"]]\n    if len(event_ids) != len(set(event_ids)):\n        raise ValueError("research ledger contains duplicate event IDs")\n    return out\n\n\ndef ledger_fingerprint(ledger: dict[str, Any]) -> str:\n    x = validate_ledger(ledger)\n    # Counters are derived bookkeeping; evidence content defines the fingerprint.\n    payload = {\n        "format_version": x["format_version"],\n        "protocol_version": x["protocol_version"],\n        "batches": sorted(x["batches"], key=lambda z: (int(z.get("sequence", 0)), str(z.get("batch_id", "")))),\n        "events": sorted(x["events"], key=lambda z: (int(z.get("sequence", 0)), str(z.get("event_id", "")))),\n        "notes": x.get("notes", []),\n    }\n    return _sha(payload, 16)\n\n\ndef ledger_json_bytes(ledger: dict[str, Any]) -> bytes:\n    x = validate_ledger(ledger)\n    return json.dumps(x, sort_keys=True, indent=2, ensure_ascii=False).encode("utf-8")\n\n\ndef load_ledger_json(raw: bytes) -> dict[str, Any]:\n    if not raw:\n        raise ValueError("empty research ledger")\n    return validate_ledger(json.loads(raw.decode("utf-8")))\n\n\ndef strategy_fingerprint(strategy_name: str) -> str:\n    by_name = {fn.__name__: fn for fn in STRATEGIES}\n    if strategy_name not in by_name:\n        raise ValueError(f"unknown strategy: {strategy_name}")\n    fn = by_name[strategy_name]\n    try:\n        src = inspect.getsource(fn)\n    except Exception:\n        code = getattr(fn, "__code__", None)\n        src = repr((\n            getattr(code, "co_code", b""),\n            getattr(code, "co_consts", ()),\n            getattr(code, "co_names", ()),\n        ))\n    return hashlib.sha256(src.encode("utf-8") if isinstance(src, str) else bytes(src)).hexdigest()[:16]\n\n\ndef risk_signature(cfg) -> str:\n    return _sha(_jsonable(cfg), 16)\n\n\ndef hypothesis_id(symbol: str, strategy: str, cfg, scientific_protocol: str = "0.6") -> str:\n    payload = {\n        "scientific_protocol": str(scientific_protocol),\n        "symbol": str(symbol),\n        "strategy": str(strategy),\n        "strategy_fingerprint": strategy_fingerprint(str(strategy)),\n        "risk_signature": risk_signature(cfg),\n    }\n    return _sha(payload, 16)\n\n\ndef _batch_hypotheses(symbols, strategies, cfg, scientific_protocol="0.6") -> list[dict[str, str]]:\n    rows = []\n    for symbol in sorted(set(str(s) for s in symbols)):\n        for strategy in sorted(set(str(s) for s in strategies)):\n            rows.append({\n                "symbol": symbol,\n                "strategy": strategy,\n                "hypothesis_id": hypothesis_id(symbol, strategy, cfg, scientific_protocol),\n            })\n    return rows\n\n\ndef _find_batch(ledger, *, spec_id, dataset_id, hypotheses_signature):\n    for batch in ledger.get("batches", []):\n        if (\n            str(batch.get("spec_id")) == str(spec_id)\n            and str(batch.get("dataset_id")) == str(dataset_id)\n            and str(batch.get("hypotheses_signature")) == str(hypotheses_signature)\n        ):\n            return batch\n    return None\n\n\ndef preregister_batch(\n    ledger: dict[str, Any],\n    *,\n    spec_id: str,\n    dataset_id: str,\n    symbols,\n    strategies,\n    cfg,\n    scientific_protocol: str = "0.6",\n    engine_fingerprint: str = "",\n    legacy: bool = False,\n) -> tuple[dict[str, Any], dict[str, Any], bool]:\n    """Register a test plan before its current result is examined.\n\n    Exact reruns of an already-finalized spec+dataset+hypothesis family are\n    idempotent and are NOT counted as another look.\n    """\n    x = validate_ledger(ledger)\n    hypotheses = _batch_hypotheses(symbols, strategies, cfg, scientific_protocol)\n    hsig = _sha(hypotheses, 16)\n    existing = _find_batch(x, spec_id=spec_id, dataset_id=dataset_id, hypotheses_signature=hsig)\n    if existing is not None:\n        return x, existing, False\n\n    seq = int(x.get("next_batch_sequence", 1))\n    batch_id = _sha({\n        "protocol": LEDGER_PROTOCOL_VERSION,\n        "sequence": seq,\n        "spec_id": spec_id,\n        "dataset_id": dataset_id,\n        "hypotheses_signature": hsig,\n        "ledger_before": ledger_fingerprint(x),\n    }, 16)\n    batch = {\n        "batch_id": batch_id,\n        "sequence": seq,\n        "spec_id": str(spec_id),\n        "dataset_id": str(dataset_id),\n        "scientific_protocol": str(scientific_protocol),\n        "engine_fingerprint_at_plan": str(engine_fingerprint),\n        "hypotheses_signature": hsig,\n        "hypotheses": hypotheses,\n        "preregistered": not bool(legacy),\n        "legacy": bool(legacy),\n        "finalized": False,\n        "base_result_fingerprint": None,\n    }\n    x["batches"].append(batch)\n    x["next_batch_sequence"] = seq + 1\n    return x, batch, True\n\n\ndef _event_lookup(ledger):\n    return {str(e["event_id"]): e for e in ledger.get("events", [])}\n\n\ndef finalize_batch(\n    ledger: dict[str, Any],\n    batch_id: str,\n    matrix: pd.DataFrame,\n    *,\n    base_result_fingerprint: str,\n) -> dict[str, Any]:\n    """Attach pair results to a preregistered batch.\n\n    Re-finalizing the exact batch is allowed only if every evidence field matches.\n    A mismatch is treated as a reproducibility error, not as a fresh statistical look.\n    """\n    x = validate_ledger(ledger)\n    batch = next((b for b in x["batches"] if str(b.get("batch_id")) == str(batch_id)), None)\n    if batch is None:\n        raise ValueError("cannot finalize an unknown research batch")\n    if matrix is None or matrix.empty:\n        raise ValueError("cannot finalize an empty research matrix")\n\n    planned = {(h["symbol"], h["strategy"]): h for h in batch.get("hypotheses", [])}\n    observed_pairs = set(zip(matrix["symbol"].astype(str), matrix["strategy"].astype(str)))\n    if set(planned) != observed_pairs:\n        missing = sorted(set(planned) - observed_pairs)\n        extra = sorted(observed_pairs - set(planned))\n        raise ValueError(f"research matrix does not match preregistered hypotheses; missing={missing[:3]} extra={extra[:3]}")\n\n    existing_events = _event_lookup(x)\n    new_events = []\n    for _, row in matrix.iterrows():\n        key = (str(row["symbol"]), str(row["strategy"]))\n        h = planned[key]\n        event_id = _sha({"batch_id": batch_id, "hypothesis_id": h["hypothesis_id"]}, 20)\n        p = float(row.get("mean_r_p_value", 1.0))\n        if not np.isfinite(p):\n            p = 1.0\n        event = {\n            "event_id": event_id,\n            "sequence": None,\n            "batch_id": str(batch_id),\n            "hypothesis_id": h["hypothesis_id"],\n            "symbol": key[0],\n            "strategy": key[1],\n            "spec_id": batch["spec_id"],\n            "dataset_id": batch["dataset_id"],\n            "preregistered": bool(batch.get("preregistered", False)),\n            "legacy": bool(batch.get("legacy", False)),\n            "raw_p": float(np.clip(p, 0.0, 1.0)),\n            "batch_bh_q": float(row.get("bh_q_value", 1.0)),\n            "trades": int(row.get("trades", 0)),\n            "mean_r": float(row.get("mean_r", 0.0)),\n            "profit_factor": float(row.get("profit_factor", 0.0)),\n            "trade_ledger_hash": str(row.get("trade_ledger_hash", "")),\n            "base_status": str(row.get("status", "")),\n            "base_result_fingerprint": str(base_result_fingerprint),\n        }\n        old = existing_events.get(event_id)\n        if old is not None:\n            comparable = dict(event)\n            comparable["sequence"] = old.get("sequence")\n            if _sha(comparable, 32) != _sha(old, 32):\n                raise ValueError(\n                    f"reproducibility conflict for {key[0]} / {key[1]}: "\n                    "same preregistered batch produced different evidence"\n                )\n            continue\n        new_events.append(event)\n\n    seq = int(x.get("next_event_sequence", 1))\n    for event in new_events:\n        event["sequence"] = seq\n        seq += 1\n        x["events"].append(event)\n    x["next_event_sequence"] = seq\n\n    batch["finalized"] = True\n    batch["base_result_fingerprint"] = str(base_result_fingerprint)\n    batch["event_count"] = len(planned)\n    return x\n\n\ndef _anytime_weight(look_index: int) -> float:\n    # Sum_{j>=1} 6/(pi^2 j^2) = 1. This alpha-spending schedule gives an\n    # anytime-valid Bonferroni-style p-value for repeated looks at one hypothesis.\n    j = max(1, int(look_index))\n    return 6.0 / (math.pi ** 2 * j * j)\n\n\ndef anytime_pvalues(ledger: dict[str, Any]) -> pd.DataFrame:\n    """One conservative p-value per unique hypothesis across repeated looks."""\n    x = validate_ledger(ledger)\n    if not x["events"]:\n        return pd.DataFrame(columns=[\n            "hypothesis_id", "symbol", "strategy", "looks", "legacy_looks",\n            "preregistered_looks", "anytime_p",\n        ])\n    rows = []\n    df = pd.DataFrame(x["events"]).sort_values(["sequence", "event_id"], kind="stable")\n    for hid, g in df.groupby("hypothesis_id", sort=True):\n        g = g.sort_values(["sequence", "event_id"], kind="stable").reset_index(drop=True)\n        adjusted = []\n        for j, row in g.iterrows():\n            # Legacy/pre-ledger exploration consumes an alpha-spending position but\n            # is not allowed to supply affirmative evidence. It can only make later\n            # preregistered looks harder, never easier.\n            if bool(row.get("preregistered", False)) and not bool(row.get("legacy", False)):\n                look_index = int(j) + 1\n                adjusted.append(min(1.0, float(row["raw_p"]) / _anytime_weight(look_index)))\n        rows.append({\n            "hypothesis_id": str(hid),\n            "symbol": str(g.iloc[0]["symbol"]),\n            "strategy": str(g.iloc[0]["strategy"]),\n            "looks": int(len(g)),\n            "legacy_looks": int(g["legacy"].astype(bool).sum()),\n            "preregistered_looks": int(g["preregistered"].astype(bool).sum()),\n            "anytime_p": float(min(adjusted) if adjusted else 1.0),\n        })\n    return pd.DataFrame(rows)\n\n\ndef benjamini_yekutieli_qvalues(p_values) -> np.ndarray:\n    """BY FDR q-values; valid under arbitrary dependence, more conservative than BH."""\n    p = np.asarray(p_values, dtype=float)\n    p = np.where(np.isfinite(p), np.clip(p, 0.0, 1.0), 1.0)\n    m = len(p)\n    if m == 0:\n        return np.asarray([], dtype=float)\n    order = np.argsort(p, kind="stable")\n    ranked = p[order]\n    harmonic = float(np.sum(1.0 / np.arange(1, m + 1, dtype=float)))\n    q_ranked = ranked * m * harmonic / np.arange(1, m + 1, dtype=float)\n    q_ranked = np.minimum.accumulate(q_ranked[::-1])[::-1]\n    q_ranked = np.clip(q_ranked, 0.0, 1.0)\n    q = np.empty(m, dtype=float)\n    q[order] = q_ranked\n    return q\n\n\ndef cumulative_table(ledger: dict[str, Any]) -> pd.DataFrame:\n    out = anytime_pvalues(ledger)\n    if out.empty:\n        out["ledger_by_q"] = []\n        return out\n    out["ledger_by_q"] = benjamini_yekutieli_qvalues(out["anytime_p"].to_numpy(float))\n    return out\n\n\ndef apply_cumulative_gate(\n    matrix: pd.DataFrame,\n    ledger: dict[str, Any],\n    *,\n    cfg,\n    scientific_protocol: str = "0.6",\n    batch_id: str | None = None,\n    threshold: float = CUMULATIVE_Q_THRESHOLD,\n) -> pd.DataFrame:\n    """Add cumulative evidence columns and only further RESTRICT candidate status."""\n    x = matrix.copy()\n    table = cumulative_table(ledger)\n    by_h = {} if table.empty else table.set_index("hypothesis_id").to_dict("index")\n    batch = None\n    if batch_id is not None:\n        batch = next((b for b in ledger.get("batches", []) if str(b.get("batch_id")) == str(batch_id)), None)\n\n    hids = []\n    looks = []\n    legacy_looks = []\n    prereg_looks = []\n    anytime_ps = []\n    ledger_qs = []\n    ledger_gate = []\n\n    for idx, row in x.iterrows():\n        hid = hypothesis_id(str(row["symbol"]), str(row["strategy"]), cfg, scientific_protocol)\n        info = by_h.get(hid, {})\n        q = float(info.get("ledger_by_q", 1.0))\n        look_n = int(info.get("looks", 0))\n        leg_n = int(info.get("legacy_looks", 0))\n        pre_n = int(info.get("preregistered_looks", 0))\n        ap = float(info.get("anytime_p", 1.0))\n        current_preregistered = bool(batch and batch.get("preregistered", False))\n        current_legacy = bool(batch and batch.get("legacy", False))\n\n        if str(row.get("status")) == "CANDIDATE":\n            if current_legacy or not current_preregistered:\n                x.at[idx, "status"] = "WATCH"\n                x.at[idx, "why"] = str(row.get("why", "")) + "; cumulative ledger: current evidence is legacy/not preregistered"\n                gate = "BLOCK_LEGACY"\n            elif q > float(threshold):\n                x.at[idx, "status"] = "WATCH"\n                x.at[idx, "why"] = str(row.get("why", "")) + f"; cumulative BY q={q:.3f}>{threshold:.2f}"\n                gate = "BLOCK_CUMULATIVE_FDR"\n            else:\n                x.at[idx, "why"] = str(row.get("why", "")) + f"; cumulative BY q={q:.3f}<={threshold:.2f}"\n                gate = "PASS_CUMULATIVE_GATE"\n        else:\n            gate = "NOT_BATCH_CANDIDATE"\n\n        hids.append(hid)\n        looks.append(look_n)\n        legacy_looks.append(leg_n)\n        prereg_looks.append(pre_n)\n        anytime_ps.append(ap)\n        ledger_qs.append(q)\n        ledger_gate.append(gate)\n\n    x["hypothesis_id"] = hids\n    x["ledger_looks"] = looks\n    x["ledger_legacy_looks"] = legacy_looks\n    x["ledger_preregistered_looks"] = prereg_looks\n    x["ledger_anytime_p"] = anytime_ps\n    x["ledger_by_q"] = ledger_qs\n    x["ledger_gate"] = ledger_gate\n    return x\n\n\ndef ledger_selection_id(matrix: pd.DataFrame, *, spec_id: str, dataset_id: str, ledger: dict[str, Any]) -> tuple[dict[str, str], str]:\n    """Freeze at most one final CANDIDATE per asset after the cumulative gate."""\n    selections: dict[str, str] = {}\n    if matrix is not None and not matrix.empty:\n        for symbol, g in matrix.groupby("symbol"):\n            c = g[g["status"] == "CANDIDATE"]\n            if c.empty:\n                continue\n            c = c.sort_values(\n                ["ledger_by_q", "bh_q_value", "prob_mean_r_positive", "mean_r", "profit_factor"],\n                ascending=[True, True, False, False, False],\n            )\n            selections[str(symbol)] = str(c.iloc[0]["strategy"])\n    payload = {\n        "spec_id": str(spec_id),\n        "dataset_id": str(dataset_id),\n        "ledger_fingerprint": ledger_fingerprint(ledger),\n        "selections": selections,\n    }\n    return selections, _sha(payload, 12)\n\n\ndef matrix_with_ledger_fingerprint(matrix: pd.DataFrame) -> str:\n    if matrix is None or matrix.empty:\n        return hashlib.sha256(b"empty-v07-ledger-matrix").hexdigest()[:16]\n    cols = [\n        "symbol", "strategy", "status", "folds", "trades", "return_pct",\n        "profit_factor", "mean_r", "mean_r_p_value", "bh_q_value",\n        "trade_ledger_hash", "hypothesis_id", "ledger_looks", "ledger_legacy_looks",\n        "ledger_preregistered_looks", "ledger_anytime_p", "ledger_by_q", "ledger_gate",\n    ]\n    x = matrix.copy().sort_values(["symbol", "strategy"]).reset_index(drop=True)\n    payload = []\n    for _, row in x.iterrows():\n        item = {}\n        for c in cols:\n            value = row[c] if c in x.columns else None\n            if isinstance(value, (float, np.floating)):\n                f = float(value)\n                value = "NaN" if math.isnan(f) else ("+Inf" if math.isinf(f) and f > 0 else ("-Inf" if math.isinf(f) else f.hex()))\n            elif isinstance(value, (np.integer, int)):\n                value = int(value)\n            elif value is not None:\n                value = str(value)\n            item[c] = value\n        payload.append(item)\n    return _sha(payload, 16)\n\n\ndef merge_ledgers(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:\n    """Idempotent evidence merge; conflicts are rejected rather than silently resolved."""\n    left = validate_ledger(a)\n    right = validate_ledger(b)\n    out = copy.deepcopy(left)\n\n    batches = {str(x["batch_id"]): x for x in out["batches"]}\n    for batch in right["batches"]:\n        bid = str(batch["batch_id"])\n        if bid in batches and _sha(batches[bid], 32) != _sha(batch, 32):\n            raise ValueError(f"conflicting ledger batch {bid}")\n        if bid not in batches:\n            out["batches"].append(copy.deepcopy(batch))\n            batches[bid] = batch\n\n    events = {str(x["event_id"]): x for x in out["events"]}\n    for event in right["events"]:\n        eid = str(event["event_id"])\n        if eid in events and _sha(events[eid], 32) != _sha(event, 32):\n            raise ValueError(f"conflicting ledger event {eid}")\n        if eid not in events:\n            out["events"].append(copy.deepcopy(event))\n            events[eid] = event\n\n    # Re-sequence deterministically by original sequence, batch/event ID. Sequence\n    # is bookkeeping only; evidence identity is preserved.\n    out["batches"] = sorted(out["batches"], key=lambda z: (int(z.get("sequence", 0)), str(z["batch_id"])))\n    out["events"] = sorted(out["events"], key=lambda z: (int(z.get("sequence", 0)), str(z["event_id"])))\n    out["next_batch_sequence"] = max([int(x.get("sequence", 0)) for x in out["batches"]] + [0]) + 1\n    out["next_event_sequence"] = max([int(x.get("sequence", 0)) for x in out["events"]] + [0]) + 1\n    out["notes"] = list(dict.fromkeys(list(out.get("notes", [])) + list(right.get("notes", []))))\n    return validate_ledger(out)\n'}

_alpha_root = _AlphaPath(_alpha_tempfile.gettempdir()) / "alpha_money_hunter_runtime_v07"
_alpha_pkg = _alpha_root / "alpha"
_alpha_pkg.mkdir(parents=True, exist_ok=True)
for _name, _content in _ALPHA_SOURCES.items():
    (_alpha_pkg / _name).write_text(_content, encoding="utf-8")
if str(_alpha_root) not in _alpha_sys.path:
    _alpha_sys.path.insert(0, str(_alpha_root))


import streamlit as st
import pandas as pd
import hashlib
import json
import time
from datetime import date, timedelta

from alpha.config import RiskConfig
from alpha.data_provider import StooqDailyProvider, DataRequest, normalize_uploaded_csv
from alpha.data_quality import validate_ohlcv
from alpha.walkforward import run_walk_forward, cost_stress_walk_forward
from alpha.leaderboard import strategy_leaderboard
from alpha.portfolio import run_portfolio_backtest
from alpha.scanner import rank_watchlist
from alpha.registry import ExperimentRegistry, code_fingerprint
from alpha.integrity import run_integrity_checks
from alpha.research import discovery_screen, evaluate_frozen_holdout, family_summary, STRATEGY_NAMES
from alpha.snapshot import (
    build_snapshot_zip, compare_manifests, dataset_manifest, load_snapshot_zip,
    load_snapshot_ledger, matrix_result_fingerprint,
)
from alpha.ledger import (
    apply_cumulative_gate, cumulative_table, empty_ledger, finalize_batch,
    ledger_fingerprint, ledger_json_bytes, ledger_selection_id, load_ledger_json,
    matrix_with_ledger_fingerprint, merge_ledgers, preregister_batch, validate_ledger,
)

st.set_page_config(page_title="Alpha v0.7.1", layout="wide")
st.title("Alpha v0.7.1 — Mobile-safe Research Exports")
st.caption("Broker-agnostic, paper/research only. Exact datasets are reproducible and every hypothesis look is tracked cumulatively.")

cfg = RiskConfig()
provider = StooqDailyProvider()
registry = ExperimentRegistry()

with st.sidebar:
    st.subheader("Hard risk rules")
    st.write(f"Starting equity: **${cfg.starting_equity:,.0f}**")
    st.write(f"Risk / trade: **{cfg.max_risk_per_trade_pct:.1%}**")
    st.write(f"Portfolio heat: **{cfg.max_portfolio_heat_pct:.1%}**")
    st.write(f"Daily loss kill: **{cfg.max_daily_loss_pct:.1%}**")
    st.write(f"Max drawdown kill: **{cfg.max_drawdown_pct:.1%}**")
    st.write(f"Max position value: **{cfg.max_position_value_pct:.0%}**")
    st.write(f"Max open positions: **{cfg.max_open_positions}**")
    st.write(f"Max pair correlation: **{cfg.max_pair_correlation:.2f}**")
    st.write("Leverage: **OFF (1.0x max)**")
    st.divider()
    st.caption(f"Code fingerprint: `{code_fingerprint()}`")
    st.warning("A PASS is only a research filter. It is not authorization for real-money trading.")

mode = st.radio(
    "Mode",
    [
        "Audit overview", "Integrity audit", "Walk-forward lab", "Cost stress", "Multi-asset portfolio",
        "Research universe", "Research ledger", "Strategy leaderboard", "Current scanner", "CSV walk-forward", "Experiment registry"
    ],
    horizontal=True,
)

DEFAULT = "SPY.US, QQQ.US, IWM.US, DIA.US, MDY.US, EFA.US, EEM.US, GLD.US, SLV.US, TLT.US, IEF.US, HYG.US, XLE.US, XLF.US, XLK.US, XLV.US, XLI.US, XLP.US"


def _normalize_symbols(symbols):
    seen = set()
    out = []
    for raw in symbols:
        symbol = str(raw).strip().upper()
        if symbol and symbol not in seen:
            seen.add(symbol)
            out.append(symbol)
    return tuple(out)


def _cfg_key(cfg_obj):
    return tuple(sorted(cfg_obj.__dict__.items()))


@st.cache_data(ttl=21600, show_spinner=False)
def _cached_fetch_many(symbols_key, years, as_of):
    p = StooqDailyProvider()
    data = p.fetch_many(symbols_key, years=int(years))
    return data, tuple(p.last_errors), dict(p.source_by_symbol)


def load_many(symbols, years):
    symbols_key = _normalize_symbols(symbols)
    data, errors, sources = _cached_fetch_many(
        symbols_key, int(years), date.today().isoformat()
    )
    provider.last_errors = list(errors)
    provider.source_by_symbol = dict(sources)
    return data


def research_spec_payload(symbols_key, years, holdout_bars, burn_in_bars, fold_bars):
    return {
        "ruleset": "0.6",  # scientific rules unchanged; v0.7 adds cumulative hypothesis accounting.
        "symbols": list(symbols_key),
        "years": int(years),
        "holdout_bars": int(holdout_bars),
        "burn_in_bars": int(burn_in_bars),
        "fold_bars": int(fold_bars),
    }


def research_spec_id(symbols_key, years, holdout_bars, burn_in_bars, fold_bars):
    payload = research_spec_payload(symbols_key, years, holdout_bars, burn_in_bars, fold_bars)
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


def _snapshot_store():
    return st.session_state.setdefault("alpha_v07_snapshots", {})


def _run_store():
    return st.session_state.setdefault("alpha_v07_runs", {})


def _current_ledger():
    led = st.session_state.get("alpha_v07_ledger")
    if led is None:
        led = empty_ledger()
        st.session_state["alpha_v07_ledger"] = led
    return validate_ledger(led)


def _set_ledger(ledger):
    checked = validate_ledger(ledger)
    st.session_state["alpha_v07_ledger"] = checked
    return checked


def _register_snapshot(spec_id, data, sources, manifest, errors=()):
    store = _snapshot_store()
    bucket = store.setdefault(spec_id, {"records": {}, "active": None})
    dataset_id = manifest["dataset_id"]
    bucket["records"][dataset_id] = {
        "data": data,
        "sources": dict(sources),
        "manifest": manifest,
        "errors": tuple(errors),
    }
    bucket["active"] = dataset_id
    return bucket["records"][dataset_id]


def _active_snapshot(spec_id):
    bucket = _snapshot_store().get(spec_id)
    if not bucket or not bucket.get("active"):
        return None
    return bucket["records"].get(bucket["active"])


def _fetch_new_snapshot(symbols_key, years, spec_id, spec_payload):
    p = StooqDailyProvider()
    data = p.fetch_many(symbols_key, years=int(years))
    sources = dict(p.source_by_symbol)
    manifest = dataset_manifest(
        data, sources,
        spec_id=spec_id,
        spec_payload=spec_payload,
        engine_fingerprint=code_fingerprint(),
    )
    return _register_snapshot(spec_id, data, sources, manifest, p.last_errors)


def _compute_locked_research(snapshot, spec_id, spec_payload, holdout_bars, burn_in_bars, fold_bars, *, force=False):
    dataset_id = snapshot["manifest"]["dataset_id"]
    engine_fp = code_fingerprint()
    run_key = f"{spec_id}:{dataset_id}:{engine_fp}"
    runs = _run_store()
    previous = runs.get(run_key)

    # Statistical plan registration happens BEFORE any fresh discovery computation.
    # If an older v0.6.x bundle already contains a result but no v0.7 ledger, that
    # evidence is imported as legacy: it contributes to cumulative penalties but
    # cannot by itself unlock a candidate.
    ledger = _current_ledger()
    matching_existing_batch = any(
        str(b.get("spec_id")) == str(spec_id) and str(b.get("dataset_id")) == str(dataset_id)
        for b in ledger.get("batches", [])
    )
    legacy_import = bool(snapshot["manifest"].get("result_fingerprint")) and not matching_existing_batch
    ledger, batch, batch_is_new = preregister_batch(
        ledger,
        spec_id=spec_id,
        dataset_id=dataset_id,
        symbols=sorted(snapshot["data"].keys()),
        strategies=STRATEGY_NAMES,
        cfg=cfg,
        scientific_protocol="0.6",
        engine_fingerprint=engine_fp,
        legacy=legacy_import,
    )
    _set_ledger(ledger)

    recomputed = False
    if previous is not None and not force:
        base_matrix = previous["base_matrix"].copy()
        compute_seconds = 0.0
        reused_base = True
    else:
        t0 = time.perf_counter()
        base_matrix, _, _ = discovery_screen(
            snapshot["data"], cfg,
            holdout_bars=int(holdout_bars),
            burn_in_bars=int(burn_in_bars),
            fold_bars=int(fold_bars),
        )
        compute_seconds = time.perf_counter() - t0
        reused_base = False
        recomputed = True

    base_result_fp = matrix_result_fingerprint(base_matrix)
    expected_fp = snapshot["manifest"].get("result_fingerprint")
    expected_match = None if not expected_fp else (str(expected_fp) == base_result_fp)

    # Finalization is idempotent. Same spec+dataset+hypotheses cannot silently
    # become a second look; a changed ledger/trade result raises a conflict.
    ledger = finalize_batch(
        _current_ledger(),
        batch["batch_id"],
        base_matrix,
        base_result_fingerprint=base_result_fp,
    )
    _set_ledger(ledger)

    gated_matrix = apply_cumulative_gate(
        base_matrix,
        ledger,
        cfg=cfg,
        scientific_protocol="0.6",
        batch_id=batch["batch_id"],
    )
    selections, selection_id = ledger_selection_id(
        gated_matrix,
        spec_id=spec_id,
        dataset_id=dataset_id,
        ledger=ledger,
    )
    ledger_result_fp = matrix_with_ledger_fingerprint(gated_matrix)
    ledger_fp = ledger_fingerprint(ledger)

    current = {
        "base_matrix": base_matrix,
        "matrix": gated_matrix,
        "selections": selections,
        "selection_id": selection_id,
        "base_result_fingerprint": base_result_fp,
        "result_fingerprint": ledger_result_fp,
        "ledger_result_fingerprint": ledger_result_fp,
        "ledger_fingerprint": ledger_fp,
        "batch_id": batch["batch_id"],
        "batch_preregistered": bool(batch.get("preregistered", False)),
        "batch_legacy": bool(batch.get("legacy", False)),
        "batch_is_new": bool(batch_is_new),
        "compute_seconds": float(compute_seconds),
        "expected_result_fingerprint": expected_fp,
        "expected_match": expected_match,
    }

    verification = None
    if previous is not None and recomputed:
        verification = {
            "previous_base": previous["base_result_fingerprint"],
            "current_base": base_result_fp,
            "base_match": previous["base_result_fingerprint"] == base_result_fp,
            "previous_ledger": previous["ledger_result_fingerprint"],
            "current_ledger": ledger_result_fp,
            "ledger_match": previous["ledger_result_fingerprint"] == ledger_result_fp,
        }

    runs[run_key] = current

    # Backward-compatible reproducibility fingerprint for exact OHLCV computation.
    # The cumulative-ledger fingerprint is separate because it depends on research history.
    if not expected_fp:
        snapshot["manifest"]["result_fingerprint"] = base_result_fp
        snapshot["manifest"]["engine_fingerprint"] = engine_fp
    snapshot["manifest"]["ledger_fingerprint"] = ledger_fp
    snapshot["manifest"]["ledger_result_fingerprint"] = ledger_result_fp

    return current, reused_base, verification



def show_load_errors():
    if provider.last_errors:
        with st.expander(f"Data load warnings ({len(provider.last_errors)})"):
            for e in provider.last_errors:
                st.write("-", e)


def save_result(kind, label, df, summary):
    rid = registry.save(
        kind=kind,
        label=label,
        cfg=cfg,
        result=summary,
        data_start=str(df.index[0]) if len(df) else "",
        data_end=str(df.index[-1]) if len(df) else "",
        notes="Saved from Alpha v0.7.1 UI",
    )
    st.success(f"Experiment saved with ID {rid}.")


if mode == "Audit overview":
    st.subheader("What v0.7 adds")
    st.markdown(
        """
**Cumulative research hygiene**
- The scientific **v0.6 trading rules are unchanged**; v0.7 changes how repeated research is accounted for.
- Every asset/strategy pair has a deterministic **Hypothesis ID** tied to the strategy implementation and risk configuration.
- Exact reruns on the same Research Spec + Dataset are idempotent and **do not count as new evidence looks**.
- A genuinely new dataset/specification becomes another look at the same hypothesis.
- Repeated looks use a conservative **anytime alpha-spending p-value**; cumulative cross-hypothesis FDR uses **Benjamini-Yekutieli**, which is stricter than the batch BH screen.
- Pre-v0.7 results can be imported as **legacy evidence**: they increase the cumulative penalty but cannot directly unlock a candidate.
- Downloadable reproducibility bundles now carry both the exact OHLCV snapshot and the cumulative research ledger.

**Selection discipline**
- Screens the ETF universe across **six fixed strategy families** under the unchanged v0.6 absolute thresholds.
- The reserved historical segment is physically removed before discovery scoring.
- A pair must pass both the current-batch **BH q≤0.10** and the cumulative-ledger **BY q≤0.10**.
- Only a preregistered, cumulative-gate **CANDIDATE** may be frozen; **WATCH** never unlocks the historical holdout.
- The historical holdout is a secondary robustness check; **future forward-paper data remains the true untouched validation**.

**Portfolio discipline**
- A selected strategy can be frozen per asset for a secondary historical robustness test.
- Heat sizing includes entry fees, and a heat breach caused by mark-to-market triggers next-open de-risking.
- The original walk-forward, cost-stress, correlation, leverage and drawdown controls remain active.

**Execution realism**
- Signal at close **t**, entry only at open **t+1**.
- Same-bar stop/target after entry is now modeled.
- Overnight stop gaps fill at the worse opening price; profit gaps do not get optimistic price improvement.
- Entry is skipped when the next open is too far from the original signal in risk units.

**Bias control**
- Portfolio event order is strict: **open gap → allocation → intraday high/low → close mark**.
- No same-day close is used to decide an open entry.
- Walk-forward adds an embargo between training and unseen test windows.
- Cross-asset leaderboard is exploratory only; it is not the final validation gate.

**Risk**
- Drawdown kill switch is latched and existing exposure is flattened at the next executable open.
- Portfolio heat limits total stop-risk across all open positions.
- Pair-correlation veto reduces duplicate bets.
- Gross exposure, position size, ADV participation and no-leverage caps are enforced together.

**Evidence**
- Sharpe annualization adapts to the actual sampling frequency instead of assuming 252 bars/year.
- Adds Sortino, Calmar, R-multiples, exposure, turnover, average holding period and bootstrap uncertainty.
- Cost stress reruns walk-forward at 1×, 2× and 3× assumed fees/slippage.
- Every saved experiment records a code fingerprint so later tuning cannot be confused with the original test.
        """
    )
    st.info(
        "Still intentionally missing: live execution, leverage, options, shorts, LLM-directed trade decisions, "
        "automatic parameter optimization, and autonomous risk-limit changes. Those are excluded until forward paper evidence exists."
    )

elif mode == "Integrity audit":
    st.subheader("Automated integrity checks")
    st.write(
        "These tests audit mechanics rather than profitability: future-data isolation, conservative execution, "
        "candidate-only freezing, batch and cumulative multiple-testing math, ledger idempotence, holdout isolation, "
        "dataset hashing, snapshot round-trips, and result fingerprints."
    )
    if st.button("Run integrity audit", type="primary"):
        with st.spinner("Running deterministic synthetic-data checks..."):
            audit = run_integrity_checks()
        st.dataframe(audit, use_container_width=True)
        if bool(audit["passed"].all()):
            st.success(f"All {len(audit)} integrity checks passed.")
        else:
            failed = audit[~audit["passed"]]
            st.error(f"{len(failed)} integrity check(s) failed. Do not trust research output until fixed.")


elif mode == "Walk-forward lab":
    st.subheader("Rolling unseen test")
    symbol = st.text_input("Market symbol", "SPY.US").strip().upper()
    years = st.slider("History", 4, 15, 10)
    c1, c2, c3 = st.columns(3)
    train_bars = c1.selectbox("Training window", [252, 504, 756], index=1)
    test_bars = c2.selectbox("Unseen test window", [63, 126, 252], index=1)
    embargo = c3.selectbox("Embargo bars", [0, 1, 3, 5], index=1)
    if st.button("Run hardened walk-forward", type="primary"):
        end = date.today(); start = end - timedelta(days=int(365.25 * years))
        try:
            with st.spinner("Testing chronological unseen windows..."):
                df = provider.fetch(DataRequest(symbol, start, end))
                q = validate_ohlcv(df)
                source_used = provider.last_source
                result = run_walk_forward(symbol, df, cfg, train_bars=train_bars, test_bars=test_bars, embargo_bars=embargo)
            s = result.summary
            if source_used:
                st.caption(f"Market data source: {source_used}")
            c = st.columns(4)
            c[0].metric("OOS return", f"{s['oos_compounded_return_pct']:.2f}%")
            c[1].metric("OOS benchmark", f"{s['oos_benchmark_return_pct']:.2f}%")
            c[2].metric("OOS alpha", f"{s['oos_alpha_vs_benchmark_pct']:+.2f}%")
            c[3].metric("Research gate", s["research_gate"])
            c = st.columns(4)
            c[0].metric("OOS trades", s["total_oos_trades"])
            c[1].metric("Positive-alpha folds", f"{s['positive_alpha_folds_pct']:.0f}%")
            c[2].metric("Mean R", f"{s['oos_mean_r']:.3f}")
            c[3].metric("P(mean R > 0)", f"{s['bootstrap_prob_mean_r_positive']:.1%}")
            st.caption(
                f"Bootstrap 95% CI mean R: [{s['bootstrap_mean_r_ci_low']:.3f}, {s['bootstrap_mean_r_ci_high']:.3f}] · "
                f"Worst OOS DD: {s['worst_oos_drawdown_pct']:.2f}% · Data rows: {q.rows}"
            )
            if s.get("mc_paths", 0) > 0:
                st.caption(
                    f"100-trade bootstrap stress: median return {s['mc_100_trade_median_return_pct']:.1f}% · "
                    f"5th-percentile return {s['mc_100_trade_p05_return_pct']:.1f}% · "
                    f"95th-percentile max DD {s['mc_100_trade_p95_drawdown_pct']:.1f}% · "
                    f"P(finish negative) {s['mc_prob_finish_negative']:.1%}. This is a sequence-risk stress test, not a forecast."
                )
            else:
                st.caption("Monte Carlo sequence-risk stress withheld: fewer than 20 OOS trades.")
            st.dataframe(pd.DataFrame([f.to_dict() for f in result.folds]), use_container_width=True)
            if s["research_gate"] == "PASS_RESEARCH_GATE":
                st.success("Passed the hardened historical research gate. Forward paper testing is still mandatory.")
            elif s["research_gate"] == "INSUFFICIENT_DATA":
                st.warning("Insufficient evidence. This is deliberately stricter than v0.3.")
            else:
                st.error("Failed the research gate.")
            if st.button("Save this experiment"):
                save_result("walk_forward", symbol, df, s)
        except Exception as exc:
            st.error(str(exc))

elif mode == "Cost stress":
    st.subheader("Does the edge survive worse execution?")
    symbol = st.text_input("Market symbol", "SPY.US", key="stress_symbol").strip().upper()
    years = st.slider("History", 4, 15, 10, key="stress_years")
    c1, c2 = st.columns(2)
    train_bars = c1.selectbox("Training bars", [252, 504, 756], index=1, key="stress_train")
    test_bars = c2.selectbox("Test bars", [63, 126, 252], index=1, key="stress_test")
    if st.button("Run 1× / 2× / 3× cost stress", type="primary"):
        end = date.today(); start = end - timedelta(days=int(365.25 * years))
        try:
            with st.spinner("Rerunning walk-forward under stressed transaction costs..."):
                df = provider.fetch(DataRequest(symbol, start, end))
                stress = cost_stress_walk_forward(symbol, df, cfg, train_bars=train_bars, test_bars=test_bars, embargo_bars=1)
            st.dataframe(stress, use_container_width=True)
            if stress.attrs.get("cost_resilient", False):
                st.success("Historical alpha remains positive at 2× assumed execution costs. That is a useful robustness sign, not proof.")
            else:
                st.error("The apparent edge does not survive the 2× cost test.")
        except Exception as exc:
            st.error(str(exc))

elif mode == "Multi-asset portfolio":
    st.subheader("Shared-capital portfolio simulation")
    raw = st.text_area("Market symbols", DEFAULT, height=90)
    years = st.slider("Portfolio history", 3, 12, 7)
    if st.button("Run portfolio test", type="primary"):
        symbols = [x.strip().upper() for x in raw.split(",") if x.strip()]
        try:
            with st.spinner("Loading assets and enforcing shared risk controls..."):
                data = load_many(symbols, years)
                result = run_portfolio_backtest(data, cfg)
            show_load_errors()
            m = result.metrics
            c = st.columns(4)
            c[0].metric("Portfolio return", f"{m['total_return_pct']:.2f}%")
            c[1].metric("Equal-weight B&H", f"{m['benchmark_return_pct']:.2f}%")
            c[2].metric("Alpha", f"{m['alpha_vs_benchmark_pct']:+.2f}%")
            c[3].metric("Max drawdown", f"{m['max_drawdown_pct']:.2f}%")
            c = st.columns(4)
            c[0].metric("Trades", m["trades"])
            c[1].metric("Profit factor", "∞" if m["profit_factor"] == float("inf") else f"{m['profit_factor']:.2f}")
            c[2].metric("Sharpe / Sortino", f"{m['sharpe']:.2f} / {m['sortino']:.2f}")
            c[3].metric("Mean R", f"{m['mean_r']:.3f}")
            c = st.columns(4)
            c[0].metric("Max heat observed", f"{m['max_portfolio_heat_pct_observed']:.2f}%")
            c[1].metric("Max gross exposure", f"{m['max_gross_exposure_pct_observed']:.1f}%")
            c[2].metric("P(mean R > 0)", f"{m['bootstrap_prob_mean_r_positive']:.1%}")
            c[3].metric("Heat de-risk events", m.get("heat_rebalance_count", 0))
            st.line_chart(result.equity_curve[["equity"]].join(result.benchmark_curve, how="left"), height=350)
            trades = pd.DataFrame([t.to_dict() for t in result.trades])
            if len(trades):
                st.dataframe(trades, use_container_width=True)
            if m["trades"] < cfg.min_oos_trades:
                st.warning("Trade sample is below the hardened evidence threshold.")
        except Exception as exc:
            st.error(str(exc))


elif mode == "Research universe":
    st.subheader("Broad discovery → cumulative ledger → candidate freeze → secondary historical holdout")
    st.write(
        "Alpha v0.7 keeps the v0.6 trading rules unchanged, but it no longer judges a run in isolation. "
        "Every hypothesis look is written to a cumulative ledger. Exact reruns do not add another look; new data/specs do."
    )
    st.caption(
        "Batch gate: ≥30 trades · PF≥1.20 · Mean R≥0.10 · P(Mean R>0)≥90% · positive in ≥60% of folds · BH q≤0.10. "
        "Final historical candidate also requires cumulative anytime/BY q≤0.10 and preregistered evidence."
    )
    st.info(
        "Pre-v0.7 bundles are accepted, but their existing result is imported as legacy evidence: it contributes to the penalty and cannot by itself unlock a candidate. "
        "Keep the new v0.7 bundle because it contains both the exact dataset and the cumulative research ledger."
    )

    raw = st.text_area("Market symbols", DEFAULT, height=90, key="research_symbols")
    years = st.slider("Total history", 8, 15, 10, key="research_years")
    c1, c2, c3 = st.columns(3)
    holdout_bars = c1.selectbox("Secondary historical holdout", [252, 504, 756], index=1, help="252≈1 year, 504≈2 years")
    burn_in_bars = c2.selectbox("Discovery burn-in", [252, 504, 756], index=1)
    fold_bars = c3.selectbox("Discovery fold", [63, 126, 252], index=1)

    symbols = [x.strip().upper() for x in raw.split(",") if x.strip()]
    symbols_key = _normalize_symbols(symbols)
    spec_payload = research_spec_payload(symbols_key, years, holdout_bars, burn_in_bars, fold_bars)
    spec_id = research_spec_id(symbols_key, years, holdout_bars, burn_in_bars, fold_bars)
    st.caption(f"Current Research Spec ID: `{spec_id}` · active Ledger ID: `{ledger_fingerprint(_current_ledger())}`")

    uploaded_snapshot = st.file_uploader(
        "Restore an exact Alpha research bundle ZIP (optional)",
        type=["zip"],
        key=f"snapshot_upload_{spec_id}",
        help="v0.7 bundles restore both the exact Dataset ID and cumulative ledger. Older v0.6.2 bundles restore the dataset and are imported as legacy evidence.",
    )
    if uploaded_snapshot is not None and st.button("Restore uploaded research bundle", key=f"restore_{spec_id}"):
        try:
            raw_bundle = uploaded_snapshot.getvalue()
            restored_data, restored_manifest = load_snapshot_zip(raw_bundle)
            snap_spec = restored_manifest.get("spec_id")
            snap_payload = restored_manifest.get("spec_payload") or {}
            if snap_spec and snap_spec != spec_id:
                raise ValueError(f"Snapshot belongs to Research Spec {snap_spec}, not current spec {spec_id}.")
            if snap_payload and snap_payload != spec_payload:
                raise ValueError("Snapshot research parameters do not match the current controls.")
            restored_sources = {a["symbol"]: a.get("source", "snapshot") for a in restored_manifest.get("assets", [])}
            _register_snapshot(spec_id, restored_data, restored_sources, restored_manifest, ())

            embedded_ledger = load_snapshot_ledger(raw_bundle)
            if embedded_ledger is not None:
                merged = merge_ledgers(_current_ledger(), embedded_ledger)
                _set_ledger(merged)
                st.success(
                    f"Restored Dataset ID {restored_manifest['dataset_id']} + cumulative Ledger ID {ledger_fingerprint(merged)}."
                )
            else:
                st.success(
                    f"Restored pre-v0.7 Dataset ID {restored_manifest['dataset_id']}. "
                    "Its existing result will be imported as legacy evidence on the next run."
                )
        except Exception as exc:
            st.error(f"Research bundle restore failed: {exc}")

    bucket = _snapshot_store().get(spec_id)
    if bucket and bucket.get("records"):
        dataset_options = list(bucket["records"].keys())
        active_id = bucket.get("active") if bucket.get("active") in dataset_options else dataset_options[-1]
        selected_id = st.selectbox(
            "Locked dataset in this session",
            dataset_options,
            index=dataset_options.index(active_id),
            key=f"dataset_select_{spec_id}",
            help="Switching here does not download data; it reactivates an exact snapshot already locked in this session.",
        )
        bucket["active"] = selected_id

    b1, b2 = st.columns(2)
    run_clicked = b1.button("1 · Run / reuse locked dataset", type="primary", use_container_width=True)
    refresh_clicked = b2.button("Refresh market data → new dataset", type="secondary", use_container_width=True)

    if run_clicked or refresh_clicked:
        try:
            old_snapshot = _active_snapshot(spec_id)
            if refresh_clicked or old_snapshot is None:
                with st.spinner("Fetching market data and creating an exact OHLCV snapshot..."):
                    new_snapshot = _fetch_new_snapshot(symbols_key, years, spec_id, spec_payload)
                if old_snapshot is not None:
                    drift = compare_manifests(old_snapshot["manifest"], new_snapshot["manifest"])
                    st.session_state["alpha_v07_drift"] = {
                        "spec_id": spec_id,
                        "old_dataset_id": old_snapshot["manifest"]["dataset_id"],
                        "new_dataset_id": new_snapshot["manifest"]["dataset_id"],
                        "table": drift,
                    }
                snapshot = new_snapshot
            else:
                snapshot = old_snapshot

            provider.last_errors = list(snapshot.get("errors", ()))
            provider.source_by_symbol = dict(snapshot.get("sources", {}))
            show_load_errors()

            request_t0 = time.perf_counter()
            with st.spinner("Running discovery against the locked dataset..."):
                result, reused, verification = _compute_locked_research(
                    snapshot, spec_id, spec_payload,
                    holdout_bars, burn_in_bars, fold_bars,
                    force=False,
                )
            request_seconds = time.perf_counter() - request_t0

            seen = st.session_state.setdefault("alpha_v07_specs_seen", [])
            if spec_id not in seen:
                seen.append(spec_id)

            st.session_state["alpha_v07_research"] = {
                "data": snapshot["data"],
                "sources": snapshot["sources"],
                "manifest": snapshot["manifest"],
                "matrix": result["matrix"],
                "base_matrix": result["base_matrix"],
                "selections": result["selections"],
                "selection_id": result["selection_id"],
                "base_result_fingerprint": result["base_result_fingerprint"],
                "result_fingerprint": result["result_fingerprint"],
                "ledger_result_fingerprint": result["ledger_result_fingerprint"],
                "ledger_fingerprint": result["ledger_fingerprint"],
                "batch_id": result["batch_id"],
                "batch_preregistered": result["batch_preregistered"],
                "batch_legacy": result["batch_legacy"],
                "spec_id": spec_id,
                "spec_payload": spec_payload,
                "holdout_bars": int(holdout_bars),
                "years": int(years),
                "compute_seconds": float(result["compute_seconds"]),
                "request_seconds": float(request_seconds),
                "reused": bool(reused),
                "iteration_count": len(seen),
                "expected_result_fingerprint": result.get("expected_result_fingerprint"),
                "expected_match": result.get("expected_match"),
                "verification": verification,
            }
            st.session_state.pop("alpha_v07_holdout", None)
        except Exception as exc:
            st.error(str(exc))

    drift_state = st.session_state.get("alpha_v07_drift")
    if drift_state and drift_state.get("spec_id") == spec_id:
        changed = drift_state["table"]
        changed_assets = changed[changed["changed"]]
        if drift_state["old_dataset_id"] == drift_state["new_dataset_id"]:
            st.success(f"Refresh returned the identical Dataset ID `{drift_state['new_dataset_id']}`; no OHLCV drift detected.")
        else:
            st.warning(
                f"Market-data drift detected: Dataset `{drift_state['old_dataset_id']}` → `{drift_state['new_dataset_id']}`. "
                f"Changed assets: {len(changed_assets)}. This is a NEW dataset, not a rerun of the old experiment."
            )
            if len(changed_assets):
                st.dataframe(changed_assets, use_container_width=True)

    state = st.session_state.get("alpha_v07_research")
    if state and state.get("spec_id") == spec_id:
        matrix = state["matrix"].copy()
        order = {"CANDIDATE": 0, "WATCH": 1, "INSUFFICIENT": 2, "REJECT": 3}
        matrix["_order"] = matrix["status"].map(order).fillna(9)
        matrix = matrix.sort_values(
            ["_order", "ledger_by_q", "bh_q_value", "prob_mean_r_positive", "mean_r"],
            ascending=[True, True, True, False, False],
        ).drop(columns=["_order"])

        dataset_id = state["manifest"]["dataset_id"]
        ledger_now = _current_ledger()
        ledger_table = cumulative_table(ledger_now)
        total_looks = len(ledger_now.get("events", []))
        legacy_looks = int(sum(bool(e.get("legacy", False)) for e in ledger_now.get("events", [])))
        prereg_looks = int(sum(bool(e.get("preregistered", False)) for e in ledger_now.get("events", [])))

        st.caption(
            f"Research Spec `{state['spec_id']}` · Dataset `{dataset_id}` · base result `{state['base_result_fingerprint']}` · "
            f"ledger result `{state['ledger_result_fingerprint']}`"
        )
        st.caption(
            f"Ledger `{state['ledger_fingerprint']}` · batch `{state['batch_id']}` · candidate-freeze `{state['selection_id']}` · "
            f"unique hypotheses {len(ledger_table)} · evidence looks {total_looks} "
            f"(legacy {legacy_looks}, preregistered {prereg_looks})"
        )

        if state.get("batch_legacy"):
            st.warning(
                "This exact batch was imported as PRE-v0.7 legacy evidence. It contributes to cumulative penalties but cannot directly unlock a historical CANDIDATE."
            )
        elif state.get("batch_preregistered"):
            st.success("This batch was preregistered in the cumulative ledger before its result was computed.")
        if state.get("reused"):
            st.success("Exact locked dataset + base engine result reused; no market data was downloaded and no new evidence look was added.")
        st.caption(
            f"Discovery compute time: {state.get('compute_seconds', 0.0):.1f}s · this request: {state.get('request_seconds', 0.0):.1f}s."
        )

        expected_fp = state.get("expected_result_fingerprint")
        if expected_fp:
            if state.get("expected_match"):
                st.success(f"Exact OHLCV reproduction VERIFIED: legacy/base result `{expected_fp}` matches exactly.")
            else:
                st.error(
                    f"REPRODUCIBILITY FAILURE: restored bundle expected base result `{expected_fp}`, "
                    f"but current base result is `{state['base_result_fingerprint']}`. "
                    "Do not trust this run until the code/data difference is explained."
                )

        verify_clicked = st.button("Verify exact rerun on this locked dataset + ledger", key=f"verify_{state['spec_id']}_{dataset_id}")
        if verify_clicked:
            try:
                snapshot = _active_snapshot(spec_id)
                with st.spinner("Recomputing from the same OHLCV bytes and verifying both base and cumulative-ledger fingerprints..."):
                    result2, _, verification = _compute_locked_research(
                        snapshot, spec_id, spec_payload,
                        holdout_bars, burn_in_bars, fold_bars,
                        force=True,
                    )
                if verification and verification.get("base_match"):
                    if verification.get("ledger_match"):
                        st.success(
                            f"Exact rerun VERIFIED: base `{verification['current_base']}` and cumulative ledger context are unchanged."
                        )
                    else:
                        st.warning(
                            f"Base rerun VERIFIED (`{verification['current_base']}`), but the cumulative ledger context changed since the prior view. "
                            "That is expected after other hypotheses/batches are added; the batch evidence itself remained exact."
                        )
                else:
                    st.error(f"REPRODUCIBILITY FAILURE: {verification}")
            except Exception as exc:
                st.error(str(exc))

        st.dataframe(
            matrix[[
                "symbol", "strategy", "status", "why", "folds", "trades",
                "return_pct", "alpha_pct", "profit_factor", "mean_r",
                "prob_mean_r_positive", "mean_r_p_value", "bh_q_value",
                "ledger_looks", "ledger_legacy_looks", "ledger_preregistered_looks",
                "ledger_anytime_p", "ledger_by_q", "ledger_gate",
                "positive_return_folds_pct", "max_drawdown_pct"
            ]],
            use_container_width=True,
        )
        st.caption(
            "BH q is the current-batch screen. ledger_by_q is the stricter cumulative Benjamini-Yekutieli q-value after anytime alpha-spending across repeated looks. "
            "alpha_pct is context versus full-exposure buy-and-hold and is not itself a candidate requirement."
        )

        with st.expander("Dataset manifest · per-asset SHA-256"):
            manifest_df = pd.DataFrame(state["manifest"].get("assets", []))
            if len(manifest_df):
                st.dataframe(manifest_df[["symbol", "rows", "start", "end", "source", "sha256"]], use_container_width=True)

        # v0.7.1 mobile-safe export flow. Preparing exports is an explicit action;
        # the resulting bytes are pinned in session_state and the actual download
        # buttons use on_click="ignore" so downloading does not rerun the app.
        export_token = f"{state['spec_id']}:{dataset_id}:{state['ledger_result_fingerprint']}:{ledger_fingerprint(ledger_now)}"
        prepared = st.session_state.get("alpha_v071_prepared_exports")
        if st.button(
            "Prepare mobile-safe exports",
            key=f"prepare_exports_{dataset_id}_{state['ledger_result_fingerprint']}",
            type="primary",
        ):
            bundle_bytes, _ = build_snapshot_zip(
                state["data"], state["sources"],
                spec_id=state["spec_id"],
                spec_payload=state["spec_payload"],
                result_fingerprint=state["base_result_fingerprint"],
                engine_fingerprint=code_fingerprint(),
                matrix=state["matrix"],
                ledger=ledger_now,
                ledger_result_fingerprint=state["ledger_result_fingerprint"],
            )
            st.session_state["alpha_v071_prepared_exports"] = {
                "token": export_token,
                "bundle": bundle_bytes,
                "bundle_name": f"alpha_v071_{state['spec_id']}_{dataset_id}_{state['ledger_result_fingerprint']}.zip",
                "ledger": ledger_json_bytes(ledger_now),
                "ledger_name": f"alpha_research_ledger_{ledger_fingerprint(ledger_now)}.json",
            }
            prepared = st.session_state["alpha_v071_prepared_exports"]

        if prepared and prepared.get("token") == export_token:
            st.success("Exports prepared and pinned in this session. Downloading them will not rerun Alpha.")
            st.download_button(
                "Download exact v0.7.1 research bundle (.zip)",
                data=prepared["bundle"],
                file_name=prepared["bundle_name"],
                mime="application/zip",
                key=f"download_bundle_v071_{dataset_id}_{state['ledger_result_fingerprint']}",
                on_click="ignore",
            )
            st.download_button(
                "Download cumulative research ledger (.json)",
                data=prepared["ledger"],
                file_name=prepared["ledger_name"],
                mime="application/json",
                key=f"download_ledger_v071_{ledger_fingerprint(ledger_now)}",
                on_click="ignore",
            )
            st.caption(
                "Keep the ZIP. It contains the exact OHLCV snapshot, pair matrix, and cumulative ledger needed to reproduce this historical research state. "
                "The download buttons are configured not to rerun the Streamlit script."
            )
        else:
            st.info("Tap Prepare mobile-safe exports first. This freezes the ZIP and ledger bytes before you download them.")

        fam = family_summary(matrix)
        if len(fam):
            st.subheader("Strategy-family breadth")
            st.dataframe(fam, use_container_width=True)
            st.caption("This table is descriptive. A family with one spectacular pair but weak breadth is not automatically promoted.")

        selections = state["selections"]
        if selections:
            selected_rows = []
            for sym, strat in selections.items():
                r = matrix[(matrix["symbol"] == sym) & (matrix["strategy"] == strat)].iloc[0]
                selected_rows.append({
                    "symbol": sym, "frozen_strategy": strat, "discovery_status": r["status"],
                    "discovery_trades": int(r["trades"]), "discovery_mean_r": r["mean_r"],
                    "P(mean R > 0)": r["prob_mean_r_positive"],
                    "discovery_profit_factor": r["profit_factor"],
                    "trade_ledger_hash": r.get("trade_ledger_hash", ""),
                })
            st.subheader("Frozen selections")
            st.dataframe(pd.DataFrame(selected_rows), use_container_width=True)
            st.warning(
                "Do not change strategies after seeing the secondary historical holdout. If it fails, the hypothesis fails. "
                "Changing the rules after seeing it would contaminate the test."
            )
            confirm = st.checkbox(
                "I understand that revealing the secondary historical holdout consumes it for this exact frozen selection + Dataset ID.",
                key=f"consume_{state['selection_id']}_{dataset_id}"
            )
            if confirm and st.button("2 · Consume secondary historical holdout", type="secondary"):
                try:
                    with st.spinner("Evaluating frozen selections on the reserved historical segment of the locked dataset..."):
                        pair_df, port, gate = evaluate_frozen_holdout(
                            state["data"], selections, cfg,
                            holdout_bars=state["holdout_bars"],
                        )
                    st.session_state["alpha_v07_holdout"] = {
                        "selection_id": state["selection_id"],
                        "dataset_id": dataset_id,
                        "pair_df": pair_df,
                        "port": port,
                        "gate": gate,
                    }
                except Exception as exc:
                    st.error(str(exc))
        else:
            st.info("No preregistered pair passed BOTH the current-batch and cumulative-ledger candidate gates. WATCH pairs remain visible, but the historical holdout stays locked.")

    hold = st.session_state.get("alpha_v07_holdout")
    if hold and state and hold["selection_id"] == state["selection_id"] and hold.get("dataset_id") == state["manifest"]["dataset_id"]:
        st.divider()
        st.subheader("Secondary historical holdout — consumed")
        st.caption(f"Consumed against Dataset ID `{hold['dataset_id']}`.")
        st.dataframe(hold["pair_df"], use_container_width=True)
        port = hold["port"]
        gate = hold["gate"]
        if port is not None:
            m = port.metrics
            c = st.columns(4)
            c[0].metric("Holdout portfolio return", f"{m['total_return_pct']:.2f}%")
            c[1].metric("Holdout benchmark", f"{m['benchmark_return_pct']:.2f}%")
            c[2].metric("Holdout alpha", f"{m['alpha_vs_benchmark_pct']:+.2f}%")
            c[3].metric("Historical holdout gate", gate)
            c = st.columns(4)
            c[0].metric("Trades", m["trades"])
            c[1].metric("PF", f"{m['profit_factor']:.2f}")
            c[2].metric("Mean R", f"{m['mean_r']:.3f}")
            c[3].metric("P(mean R > 0)", f"{m['bootstrap_prob_mean_r_positive']:.1%}")
            st.line_chart(port.equity_curve[["equity"]].join(port.benchmark_curve, how="left"), height=350)
        if gate == "HISTORICAL_HOLDOUT_PASS":
            st.success("The frozen historical hypothesis passed its secondary robustness check. It still requires forward paper evidence before any live-money consideration.")
        elif gate == "INCONCLUSIVE":
            st.warning("The reserved historical segment did not contain enough trades for a conclusion. Do not relax the rule after seeing it.")
        else:
            st.error("The frozen hypothesis failed the secondary historical robustness check. Do not retune it on this consumed segment.")


elif mode == "Research ledger":
    st.subheader("Cumulative research ledger")
    st.write(
        "This ledger tracks every registered asset/strategy hypothesis look. "
        "Exact reruns of the same spec+dataset do not increase the look count; new data/specifications do."
    )
    ledger_now = _current_ledger()

    uploaded_ledger = st.file_uploader(
        "Merge a cumulative ledger JSON",
        type=["json"],
        key="ledger_json_upload",
        help="Use this only to restore or merge a previously exported Alpha v0.7 ledger.",
    )
    if uploaded_ledger is not None and st.button("Merge uploaded ledger", key="merge_uploaded_ledger"):
        try:
            incoming = load_ledger_json(uploaded_ledger.getvalue())
            merged = merge_ledgers(ledger_now, incoming)
            _set_ledger(merged)
            ledger_now = merged
            st.success(f"Ledger merged. Active Ledger ID `{ledger_fingerprint(merged)}`.")
        except Exception as exc:
            st.error(f"Ledger merge failed: {exc}")

    table = cumulative_table(ledger_now)
    events = pd.DataFrame(ledger_now.get("events", []))
    batches = pd.DataFrame(ledger_now.get("batches", []))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Ledger ID", ledger_fingerprint(ledger_now))
    c2.metric("Unique hypotheses", int(len(table)))
    c3.metric("Evidence looks", int(len(events)))
    c4.metric("Batches", int(len(batches)))

    if len(table):
        st.subheader("Current cumulative evidence")
        st.dataframe(
            table.sort_values(["ledger_by_q", "anytime_p", "looks"]).reset_index(drop=True),
            use_container_width=True,
        )
        st.caption(
            "anytime_p spends alpha across repeated looks of the same hypothesis. "
            "ledger_by_q applies Benjamini-Yekutieli FDR across all unique hypotheses in the ledger."
        )
    else:
        st.info("The cumulative ledger is empty. Run Research universe or restore a v0.7 bundle.")

    if len(batches):
        with st.expander("Registered research batches"):
            display_cols = [
                c for c in [
                    "sequence", "batch_id", "spec_id", "dataset_id", "preregistered",
                    "legacy", "finalized", "event_count", "base_result_fingerprint"
                ] if c in batches.columns
            ]
            st.dataframe(batches[display_cols], use_container_width=True)

    if len(events):
        with st.expander("Evidence-event audit trail"):
            display_cols = [
                c for c in [
                    "sequence", "event_id", "hypothesis_id", "symbol", "strategy",
                    "spec_id", "dataset_id", "preregistered", "legacy", "raw_p",
                    "batch_bh_q", "trades", "mean_r", "profit_factor", "trade_ledger_hash"
                ] if c in events.columns
            ]
            st.dataframe(events[display_cols], use_container_width=True)

    st.download_button(
        "Download cumulative research ledger (.json)",
        data=ledger_json_bytes(ledger_now),
        file_name=f"alpha_research_ledger_{ledger_fingerprint(ledger_now)}.json",
        mime="application/json",
        key="ledger_mode_download",
    )
    st.warning(
        "The ledger controls historical research hygiene; it does not turn backtests into proof of future profitability. "
        "Forward paper evidence remains mandatory before any live-money consideration."
    )

elif mode == "Strategy leaderboard":
    st.subheader("Exploratory cross-asset holdout screen")
    st.warning("Diagnostic only. This screen uses recent historical data and must NOT be used to tune the v0.6 candidate rules. Use Research universe for disciplined selection.")
    raw = st.text_area("Market symbols", DEFAULT, height=90, key="lb_symbols")
    years = st.slider("History", 3, 12, 7, key="lb_years")
    if st.button("Build exploratory leaderboard", type="primary"):
        symbols = [x.strip().upper() for x in raw.split(",") if x.strip()]
        try:
            data = load_many(symbols, years)
            show_load_errors()
            summary, detail = strategy_leaderboard(data, cfg)
            st.dataframe(summary, use_container_width=True)
            with st.expander("Per-asset holdout detail"):
                st.dataframe(detail, use_container_width=True)
        except Exception as exc:
            st.error(str(exc))

elif mode == "Current scanner":
    st.subheader("Current signal scanner — no execution")
    raw = st.text_area("Market symbols", DEFAULT, height=90, key="scan_symbols")
    years = st.slider("History", 1, 5, 3, key="scan_years")
    if st.button("Scan", type="primary"):
        symbols = [x.strip().upper() for x in raw.split(",") if x.strip()]
        try:
            data = load_many(symbols, years)
            show_load_errors()
            ranking = rank_watchlist(data)
            if ranking.empty:
                st.info("No setup passed the current triggers.")
            else:
                st.dataframe(ranking, use_container_width=True)
                st.caption("quality_score is a heuristic ranking score, not a probability of profit.")
        except Exception as exc:
            st.error(str(exc))

elif mode == "CSV walk-forward":
    st.subheader("Any OHLC market from CSV")
    st.write("Useful for crypto/intraday research. Required: timestamp/date + open/high/low/close; volume optional.")
    uploaded = st.file_uploader("CSV", type=["csv"])
    symbol = st.text_input("Label", "BTC-USD")
    if uploaded is not None:
        try:
            df = normalize_uploaded_csv(uploaded)
            q = validate_ohlcv(df)
            st.json(q.to_dict())
            if not q.passed:
                st.error("Data quality failed. Fix the CSV before testing.")
                st.stop()
            max_train = max(150, min(3000, len(df)//2))
            train_bars = st.number_input("Training bars", min_value=100, max_value=max_train, value=min(500, max_train), step=25)
            test_bars = st.number_input("Test bars", min_value=20, max_value=max(20, len(df)//4), value=min(100, max(20, len(df)//4)), step=10)
            if st.button("Run CSV walk-forward", type="primary"):
                result = run_walk_forward(symbol, df, cfg, train_bars=int(train_bars), test_bars=int(test_bars), embargo_bars=1)
                st.json(result.summary)
                st.dataframe(pd.DataFrame([f.to_dict() for f in result.folds]), use_container_width=True)
        except Exception as exc:
            st.error(str(exc))

else:
    st.subheader("Frozen experiment registry")
    st.write("Each saved result records the exact Alpha code fingerprint and risk configuration used at that moment.")
    rows = registry.list()
    if rows.empty:
        st.info("No experiments saved yet.")
    else:
        st.dataframe(rows, use_container_width=True)
