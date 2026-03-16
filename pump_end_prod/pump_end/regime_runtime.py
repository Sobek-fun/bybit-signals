from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from pump_end_prod.infra.logging import log
from pump_end_threshold.features.regime_feature_builder import RegimeFeatureBuilder
from pump_end_threshold.infra.clickhouse import DataLoader
from pump_end_threshold.ml.regime_dataset import build_strategy_state_live
from pump_end_threshold.ml.regime_inference import load_guard_artifacts
from pump_end_threshold.ml.regime_policy import RegimePolicy


class RegimeGuardRuntime:
    def __init__(self, ch_dsn: str, guard_model_dir: str):
        self.ch_dsn = ch_dsn
        self.guard_model_dir = Path(guard_model_dir)
        self.artifacts = load_guard_artifacts(self.guard_model_dir, ch_dsn)
        self.policy = RegimePolicy(**self.artifacts.policy_params)
        self.loader = DataLoader(ch_dsn)
        self.builder = RegimeFeatureBuilder(
            ch_dsn=ch_dsn,
            liquid_universe=self.artifacts.liquid_universe,
            top_n=self.artifacts.top_n_universe,
        )
        self.raw_signals = pd.DataFrame(columns=['symbol', 'open_time', 'signal_id'])

    def history_restore_window(self) -> timedelta:
        horizon = timedelta(minutes=(self.artifacts.max_horizon_bars + 1) * 15)
        return max(timedelta(hours=24), horizon) + timedelta(hours=1)

    def bootstrap_history(self, raw_payloads: list[dict]):
        if not raw_payloads:
            return
        rows = []
        for payload in raw_payloads:
            open_time = pd.to_datetime(payload['event_time'])
            signal_id = payload.get('signal_id') or f"{payload['symbol']}|{open_time.strftime('%Y%m%d_%H%M%S')}"
            rows.append({
                'symbol': payload['symbol'],
                'open_time': open_time,
                'signal_id': signal_id,
            })
        add_df = pd.DataFrame(rows)
        self.raw_signals = pd.concat([self.raw_signals, add_df], ignore_index=True)
        self.raw_signals = self.raw_signals.drop_duplicates(subset=['signal_id'], keep='last')
        self.raw_signals = self.raw_signals.sort_values('open_time').reset_index(drop=True)

    def process_bucket(self, raw_payloads: list[dict], bucket_time: datetime) -> tuple[list[dict], list[dict]]:
        if raw_payloads:
            self.bootstrap_history(raw_payloads)

        if not raw_payloads:
            return [], []

        if self.raw_signals.empty:
            return [], raw_payloads

        asof_time = pd.to_datetime(bucket_time)
        trades_df = build_strategy_state_live(
            self.raw_signals,
            self.loader,
            asof_time=asof_time,
            tp_pct=self.artifacts.tp_pct,
            sl_pct=self.artifacts.sl_pct,
            max_horizon_bars=self.artifacts.max_horizon_bars,
        )

        guard_features = self.builder.build_batch(self.raw_signals, batch_size=100, trades_df=trades_df)
        if guard_features.empty:
            return raw_payloads, []

        available = [c for c in self.artifacts.feature_columns if c in guard_features.columns]
        if len(available) < len(self.artifacts.feature_columns):
            missing = set(self.artifacts.feature_columns) - set(available)
            for col in missing:
                guard_features[col] = np.nan

        X_guard = guard_features[self.artifacts.feature_columns]
        p_bad = self.artifacts.model.predict_proba(X_guard)[:, 1]

        scored = self.raw_signals.copy()
        scored['p_bad'] = p_bad
        scored = self.policy.apply(scored, p_bad_col='p_bad')
        scored_bucket = scored[scored['open_time'] == asof_time]

        blocked_map = {}
        for _, row in scored_bucket.iterrows():
            blocked_map[row['signal_id']] = bool(row['blocked_by_policy'])

        accepted = []
        blocked = []
        for payload in raw_payloads:
            open_time = pd.to_datetime(payload['event_time'])
            signal_id = payload.get('signal_id') or f"{payload['symbol']}|{open_time.strftime('%Y%m%d_%H%M%S')}"
            if blocked_map.get(signal_id, False):
                blocked.append(payload)
            else:
                accepted.append(payload)

        log("INFO", "REGIME", f"bucket={asof_time.strftime('%Y-%m-%d %H:%M:%S')} raw={len(raw_payloads)} accepted={len(accepted)} blocked={len(blocked)}")
        return accepted, blocked
