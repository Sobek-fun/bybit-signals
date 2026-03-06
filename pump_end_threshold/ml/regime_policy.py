import pandas as pd


class RegimePolicy:
    STATE_ACTIVE = 'ACTIVE'
    STATE_PAUSED = 'PAUSED'

    def __init__(
            self,
            pause_on_threshold: float = 0.65,
            resume_threshold: float = 0.30,
            resume_confirm_signals: int = 2,
    ):
        self.pause_on_threshold = pause_on_threshold
        self.resume_threshold = resume_threshold
        self.resume_confirm_signals = resume_confirm_signals

    def apply(self, signals_df: pd.DataFrame, p_bad_col: str = 'p_bad') -> pd.DataFrame:
        df = signals_df.sort_values('open_time').copy()
        df['regime_state'] = self.STATE_ACTIVE
        df['blocked_by_policy'] = False

        state = self.STATE_ACTIVE
        resume_streak = 0

        for idx in df.index:
            p_bad = df.at[idx, p_bad_col]

            if state == self.STATE_ACTIVE:
                if p_bad >= self.pause_on_threshold:
                    state = self.STATE_PAUSED
                    resume_streak = 0
                    df.at[idx, 'regime_state'] = self.STATE_PAUSED
                    df.at[idx, 'blocked_by_policy'] = True
                else:
                    df.at[idx, 'regime_state'] = self.STATE_ACTIVE
                    df.at[idx, 'blocked_by_policy'] = False

            elif state == self.STATE_PAUSED:
                df.at[idx, 'regime_state'] = self.STATE_PAUSED
                df.at[idx, 'blocked_by_policy'] = True

                if p_bad <= self.resume_threshold:
                    resume_streak += 1
                    if resume_streak >= self.resume_confirm_signals:
                        state = self.STATE_ACTIVE
                        resume_streak = 0
                        df.at[idx, 'regime_state'] = self.STATE_ACTIVE
                        df.at[idx, 'blocked_by_policy'] = False
                else:
                    resume_streak = 0

        return df

    def simulate_pnl(self, signals_df: pd.DataFrame, p_bad_col: str = 'p_bad') -> dict:
        df = self.apply(signals_df, p_bad_col)

        all_outcomes = signals_df.sort_values('open_time')
        has_outcome = 'trade_outcome' in all_outcomes.columns
        has_pnl_pct = 'pnl_pct' in all_outcomes.columns

        if not has_outcome:
            return {
                'signals_before': len(df),
                'signals_after': int((~df['blocked_by_policy']).sum()),
                'blocked_share': float(df['blocked_by_policy'].mean()),
            }

        before_tp = int((all_outcomes['trade_outcome'] == 'TP').sum())
        before_sl = int((all_outcomes['trade_outcome'] == 'SL').sum())
        before_resolved = before_tp + before_sl

        if has_pnl_pct:
            pnl_before = all_outcomes['pnl_pct'].fillna(0).sum()
        else:
            pnl_map = {'TP': 4.5, 'SL': -10.0, 'TIMEOUT': 0.0}
            pnl_before = sum(pnl_map.get(o, 0) for o in all_outcomes['trade_outcome'])

        accepted = df[~df['blocked_by_policy']]
        after_tp = int((accepted['trade_outcome'] == 'TP').sum()) if has_outcome else 0
        after_sl = int((accepted['trade_outcome'] == 'SL').sum()) if has_outcome else 0

        if has_pnl_pct:
            pnl_after = accepted['pnl_pct'].fillna(0).sum()
        else:
            pnl_map = {'TP': 4.5, 'SL': -10.0, 'TIMEOUT': 0.0}
            pnl_after = sum(pnl_map.get(o, 0) for o in accepted['trade_outcome'])

        blocked = df[df['blocked_by_policy']]
        tp_blocked = int((blocked['trade_outcome'] == 'TP').sum()) if has_outcome else 0
        sl_blocked = int((blocked['trade_outcome'] == 'SL').sum()) if has_outcome else 0
        blocked_total = tp_blocked + sl_blocked

        def max_losing_streak(outcomes):
            streak = 0
            max_s = 0
            for o in outcomes:
                if o == 'SL':
                    streak += 1
                    max_s = max(max_s, streak)
                else:
                    streak = 0
            return max_s

        max_streak_before = max_losing_streak(all_outcomes['trade_outcome'].tolist())
        max_streak_after = max_losing_streak(accepted['trade_outcome'].tolist()) if len(accepted) > 0 else 0

        blocked_bad_precision = sl_blocked / max(1, blocked_total)
        tp_block_share_total = tp_blocked / max(1, before_tp)
        sl_block_share_total = sl_blocked / max(1, before_sl)

        return {
            'signals_before': len(df),
            'signals_after': len(accepted),
            'blocked_share': float(df['blocked_by_policy'].mean()),
            'pnl_before': pnl_before,
            'pnl_after': pnl_after,
            'pnl_improvement': pnl_after - pnl_before,
            'tp_kept': after_tp,
            'sl_blocked': sl_blocked,
            'tp_blocked': tp_blocked,
            'blocked_bad_precision': blocked_bad_precision,
            'tp_block_share_total': tp_block_share_total,
            'sl_block_share_total': sl_block_share_total,
            'max_losing_streak_before': max_streak_before,
            'max_losing_streak_after': max_streak_after,
        }
