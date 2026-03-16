import numpy as np
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
        df['policy_episode_id'] = -1
        df['pause_reason'] = ''
        df['resume_reason'] = ''
        df['bucket_p_bad'] = pd.to_numeric(df[p_bad_col], errors='coerce')

        state = self.STATE_ACTIVE
        resume_streak = 0
        episode_id = 0
        pending_pause_time = None

        for idx in df.index:
            current_time = df.at[idx, 'open_time']
            p_bad_val = pd.to_numeric(df.at[idx, p_bad_col], errors='coerce')

            if (
                    state == self.STATE_ACTIVE and
                    pending_pause_time is not None and
                    current_time > pending_pause_time
            ):
                state = self.STATE_PAUSED
                episode_id += 1
                resume_streak = 0
                pending_pause_time = None

            if state == self.STATE_ACTIVE:
                df.at[idx, 'regime_state'] = self.STATE_ACTIVE
                df.at[idx, 'blocked_by_policy'] = False
                df.at[idx, 'policy_episode_id'] = -1
                if pd.notna(p_bad_val) and p_bad_val >= self.pause_on_threshold:
                    pending_pause_time = current_time
                    df.at[idx, 'pause_reason'] = f'p_bad={p_bad_val:.3f}'
                continue

            df.at[idx, 'regime_state'] = self.STATE_PAUSED
            df.at[idx, 'blocked_by_policy'] = True
            df.at[idx, 'policy_episode_id'] = episode_id

            if pd.notna(p_bad_val) and p_bad_val <= self.resume_threshold:
                resume_streak += 1
                if resume_streak >= self.resume_confirm_signals:
                    state = self.STATE_ACTIVE
                    df.at[idx, 'regime_state'] = self.STATE_ACTIVE
                    df.at[idx, 'blocked_by_policy'] = False
                    df.at[idx, 'policy_episode_id'] = -1
                    df.at[idx, 'resume_reason'] = f'{resume_streak}_consecutive_low_risk_signals'
                    resume_streak = 0
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

        if has_pnl_pct:
            pnl_before = float(all_outcomes['pnl_pct'].fillna(0).sum())
        else:
            pnl_map = {'TP': 4.5, 'SL': -10.0, 'TIMEOUT': 0.0}
            pnl_before = sum(pnl_map.get(o, 0) for o in all_outcomes['trade_outcome'])

        accepted = df[~df['blocked_by_policy']]
        after_tp = int((accepted['trade_outcome'] == 'TP').sum())
        after_sl = int((accepted['trade_outcome'] == 'SL').sum())

        if has_pnl_pct:
            pnl_after = float(accepted['pnl_pct'].fillna(0).sum())
        else:
            pnl_map = {'TP': 4.5, 'SL': -10.0, 'TIMEOUT': 0.0}
            pnl_after = sum(pnl_map.get(o, 0) for o in accepted['trade_outcome'])

        blocked = df[df['blocked_by_policy']]
        tp_blocked = int((blocked['trade_outcome'] == 'TP').sum())
        sl_blocked = int((blocked['trade_outcome'] == 'SL').sum())
        blocked_total = tp_blocked + sl_blocked

        if has_pnl_pct:
            blocked_pnl_sum = float(blocked['pnl_pct'].fillna(0).sum())
        else:
            pnl_map = {'TP': 4.5, 'SL': -10.0, 'TIMEOUT': 0.0}
            blocked_pnl_sum = sum(pnl_map.get(o, 0) for o in blocked['trade_outcome'])

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

        episode_ids = df.loc[df['policy_episode_id'] > 0, 'policy_episode_id'].unique()
        pause_episodes_count = len(episode_ids)

        episode_durations = []
        for eid in episode_ids:
            ep_df = df[(df['policy_episode_id'] == eid) & df['blocked_by_policy']]
            if 'open_time' in ep_df.columns and len(ep_df) > 0:
                start = ep_df['open_time'].min()
                end = ep_df['open_time'].max()
                duration_hours = (end - start).total_seconds() / 3600
                episode_durations.append(duration_hours)

        avg_pause_duration_hours = float(np.mean(episode_durations)) if episode_durations else 0.0

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
            'blocked_pnl_sum': blocked_pnl_sum,
            'pause_episodes_count': pause_episodes_count,
            'avg_pause_duration_hours': avg_pause_duration_hours,
        }
