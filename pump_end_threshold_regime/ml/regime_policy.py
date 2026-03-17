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
        df['bucket_p_bad'] = np.nan

        state = self.STATE_ACTIVE
        resume_streak = 0
        episode_id = 0

        buckets = df.groupby('open_time', sort=True)

        for bucket_time, bucket_idx in buckets.groups.items():
            bucket_p_bads = df.loc[bucket_idx, p_bad_col]
            bucket_p_bad_val = float(bucket_p_bads.max())
            df.loc[bucket_idx, 'bucket_p_bad'] = bucket_p_bad_val

            if state == self.STATE_ACTIVE:
                if bucket_p_bad_val >= self.pause_on_threshold:
                    state = self.STATE_PAUSED
                    episode_id += 1
                    resume_streak = 0
                    df.loc[bucket_idx, 'regime_state'] = self.STATE_PAUSED
                    df.loc[bucket_idx, 'blocked_by_policy'] = True
                    df.loc[bucket_idx, 'policy_episode_id'] = episode_id
                    df.loc[bucket_idx, 'pause_reason'] = f'bucket_p_bad={bucket_p_bad_val:.3f}'
                else:
                    df.loc[bucket_idx, 'regime_state'] = self.STATE_ACTIVE
                    df.loc[bucket_idx, 'blocked_by_policy'] = False

            elif state == self.STATE_PAUSED:
                df.loc[bucket_idx, 'regime_state'] = self.STATE_PAUSED
                df.loc[bucket_idx, 'blocked_by_policy'] = True
                df.loc[bucket_idx, 'policy_episode_id'] = episode_id

                if bucket_p_bad_val <= self.resume_threshold:
                    resume_streak += 1
                    if resume_streak >= self.resume_confirm_signals:
                        state = self.STATE_ACTIVE
                        reason = f'{resume_streak}_consecutive_low_risk_buckets'
                        df.loc[bucket_idx, 'regime_state'] = self.STATE_ACTIVE
                        df.loc[bucket_idx, 'blocked_by_policy'] = False
                        df.loc[bucket_idx, 'policy_episode_id'] = -1
                        df.loc[bucket_idx, 'resume_reason'] = reason
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
