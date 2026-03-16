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
