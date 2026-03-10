import numpy as np
import pandas as pd


def compute_regime_scorecard(filtered_df: pd.DataFrame) -> dict:
    has_outcome = 'trade_outcome' in filtered_df.columns
    has_pnl = 'pnl_pct' in filtered_df.columns

    if not has_outcome or len(filtered_df) == 0:
        return {k: np.nan for k in [
            'pnl_improvement', 'blocked_bad_precision', 'sl_capture',
            'tp_tax', 'worst_window_improvement_12h',
        ]}

    all_tp = int((filtered_df['trade_outcome'] == 'TP').sum())
    all_sl = int((filtered_df['trade_outcome'] == 'SL').sum())

    blocked = filtered_df[filtered_df['blocked_by_policy']]
    accepted = filtered_df[~filtered_df['blocked_by_policy']]

    sl_blocked = int((blocked['trade_outcome'] == 'SL').sum())
    tp_blocked = int((blocked['trade_outcome'] == 'TP').sum())

    if has_pnl:
        pnl_before = float(filtered_df['pnl_pct'].fillna(0).sum())
        pnl_after = float(accepted['pnl_pct'].fillna(0).sum())
    else:
        pnl_map = {'TP': 4.5, 'SL': -10.0, 'TIMEOUT': 0.0}
        pnl_before = sum(pnl_map.get(o, 0) for o in filtered_df['trade_outcome'])
        pnl_after = sum(pnl_map.get(o, 0) for o in accepted['trade_outcome'])

    pnl_improvement = pnl_after - pnl_before
    blocked_bad_precision = sl_blocked / max(1, sl_blocked + tp_blocked)
    sl_capture = sl_blocked / max(1, all_sl)
    tp_tax = tp_blocked / max(1, all_tp)

    worst_12h_before = 0.0
    worst_12h_after = 0.0
    if 'open_time' in filtered_df.columns and has_pnl:
        sorted_df = filtered_df.sort_values('open_time').reset_index(drop=True)
        window = pd.Timedelta(hours=12)
        for i in range(len(sorted_df)):
            start_time = sorted_df.iloc[i]['open_time']
            end_time = start_time + window
            w = sorted_df[
                (sorted_df['open_time'] >= start_time) &
                (sorted_df['open_time'] < end_time)
            ]
            if len(w) > 0:
                w_pnl_before = float(w['pnl_pct'].fillna(0).sum())
                w_pnl_after = float(w.loc[~w['blocked_by_policy'], 'pnl_pct'].fillna(0).sum())
                worst_12h_before = min(worst_12h_before, w_pnl_before)
                worst_12h_after = min(worst_12h_after, w_pnl_after)

    return {
        'pnl_improvement': float(pnl_improvement),
        'blocked_bad_precision': float(blocked_bad_precision),
        'sl_capture': float(sl_capture),
        'tp_tax': float(tp_tax),
        'worst_window_improvement_12h': float(worst_12h_after - worst_12h_before),
    }


def _compute_episode_recall(filtered_df: pd.DataFrame) -> float:
    if 'trade_outcome' not in filtered_df.columns:
        return 0.0

    sorted_df = filtered_df.sort_values('open_time').reset_index(drop=True)

    sl_streaks = []
    current_streak_start = None
    current_streak_len = 0

    for i in range(len(sorted_df)):
        if sorted_df.iloc[i]['trade_outcome'] == 'SL':
            if current_streak_start is None:
                current_streak_start = i
            current_streak_len += 1
        else:
            if current_streak_len >= 3:
                sl_streaks.append({'start': current_streak_start, 'length': current_streak_len})
            current_streak_start = None
            current_streak_len = 0

    if current_streak_len >= 3:
        sl_streaks.append({'start': current_streak_start, 'length': current_streak_len})

    if not sl_streaks:
        return 0.0

    detected = 0
    for streak in sl_streaks:
        start_idx = streak['start']
        end_idx = min(start_idx + 2, start_idx + streak['length'])
        streak_slice = sorted_df.iloc[start_idx:end_idx]
        if streak_slice['blocked_by_policy'].any():
            detected += 1

    return detected / len(sl_streaks)


def _compute_worst_window(filtered_df: pd.DataFrame, hours: int) -> tuple:
    worst_before = 0.0
    worst_after = 0.0
    if 'open_time' not in filtered_df.columns or 'pnl_pct' not in filtered_df.columns:
        return worst_before, worst_after

    sorted_df = filtered_df.sort_values('open_time').reset_index(drop=True)
    window = pd.Timedelta(hours=hours)

    for i in range(len(sorted_df)):
        start_time = sorted_df.iloc[i]['open_time']
        end_time = start_time + window
        w = sorted_df[
            (sorted_df['open_time'] >= start_time) &
            (sorted_df['open_time'] < end_time)
        ]
        if len(w) > 0:
            w_pnl_before = float(w['pnl_pct'].fillna(0).sum())
            w_pnl_after = float(w.loc[~w['blocked_by_policy'], 'pnl_pct'].fillna(0).sum())
            worst_before = min(worst_before, w_pnl_before)
            worst_after = min(worst_after, w_pnl_after)

    return worst_before, worst_after


def compute_regime_diagnostics(
        filtered_df: pd.DataFrame,
        target_col: str = 'target_bad_next_5',
) -> dict:
    if len(filtered_df) == 0:
        return {
            'blocked_share': 0.0,
            'signal_keep_rate': 0.0,
            'episode_recall': 0.0,
            'brier_score': np.nan,
            'worst_window_6h_before': 0.0,
            'worst_window_6h_after': 0.0,
            'worst_window_12h_before': 0.0,
            'worst_window_12h_after': 0.0,
        }

    accepted = filtered_df[~filtered_df['blocked_by_policy']]

    blocked_share = float(filtered_df['blocked_by_policy'].mean())
    signal_keep_rate = len(accepted) / max(1, len(filtered_df))
    episode_recall = _compute_episode_recall(filtered_df)

    brier_score = np.nan
    if 'p_bad' in filtered_df.columns and target_col in filtered_df.columns:
        valid = filtered_df.dropna(subset=[target_col])
        if len(valid) > 0:
            brier_score = float(np.mean((valid['p_bad'] - valid[target_col]) ** 2))

    worst_6h_before, worst_6h_after = _compute_worst_window(filtered_df, 6)
    worst_12h_before, worst_12h_after = _compute_worst_window(filtered_df, 12)

    return {
        'blocked_share': blocked_share,
        'signal_keep_rate': signal_keep_rate,
        'episode_recall': episode_recall,
        'brier_score': brier_score,
        'worst_window_6h_before': worst_6h_before,
        'worst_window_6h_after': worst_6h_after,
        'worst_window_12h_before': worst_12h_before,
        'worst_window_12h_after': worst_12h_after,
    }


def _max_losing_streak(outcomes: list) -> int:
    streak = 0
    max_s = 0
    for o in outcomes:
        if o == 'SL':
            streak += 1
            max_s = max(max_s, streak)
        else:
            streak = 0
    return max_s


def evaluate_regime(
        filtered_df: pd.DataFrame,
        target_col: str = 'target_bad_next_5',
) -> dict:
    scorecard = compute_regime_scorecard(filtered_df)
    diagnostics = compute_regime_diagnostics(filtered_df, target_col)

    accepted = filtered_df[~filtered_df['blocked_by_policy']]
    blocked = filtered_df[filtered_df['blocked_by_policy']]
    has_pnl = 'pnl_pct' in filtered_df.columns
    has_outcome = 'trade_outcome' in filtered_df.columns

    context = {
        'signals_before': len(filtered_df),
        'signals_after': len(accepted),
    }

    if has_pnl:
        context['pnl_before'] = float(filtered_df['pnl_pct'].fillna(0).sum())
        context['pnl_after'] = float(accepted['pnl_pct'].fillna(0).sum())
        context['blocked_pnl_sum'] = float(blocked['pnl_pct'].fillna(0).sum())

    if has_outcome:
        context['tp_blocked'] = int((blocked['trade_outcome'] == 'TP').sum())
        context['sl_blocked'] = int((blocked['trade_outcome'] == 'SL').sum())
        context['tp_kept'] = int((accepted['trade_outcome'] == 'TP').sum())
        context['sl_kept'] = int((accepted['trade_outcome'] == 'SL').sum())

        sorted_df = filtered_df.sort_values('open_time')
        context['max_losing_streak_before'] = _max_losing_streak(sorted_df['trade_outcome'].tolist())
        context['max_losing_streak_after'] = _max_losing_streak(
            sorted_df.loc[~sorted_df['blocked_by_policy'], 'trade_outcome'].tolist()
        ) if len(accepted) > 0 else 0

    if 'policy_episode_id' in filtered_df.columns:
        episode_ids = filtered_df.loc[
            filtered_df['policy_episode_id'] > 0, 'policy_episode_id'
        ].unique()
        context['pause_episodes_count'] = len(episode_ids)

        durations = []
        for eid in episode_ids:
            ep = filtered_df[(filtered_df['policy_episode_id'] == eid) & filtered_df['blocked_by_policy']]
            if 'open_time' in ep.columns and len(ep) > 0:
                dur = (ep['open_time'].max() - ep['open_time'].min()).total_seconds() / 3600
                durations.append(dur)
        context['avg_pause_duration_hours'] = float(np.mean(durations)) if durations else 0.0

    if 'p_bad' in filtered_df.columns:
        p_bad = filtered_df['p_bad'].values
        percentiles = np.percentile(p_bad, [50, 90, 95, 99])
        context['p_bad_p50'] = float(percentiles[0])
        context['p_bad_p90'] = float(percentiles[1])
        context['p_bad_p95'] = float(percentiles[2])
        context['p_bad_p99'] = float(percentiles[3])

    return {**scorecard, **diagnostics, **context}


def compute_cv_score(
        metrics: dict,
        score_mode: str = 'pnl_improvement',
        max_blocked_share: float = 0.35,
        min_signal_keep_rate: float = 0.45,
) -> float:
    blocked_share = metrics.get('blocked_share', 0)
    signal_keep_rate = metrics.get('signal_keep_rate', 1.0)

    if blocked_share > max_blocked_share or signal_keep_rate < min_signal_keep_rate:
        return -np.inf

    no_op_penalty = -1000 if blocked_share == 0 else 0

    pnl_improvement = metrics.get('pnl_improvement', 0)
    blocked_bad_precision = metrics.get('blocked_bad_precision', 0)
    tp_tax = metrics.get('tp_tax', 0)
    episode_recall = metrics.get('episode_recall', 0)
    worst_window_improvement_12h = metrics.get('worst_window_improvement_12h', 0)

    worst_6h_before = metrics.get('worst_window_6h_before', 0)
    worst_6h_after = metrics.get('worst_window_6h_after', 0)
    worst_window_improvement_6h = worst_6h_after - worst_6h_before

    streak_before = metrics.get('max_losing_streak_before', 0)
    streak_after = metrics.get('max_losing_streak_after', 0)
    streak_reduction = streak_before - streak_after

    pnl_after = metrics.get('pnl_after', 0)
    sl_blocked = metrics.get('sl_blocked', 0)
    tp_blocked = metrics.get('tp_blocked', 0)
    block_value = 10.0 * sl_blocked - 4.5 * tp_blocked

    if score_mode == 'pnl_after':
        score = pnl_after
    elif score_mode == 'block_value':
        score = block_value
    elif score_mode == 'comprehensive':
        score = (
            pnl_improvement * 1.0 +
            streak_reduction * 20.0 +
            blocked_bad_precision * 50.0 +
            worst_window_improvement_6h * 2.0 +
            worst_window_improvement_12h * 1.0 +
            episode_recall * 100.0 +
            no_op_penalty
        )
        if blocked_bad_precision < 0.35:
            score -= 100
        if tp_tax > 0.30:
            score -= (tp_tax - 0.30) * 100
        if blocked_share > 0.25:
            score -= (blocked_share - 0.25) * 100
    else:
        score = (
            pnl_improvement +
            worst_window_improvement_12h * 0.5 +
            streak_reduction * 10.0 +
            episode_recall * 50.0
        )
        if tp_tax > 0.30:
            score -= (tp_tax - 0.30) * 50
        if blocked_share > 0.25:
            score -= (blocked_share - 0.25) * 100
        score += no_op_penalty

    return score


def build_pause_episodes(filtered_df: pd.DataFrame) -> pd.DataFrame:
    if 'policy_episode_id' not in filtered_df.columns:
        return pd.DataFrame()

    episode_ids = filtered_df.loc[
        filtered_df['policy_episode_id'] > 0, 'policy_episode_id'
    ].unique()

    episodes = []
    for eid in sorted(episode_ids):
        ep = filtered_df[filtered_df['policy_episode_id'] == eid]
        blocked_ep = ep[ep['blocked_by_policy']]

        if blocked_ep.empty:
            continue

        start_time = blocked_ep['open_time'].min()
        end_time = blocked_ep['open_time'].max()
        duration_hours = (end_time - start_time).total_seconds() / 3600

        has_outcome = 'trade_outcome' in blocked_ep.columns
        signals_blocked = len(blocked_ep)
        sl_blocked = int((blocked_ep['trade_outcome'] == 'SL').sum()) if has_outcome else 0
        tp_blocked = int((blocked_ep['trade_outcome'] == 'TP').sum()) if has_outcome else 0

        blocked_pnl = float(blocked_ep['pnl_pct'].fillna(0).sum()) if 'pnl_pct' in blocked_ep.columns else 0.0

        trigger_p_bad = float(ep.iloc[0]['bucket_p_bad']) if 'bucket_p_bad' in ep.columns else np.nan

        pause_reason = ep.iloc[0].get('pause_reason', '') if 'pause_reason' in ep.columns else ''
        resume_rows = ep[ep['resume_reason'] != ''] if 'resume_reason' in ep.columns else pd.DataFrame()
        resume_reason = resume_rows.iloc[-1]['resume_reason'] if len(resume_rows) > 0 else 'end_of_data'

        episodes.append({
            'episode_id': int(eid),
            'start_time': start_time,
            'end_time': end_time,
            'duration_hours': round(duration_hours, 2),
            'signals_blocked': signals_blocked,
            'sl_blocked': sl_blocked,
            'tp_blocked': tp_blocked,
            'blocked_pnl_sum': round(blocked_pnl, 2),
            'trigger_p_bad': round(trigger_p_bad, 4) if not np.isnan(trigger_p_bad) else np.nan,
            'resume_reason': resume_reason,
        })

    return pd.DataFrame(episodes)


def build_bucket_summary_6h(filtered_df: pd.DataFrame) -> pd.DataFrame:
    if 'open_time' not in filtered_df.columns or len(filtered_df) == 0:
        return pd.DataFrame()

    df = filtered_df.copy()
    df['bucket_6h'] = df['open_time'].dt.floor('6h')
    has_outcome = 'trade_outcome' in df.columns
    has_pnl = 'pnl_pct' in df.columns

    rows = []
    for bucket, bdf in df.groupby('bucket_6h'):
        blocked = bdf[bdf['blocked_by_policy']]
        accepted = bdf[~bdf['blocked_by_policy']]

        row = {
            'bucket_6h': bucket,
            'signals_before': len(bdf),
            'signals_after': len(accepted),
            'blocked': len(blocked),
        }

        if has_outcome:
            row['blocked_tp'] = int((blocked['trade_outcome'] == 'TP').sum())
            row['blocked_sl'] = int((blocked['trade_outcome'] == 'SL').sum())

        if has_pnl:
            row['pnl_before'] = float(bdf['pnl_pct'].fillna(0).sum())
            row['pnl_after'] = float(accepted['pnl_pct'].fillna(0).sum())

        for col in ['btc_ret_16', 'breadth_pos_16', 'breadth_mean_ret_16']:
            if col in bdf.columns:
                row[col] = float(bdf[col].mean())

        rows.append(row)

    return pd.DataFrame(rows)


def build_p_bad_deciles(df: pd.DataFrame) -> pd.DataFrame:
    if 'p_bad' not in df.columns or len(df) == 0:
        return pd.DataFrame()

    tmp = df.copy()
    try:
        tmp['p_bad_decile'] = pd.qcut(tmp['p_bad'], 10, labels=False, duplicates='drop')
    except ValueError:
        return pd.DataFrame()

    rows = []
    for decile, ddf in tmp.groupby('p_bad_decile'):
        row = {
            'decile': int(decile),
            'mean_p_bad': float(ddf['p_bad'].mean()),
            'count': len(ddf),
        }

        if 'trade_outcome' in ddf.columns:
            resolved = ddf[ddf['trade_outcome'].isin(['TP', 'SL'])]
            row['bad_rate'] = float((resolved['trade_outcome'] == 'SL').mean()) if len(resolved) > 0 else np.nan

        if 'target_future_block_value_12h' in ddf.columns:
            vals = ddf['target_future_block_value_12h'].dropna()
            row['future_block_value_12h_mean'] = float(vals.mean()) if len(vals) > 0 else np.nan

        if 'target_future_pnl_sum_12h' in ddf.columns:
            vals = ddf['target_future_pnl_sum_12h'].dropna()
            row['future_pnl_sum_12h_mean'] = float(vals.mean()) if len(vals) > 0 else np.nan

        if 'target_next_5_sl_rate' in ddf.columns:
            vals = ddf['target_next_5_sl_rate'].dropna()
            row['sl_rate_next_5_mean'] = float(vals.mean()) if len(vals) > 0 else np.nan

        rows.append(row)

    return pd.DataFrame(rows)


def save_btc_pause_overlay(filtered_df: pd.DataFrame, btc_candles: pd.DataFrame, out_path):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError:
        return

    if btc_candles.empty or filtered_df.empty or 'open_time' not in filtered_df.columns:
        return

    start = filtered_df['open_time'].min()
    end = filtered_df['open_time'].max()
    btc = btc_candles[(btc_candles.index >= start) & (btc_candles.index <= end)]

    if btc.empty:
        return

    df = filtered_df.copy()
    df['bucket_6h'] = df['open_time'].dt.floor('6h')
    blocked = df[df['blocked_by_policy']]
    has_outcome = 'trade_outcome' in blocked.columns

    bucket_stats = []
    if not blocked.empty:
        for bucket, bdf in blocked.groupby('bucket_6h'):
            tp_b = int((bdf['trade_outcome'] == 'TP').sum()) if has_outcome else 0
            sl_b = int((bdf['trade_outcome'] == 'SL').sum()) if has_outcome else 0
            bucket_stats.append({'time': bucket, 'blocked_tp': tp_b, 'blocked_sl': sl_b})

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(16, 8),
        gridspec_kw={'height_ratios': [3, 1]}, sharex=True,
    )

    btc_6h = btc['close'].resample('6h').last().dropna()
    ax1.plot(btc_6h.index, btc_6h.values, color='black', linewidth=0.8)
    ax1.set_ylabel('BTC Price')

    episodes = build_pause_episodes(filtered_df)
    for _, ep in episodes.iterrows():
        ax1.axvspan(ep['start_time'], ep['end_time'], alpha=0.15, color='red')

    if bucket_stats:
        bs_df = pd.DataFrame(bucket_stats)
        ax2.bar(bs_df['time'], bs_df['blocked_sl'], width=0.2, color='red', alpha=0.7, label='Blocked SL')
        ax2.bar(
            bs_df['time'], bs_df['blocked_tp'],
            bottom=bs_df['blocked_sl'], width=0.2, color='green', alpha=0.7, label='Blocked TP',
        )
        ax2.legend(fontsize=8)

    ax2.set_ylabel('Blocked Count (6h)')
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=100)
    plt.close(fig)


def build_pause_start_explanations(
        filtered_df: pd.DataFrame,
        model,
        feature_columns: list,
) -> pd.DataFrame:
    if 'policy_episode_id' not in filtered_df.columns:
        return pd.DataFrame()

    episode_ids = filtered_df.loc[
        filtered_df['policy_episode_id'] > 0, 'policy_episode_id'
    ].unique()

    available_features = [c for c in feature_columns if c in filtered_df.columns]
    if not available_features:
        return pd.DataFrame()

    rows = []
    for eid in sorted(episode_ids):
        ep = filtered_df[filtered_df['policy_episode_id'] == eid]
        first_row = ep.sort_values('open_time').iloc[0]

        x = first_row[available_features].values.reshape(1, -1)

        row = {
            'episode_id': int(eid),
            'start_time': first_row['open_time'],
            'p_bad': float(first_row.get('p_bad', np.nan)),
        }

        try:
            from catboost import Pool
            pool = Pool(x, feature_names=available_features)
            shap_values = model.get_feature_importance(pool, type='ShapValues')
            contributions = shap_values[0][:-1]

            top_indices = np.argsort(np.abs(contributions))[::-1][:5]
            for rank, idx in enumerate(top_indices, 1):
                row[f'top_{rank}_feature'] = available_features[idx]
                row[f'top_{rank}_shap'] = float(contributions[idx])
                val = x[0][idx]
                row[f'top_{rank}_value'] = float(val) if not np.isnan(val) else None
        except Exception:
            pass

        rows.append(row)

    return pd.DataFrame(rows)
