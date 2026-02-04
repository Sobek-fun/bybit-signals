import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from pump_long.features.feature_builder import PumpLongFeatureBuilder, get_feature_columns, META_COLUMNS
from pump_long.infra.logging import log


def load_labels(path: str, start_date: datetime = None, end_date: datetime = None) -> pd.DataFrame:
    df = pd.read_csv(path)

    if 'event_open_time' not in df.columns:
        if 'timestamp' in df.columns:
            df = df.rename(columns={'timestamp': 'event_open_time'})

    df['event_open_time'] = pd.to_datetime(df['event_open_time'], utc=True).dt.tz_localize(None)

    if start_date:
        df = df[df['event_open_time'] >= start_date]
    if end_date:
        df = df[df['event_open_time'] < end_date]

    return df


def build_training_points_from_labels(
        labels_df: pd.DataFrame,
        neg_before: int = 60,
        neg_after: int = 10,
        pos_offsets: list = None
) -> pd.DataFrame:
    if pos_offsets is None:
        pos_offsets = [0]

    events = labels_df[labels_df['pump_la_type'] == 'A'].copy()

    if events.empty:
        return pd.DataFrame()

    events['event_id'] = events['symbol'] + '|' + events['event_open_time'].dt.strftime('%Y%m%d_%H%M%S')

    all_offsets = list(range(-neg_before, 0)) + pos_offsets + list(
        range(max(pos_offsets) + 1, max(pos_offsets) + neg_after + 1))
    all_offsets = sorted(set(all_offsets))

    n_events = len(events)
    n_offsets = len(all_offsets)

    event_ids = np.repeat(events['event_id'].values, n_offsets)
    symbols = np.repeat(events['symbol'].values, n_offsets)
    base_times = np.repeat(events['event_open_time'].values, n_offsets)
    runup_pcts = np.repeat(events['runup_pct'].values if 'runup_pct' in events.columns else np.nan, n_offsets)

    offsets = np.tile(all_offsets, n_events)
    time_deltas = pd.to_timedelta(offsets * 15, unit='m')
    open_times = pd.to_datetime(base_times) + time_deltas

    pos_offsets_set = set(pos_offsets)
    y_values = np.where(np.isin(offsets, list(pos_offsets_set)), 1, 0)

    points_df = pd.DataFrame({
        'event_id': event_ids,
        'symbol': symbols,
        'open_time': open_times,
        'offset': offsets,
        'y': y_values,
        'runup_pct': runup_pcts
    })

    return points_df


def generate_random_negatives(
        labels_df: pd.DataFrame,
        neg_before: int,
        neg_after: int,
        random_neg_mult: int,
        random_neg_seed: int,
        random_neg_min_gap_bars: int
) -> pd.DataFrame:
    events = labels_df[labels_df['pump_la_type'] == 'A'].copy()

    if events.empty or random_neg_mult <= 0:
        return pd.DataFrame()

    events['event_open_time'] = pd.to_datetime(events['event_open_time'])

    n_positives = len(events)
    n_random_neg = n_positives * random_neg_mult

    rng = np.random.default_rng(random_neg_seed)

    symbols = events['symbol'].unique()
    symbol_events = {}
    for sym in symbols:
        sym_events = events[events['symbol'] == sym]['event_open_time'].values
        symbol_events[sym] = pd.to_datetime(sym_events)

    global_min_time = events['event_open_time'].min()
    global_max_time = events['event_open_time'].max()

    all_random_negatives = []

    neg_per_symbol = max(1, n_random_neg // len(symbols))

    for sym in symbols:
        sym_event_times = symbol_events[sym]

        forbidden_zones = []
        for evt in sym_event_times:
            zone_start = evt - pd.Timedelta(minutes=neg_before * 15)
            zone_end = evt + pd.Timedelta(minutes=neg_after * 15)
            forbidden_zones.append((zone_start, zone_end))

        candidates = []
        attempt_count = 0
        max_attempts = neg_per_symbol * 20

        while len(candidates) < neg_per_symbol and attempt_count < max_attempts:
            attempt_count += 1

            random_minutes = rng.integers(0, int((global_max_time - global_min_time).total_seconds() / 60) + 1)
            candidate_time = global_min_time + pd.Timedelta(minutes=int(random_minutes))

            candidate_time = candidate_time.floor('15min')

            in_forbidden = False
            for zone_start, zone_end in forbidden_zones:
                if zone_start <= candidate_time <= zone_end:
                    in_forbidden = True
                    break

            if in_forbidden:
                continue

            too_close = False
            for existing_time in candidates:
                gap_bars = abs((candidate_time - existing_time).total_seconds()) / (15 * 60)
                if gap_bars < random_neg_min_gap_bars:
                    too_close = True
                    break

            if too_close:
                continue

            candidates.append(candidate_time)

        for i, cand_time in enumerate(candidates):
            event_id = f"NEG|{sym}|{cand_time.strftime('%Y%m%d_%H%M%S')}"
            all_random_negatives.append({
                'event_id': event_id,
                'symbol': sym,
                'open_time': cand_time,
                'offset': 0,
                'y': 0,
                'runup_pct': np.nan
            })

    if not all_random_negatives:
        return pd.DataFrame()

    return pd.DataFrame(all_random_negatives)


def parse_pos_offsets(offsets_str: str) -> list:
    return [int(x.strip()) for x in offsets_str.split(',')]


def parse_date_exclusive(date_str: str) -> datetime:
    dt = datetime.strptime(date_str, '%Y-%m-%d')
    return dt + timedelta(days=1)


def validate_feature_columns(df: pd.DataFrame, expected_columns: list, feature_set: str):
    actual_feature_cols = [c for c in df.columns if c not in META_COLUMNS]
    expected_set = set(expected_columns)
    actual_set = set(actual_feature_cols)

    missing = expected_set - actual_set
    extra = actual_set - expected_set

    if missing:
        raise ValueError(f"Missing feature columns for {feature_set}: {sorted(missing)[:10]}... ({len(missing)} total)")

    if extra:
        log("WARN", "BUILD", f"Extra columns detected (will be ignored): {sorted(extra)[:5]}...")


def main():
    parser = argparse.ArgumentParser(description="Build features dataset for pump long model")

    parser.add_argument("--clickhouse-dsn", type=str, required=True)
    parser.add_argument("--labels", type=str, required=True)
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    parser.add_argument("--neg-before", type=int, default=60)
    parser.add_argument("--neg-after", type=int, default=10)
    parser.add_argument("--pos-offsets", type=str, default="0")
    parser.add_argument("--window-bars", type=int, default=60)
    parser.add_argument("--warmup-bars", type=int, default=150)
    parser.add_argument("--feature-set", type=str, choices=["base", "extended"], default="extended")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out-dir", type=str, required=True)

    parser.add_argument("--random-neg-mult", type=int, default=0)
    parser.add_argument("--random-neg-seed", type=int, default=42)
    parser.add_argument("--random-neg-min-gap-bars", type=int, default=4)

    args = parser.parse_args()

    start_date = datetime.strptime(args.start_date, '%Y-%m-%d') if args.start_date else None
    end_date = parse_date_exclusive(args.end_date) if args.end_date else None

    log("INFO", "BUILD", f"loading labels from {args.labels}")
    labels_df = load_labels(args.labels, start_date, end_date)
    log("INFO", "BUILD", f"loaded {len(labels_df)} labels")

    pos_offsets = parse_pos_offsets(args.pos_offsets)
    log("INFO", "BUILD",
        f"building training points neg_before={args.neg_before} neg_after={args.neg_after} pos_offsets={pos_offsets}")

    points_df = build_training_points_from_labels(
        labels_df,
        neg_before=args.neg_before,
        neg_after=args.neg_after,
        pos_offsets=pos_offsets
    )
    log("INFO", "BUILD",
        f"event-window points: {len(points_df)} (y=1: {len(points_df[points_df['y'] == 1])}, y=0: {len(points_df[points_df['y'] == 0])})")

    random_neg_total = 0
    if args.random_neg_mult > 0:
        log("INFO", "BUILD", f"generating random negatives: mult={args.random_neg_mult}, seed={args.random_neg_seed}, min_gap={args.random_neg_min_gap_bars}")

        random_neg_df = generate_random_negatives(
            labels_df,
            neg_before=args.neg_before,
            neg_after=args.neg_after,
            random_neg_mult=args.random_neg_mult,
            random_neg_seed=args.random_neg_seed,
            random_neg_min_gap_bars=args.random_neg_min_gap_bars
        )

        if not random_neg_df.empty:
            random_neg_total = len(random_neg_df)
            points_df = pd.concat([points_df, random_neg_df], ignore_index=True)
            log("INFO", "BUILD", f"added {random_neg_total} random negatives")

    log("INFO", "BUILD",
        f"total training points: {len(points_df)} (y=1: {len(points_df[points_df['y'] == 1])}, y=0: {len(points_df[points_df['y'] == 0])})")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    points_path = out_dir / "training_points.parquet"
    points_df.to_parquet(points_path, index=False)
    log("INFO", "BUILD", f"training points saved to {points_path}")

    log("INFO", "BUILD", f"building features from ClickHouse (feature_set={args.feature_set})")
    builder = PumpLongFeatureBuilder(
        ch_dsn=args.clickhouse_dsn,
        window_bars=args.window_bars,
        warmup_bars=args.warmup_bars,
        feature_set=args.feature_set
    )

    unique_times = points_df[['symbol', 'open_time']].drop_duplicates()
    feature_input = unique_times.copy()
    feature_input = feature_input.rename(columns={'open_time': 'event_open_time'})
    feature_input['pump_la_type'] = 'A'
    feature_input['runup_pct'] = 0

    features_df = builder.build(feature_input, max_workers=args.workers)

    if 'runup_pct' in features_df.columns:
        features_df = features_df.drop(columns=['runup_pct'])

    features_df = features_df.merge(
        points_df[['symbol', 'open_time', 'event_id', 'offset', 'y', 'runup_pct']],
        on=['symbol', 'open_time'],
        how='inner'
    )

    features_df = features_df.sort_values(['event_id', 'offset']).reset_index(drop=True)

    expected_feature_columns = get_feature_columns(args.feature_set)
    validate_feature_columns(features_df, expected_feature_columns, args.feature_set)

    meta_cols_present = [c for c in META_COLUMNS if c in features_df.columns]
    feature_cols_present = [c for c in expected_feature_columns if c in features_df.columns]
    final_columns = meta_cols_present + feature_cols_present
    features_df = features_df[final_columns]

    log("INFO", "BUILD", f"features shape: {features_df.shape}")

    features_path = out_dir / "features.parquet"
    features_df.to_parquet(features_path, index=False)
    log("INFO", "BUILD", f"features saved to {features_path}")

    manifest = {
        'feature_columns': feature_cols_present,
        'meta_columns': meta_cols_present,
        'feature_set': args.feature_set,
        'neg_before': args.neg_before,
        'neg_after': args.neg_after,
        'pos_offsets': pos_offsets,
        'window_bars': args.window_bars,
        'warmup_bars': args.warmup_bars,
        'labels_path': args.labels,
        'start_date': args.start_date,
        'end_date': args.end_date,
        'total_rows': len(features_df),
        'total_events': features_df['event_id'].nunique() if 'event_id' in features_df.columns else 0,
        'random_neg_mult': args.random_neg_mult,
        'random_neg_seed': args.random_neg_seed,
        'random_neg_min_gap_bars': args.random_neg_min_gap_bars,
        'random_neg_total': random_neg_total,
        'created_at': datetime.now().isoformat()
    }

    manifest_path = out_dir / "dataset_manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2, default=str)

    log("INFO", "BUILD", f"manifest saved to {manifest_path}")
    log("INFO", "BUILD", f"dataset saved to {out_dir}")


if __name__ == "__main__":
    main()
