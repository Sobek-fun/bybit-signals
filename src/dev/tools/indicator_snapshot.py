import json
from datetime import datetime, timedelta

from src.shared.clickhouse import DataLoader
from src.shared.indicators import IndicatorCalculator
from src.shared.pump.detector import PumpDetector


def get_indicator_snapshot(ch_dsn: str, symbol: str, timestamp_str: str, timestamp_kind: str, lookback_candles: int):
    timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')

    if timestamp_kind == "close":
        bucket_start = timestamp - timedelta(minutes=15)
    else:
        bucket_start = timestamp

    if bucket_start.minute % 15 != 0 or bucket_start.second != 0 or bucket_start.microsecond != 0:
        raise ValueError(f"bucket_start must be aligned to 15-minute boundaries: {bucket_start}")

    start_bucket = bucket_start - timedelta(minutes=(lookback_candles - 1) * 15)
    end_bucket = bucket_start + timedelta(minutes=15)

    loader = DataLoader(ch_dsn)
    df = loader.load_candles_range(symbol, start_bucket, end_bucket)

    if df.empty:
        print(json.dumps({"error": "no data loaded"}))
        return

    calculator = IndicatorCalculator()
    df = calculator.calculate(df)

    detector = PumpDetector()
    df = detector.detect(df)

    prev_bucket = bucket_start - timedelta(minutes=15)
    cur_bucket = bucket_start
    next_bucket = bucket_start + timedelta(minutes=15)

    result = {}

    if prev_bucket in df.index:
        row = df.loc[prev_bucket]
        prev_close_time = prev_bucket + timedelta(minutes=15)
        result[prev_close_time.strftime('%Y-%m-%d %H:%M:%S')] = _row_to_dict(prev_bucket, row)

    if cur_bucket in df.index:
        row = df.loc[cur_bucket]
        cur_close_time = cur_bucket + timedelta(minutes=15)
        result[cur_close_time.strftime('%Y-%m-%d %H:%M:%S')] = _row_to_dict(cur_bucket, row)

    if next_bucket in df.index:
        row = df.loc[next_bucket]
        next_close_time = next_bucket + timedelta(minutes=15)
        result[next_close_time.strftime('%Y-%m-%d %H:%M:%S')] = _row_to_dict(next_bucket, row)

    filename = f"snapshot_{symbol}_{bucket_start.strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, 'w') as f:
        json.dump(result, f, indent=2, default=str)

    print(f"Snapshot saved to {filename}")


def _row_to_dict(bucket_start: datetime, row):
    close_time = bucket_start + timedelta(minutes=15)
    data = {
        "bucket_start": bucket_start.strftime('%Y-%m-%d %H:%M:%S'),
        "close_time": close_time.strftime('%Y-%m-%d %H:%M:%S')
    }
    for col in row.index:
        val = row[col]
        if val is None or (hasattr(val, '__iter__') and not isinstance(val, str) and len(val) == 0):
            data[col] = None
        else:
            data[col] = val
    return data
