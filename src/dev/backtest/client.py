import time
from typing import Optional

import requests


def submit_experiment(
        signals: list[dict],
        strategy_grid: dict,
        meta: dict,
        base_url: str,
        api_key: str
) -> dict:
    url = f"{base_url}/v1/experiments"
    headers = {
        "Content-Type": "application/json",
        "X-API-Key": api_key
    }
    payload = {
        "signals": signals,
        "strategy_grid": strategy_grid,
        "meta": meta
    }
    response = requests.post(url, json=payload, headers=headers)
    if response.status_code >= 400:
        print(f"[ERROR] Backtest API error: {response.status_code} {response.text}")
    response.raise_for_status()
    return response.json()

def poll_job(
        job_id: str,
        base_url: str,
        api_key: str,
        timeout_sec: int = 120,
        poll_interval_sec: int = 1
) -> dict:
    url = f"{base_url}/v1/experiments/{job_id}"
    headers = {"X-API-Key": api_key}

    start_time = time.time()
    while True:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        status_data = response.json()

        if status_data["status"] in ("done", "failed"):
            return status_data

        elapsed = time.time() - start_time
        if elapsed >= timeout_sec:
            raise TimeoutError(f"Job {job_id} did not complete within {timeout_sec}s")

        time.sleep(poll_interval_sec)


def get_result(job_id: str, base_url: str, api_key: str) -> dict:
    url = f"{base_url}/v1/experiments/{job_id}/result"
    headers = {"X-API-Key": api_key}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.json()


def download_artifact(artifact_path: str, base_url: str, api_key: str) -> bytes:
    url = f"{base_url}{artifact_path}"
    headers = {"X-API-Key": api_key}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.content


def select_best_strategy_winrate_first(
        experiments_csv_content: bytes,
        min_trades: int = 200,
        winrate_eps: float = 1.0
) -> Optional[dict]:
    import pandas as pd
    from io import BytesIO

    df = pd.read_csv(BytesIO(experiments_csv_content))

    df = df[df['total_trades'] >= min_trades]
    if df.empty:
        return None

    max_winrate = df['winrate_all_pct'].max()
    df = df[df['winrate_all_pct'] >= max_winrate - winrate_eps]

    df = df.sort_values(
        by=['sl_pct', 'tp_pct', 'timeout_pct', 'profit_factor', 'total_pnl_usdt'],
        ascending=[True, False, True, False, False]
    )

    best_row = df.iloc[0]
    return {
        'tp_pct': best_row['tp_pct'],
        'sl_pct': best_row['sl_pct'],
        'max_holding_hours': int(best_row['max_holding_hours']),
        'winrate_all_pct': best_row['winrate_all_pct'],
        'total_trades': int(best_row['total_trades']),
        'total_pnl_usdt': best_row['total_pnl_usdt'],
        'profit_factor': best_row['profit_factor'],
        'timeout_pct': best_row['timeout_pct']
    }
