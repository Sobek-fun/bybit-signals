#!/usr/bin/env python3
import argparse
import json
import os
import time
import traceback
from datetime import datetime

import clickhouse_connect

SOL_MINTS = [
    "11111111111111111111111111111111",
    "So11111111111111111111111111111111111111112",
]

STABLE_MINTS = [
    "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
    "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB",
    "USD1ttGY1N17NEEHLmELoaybftRBUSErhqYiQzvEmuB",
    "2u1tszSeqZ3qBWF3uNGPFc8TzMk2tdiwknnRMWGWjGWH",
    "2b1kV6DkPAnxd5ixfnxCpjxmKwqjjaYmCZfHsFu24GXo",
    "EjmyN6qEC1Tf1JxiG1ae7UTJhUxSwk1TCWNWqxWV4J6o",
    "9zNQRsGLjNKwCUU5Gq5LR8beUCPzQMVMqKAi3SSZh54u",
]


def bytes_expr(base58: str) -> str:
    # important: addresses stored in bytes FixedString(32)
    return f"toFixedString(base58Decode('{base58}'), 32)"


def list_expr(addrs) -> str:
    return ",\n        ".join(bytes_expr(a) for a in addrs)


SOL_LIST = list_expr(SOL_MINTS)
STABLE_LIST = list_expr(STABLE_MINTS)

FILTER_EXPR = f"""
(
    (from_mint IN [
        {SOL_LIST}
    ] OR to_mint IN [
        {SOL_LIST}
    ])
    AND
    (
        (from_mint IN [
            {SOL_LIST}
        ]) != (to_mint IN [
            {SOL_LIST}
        ])
    )
    AND NOT
    (
        from_mint IN [
            {STABLE_LIST}
        ]
        OR
        to_mint IN [
            {STABLE_LIST}
        ]
    )
)
""".strip()


def now_s() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def load_state(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_state(path: str, state: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2, default=str)
    os.replace(tmp, path)


def run_command(client, sql: str, label: str = "", settings: dict | None = None) -> None:
    t0 = time.time()
    client.command(sql, settings=settings)
    dt = time.time() - t0
    print(f"[{now_s()}] OK {label} ({dt:.2f}s)")


def run_query_one(client, sql: str, label: str = "", settings: dict | None = None):
    t0 = time.time()
    res = client.query(sql, settings=settings)
    dt = time.time() - t0
    rows = res.result_rows
    if not rows:
        print(f"[{now_s()}] OK {label} -> no rows ({dt:.2f}s)")
        return None
    print(f"[{now_s()}] OK {label} ({dt:.2f}s)")
    return rows[0]


def ensure_backup_table(client, backup_table: str) -> None:
    # IMPORTANT: create schema-only copy WITHOUT projections (AS SELECT ... LIMIT 0)
    sql = f"""
CREATE TABLE IF NOT EXISTS {backup_table}
ENGINE = MergeTree
PARTITION BY toYYYYMM(block_time)
ORDER BY (block_time, slot, tx_index, inst_index)
AS
SELECT *
FROM swaps
LIMIT 0
"""
    run_command(client, sql, label=f"ensure backup table {backup_table}")


def get_swaps_columns(client) -> list[str]:
    sql = """
SELECT name
FROM system.columns
WHERE database = currentDatabase() AND table = 'swaps'
ORDER BY position
"""
    res = client.query(sql)
    return [r[0] for r in res.result_rows]


def stats_for_range(client, s_from: int, s_to: int):
    sql = f"""
SELECT
    count() AS rows,
    min(block_time) AS bt_from,
    max(block_time) AS bt_to
FROM swaps
PREWHERE slot BETWEEN {s_from} AND {s_to}
WHERE {FILTER_EXPR}
"""
    row = run_query_one(client, sql, label=f"stats swaps [{s_from},{s_to}]")
    if row is None:
        return 0, None, None
    rows, bt_from, bt_to = row
    return int(rows), bt_from, bt_to


def count_backup_for_range(client, backup_table: str, s_from: int, s_to: int, bt_from, bt_to) -> int:
    bt_from_s = bt_from.strftime("%Y-%m-%d %H:%M:%S")
    bt_to_s = bt_to.strftime("%Y-%m-%d %H:%M:%S")
    sql = f"""
SELECT count()
FROM {backup_table}
PREWHERE
    block_time BETWEEN toDateTime('{bt_from_s}') AND toDateTime('{bt_to_s}')
    AND slot BETWEEN {s_from} AND {s_to}
WHERE {FILTER_EXPR}
"""
    row = run_query_one(client, sql, label=f"count backup [{s_from},{s_to}]")
    return int(row[0]) if row else 0


def backup_insert(client, backup_table: str, s_from: int, s_to: int, bt_from, bt_to) -> None:
    bt_from_s = bt_from.strftime("%Y-%m-%d %H:%M:%S")
    bt_to_s = bt_to.strftime("%Y-%m-%d %H:%M:%S")
    sql = f"""
INSERT INTO {backup_table}
SELECT *
FROM swaps
PREWHERE
    block_time BETWEEN toDateTime('{bt_from_s}') AND toDateTime('{bt_to_s}')
    AND slot BETWEEN {s_from} AND {s_to}
WHERE {FILTER_EXPR}
"""
    run_command(client, sql, label=f"backup insert [{s_from},{s_to}]")


def delete_swaps_range(client, s_from: int, s_to: int, bt_from, bt_to) -> None:
    bt_from_s = bt_from.strftime("%Y-%m-%d %H:%M:%S")
    bt_to_s = bt_to.strftime("%Y-%m-%d %H:%M:%S")
    sql = f"""
ALTER TABLE swaps
DELETE WHERE
    block_time BETWEEN toDateTime('{bt_from_s}') AND toDateTime('{bt_to_s}')
    AND slot BETWEEN {s_from} AND {s_to}
    AND {FILTER_EXPR}
SETTINGS mutations_sync = 1
"""
    run_command(client, sql, label=f"delete swaps [{s_from},{s_to}]")


def insert_repriced_from_backup(
    client,
    backup_table: str,
    swaps_cols: list[str],
    s_from: int,
    s_to: int,
    bt_from,
    bt_to,
) -> None:
    # Build SELECT list in exact swaps column order, replacing usd_amount with computed expression
    if "usd_amount" not in swaps_cols:
        raise RuntimeError("Column 'usd_amount' not found in swaps schema.")

    bt_from_s = bt_from.strftime("%Y-%m-%d %H:%M:%S")
    bt_to_s = bt_to.strftime("%Y-%m-%d %H:%M:%S")

    # computed usd_amount: EXACT microCents logic that worked
    computed_usd = f"""
CAST(
    if(
        sp.price IS NULL OR sp.price = 0 OR toUnixTimestamp(b.block_time) = 0,
        NULL,
        toDecimal128(
            toFloat64(
                toInt64(
                    (
                        (
                            (toFloat64(if(b.from_mint IN sol_mints, b.from_amount, b.to_amount)) / 1000000000.0)
                            * toFloat64(sp.price)
                        ) * 1000000000000.0
                    )
                )
            ) / 1000000000000.0,
            12
        )
    )
    AS Nullable(Decimal(28,12))
) AS usd_amount
""".strip()

    select_items = []
    for c in swaps_cols:
        if c == "usd_amount":
            select_items.append(computed_usd)
        else:
            select_items.append(f"b.{c}")

    select_list = ",\n    ".join(select_items)
    insert_cols = ", ".join(swaps_cols)

    # We try to be restart-safe:
    # - select from backup (because swaps may already be deleted)
    # - dedupe at source ONLY if backup got duplicated: we do a light DISTINCT on (all columns) inside the batch
    #   You can disable DISTINCT for maximum speed if you are 100% sure backup has no duplicates.
    #   Here we keep it ON for reliability.
    sql = f"""
INSERT INTO swaps ({insert_cols})
WITH
    [
        {SOL_LIST}
    ] AS sol_mints
SELECT
    {select_list}
FROM
(
    SELECT DISTINCT *
    FROM {backup_table}
    PREWHERE
        block_time BETWEEN toDateTime('{bt_from_s}') AND toDateTime('{bt_to_s}')
        AND slot BETWEEN {s_from} AND {s_to}
    WHERE {FILTER_EXPR}
) AS b
ANY LEFT JOIN sol_prices AS sp
    ON sp.timestamp = toDateTime(intDiv(toUnixTimestamp(b.block_time), 900) * 900)
"""
    run_command(client, sql, label=f"insert repriced [{s_from},{s_to}]")


def propose_span_slots(client, slot_from: int, end_slot: int, target_rows: int, probe_span: int, last_rows_per_slot: float | None):
    # Estimate rows/slot on a probe window to choose a slot span that ~target_rows.
    # Keep this light: one stats query on a small range.
    if slot_from >= end_slot:
        return 0, None

    probe_to = min(end_slot, slot_from + probe_span)
    rows, bt_from, bt_to = stats_for_range(client, slot_from, probe_to)
    if rows <= 0:
        # If no rows in probe, just move forward by probe_span
        return min(probe_span, end_slot - slot_from), last_rows_per_slot

    rows_per_slot = rows / max(1, (probe_to - slot_from))
    if last_rows_per_slot is not None:
        # smooth a bit
        rows_per_slot = 0.6 * last_rows_per_slot + 0.4 * rows_per_slot

    span = int(target_rows / max(rows_per_slot, 1e-12))
    span = max(10_000, min(span, 5_000_000))  # safety clamp
    span = min(span, end_slot - slot_from)
    return span, rows_per_slot


def main():
    ap = argparse.ArgumentParser(description="Reprice SOL<->token swaps in ClickHouse using sol_prices, with backup + batched delete/insert.")
    ap.add_argument("--host", default="84.32.97.20")
    ap.add_argument("--port", type=int, default=8123)
    ap.add_argument("--user", default="admin")
    ap.add_argument("--password", default="Wv4A4QyWk4Lvzs74")
    ap.add_argument("--database", default="solana")

    ap.add_argument("--start-slot", type=int, default=358557506)
    ap.add_argument("--end-slot", type=int, default=387603957)

    ap.add_argument("--backup-table", default="swaps_backup_sol_reprice_358557506_387603957")

    ap.add_argument("--target-rows", type=int, default=10_000_000)
    ap.add_argument("--probe-span-slots", type=int, default=200_000)

    ap.add_argument("--min-span-slots", type=int, default=20_000)

    ap.add_argument("--state-file", default="swap_reprice_state.json")
    ap.add_argument("--http-timeout", type=int, default=1800, help="send/receive timeout (seconds)")

    args = ap.parse_args()

    print(f"[{now_s()}] Connecting to ClickHouse {args.host}:{args.port} db={args.database} user={args.user}")
    client = clickhouse_connect.get_client(
        host=args.host,
        port=args.port,
        username=args.user,
        password=args.password,
        database=args.database,
        connect_timeout=10,
        send_receive_timeout=args.http_timeout,
    )

    # Ensure backup table exists (no projections)
    ensure_backup_table(client, args.backup_table)

    swaps_cols = get_swaps_columns(client)
    print(f"[{now_s()}] swaps columns: {len(swaps_cols)}")

    state = load_state(args.state_file)
    slot = int(state.get("next_slot", args.start_slot))
    last_rows_per_slot = state.get("rows_per_slot", None)

    # Clamp resume slot into requested bounds
    slot = max(args.start_slot, min(slot, args.end_slot + 1))

    print(f"[{now_s()}] Start processing: slot={slot} .. {args.end_slot}, target_rows≈{args.target_rows}")

    while slot <= args.end_slot:
        try:
            span, last_rows_per_slot = propose_span_slots(
                client,
                slot_from=slot,
                end_slot=args.end_slot,
                target_rows=args.target_rows,
                probe_span=args.probe_span_slots,
                last_rows_per_slot=last_rows_per_slot,
            )
            span = max(args.min_span_slots, span)
            s_to = min(args.end_slot, slot + span)

            # Get accurate stats for proposed batch
            rows, bt_from, bt_to = stats_for_range(client, slot, s_to)
            if rows == 0 or bt_from is None or bt_to is None:
                print(f"[{now_s()}] No matching swaps in [{slot},{s_to}] -> advance")
                slot = s_to + 1
                state["next_slot"] = slot
                state["rows_per_slot"] = last_rows_per_slot
                save_state(args.state_file, state)
                continue

            print(f"[{now_s()}] Batch candidate [{slot},{s_to}] rows={rows} bt=[{bt_from} .. {bt_to}]")

            # If rows far above target, shrink span a bit (without heavy binary search)
            if rows > int(args.target_rows * 1.4) and span > args.min_span_slots:
                # shrink proportionally, retry selection
                factor = rows / args.target_rows
                new_span = int(span / factor)
                new_span = max(args.min_span_slots, new_span)
                s_to = min(args.end_slot, slot + new_span)
                rows, bt_from, bt_to = stats_for_range(client, slot, s_to)
                print(f"[{now_s()}] Adjusted batch [{slot},{s_to}] rows={rows} bt=[{bt_from} .. {bt_to}]")

            # --- Backup logic (skip if already backed up enough rows) ---
            backup_cnt = count_backup_for_range(client, args.backup_table, slot, s_to, bt_from, bt_to)
            if backup_cnt >= int(rows * 0.995):
                print(f"[{now_s()}] Backup already present: backup_cnt={backup_cnt} expected≈{rows} -> skip backup insert")
            else:
                print(f"[{now_s()}] Backup missing: backup_cnt={backup_cnt} expected≈{rows} -> backup insert")
                backup_insert(client, args.backup_table, slot, s_to, bt_from, bt_to)
                backup_cnt2 = count_backup_for_range(client, args.backup_table, slot, s_to, bt_from, bt_to)
                if backup_cnt2 < int(rows * 0.995):
                    raise RuntimeError(f"Backup insert seems incomplete: backup_cnt={backup_cnt2}, expected≈{rows}")

            # --- Update: delete + insert repriced ---
            delete_swaps_range(client, slot, s_to, bt_from, bt_to)
            insert_repriced_from_backup(client, args.backup_table, swaps_cols, slot, s_to, bt_from, bt_to)

            # Success -> advance
            slot = s_to + 1
            state["next_slot"] = slot
            state["rows_per_slot"] = last_rows_per_slot
            save_state(args.state_file, state)
            print(f"[{now_s()}] ✅ Completed batch. Next slot={slot}")

        except Exception as e:
            print(f"[{now_s()}] ❌ ERROR in batch starting at slot={slot}: {e}")
            traceback.print_exc()

            # Halve the span for next attempt
            # We do this by decreasing probe-derived rows_per_slot influence and forcing smaller span.
            prev_next = state.get("next_slot", slot)
            state["next_slot"] = slot  # retry same slot
            # Nudge rows_per_slot upward to force smaller spans (conservative)
            if last_rows_per_slot is not None:
                last_rows_per_slot *= 1.8
                state["rows_per_slot"] = last_rows_per_slot
            save_state(args.state_file, state)

            # Also reduce probe_span temporarily to be more local
            args.probe_span_slots = max(50_000, args.probe_span_slots // 2)
            args.target_rows = max(250_000, args.target_rows // 2)
            print(
                f"[{now_s()}] Retrying with smaller batches: "
                f"target_rows≈{args.target_rows}, probe_span_slots={args.probe_span_slots}"
            )
            # short backoff
            time.sleep(3)

    print(f"[{now_s()}] 🎉 DONE. All slots processed up to {args.end_slot}.")


if __name__ == "__main__":
    main()
