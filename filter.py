"""
CSV Filter Tool — best_candidate
Apply AND / OR filters on CSV columns and export the result.
"""

import pandas as pd
import os
import sys

# ── Source CSV path ────────────────────────────────────────────────────────────
CSV_PATH = r"C:\Users\arnau\Documents\projetde\runs-in-behind\best_candidate\best_candidate_DFL-MAT-J03WMX.csv"

OUTPUT_DIR = os.path.dirname(CSV_PATH)


# ── Utilities ─────────────────────────────────────────────────────────────────

def load_data(path: str) -> pd.DataFrame:
    print(f"\n📂 Loading: {path}")
    df = pd.read_csv(path)
    print(f"   → {len(df)} rows  |  {len(df.columns)} columns\n")
    return df


def print_columns(df: pd.DataFrame):
    print("═" * 60)
    print(f"{'#':<4} {'Column':<40} {'Type'}")
    print("─" * 60)
    for i, col in enumerate(df.columns):
        print(f"{i:<4} {col:<40} {str(df[col].dtype)}")
    print("═" * 60)


def resolve_column(df: pd.DataFrame, raw: str):
    """Resolves a column number or name. Returns the name or None."""
    raw = raw.strip()
    if raw.isdigit():
        idx = int(raw)
        if 0 <= idx < len(df.columns):
            return df.columns[idx]
    elif raw in df.columns:
        return raw
    return None


def get_unique_values(df: pd.DataFrame, col: str, max_show: int = 30):
    vals = sorted(df[col].dropna().astype(str).unique().tolist())
    if len(vals) <= max_show:
        return vals
    return vals[:max_show] + [f"... ({len(vals)} values total)"]


# ── Numeric condition parser ───────────────────────────────────────────────────

OPS = [">=", "<=", "!=", ">", "<", "=="]

def parse_numeric_condition(token: str):
    """
    Parses a numeric condition such as '>= 5.0'.
    Returns (op, val) or (None, None) on error.
    """
    token = token.strip()
    for op in OPS:
        if token.startswith(op):
            try:
                val = float(token[len(op):].strip())
                return op, val
            except ValueError:
                return None, None
    return None, None


def build_numeric_mask(series: pd.Series, op: str, val: float) -> pd.Series:
    if op == ">":  return series > val
    if op == ">=": return series >= val
    if op == "<":  return series < val
    if op == "<=": return series <= val
    if op == "==": return series == val
    if op == "!=": return series != val


# ── Numeric filter with AND / OR support ──────────────────────────────────────

def apply_numeric_filter(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Numeric filter.
    AND syntax : >= 5 AND < 20      → keeps rows matching BOTH conditions
    OR  syntax : < 5 OR > 20        → keeps rows matching EITHER condition
    Single     : >= 5
    """
    s = df[col].dropna()
    print(f"\n  Values: min={s.min():.4g}  max={s.max():.4g}  mean={s.mean():.4g}")
    print("  Operators: >  >=  <  <=  ==  !=")
    print("  ┌─ Examples ──────────────────────────────────┐")
    print("  │  >= 5                 (single condition)    │")
    print("  │  >= 5 AND < 20        (AND — intersection)  │")
    print("  │  < 5 OR > 20          (OR  — union)         │")
    print("  └─────────────────────────────────────────────┘")
    raw = input("  Filter: ").strip()
    if not raw:
        print("  (No filter applied)")
        return df

    raw_up = raw.upper()
    if " AND " in raw_up:
        parts = [p.strip() for p in raw.split(" AND ", maxsplit=1)]
        logic = "AND"
    elif " OR " in raw_up:
        parts = [p.strip() for p in raw.split(" OR ", maxsplit=1)]
        logic = "OR"
    else:
        parts = [raw]
        logic = None

    masks = []
    for part in parts:
        op, val = parse_numeric_condition(part)
        if op is None:
            print(f"  ⚠️  Unrecognised condition: «{part}», filter ignored.")
            return df
        masks.append(build_numeric_mask(df[col], op, val))

    if logic == "AND":
        mask = masks[0] & masks[1]
        desc = f"{parts[0]} AND {parts[1]}"
    elif logic == "OR":
        mask = masks[0] | masks[1]
        desc = f"{parts[0]} OR {parts[1]}"
    else:
        mask = masks[0]
        desc = parts[0]

    before = len(df)
    df = df[mask]
    print(f"  ✅ Filter «{desc}» → {before - len(df)} rows removed, {len(df)} remaining")
    return df


# ── Text filter with OR support ───────────────────────────────────────────────

def apply_text_filter(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Text filter.
    A comma implicitly means OR: Home, Away → keeps Home OR Away.
    The keyword OR is also accepted: Home OR Away
    """
    uniques = get_unique_values(df, col)
    print(f"\n  Available values: {uniques}")
    print("  ┌─ Examples ──────────────────────────────────────────┐")
    print("  │  Home                      (single value)           │")
    print("  │  Home, Away                (OR — comma)             │")
    print("  │  1. FC Köln OR FC Bayern   (OR — keyword)           │")
    print("  └─────────────────────────────────────────────────────┘")
    raw = input("  Filter: ").strip()
    if not raw:
        print("  (No filter applied)")
        return df

    if "," in raw:
        values = [v.strip() for v in raw.split(",") if v.strip()]
    elif " OR " in raw.upper():
        values = [v.strip() for v in raw.replace(" or ", " OR ").replace(" Or ", " OR ").split(" OR ") if v.strip()]
    else:
        values = [raw]

    before = len(df)
    df = df[df[col].astype(str).isin(values)]
    print(f"  ✅ OR filter ({values}) → {before - len(df)} rows removed, {len(df)} remaining")
    return df


# ── Boolean filter ────────────────────────────────────────────────────────────

def apply_bool_filter(df: pd.DataFrame, col: str) -> pd.DataFrame:
    print(f"\n  Available values: True / False")
    raw = input("  Keep [True/False]: ").strip().lower()
    if raw not in ("true", "false"):
        print("  ⚠️  Invalid value, filter ignored.")
        return df
    keep = raw == "true"
    before = len(df)
    df = df[df[col] == keep]
    print(f"  ✅ {before - len(df)} rows removed → {len(df)} remaining")
    return df


# ── Main loop ─────────────────────────────────────────────────────────────────

def filter_loop(df: pd.DataFrame) -> pd.DataFrame:
    applied = []

    while True:
        print(f"\n{'─'*60}")
        print(f"  Current rows: {len(df)}")
        if applied:
            print(f"  Applied filters: {', '.join(applied)}")
        print("─" * 60)
        print("  [l] List columns")
        print("  [f] Add a column filter")
        print("  [r] Reset (reload CSV)")
        print("  [s] Save filtered CSV and quit")
        print("  [q] Quit without saving")
        print("─" * 60)

        choice = input("  Your choice: ").strip().lower()

        if choice == "l":
            print_columns(df)

        elif choice == "f":
            print_columns(df)
            raw_col = input("\n  Column number or name: ").strip()
            col = resolve_column(df, raw_col)
            if col is None:
                print("  ⚠️  Column not found.")
                continue

            dtype = df[col].dtype
            print(f"\n  → Selected column: «{col}»  (type: {dtype})")

            if pd.api.types.is_bool_dtype(dtype):
                df = apply_bool_filter(df, col)
                applied.append(f"{col}(bool)")
            elif pd.api.types.is_numeric_dtype(dtype):
                df = apply_numeric_filter(df, col)
                applied.append(f"{col}(num)")
            else:
                df = apply_text_filter(df, col)
                applied.append(f"{col}(txt)")

        elif choice == "r":
            confirm = input("  ⚠️  Reset? All filters will be lost. [y/n]: ").strip().lower()
            if confirm == "y":
                df = load_data(CSV_PATH)
                applied = []

        elif choice == "s":
            default_name = "best_candidate_filtered.csv"
            name = input(f"\n  Output file name [{default_name}]: ").strip()
            if not name:
                name = default_name
            if not name.endswith(".csv"):
                name += ".csv"
            out_path = os.path.join(OUTPUT_DIR, name)
            df.to_csv(out_path, index=False)
            print(f"\n  ✅ Filtered CSV saved: {out_path}")
            print(f"     {len(df)} rows  |  {len(df.columns)} columns")
            break

        elif choice == "q":
            print("\n  Exiting without saving.")
            break

        else:
            print("  ⚠️  Unrecognised choice.")

    return df


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    print("\n" + "═" * 60)
    print("   CSV Filter Tool — best_candidate")
    print("═" * 60)

    if not os.path.isfile(CSV_PATH):
        print(f"\n❌ File not found: {CSV_PATH}")
        print("   Check the path in the CSV_PATH variable at the top of the script.")
        sys.exit(1)

    df = load_data(CSV_PATH)
    filter_loop(df)


if __name__ == "__main__":
    main()