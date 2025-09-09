import json
from pathlib import Path
from datetime import datetime, timezone
import yaml
import pandas as pd
import numpy as np

# ---- Config ---------------------------------------------------------------
YAML_PATH   = Path("dir_struct.yaml")      
SOURCE_ROOT = Path("cleaned_data")        
OUT_ROOT    = Path("data_ML")              
# --------------------------------------------------------------------------

# Rolling windows (days)
ATR_WINDOWS   = (7, 14, 30)
VOL_WINDOWS   = (7, 14, 30)

# Intraday resample frequency for cross-outcome spread metrics
RESAMPLE_FREQ = None
# Fill strategy for missing daily trade metrics
TRADE_GAP_FILL      = "ffill_then_mean"  
TRADE_GAP_WINDOW    = 7                   
# --------------------------


def getmarkets_path(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        markets = yaml.safe_load(f)
    return list(markets.keys())


# ---- Parsing utils --------------------------------------------------------
def _iter_json_objects_from_text(s: str):
    """
    Works for either:
      • concatenated JSON objects with no separators, or
      • standard JSON lines (if you pass a single line).
    """
    dec = json.JSONDecoder()
    i, n = 0, len(s)
    while i < n:
        # skip whitespace
        while i < n and s[i].isspace():
            i += 1
        if i >= n:
            break
        obj, j = dec.raw_decode(s, i)
        yield obj
        i = j


def _iter_objects_from_file(path: Path, chunked: bool = False, chunk_size: int = 1 << 20):
    """
    Yields JSON objects from:
      • newline-delimited JSON (NDJSON), OR
      • concatenated JSON streams with no newlines.
    For huge files, set chunked=True to stream with bounded memory.
    """
    if not chunked:
        txt = path.read_text(encoding="utf-8", errors="replace")
        if "\n" in txt:
            # NDJSON or mixed: handle lines; each line might still contain >1 object
            for line in txt.splitlines():
                if not line.strip():
                    continue
                for obj in _iter_json_objects_from_text(line):
                    yield obj
        else:
            # pure concatenated stream
            yield from _iter_json_objects_from_text(txt)
        return

    # Streaming mode for very large files
    dec = json.JSONDecoder()
    buf = ""
    with path.open("r", encoding="utf-8", errors="replace") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            buf += chunk
            i = 0
            while True:
                # skip whitespace
                while i < len(buf) and buf[i].isspace():
                    i += 1
                if i >= len(buf):
                    buf = ""
                    break
                try:
                    obj, j = dec.raw_decode(buf, i)
                except json.JSONDecodeError:
                    # need more data
                    buf = buf[i:]
                    break
                else:
                    yield obj
                    i = j
            # keep tail for next chunk
            buf = buf[i:]
def read_holders(path_holder: Path) -> pd.DataFrame:
    """
    Returns a tidy DF with columns:
      time(UTC), date, conditionId, asset, outcome_index(Int64), address, amount(float)
    Works for files that are NDJSON *or* space-concatenated JSON objects.
    """
    rows = []
    for fp in path_holder.glob("*.ndjson"):
        txt = fp.read_text(encoding="utf-8", errors="replace")
        for obj in _iter_json_objects_from_text(txt):
            cid   = obj.get("conditionId")
            ts_ms = obj.get("capture_ts_ms")
            t     = pd.to_datetime(ts_ms, unit="ms", utc=True) if ts_ms is not None else pd.NaT
            holders = obj.get("holders") or []
            for h in holders:
                try:
                    amt = float(h.get("amount") or h.get("balance") or h.get("size") or 0.0)
                except Exception:
                    amt = np.nan
                rows.append({
                    "conditionId": cid,
                    "time": t,
                    "asset": h.get("asset") or h.get("token") or h.get("token_id") or obj.get("token_id"),
                    "outcome_index": h.get("outcomeIndex") if h.get("outcomeIndex") is not None else h.get("outcome_index"),
                    "address": h.get("proxyWallet") or h.get("address") or h.get("holder") or h.get("wallet"),
                    "amount": amt,
                })
    df = pd.DataFrame(rows)
    if df.empty: 
        return df
    df["time"]   = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df["date"]   = df["time"].dt.date
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df["outcome_index"] = pd.to_numeric(df["outcome_index"], errors="coerce").astype("Int64")
    return df.dropna(subset=["date", "outcome_index", "address"])

def read_price_records(path_price: Path) -> list[list]:
    """
    Returns rows: [price(float), time(datetime, UTC), outcome_index(int)]
    Accepts files that are NDJSON or concatenated JSON streams.
    """
    rows = []
    for fp in path_price.glob("*.ndjson"):
        for row in _iter_objects_from_file(fp):

            try:
                p = float(row["p"])
                ts = float(row["t"])
          
                if ts > 1e12:
                    ts /= 1000.0
                t = datetime.fromtimestamp(ts, tz=timezone.utc)
                oi = int(row["outcome_index"])
            except Exception:
                continue
            rows.append([p, t, oi])
    return rows
# ---- concentration helpers ----
def _gini(x: np.ndarray) -> float:
    """Gini coefficient for a non-negative vector x."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    x[x < 0] = 0.0
    if x.size == 0:
        return np.nan
    s = x.sum()
    if s <= 0:
        return 0.0
    x.sort()
    n = x.size
    cumx = np.cumsum(x)
    # Equivalent formula: 1 - 2 * sum((n+1-i)*x_i)/(n*sum x)
    return float((n + 1 - 2.0 * (cumx / s).sum()) / n)


def _conc_from_amounts(by_wallet: pd.Series) -> dict:
    """by_wallet: index=address, values=amount at one snapshot."""
    # Robust numeric array
    x = pd.to_numeric(by_wallet, errors="coerce").to_numpy(dtype=float)
    x[~np.isfinite(x)] = 0.0
    x[x < 0] = 0.0

    s = x.sum()
    if x.size == 0 or s <= 0.0:
        return {
            "holders_count": 0, "supply": 0.0,
            "hhi": np.nan, "effective_holders": np.nan,
            "top10_share": np.nan, "top1_share": np.nan, "gini": np.nan,
            "avg_balance": np.nan, "median_balance": np.nan,
        }

    shares = x / s
    shares.sort()  # ascending
    hhi = float(np.dot(shares, shares))
    return {
        "holders_count": int(x.size),
        "supply": float(s),
        "hhi": hhi,
        "effective_holders": (1.0 / hhi) if hhi > 0 else np.nan,
        "top10_share": float(shares[-10:].sum()) if shares.size else np.nan,
        "top1_share": float(shares[-1]) if shares.size else np.nan,
        "gini": _gini(x),
        "avg_balance": float(x.mean()),
        "median_balance": float(np.median(x)),
    }

# ---- per-snapshot concentration (time, outcome) ----
def snapshot_concentration_metrics(holders_df: pd.DataFrame) -> pd.DataFrame:
    """
    Input: tidy DF from read_holders() with columns ['time','date','outcome_index','address','amount'].
    Output: one row per (time, outcome) with concentration metrics.
    """
    df = holders_df.dropna(subset=["time","outcome_index","address","amount"]).copy()
    df["outcome"] = df["outcome_index"].astype(int)

    rows = []
    for (t, oi), grp in df.groupby(["time","outcome"], sort=True):
        by_wallet = grp.groupby("address")["amount"].sum()
        m = _conc_from_amounts(by_wallet)
        m.update({"time": t, "date": t.date(), "outcome": int(oi)})
        rows.append(m)

    snap = pd.DataFrame(rows).sort_values(["date","outcome","time"]).reset_index(drop=True)
    return snap

# ---- daily average of snapshot metrics (mean or time-weighted) ----
def daily_avg_concentration(snap_df: pd.DataFrame, method: str = "time_weighted") -> pd.DataFrame:
    """
    method: 'mean' (simple average of snapshots that day) or 'time_weighted'
            (weight each snapshot by seconds until the next snapshot or day-end).
    """
    if snap_df.empty:
        return pd.DataFrame(columns=["date","outcome","holders_count","supply","hhi",
                                     "effective_holders","top10_share","top1_share",
                                     "gini","avg_balance","median_balance"])

    metr = ["holders_count","supply","hhi","effective_holders",
            "top10_share","top1_share","gini","avg_balance","median_balance"]

    if method == "mean":
        out = (snap_df.groupby(["date","outcome"], as_index=False)[metr]
                      .mean(numeric_only=True))
        return out

    # time-weighted
    out_rows = []
    for (d, oi), grp in snap_df.groupby(["date","outcome"]):
        grp = grp.sort_values("time").copy()
        # day start/end in the same tz as 'time'
        tz = grp["time"].dt.tz
        day_start = pd.Timestamp(d).tz_localize(tz)
        day_end   = day_start + pd.Timedelta(days=1)

        times = grp["time"].to_list()
        # durations: from snapshot i -> next snapshot (or day_end for last)
        durs = []
        for i, t in enumerate(times):
            t_next = times[i+1] if i+1 < len(times) else day_end
            dur = (t_next - t).total_seconds()
            durs.append(max(dur, 0.0))

        w = np.array(durs, dtype=float)
        W = w.sum()
        if W <= 0:
            vals = grp[metr].mean(numeric_only=True).to_dict()
        else:
            M = grp[metr].to_numpy(dtype=float)
            vals = dict(zip(metr, (M * w[:, None]).sum(axis=0) / W))

        out_rows.append({"date": d, "outcome": int(oi), **vals})

    return pd.DataFrame(out_rows).sort_values(["date","outcome"]).reset_index(drop=True)

def token_outcome_map_from_trades(path_trades: Path) -> dict[str, int]:
    m = {}
    for fp in path_trades.glob("*.ndjson"):
        for r in _iter_objects_from_file(fp):
            try:
                tok = str(r.get("asset") or r.get("token") or r.get("token_id"))
                oi  = int(r.get("outcomeIndex"))
            except Exception:
                continue
            if tok and oi in (0,1):   
                m[tok] = oi
    return m



def read_trades(path_trades: Path) -> pd.DataFrame:
    """Return a tidy DataFrame with time(UTC), side, price, size, is_yes, date, notional."""
    rows = []
    for fp in path_trades.glob("*.ndjson"):
        for r in _iter_objects_from_file(fp):
            try:
                ts = float(r["timestamp"])
                if ts > 1e12: ts /= 1000.0  
                t  = pd.to_datetime(ts, unit="s", utc=True)
                side = str(r["side"]).upper()
                price = float(r["price"])
                size  = float(r["size"])
                
                outcome = str(r.get("outcome","")).lower()
                oi = int(r.get("outcomeIndex", -1))
                is_yes = (outcome == "yes") or (oi == 1)  
               
            except Exception:
                continue
            rows.append((t, side, price, size, is_yes))
    df = pd.DataFrame(rows, columns=["time","side","price","size","is_yes"])
    if df.empty: return df
    eps = 1e-12
    df["price"] = df["price"].clip(eps, 1 - eps)
    df["date"]  = df["time"].dt.date
    df["notional_quote"] = df["price"] * df["size"]
    df["side_sign"] = np.where(df["side"] == "BUY", 1, -1)
    # YES-direction sign: Yes BUY +, Yes SELL -, No BUY -, No SELL +
    df["yes_dir_sign"] = np.where(df["is_yes"], df["side_sign"], -df["side_sign"])
    df["yes_dir_quote"] = df["yes_dir_sign"] * df["notional_quote"]
    df["yes_dir_base"]  = df["yes_dir_sign"] * df["size"]
    # Split columns for easy groupby sums
    df["notional_yes"] = np.where(df["is_yes"], df["notional_quote"], 0.0)
    df["notional_no"]  = np.where(~df["is_yes"], df["notional_quote"], 0.0)
    df["base_yes"]     = np.where(df["is_yes"], df["size"], 0.0)
    df["base_no"]      = np.where(~df["is_yes"], df["size"], 0.0)
    df["buy_notional"] = np.where(df["side"]=="BUY",  df["notional_quote"], 0.0)
    df["sell_notional"]= np.where(df["side"]=="SELL", df["notional_quote"], 0.0)
    return df

def daily_trade_metrics_by_outcome(trades: pd.DataFrame, whale_q: float = 0.9) -> pd.DataFrame:
    """
    Return one row per (date, outcome) with outcome-specific volumes/flows.
    outcome = 1 for YES, 0 for NO.
    """
    if trades.empty:
        return pd.DataFrame(columns=[
            "date","outcome",
            "trades_count","vol_quote","vol_base",
            "buy_vol_quote","sell_vol_quote",
            "flow_imbalance",
            "avg_trade_size_quote","median_trade_size_quote",
            "yes_dir_flow_quote","yes_dir_flow_base",
            "whale_flow_top10p_quote"
        ])

    df = trades.copy()
    df["outcome"] = np.where(df["is_yes"], 1, 0)
    df["notional_quote"] = df["price"] * df["size"]

    def _one_group(g):
        buy_q  = float(g.loc[g["side"]=="BUY",  "notional_quote"].sum())
        sell_q = float(g.loc[g["side"]=="SELL", "notional_quote"].sum())
        vol_q  = float(g["notional_quote"].sum())
        vol_b  = float(g["size"].sum())
        ntr    = int(len(g))

        # outcome-directional flow (BUY=+1, SELL=-1) within this outcome
        sign = np.where(g["side"]=="BUY", 1.0, -1.0)
        dir_flow_q = float((sign * g["notional_quote"]).sum())
        dir_flow_b = float((sign * g["size"]).sum())

        # flow imbalance for THIS outcome only
        denom = buy_q + sell_q
        fi = (buy_q - sell_q) / denom if denom > 0 else 0.0

        # whale flow: top 10% by notional within THIS outcome-day
        thr = g["notional_quote"].quantile(whale_q) if not g.empty else np.nan
        whale_q_sum = float((sign * g["notional_quote"])[g["notional_quote"] >= thr].sum()) if not np.isnan(thr) else 0.0

        return pd.Series({
            "trades_count": ntr,
            "vol_quote": vol_q,
            "vol_base": vol_b,
            "buy_vol_quote": buy_q,
            "sell_vol_quote": sell_q,
            "flow_imbalance": fi,
            "avg_trade_size_quote": float(g["notional_quote"].mean()) if ntr>0 else 0.0,
            "median_trade_size_quote": float(g["notional_quote"].median()) if ntr>0 else 0.0,
            # Keep "yes-directional" strictly on outcome==1; set 0 on outcome==0
            "yes_dir_flow_quote": dir_flow_q if int(g["outcome"].iloc[0])==1 else 0.0,
            "yes_dir_flow_base":  dir_flow_b if int(g["outcome"].iloc[0])==1 else 0.0,
            "whale_flow_top10p_quote": whale_q_sum,  # per outcome
        })

    out = (
        df.groupby(["date","outcome"], as_index=False, group_keys=False)
          .apply(_one_group)
          .reset_index()
          .drop(columns=["index"], errors="ignore")
    )
    return out.sort_values(["date","outcome"]).reset_index(drop=True)


def fill_trade_gaps_grouped(df: pd.DataFrame,
                            method: str = "ffill_then_mean",
                            window: int = 7) -> pd.DataFrame:
    trade_cols = [
        "trades_count","vol_quote","vol_base",
        "buy_vol_quote","sell_vol_quote",
        "flow_imbalance","avg_trade_size_quote","median_trade_size_quote",
        "yes_dir_flow_quote","yes_dir_flow_base","whale_flow_top10p_quote",
       
    ]
    cols = [c for c in trade_cols if c in df.columns]
    if not cols or df.empty:
        return df

    out = df.sort_values(["outcome","date"]).copy()
    g = out.groupby("outcome", group_keys=False)

    if method == "ffill":
        out[cols] = g[cols].ffill()
    elif method == "mean":
        roll = g[cols].apply(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
        roll.index = out.index
        out[cols] = out[cols].fillna(roll)
    else:  # ffill_then_mean
        tmp = g[cols].ffill()
        roll = g[cols].apply(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
        roll.index = out.index
        out[cols] = tmp.fillna(roll)


    out[cols] = out[cols].fillna(0.0)
    return out




def yes_no_panel(records):
    """
    records: [[price, time(tz-aware UTC), outcome_index], ...]
    outcome_index: 1=YES, 0=NO
    returns a DataFrame at native (~3H) timestamps with p_yes, p_no and spread metrics
    """
    df = pd.DataFrame(records, columns=["price","time","outcome"]).copy()
    df = df.sort_values("time")

    wide = (df.pivot_table(index="time", columns="outcome", values="price", aggfunc="last")
              .sort_index()
              .rename(columns={0:"p_no", 1:"p_yes"}))

    wide = wide.ffill()
    wide = wide.dropna(subset=["p_yes","p_no"])

    eps = 1e-12
    wide["p_yes"] = wide["p_yes"].clip(eps, 1-eps)
    wide["p_no"]  = wide["p_no"].clip(eps, 1-eps)

    wide["market_sum"] = wide["p_yes"] + wide["p_no"]          # ≈1 in a tight market
    wide["market_gap"] = (wide["market_sum"] - 1.0).abs()
    wide["mid_price"]  = 0.5 * (wide["p_yes"] + (1.0 - wide["p_no"]))
    wide["arb_slack"]  = 1.0 - wide["market_sum"]              # >0 means combined < 1
    return wide



def _cross_outcome_spread_intraday(df: pd.DataFrame, freq: str | None = None) -> pd.DataFrame:
    """
    Compute per-day cross-outcome YES+NO sum metrics on your native (~3H) timestamps.
    Assumes 'df' has: price, time (tz-aware), outcome (0/1), date.
    """
    out_rows = []
    for d, day in df.groupby("date"):
        wide = (day.pivot_table(index="time", columns="outcome", values="price", aggfunc="last")
                  .sort_index()
                  .rename(columns={0:"p_no", 1:"p_yes"})).ffill()

        # need both sides to compute sums
        if not {"p_yes","p_no"}.issubset(wide.columns) or wide[["p_yes","p_no"]].dropna().empty:
            out_rows.append({
                "date": d,
                "market_sum_close": np.nan,
                "market_gap_close": np.nan,
                "market_sum_min_native": np.nan,
                "market_sum_max_native": np.nan,
                "market_gap_maxabs_native": np.nan,
                "market_gap_mean_native": np.nan,
            })
            continue

        s_sum = (wide["p_yes"] + wide["p_no"]).dropna()

        close_sum   = float(wide["p_yes"].iloc[-1] + wide["p_no"].iloc[-1])
        close_gap   = close_sum - 1.0
        min_sum     = float(s_sum.min())
        max_sum     = float(s_sum.max())
        max_abs_gap = float(np.abs(s_sum - 1.0).max())
        mean_gap    = float((s_sum - 1.0).mean())

        out_rows.append({
            "date": d,
            "market_sum_close": close_sum,
            "market_gap_close": close_gap,
            "market_sum_min_native": min_sum,
            "market_sum_max_native": max_sum,
            "market_gap_maxabs_native": max_abs_gap,
            "market_gap_mean_native": mean_gap,
        })

    return pd.DataFrame(out_rows)



def daily_metrics(records: list[list]) -> pd.DataFrame:
    """
    Per-day, per-outcome OHLC + derived metrics, plus:
      • rolling volatility of p_close & logit_close (windows in VOL_WINDOWS)
      • ATR_n (SMA over True Range; windows in ATR_WINDOWS) and ATR_n relative
      • cross-outcome spread/inefficiency columns duplicated on each outcome row
    """
    base_cols = [
        "date", "outcome",
        "p_open", "p_high", "p_low", "p_close", "count",
        "odds_close", "logit_close",
        "range_abs", "range_rel",
        "parkinson_var",
    ]
    parkinson_fill = 0.0


    if not records:
        return pd.DataFrame(columns=base_cols)

    df = pd.DataFrame(records, columns=["price", "time", "outcome"])
    eps = 1e-12
    df["price"] = pd.to_numeric(df["price"], errors="coerce").clip(eps, 1 - eps)
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df.dropna(subset=["price", "time"])
    df["date"] = df["time"].dt.date

    # ----- Outcome-level daily OHLC -----
    df = df.sort_values(["outcome", "date", "time"])
    agg = df.groupby(["outcome", "date"], as_index=False).agg(
        p_open=("price", "first"),
        p_high=("price", "max"),
        p_low =("price", "min"),
        p_close=("price", "last"),
        count =("price", "size"),
    )

    p  = agg["p_close"].to_numpy()
    hi = agg["p_high"].to_numpy()
    lo = agg["p_low"].to_numpy()
    n  = agg["count"].to_numpy()

    agg["odds_close"]  = p / (1.0 - p)
    agg["logit_close"] = np.log(p) - np.log1p(-p)     # log(p/(1-p))
    agg["range_abs"]   = hi - lo
    agg["range_rel"]   = (hi - lo) / p

    with np.errstate(divide="ignore", invalid="ignore"):
        pv = (np.log(hi / lo) ** 2) / (4.0 * np.log(2.0))
    pv[(n < 2) | (hi <= lo)] = np.nan
    agg["parkinson_var"] = pv
    if parkinson_fill is not None:
        agg["parkinson_var"] = agg["parkinson_var"].fillna(parkinson_fill)

    # ----- True Range & ATR (per outcome) -----
    # TR_t = max( high-low, |high - prev_close|, |low - prev_close| )
    agg = agg.sort_values(["outcome", "date"]).reset_index(drop=True)
    prev_close = agg.groupby("outcome")["p_close"].shift(1)

    tr1 = agg["p_high"] - agg["p_low"]
    tr2 = (agg["p_high"] - prev_close).abs()
    tr3 = (agg["p_low"]  - prev_close).abs()
    agg["tr"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1, skipna=True)

    # ATR_n (simple moving average) and relative ATR
    g = agg.groupby("outcome", group_keys=False)
    for nwin in ATR_WINDOWS:
        agg[f"atr_{nwin}"] = g["tr"].apply(lambda s: s.rolling(nwin, min_periods=1).mean())
        agg[f"atr_rel_{nwin}"] = agg[f"atr_{nwin}"] / agg["p_close"]

    # ----- Rolling volatility (per outcome) -----
    for nwin in VOL_WINDOWS:
        agg[f"roll_std_p_{nwin}"] = g["p_close"].apply(
            lambda s: s.rolling(nwin, min_periods=2).std(ddof=0)
        )
        agg[f"roll_std_logit_{nwin}"] = g["logit_close"].apply(
            lambda s: s.rolling(nwin, min_periods=2).std(ddof=0)
        )

    # ----- Cross-outcome daily spread/inefficiency (YES+NO) from intraday -----
    cross = _cross_outcome_spread_intraday(df[["price", "time", "outcome", "date"]].copy(),
                                           freq=RESAMPLE_FREQ)
    agg = agg.merge(cross, on="date", how="left")

    cross_cols = [c for c in agg.columns if c.startswith("market_")]
    empty_cols = [c for c in cross_cols if agg[c].isna().all()]
    if empty_cols:
        agg = agg.drop(columns=empty_cols)

    # Final ordering
    ocols = [
        "date", "outcome",
        "p_open", "p_high", "p_low", "p_close", "count",
        "odds_close", "logit_close",
        "range_abs", "range_rel", "parkinson_var",
        "tr",
        *[f"atr_{n}" for n in ATR_WINDOWS],
        *[f"atr_rel_{n}" for n in ATR_WINDOWS],
        *[f"roll_std_p_{n}" for n in VOL_WINDOWS],
        *[f"roll_std_logit_{n}" for n in VOL_WINDOWS],
        "market_sum_close", "market_gap_close",
        "market_sum_min_native","market_sum_max_native",
         "market_gap_maxabs_native","market_gap_mean_native",
       
    ]
    existing_cols = [c for c in ocols if c in agg.columns]
    return agg[existing_cols].sort_values(["date", "outcome"]).reset_index(drop=True)



def write_daily_csvs(market_key: str, metrics_df: pd.DataFrame) -> None:
    """
    Writes one CSV per date under OUT_ROOT/<market_key>/<YYYY-MM-DD>.csv.
    Each CSV contains both outcomes' rows for that day.
    """
    out_dir = OUT_ROOT / market_key
    out_dir.mkdir(parents=True, exist_ok=True)

    if metrics_df.empty:
        print(f"[skip] No usable price data in: {SOURCE_ROOT}/{market_key}")
        return

    for d, day_df in metrics_df.groupby("date"):
        out_fp = out_dir / f"{d}.csv"
        day_df.to_csv(out_fp, index=False)
        print(f"[write] {market_key}  {d}  rows={len(day_df)}  -> {out_fp}")


# ---- Driver ---------------------------------------------------------------
def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    market_keys = getmarkets_path(YAML_PATH)

    for key in market_keys:
        price_path = SOURCE_ROOT / key / "price"
        book_path = SOURCE_ROOT / key / "book"
        holder_path = SOURCE_ROOT / key / "holder"
        trade_path = SOURCE_ROOT / key / "trade"
        if not price_path.exists():
            print(f"[skip] Missing price dir: {price_path}")
            continue

        records = read_price_records(price_path)
        metrics  = daily_metrics(records)
        tdf = read_trades(trade_path)
        trade_daily = daily_trade_metrics_by_outcome(tdf)
        metrics = metrics.merge(trade_daily,on = ["date","outcome"],how = "left")
        metrics = fill_trade_gaps_grouped(metrics,method=TRADE_GAP_FILL,window=TRADE_GAP_WINDOW)
        hdf = read_holders(holder_path)
        tokmap = token_outcome_map_from_trades(trade_path)
        if tokmap:
            hdf["outcome_index"] = hdf["outcome_index"].fillna(
            hdf["asset"].map(tokmap)
            )
            hdf = hdf.dropna(subset=["outcome_index"])
            hdf["outcome_index"] = hdf["outcome_index"].astype(int)
        
        snap = snapshot_concentration_metrics(hdf)
        conc_daily = daily_avg_concentration(snap,method="time_weighted")

        metrics = metrics.merge(conc_daily,on=["date","outcome"],how="left")
        write_daily_csvs(key,metrics)



