import json
import os
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Optional, Any, Tuple
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

# =========================
# Config knobs (edit here)
# =========================
MARKET_LIMIT           = 100000  # how many markets to pull (total across pages)
PRICE_INTERVAL         = "max"    # or '1d','6h','1h','1m'
PRICE_FIDELITY_MIN     = 180      # minutes per candle (e.g., 180 = 3h)
BOOK_CONCURRENCY       = 8        # threads for bulk book fetch
INCLUDE_TRADES         = True     # keep attaching a small sample of trades to JSON
TRADES_LIMIT_PER_BATCH = 500
TRADES_TAKER_ONLY      = True     # taker-only for the summary list

# ========= NDJSON output switches =========
WRITE_NDJSON          = True                   # master switch for NDJSON
WRITE_AGGREGATE_JSON = False                  # set False to avoid huge in-memory JSON
DATA_DIR             = "data_2024/raw"             # base folder for NDJSON
NDJSON_PRICES        = True
NDJSON_BOOKS         = True
NDJSON_TRADES        = True                   # FULL trade dump (paged) to NDJSON
NDJSON_HOLDERS       = True                   # market- and asset-level holders to NDJSON
NDJSON_MARKETS       = True                   # minimal per-market metadata

# --- Endpoints ---
market_url       = "https://gamma-api.polymarket.com/markets"
price_url        = "https://clob.polymarket.com/prices-history"
DATA_TRADES_URL  = "https://data-api.polymarket.com/trades"
CLOB_BOOK_URL    = "https://clob.polymarket.com/book"
DATA_HOLDERS_URL = "https://data-api.polymarket.com/holders"

# --- HTTP session ---
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "PolymarketIngest/0.5"})

# =========================
# Helpers
# =========================
def _to_float_or_none(x):
    try:
        return float(x)
    except Exception:
        return None


def _parse_jsonish(val):
    if isinstance(val, str):
        try:
            return json.loads(val)
        except json.JSONDecodeError:
            return []
    return val or []


def _is_valid_condition_id(val: Optional[str]) -> bool:
    return isinstance(val, str) and val.startswith("0x") and len(val) == 66


def _safe_int_ms(x) -> Optional[int]:
    """Best-effort convert to milliseconds int; None if unknown."""
    if x is None:
        return None
    try:
        if isinstance(x, (int, float)):
            return int(x)
        return int(str(x).strip())
    except Exception:
        return None

# ---- NDJSON helpers ----
def _ensure_dir(p: str) -> None:
    Path(p).parent.mkdir(parents=True, exist_ok=True)


def _append_ndjson(path: str, obj: dict) -> None:
    if not WRITE_NDJSON:
        return
    _ensure_dir(path)
    with open(path, "a", encoding="utf-8") as fp:
        fp.write(json.dumps(obj) + " ")


def _utc_ms_now() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


# =========================
# API calls
# =========================
def _iter_markets(limit_total: int, **filters):
    """
    Yields market dicts across many pages until limit_total is reached.
    Pass filters like active=None to get all, start_date_min="2020-01-01T00:00:00Z", etc.
    """
    page_size = min(200, int(filters.pop("limit", 200) or 200))
    fetched, offset = 0, 0
    while fetched < limit_total:
        params = {"limit": page_size, "offset": offset, **filters}
        r = SESSION.get(market_url, params=params, timeout=10)
        r.raise_for_status()
        batch = r.json() or []
        if not batch:
            break
        for m in batch:
            yield m
            fetched += 1
            if fetched >= limit_total:
                break
        if len(batch) < page_size:
            break
        offset += len(batch)


def _getPrice_(token_id: str) -> List[Dict]:
    params = {"market": token_id, "fidelity": PRICE_FIDELITY_MIN, "interval": PRICE_INTERVAL}
    r = SESSION.get(price_url, params=params, timeout=10)
    r.raise_for_status()
    data = r.json() or {}
    return data.get("history", []) or []


def _get_trades_many(condition_ids: List[str], *, limit: int = TRADES_LIMIT_PER_BATCH, taker_only: bool = True) -> List[dict]:
    ids = [cid for cid in condition_ids if _is_valid_condition_id(cid)]
    if not ids:
        return []
    r = SESSION.get(
        DATA_TRADES_URL,
        params={"market": ",".join(ids), "limit": limit, "takerOnly": taker_only},
        timeout=10,
    )
    r.raise_for_status()
    return r.json() or []


def _get_trades_paged(condition_id: str, *, taker_only: bool = False, page_limit: int = 500):
    """Generator yielding ALL trades for one market, paged by offset."""
    if not _is_valid_condition_id(condition_id):
        return
    offset = 0
    while True:
        r = SESSION.get(
            DATA_TRADES_URL,
            params={
                "market": condition_id,
                "takerOnly": taker_only,
                "limit": page_limit,
                "offset": offset,
                
            },
            timeout=10,
        )
        r.raise_for_status()
        batch = r.json() or []
        if not batch:
            break
        for tr in batch:
            yield tr
        if len(batch) < page_limit:
            break
        offset += page_limit


def _get_order_book(token_id: str) -> dict:
    r = SESSION.get(CLOB_BOOK_URL, params={"token_id": token_id}, timeout=30)
    r.raise_for_status()
    return r.json() or {}


# ---------- bulk order-books ----------
def _get_books_bulk(token_ids: List[str], *, concurrency: int = BOOK_CONCURRENCY) -> Dict[str, dict]:
    """Fetch many books concurrently. Returns token_id -> raw book dict (or {})."""
    out: Dict[str, dict] = {}
    unique = list({tid for tid in token_ids if tid})
    if not unique:
        return out

    def worker(tid):
        try:
            return tid, _get_order_book(tid)
        except requests.RequestException:
            return tid, {}

    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = [ex.submit(worker, tid) for tid in unique]
        for fut in as_completed(futures):
            tid, book = fut.result()
            out[tid] = book
    return out


# ---------- book summarizer ----------
def _first_price(levels):
    for L in levels:
        if not isinstance(L, dict):
            continue
        val = L.get("price") or L.get("p")  # support both shapes
        try:
            if val is not None:
                return float(val)
        except Exception:
            pass
    return None


def _level_size(levels):
    if not levels:
        return 0.0
    L0 = levels[0] if isinstance(levels[0], dict) else {}
    val = L0.get("size") or L0.get("s")
    try:
        return float(val) if val is not None else 0.0
    except Exception:
        return 0.0


def _summarize_book(book: dict) -> dict:
    bids = book.get("bids") or []
    asks = book.get("asks") or []
    best_bid = _first_price(bids)
    best_ask = _first_price(asks)
    bid_depth = _level_size(bids)
    ask_depth = _level_size(asks)

    mid = spread_bps = None
    if best_bid is not None and best_ask is not None and best_ask > 0:
        mid = 0.5 * (best_bid + best_ask)
        spread_bps = ((best_ask - best_bid) / best_ask) * 10_000.0

    ts_ms = _safe_int_ms(book.get("timestamp"))

    imbalance = None
    denom = (bid_depth or 0.0) + (ask_depth or 0.0)
    if denom > 0:
        imbalance = (bid_depth - ask_depth) / denom

    return {
        "best_bid": best_bid,
        "best_ask": best_ask,
        "mid": mid,
        "spread_bps": spread_bps,
        "bid_depth": bid_depth,
        "ask_depth": ask_depth,
        "imbalance": imbalance,
        "timestamp_ms": ts_ms,
    }


# ---------- market summary from per-outcome books ----------
def _safe_median(vals: List[Optional[float]]) -> Optional[float]:
    xs = [v for v in vals if isinstance(v, (int, float))]
    if not xs:
        return None
    xs.sort()
    n = len(xs)
    mid = n // 2
    if n % 2:
        return float(xs[mid])
    return float((xs[mid - 1] + xs[mid]) / 2.0)


def _build_market_summary(books_by_idx: Dict[str, Optional[dict]]) -> Optional[dict]:
    if not books_by_idx:
        return None
    best_bids = []
    best_asks = []
    mids      = []
    spreads   = []
    bid_depth_sum = 0.0
    ask_depth_sum = 0.0
    ts_list   = []

    for b in books_by_idx.values():
        if not b:
            continue
        best_bids.append(b.get("best_bid"))
        best_asks.append(b.get("best_ask"))
        mids.append(b.get("mid"))
        spreads.append(b.get("spread_bps"))
        bid_depth_sum += float(b.get("bid_depth") or 0.0)
        ask_depth_sum += float(b.get("ask_depth") or 0.0)
        t = _safe_int_ms(b.get("timestamp_ms"))
        if t is not None:
            ts_list.append(t)

    if not (best_bids or best_asks or mids or spreads or bid_depth_sum or ask_depth_sum):
        return None

    return {
        "best_bid_median": _safe_median(best_bids),
        "best_ask_median": _safe_median(best_asks),
        "mid_median": _safe_median(mids),
        "spread_bps_median": _safe_median(spreads),
        "total_bid_depth": bid_depth_sum,
        "total_ask_depth": ask_depth_sum,
        "book_timestamp_ms_max": max(ts_list) if ts_list else None,
    }


# ---------- Open Interest via holders ----------
_SUPPLY_CACHE: Dict[str, float] = {}  # token_id -> summed supply


def _sum_token_supply(token_id: str, *, page_limit: int = 1000) -> float:
    """Sum circulating supply for a token by paging holders (asset=<token_id>)."""
    if not token_id:
        return 0.0
    if token_id in _SUPPLY_CACHE:
        return _SUPPLY_CACHE[token_id]

    total = 0.0
    offset = 0
    while True:
        try:
            r = SESSION.get(
                DATA_HOLDERS_URL,
                params={"asset": token_id, "limit": page_limit, "offset": offset},
                timeout=10,
            )
            r.raise_for_status()
            payload = r.json() or {}
        except requests.RequestException:
            break

        if isinstance(payload, dict):
            holders = payload.get("holders") or []
        elif isinstance(payload, list):
            holders = payload
        else:
            holders = []

        got = 0
        for h in holders:
            if isinstance(h, dict):
                v = _to_float_or_none(h.get("amount") or h.get("balance") or h.get("size"))
                if v is not None:
                    total += v
                    got += 1
        if got < page_limit:
            break
        offset += page_limit

    _SUPPLY_CACHE[token_id] = total
    return total


def _estimate_market_open_interest(condition_id: Optional[str], outcome_specs: List[dict]) -> Optional[float]:
    """
    Estimate open interest:
      1) Try market-level holders (market=<condition_id>) -> supplies per outcome.
      2) Fallback: sum asset-level holders per token (asset=<token_id>).
      Binary: min(YES, NO); multi-outcome: sum/2 to avoid double counting.
    """
    supplies: List[float] = []

    # 1) market-level holders
    if condition_id and _is_valid_condition_id(condition_id):
        try:
            r = SESSION.get(DATA_HOLDERS_URL, params={"market": condition_id, "limit": 500}, timeout=30)
            r.raise_for_status()
            payload = r.json() or []
            if isinstance(payload, list):
                for token_entry in payload:
                    if isinstance(token_entry, dict):
                        holders = token_entry.get("holders") or []
                        s = 0.0
                        for h in holders:
                            if isinstance(h, dict):
                                v = _to_float_or_none(h.get("amount") or h.get("balance") or h.get("size"))
                                if v is not None:
                                    s += v
                        if s > 0:
                            supplies.append(s)
        except requests.RequestException:
            pass

    # 2) Fallback: asset-level per outcome
    if not supplies:
        for spec in outcome_specs:
            tok = spec.get("token_id")
            if not tok:
                continue
            try:
                s = _sum_token_supply(tok)
                if s > 0:
                    supplies.append(s)
            except requests.RequestException:
                pass

    if not supplies:
        return None
    return min(supplies) if len(supplies) == 2 else (sum(supplies) / 2.0)


# ---------- concentration metrics (HHI & top-10 share) ----------
def _normalize_holders(obj: Any) -> List[Dict[str, Any]]:
    """Coerce various API shapes into List[Dict[str, Any]] for type-checkers and runtime."""
    if isinstance(obj, dict) and "holders" in obj:
        obj = obj.get("holders")
    if isinstance(obj, list):
        return [h for h in obj if isinstance(h, dict)]
    return []


def _hhi_from_holders(holders: List[Dict[str, Any]]) -> Tuple[Optional[float], Optional[float]]:
    """
    holders: list like [{"amount": "...", ...}, ...]
    Returns (hhi, top10_share) in [0,1]. None if not computable.
    """
    qty: List[float] = []
    for h in holders:
        v = _to_float_or_none(h.get("amount") or h.get("balance") or h.get("size"))
        if v and v > 0:
            qty.append(v)
    s = sum(qty)
    if s <= 0:
        return None, None
    shares = [q / s for q in qty]
    hhi = sum(x * x for x in shares)
    top10 = sum(sorted(shares, reverse=True)[:10])
    return hhi, top10


# ---------- market-holders iterator for NDJSON ----------
def _iter_market_holders(condition_id: str, *, page_limit: int = 500):
    """
    Yields dicts like {"token": "...", "holders": [...] } for a market.
    Works whether the API is paged or returns a single full list.
    """
    if not condition_id or not _is_valid_condition_id(condition_id):
        return
    offset = 0
    while True:
        try:
            r = SESSION.get(
                DATA_HOLDERS_URL,
                params={"market": condition_id, "limit": page_limit, "offset": offset},
                timeout=10,
            )
            r.raise_for_status()
            payload = r.json() or []
        except requests.RequestException:
            return

        if isinstance(payload, list):
            if not payload:
                return
            for entry in payload:
                if isinstance(entry, dict):
                    yield entry
            if len(payload) < page_limit:
                return
            offset += page_limit
        else:
            if isinstance(payload, dict):
                yield payload
            return


# =========================
# Main assembly
# =========================
def get_structured_market_data() -> Dict:
    kept_count = 0
    structured_markets: List[Dict] = []
    kept_condition_ids: List[str] = []

    # Stream markets across pages
    markets = _iter_markets(
        MARKET_LIMIT,
        ascending=True,
        order="volume",
        start_date_max="2025-01-01T00:00:00Z",
        limit=200,
    )

    # Build rows we'll actually keep (require price data)
    tmp_rows = []  # (m, outcome_specs, prices)
    for m in markets:
        cond_id = m.get("conditionId") or m.get("condition_id")
        token_ids = _parse_jsonish(m.get("clobTokenIds"))
        outcomes  = _parse_jsonish(m.get("outcomes"))
        if not outcomes or not token_ids:
            continue

        L = min(len(outcomes), len(token_ids))
        if L == 0:
            continue

        outcome_specs = [{"index": i, "name": outcomes[i], "token_id": token_ids[i]} for i in range(L)]

        # prices (require at least some to keep the market)
        prices = []
        seen = set()
        for i, token_id in enumerate(token_ids[:L]):
            if not token_id:
                continue
            try:
                hist = _getPrice_(token_id)
            except requests.RequestException:
                hist = []
            if not hist:
                continue
            for pt in hist:
                t = pt.get("t"); p = pt.get("p")
                if t is None or p is None:
                    continue
                key = (token_id, int(t), round(float(p), 8))
                if key in seen:
                    continue
                seen.add(key)
                row = {"t": int(t), "p": float(p), "outcome_index": i}
                prices.append(row)

                # NDJSON: per-price point (token-level)
                if WRITE_NDJSON and NDJSON_PRICES:
                    _append_ndjson(
                        f"{DATA_DIR}/prices/token={token_id}.ndjson",
                        {
                            "token_id": token_id,
                            "conditionId": cond_id,
                            "market_id": m.get("id"),
                            "outcome_index": i,
                            "t": int(t),
                            "p": float(p),
                        },
                    )

        if not prices:
            continue

        tmp_rows.append((m, outcome_specs, prices))

        # NDJSON: a minimal per-market metadata row as soon as we decide to keep it
        if WRITE_NDJSON and NDJSON_MARKETS and cond_id:
            _append_ndjson(
                f"{DATA_DIR}/markets/kept_markets.ndjson",
                {
                    "conditionId": cond_id,
                    "market_id": m.get("id"),
                    "question": m.get("question"),
                    "slug": m.get("slug"),
                    "active": m.get("active"),
                    "startDate": m.get("start_date") or m.get("startDate"),
                    "endDate": m.get("end_date") or m.get("endDate"),
                    "capture_ts_ms": _utc_ms_now(),
                },
            )

    # ---- BULK order books across all tokens in kept markets ----
    all_token_ids = []
    for _, outcome_specs, _ in tmp_rows:
        for spec in outcome_specs:
            if spec["token_id"]:
                all_token_ids.append(spec["token_id"])

    books_raw_by_token = _get_books_bulk(all_token_ids, concurrency=BOOK_CONCURRENCY)
    books_sum_by_token = {tid: (_summarize_book(bk) if bk else None) for tid, bk in books_raw_by_token.items()}

    # ---- Finalize rows, add book summaries, OI, concentration, NDJSON writes ----
    for m, outcome_specs, prices in tmp_rows:
        cond_id = m.get("conditionId") or m.get("condition_id")
        books = {}
        for spec in outcome_specs:
            tok = spec["token_id"]
            books[str(spec["index"])] = books_sum_by_token.get(tok)
            # NDJSON: one book-snapshot summary per token
            if WRITE_NDJSON and NDJSON_BOOKS and tok:
                summary = books_sum_by_token.get(tok) or {}
                _append_ndjson(
                    f"{DATA_DIR}/books/token={tok}.ndjson",
                    {"token_id": tok, "conditionId": cond_id, "market_id": m.get("id"), "capture_ts_ms": _utc_ms_now(), **summary},
                )

        market_summary = _build_market_summary(books)

        if cond_id:
            kept_condition_ids.append(cond_id)
        kept_count += 1

        liquidity_value = _to_float_or_none(m.get("liquidity"))
        if liquidity_value is None:
            liquidity_value = _to_float_or_none(m.get("liquidity_num"))

        oi_raw = (
            m.get("openInterest") or m.get("open_interest") or
            m.get("openInterestNum") or m.get("open_interest_num") or
            m.get("oi")
        )
        open_interest = _to_float_or_none(oi_raw)
        if open_interest is None:
            open_interest = _estimate_market_open_interest(cond_id, outcome_specs)

        # holders concentration metrics (by outcome)
        holders_metrics = {"by_outcome": {}}

        # try market-level holders first to compute HHI/top10 AND write NDJSON
        if NDJSON_HOLDERS and cond_id and _is_valid_condition_id(cond_id):
            for chunk in _iter_market_holders(cond_id, page_limit=500):
                tok = (
                    chunk.get("token")
                    or chunk.get("asset")
                    or chunk.get("token_id")
                    or chunk.get("id")
                )
                raw_holders = chunk.get("holders") or []
                hhi, top10 = _hhi_from_holders(_normalize_holders(raw_holders))
                if hhi is not None:
                    # map token -> outcome index (if possible)
                    idx = None
                    for spec in outcome_specs:
                        if spec["token_id"] == tok:
                            idx = str(spec["index"])
                            break
                    if idx is not None:
                        holders_metrics["by_outcome"][idx] = {"hhi": hhi, "top10_share": top10}

                _append_ndjson(
                    f"{DATA_DIR}/holders/market={cond_id}.ndjson",
                    {
                        "conditionId": cond_id,
                        "token_id": tok,
                        "holders": [h for h in raw_holders if isinstance(h, dict)],
                        "capture_ts_ms": _utc_ms_now(),
                    },
                )

        # fallback to per-asset holders if we didn't fill anything
        if NDJSON_HOLDERS and not holders_metrics["by_outcome"]:
            for spec in outcome_specs:
                tok = spec["token_id"]
                if not tok:
                    continue
                try:
                    r = SESSION.get(DATA_HOLDERS_URL, params={"asset": tok, "limit": 1000}, timeout=30)
                    r.raise_for_status()
                    payload = r.json() or {}
                    raw_holders = payload.get("holders") if isinstance(payload, dict) else (payload if isinstance(payload, list) else [])
                    hhi, top10 = _hhi_from_holders(_normalize_holders(raw_holders))
                    if hhi is not None:
                        holders_metrics["by_outcome"][str(spec["index"])] = {"hhi": hhi, "top10_share": top10}
                    _append_ndjson(
                        f"{DATA_DIR}/holders/asset={tok}.ndjson",
                        {
                            "token_id": tok,
                            "conditionId": cond_id,
                            "holders": [h for h in _normalize_holders(raw_holders) if isinstance(h, dict)],
                            "capture_ts_ms": _utc_ms_now(),
                        },
                    )
                except requests.RequestException:
                    pass

        # FULL trades -> NDJSON (paged per market)
        if WRITE_NDJSON and NDJSON_TRADES and cond_id and _is_valid_condition_id(cond_id):
            try:
                for tr in _get_trades_paged(cond_id, taker_only=False, page_limit=500,):
                    _append_ndjson(
                        f"{DATA_DIR}/trades/market={cond_id}.ndjson",
                        {
                            "conditionId": cond_id,
                            "market_id": m.get("id"),
                            "asset": tr.get("asset") or tr.get("token"),
                            "timestamp": tr.get("timestamp"),
                            "side": tr.get("side"),
                            "price": tr.get("price"),
                            "size": tr.get("size"),
                            "outcome": tr.get("outcome"),
                            "outcomeIndex": tr.get("outcomeIndex"),
                           
                        },
                    )
            except requests.RequestException:
                pass

        if WRITE_AGGREGATE_JSON:
            structured_markets.append({
                "market_id": m.get("id"),
                "conditionId": cond_id,
                "question": m.get("question"),
                "category": m.get("slug"),
                "active": m.get("active"),
                "startDate": m.get("start_date") or m.get("startDate"),
                "endDate": m.get("end_date") or m.get("endDate"),
                "liquidity_num": liquidity_value,
                "openInterest": open_interest,
                "volume": _to_float_or_none(m.get("volume")),
                "outcomes": outcome_specs,
                "prices": prices,
                "order_books": books,
                "market_summary": market_summary,
                "holders_metrics": holders_metrics,
                "trades": [],  
                "meta": {"interval": PRICE_INTERVAL, "fidelity_minutes": PRICE_FIDELITY_MIN},
            })

    # ---- Batch trades (small summary attached to JSON) ----
    if INCLUDE_TRADES and kept_condition_ids and WRITE_AGGREGATE_JSON:
        valid_cids = [cid for cid in kept_condition_ids if _is_valid_condition_id(cid)]
        trades_flat: List[dict] = []
        try:
            trades_flat = _get_trades_many(valid_cids, limit=TRADES_LIMIT_PER_BATCH, taker_only=TRADES_TAKER_ONLY)
        except requests.RequestException:
            trades_flat = []

        trades_by_cid: Dict[str, List[dict]] = {}
        for tr in trades_flat:
            cid = tr.get("conditionId")
            if cid is None:
                continue
            trades_by_cid.setdefault(cid, []).append({
                "timestamp": tr.get("timestamp"),
                "size": tr.get("size"),
                "price": tr.get("price"),
                "side": tr.get("side"),
                "proxyWallet": tr.get("proxyWallet"),
            })

        for rec in structured_markets:
            cid = rec.get("conditionId")
            rec["trades"] = trades_by_cid.get(str(cid), []) if cid else []

    # count ONLY markets we kept
    return {"count": kept_count, "markets": structured_markets if WRITE_AGGREGATE_JSON else []}


def run():
    try:
        return get_structured_market_data()
    except Exception as e:
        print(f"error {e}")
        return {"count": 0, "markets": []}



