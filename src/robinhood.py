"""
Robinhood Crypto API, READ ONLY.

Reads crypto market data and (if your key has it) account holdings. There are
deliberately NO order/trade functions in this module: it can look, never touch.

Auth (per Robinhood Crypto docs): Ed25519-signed requests.
  message  = f"{api_key}{timestamp}{path}{method}{body}"
  signature = base64( Ed25519_sign(message) )
  headers  = x-api-key, x-timestamp, x-signature

Env: ROBINHOOD_API_KEY (from Robinhood after you register the public key),
     ROBINHOOD_PRIVATE_KEY (base64 Ed25519 seed, kept secret).

Usage: python src/robinhood.py --symbols BTC-USD ETH-USD SOL-USD
"""

from __future__ import annotations
import os
import time
import base64
import json
import argparse

import requests

BASE = "https://trading.robinhood.com"
TIMEOUT = 20


def _signer():
    import nacl.signing
    key = os.environ.get("ROBINHOOD_PRIVATE_KEY")
    if not key:
        raise RuntimeError("ROBINHOOD_PRIVATE_KEY not set.")
    return nacl.signing.SigningKey(base64.b64decode(key))


def _headers(method: str, path: str, body: str = "") -> dict:
    api_key = os.environ.get("ROBINHOOD_API_KEY")
    if not api_key:
        raise RuntimeError("ROBINHOOD_API_KEY not set (register the public key at Robinhood first).")
    ts = str(int(time.time()))
    message = f"{api_key}{ts}{path}{method}{body}"
    sig = _signer().sign(message.encode()).signature
    return {
        "x-api-key": api_key,
        "x-timestamp": ts,
        "x-signature": base64.b64encode(sig).decode(),
        "Content-Type": "application/json",
    }


def _get(path: str):
    r = requests.get(BASE + path, headers=_headers("GET", path), timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


# ── Read-only endpoints ────────────────────────────────────────────────
def best_bid_ask(symbols: list[str]) -> dict:
    q = "&".join(f"symbol={s}" for s in symbols)
    return _get(f"/api/v1/crypto/marketdata/best_bid_ask/?{q}")


def estimated_price(symbol: str, side: str = "bid", quantity: str = "1") -> dict:
    return _get(f"/api/v1/crypto/marketdata/estimated_price/"
                f"?symbol={symbol}&side={side}&quantity={quantity}")


def holdings() -> dict:
    return _get("/api/v1/crypto/trading/holdings/")


def accounts() -> dict:
    return _get("/api/v1/crypto/trading/accounts/")


def prices(symbols: list[str]) -> dict:
    """Simple {symbol: mid_price} from best bid/ask."""
    out = {}
    try:
        data = best_bid_ask(symbols)
        for r in data.get("results", []):
            bid = float(r.get("bid_inclusive_of_buy_spread") or r.get("price") or 0)
            ask = float(r.get("ask_inclusive_of_sell_spread") or r.get("price") or 0)
            mid = (bid + ask) / 2 if (bid and ask) else (bid or ask)
            out[r.get("symbol")] = round(mid, 2)
    except Exception as e:
        print(f"Robinhood crypto read failed: {e}")
    return out


if __name__ == "__main__":
    from dotenv import load_dotenv
    from pathlib import Path
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="+", default=["BTC-USD", "ETH-USD", "SOL-USD"])
    args = ap.parse_args()
    if not os.environ.get("ROBINHOOD_API_KEY"):
        print("ROBINHOOD_API_KEY not set yet. Register the public key at Robinhood:")
        print("  Public key:", os.environ.get("ROBINHOOD_PUBLIC_KEY", "(see .env)"))
        print("  Then paste the API key into .env as ROBINHOOD_API_KEY and re-run.")
    else:
        print("Crypto mid prices:", json.dumps(prices(args.symbols), indent=2))
