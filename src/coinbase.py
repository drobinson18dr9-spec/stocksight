"""
Coinbase Advanced Trade API, READ ONLY (crypto prices/quotes).

Auth: short-lived JWT signed ES256 with the EC private key (Coinbase CDP).
No order/trade functions exist in this module by design.

Env: COINBASE_API_KEY_NAME (organizations/.../apiKeys/...),
     COINBASE_API_PRIVATE_KEY (EC PEM; \\n escapes are decoded automatically).

Usage: python src/coinbase.py --products BTC-USD ETH-USD SOL-USD
"""

from __future__ import annotations
import os
import time
import secrets
import argparse

import requests

HOST = "api.coinbase.com"
BASE = f"https://{HOST}"
TIMEOUT = 20


def _build_jwt(uri: str) -> str:
    import jwt
    from cryptography.hazmat.primitives import serialization
    name = os.environ.get("COINBASE_API_KEY_NAME")
    secret = os.environ.get("COINBASE_API_PRIVATE_KEY", "")
    if not (name and secret):
        raise RuntimeError("COINBASE_API_KEY_NAME / COINBASE_API_PRIVATE_KEY not set.")
    pem = secret.replace("\\n", "\n").encode()
    private_key = serialization.load_pem_private_key(pem, password=None)
    now = int(time.time())
    payload = {"sub": name, "iss": "cdp", "nbf": now, "exp": now + 120, "uri": uri}
    headers = {"kid": name, "nonce": secrets.token_hex()}
    return jwt.encode(payload, private_key, algorithm="ES256", headers=headers)


def _get(path: str):
    uri = f"GET {HOST}{path}"
    token = _build_jwt(uri)
    r = requests.get(BASE + path, headers={"Authorization": f"Bearer {token}"}, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def product(product_id: str) -> dict:
    """Full product quote (price, bid/ask, 24h volume, etc.)."""
    return _get(f"/api/v3/brokerage/products/{product_id}")


def prices(product_ids: list[str]) -> dict:
    """{product_id: spot_price} for crypto pairs."""
    out = {}
    for pid in product_ids:
        try:
            d = product(pid)
            p = d.get("price")
            if p is not None:
                out[pid] = round(float(p), 2)
        except Exception as e:
            print(f"Coinbase {pid} failed: {e}")
    return out


def quote(product_id: str) -> dict:
    """Compact quote: price, best bid/ask, 24h volume."""
    d = product(product_id)
    return {
        "product": product_id,
        "price": d.get("price"),
        "bid": d.get("best_bid"),
        "ask": d.get("best_ask"),
        "volume_24h": d.get("volume_24h"),
        "price_change_24h_pct": d.get("price_percentage_change_24h"),
    }


if __name__ == "__main__":
    from dotenv import load_dotenv
    from pathlib import Path
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    ap = argparse.ArgumentParser()
    ap.add_argument("--products", nargs="+", default=["BTC-USD", "ETH-USD", "SOL-USD"])
    args = ap.parse_args()
    import json
    print("Prices:", json.dumps(prices(args.products), indent=2))
    print("Sample quote:", json.dumps(quote(args.products[0]), indent=2))
