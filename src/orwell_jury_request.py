"""
orwell_jury_request.py - VM-side client for the Project Orwell Gate 4 host bridge.

Runs inside the cowork VM (no egress to the model APIs). Bundles the ledger plus the
evidence packet for each claim into one request, drops it in the shared Downloads
queue, and polls for the host watcher's verdict. The host (orwell_bridge.py) runs the
real multi-model jury and writes the verdict back. So the full panel verifies live
VM claims without the VM ever needing egress.

If the host watcher does not answer within the timeout (it is not running, or the
queue is not shared), this exits 3 so Gate 4 falls back to the Tier 1 Claude panel.
A bridge timeout is never a kill.

The queue path: on the host it defaults to <home>/Downloads/orwell_jury_queue. Inside
the VM, point --queue (or env ORWELL_JURY_QUEUE) at the Downloads MOUNT, e.g.
/sessions/<id>/mnt/Downloads/orwell_jury_queue.

Usage:
  python orwell_jury_request.py --ledger ledger.json --evidence-dir ev/ [--timeout 240] [--queue DIR]
EXIT: 0 all claims SURVIVED, 1 some KILLED, 3 bridge unavailable (use Tier 1).
"""
from __future__ import annotations
import argparse
import json
import os
import time
import uuid
from pathlib import Path

DEFAULT_QUEUE = os.environ.get("ORWELL_JURY_QUEUE",
                               str(Path.home() / "Downloads" / "orwell_jury_queue"))


def _load_evidence(claims, evidence_dir, evidence_file):
    ev = {}
    if evidence_dir:
        d = Path(evidence_dir)
        for c in claims:
            p = d / f"{c.get('id', '')}.txt"
            if p.exists():
                ev[c.get("id")] = p.read_text(encoding="utf-8")
    elif evidence_file:
        shared = Path(evidence_file).read_text(encoding="utf-8")
        ev = {c.get("id"): shared for c in claims}
    return ev


def submit_and_wait(ledger_path, evidence_dir=None, evidence_file=None,
                    queue=DEFAULT_QUEUE, timeout=240.0, poll=3.0) -> dict:
    q = Path(queue)
    q.mkdir(parents=True, exist_ok=True)
    data = json.loads(Path(ledger_path).read_text(encoding="utf-8"))
    claims = data.get("claims", [])
    evidence = _load_evidence(claims, evidence_dir, evidence_file)

    rid = uuid.uuid4().hex[:12]
    req = q / f"{rid}.request.json"
    verdict = q / f"{rid}.verdict.json"
    tmp = q / f"{rid}.request.tmp"
    tmp.write_text(json.dumps({"id": rid, "ledger": {"claims": claims},
                               "evidence": evidence}, indent=2), encoding="utf-8")
    tmp.replace(req)                                   # atomic publish so the host never reads a half file

    deadline = time.time() + timeout
    while time.time() < deadline:
        if verdict.exists():
            try:
                res = json.loads(verdict.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                time.sleep(poll)
                continue
            for f in (req, verdict):                   # clean up our own files
                try:
                    f.unlink()
                except OSError:
                    pass
            return res
        time.sleep(poll)
    try:
        req.unlink()
    except OSError:
        pass
    return {"ran": False, "timeout": True,
            "reason": "host bridge did not answer within the timeout; it may not be "
                      "running or the queue is not shared. Fall back to the Tier 1 "
                      "Claude panel.", "all_clear": False, "killed": [], "results": []}


def main() -> int:
    ap = argparse.ArgumentParser(description="Project Orwell Gate 4 bridge client (VM side).")
    ap.add_argument("--ledger", required=True)
    ap.add_argument("--evidence-dir")
    ap.add_argument("--evidence")
    ap.add_argument("--queue", default=DEFAULT_QUEUE)
    ap.add_argument("--timeout", type=float, default=240.0)
    ap.add_argument("--poll", type=float, default=3.0)
    a = ap.parse_args()

    res = submit_and_wait(a.ledger, a.evidence_dir, a.evidence, a.queue, a.timeout, a.poll)
    print("=" * 64)
    print("PROJECT ORWELL GATE 4 (Tier 2 via host bridge)")
    print("=" * 64)
    if res.get("ran") is False:
        print(res.get("reason", "bridge unavailable"))
        print("=" * 64)
        return 3
    for r in res.get("results", []):
        head = "KILLED" if r.get("killed") else "SURVIVES"
        print(f"[{head}] {r.get('id','?')}: {(r.get('claim') or '')[:90]}")
        jt = (r.get("judge_text") or "").strip().splitlines()
        if jt:
            print(f"    judge: {jt[0][:110]}")
    print("=" * 64)
    if res.get("all_clear"):
        print(f"ALL {res.get('claim_count', 0)} CLAIMS SURVIVED. Gate 4 green.")
        return 0
    print(f"KILLED: {res.get('killed')}. Re-derive/fix/drop, then re-run from Gate 1.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
