"""
orwell_bridge.py - Host-side jury bridge for Project Orwell Gate 4 (Tier 2).

The Orwell task runs inside the Claude desktop cowork VM, which has NO egress to
the model API endpoints, so the multi-model jury cannot call OpenAI/Gemini/xAI from
there. This host-side watcher gives the VM the full panel anyway: the VM drops a
request into the shared Downloads queue (the VM mounts Downloads), the host (which
has egress) runs orwell_jury.verify_ledger, and writes the verdict back for the VM
to read.

Queue (shared between host and VM via the Downloads mount):
  <home>/Downloads/orwell_jury_queue/
    <id>.request.json   written by the VM client: {"id", "ledger":{"claims":[...]}, "evidence":{claim_id: text}}
    <id>.processing      transient lock while the host runs the jury
    <id>.verdict.json   written here: the verify_ledger result (results, killed, all_clear, ran)

Run standalone on the host:
    python orwell_bridge.py --watch [--interval 3] [--queue DIR]
Or import and call process_pending(queue_dir) from a host loop. The SMS daemon
calls it every cycle, so no separate process is needed when the daemon is running.
"""
from __future__ import annotations
import argparse
import json
import time
import traceback
from pathlib import Path

import orwell_jury

DEFAULT_QUEUE = Path.home() / "Downloads" / "orwell_jury_queue"
REQ_SUFFIX = ".request.json"


def process_pending(queue_dir=DEFAULT_QUEUE) -> int:
    """Run the jury on every unanswered request in the queue. Returns how many it
    processed. Best-effort and exception-safe: a bad request gets an error verdict,
    never crashes the watcher (or the SMS daemon that calls this)."""
    q = Path(queue_dir)
    if not q.exists():
        return 0
    processed = 0
    for req in sorted(q.glob("*" + REQ_SUFFIX)):
        rid = req.name[: -len(REQ_SUFFIX)]
        verdict = q / f"{rid}.verdict.json"
        lock = q / f"{rid}.processing"
        if verdict.exists() or lock.exists():
            continue
        try:
            lock.write_text("1", encoding="utf-8")
            data = json.loads(req.read_text(encoding="utf-8"))
            claims = (data.get("ledger") or {}).get("claims") or data.get("claims") or []
            evidence = data.get("evidence") or {}
            res = orwell_jury.verify_ledger(claims, evidence)
            verdict.write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
            processed += 1
        except Exception as e:
            try:
                verdict.write_text(json.dumps({
                    "ran": False, "error": str(e), "trace": traceback.format_exc(),
                    "all_clear": False, "killed": [], "results": [],
                    "reason": "host bridge could not process this request"},
                    indent=2), encoding="utf-8")
            except OSError:
                pass
        finally:
            try:
                lock.unlink()
            except OSError:
                pass
    return processed


def main():
    ap = argparse.ArgumentParser(description="Project Orwell host-side jury bridge.")
    ap.add_argument("--watch", action="store_true", help="Loop forever processing requests.")
    ap.add_argument("--interval", type=float, default=3.0, help="Seconds between polls in --watch.")
    ap.add_argument("--queue", default=str(DEFAULT_QUEUE), help="Shared queue dir.")
    a = ap.parse_args()
    q = Path(a.queue)
    q.mkdir(parents=True, exist_ok=True)
    if not a.watch:
        n = process_pending(q)
        print(f"processed {n} request(s) in {q}")
        return
    print(f"orwell_bridge watching {q} every {a.interval}s")
    while True:
        try:
            process_pending(q)
        except Exception:
            traceback.print_exc()
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
