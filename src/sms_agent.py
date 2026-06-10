"""
StockSight two-way SMS agent.

Text your Twilio number -> this daemon (running on your PC) sees the inbound
message, hands it to a headless Claude Code session (claude -p) running inside
the stocksight repo (so it has the full project context and can run the math),
and texts Claude's answer back to you via Twilio.

How it works (poll-based, so no public webhook/server is needed):
  1. Every POLL_SECONDS it lists recent inbound messages to TWILIO_FROM.
  2. New message from SMS_TO (your phone) -> builds a prompt -> runs
     `claude -p` headless with a turn/time budget.
  3. Reply is trimmed to SMS size and sent back through the Twilio API.
  4. Processed message SIDs are remembered in data/sms_agent_seen.json so
     restarts do not re-answer old texts.

Commands you can text:
  "summary"                  -> quick scorecard summary (cached, fast)
  "question..."              -> Claude answers from the stocksight repo
  "@windows question..."     -> Claude runs in this machine's user profile
                                (on the Mac daemon, use @mac). Allowlist:
                                @stocksight @downloads @claude @windows/@mac;
                                the OTHER platform only via CROSS_HOME mount.
  "!do task..."              -> ACTION mode: file edits allowed (acceptEdits).
                                Combine: "@claude !do rename X to Y"
  Default is answer-only: no edits or state changes without "!do".

Safety: replies only to SMS_TO (your number). STOP/HELP are handled by
Twilio/A2P itself and are ignored here.

Run it:  python src/sms_agent.py          (leave running in a terminal)
         python src/sms_agent.py --once   (single poll, for testing)
"""

from __future__ import annotations
import argparse
import json
import subprocess
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
SEEN_FILE = ROOT / "data" / "sms_agent_seen.json"
POLL_SECONDS = 20
SMS_LIMIT = 1500              # Twilio splits long SMS; keep replies reasonable
CLAUDE_TIMEOUT = 240          # seconds for a headless Claude run


def _env():
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    import os
    return (os.environ["TWILIO_SID"], os.environ["TWILIO_TOKEN"],
            os.environ["TWILIO_FROM"], os.environ["SMS_TO"])


def _seen() -> set:
    try:
        return set(json.loads(SEEN_FILE.read_text()))
    except Exception:
        return set()


def _save_seen(s: set):
    SEEN_FILE.parent.mkdir(exist_ok=True)
    SEEN_FILE.write_text(json.dumps(sorted(s)[-500:]))


def fetch_inbound(sid, tok, to_number, since_minutes=90) -> list:
    """Recent inbound messages TO our Twilio number."""
    url = f"https://api.twilio.com/2010-04-01/Accounts/{sid}/Messages.json"
    r = requests.get(url, auth=(sid, tok),
                     params={"To": to_number, "PageSize": 20}, timeout=20)
    r.raise_for_status()
    out = []
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=since_minutes)
    for m in r.json().get("messages", []):
        if m.get("direction") != "inbound":
            continue
        try:
            sent = datetime.strptime(m["date_created"], "%a, %d %b %Y %H:%M:%S %z")
            if sent < cutoff:
                continue
        except Exception:
            pass
        out.append(m)
    return out


def send_sms(sid, tok, frm, to, body):
    url = f"https://api.twilio.com/2010-04-01/Accounts/{sid}/Messages.json"
    r = requests.post(url, auth=(sid, tok),
                      data={"From": frm, "To": to, "Body": body[:SMS_LIMIT]},
                      timeout=20)
    r.raise_for_status()
    return r.json().get("sid")


# Workspace routing: text "@name question" to point Claude somewhere else.
# Only these allowlisted roots are reachable; everything else is refused.
import os as _os
import platform as _plat

_IS_WINDOWS = _os.name == "nt"
THIS_PLATFORM = "windows" if _IS_WINDOWS else "mac"

# Cross-platform homes. The machine the daemon runs on maps its own platform
# label to the real local home. The OTHER platform is reachable only if you
# point CROSS_HOME at a mounted/synced path (env), else it is refused, not guessed.
_LOCAL_HOME = Path.home()
_CROSS_HOME = _os.environ.get("CROSS_HOME")              # e.g. a synced/SMB path
WORKSPACES = {
    "stocksight": ROOT,                                  # default
    "downloads": _LOCAL_HOME / "Downloads",
    "claude": _LOCAL_HOME / "Claude",
    THIS_PLATFORM: _LOCAL_HOME,                           # @windows or @mac = this home
}
_other = "mac" if _IS_WINDOWS else "windows"
if _CROSS_HOME and Path(_CROSS_HOME).exists():
    WORKSPACES[_other] = Path(_CROSS_HOME)               # the other box, if mounted


def parse_routing(body: str):
    """'@windows !do clean up X' -> (workspace_path, action_mode, question, error)."""
    ws, action, err = WORKSPACES["stocksight"], False, None
    parts = body.strip().split()
    while parts:
        head = parts[0].lower()
        if head.startswith("@"):
            name = head[1:]
            if name in WORKSPACES:
                ws = WORKSPACES[name]; parts = parts[1:]
            elif name in ("windows", "mac"):
                err = (f"@{name} is the other machine and is not reachable from "
                       f"this {THIS_PLATFORM} box (set CROSS_HOME to a mounted path).")
                parts = parts[1:]
            else:
                err = f"Unknown workspace @{name}. Use @stocksight @downloads @claude @{THIS_PLATFORM}."
                parts = parts[1:]
        elif head == "!do":
            action = True; parts = parts[1:]
        else:
            break
    return ws, action, " ".join(parts), err


def ask_claude(question: str, cwd: Path = ROOT, action: bool = False) -> str:
    """Headless Claude Code run in an allowlisted workspace.
    Default is ANSWER mode: Claude can read/search but permission-gated tools
    (edits, most commands) are blocked in headless mode, so a text cannot
    silently change your machine. Prefix '!do' for ACTION mode, which allows
    file edits (acceptEdits). Full bypass is deliberately NOT offered here."""
    prompt = (
        "You are answering a text message (SMS) from the machine owner. "
        f"Working directory: {cwd}. "
        "Be direct and compact: the ENTIRE answer must fit in under 900 characters "
        "of plain text. No markdown, no headers, no em dashes. "
        f"Owner's text: {question}"
    )
    cmd = ["claude", "-p", prompt, "--max-turns", "8"]
    if action:
        cmd += ["--permission-mode", "acceptEdits"]
    try:
        res = subprocess.run(
            cmd, cwd=str(cwd), capture_output=True, text=True,
            timeout=CLAUDE_TIMEOUT, shell=True,
        )
        ans = (res.stdout or "").strip() or (res.stderr or "").strip()
        return ans[:SMS_LIMIT] if ans else "No answer produced. Try again."
    except subprocess.TimeoutExpired:
        return "Claude run timed out. Try a simpler question."
    except FileNotFoundError:
        return "Claude CLI not found on this machine."


def quick_summary() -> str:
    """Fast path for 'summary': latest saved scorecard, no recompute."""
    try:
        reports = sorted((ROOT / "reports").glob("*_scorecard.md"))
        if not reports:
            return "No saved scorecard yet. Text anything else to ask Claude."
        lines = reports[-1].read_text(encoding="utf-8").splitlines()
        picks = [l for l in lines if l.startswith("| ") and "$" in l][:6]
        head = reports[-1].stem.replace("_scorecard", "")
        body = f"StockSight {head} latest picks:\n" + "\n".join(
            " ".join(p.split("|")[1:4]).strip() for p in picks)
        return body[:SMS_LIMIT]
    except Exception as e:
        return f"Summary unavailable: {e}"


def main(once=False):
    sid, tok, frm, me = _env()
    seen = _seen()
    print(f"SMS agent up on {THIS_PLATFORM.upper()}. Polling {frm} every {POLL_SECONDS}s.")
    print(f"Workspaces: {', '.join('@'+k for k in WORKSPACES)}  (default @stocksight)")
    print("Ctrl+C to stop.")
    while True:
        try:
            for m in fetch_inbound(sid, tok, frm):
                msid = m["sid"]
                if msid in seen:
                    continue
                seen.add(msid); _save_seen(seen)
                body = (m.get("body") or "").strip()
                sender = m.get("from")
                if sender != me or not body:
                    continue
                if body.upper() in ("STOP", "STOPALL", "CANCEL", "END", "QUIT",
                                    "UNSUBSCRIBE", "REVOKE", "HELP", "INFO",
                                    "START", "YES"):
                    continue                      # carrier keywords, leave alone
                print(f"[{datetime.now():%H:%M:%S}] inbound: {body[:80]}")
                if body.lower().strip() in ("summary", "stocks", "picks"):
                    reply = quick_summary()
                else:
                    ws, action, q, err = parse_routing(body)
                    if err:
                        reply = err
                    elif not q:
                        reply = "Empty question. Try: @windows what big files are in Downloads"
                    else:
                        reply = ask_claude(q, cwd=ws, action=action)
                send_sms(sid, tok, frm, me, reply)
                print(f"  replied ({len(reply)} chars)")
        except Exception as e:
            print(f"poll error: {e}")
        if once:
            break
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()
    main(once=args.once)
