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
    "home": _LOCAL_HOME,                                  # forks to @windows/@mac
    THIS_PLATFORM: _LOCAL_HOME,                           # @windows or @mac = this home
}
_other = "mac" if _IS_WINDOWS else "windows"
if _CROSS_HOME and Path(_CROSS_HOME).exists():
    WORKSPACES[_other] = Path(_CROSS_HOME)               # the other box, if mounted

# Which daemon answers an UNtargeted text when both machines are running.
# A text addressed @windows/@mac is answered only by that machine; an untargeted
# text is answered only by the PRIMARY. Default primary is windows; override with
# STOCKSIGHT_PRIMARY=mac on the Mac (or leave windows as the sole responder).
PRIMARY = _os.environ.get("STOCKSIGHT_PRIMARY", "windows").lower()


def addressed_platform(body: str):
    """Return 'windows'/'mac' if the text is explicitly addressed to a machine,
    else None. Used so two running daemons do not both reply."""
    for tok in body.lower().split():
        if tok in ("@windows", "@mac"):
            return tok[1:]
        if not tok.startswith("@"):
            break
    return None


def should_answer(body: str) -> bool:
    tgt = addressed_platform(body)
    if tgt:
        return tgt == THIS_PLATFORM        # only the addressed machine answers
    return PRIMARY == THIS_PLATFORM        # untargeted -> only the primary answers


MENU = (
    "StockSight SMS commands:\n"
    "summary - latest picks (fast)\n"
    "<question> - Claude answers from the repo\n"
    "@windows / @mac <q> - run on that machine's home\n"
    "@home <q> - this machine's home (forks to @windows on PC, @mac on Mac)\n"
    "@downloads / @claude / @stocksight <q> - those folders\n"
    "!do <task> - allow file edits (e.g. @downloads !do rename X to Y)\n"
    "!open - launch the Claude desktop app here\n"
    "Pipeline: [@windows/@mac] [+cowork|+code] !prompt <text>\n"
    "  +cowork !prompt <text> - start a Cowork session in the desktop app\n"
    "  +code !prompt <text>   - open a Claude Code session (terminal) with it\n"
    "  e.g. @windows +code !prompt scope a handshake omega task\n"
    "Stack with ; -> !open ; summary\n"
    "@menu - this list"
)


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


def launch_app() -> str:
    """Open the NATIVE Claude desktop app (the GUI with Cowork), not the CLI.
    On Windows the app is a Store/MSIX package launched via its AppUserModelID
    (CLAUDE_APP_AUMID overrides). On macOS, `open -a Claude`."""
    try:
        if _IS_WINDOWS:
            p = _os.environ.get("CLAUDE_APP_PATH")
            if p and Path(p).exists():
                subprocess.Popen([p])
            else:
                aumid = _os.environ.get("CLAUDE_APP_AUMID", "Claude_pzs8sxrjxfjjc!Claude")
                # explorer resolves shell:AppsFolder\<AUMID> for Store apps.
                subprocess.Popen(["explorer.exe", f"shell:AppsFolder\\{aumid}"])
        else:  # macOS
            subprocess.Popen(["open", "-a", _os.environ.get("CLAUDE_APP_PATH", "Claude")])
        return "Opening the Claude desktop app on " + THIS_PLATFORM + "."
    except Exception as e:
        return f"Could not open Claude app: {e}"


def _set_clipboard(text: str):
    """Put text on the OS clipboard (used to paste into the desktop app)."""
    try:
        import pyperclip
        pyperclip.copy(text)
        return
    except Exception:
        pass
    if _IS_WINDOWS:
        subprocess.run("clip", input=text, text=True, shell=True, timeout=10)
    else:
        subprocess.run("pbcopy", input=text, text=True, timeout=10)


def cowork_prompt(text: str) -> str:
    """Open the Claude desktop app and drop `text` into its input box, then submit,
    starting a Cowork session with that prompt. This is real UI automation: it
    needs the machine logged in with the screen UNLOCKED (a locked screen blocks
    synthetic input). Best-effort; tune COWORK_LAUNCH_WAIT if the window is slow."""
    text = (text or "").strip()
    if not text:
        return "Usage: !prompt <what you want Claude to do in the desktop app>"
    wait = float(_os.environ.get("COWORK_LAUNCH_WAIT", "4"))
    try:
        _set_clipboard(text)
        launch_app()
        time.sleep(wait)                       # let the window come up / focus input
        if _IS_WINDOWS:
            from pywinauto import Desktop
            from pywinauto.keyboard import send_keys
            w = Desktop(backend="uia").window(title_re=".*[Cc]laude.*")
            w.wait("visible ready", timeout=15)
            w.set_focus()
            time.sleep(0.6)
            send_keys("^a{BACKSPACE}", pause=0.05)   # clear any leftover text
            send_keys("^v", pause=0.05)              # paste the prompt
            time.sleep(0.4)
            send_keys("{ENTER}", pause=0.05)         # submit -> starts the session
        else:  # macOS via AppleScript
            script = ('tell application "Claude" to activate\n'
                      'delay 1\n'
                      'tell application "System Events"\n'
                      '  keystroke "v" using command down\n'
                      '  delay 0.3\n'
                      '  key code 36\n'              # Return
                      'end tell')
            subprocess.run(["osascript", "-e", script], check=True, timeout=30)
        return f"Started a Cowork session in the Claude desktop app on {THIS_PLATFORM} with your prompt."
    except Exception as e:
        return ("Opened the app but could not auto-type the prompt "
                f"({e}). Is the screen unlocked? Prompt is on your clipboard, paste it.")


def code_session(text: str, cwd: Path = ROOT) -> str:
    """Open a VISIBLE interactive Claude Code session (a terminal window) in
    `cwd`, seeded with `text` so you can take over on the desktop. This is the
    '+code' surface, as opposed to the '+cowork' desktop app. Needs the user
    logged in with the screen unlocked."""
    text = (text or "").strip()
    try:
        if _IS_WINDOWS:
            safe = text.replace('"', "'")
            cmd = 'start "Claude Code" cmd /k claude' + (f' "{safe}"' if safe else "")
            subprocess.Popen(cmd, shell=True, cwd=str(cwd))
        else:
            safe = text.replace('"', "'")
            inner = f"cd {cwd} && claude" + (f' \\"{safe}\\"' if safe else "")
            subprocess.Popen(["osascript", "-e",
                              f'tell application "Terminal" to do script "{inner}"'])
        return (f"Opened a Claude Code session on {THIS_PLATFORM} in {cwd}"
                + (" with your prompt." if text else "."))
    except Exception as e:
        return f"Could not open Claude Code session: {e}"


def parse_pipeline(seg: str):
    """Parse the command pipeline: leading [@workspace] [!open] [+cowork|+code]
    [!do] modifiers, then the command/prompt. Returns a plan dict."""
    ws, surface, want_open, action, err = WORKSPACES["stocksight"], None, False, False, None
    toks = seg.split()
    i = 0
    while i < len(toks):
        t = toks[i].lower()
        if t.startswith("@"):
            name = t[1:]
            if name in WORKSPACES:
                ws = WORKSPACES[name]
            elif name in ("windows", "mac"):
                err = (f"@{name} targets the other machine; not reachable from this "
                       f"{THIS_PLATFORM} box without CROSS_HOME.")
            else:
                err = f"Unknown workspace @{name}. Text @menu."
            i += 1
        elif t == "!open":
            want_open = True; i += 1
        elif t in ("+cowork", "+coworks"):
            surface = "cowork"; i += 1
        elif t == "+code":
            surface = "code"; i += 1
        elif t == "!do":
            action = True; i += 1
        else:
            break
    rest = " ".join(toks[i:]).strip()
    mode = "plain"
    if rest.lower().startswith("!prompt") or rest.lower().startswith("!cowork"):
        mode = "prompt"
        rest = rest.split(None, 1)[1] if " " in rest else ""
    return {"ws": ws, "surface": surface, "open": want_open,
            "action": action, "mode": mode, "text": rest, "err": err}


QUICK_CMDS = ("summary", "stocks", "picks", "!open", "open claude",
              "!open claude", "@menu", "menu")


def is_quick(seg: str) -> bool:
    """A command that returns instantly (no Claude call) - so we skip the ack."""
    return seg.lower().strip() in QUICK_CMDS


def handle_one(seg: str) -> str:
    """Run a single (non-stacked) command and return its reply.

    Pipeline grammar:  [@workspace] [!open] [+cowork|+code] [!do] <command>
    where <command> is: !prompt <text> | summary | a question | a !do task.
    Examples:
      @windows +cowork !prompt scope a handshake omega task
      @stocksight +code !prompt fix the IPO branch in predict.py
      @downloads !do delete the old zip
      !open
      summary
    """
    seg = (seg or "").strip()
    low = seg.lower().strip()
    if low in ("@menu", "menu"):
        return MENU
    if low in ("open claude", "!open claude"):
        return launch_app()

    p = parse_pipeline(seg)
    if p["err"]:
        return p["err"]
    surface, text = p["surface"], p["text"]

    # An explicit prompt: route to the chosen surface (default Cowork app).
    if p["mode"] == "prompt":
        if surface == "code":
            return code_session(text, p["ws"])
        return cowork_prompt(text)          # +cowork (default): desktop app

    # No !prompt. If a surface was named with text, treat the text as the prompt.
    if surface == "code":
        return code_session(text, p["ws"])
    if surface == "cowork":
        return cowork_prompt(text) if text else launch_app()

    # Plain commands.
    if low in ("summary", "stocks", "picks"):
        return quick_summary()
    if not text:
        return launch_app() if p["open"] else "Empty command. Text @menu for the list."
    return ask_claude(text, cwd=p["ws"], action=p["action"])


def handle_text(body: str) -> str:
    """Turn one inbound SMS body into a reply. Supports stacking with ';'
    (e.g. '!open ; summary'); each segment runs in order and replies are joined."""
    body = (body or "").strip()
    if body.upper() in ("STOP", "STOPALL", "CANCEL", "END", "QUIT",
                        "UNSUBSCRIBE", "REVOKE", "HELP", "INFO", "START", "YES"):
        return ""                                  # carrier keywords, no reply
    segs = [s for s in (p.strip() for p in body.split(";")) if s]
    if len(segs) <= 1:
        return handle_one(body)
    out = []
    for s in segs:
        out.append(f"[{s[:24]}] {handle_one(s)}")
    return "\n".join(out)[:SMS_LIMIT]


def serve_webhook(port=8787):
    """Run a tiny HTTP listener for Twilio inbound webhooks. Point a tunnel
    (Cloudflare Tunnel / ngrok) at http://localhost:PORT/sms and set that public
    URL as the number's 'A MESSAGE COMES IN' webhook in the Twilio console.
    No polling; Twilio pushes each text here."""
    from http.server import BaseHTTPRequestHandler, HTTPServer
    from urllib.parse import parse_qs
    sid, tok, frm, me = _env()

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):  # quiet default logging
            pass

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            data = parse_qs(self.rfile.read(n).decode("utf-8", "ignore"))
            sender = (data.get("From") or [""])[0]
            body = (data.get("Body") or [""])[0]
            print(f"[{datetime.now():%H:%M:%S}] webhook from {sender}: {body[:80]}")
            reply = handle_text(body) if (sender == me and should_answer(body)) else ""
            # TwiML response: Twilio sends `reply` back to the user automatically
            twiml = f"<?xml version='1.0' encoding='UTF-8'?><Response>{('<Message>'+_esc(reply)+'</Message>') if reply else ''}</Response>"
            self.send_response(200)
            self.send_header("Content-Type", "text/xml")
            self.end_headers()
            self.wfile.write(twiml.encode("utf-8"))

    print(f"Webhook listener on http://localhost:{port}/  (POST). Point a tunnel here.")
    print(f"Platform {THIS_PLATFORM}. Workspaces: {', '.join('@'+k for k in WORKSPACES)}")
    HTTPServer(("0.0.0.0", port), H).serve_forever()


def _esc(s):
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))[:SMS_LIMIT]


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
                if (m.get("from") or "") != me:
                    continue
                body = m.get("body") or ""
                print(f"[{datetime.now():%H:%M:%S}] inbound: {body[:80]}")
                if not should_answer(body):
                    print(f"  not for {THIS_PLATFORM} (addressed elsewhere / not primary)")
                    continue
                # Confirm receipt for anything that is not instant, so you get a
                # 'task started' text right away and the result when it finishes.
                segs = [s.strip() for s in body.split(";") if s.strip()]
                if any(not is_quick(s) for s in segs):
                    send_sms(sid, tok, frm, me, f"Started on {THIS_PLATFORM}: {body[:120]}")
                    print("  ack sent")
                reply = handle_text(body)
                if reply:
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
    ap.add_argument("--webhook", action="store_true", help="run webhook listener instead of polling")
    ap.add_argument("--port", type=int, default=8787)
    args = ap.parse_args()
    if args.webhook:
        serve_webhook(args.port)
    else:
        main(once=args.once)
