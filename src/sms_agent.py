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

# Make the process per-monitor DPI aware so pywinauto's window rectangle and the
# mouse click use the SAME pixel space. Without this, display scaling (e.g. 150%)
# makes clicks land low-and-right of the target (it was hitting the model
# dropdown instead of the composer). Must run before any UI interaction.
if _IS_WINDOWS:
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(2)   # PER_MONITOR_AWARE
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass

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
    "Claude Commands\n"
    "<msg> : chat with web access\n"
    "!jury <q> : Claude+GPT+Gemini answer, a judge synthesizes\n"
    "!gpt <q> / !gemini <q> : one model solo\n"
    "!conversation : resume chat\n"
    "!exit : leave chat\n"
    "summary : latest picks\n"
    "!open : open desktop app\n"
    "+cowork !prompt <txt> : Cowork session, streams replies\n"
    "+code !prompt <txt> : Code session\n"
    "!watch / !endwatch : stream / stop replies\n"
    "!do <task> : allow edits\n"
    "@windows @mac @downloads @claude @stocksight : target\n"
    "Stack with ; . @menu : this list"
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


# Set by the poll loop before each message so a streaming turn can text you
# milestones as they happen. Signature: _NOTIFY(text) -> sends one SMS.
_NOTIFY = None


def _stream_claude(prompt, cwd, action, cont, notify, max_updates=6):
    """Run claude in stream-json mode and text milestone updates (tool uses) as
    they happen. Returns the final answer text. Falls back to '' on any trouble
    so the caller can retry non-streamed."""
    cmd = ["claude", "-p", prompt, "--max-turns", "12",
           "--allowed-tools", "WebSearch,WebFetch,Read,Glob,Grep",
           "--output-format", "stream-json", "--verbose"]
    if cont:
        cmd.append("--continue")
    if action:
        cmd += ["--permission-mode", "acceptEdits"]
    final, sent = "", 0
    try:
        proc = subprocess.Popen(cmd, cwd=str(cwd), stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL, text=True, shell=True)
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except Exception:
                continue
            if ev.get("type") == "assistant":
                for blk in ev.get("message", {}).get("content", []):
                    if blk.get("type") == "tool_use" and notify and sent < max_updates:
                        nm = blk.get("name", "tool")
                        notify(f"... {nm}")
                        sent += 1
            elif ev.get("type") == "result":
                final = (ev.get("result") or "").strip()
        proc.wait(timeout=CLAUDE_TIMEOUT)
    except Exception:
        return final
    return final


def ask_claude(question: str, cwd: Path = ROOT, action: bool = False,
               continue_session: bool = False) -> str:
    """Headless Claude Code run in an allowlisted workspace.
    Default is ANSWER mode: Claude can read/search but permission-gated tools
    (edits, most commands) are blocked in headless mode, so a text cannot
    silently change your machine. Prefix '!do' for ACTION mode, which allows
    file edits (acceptEdits). continue_session=True resumes the prior turn in
    this directory so a series of texts is one threaded conversation."""
    prompt = (
        "You are in an ongoing SMS conversation with the machine owner. "
        f"Working directory: {cwd}. "
        "Output ONLY the message to send, nothing else. No preamble (never write "
        "'Here is the reply' or 'Here is...'), no markdown, no headers, no em "
        "dashes. Plain text under 1200 characters. "
        f"Owner's latest text: {question}"
    )
    cmd = ["claude", "-p", prompt, "--max-turns", "12",
           "--allowed-tools", "WebSearch,WebFetch,Read,Glob,Grep"]
    if continue_session:
        cmd.append("--continue")                 # resume the prior turn (threaded)
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
        return "Still working on that one. Text 'status' in a minute, or send more detail."
    except FileNotFoundError:
        return "Claude CLI not found on this machine."


# --- Conversation state: one running thread per working directory ----------
def _conv_flag(cwd: Path) -> Path:
    safe = str(cwd).replace(":", "_").replace("\\", "_").replace("/", "_")
    return ROOT / "data" / f"conv_{safe}.flag"


def converse(question: str, cwd: Path = ROOT, action: bool = False) -> str:
    """A conversation turn: continue the running thread for this directory if one
    exists, else start it. Each inbound text is one turn; the user can interject
    any time by texting again. Text '!new' to start a fresh thread."""
    flag = _conv_flag(cwd)
    cont = flag.exists()
    # If a notifier is wired and streaming is on, stream tool-use milestones.
    ans = ""
    if _NOTIFY and _os.environ.get("CONV_STREAM", "1") == "1":
        prompt = (
            "You are in an ongoing SMS conversation with the machine owner. "
            f"Working directory: {cwd}. Reply in plain text under 1200 characters, "
            "no markdown, no headers, no em dashes. "
            f"Owner's latest text: {question}"
        )
        ans = _stream_claude(prompt, cwd, action, cont, _NOTIFY)
    if not ans:                                  # fallback / streaming disabled
        ans = ask_claude(question, cwd=cwd, action=action, continue_session=cont)
    try:
        flag.parent.mkdir(exist_ok=True)
        flag.write_text("1")                     # a thread now exists for this cwd
        st = _agent_state(); st["active"] = True; st["exited"] = False
        st["last_activity"] = _now(); _save_agent_state(st)   # refresh idle timer
    except Exception:
        pass
    return ans


def reset_conversation() -> str:
    n = 0
    for f in (ROOT / "data").glob("conv_*.flag"):
        try:
            f.unlink(); n += 1
        except Exception:
            pass
    _save_agent_state({})                        # clear idle/awaiting state too
    return f"Started a fresh conversation ({n} thread(s) cleared)."


# --- Agent state: conversation mode, idle timer, resume prompt --------------
_STATE_FILE = ROOT / "data" / "agent_state.json"
IDLE_SECONDS = int(_os.environ.get("CONV_IDLE_SECONDS", "900"))   # 15 min default
# Cowork watch: ping the transcript every PING seconds (digest, not a flood) and
# stop watching after IDLE seconds with no text from you (so it never exhausts).
WATCH_PING_SECONDS = int(_os.environ.get("WATCH_PING_SECONDS", "900"))    # 15 min
WATCH_IDLE_SECONDS = int(_os.environ.get("WATCH_IDLE_SECONDS", "2700"))   # 45 min


def _agent_state() -> dict:
    try:
        return json.loads(_STATE_FILE.read_text())
    except Exception:
        return {}


def _save_agent_state(s: dict):
    try:
        _STATE_FILE.parent.mkdir(exist_ok=True)
        _STATE_FILE.write_text(json.dumps(s))
    except Exception:
        pass


def _conversation_exists() -> bool:
    return any((ROOT / "data").glob("conv_*.flag"))


def enter_conversation() -> str:
    """!conversation: resume the saved thread if one exists (ask continue/reboot),
    else start fresh."""
    st = _agent_state()
    st["active"] = True
    st["exited"] = False
    st["last_activity"] = _now()
    if _conversation_exists():
        st["awaiting_resume"] = True
        _save_agent_state(st)
        return ("You have a saved conversation. Reply 'continue' to resume it, "
                "or 'reboot' to start fresh.")
    st["awaiting_resume"] = False
    _save_agent_state(st)
    return "Conversation started. Text me anything. !exit when you are done."


def exit_conversation() -> str:
    """!exit: save and leave conversation mode (the thread is kept for resuming)."""
    st = _agent_state()
    st["active"] = False
    st["exited"] = True
    st["awaiting_resume"] = False
    _save_agent_state(st)
    return "Conversation saved and exited. Text !conversation to resume."


def _now() -> float:
    import time as _t
    return _t.time()


# --- Cowork read-bridge: tail the desktop app's on-disk session transcript ---
# The desktop Cowork/Code app writes every turn to audit.jsonl. We tail the
# newest session's file and text new assistant turns back to you. Same schema on
# both platforms; only the data dir differs.
def _claude_data_dir() -> Path:
    if _IS_WINDOWS:
        base = _os.environ.get("APPDATA", str(Path.home() / "AppData" / "Roaming"))
        return Path(base) / "Claude"
    return Path.home() / "Library" / "Application Support" / "Claude"   # macOS


def latest_cowork_audit():
    """Newest session transcript (by modified time)."""
    d = _claude_data_dir() / "local-agent-mode-sessions"
    if not d.exists():
        return None
    audits = list(d.rglob("audit.jsonl"))
    if not audits:
        return None
    return max(audits, key=lambda p: p.stat().st_mtime)


def cowork_watch_start() -> bool:
    """Begin tailing the latest Cowork session from its current end (new turns
    only). Returns True if a session was found."""
    p = latest_cowork_audit()
    if not p:
        return False
    st = _agent_state()
    st["cowork_watch"] = {"path": str(p), "offset": p.stat().st_size,
                          "last_ping": 0, "started": _now()}
    _save_agent_state(st)
    return True


def cowork_watch_stop() -> str:
    st = _agent_state(); st.pop("cowork_watch", None); _save_agent_state(st)
    return "Stopped watching the Cowork session."


def _assistant_text(ev) -> str:
    if ev.get("type") != "assistant":
        return ""
    content = ev.get("message", {}).get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return "\n".join(b["text"] for b in content
                         if isinstance(b, dict) and b.get("type") == "text"
                         and b.get("text")).strip()
    return ""


def cowork_watch_poll(notify, max_msgs=4):
    """Read any new complete lines from the watched transcript and text new
    assistant turns. Byte-offset tail so partial lines are not split."""
    st = _agent_state()
    w = st.get("cowork_watch")
    if not w:
        return
    # Only ping every WATCH_PING_SECONDS so it does not flood you or exhaust itself.
    if _now() - float(w.get("last_ping", 0)) < WATCH_PING_SECONDS:
        return
    w["last_ping"] = _now()
    st["cowork_watch"] = w; _save_agent_state(st)    # persist the ping time now
    p = Path(w["path"]); off = int(w.get("offset", 0))
    try:
        if not p.exists() or p.stat().st_size <= off:
            return
        with open(p, "rb") as f:
            f.seek(off)
            data = f.read()
        last_nl = data.rfind(b"\n")
        if last_nl == -1:
            return                               # no complete line yet
        chunk = data[:last_nl + 1].decode("utf-8", "ignore")
        sent = 0
        for line in chunk.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except Exception:
                continue
            txt = _assistant_text(ev)
            if txt and notify and sent < max_msgs:
                notify(("[cowork] " + txt)[:SMS_LIMIT]); sent += 1
        w["offset"] = off + last_nl + 1
        st["cowork_watch"] = w
        _save_agent_state(st)
    except Exception:
        pass


PICKS_URL = _os.environ.get(
    "PICKS_URL", "https://drobinson18dr9-spec.github.io/stocksight/latest_picks.json")


def quick_summary() -> str:
    """Fast path for 'summary': pull the LIVE deployed picks (what the site shows),
    so it is never the stale local file. Falls back to a local report if offline."""
    try:
        r = requests.get(PICKS_URL, params={"cb": _now()}, timeout=15)
        if r.ok:
            d = r.json()
            picks = d.get("picks", [])[:6]
            if picks:
                lines = "\n".join(f"{i+1}  {p['ticker']}  ${p['price']:.2f}"
                                  for i, p in enumerate(picks))
                return f"StockSight {d.get('asof','')} latest picks:\n{lines}"[:SMS_LIMIT]
    except Exception:
        pass
    # Offline fallback: newest local report.
    try:
        reports = sorted((ROOT / "reports").glob("*_scorecard.md"))
        if not reports:
            return "Picks unavailable right now. Try again shortly."
        lines = reports[-1].read_text(encoding="utf-8").splitlines()
        picks = [l for l in lines if l.startswith("| ") and "$" in l][:6]
        head = reports[-1].stem.replace("_scorecard", "")
        return (f"StockSight {head} latest picks (local):\n" + "\n".join(
            " ".join(p.split("|")[1:4]).strip() for p in picks))[:SMS_LIMIT]
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
                # The claude:// protocol opens AND focuses the existing instance,
                # which is more reliable than the AUMID when windows are stacked.
                try:
                    _os.startfile("claude://")
                except Exception:
                    aumid = _os.environ.get("CLAUDE_APP_AUMID", "Claude_pzs8sxrjxfjjc!Claude")
                    subprocess.Popen(["explorer.exe", f"shell:AppsFolder\\{aumid}"])
        else:  # macOS
            subprocess.Popen(["open", "-a", _os.environ.get("CLAUDE_APP_PATH", "Claude")])
        return "Opening the Claude desktop app on " + THIS_PLATFORM + "."
    except Exception as e:
        return f"Could not open Claude app: {e}"


PROMPTS_DIR = ROOT / "prompts"


def _load_named_prompt(text: str) -> str:
    """Expand a short keyword into a full saved prompt. SMS caps out near 1600
    chars, so long recurring briefs live in prompts/<name>.md and you trigger
    them by name: '+cowork !prompt orwell' loads prompts/orwell.md in full.
    A bare single token (or 'file:<name>') that matches a file is expanded;
    anything with spaces is treated as a literal prompt."""
    t = (text or "").strip()
    name = None
    if t.lower().startswith("file:"):
        name = t[5:].strip()
    elif t and " " not in t and "\n" not in t:
        name = t
    if name:
        for ext in (".md", ".txt", ""):
            p = PROMPTS_DIR / f"{name}{ext}"
            if p.exists() and p.is_file():
                return p.read_text(encoding="utf-8").strip()
    return text


def _with_context(text: str) -> str:
    """Prepend an instruction to read the standing context file (if present),
    so a Cowork/Code session starts oriented without a long SMS."""
    text = _load_named_prompt(text)          # expand saved-prompt keywords first
    ctx = _os.environ.get("COWORK_CONTEXT_FILE", str(ROOT / "cowork_context.md"))
    if ctx and Path(ctx).exists():
        return (f"First read the file at {ctx} for full context and operating "
                f"rules, then complete this request: {text}")
    return text


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


def desktop_session(text: str, mode: str = "cowork") -> str:
    """Open the Claude desktop app, switch to the given surface, drop `text` into
    the composer, and submit. mode='cowork' uses Ctrl+2, mode='code' uses Ctrl+3
    (the app's tab shortcuts). Real UI automation: needs the screen UNLOCKED.
    The composer click is a window-rect fraction, tunable via env per mode."""
    text = (text or "").strip()
    if not text:
        return f"Usage: +{mode} !prompt <what you want Claude to do>"
    wait = float(_os.environ.get("COWORK_LAUNCH_WAIT", "1.5"))
    shortcut = "^2" if mode == "cowork" else "^3"        # Ctrl+2 cowork / Ctrl+3 code
    try:
        _set_clipboard(_with_context(text))
        if _IS_WINDOWS:
            from pywinauto import Desktop, mouse
            from pywinauto.keyboard import send_keys
            # Open via claude://, focus + maximize the window (so composer position
            # is fixed), Ctrl+2/Ctrl+3 to switch tab, then CLICK the composer to
            # focus it (the keyboard-only 'new task' dance does not reliably focus
            # the input, so keys leaked to Search / the model dropdown). The click
            # is now accurate because the process is DPI-aware (see top of file).
            try:
                _os.startfile("claude://")
            except Exception:
                launch_app()
            time.sleep(wait)
            w = None
            for _ in range(10):
                wins = Desktop(backend="uia").windows(title_re=".*[Cc]laude.*",
                                                      top_level_only=True)
                big = []
                for x in wins:
                    try:
                        rr = x.rectangle()
                        if (rr.right - rr.left) > 300 and (rr.bottom - rr.top) > 300:
                            big.append(x)
                    except Exception:
                        pass
                if big:
                    w = max(big, key=lambda x: (lambda r: (r.right - r.left) * (r.bottom - r.top))(x.rectangle()))
                    break
                time.sleep(0.5)
            if w is None:
                raise RuntimeError("no Claude window found")
            if w.is_minimized():
                w.restore()
            # Force the window to the foreground and grab keyboard focus. Retry,
            # because Windows can briefly refuse the first SetForegroundWindow.
            for _ in range(3):
                try:
                    w.set_focus()
                    break
                except Exception:
                    time.sleep(0.4)
            try:
                w.maximize()                          # fix the layout for the click
            except Exception:
                pass
            time.sleep(0.7)
            send_keys(shortcut, pause=0.08)          # Ctrl+2 cowork / Ctrl+3 code
            time.sleep(0.9)
            r = w.rectangle(); W = r.right - r.left; H = r.bottom - r.top

            def _pt(ex, ey, dx, dy):
                fx = float(_os.environ.get(ex, dx)); fy = float(_os.environ.get(ey, dy))
                return (int(r.left + W * fx), int(r.top + H * fy))

            # Click the UPPER text area of the composer (placeholder line), safely
            # above the model/Ask row so a small miss cannot hit the model dropdown.
            if mode == "cowork":
                comp = _pt("COWORK_COMPOSER_FX", "COWORK_COMPOSER_FY", 0.50, 0.35)
            else:
                comp = _pt("CODE_COMPOSER_FX", "CODE_COMPOSER_FY", 0.50, 0.90)
            mouse.click(coords=comp)
            time.sleep(0.4)
            send_keys("^a{BACKSPACE}", pause=0.06)   # clear any stray text
            send_keys("^v", pause=0.08)              # paste the prompt
            time.sleep(0.5)
            send_keys("{ENTER}", pause=0.08)         # submit -> starts the session
        else:  # macOS: activate, Cmd+2/Cmd+3 switch tab, paste, Return
            digit = "2" if mode == "cowork" else "3"
            script = ('tell application "Claude" to activate\n'
                      'delay 1\n'
                      'tell application "System Events"\n'
                      f'  keystroke "{digit}" using command down\n'
                      '  delay 1.0\n'
                      '  keystroke "v" using command down\n'
                      '  delay 0.4\n'
                      '  key code 36\n'
                      'end tell')
            subprocess.run(["osascript", "-e", script], check=True, timeout=30)
        if mode == "cowork":
            cowork_watch_start()                 # stream its replies back to you
        return (f"Started a {mode} session in the Claude desktop app on {THIS_PLATFORM} "
                "with your prompt." + (" Streaming its replies to you."
                                       if mode == "cowork" else ""))
    except Exception as e:
        return ("Opened the app but could not auto-type the prompt "
                f"({e}). Is the screen unlocked? Prompt is on your clipboard, paste it.")


def cowork_prompt(text: str) -> str:
    return desktop_session(text, "cowork")


def code_session(text: str, cwd: Path = ROOT) -> str:
    """Open a VISIBLE interactive Claude Code session (a terminal window) in
    `cwd`, seeded with `text` so you can take over on the desktop. This is the
    '+code' surface, as opposed to the '+cowork' desktop app. Needs the user
    logged in with the screen unlocked."""
    text = _with_context((text or "").strip()) if text else ""
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
        elif t == "!watch":
            i += 1                                # +cowork already auto-watches; accept it
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
    # Conversation mode controls.
    if low in ("!conversation", "!convo", "conversation"):
        return enter_conversation()
    if low in ("!exit", "exit", "!quit", "!stop chat"):
        cowork_watch_stop()                      # also stop any cowork stream
        return exit_conversation()
    if low in ("!watch", "watch"):
        return ("Now streaming the latest Cowork session's replies to you."
                if cowork_watch_start() else
                "No Cowork session found yet. Start one with +cowork !prompt ...")
    if low in ("!unwatch", "!endwatch", "!watch off", "stop watching", "endwatch"):
        return cowork_watch_stop()
    # Multi-model jury+judge, and single-model routing.
    if low.startswith("!jury ") or low == "!jury":
        q = seg.split(None, 1)[1] if " " in seg else ""
        if not q:
            return "Usage: !jury <question> (Claude+GPT+Gemini answer, a judge synthesizes)"
        try:
            import jury
            return jury.jury_sms(q)
        except Exception as e:
            return f"Jury unavailable: {e}"
    if low.startswith("!gpt ") or low.startswith("!gemini "):
        model = "openai" if low.startswith("!gpt ") else "gemini"
        q = seg.split(None, 1)[1]
        try:
            import jury
            return jury.JURORS[model](q)[:SMS_LIMIT]
        except Exception as e:
            return f"{model} unavailable: {e}"
    if low in ("!new", "!reset", "new chat", "reset", "reboot"):
        return reset_conversation() if low != "reboot" else (
            reset_conversation() + " Text me to begin.")
    # Resolve a pending continue/reboot prompt from !conversation.
    if _agent_state().get("awaiting_resume"):
        if low in ("continue", "resume", "yes"):
            st = _agent_state(); st["awaiting_resume"] = False; st["active"] = True
            st["last_activity"] = _now(); _save_agent_state(st)
            return "Resuming your conversation. Go ahead."
        if low in ("reboot", "restart", "fresh", "new"):
            reset_conversation()
            st = _agent_state(); st["active"] = True; st["last_activity"] = _now()
            _save_agent_state(st)
            return "Fresh conversation started. Go ahead."

    p = parse_pipeline(seg)
    if p["err"]:
        return p["err"]
    surface, text = p["surface"], p["text"]

    # An explicit prompt: route to the chosen desktop surface (default Cowork).
    if p["mode"] == "prompt":
        return desktop_session(text, "code" if surface == "code" else "cowork")

    # No !prompt. If a surface was named with text, treat the text as the prompt.
    if surface in ("code", "cowork"):
        if text:
            return desktop_session(text, surface)
        return launch_app()

    # Plain commands.
    if low in ("summary", "stocks", "picks"):
        return quick_summary()
    if not text:
        return launch_app() if p["open"] else "Empty command. Text @menu for the list."
    # Default: a threaded conversation turn (continues the prior text). This is the
    # 2-way chat: you text, it answers when ready, you interject with more texts.
    return converse(text, cwd=p["ws"], action=p["action"])


def handle_text(body: str) -> str:
    """Turn one inbound SMS body into a reply. Supports stacking with ';'
    (e.g. '!open ; summary'); each segment runs in order and replies are joined."""
    body = (body or "").strip()
    if body.upper() in ("STOP", "STOPALL", "CANCEL", "END", "QUIT",
                        "UNSUBSCRIBE", "REVOKE", "HELP", "INFO", "START", "YES"):
        return ""                                  # carrier keywords, no reply
    # A prompt is freeform text (it can contain ';', paths, etc.), so never split
    # a command that carries one. Only plain command chains stack on ';'.
    low = body.lower()
    if any(k in low for k in ("!prompt", "!cowork", "+cowork", "+code")):
        return handle_one(body)
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
                # Any text from you counts as activity (resets conversation +
                # cowork-watch idle timers).
                _stt = _agent_state(); _stt["last_activity"] = _now()
                _save_agent_state(_stt)
                # No corny "started" ack and no tool-use chatter; just answer.
                global _NOTIFY
                _NOTIFY = None
                reply = handle_text(body)
                if reply:
                    send_sms(sid, tok, frm, me, reply)
                    print(f"  replied ({len(reply)} chars)")
            # Orwell Gate 4 host bridge: run the multi-model jury on any request the
            # cowork VM dropped in the shared Downloads queue (the VM has no egress to
            # the model APIs; this host does). Isolated so a bridge error never stops SMS.
            try:
                import orwell_bridge
                done = orwell_bridge.process_pending()
                if done:
                    print(f"  orwell bridge: ran jury on {done} request(s)")
            except Exception as e:
                print(f"  orwell bridge error: {e}")
            # Stream any new Cowork session replies back to you (disk tail),
            # gated to a 15-min digest. Stop watching after 45 min of no texts.
            cowork_watch_poll(lambda t: send_sms(sid, tok, frm, me, t[:SMS_LIMIT]))
            _st2 = _agent_state()
            if _st2.get("cowork_watch") and _st2.get("last_activity"):
                if _now() - float(_st2["last_activity"]) > WATCH_IDLE_SECONDS:
                    cowork_watch_stop()
                    try:
                        send_sms(sid, tok, frm, me,
                                 f"Stopped watching the Cowork session after "
                                 f"{WATCH_IDLE_SECONDS // 60} min idle. Text !watch to resume.")
                        print("  watch idle-stop sent")
                    except Exception:
                        pass
            # Idle auto-save-and-exit: if a conversation is active and untouched
            # for IDLE_SECONDS, save and exit it (resume later with !conversation).
            st = _agent_state()
            if st.get("active") and not st.get("exited") and st.get("last_activity"):
                if _now() - float(st["last_activity"]) > IDLE_SECONDS:
                    exit_conversation()
                    try:
                        mins = IDLE_SECONDS // 60
                        send_sms(sid, tok, frm, me,
                                 f"Conversation saved and exited after {mins} min idle. "
                                 "Text !conversation to resume.")
                        print("  idle save+exit sent")
                    except Exception:
                        pass
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
