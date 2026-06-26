"""
Judge-and-jury multi-LLM engine.

Each JUROR answers the same question in ISOLATION (its own API call, no model
sees another's answer). Then a separate JUDGE reads the juror answers and
produces the final synthesized verdict (agreement, disagreement, best call).

Impartiality rule (hard): the judge NEVER grades an answer from its own lab.
If the judge is GPT, it judges only the non-OpenAI jurors; if it is Claude, it
judges only the non-Anthropic jurors; and so on. A judge from a 4th, fully
independent lab (Grok / xAI) grades all three big-lab jurors with no allegiance.

Jurors (any subset, by which keys are in .env):
  - claude  (always; via the Claude Code CLI)         lab: anthropic
  - openai  (OPENAI_API_KEY)                           lab: openai
  - gemini  (GEMINI_API_KEY)                           lab: google

Judges (separate role; pick the best independent one available):
  - grok    (XAI_API_KEY)        lab: xai    <- best independent judge, frontier reasoning
  - openai  (gpt-5.5-pro)        lab: openai (reasoning flagship)
  - gemini  / claude             fall-backs

Use from the SMS agent ("!jury <question>") or directly:
    from jury import jury_verdict
    jury_verdict("Is NVDA overvalued at today's price?")
"""

from __future__ import annotations
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TIMEOUT = int(os.environ.get("JURY_TIMEOUT", "200"))

# Which lab each model belongs to. The judge is barred from grading its own lab.
LAB = {"claude": "anthropic", "openai": "openai", "gemini": "google", "grok": "xai"}


# ------------------------------------------------------------------ model calls

LIVE = os.environ.get("JURY_LIVE", "1") != "0"   # web/search access on by default

LIVE_PREFIX = (
    "You have live web and search access. For anything time-sensitive (prices, "
    "weather, news, current events, today's date), you MUST search and use the "
    "real current value. Never tell the user to go check a website; fetch it "
    "yourself. State the value with its source and the as-of time (and whether a "
    "temperature is the current reading or the day's high). If you genuinely "
    "cannot retrieve it, say so plainly.\n\nQuestion: ")


def _live(question: str) -> str:
    return (LIVE_PREFIX + question) if LIVE else question


def _claude(question: str, live: bool | None = None) -> str:
    """Claude juror. Prefers the Anthropic API (uses ANTHROPIC_API_KEY, uniform with
    the other jurors, works headless in the host bridge); falls back to the Claude
    Code CLI (your subscription login) if no API key is set. Pass live=False to forbid
    web access (Orwell verification reasons over provided evidence only).
    """
    lv = LIVE if live is None else live

    # Preferred path: the Anthropic Messages API (the API key actually gets used).
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=key)
            base = {"model": os.environ.get("CLAUDE_MODEL", "claude-opus-4-8"),
                    "max_tokens": 1500,
                    "messages": [{"role": "user", "content": question}]}
            attempts = []
            if lv:                                       # server-side web search for live data
                attempts.append({**base, "tools": [
                    {"type": "web_search_20250305", "name": "web_search", "max_uses": 5}]})
            attempts.append(base)                        # plain, no tools
            for kw in attempts:
                try:
                    r = client.messages.create(**kw)
                    text = "".join(getattr(b, "text", "") for b in r.content
                                   if getattr(b, "type", "") == "text").strip()
                    if text:
                        return text
                except Exception:
                    continue
        except Exception:
            pass                                         # fall through to the CLI

    # Fallback path: the Claude Code CLI (no API key needed; uses your subscription).
    try:
        parts = ["claude", "-p",
                 "--model", os.environ.get("CLAUDE_MODEL", "claude-opus-4-8"),
                 "--max-turns", "6"]
        if lv:
            parts += ["--allowed-tools", "WebSearch,WebFetch"]   # let it pull live data
        cmdline = " ".join(parts)                                 # args are all safe tokens
        r = subprocess.run(cmdline, input=question, cwd=str(ROOT),
                           capture_output=True, text=True, timeout=TIMEOUT, shell=True)
        return (r.stdout or r.stderr or "").strip() or "(claude: no output)"
    except Exception as e:
        return f"(claude error: {e})"


def _openai(question: str, model: str | None = None, live: bool | None = None) -> str:
    lv = LIVE if live is None else live
    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        model = model or os.environ.get("OPENAI_MODEL", "gpt-5.5")
        # Responses API handles the whole gpt-5.x line incl. the -pro reasoning
        # models (which the chat endpoint rejects). Big token budget leaves room
        # for reasoning tokens before the visible answer. With LIVE on, attach the
        # web_search tool so the model can pull current data.
        attempts = []
        if lv:
            attempts = [{"tools": [{"type": "web_search"}]},
                        {"tools": [{"type": "web_search_preview"}]}]
        attempts.append({})                              # no-tools fallback
        for extra in attempts:
            try:
                r = client.responses.create(model=model, input=question,
                                            max_output_tokens=4000, **extra)
                t = (getattr(r, "output_text", "") or "").strip()
                if t:
                    return t
            except Exception:
                continue
        # Fallback for older chat-only models.
        kw = {"model": model, "messages": [{"role": "user", "content": question}]}
        try:
            r = client.chat.completions.create(max_completion_tokens=1200, **kw)
        except Exception:
            r = client.chat.completions.create(max_tokens=1200, **kw)
        return (r.choices[0].message.content or "").strip()
    except Exception as e:
        return f"(openai error: {e})"


def _gemini(question: str, model: str | None = None, live: bool | None = None) -> str:
    lv = LIVE if live is None else live
    model = model or os.environ.get("GEMINI_MODEL", "gemini-3.1-pro-preview")
    key = os.environ["GEMINI_API_KEY"]
    # Preferred path: new google-genai SDK with Google Search grounding (live data).
    if lv:
        try:
            from google import genai as ggenai
            from google.genai import types as gtypes
            client = ggenai.Client(api_key=key)
            cfg = gtypes.GenerateContentConfig(
                tools=[gtypes.Tool(google_search=gtypes.GoogleSearch())])
            r = client.models.generate_content(model=model, contents=question, config=cfg)
            t = (getattr(r, "text", "") or "").strip()
            if t:
                return t
        except Exception:
            pass
    # Fallback: legacy library, ungrounded.
    try:
        import google.generativeai as genai
        genai.configure(api_key=key)
        return (genai.GenerativeModel(model).generate_content(question).text or "").strip()
    except Exception as e:
        return f"(gemini error: {e})"


def _grok(question: str, model: str | None = None, live: bool | None = None) -> str:
    """Grok via xAI's OpenAI-compatible endpoint. Best independent judge."""
    lv = LIVE if live is None else live
    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.environ["XAI_API_KEY"],
                        base_url=os.environ.get("XAI_BASE_URL", "https://api.x.ai/v1"))
        model = model or os.environ.get("XAI_MODEL", "grok-4")
        kw = {"model": model, "max_tokens": 1500,
              "messages": [{"role": "user", "content": question}]}
        if lv:
            # xAI Live Search: let Grok pull from web/X when it helps.
            try:
                r = client.chat.completions.create(
                    extra_body={"search_parameters": {"mode": "auto"}}, **kw)
                return (r.choices[0].message.content or "").strip()
            except Exception:
                pass
        r = client.chat.completions.create(**kw)
        return (r.choices[0].message.content or "").strip()
    except Exception as e:
        return f"(grok error: {e})"


# Juror callables (the panel). Judges reuse the same callables.
JURORS = {"claude": _claude, "openai": _openai, "gemini": _gemini}

# Judge callables. OpenAI judge uses the reasoning flagship, not the juror model.
JUDGES = {
    "grok": _grok,
    "openai": lambda q: _openai(q, os.environ.get("OPENAI_JUDGE_MODEL", "gpt-5.5-pro")),
    "gemini": _gemini,
    "claude": _claude,
}


# ------------------------------------------------------------------ availability

def available_jurors() -> list[str]:
    out = ["claude"]                                   # CLI, always available
    if os.environ.get("OPENAI_API_KEY"):
        out.append("openai")
    if os.environ.get("GEMINI_API_KEY"):
        out.append("gemini")
    return out


def default_judge() -> str:
    """Best independent judge available. Grok (4th lab) > OpenAI reasoning > Gemini > Claude."""
    forced = os.environ.get("JURY_JUDGE")
    if forced:
        return forced
    if os.environ.get("XAI_API_KEY"):
        return "grok"                                  # truly independent, no labmate in the box
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"                                # gpt-5.5-pro, grades only non-OpenAI jurors
    if os.environ.get("GEMINI_API_KEY"):
        return "gemini"
    return "claude"


# ------------------------------------------------------------------ the panel

def jury_verdict(question: str, jurors: list[str] | None = None,
                 judge: str | None = None) -> dict:
    """Run the panel in isolation, then have an independent `judge` synthesize."""
    jurors = jurors or available_jurors()
    judge = judge or default_judge()

    # 1. Jury, in parallel but each fully isolated (no shared context). Each gets
    #    the live-data instruction so time-sensitive questions are actually fetched.
    q_live = _live(question)
    with ThreadPoolExecutor(max_workers=max(1, len(jurors))) as ex:
        results = list(ex.map(lambda n: (n, JURORS[n](q_live)), jurors))
    answers = dict(results)

    # 2. Impartiality: the judge never grades an answer from its own lab.
    judged = {n: a for n, a in answers.items() if LAB.get(n) != LAB.get(judge)}
    if not judged:                                     # judge shares a lab with every juror
        judged = answers                               # fall back rather than judge nothing

    panel = "\n\n".join(f"=== {n.upper()} juror ===\n{a}" for n, a in judged.items())
    judge_prompt = (
        f"You are an impartial JUDGE on a panel. You did not answer this question; "
        f"your job is only to weigh the jurors below. The question was:\n{question}\n\n"
        f"Independent juror answers (they did not see each other):\n\n{panel}\n\n"
        "Deliver the final verdict in plain text, no markdown, no em dashes: "
        "(1) where the jurors agree, (2) any real disagreement and who is right and why, "
        "(3) the single best reasoned answer. Be concise and decisive.")
    verdict = JUDGES.get(judge, JURORS.get(judge, _claude))(judge_prompt)

    return {"question": question, "answers": answers, "verdict": verdict,
            "jurors": jurors, "judge": judge, "judged": list(judged.keys())}


def jury_sms(question: str) -> str:
    """Compact verdict for SMS: the judge's call plus who weighed in."""
    r = jury_verdict(question)
    who = ", ".join(r["jurors"])
    return f"Jury [{who}] -> JUDGE {r['judge'].upper()} (graded {', '.join(r['judged'])}):\n{r['verdict']}"


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "Say hello and name which model you are."
    out = jury_verdict(q)
    for n, a in out["answers"].items():
        print(f"\n--- {n} juror ---\n{a[:400]}")
    print(f"\n=== JUDGE: {out['judge']} (graded {out['judged']}) ===\n{out['verdict']}")
