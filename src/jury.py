"""
Judge-and-jury multi-LLM engine.

Each JUROR answers the same question in ISOLATION (its own API call, no model
sees another's answer). Then one model acts as JUDGE: it reads every juror's
answer and produces the final synthesized verdict (agreement, disagreement, and
the single best reasoned call).

Jurors available depend on which keys are set in .env:
  - claude  (always; via the Claude Code CLI you already use)
  - openai  (if OPENAI_API_KEY set)
  - gemini  (if GEMINI_API_KEY set)

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


def _claude(question: str) -> str:
    """Claude juror via the Claude Code CLI (the existing access, no API key)."""
    try:
        r = subprocess.run(
            ["claude", "-p", question + "\n\nAnswer directly and concisely.",
             "--max-turns", "4"],
            cwd=str(ROOT), capture_output=True, text=True, timeout=TIMEOUT, shell=True)
        return (r.stdout or r.stderr or "").strip() or "(claude: no output)"
    except Exception as e:
        return f"(claude error: {e})"


def _openai(question: str) -> str:
    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        model = os.environ.get("OPENAI_MODEL", "gpt-4o")
        r = client.chat.completions.create(
            model=model, max_tokens=700,
            messages=[{"role": "user", "content": question}])
        return (r.choices[0].message.content or "").strip()
    except Exception as e:
        return f"(openai error: {e})"


def _gemini(question: str) -> str:
    try:
        import google.generativeai as genai
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        model = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
        return (genai.GenerativeModel(model).generate_content(question).text or "").strip()
    except Exception as e:
        return f"(gemini error: {e})"


JURORS = {"claude": _claude, "openai": _openai, "gemini": _gemini}


def available_jurors() -> list[str]:
    """Which jurors can run given the keys present."""
    out = ["claude"]                                   # CLI, always available
    if os.environ.get("OPENAI_API_KEY"):
        out.append("openai")
    if os.environ.get("GEMINI_API_KEY"):
        out.append("gemini")
    return out


def jury_verdict(question: str, jurors: list[str] | None = None,
                 judge: str = "claude") -> dict:
    """Run the panel in isolation, then have `judge` synthesize the verdict."""
    jurors = jurors or available_jurors()
    # 1. Jury, in parallel but each fully isolated (no shared context).
    with ThreadPoolExecutor(max_workers=len(jurors)) as ex:
        results = list(ex.map(lambda n: (n, JURORS[n](question)), jurors))
    answers = dict(results)

    # 2. Judge reads all juror answers and renders the final verdict.
    panel = "\n\n".join(f"=== {n.upper()} juror ===\n{a}" for n, a in answers.items())
    judge_prompt = (
        f"You are the JUDGE on a panel. The question was:\n{question}\n\n"
        f"Here are the independent juror answers (they did not see each other):\n\n"
        f"{panel}\n\n"
        "Deliver the final verdict in plain text, no markdown, no em dashes: "
        "(1) where the jurors agree, (2) any real disagreement and who is right, "
        "(3) the single best reasoned answer. Be concise.")
    verdict = JURORS[judge](judge_prompt)
    return {"question": question, "answers": answers, "verdict": verdict,
            "jurors": jurors, "judge": judge}


def jury_sms(question: str) -> str:
    """Compact verdict for SMS: the judge's call plus which jurors weighed in."""
    r = jury_verdict(question)
    who = ", ".join(r["jurors"])
    return f"Jury ({who}) -> judge verdict:\n{r['verdict']}"


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "Say hello and name which model you are."
    out = jury_verdict(q)
    for n, a in out["answers"].items():
        print(f"\n--- {n} ---\n{a[:400]}")
    print(f"\n=== JUDGE ({out['judge']}) VERDICT ===\n{out['verdict']}")
