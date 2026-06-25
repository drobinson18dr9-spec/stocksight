"""
orwell_jury.py - Multi-model grounding jury for Project Orwell (T1) Gate 4.

Replaces the single-model (Claude-only) adversarial grounding agents in the
orwell-gates skill with a real judge-and-jury panel. Claude (the orchestrator)
fetches each rollout transcript to disk and recomputes the answer key in Python,
then hands the SAME evidence packet (transcript excerpts + source rows + the
recomputed figure) to every juror. So all models read the identical source; only
the fetch and the Python recompute stay Claude's job (the API models cannot do
those, but they do not need to once the evidence is in front of them).

Panel (matches the SMS jury, but web OFF: a grounding audit reasons over the
evidence shown, never the open web):
  - Jurors:  claude (Opus 4.8), openai (GPT-5.5), gemini (Gemini 3.1 Pro)
  - Judge:   grok (Grok 4, independent 4th lab, grades all three with no labmate)

Contract (from the orwell-gates SKILL.md Gate 4): DEFAULT TO REJECT. A claim
survives only if clearly anchored to a named artifact in the evidence AND the
number is independently confirmable from the rows shown AND it is about work
product, not the trajectory. Majority-UNVERIFIED kills it. A killed claim must be
re-derived from source, fixed, or dropped from the ledger and the prose before
text enters the form.

Use:
    from orwell_jury import verify_claim, verify_ledger
    r = verify_claim("Rollout 2 misses the BND-012 recount", evidence_text)
    # r["killed"] -> bool ; r["verdicts"] -> per-juror ; r["judge"] -> final

CLI:
    python orwell_jury.py --claim "<claim>" --evidence evidence.txt [--rank A+]
    python orwell_jury.py --ledger ledger.json --evidence-dir ./ev   # one <id>.txt per claim
"""

from __future__ import annotations
import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Load the stocksight .env (the API keys) so the Gate-4 CLI runs standalone from
# the Orwell workspace, not just when a caller pre-loads dotenv.
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except Exception:
    pass

import jury   # sibling module: model callables, LAB map, default_judge

VERIFY_JURORS = ["claude", "openai", "gemini"]
VERIFY_JUDGE = "grok"

# --- network reachability preflight ------------------------------------------
# Orwell tasks run inside the Claude desktop cowork VM, which may have NO egress
# to the model API endpoints. A grounding audit must NEVER hang on an unreachable
# endpoint, and must NEVER kill a claim just because the network was down. We TCP
# preflight each endpoint (a few seconds, client-side timeout) and run the jury
# only with the jurors that are actually reachable. If too few are reachable, the
# multi-model jury reports "cannot run here" so the caller falls back to the
# in-environment Claude Agent-tool panel (the Gate 4 floor), rather than killing.
import socket

ENDPOINTS = {
    "openai": ("api.openai.com", 443),
    "gemini": ("generativelanguage.googleapis.com", 443),
    "grok":   ("api.x.ai", 443),
    "claude": ("api.anthropic.com", 443),
}
_REACH_CACHE: dict = {}


def _tcp_ok(host: str, port: int, timeout: float) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except Exception:
        return False


def reachable_providers(timeout: float = 4.0, force: bool = False) -> dict:
    """Which model endpoints answer a TCP connect within `timeout` seconds.
    Cached after the first call (the network does not change mid-run)."""
    if _REACH_CACHE and not force:
        return _REACH_CACHE
    with ThreadPoolExecutor(max_workers=len(ENDPOINTS)) as ex:
        futs = {n: ex.submit(_tcp_ok, h, p, timeout) for n, (h, p) in ENDPOINTS.items()}
        res = {n: f.result() for n, f in futs.items()}
    _REACH_CACHE.clear()
    _REACH_CACHE.update(res)
    return res


def _not_reachable_result(reach: dict, claim: str = "", judge: str = "") -> dict:
    return {
        "ran": False,
        "reachable": reach,
        "reason": ("fewer than 2 model endpoints reachable from this environment, so the "
                   "multi-model jury cannot run here. This is expected inside the cowork VM "
                   "(no egress). Fall back to the in-environment Claude Agent-tool grounding "
                   "panel (the Gate 4 floor), which works without external egress."),
        "killed": False, "claim": claim, "verdicts": {}, "judge": judge,
        "unverified_count": 0, "juror_total": 0, "judge_text": "",
    }

_JUROR_PROMPT = """You are an impartial grounding auditor on a Project Orwell rank submission. You are shown SOURCE EVIDENCE (rollout transcript excerpts and the source data behind it) and ONE CLAIM the submission makes. Judge ONLY from the evidence shown. Do not use outside knowledge and do not assume any fact that is not present in the evidence.

For the CLAIM decide three things:
1. ANCHORED: is it tied to a specific named artifact, row, SKU, account, tab, or figure that actually appears in the evidence?
2. VERIFIED: is the stated number or fact independently confirmable from the evidence, and does it match? Recompute any arithmetic yourself from the rows shown. A behavioral or structural claim (the rollout did X, the workbook lacks Y) must be supported by a specific line in the evidence, not inferred.
3. WORK-PRODUCT: is the claim about the delivered artifact, not the model's reasoning or tool-call trajectory?

Default to REJECT. If the claim is not clearly anchored in the evidence, or the number does not match, or you cannot independently confirm it from what is shown, answer UNVERIFIED. Do not give the benefit of the doubt.

Return EXACTLY two lines, no markdown, no em dashes:
VERDICT: ANCHORED+VERIFIED   (or)   VERDICT: UNVERIFIED
REASON: one or two sentences citing the specific evidence you used.

=== SOURCE EVIDENCE ===
{evidence}

=== CLAIM ===
{claim}
"""

_JUDGE_PROMPT = """You are the independent JUDGE adjudicating a grounding audit on a Project Orwell rank submission. You did not audit; your job is to weigh the jurors against the evidence. Below are the CLAIM, the SOURCE EVIDENCE, the juror tally, and each juror's verdict.

JUROR TALLY: {verified} of {total} jurors marked ANCHORED+VERIFIED, {unverified} marked UNVERIFIED.

Decision rule, in order:
1. If a MAJORITY of jurors marked UNVERIFIED, KILL it.
2. If you can quote a SPECIFIC line in the evidence that the claim's number or fact contradicts, KILL it (a backstop even when jurors passed it).
3. Otherwise, if the claim's factual core is anchored to a named artifact in the evidence and a majority verified it, it SURVIVES. Do not kill a majority-verified, anchored claim just because one juror dissented or because the claim contains an interpretive phrase. If a phrase asserts something not in the evidence (for example what an owner "cares about"), keep FINAL: SURVIVES and put the tightening in FIX. Killing an anchored, majority-verified claim with no contradicting evidence line is a false reject; do not do it.

Return EXACTLY, no markdown, no em dashes:
FINAL: SURVIVES   (or)   FINAL: KILLED
REASON: why, citing the specific evidence line and the juror split.
FIX: if killed, what must be re-derived from source, corrected, or dropped. If it survives but a phrase should be tightened, say how. If fully clean, write none.

=== SOURCE EVIDENCE ===
{evidence}

=== CLAIM ===
{claim}

=== JUROR VERDICTS ===
{panel}
"""


def _is_unverified(text: str) -> bool:
    """A juror result counts as UNVERIFIED unless it clearly says ANCHORED+VERIFIED.
    Default-to-reject: a malformed or errored juror answer is treated as a no-confirm."""
    low = (text or "").lower()
    if low.startswith("(") and "error" in low[:40]:
        return True                                  # model errored -> no confirmation
    m = re.search(r"verdict:\s*(anchored\+verified|anchored and verified|unverified|verified)", low)
    if m:
        tok = m.group(1)
        return "unverified" in tok or tok == "verified" and "anchored" not in low
    # No parseable verdict line -> reject.
    return "anchored+verified" not in low and "anchored and verified" not in low


def verify_claim(claim: str, evidence: str,
                 jurors: list[str] | None = None, judge: str | None = None) -> dict:
    """Run the grounding panel on one claim against one evidence packet (web off)."""
    jurors = jurors or VERIFY_JURORS
    judge = judge or VERIFY_JUDGE

    # Preflight: only call jurors whose endpoint actually answers. An unreachable
    # juror is EXCLUDED from the vote, never counted as a kill.
    reach = reachable_providers()
    live_jurors = [j for j in jurors if reach.get(j, False)]
    if len(live_jurors) < 2:
        return _not_reachable_result(reach, claim, judge)
    if not reach.get(judge, False):                    # judge endpoint down: pick a reachable one
        judge = next((j for j in ("grok", "openai", "gemini", "claude") if reach.get(j)),
                     live_jurors[0])
    jurors = live_jurors

    prompt = _JUROR_PROMPT.format(evidence=evidence, claim=claim)

    def _ask(name):
        fn = jury.JURORS.get(name, jury._claude)
        return name, fn(prompt, live=False)           # identical evidence, no web

    with ThreadPoolExecutor(max_workers=max(1, len(jurors))) as ex:
        results = dict(ex.map(_ask, jurors))

    n_unv = sum(1 for v in results.values() if _is_unverified(v))
    n_ver = len(jurors) - n_unv
    majority_unverified = n_unv > n_ver               # strict majority UNVERIFIED is a hard kill

    panel = "\n\n".join(f"--- {n.upper()} juror ---\n{v}" for n, v in results.items())
    judge_fn = jury.JUDGES.get(judge, jury.JURORS.get(judge, jury._grok))
    judge_out = judge_fn(_JUDGE_PROMPT.format(
        evidence=evidence, claim=claim, panel=panel,
        verified=n_ver, unverified=n_unv, total=len(jurors)))

    jl = (judge_out or "").lower()
    judge_killed = "final: killed" in jl
    # Kill on a strict juror majority UNVERIFIED, OR on the judge's backstop kill
    # (the judge is instructed to spare anchored, majority-verified claims absent a
    # contradicting evidence line, so a judge KILLED here means a real defect).
    killed = majority_unverified or judge_killed

    return {
        "ran": True,
        "claim": claim,
        "verdicts": results,
        "unverified_count": n_unv,
        "juror_total": len(jurors),
        "judge": judge,
        "judge_text": judge_out,
        "killed": killed,
    }


def verify_ledger(claims: list[dict], evidence_map: dict,
                  jurors: list[str] | None = None, judge: str | None = None) -> dict:
    """Verify each ledger claim against its evidence packet. Returns per-claim
    results plus the kill list that must be fixed before any text enters the form."""
    # One preflight for the whole ledger: if the jury cannot run here, say so once
    # and let the caller fall back to the Claude Agent-tool panel. Do not kill.
    reach = reachable_providers()
    if sum(1 for j in (jurors or VERIFY_JURORS) if reach.get(j, False)) < 2:
        return {"ran": False, "reachable": reach, "results": [], "killed": [],
                "all_clear": False, "claim_count": len(claims),
                "reason": _not_reachable_result(reach)["reason"]}
    out = []
    for c in claims:
        cid = c.get("id", "?")
        text = c.get("text", "")
        ev = evidence_map.get(cid) or evidence_map.get(text) or ""
        if not ev.strip():
            out.append({"id": cid, "claim": text, "killed": True,
                        "judge_text": "FINAL: KILLED\nREASON: no evidence packet supplied for this claim.\nFIX: attach the transcript excerpt and source rows behind this claim.",
                        "verdicts": {}, "unverified_count": 0, "juror_total": 0})
            continue
        r = verify_claim(text, ev, jurors, judge)
        r["id"] = cid
        out.append(r)
    killed = [r for r in out if r["killed"]]
    return {"ran": True, "results": out, "killed": [r["id"] for r in killed],
            "all_clear": not killed, "claim_count": len(out)}


def _fmt(r: dict) -> str:
    if r.get("ran") is False:
        return f"[JURY DID NOT RUN]  {r.get('reason','')}\n    reachable: {r.get('reachable', {})}"
    head = "KILLED" if r["killed"] else "SURVIVES"
    tally = f"{r['unverified_count']}/{r['juror_total']} jurors UNVERIFIED"
    lines = [f"[{head}]  ({tally})  claim: {r['claim'][:90]}"]
    for n, v in r.get("verdicts", {}).items():
        first = (v.strip().splitlines() or [""])[0]
        lines.append(f"    {n:7s}: {first[:100]}")
    jt = (r.get("judge_text") or "").strip().splitlines()
    if jt:
        lines.append(f"    JUDGE ({r['judge']}): {jt[0][:120]}")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Project Orwell multi-model grounding jury (Gate 4).")
    ap.add_argument("--claim", help="A single claim to verify.")
    ap.add_argument("--evidence", help="Path to the evidence packet for --claim (or for all --ledger claims).")
    ap.add_argument("--ledger", help="ledger.json; verifies every claim in claims[].")
    ap.add_argument("--evidence-dir", help="Directory with one <claim-id>.txt evidence packet per claim.")
    args = ap.parse_args(argv)

    if args.claim:
        ev = Path(args.evidence).read_text(encoding="utf-8") if args.evidence else ""
        r = verify_claim(args.claim, ev)
        print(_fmt(r))
        if r.get("ran") is False:
            return 3                                  # jury could not run; caller falls back
        print("\nJUDGE FULL:\n" + (r["judge_text"] or ""))
        return 1 if r["killed"] else 0

    if args.ledger:
        data = json.loads(Path(args.ledger).read_text(encoding="utf-8"))
        claims = data.get("claims", [])
        ev_map = {}
        if args.evidence_dir:
            d = Path(args.evidence_dir)
            for c in claims:
                p = d / f"{c.get('id','')}.txt"
                if p.exists():
                    ev_map[c.get("id")] = p.read_text(encoding="utf-8")
        elif args.evidence:
            shared = Path(args.evidence).read_text(encoding="utf-8")
            ev_map = {c.get("id"): shared for c in claims}
        res = verify_ledger(claims, ev_map)
        print("=" * 64)
        print("PROJECT ORWELL GATE 4: multi-model grounding jury")
        print("=" * 64)
        if res.get("ran") is False:
            print("JURY DID NOT RUN (no egress to the model endpoints from here).")
            print(res.get("reason", ""))
            print(f"reachable: {res.get('reachable', {})}")
            print("Action: run Gate 4 as the in-environment Claude Agent-tool grounding")
            print("panel instead (works without external egress), or hand the ledger to the")
            print("host-side jury where the model endpoints are reachable.")
            print("=" * 64)
            return 3
        for r in res["results"]:
            print(_fmt(r))
            print()
        print("=" * 64)
        if res["all_clear"]:
            print(f"ALL {res['claim_count']} CLAIMS SURVIVED. Gate 4 green.")
        else:
            print(f"KILLED: {res['killed']}. Re-derive/fix/drop these, then re-run Gates 1-4.")
        print("=" * 64)
        return 0 if res["all_clear"] else 1

    ap.error("provide --claim with --evidence, or --ledger with --evidence/--evidence-dir")


if __name__ == "__main__":
    sys.exit(main())
