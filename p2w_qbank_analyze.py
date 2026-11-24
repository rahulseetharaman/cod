"""
Which clarification questions are worth asking?  (analysis of a p2w_qbank_*.json results file)

1. Informativeness per question template, measured within a slot (so slot difficulty cancels):
   pairwise accuracy = P(score(correct candidate) > score(incorrect candidate)), 0.5 = useless.
     hypothesis / negative templates: score = Jev answer (negatives flipped)
     facet templates: score = agreement between Jev's answer from the history and from the candidate
2. Learned bank, cross-fitted by persona: pick templates with pairwise acc > THR on one half of
   the personas, decide on the other half with only those templates, swap, and pool.

Run: browser_env/bin/python3 p2w_qbank_analyze.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
RESULTS = json.load(open(sys.argv[1] if len(sys.argv) > 1 else ROOT / "results" / "p2w_qbank_full_q.json"))
THR = 0.55        # keep a template if it orders correct above incorrect candidates > 55% of the time
MIN_PAIRS = 8


def persona(tid: str) -> str:
    return "_".join(tid.split("_")[:2])


def decisions(results):
    """Yield (persona, decision record, correctness labels) for decisions with a contest to win."""
    for r in results:
        recs = list(r["slots"])
        if r["site"].get("questions"):
            recs.append(r["site"])
        for d in recs:
            ok = d.get("cand_ok", [])
            if len(d["candidates"]) > 1 and any(ok) and not all(ok):
                yield persona(r["task_id"]), d, ok


def template_scores(d) -> dict[str, list[float | None]]:
    """Per template id → per-candidate score (aligned with d['candidates'])."""
    cands, out = d["candidates"], {}
    for q in d["questions"]:
        if q["kind"] in ("hypothesis", "negative") and q.get("p") is not None:
            s = q["p"] if q["kind"] == "hypothesis" else 1 - q["p"]
            out.setdefault(q["tid"], [None] * len(cands))[cands.index(q["cand"])] = s
    facets = [q for q in d["questions"] if q["kind"] == "facet" and q.get("p") is not None]
    fca = d.get("facet_cand_answers") or {}
    for fi, q in enumerate(facets):
        p = q["p"]
        col = []
        for c in cands:
            ans = fca.get(c)
            cp = ans[fi] if ans and fi < len(ans) else None
            col.append(None if cp is None else p * cp + (1 - p) * (1 - cp))
        out[q["tid"]] = col
    return out


def pairwise(scores: list[float | None], ok: list[bool]) -> tuple[float, int]:
    wins = n = 0.0
    for i, si in enumerate(scores):
        for j, sj in enumerate(scores):
            if ok[i] and not ok[j] and si is not None and sj is not None:
                n += 1
                wins += 1.0 if si > sj else (0.5 if si == sj else 0.0)
    return wins, int(n)


def informativeness(decs) -> dict[str, tuple[float, int]]:
    agg = defaultdict(lambda: [0.0, 0])
    for _, d, ok in decs:
        for tid, sc in template_scores(d).items():
            w, n = pairwise(sc, ok)
            agg[tid][0] += w
            agg[tid][1] += n
    return {t: (w / n if n else 0.5, n) for t, (w, n) in agg.items()}


def decide(d, keep: set[str] | None) -> int:
    """Fuse the Jev-decider prior with the mean of kept template scores (log-linear)."""
    cands = d["candidates"]
    prior = d.get("jc_noq_probs") or []
    prior = np.asarray(prior if len(prior) == len(cands) else [1 / len(cands)] * len(cands), float)
    logit = np.log(np.clip(prior, 1e-4, 1))
    cols = [sc for t, sc in template_scores(d).items() if keep is None or t in keep]
    if cols:
        m = np.array([[np.nan if v is None else v for v in col] for col in cols], float)
        mean = np.nan_to_num(np.nanmean(m, axis=0), nan=0.5) if not np.all(np.isnan(m)) else np.full(len(cands), .5)
        logit = logit + np.log(np.clip(mean, 1e-4, 1)) * len(cols) ** 0.5
    return int(np.argmax(logit))


def main():
    decs = list(decisions(RESULTS))
    print(f"{len(RESULTS)} tasks, {len(decs)} contested decisions (>1 option, mixed correctness)\n")

    info = informativeness(decs)
    print(f"{'template':<28}{'pairwise acc':>13}{'pairs':>7}")
    for t, (acc, n) in sorted(info.items(), key=lambda kv: -kv[1][0]):
        if n >= MIN_PAIRS:
            bar = "█" * int(max(0, acc - 0.5) * 40)
            print(f"{t:<28}{acc:>13.1%}{n:>7}  {bar}")

    # Cross-fitted learned bank
    personas = sorted({p for p, _, _ in decs})
    halves = [set(personas[0::2]), set(personas[1::2])]
    res = defaultdict(list)
    kept_sizes = []
    for k in (0, 1):
        train = [x for x in decs if x[0] in halves[k]]
        test = [x for x in decs if x[0] in halves[1 - k]]
        tinfo = informativeness(train)
        keep = {t for t, (acc, n) in tinfo.items() if n >= MIN_PAIRS and acc > THR}
        kept_sizes.append(len(keep))
        for _, d, ok in test:
            res["jev_decider_noq"].append(ok[d["candidates"].index(d["jc_noq"])] if d.get("jc_noq") in d["candidates"] else False)
            res["jev_decider_q"].append(ok[d["candidates"].index(d["jc_q"])] if d.get("jc_q") in d["candidates"] else False)
            res["full_bank_fuse"].append(ok[decide(d, None)])
            res["learned_bank_fuse"].append(ok[decide(d, keep)])
            res["random"].append(sum(ok) / len(ok))
    print(f"\nCross-fitted by persona (kept {kept_sizes} templates per fold, thr {THR:.0%}):")
    for name, v in res.items():
        print(f"  {name:<20}{np.mean(v):7.1%}   (n={len(v)})")


if __name__ == "__main__":
    main()
