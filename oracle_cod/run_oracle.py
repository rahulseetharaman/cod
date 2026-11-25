"""
Oracle chain-of-decisions experiment.

24 slot decisions (12 the no-question Jev decider got wrong, 12 it got right), shuffled.
Questions and verified answers were written blind to labels, from the same 15 history entries Jev sees.

Conditions (identical evidence, identical Jev Choice decider):
  baseline  : task + history
  oracle_q  : task + history + oracle questions, answered by Jev
  oracle_qa : task + history + oracle questions with verified answers
Each condition is repeated REPS times (Jev is slightly non-deterministic).

Run from the repo root: browser_env/bin/python3 oracle_cod/run_oracle.py
"""

from __future__ import annotations

import asyncio
import glob
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import p2w_qbank_eval as E  # noqa: E402

REPS = 3
OUT = HERE / __import__("os").environ.get("ORACLE_OUT", "oracle_results.json")   # e.g. oracle_results_kev.json
CONDITIONS = ("baseline", "oracle_q", "oracle_qa")


def load_questions() -> dict:
    qs = {}
    for f in sorted(glob.glob(str(HERE / "oracle_questions_part*.json"))):
        qs.update(json.load(open(f)))
    return qs


async def run_case(case: dict, oq: list[dict]) -> dict:
    state, cands = case["state"], case["candidates"]
    question = f"Which value is the user's usual \"{case['phrase']}\" for this task?"
    texts = [q["q"] for q in oq]
    verified = [1.0 if q["a"] == "yes" else 0.0 for q in oq]
    qa_state = state + "\n\n" + E.trace_block(texts, verified)

    out = {c: [] for c in CONDITIONS}
    jev_answers = []
    for _ in range(REPS):
        answers = await E.jev_answer(state, texts)
        jev_answers.append(answers)
        q_state = state + "\n\n" + E.trace_block(texts, answers)
        picks = await asyncio.gather(E.jev_choose(state, question, cands),
                                     E.jev_choose(q_state, question, cands),
                                     E.jev_choose(qa_state, question, cands))
        for c, (pick, _probs) in zip(CONDITIONS, picks):
            out[c].append(pick)
    return {"case_id": case["case_id"], "picks": out, "jev_answers": jev_answers}


def majority(picks: list[str]) -> str:
    return Counter(picks).most_common(1)[0][0]


async def main():
    blind = {c["case_id"]: c for c in json.load(open(HERE / "cases_blind.json"))}
    labels = {c["case_id"]: c for c in json.load(open(HERE / "cases_with_labels.json"))}
    oq = load_questions()
    missing = sorted(set(blind) - set(oq))
    assert not missing, f"no oracle questions for {missing}"

    results = await asyncio.gather(*[run_case(blind[cid], oq[cid]) for cid in sorted(blind)])
    json.dump(results, open(OUT, "w"), indent=1)

    def correct(cid, pick):
        lab = labels[cid]
        return lab["cand_ok"][lab["candidates"].index(pick)] if pick in lab["candidates"] else False

    groups = {"pilot-wrong (12)": [c for c in labels if not labels[c]["baseline_ok"]],
              "pilot-right (12)": [c for c in labels if labels[c]["baseline_ok"]],
              "all (24)": list(labels)}
    res = {r["case_id"]: r for r in results}

    print(f"\nAccuracy, majority vote over {REPS} runs (mean single-run accuracy in brackets)")
    print(f"{'group':<18}" + "".join(f"{c:>20}" for c in CONDITIONS))
    for g, ids in groups.items():
        row = f"{g:<18}"
        for c in CONDITIONS:
            maj = sum(correct(i, majority(res[i]["picks"][c])) for i in ids) / len(ids)
            mean = sum(correct(i, p) for i in ids for p in res[i]["picks"][c]) / (len(ids) * REPS)
            row += f"{maj:>12.0%} [{mean:4.0%}]"
        print(row)

    print("\nFlips vs the re-run baseline (majority votes), all 24 cases:")
    for c in ("oracle_q", "oracle_qa"):
        fixed = [i for i in labels if correct(i, majority(res[i]["picks"][c]))
                 and not correct(i, majority(res[i]["picks"]["baseline"]))]
        broke = [i for i in labels if not correct(i, majority(res[i]["picks"][c]))
                 and correct(i, majority(res[i]["picks"]["baseline"]))]
        print(f"  {c:<10} fixed {len(fixed)} {fixed}   broke {len(broke)} {broke}")

    agree = tot = 0
    per_case = {}
    for cid in labels:
        ver = [1 if q["a"] == "yes" else 0 for q in oq[cid]]
        hits = []
        for run in res[cid]["jev_answers"]:
            for v, p in zip(ver, run):
                if p is None:
                    continue
                tot += 1
                ok = (p >= 0.5) == bool(v)
                agree += ok
                hits.append(ok)
        per_case[cid] = sum(hits) / len(hits) if hits else None
    print(f"\nJev's answers vs verified answers: {agree}/{tot} = {agree/tot:.0%} agree")
    worst = sorted((v, k) for k, v in per_case.items() if v is not None)[:5]
    print("  lowest agreement:", ", ".join(f"{k} {v:.0%}" for v, k in worst))


if __name__ == "__main__":
    asyncio.run(main())
