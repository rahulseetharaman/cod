"""
Re-score saved question-bank results with per-slot labels (each slot judged only against its own
ground-truth preference) instead of "matches ANY ground-truth preference". Decisions are not rerun.

Run: browser_env/bin/python3 p2w_qbank_rescore.py FILE [FILE ...] [--oracle]
Writes FILE with a _rescored suffix; labels are cached in p2w_qbank_cache.json.
"""

from __future__ import annotations

import asyncio
import json
import sys
from collections import Counter
from pathlib import Path

import p2w_qbank_eval as E

ROOT = Path(__file__).resolve().parent


async def rescore_file(path: Path) -> None:
    results = json.load(open(path))
    for r in results:
        await E.slot_labels(E._CACHE.setdefault(r["task_id"], {}), r["intent"], r["gt"], r["slots"])
    E.save_cache()
    out = path.with_name(path.stem + "_rescored.json")
    json.dump(results, open(out, "w"), indent=1)
    print(f"\n######## {path.name}  ->  {out.name}")
    E.summarise(results)


def rescore_oracle() -> None:
    """Oracle experiment: re-label the 24 cases with per-slot labels (from the cache filled above)."""
    d = ROOT / "oracle_cod"
    cases = json.load(open(d / "cases_with_labels.json"))
    res = {r["case_id"]: r for r in json.load(open(d / "oracle_results.json"))}
    maj = lambda ps: Counter(ps).most_common(1)[0][0]
    rows = []
    for c in cases:
        t = E._CACHE.get(c["task_id"], {})
        if t.get("slot_gt2", {}).get(c["phrase"]) is None:
            rows.append((c, None))
            continue
        lab = t["slot_labels2"][c["phrase"]]
        ok = {cond: lab.get(maj(res[c["case_id"]]["picks"][cond]), False)
              for cond in ("baseline", "oracle_q", "oracle_qa")}
        rows.append((c, ok if any(lab.get(x, False) for x in c["candidates"]) else None))
    scored = [(c, ok) for c, ok in rows if ok is not None]
    print(f"\n######## oracle experiment, per-slot labels: {len(scored)} of {len(rows)} cases scorable "
          f"({len(rows) - len(scored)} have no GT value for the slot or none in the pool)")
    for cond in ("baseline", "oracle_q", "oracle_qa"):
        acc = sum(ok[cond] for _, ok in scored) / len(scored)
        fixed = [c["case_id"] for c, ok in scored if ok[cond] and not ok["baseline"]]
        broke = [c["case_id"] for c, ok in scored if ok["baseline"] and not ok[cond]]
        print(f"  {cond:<10} {acc:6.1%}   fixed {fixed}  broke {broke}")


async def main():
    E._CACHE.update(E.load_cache())
    files = [Path(a) for a in sys.argv[1:] if not a.startswith("--")]
    for f in files:
        await rescore_file(f)
    if "--oracle" in sys.argv:
        rescore_oracle()


if __name__ == "__main__":
    asyncio.run(main())
