# cod — clarification questions before decisions

Does asking a decision model clarification questions (yes/no or multiple choice), answered from the user's
history, make its decisions better? Tested on the 149 personalised web tasks of the Persona2Web benchmark,
with Jev as both the question answerer and the decider.

**Result: no.** No question format beat deciding without questions; at full scale, questions made slot
decisions measurably worse. See **[question_findings.md](question_findings.md)** for the full write-up.

## Layout

| Path | What |
|---|---|
| `question_findings.md` | Findings: setup, results, analysis, recommendations |
| `p2w_qbank_eval.py` | Pipeline: slots → candidates → questions → Jev answers → Jev decision → scoring |
| `question_bank.json` | Hand-written question bank (slot types and templates) |
| `p2w_qbank_analyze.py` | Per-question informativeness and cross-fitted "learned bank" |
| `p2w_qbank_rescore.py` | Re-score saved results with per-slot labels |
| `p2w_qbank_cache.json` | Frozen slots, candidates, generated questions, and per-slot labels for all 149 tasks |
| `oracle_cod/` | Oracle experiment: blind hand-written questions and verified answers |
| `results/` | Full 149-task runs and the 20-task pilots behind the findings |

## Setup

Requires Python 3.12 and the Persona2Web data checked out at `Persona2Web/browser-use/` (uses `task/` and the
`cache/*_memories.pkl` memory banks).

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=...       # candidate generation, gpt-4o judge
export TYPESAFE_API_KEY=...     # Jev
```

Optional local question generator (any OpenAI-compatible server, e.g. Ollama):

```bash
export QGEN_BASE_URL=http://localhost:11434/v1 QGEN_MODEL=qwen3.5:9b
```

## Reproduce

```bash
# full runs (candidates are read from p2w_qbank_cache.json, so pools are identical)
python p2w_qbank_eval.py --arms noq    --out results/p2w_qbank_full_noq.json      # no questions
python p2w_qbank_eval.py --arms q      --out results/p2w_qbank_full_q.json        # bank yes/no questions
python p2w_qbank_eval.py --arms choiceq --out results/p2w_qbank_full_choiceq.json # generated multiple choice

# 20-task pilot with every arm
python p2w_qbank_eval.py --tasks 20 --out results/pilot.json

# oracle experiment
python oracle_cod/run_oracle.py

# per-question analysis
python p2w_qbank_analyze.py results/p2w_qbank_full_q.json
```
