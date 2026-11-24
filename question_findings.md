# Clarification questions before decisions — findings

**Question:** if a decision model is given yes/no (or multiple-choice) clarification questions answered from
the user's history before it decides, does it decide better?

**Short answer:** no. Across every question format, generator, and selection rule tested, adding questions
never beat letting Jev decide without them. At full scale (149 tasks) the questions made slot decisions
measurably worse.

---

## Setup

- **Benchmark:** Persona2Web, 149 tasks. Each level-1 task has vague phrases ("my usual budget", "my area")
  whose true values are given in the level-3 version of the task.
- **Decisions per task:** the website, plus one value per vague phrase ("slot"). 421 slot decisions in total.
- **Candidates:** for each slot, gpt-4o-mini proposes up to 5 values copied from the user's history
  (15 entries retrieved with MiniLM). Website candidates are the sites in the retrieved history.
- **Decider:** Jev (`jev-1.14.0`) `Choice` over the candidates.
- **Answerer:** Jev answers each question from the same 15 history entries (`Noul` for yes/no,
  `Choice` for multiple choice).
- **Control:** `jc_noq`, the same Jev decider with no questions.
- **Metric:** share of slot decisions where Jev picks a correct candidate. Each slot is judged only against
  its own ground-truth value, and only slots whose candidate pool contains a correct value count
  (260 of 421 at full scale). Plan-level preference accuracy (GPT-4o judge) is also reported.
- **Offline only:** no browser runs; these measure decision quality, not task success.

Code: `p2w_qbank_eval.py` (pipeline and all arms), `question_bank.json` (hand-written bank),
`p2w_qbank_analyze.py` (per-question informativeness), `p2w_qbank_rescore.py` (per-slot re-scoring),
`oracle_cod/` (oracle experiment).

---

## Results

### Full runs — all 149 tasks (260 scored slot decisions)

| Arm | Slot decisions correct | Slots fixed / broke vs no questions | Websites fixed / broke |
|---|---|---|---|
| **No questions (`jc_noq`)** | **85.4%** | — | — |
| Bank yes/no questions (`jc_q`) | 81.5% | 4 / 14 (sign test p = 0.03) | 3 / 15 |
| Qwen multiple-choice questions (`jc_choiceq`) | 81.9% | 4 / 13 (p = 0.05) | 2 / 4 (p = 0.69) |

Run-to-run noise of the control: 5 of 260 picks change between identical runs (~2%), so these effects are
real.

### Pilots — 20 tasks (37 scored slot decisions)

| Arm | How questions were made | No questions | With questions | Fixed / broke |
|---|---|---|---|---|
| Bank yes/no | `question_bank.json` templates, full Q/A text in the prompt | 86.5% | 86.5% | — |
| Comparisons + requirement checks | "is A better supported than B", both orders; "meets k of n requirements" | 86.5% | 75.7% | — |
| Generated yes/no | gpt-4o-mini, up to 6 per decision | 86.5% | 81.1% | 1 / 3 |
| Generated, confident answers only | keep the 3–5 most confidently answered | 86.5% | 78.4% | — |
| Generated yes/no | qwen3.5:9b (Ollama), first prompt | 83.8% | 75.7% | 1 / 4 |
| Generated yes/no | qwen3.5:9b, prompt built from oracle question types | 86.5% | 78.4% | 0 / 3 |
| Multiple choice | qwen3.5:9b, 3–5 per decision | 83.8% | 81.1% | 1 / 2 |
| Mixed yes/no + multiple choice | qwen3.5:9b | 83.8% | 81.1% | 1 / 2 |
| Answers only, no decider call | argmax of hypothesis-question scores | ~86% | 54–70% | — |

The 83.8% vs 86.5% control values differ by one decision (Jev is slightly non-deterministic).

### Oracle experiment — 24 decisions

12 decisions Jev got wrong and 12 it got right, shuffled. 3–5 questions per decision were written by hand
(by Claude) **blind to the answer and to Jev's result**, from the same 15 history entries, each with a
verified answer citing entries.

| Condition | Result |
|---|---|
| Oracle questions, Jev answers them (Oracle-Q) | fixed 1, broke 0 |
| Oracle questions, verified answers given (Oracle-QA) | fixed 1, broke 0 (identical to Oracle-Q) |
| Jev's answers vs verified answers | **99% agree** (306 / 309) |

Of the 12 "errors", only 3 were genuine decision errors once scored per slot; the rest were scoring
artifacts or cases where the history does not support the ground truth.

---

## Findings

1. **No question format beats deciding without questions.** Yes/no questions hurt both slot and website
   decisions; multiple-choice questions no longer hurt website decisions but still hurt slot decisions.

2. **The answers carry no new information.** The same model answers from the same evidence the decider
   already reads. Answers favoured the option the decider already picked 66% of the time; otherwise they
   add framing noise.

3. **Jev answers well; the questions are the weak link.** With well-posed questions Jev's answers match
   verified ones 99% of the time, and handing it verified answers changes nothing. Hand-written questions
   broke no correct decisions; generated questions broke 2–5 per 37.

4. **How generated questions go wrong:**
   - **Yes-bias:** Jev answers yes to ~67% of yes/no questions (13% no), so whatever a question names tends
     to be confirmed.
   - **Non-discriminative questions:** ~30% of bank questions get the same answer for every option;
     "has the user ever…" questions are yes for everything.
   - **Leading, wording, and off-topic questions:** e.g. "is the area defined by proximity to the campus
     rather than the city?", "does the user write 'near' before landmarks?", seating questions while
     deciding a city.
   - **Detail over correctness:** many breaks swap a correct general value for a more detailed one the
     history emphasises ("Charlotte" → "near the UNC Charlotte campus", "same-day delivery" →
     "same-day service in San Francisco").

5. **Multiple choice is the best format tested.** Forcing a choice among named alternatives removes the
   yes-bias and left website decisions intact (2 fixed / 4 broken vs 3 / 15 for yes/no). It does not stop
   slot breaks driven by the history favouring a more detailed value.

6. **Selecting by answer confidence does not help.** A confident answer is not an informative one; keeping
   only the most confident answers changed one decision, for the worse.

7. **There is little headroom for questions here.** Without questions Jev is right on 85% of scorable slot
   decisions; the remaining limit is the candidate pool (the correct value is missing for 57 slots, capping
   plan-level preference accuracy at ~76%).

---

## Methodology notes

- **Per-slot scoring was essential.** The first scoring counted a candidate as correct if it matched *any*
  of the task's ground-truth preferences, which made some slots unwinnable and penalised values the ground
  truth bundles ("3–5 hours/week over 4–6 weeks" covers both workload and course length). All numbers
  above use the corrected per-slot scoring; 104 of 421 slot decisions have no ground-truth value and are
  excluded.
- **Candidates must be frozen.** Regenerating candidates moved the control by 13 points between runs.
  Candidates, slots, and labels are cached in `p2w_qbank_cache.json`.
- **Local generator settings:** qwen3.5:9b via Ollama rejects JSON mode, and its thinking step made calls
  25–260 s; `reasoning_effort="none"` brings this to 2–10 s.
- **Caveats:** offline, plan-level evaluation only; the oracle questions were written by a much stronger
  model than the generators, and two oracle cases had been seen earlier in the analysis.

---

## Recommendations

1. **Do not add pre-execution questions to the decision pipeline** on this benchmark.
2. **If continuing the idea, move questions into browsing**, answered from the live page — a source the
   decider has not seen — and use the multiple-choice format.
3. **Generate questions to separate the current top options** (expected information gain), chosen before
   looking at the evidence, rather than writing a fixed batch.
4. **Invest in candidate generation**, which now limits accuracy more than the decision does.
