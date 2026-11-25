"""
Persona2Web — "planning as asking better questions" (offline, no browser).

For every decision (which site + one value per vague slot in the task):
  1. Retrieve history evidence (MiniLM)
  2. Propose candidate values, worded as in the history (gpt-4o-mini)
  3. Ask yes/no clarification questions from question_bank.json, answered by cloud Jev
  4. Decide with a Jev Choice call; all arms share the same candidates and the same answers:
       jc_noq  : Jev decider, prompt = task + evidence                      (control)
       jc_q    : Jev decider, prompt = task + evidence + yes/no Q/A text    (the idea)
       jev     : argmax of hypothesis-question scores, no decider call
       facet   : best agreement between facet answers from history and from each candidate
       jc_fuse : jc_noq probabilities x hypothesis scores x facet scores
Scored against level-3 ground truth: site accuracy + preference-value accuracy
(GPT-4o equivalence judge; also applied to the earlier Kev pre-exec plans for reference).

Run: browser_env/bin/python3 p2w_qbank_eval.py [--tasks N]
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import json
import os
import pickle
import re
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import AsyncOpenAI
from typesafe_sdk import Choice, Noul, TypeSafeClient

import sys
import types


class _MemoryItem:
    """Stand-in for the class pickled into Persona2Web's *_memories.pkl caches."""
    def __init__(self, data, index):
        self.data = data
        self.index = index


for _mod in ("retriever", "browser_use.personalization.retriever"):
    if _mod not in sys.modules:
        _m = types.ModuleType(_mod)
        _m.MemoryItem = _MemoryItem
        sys.modules[_mod] = _m

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")


def _jev_api_key() -> str:
    """TYPESAFE_API_KEY from the environment / .env (falls back to a local-only module if present)."""
    key = os.environ.get("TYPESAFE_API_KEY")
    if key:
        return key
    try:
        import p2w_jev_eval  # local-only, not part of the repo
        return p2w_jev_eval.TYPESAFE_API_KEY
    except ImportError:
        raise SystemExit("Set TYPESAFE_API_KEY (environment or .env) to use Jev.")

P2W_DIR   = ROOT / "Persona2Web" / "browser-use"
TASK_DIR  = P2W_DIR / "task"
CACHE_DIR = P2W_DIR / "cache"
BANK      = json.load(open(ROOT / "question_bank.json"))
OUT_FILE  = ROOT / "p2w_qbank_results.json"
CACHE_FILE = ROOT / "p2w_qbank_cache.json"   # gpt-4o-mini slots/candidates + judge labels, per task

LLM_MODEL   = "gpt-4o-mini"
JUDGE_MODEL = "gpt-4o"
SLOT_TYPES  = [k for k in BANK if not k.startswith("_") and k != "site"]
EVIDENCE_K  = 15
MAX_CANDS   = 5
MAX_SITES   = 8
JEV_CHUNK   = 12
ARMS        = ("jc_noq", "jc_q", "jc_cmp", "jc_gen", "jc_gen_top", "pair", "jev", "facet", "jc_fuse")
MODE        = "all"   # set by --arms: all | noq | q
PHANTOM_SITES = {"student", "weather", "transit", "bus", "campus", "local", "calendar",
                 "reminder", "notes", "google search", "google map", "google maps"}

# Context-specific question generator: any OpenAI-compatible endpoint. For a local model, e.g. Ollama:
#   QGEN_BASE_URL=http://localhost:11434/v1 QGEN_MODEL=qwen3.5:9b
# Local servers: no JSON mode (Ollama rejects it for some models) and thinking off by default
# (qwen3.5:9b: 25s -> 2.4s per call). QGEN_THINK=1 turns thinking back on.
QGEN_MODEL    = os.environ.get("QGEN_MODEL", "gpt-4o-mini")
QGEN_BASE_URL = os.environ.get("QGEN_BASE_URL")
QGEN_THINK    = os.environ.get("QGEN_THINK", "0") == "1"
MAX_GEN_QS    = 6
QGEN_PROMPT   = "v3-oracle-types"   # prompt version; part of the generated-question cache key

oai = AsyncOpenAI()
qgen = AsyncOpenAI(base_url=QGEN_BASE_URL, api_key=os.environ.get("QGEN_API_KEY", "local")) \
    if QGEN_BASE_URL else oai
_CACHE: dict = {}
JEV_MODEL = "jev-1.14.0"   # pinned so pilots stay comparable if the server default moves


def _typesafe_client(base_url: str | None) -> TypeSafeClient:
    """Cloud Jev (pinned model) by default; a local Kev server when a base URL is given."""
    if base_url:   # local server: generous timeout (the SDK default of 10 s fails under load)
        return TypeSafeClient(api_key=_jev_api_key(), base_url=base_url, timeout=120.0)
    return TypeSafeClient(api_key=_jev_api_key(), model=JEV_MODEL, timeout=60.0)


# Decider and question answerer can be different models, e.g. local Kev deciding with Jev answering:
#   DECIDER_BASE_URL=http://localhost:8009   ANSWERER_BASE_URL=http://localhost:8009
DECIDER_BASE_URL  = os.environ.get("DECIDER_BASE_URL")
ANSWERER_BASE_URL = os.environ.get("ANSWERER_BASE_URL")
decider  = _typesafe_client(DECIDER_BASE_URL)
answerer = _typesafe_client(ANSWERER_BASE_URL)


# ── Data ──────────────────────────────────────────────────────────────────────

def parse_gt(intent3: str) -> dict:
    site = None
    for pat in [r"\bAt\s+([\w.-]+\.\w{2,6})", r"\bon\s+([\w.-]+\.\w{2,6})"]:
        m = re.search(pat, intent3, re.IGNORECASE)
        if m:
            site = m.group(1).lower().rstrip(".,")
            break
    return {"website": site, "preferences": re.findall(r"\*\*([^*]+)\*\*", intent3)}


def load_pairs() -> list[dict]:
    def lv(n):
        out = {}
        for p in sorted(glob.glob(str(TASK_DIR / f"personalization_{n}" / "*.json"))):
            t = json.load(open(p))
            out[t["task_id"][:-1]] = t["intent"].replace("Query: ", "").strip()
        return out
    l1, l3 = lv(1), lv(3)
    pairs = []
    for base, intent in l1.items():
        if base not in l3:
            continue
        gt = parse_gt(l3[base])
        if gt["website"]:
            pairs.append({"task_id": base + "1", "persona": "_".join(base.split("_")[:2]),
                          "intent": intent, "gt": gt})
    return pairs


def norm_site(s: str) -> str:
    s = (s or "").strip().lower().replace("www.", "").rstrip("/")
    return s if "." in s else (s + ".com" if s else "")


def site_match(pred: str | None, gt: str) -> bool:
    if not pred:
        return False
    p, g = norm_site(pred), norm_site(gt)
    return p == g or p in g or g in p


# ── Retrieval (MiniLM, cached per persona) ────────────────────────────────────

_ENCODER = None
_PERSONA: dict[str, tuple[list[dict], np.ndarray]] = {}


def persona_memory(persona: str) -> tuple[list[dict], np.ndarray]:
    global _ENCODER
    if persona not in _PERSONA:
        if _ENCODER is None:
            from sentence_transformers import SentenceTransformer
            _ENCODER = SentenceTransformer("all-MiniLM-L6-v2")
        pkl = sorted(glob.glob(str(CACHE_DIR / f"{persona}_*_memories.pkl")))[0]
        items = [it.data for it in pickle.load(open(pkl, "rb"))["memory_items"]]
        corpus = [f"{d.get('type','')} {d.get('website','')} {d.get('object','')}" for d in items]
        emb = _ENCODER.encode(corpus, normalize_embeddings=True, batch_size=256, show_progress_bar=False)
        _PERSONA[persona] = (items, emb)
    return _PERSONA[persona]


def retrieve(persona: str, query: str, k: int) -> list[dict]:
    items, emb = persona_memory(persona)
    q = _ENCODER.encode([query], normalize_embeddings=True)[0]
    return [items[i] for i in np.argsort(emb @ q)[::-1][:k]]


def fmt_mem(m: dict, n: int = 180) -> str:
    return f"[{m.get('timestamp','')[:10]}] [{m.get('type','')}] {str(m.get('object',''))[:n]} (site: {m.get('website','')})"


# ── LLM helpers ───────────────────────────────────────────────────────────────

def _parse_json(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:            # small local models often wrap JSON in prose or fences
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            raise
        return json.loads(m.group(0))


async def llm_json(prompt: str, model: str = LLM_MODEL, client: AsyncOpenAI | None = None) -> dict:
    client = client or oai
    local = client is not oai
    err = None
    for attempt in range(3):
        try:
            if local:
                kwargs = {} if QGEN_THINK else {"extra_body": {"reasoning_effort": "none"}}
            else:
                kwargs = {"response_format": {"type": "json_object"}} if attempt < 2 else {}
            temp = 0.3 * attempt if local else 0     # temp 0 retries reproduce the same broken JSON
            r = await client.chat.completions.create(
                model=model, temperature=temp, messages=[{"role": "user", "content": prompt}], **kwargs)
            return _parse_json(r.choices[0].message.content)
        except Exception as e:  # transient API / parse errors; last try drops JSON mode for local servers
            err = e
            await asyncio.sleep(2)
    print(f"    [llm error] {err}", flush=True)
    return {}


async def generate_questions(intent: str, phrase: str, cands: list[str], evidence_txt: str) -> list[str]:
    """Cheap LLM writes yes/no questions specific to this decision; Jev answers them afterwards."""
    opts = "\n".join(f"  {i+1}. {c}" for i, c in enumerate(cands))
    prompt = f"""Task: "{intent}"
Decision: choose the right value for "{phrase}" from these options:
{opts}

{evidence_txt}

Write 3 to {MAX_GEN_QS} short yes/no questions that, once answered from the user's history or the task
wording, would best tell these options apart. Use these kinds of questions:
  1. Right kind of value: is an option the kind of value asked for, or something else?
     e.g. "Is '4.0+ stars' a rating requirement rather than a price?"
  2. Frequency / consistency: which value does the user state most often or consistently?
     e.g. "Does the user cap streaming rentals at $5 in most entries?"
  3. Context: does an option belong to a different item, trip, or occasion than this task?
     e.g. "Are the Lake Tahoe entries about a holiday trip rather than the user's home area?"
  4. Acted on vs looked at: did the user actually buy or book it, or only browse?
     e.g. "Did the user actually book a table for six?"
  5. What the task asks for: what kind of value does the vague phrase request?
     e.g. "Does 'my home gym' ask for a gym name rather than a neighbourhood?"
Rules:
  - Ask about meaning, never about wording: no questions about which words, phrases, abbreviations, or
    formats the user types (NOT "Does the user write 'near' before landmarks?").
  - One question, one claim: no "or" between alternatives, so it can be answered yes or no.
  - Neutral: do not frame a question to favour one option, and do not ask "is option N correct".
  - Stay on this decision: no questions about other parts of the task.
  - Refer to the user in the third person; do not answer the questions.

Return JSON: {{"questions": [str]}}"""
    out = await llm_json(prompt, QGEN_MODEL, qgen)
    qs = [str(q).strip() for q in out.get("questions", []) if str(q).strip()]
    return list(dict.fromkeys(qs))[:MAX_GEN_QS]


TOP_MIN, TOP_MAX, TOP_CONF = 3, 5, 0.8


def select_confident(questions: list[str], answers: list[float | None]) -> tuple[list[str], list[float]]:
    """Keep the 3–5 most confidently answered questions: up to 5 with p ≥ 0.8 or ≤ 0.2, topped up to 3."""
    ranked = sorted(((q, p) for q, p in zip(questions, answers) if p is not None),
                    key=lambda qp: -abs(qp[1] - 0.5))
    confident = [qp for qp in ranked if qp[1] >= TOP_CONF or qp[1] <= 1 - TOP_CONF][:TOP_MAX]
    keep = confident if len(confident) >= TOP_MIN else ranked[:TOP_MIN]
    order = {q: i for i, q in enumerate(questions)}          # keep the generator's original order
    keep = sorted(keep, key=lambda qp: order[qp[0]])
    return [q for q, _ in keep], [p for _, p in keep]


def trace_block(questions: list[str], answers: list[float | None]) -> str:
    lines = ["Clarifying questions answered from the user's history:"]
    for q, p in zip(questions, answers):
        if p is not None:
            lines.append(f"  Q: {q}\n  A: {_verdict(p)}")
    return "\n".join(lines) if len(lines) > 1 else ""


# ── Typed questions: yes/no (Noul) and multiple choice (Choice) ───────────────

TYPED_PROMPT = "typed-v1"
_TYPE_GUIDE = """Kinds of questions that work well:
  1. Right kind of value: is an option the kind of value asked for, or something else?
  2. Frequency / consistency: which value does the user state most often?
  3. Context: does a value belong to a different item, trip, or occasion than this task?
  4. Acted on vs looked at: did the user actually buy or book it, or only browse?
  5. What the task asks for: what kind of value does the vague phrase request (e.g. a city vs a venue)?"""

_TYPED_RULES = """Rules:
  - Ask about meaning, never about wording (no questions about which words or formats the user types).
  - Stay neutral and on this decision; do not ask "which option is correct".
  - Multiple-choice options must be about the history or the task, NOT the candidate values listed above
    (asking the user to pick among those options is just the decision again).
  - Multiple-choice options must cover the reasonable alternatives (2 to 5 options); include
    "none of these" when the alternatives may not be exhaustive.
  - Refer to the user in the third person; do not answer the questions."""


async def generate_typed_questions(intent: str, phrase: str, cands: list[str], evidence_txt: str,
                                   mode: str) -> list[dict]:
    """mode 'choice': multiple-choice only; 'mixed': the generator picks yes/no or multiple choice per question."""
    opts = "\n".join(f"  {i+1}. {c}" for i, c in enumerate(cands))
    if mode == "choice":
        form = ('Write 3 to 5 MULTIPLE-CHOICE questions.\n'
                'Return JSON: {"questions": [{"type": "choice", "q": str, "options": [str, ...]}]}')
        example = ('e.g. {"type": "choice", "q": "What kind of place does \'my home gym\' ask for?", '
                   '"options": ["a gym name", "a neighbourhood", "a city", "none of these"]}')
    else:
        form = ('Write 3 to 6 questions. For each, choose the better form: yes/no for a single claim, '
                'multiple choice when there are several alternatives to tell apart.\n'
                'Return JSON: {"questions": [{"type": "yesno", "q": str} | '
                '{"type": "choice", "q": str, "options": [str, ...]}]}')
        example = ('e.g. {"type": "yesno", "q": "Did the user actually book a table for six?"} and '
                   '{"type": "choice", "q": "What are the user\'s Lake Tahoe entries mostly about?", '
                   '"options": ["their home area", "a holiday trip", "work travel"]}')
    prompt = f"""Task: "{intent}"
Decision: choose the right value for "{phrase}" from these options:
{opts}

{evidence_txt}

Write clarifying questions that, once answered from the user's history or the task wording, would best tell
these options apart.
{_TYPE_GUIDE}
{_TYPED_RULES}
{form}
{example}"""
    out = await llm_json(prompt, QGEN_MODEL, qgen)
    cand_norm = {_norm_txt(c) for c in cands}
    qs, seen = [], set()
    for item in out.get("questions", []):
        if not isinstance(item, dict) or not str(item.get("q", "")).strip():
            continue
        q = str(item["q"]).strip()
        if q in seen:
            continue
        options = [str(o).strip() for o in item.get("options") or [] if str(o).strip()]
        is_choice = item.get("type") == "choice" or (mode == "choice")
        if is_choice:
            options = list(dict.fromkeys(options))[:5]
            if len(options) < 2:
                continue
            # enforce: options must not simply be the candidate values (that re-asks the decision)
            if sum(_norm_txt(o) in cand_norm for o in options) >= 2:
                continue
            qs.append({"type": "choice", "q": q, "options": options})
        else:
            # a yes/no question offering two alternatives ("X, or is it Y?") has no meaningful yes/no answer
            if re.search(r",\s*or\b|\bor\s+(is|are|does|do|did|was|were|has|have|strictly|specifically)\b", q, re.I):
                continue
            qs.append({"type": "yesno", "q": q})
        seen.add(q)
    return qs[:MAX_GEN_QS]


def _jev_typed_sync(state: str, questions: list[dict]) -> list:
    """Batched: Noul for yes/no (-> float), Choice for multiple choice (-> {option: prob})."""
    out: list = []
    for i in range(0, len(questions), JEV_CHUNK):
        chunk = questions[i:i + JEV_CHUNK]
        spec = {}
        for j, q in enumerate(chunk):
            if q["type"] == "choice":
                spec[f"q{j}"] = Choice(instructions=q["q"],
                                       criteria={f"o{k}": o for k, o in enumerate(q["options"])})
            else:
                spec[f"q{j}"] = Noul(instructions=q["q"])
        try:
            r = call_with_retry(answerer, state, spec, "typed answer")
            for j, q in enumerate(chunk):
                if q["type"] == "choice":
                    probs = r.choices[f"q{j}"].probabilities
                    out.append({o: float(probs.get(f"o{k}", 0.0)) for k, o in enumerate(q["options"])})
                else:
                    out.append(float(r.nouls[f"q{j}"].noul))
        except Exception:
            out += [None] * len(chunk)
    return out


async def jev_answer_typed(state: str, questions: list[dict]) -> list:
    return await asyncio.to_thread(_jev_typed_sync, state, questions) if questions else []


def typed_trace_block(questions: list[dict], answers: list) -> str:
    lines = ["Clarifying questions answered from the user's history:"]
    for q, a in zip(questions, answers):
        if a is None:
            continue
        if q["type"] == "choice":
            best = max(a, key=a.get)
            lines.append(f"  Q: {q['q']} (options: {'; '.join(q['options'])})\n  A: {best} ({a[best]:.2f})")
        else:
            lines.append(f"  Q: {q['q']}\n  A: {_verdict(a)}")
    return "\n".join(lines) if len(lines) > 1 else ""


async def extract_slots(intent: str) -> dict:
    prompt = f"""A user gave a web task that refers to their personal history with vague phrases.
Task: "{intent}"

List every vague reference that can only be resolved from the user's history (e.g. "my usual city",
"my typical budget", "my preferred restaurant"). Do NOT list things the task already states explicitly.
Keep compound phrases as ONE slot (e.g. "my usual waist/inseam" is one slot, not two).
Give each a type from: {SLOT_TYPES}.
Also report the website if the task names one explicitly (e.g. "On IKEA" -> "ikea.com"), else null.

Return JSON: {{"explicit_site": str|null, "slots": [{{"phrase": str, "type": str}}]}}"""
    out = await llm_json(prompt)
    slots = [s for s in out.get("slots", []) if s.get("phrase")]
    for s in slots:
        if s.get("type") not in SLOT_TYPES:
            s["type"] = "attributes"
    return {"explicit_site": out.get("explicit_site"), "slots": slots[:5]}


_VAGUE = re.compile(r"\bmy\s+(usual|typical|preferred|regular|favou?rite|go-to)\b", re.I)


async def propose_candidates(intent: str, phrase: str, kind: str, others: list[str],
                             evidence: list[dict]) -> list[str]:
    lines = "\n".join(f"{i+1}. {fmt_mem(m, 220)}" for i, m in enumerate(evidence))
    other_txt = "; ".join(others) if others else "none"
    prompt = f"""Task: "{intent}"
Vague reference to resolve: "{phrase}"  (kind of value: {BANK[kind]['description']})
Other vague parts of the task, resolved separately — do NOT propose values for these: {other_txt}

User history:
{lines}

Propose up to {MAX_CANDS} distinct candidate values for "{phrase}" supported by this history.
Every candidate must be a value of the kind above, never the vague phrase itself.
Copy wording from the history as closely as possible and keep every qualifier, number, unit, and
combined condition (e.g. "4.4+ stars with 300+ reviews", "32x32", "$35-$70 per person").
Where the history supports it, include candidates at different levels of detail.

Return JSON: {{"candidates": [str]}}"""
    out = await llm_json(prompt)
    seen, cands = set(), []
    for c in out.get("candidates", []):
        c = str(c).strip()
        if c and c.lower() not in seen and not _VAGUE.search(c) and c.lower() != phrase.lower():
            seen.add(c.lower())
            cands.append(c)
    return cands[:MAX_CANDS]


# ── Jev: answer clarification questions ───────────────────────────────────────

CALL_RETRIES = (2, 5, 15)      # seconds between attempts; 4 attempts in total


def call_with_retry(client: TypeSafeClient, state: str, spec: dict, label: str):
    """system_one with retries. Raises after the last attempt — callers must record the failure, never guess."""
    import time as _t
    for attempt in range(len(CALL_RETRIES) + 1):
        try:
            return client.system_one(state, spec)
        except Exception as e:
            if attempt == len(CALL_RETRIES):
                print(f"    [{label} failed after {attempt + 1} attempts] {e}", flush=True)
                raise
            _t.sleep(CALL_RETRIES[attempt])


def _jev_answer_sync(state: str, questions: list[str]) -> list[float | None]:
    out: list[float | None] = []
    for i in range(0, len(questions), JEV_CHUNK):
        chunk = questions[i:i + JEV_CHUNK]
        try:
            r = call_with_retry(answerer, state, {f"q{j}": Noul(instructions=q) for j, q in enumerate(chunk)},
                                "answer")
            out += [float(r.nouls[f"q{j}"].noul) for j in range(len(chunk))]
        except Exception:
            out += [None] * len(chunk)
    return out


async def jev_answer(state: str, questions: list[str]) -> list[float | None]:
    return await asyncio.to_thread(_jev_answer_sync, state, questions)


def _jev_choose_sync(state: str, question: str, options: list[str]) -> list[float]:
    try:
        r = call_with_retry(decider, state, {"pick": Choice(
            instructions=question, criteria={f"o{i}": o for i, o in enumerate(options)})}, "decide")
        probs = r.choices["pick"].probabilities
        return [float(probs.get(f"o{i}", 0.0)) for i in range(len(options))]
    except Exception:
        return []


async def jev_choose(state: str, question: str, options: list[str]) -> tuple[str | None, list[float]]:
    """Jev as the decision maker: one Choice call, full per-option distribution.
    On a failed call returns (None, []) — a failure is never turned into a pick."""
    if not options:
        return None, []
    if len(options) == 1:
        return options[0], [1.0]
    probs = await asyncio.to_thread(_jev_choose_sync, state, question, options)
    if not probs:
        return None, []
    return options[int(np.argmax(probs))], probs


def build_questions(kind: str, task: str, slot_phrase: str, cands: list[str]) -> list[dict]:
    """Instantiate bank templates. Each question records its template id for later analysis."""
    spec = BANK[kind]
    qs = []
    for t_i, tmpl in enumerate(spec.get("facet", [])):
        qs.append({"tid": f"{kind}.facet.{t_i}", "kind": "facet", "cand": None,
                   "q": tmpl.replace("{slot}", slot_phrase)})
    for group in ("hypothesis", "negative"):
        for t_i, tmpl in enumerate(spec.get(group, [])):
            for c in cands:
                qs.append({"tid": f"{kind}.{group}.{t_i}", "kind": group, "cand": c,
                           "q": tmpl.replace("{cand}", c).replace("{slot}", slot_phrase)
                                    .replace("{task}", task)})
    return qs


def _verdict(p: float) -> str:
    return "yes" if p >= 0.65 else ("no" if p <= 0.35 else "unclear")


def qa_block(qs: list[dict], cands: list[str]) -> str:
    """Yes/no answers as plain text for the decider: general questions first, then each option's own."""
    out = ["Clarifying questions answered from the user's history:"]
    facets = [q for q in qs if q["kind"] == "facet" and q.get("p") is not None]
    if facets:
        out.append("About the user's preference in general:")
        out += [f"  Q: {q['q']}\n  A: {_verdict(q['p'])}" for q in facets]
    for i, c in enumerate(cands):
        mine = [q for q in qs if q["cand"] == c and q.get("p") is not None]
        if mine:
            out.append(f"About option {i+1} (\"{c}\"):")
            out += [f"  Q: {q['q']}\n  A: {_verdict(q['p'])}" for q in mine]
    return "\n".join(out)


PAIR_TEMPLATES = {
    "support":  "Does the user's history support \"{a}\" better than \"{b}\" as the user's {slot}?",
    "complete": "Does \"{a}\" capture more of the specific numbers, qualifiers, and names that the user's "
                "history gives for {slot} than \"{b}\" does?",
}
SITE_PAIR_TEMPLATES = {
    "support":  "Based on the user's history, is \"{a}\" more likely than \"{b}\" to be the site where the user "
                "would do this task?",
    "complete": "Has the user done this specific kind of task on \"{a}\" more directly than on \"{b}\"?",
}
MAX_PAIR_CANDS = 5
FACET_CONF = 0.8      # only confident history answers become requirements


async def pair_compare(state: str, slot_phrase: str, cands: list[str], site: bool) -> dict:
    """
    Fix 1 — comparative questions. Every ordered pair (a, b) of the first MAX_PAIR_CANDS options is asked
    both ways; P(a ≻ b) = (p(a,b) + 1 − p(b,a)) / 2 cancels position bias.
    Returns {template: {a: {b: P(a≻b)}}}.
    """
    opts = cands[:MAX_PAIR_CANDS]
    templates = SITE_PAIR_TEMPLATES if site else PAIR_TEMPLATES
    asks = [(t, a, b) for t in templates for a in opts for b in opts if a != b]
    texts = [templates[t].replace("{a}", a).replace("{b}", b).replace("{slot}", f'"{slot_phrase}"')
             for t, a, b in asks]
    answers = await jev_answer(state, texts)
    raw = {(t, a, b): p for (t, a, b), p in zip(asks, answers)}
    out: dict = {t: {a: {} for a in opts} for t in templates}
    for t in templates:
        for a in opts:
            for b in opts:
                if a == b:
                    continue
                pab, pba = raw.get((t, a, b)), raw.get((t, b, a))
                if pab is None or pba is None:
                    continue
                out[t][a][b] = (pab + 1 - pba) / 2
    return out


def pair_scores(pairs: dict, cands: list[str]) -> list[float]:
    """Mean P(option beats another) over templates and opponents; options beyond the cap get 0.5."""
    scores = []
    for c in cands:
        vals = [v for t in pairs.values() for v in t.get(c, {}).values()]
        scores.append(float(np.mean(vals)) if vals else 0.5)
    return scores


def comparison_block(qs: list[dict], cands: list[str], pairs: dict, facet_detail: dict, site: bool) -> str:
    """
    Prompt for the jc_cmp decider: fix 2 (facet requirements checked against each option)
    + fix 1 (pairwise comparison tallies). Isolated per-option yes/no answers are left out.
    """
    out: list[str] = []
    facets = [q for q in qs if q["kind"] == "facet" and q.get("p") is not None]
    reqs = [(i, q) for i, q in enumerate(facets) if q["p"] >= FACET_CONF or q["p"] <= 1 - FACET_CONF]
    if reqs:
        out.append("Requirements, answered from the user's history:")
        for k, (_, q) in enumerate(reqs, 1):
            out.append(f"  R{k}. {q['q']} -> {'yes' if q['p'] >= 0.5 else 'no'}")
        out.append("Which requirements each option meets:")
        for i, c in enumerate(cands):
            ans = facet_detail.get(c) or []
            marks, met = [], 0
            for k, (fi, q) in enumerate(reqs, 1):
                cp = ans[fi] if fi < len(ans) else None
                if cp is None:
                    marks.append(f"R{k} ?")
                    continue
                ok = (cp >= 0.5) == (q["p"] >= 0.5)
                met += ok
                marks.append(f"R{k} {'meets' if ok else 'FAILS'}")
            out.append(f"  Option {i+1} \"{c}\": meets {met} of {len(reqs)} ({', '.join(marks)})")
    labels = ({"support": "more likely than", "complete": "more directly used than"} if site else
              {"support": "better supported than", "complete": "more complete and specific than"})
    opts = cands[:MAX_PAIR_CANDS]
    if len(opts) > 1 and any(pairs.get(t) for t in labels):
        out.append("Head-to-head comparisons, answered from the user's history:")
        for i, c in enumerate(opts):
            parts = []
            for t, label in labels.items():
                wins = sum(v > 0.5 for v in pairs.get(t, {}).get(c, {}).values())
                parts.append(f"{label} {wins} of {len(opts) - 1} other options")
            out.append(f"  Option {i+1} \"{c}\": {'; '.join(parts)}")
    return "\n".join(out)


def jev_scores(qs: list[dict], cands: list[str]) -> list[float]:
    """Per-candidate support in [0,1] from hypothesis (+) and negative (−) questions."""
    scores = []
    for c in cands:
        pos = [q["p"] for q in qs if q["cand"] == c and q["kind"] == "hypothesis" and q.get("p") is not None]
        neg = [q["p"] for q in qs if q["cand"] == c and q["kind"] == "negative" and q.get("p") is not None]
        s = np.mean(pos) if pos else 0.5
        if neg:
            s = (s + 1 - np.mean(neg)) / 2
        scores.append(float(s))
    return scores


async def facet_scores(phrase: str, qs: list[dict], cands: list[str]) -> tuple[list[float], dict]:
    """
    20-questions matching: each facet question is answered twice by Jev —
    from the history (what the user's value looks like) and from each candidate (what this value
    looks like). Score = confidence-weighted agreement. Uncertain history answers carry no weight.
    """
    facets = [q for q in qs if q["kind"] == "facet" and q.get("p") is not None]
    if not facets:
        return [0.5] * len(cands), {}
    texts = [q["q"] for q in facets]
    per_cand = await asyncio.gather(*[
        jev_answer(f"The user's value for \"{phrase}\" is: {c}", texts) for c in cands])
    scores, detail = [], {}
    for c, cps in zip(cands, per_cand):
        num = den = 0.0
        for q, cp in zip(facets, cps):
            if cp is None:
                continue
            w = abs(2 * q["p"] - 1)
            num += w * (q["p"] * cp + (1 - q["p"]) * (1 - cp))
            den += w
        scores.append(num / den if den else 0.5)
        detail[c] = [None if x is None else round(x, 3) for x in cps]
    return scores, detail


def fuse(*score_lists: list[float]) -> int:
    """Product of experts over aligned per-candidate scores."""
    logs = np.sum([np.log(np.clip(np.asarray(s, float), 1e-4, 1)) for s in score_lists if len(s)], axis=0)
    return int(np.argmax(logs))


# ── Judge ─────────────────────────────────────────────────────────────────────

JUDGE_RULES = """A preference counts as matched only if the text states an equivalent value: same numbers, units,
entities, and every component. Weaker, broader, or different values do NOT match
(e.g. "4.5 stars" does not match "4.2+ stars with 50+ reviews"; "32-inch waist" does not match "32x32";
"Taylor Swift" does not match "Beyonce"). Paraphrases with identical meaning DO match
(e.g. "table for four" matches "party of 4")."""


async def judge_plan(plan: str, prefs: list[str]) -> list[bool]:
    if not prefs:
        return []
    prompt = f"""{JUDGE_RULES}

Text:
\"\"\"{plan}\"\"\"

Ground-truth preferences: {json.dumps(prefs)}
For each preference in order, is it matched by the text?
Return JSON: {{"matches": [bool, ...]}}"""
    out = await llm_json(prompt, JUDGE_MODEL)
    m = out.get("matches", [])
    return [bool(x) for x in m][:len(prefs)] + [False] * max(0, len(prefs) - len(m))


async def label_candidates(cands: list[str], prefs: list[str]) -> list[bool]:
    """Which candidates equal some ground-truth preference (used for question-informativeness analysis)."""
    if not cands or not prefs:
        return [False] * len(cands)
    prompt = f"""{JUDGE_RULES}

Ground-truth preferences: {json.dumps(prefs)}
Candidates: {json.dumps(cands)}
For each candidate in order, does it match ANY ground-truth preference on its own?
Return JSON: {{"matches": [bool, ...]}}"""
    out = await llm_json(prompt, JUDGE_MODEL)
    m = out.get("matches", [])
    return [bool(x) for x in m][:len(cands)] + [False] * max(0, len(cands) - len(m))


# ── Per-slot scoring (replaces "matches ANY ground-truth preference") ─────────

def _ascii(s: str) -> str:
    """En/em dashes come back from the judge as control characters; send plain hyphens instead."""
    return re.sub(r"[–—]", "-", str(s))


def _norm_txt(s: str) -> str:
    s = re.sub(r"[\x00-\x1f]", "-", _ascii(s))
    return re.sub(r"\s+", " ", s).strip().lower()


def _match_pref(text: str, prefs: list[str]) -> str | None:
    """Exact (normalised) match first, then the closest preference if it is clearly the same string."""
    import difflib
    by_norm = {_norm_txt(p): p for p in prefs}
    t = _norm_txt(text)
    if t in by_norm:
        return by_norm[t]
    best = max(prefs, key=lambda p: difflib.SequenceMatcher(None, t, _norm_txt(p)).ratio(), default=None)
    if best and difflib.SequenceMatcher(None, t, _norm_txt(best)).ratio() >= 0.85:
        return best
    return None


async def map_slots_to_gt(intent: str, phrases: list[str], prefs: list[str]) -> dict[str, dict | None]:
    """
    For each vague phrase: the ground-truth preference that resolves it and the part of it that belongs to
    this phrase ({"pref": str, "value": str}), or None if the GT has no value for it.
    """
    if not phrases:
        return {}
    if not prefs:
        return {p: None for p in phrases}
    prompt = f"""Task with vague references: "{intent}"
Vague phrases: {json.dumps(phrases)}
Ground-truth preference values for this task (resolved version): {json.dumps([_ascii(p) for p in prefs])}

For each vague phrase, find the ground-truth preference that resolves it and copy it exactly as "pref".
Then give "value": only the part of that preference that answers this phrase. Two phrases may share one
bundled preference, e.g. "weekly workload" -> value "3-5 hours/week" and "course length" -> value
"4-6 weeks", both from pref "3-5 hours/week over 4-6 weeks". If the preference answers the phrase as a whole,
value = pref. If no preference is the value of that phrase, return null for it; do not force a match
(a phrase about budget has no match if no preference is a price).

Return JSON keyed by the exact phrase: {{"mapping": {{"<phrase>": {{"pref": str, "value": str}} | null, ...}}}}"""
    out = await llm_json(prompt, JUDGE_MODEL)
    m = out.get("mapping", {})
    m = m if isinstance(m, dict) else {}
    m_norm = {_norm_txt(k): v for k, v in m.items()}
    res: dict[str, dict | None] = {}
    for ph in phrases:
        e = m_norm.get(_norm_txt(ph))
        pref = _match_pref(e.get("pref", ""), prefs) if isinstance(e, dict) else None
        value = re.sub(r"[\x00-\x1f]", "-", str(e.get("value") or pref)) if pref else None
        res[ph] = {"pref": pref, "value": value} if pref else None
    return res


async def label_slot_candidates(intent: str, phrase: str, gt: dict, cands: list[str]) -> list[bool]:
    """Is each candidate the ground-truth value for THIS slot? gt = {"pref": full GT preference, "value": slot part}."""
    if not cands:
        return []
    prompt = f"""Task: "{intent}"
Vague phrase being resolved: "{phrase}"
Ground-truth value for this phrase: "{gt['value']}"
(taken from the ground-truth preference "{gt['pref']}")

For each candidate value for "{phrase}", is it the ground-truth value for this phrase?
CORRECT if it states "{gt['value']}" with all of its components (paraphrase is fine). It may also include
other parts of the full preference "{gt['pref']}". A stated budget amount equals a cap at that amount
("$100" budget = "<= $100").
WRONG if it is a different value, drops a component (e.g. "4.2+ stars" when the value is
"4.2+ stars with 50+ reviews"), is not a value of this kind at all, or adds a restriction that changes what
would be searched (e.g. a specific venue or campus when the value is a city).

Candidates: {json.dumps(cands)}
Return JSON: {{"correct": [bool, ...]}}  (same order)"""
    out = await llm_json(prompt, JUDGE_MODEL)
    m = out.get("correct", [])
    return [bool(x) for x in m][:len(cands)] + [False] * max(0, len(cands) - len(m))


async def slot_labels(task_cache: dict, intent: str, gt: dict, slot_recs: list[dict]) -> None:
    """Attach per-slot labels (cached): r['gt_pref'] and r['cand_ok'] (old any-pref labels kept as cand_ok_any)."""
    phrases = [r["phrase"] for r in slot_recs]
    mapping = task_cache.setdefault("slot_gt2", {})
    todo = [p for p in phrases if p not in mapping]
    if todo:
        mapping.update(await map_slots_to_gt(intent, todo, gt["preferences"]))
    labels = task_cache.setdefault("slot_labels2", {})
    for r in slot_recs:
        g = mapping.get(r["phrase"])
        r["gt_pref"] = g["value"] if g else None
        if "cand_ok" in r and "cand_ok_any" not in r:
            r["cand_ok_any"] = r["cand_ok"]
        if g is None:
            r["cand_ok"] = [False] * len(r["candidates"])
            continue
        known = labels.setdefault(r["phrase"], {})
        new = [c for c in r["candidates"] if c not in known]
        if new:
            for c, ok in zip(new, await label_slot_candidates(intent, r["phrase"], g, new)):
                known[c] = ok
        r["cand_ok"] = [known[c] for c in r["candidates"]]


# ── Per-task pipeline ─────────────────────────────────────────────────────────

async def decide_site(persona, intent, explicit, gen_cache) -> dict:
    evidence = retrieve(persona, intent, 30)
    freq: dict[str, int] = {}
    for m in evidence:
        raw = str(m.get("website", "")).strip().lower()
        if raw and raw not in PHANTOM_SITES and not (" " in raw and "." not in raw):
            s = norm_site(raw)
            freq[s] = freq.get(s, 0) + 1
    cands = sorted(freq, key=lambda s: -freq[s])[:MAX_SITES]
    rec = {"phrase": "website", "type": "site", "candidates": cands, "explicit": explicit}
    if explicit:
        rec.update({a: norm_site(explicit) for a in ARMS}, questions=[])
        return rec
    if not cands:
        rec.update({a: None for a in ARMS}, questions=[])
        return rec
    # Evidence: per-site top entries so every candidate has support visible
    shown, lines = set(), []
    for s in cands:
        for m in [m for m in evidence if norm_site(str(m.get("website", ""))) == s][:3]:
            if id(m) not in shown:
                shown.add(id(m))
                lines.append(fmt_mem(m))
    state = f"Task: {intent}\n\nUser history:\n" + "\n".join(lines[:24])
    question = "Which website should this task be completed on, based on the user's history?"
    if MODE == "noq":
        rec["jc_noq"], rec["jc_noq_probs"] = await jev_choose(state, question, cands)
        rec["questions"] = []
        return rec
    if MODE == "typed":
        await _decide_typed(rec, state, question, cands, "website", intent, gen_cache)
        return rec
    qs = build_questions("site", intent, "website", cands)
    for q, p in zip(qs, await jev_answer(state, [q["q"] for q in qs])):
        q["p"] = p
    if MODE == "q":
        await _decide_q(rec, state, question, cands, qs)
        return rec
    await _decide(rec, state, question, cands, qs, facet=([0.5] * len(cands), {}), phrase="website",
                  gen_cache=gen_cache, intent=intent, site=True)
    return rec


async def _decide_typed(rec, state, question, cands, phrase, intent, gen_cache) -> None:
    """--arms typed: jc_noq vs jc_choiceq (multiple-choice questions) vs jc_mixed (yes/no + multiple choice)."""
    traces = {}
    for arm, mode in (("jc_choiceq", "choice"), ("jc_mixed", "mixed")):
        if arm not in ARMS:
            continue
        key = f"{phrase}|{mode}"
        if len(cands) > 1 and key not in gen_cache:
            gen_cache[key] = await generate_typed_questions(intent, phrase, cands, state, mode)
        qs = gen_cache.get(key, []) if len(cands) > 1 else []
        answers = await jev_answer_typed(state, qs)
        traces[arm] = (qs, answers, typed_trace_block(qs, answers))
    picks = await asyncio.gather(
        jev_choose(state, question, cands),
        *[jev_choose(state + ("\n\n" + t if t else ""), question, cands) for _, _, t in traces.values()])
    rec["jc_noq"], rec["jc_noq_probs"] = picks[0]
    for (arm, (qs, answers, _)), (pick, probs) in zip(traces.items(), picks[1:]):
        rec[arm], rec[f"{arm}_probs"] = pick, probs
        rec[f"{arm}_trace"] = [{"type": q["type"], "q": q["q"], "options": q.get("options"), "a": a}
                               for q, a in zip(qs, answers)]
    rec["questions"] = []


async def _decide_q(rec, state, question, cands, qs) -> None:
    """--arms q: the no-question decider and the bank-question decider (jc_q) on the same prompt/evidence."""
    qa_state = state + "\n\n" + qa_block(qs, cands)
    (rec["jc_noq"], rec["jc_noq_probs"]), (rec["jc_q"], rec["jc_q_probs"]) = await asyncio.gather(
        jev_choose(state, question, cands), jev_choose(qa_state, question, cands))
    rec["questions"] = qs


async def _decide(rec, state, question, cands, qs, facet, phrase, gen_cache, intent, site=False) -> None:
    fs, fdetail = facet
    pairs = await pair_compare(state, phrase, cands, site) if len(cands) > 1 else {}
    qa_state = state + "\n\n" + qa_block(qs, cands)
    cmp_txt = comparison_block(qs, cands, pairs, fdetail, site)
    cmp_state = state + ("\n\n" + cmp_txt if cmp_txt else "")

    # jc_gen: cheap LLM asks context-specific questions -> Jev answers yes/no -> trace -> Jev decides
    if len(cands) > 1 and phrase not in gen_cache:
        gen_cache[phrase] = await generate_questions(intent, phrase, cands, state)
    gen_qs = gen_cache.get(phrase, []) if len(cands) > 1 else []
    gen_ps = await jev_answer(state, gen_qs) if gen_qs else []
    gen_txt = trace_block(gen_qs, gen_ps)
    gen_state = state + ("\n\n" + gen_txt if gen_txt else "")
    top_qs, top_ps = select_confident(gen_qs, gen_ps)
    top_txt = trace_block(top_qs, top_ps)
    top_state = state + ("\n\n" + top_txt if top_txt else "")

    (rec["jc_noq"], rec["jc_noq_probs"]), (rec["jc_q"], rec["jc_q_probs"]), \
        (rec["jc_cmp"], rec["jc_cmp_probs"]), (rec["jc_gen"], rec["jc_gen_probs"]), \
        (rec["jc_gen_top"], rec["jc_gen_top_probs"]) = await asyncio.gather(
            jev_choose(state, question, cands), jev_choose(qa_state, question, cands),
            jev_choose(cmp_state, question, cands), jev_choose(gen_state, question, cands),
            jev_choose(top_state, question, cands))
    rec["gen_trace"] = [{"q": q, "p": p} for q, p in zip(gen_qs, gen_ps)]
    rec["gen_top_trace"] = [{"q": q, "p": p} for q, p in zip(top_qs, top_ps)]
    js = jev_scores(qs, cands)
    ps = pair_scores(pairs, cands)
    prior = rec["jc_noq_probs"] if len(rec["jc_noq_probs"]) == len(cands) else [1 / len(cands)] * len(cands)
    rec["jev"] = cands[int(np.argmax(js))]
    rec["facet"] = cands[int(np.argmax(fs))]
    rec["pair"] = cands[int(np.argmax(ps))]
    rec["jc_fuse"] = cands[fuse(prior, js, fs)]
    rec.update(jev_scores=js, facet_scores=fs, facet_cand_answers=fdetail, pair_scores=ps,
               pairs=pairs, cmp_prompt=cmp_txt, questions=qs)


async def decide_slot(persona, intent, slot, others, cand_cache: dict, gen_cache: dict) -> dict:
    phrase, kind = slot["phrase"], slot["type"]
    evidence = retrieve(persona, f"{phrase} {intent}", EVIDENCE_K)
    if phrase not in cand_cache:                  # gpt-4o-mini step 2, cached
        cand_cache[phrase] = await propose_candidates(intent, phrase, kind, others, evidence)
    cands = cand_cache[phrase]
    rec = {"phrase": phrase, "type": kind, "candidates": cands}
    if not cands:
        rec.update({a: None for a in ARMS}, questions=[])
        return rec
    state = f"Task: {intent}\n\nUser history:\n" + "\n".join(fmt_mem(m) for m in evidence)
    question = f"Which value is the user's usual \"{phrase}\" for this task?"
    if MODE == "noq":
        rec["jc_noq"], rec["jc_noq_probs"] = await jev_choose(state, question, cands)
        rec["questions"] = []
        return rec
    if MODE == "typed":
        await _decide_typed(rec, state, question, cands, phrase, intent, gen_cache)
        return rec
    qs = build_questions(kind, intent, phrase, cands)
    for q, p in zip(qs, await jev_answer(state, [q["q"] for q in qs])):
        q["p"] = p
    if MODE == "q":
        await _decide_q(rec, state, question, cands, qs)
        return rec
    await _decide(rec, state, question, cands, qs, facet=await facet_scores(phrase, qs, cands), phrase=phrase,
                  gen_cache=gen_cache, intent=intent)
    return rec


def compose(intent: str, site_rec: dict, slot_recs: list[dict], arm: str) -> tuple[str, str | None]:
    site = site_rec.get(arm)
    parts = [f'{r["phrase"]} = {r[arm]}' for r in slot_recs if r.get(arm)]
    text = intent + (f"\nWebsite: {site}" if site else "") + \
        ("\nResolved preferences: " + "; ".join(parts) if parts else "")
    return text, site


def load_cache() -> dict:
    return json.load(open(CACHE_FILE)) if CACHE_FILE.exists() else {}


def save_cache() -> None:
    json.dump(_CACHE, open(CACHE_FILE, "w"), indent=1)


async def run_task(pair, baselines) -> dict:
    tid, persona, intent, gt = pair["task_id"], pair["persona"], pair["intent"], pair["gt"]
    cache = _CACHE.setdefault(tid, {})
    if "slots" not in cache:                      # gpt-4o-mini step 1, cached
        s = await extract_slots(intent)
        cache.update(explicit_site=s["explicit_site"], slots=s["slots"])
    slots = {"explicit_site": cache["explicit_site"], "slots": cache["slots"]}
    gen_cache = cache.setdefault("gen_q", {}).setdefault(
        f"{QGEN_MODEL}{'+think' if QGEN_THINK else ''}|{TYPED_PROMPT if MODE == 'typed' else QGEN_PROMPT}", {})   # generated questions, per model
    site_rec = await decide_site(persona, intent, slots["explicit_site"], gen_cache)
    phrases = [s["phrase"] for s in slots["slots"]]
    slot_recs = list(await asyncio.gather(*[
        decide_slot(persona, intent, s, [p for p in phrases if p != s["phrase"]],
                    cache.setdefault("cands", {}), gen_cache)
        for s in slots["slots"]]))

    arms: dict[str, dict] = {}
    for arm in ARMS:
        text, site = compose(intent, site_rec, slot_recs, arm)
        arms[arm] = {"plan": text, "site": site}
    for name, (text, site) in baselines.get(tid, {}).items():
        arms[name] = {"plan": text, "site": site}

    prefs = gt["preferences"]
    judged = await asyncio.gather(*[judge_plan(a["plan"], prefs) for a in arms.values()])
    for a, m in zip(arms.values(), judged):
        a["site_ok"] = site_match(a["site"], gt["website"])
        a["pref_matches"] = m
        a["pref_acc"] = sum(m) / len(m) if m else 1.0

    # Candidate labels: ceiling for any decision rule + data for question informativeness
    # Cached so slot-decision accuracy is identical across runs for the same picks.
    all_cands = [c for r in slot_recs for c in r["candidates"]]
    known = cache.setdefault("labels", {})
    new = list(dict.fromkeys(c for c in all_cands if c not in known))
    for c, ok in zip(new, await label_candidates(new, prefs)):
        known[c] = ok
    for r in slot_recs:
        r["cand_ok"] = [known[c] for c in r["candidates"]]
    await slot_labels(cache, intent, gt, slot_recs)      # per-slot labels replace the any-preference ones
    site_rec["cand_ok"] = [site_match(c, gt["website"]) for c in site_rec["candidates"]]
    if "ceiling" not in cache:
        cache["ceiling"] = await judge_plan("; ".join(all_cands), prefs)
    ceiling = cache["ceiling"]

    return {"task_id": tid, "intent": intent, "gt": gt, "explicit_site": slots["explicit_site"],
            "site": site_rec, "slots": slot_recs, "arms": arms,
            "ceiling_pref_acc": sum(ceiling) / len(ceiling) if ceiling else 1.0,
            "ceiling_site_ok": any(site_rec["cand_ok"]) or bool(site_rec.get("explicit") and
                                                                 site_match(site_rec["explicit"], gt["website"]))}


def decision_failed(d: dict) -> bool:
    """True if any decider pick or question answer for this decision came from a failed call."""
    if len(d.get("candidates") or []) > 1 and not d.get("explicit"):
        if any(a in d and d[a] is None for a in ARMS):
            return True
    if any(q.get("p", 0) is None for q in d.get("questions") or []):
        return True
    for k, v in d.items():
        if k.endswith("_trace") and isinstance(v, list):
            if any(t.get("a", 0) is None or t.get("p", 0) is None for t in v if isinstance(t, dict)):
                return True
    return False


def failed_calls(task_result: dict) -> int:
    return sum(decision_failed(d) for d in task_result["slots"] + [task_result["site"]])


def load_baselines() -> dict:
    out: dict[str, dict] = {}
    for name, f, key in [("kev_base", "p2w_execution_results.json", "rewritten_task")]:
        p = ROOT / f
        if not p.exists():
            continue
        rows = json.load(open(p))
        rows = rows["results"] if isinstance(rows, dict) else rows
        for r in rows:
            out.setdefault(r["task_id"], {})[name] = (r.get(key) or "", r.get("resolved_website"))
    return out


# ── Main ──────────────────────────────────────────────────────────────────────

def summarise(results: list[dict]) -> None:
    n = len(results)
    arms = ["kev_base", *ARMS]
    print(f"\n{'='*70}\nQuestion-bank eval — {n} tasks (offline, plan-level)\n")
    print(f"{'arm':<12}{'site acc':>10}{'pref acc':>10}{'all prefs':>11}{'site+all':>10}")
    for a in arms:
        rows = [r["arms"][a] for r in results if a in r["arms"]]
        if not rows:
            continue
        m = len(rows)
        site = sum(x["site_ok"] for x in rows) / m
        pref = sum(x["pref_acc"] for x in rows) / m
        allp = sum(x["pref_acc"] == 1.0 for x in rows) / m
        both = sum(x["site_ok"] and x["pref_acc"] == 1.0 for x in rows) / m
        print(f"{a:<12}{site:>10.1%}{pref:>10.1%}{allp:>11.1%}{both:>10.1%}   (n={m})")
    print(f"{'ceiling':<12}{sum(r['ceiling_site_ok'] for r in results)/n:>10.1%}"
          f"{sum(r['ceiling_pref_acc'] for r in results)/n:>10.1%}   (best candidate in pool)")

    # Slot-level decision accuracy, per-slot labels (slot judged only against its own GT preference)
    all_slots = [s for r in results for s in r["slots"] if len(s["candidates"]) > 1]
    no_gt = sum(1 for s in all_slots if s.get("gt_pref") is None)
    slots = [s for s in all_slots if any(s.get("cand_ok", []))]
    print(f"\nSlot decisions with >1 option: {len(all_slots)}  (no GT value for the slot: {no_gt}; "
          f"GT value not in pool: {len(all_slots) - no_gt - len(slots)}; scored: {len(slots)})")
    for a in ARMS:
        if not any(a in s for s in slots):
            continue
        ok = sum(s["cand_ok"][s["candidates"].index(s[a])] for s in slots if s.get(a) in s["candidates"])
        last = sum(s.get(a) == s["candidates"][-1] for s in slots)
        print(f"  {a:<6} correct {ok/len(slots):6.1%}   picks last option {last/len(slots):6.1%}")


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=int, default=149)
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--out", type=Path, default=OUT_FILE)
    ap.add_argument("--arms", choices=["all", "noq", "q", "typed", "choiceq"], default="all",
                    help="noq: only the no-question Jev decider; q: jc_noq + jc_q (bank questions) only")
    args = ap.parse_args()
    out_file = args.out
    global ARMS, MODE
    MODE = args.arms
    if MODE == "noq":
        ARMS = ("jc_noq",)
    elif MODE == "q":
        ARMS = ("jc_noq", "jc_q")
    elif MODE == "typed":
        ARMS = ("jc_noq", "jc_choiceq", "jc_mixed")
    elif MODE == "choiceq":
        MODE, ARMS = "typed", ("jc_noq", "jc_choiceq")

    pairs = sorted(load_pairs(), key=lambda p: p["task_id"])[:args.tasks]
    results = json.load(open(out_file)) if out_file.exists() else []
    done = {r["task_id"] for r in results}
    todo = [p for p in pairs if p["task_id"] not in done]
    print(f"{len(pairs)} tasks ({len(done)} done, {len(todo)} to run)", flush=True)

    baselines = load_baselines()
    _CACHE.update(load_cache())
    print(f"cache: {sum(1 for p in pairs if 'slots' in _CACHE.get(p['task_id'], {}))}/{len(pairs)} tasks "
          f"have frozen candidates", flush=True)
    for p in todo:                       # warm embeddings sequentially (encoder isn't thread-safe)
        persona_memory(p["persona"])

    sem = asyncio.Semaphore(args.concurrency)

    async def guarded(p):
        async with sem:
            try:
                r = await run_task(p, baselines)
            except Exception as e:
                print(f"  ✗ {p['task_id']}: {e!r}", flush=True)
                return
            save_cache()
            failed = failed_calls(r)
            if failed:   # never save a task with a failed model call; it is rerun on the next resume
                print(f"  ✗ {p['task_id']}: {failed} decision(s) had a failed call after retries; not saved",
                      flush=True)
                return
            results.append(r)
            json.dump(results, open(out_file, "w"), indent=1)
            a = r["arms"]
            site = "".join(str(int(a[x]["site_ok"])) for x in ARMS)
            pref = " ".join(f"{a[x]['pref_acc']:.2f}" for x in ARMS)
            print(f"  [{len(results):3d}] {r['task_id']:<20} site {site}  pref {pref}  "
                  f"ceil={r['ceiling_pref_acc']:.2f}   ({'/'.join(ARMS)})", flush=True)

    await asyncio.gather(*[guarded(p) for p in todo])
    summarise(results)


if __name__ == "__main__":
    asyncio.run(main())
