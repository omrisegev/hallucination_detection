"""
agent_utils — ReAct agent loop + tool simulator for the Phase 11 agentic experiment.

Design notes:
- The "tool" is a simulated retriever over HotpotQA's gold context (10 paragraphs).
  This matches LOS-Net's setup and keeps the experiment reproducible without
  external API calls. We score "step correctness" by whether the retrieved
  paragraph is one of the supporting facts.
- Each step records per-token entropies and a verbalized confidence c_hat in [0,1]
  (the AUQ System 1 / UAM signal, Zhang et al. 2026, arXiv:2601.15703).
- run_react_episode emits a structured trajectory dict that downstream cells
  can feed straight into spectral_utils.extract_all_features() per step.

Keep this module thin: prompt formatting, parsing, tool simulation, and the
inference loop. All feature extraction / fusion / gating lives in the notebook
and uses the existing feature_utils + fusion_utils.
"""
import re
import string
from typing import Optional

from .model_utils import generate_full


# ── Prompts ───────────────────────────────────────────────────────────────────

REACT_SYSTEM_PROMPT = """You are a research agent answering multi-hop questions.

Each turn you must produce exactly four lines in this format:

Thought: <one sentence of reasoning about what to look up next>
Action: <one of `search("<query>")` or `finish("<final answer>")`>
Confidence: <a float in [0,1] reflecting how sure you are this Action is correct>
Concern: <one short phrase naming the biggest risk in this Action, or "none">

Use `search(...)` when you need to look up a fact. Use `finish(...)` only when
you can confidently give the final answer. You have at most {max_steps} turns
total. Always emit all four lines exactly as shown.
"""


def react_system_prompt(max_steps: int = 3) -> str:
    return REACT_SYSTEM_PROMPT.format(max_steps=max_steps)


def react_user_prompt(question: str, history: list, step_idx: int, max_steps: int) -> str:
    """Build the user message for step `step_idx` (0-indexed).

    history is a list of dicts with keys {thought, action_type, action_arg,
    confidence, concern, observation}.
    """
    parts = [f"Question: {question}", ""]
    for i, h in enumerate(history):
        action_str = f'{h["action_type"]}("{h["action_arg"]}")'
        parts.append(f"Thought {i+1}: {h['thought']}")
        parts.append(f"Action {i+1}: {action_str}")
        parts.append(f"Confidence {i+1}: {h['confidence']:.2f}")
        parts.append(f"Concern {i+1}: {h['concern']}")
        parts.append(f"Observation {i+1}: {h['observation']}")
        parts.append("")
    remaining = max_steps - step_idx
    parts.append(
        f"You have {remaining} turn(s) left. Emit Thought {step_idx+1}, "
        f"Action {step_idx+1}, Confidence {step_idx+1}, Concern {step_idx+1}."
    )
    return "\n".join(parts)


# ── Parsing ───────────────────────────────────────────────────────────────────

_ACTION_RE = re.compile(r'(search|finish)\s*\(\s*["\']?(.+?)["\']?\s*\)', re.I | re.S)
_THOUGHT_RE = re.compile(r'Thought\s*\d*\s*:\s*(.+?)(?=\n\s*Action|\Z)', re.I | re.S)
_ACTION_LINE_RE = re.compile(r'Action\s*\d*\s*:\s*(.+?)(?=\n\s*Confidence|\Z)', re.I | re.S)
_CONF_RE = re.compile(r'Confidence\s*\d*\s*:\s*([0-9]*\.?[0-9]+)', re.I)
_CONCERN_RE = re.compile(r'Concern\s*\d*\s*:\s*(.+?)(?=\n|$)', re.I | re.S)


def parse_thought(text: str) -> str:
    m = _THOUGHT_RE.search(text)
    return m.group(1).strip()[:300] if m else ""


def parse_action(text: str) -> tuple[str, str]:
    """Returns (action_type, action_arg). action_type in {"search","finish","invalid"}."""
    m = _ACTION_LINE_RE.search(text)
    if not m:
        return "invalid", ""
    action_line = m.group(1).strip()
    m2 = _ACTION_RE.search(action_line)
    if not m2:
        return "invalid", action_line[:200]
    return m2.group(1).lower(), m2.group(2).strip()[:200]


def parse_confidence(text: str) -> float:
    """Extract verbalized confidence c_hat ∈ [0,1]. Returns 0.5 if unparseable."""
    m = _CONF_RE.search(text)
    if not m:
        return 0.5
    try:
        v = float(m.group(1))
    except ValueError:
        return 0.5
    if v > 1.0 and v <= 100.0:   # tolerate "Confidence: 80"
        v = v / 100.0
    return max(0.0, min(1.0, v))


def parse_concern(text: str) -> str:
    m = _CONCERN_RE.search(text)
    return m.group(1).strip()[:150] if m else "none"


# ── Tool simulator ────────────────────────────────────────────────────────────

def _normalize_tokens(s: str) -> set:
    s = s.lower()
    s = re.sub(r'\b(a|an|the|of|in|on|at|to|for|and|or)\b', ' ', s)
    s = ''.join(c for c in s if c not in string.punctuation)
    return set(s.split())


def simulate_retrieve_tool(query: str, context: dict) -> tuple[str, str]:
    """Simulate a retriever that scores HotpotQA paragraphs by token overlap.

    Args:
        query:   the model's search() argument
        context: HotpotQA row's `context` field. Structure:
                 {'title': [...], 'sentences': [[...], [...], ...]}

    Returns:
        (retrieved_title, retrieved_passage). retrieved_title is "" if no match.
    """
    titles    = context.get('title', [])
    sentences = context.get('sentences', [])
    if not titles or not sentences:
        return "", "No documents available."

    q_tokens = _normalize_tokens(query)
    if not q_tokens:
        return titles[0], " ".join(sentences[0])

    best_idx   = 0
    best_score = -1
    for i, (title, sents) in enumerate(zip(titles, sentences)):
        para = title + " " + " ".join(sents)
        p_tokens = _normalize_tokens(para)
        if not p_tokens:
            continue
        score = len(q_tokens & p_tokens) / (len(q_tokens) ** 0.5)
        if score > best_score:
            best_score = score
            best_idx   = i

    title = titles[best_idx]
    passage = " ".join(sentences[best_idx])
    # truncate to keep prompt small
    if len(passage) > 700:
        passage = passage[:700] + " ..."
    return title, passage


def _support_title_list(supporting_titles) -> list[str]:
    if isinstance(supporting_titles, dict):
        vals = supporting_titles.get('title', [])
        return [str(v) for v in vals]
    if isinstance(supporting_titles, (list, tuple, set)):
        out = []
        for item in supporting_titles:
            if isinstance(item, (list, tuple)) and item:
                out.append(str(item[0]))
            else:
                out.append(str(item))
        return out
    return []


def query_support_overlap(query: str, supporting_titles) -> float:
    q_tokens = _normalize_tokens(query)
    if not q_tokens:
        return 0.0
    best = 0.0
    for title in _support_title_list(supporting_titles):
        t_tokens = _normalize_tokens(title)
        if not t_tokens:
            continue
        best = max(best, len(q_tokens & t_tokens) / max(1, len(q_tokens)))
    return float(best)


# ── Step correctness ──────────────────────────────────────────────────────────

def step_retrieved_supporting_fact(retrieved_title: str, supporting_titles: list) -> bool:
    """True if the retrieved paragraph title is one of the supporting-fact titles."""
    titles = _support_title_list(supporting_titles)
    if not retrieved_title or not titles:
        return False
    return retrieved_title in titles


# ── Episode driver ────────────────────────────────────────────────────────────

def run_react_episode(mdl, tok, question: str, context: dict,
                      supporting_titles: list, gold_answer: str,
                      T: float = 1.0, max_steps: int = 3,
                      max_new_per_step: int = 256) -> dict:
    """Run one ReAct episode. Returns a structured trajectory dict.

    Output schema:
    {
        'question':           str,
        'gold_answer':        str,
        'supporting_titles':  list[str],
        'steps': [
            {
                'thought':           str,
                'action_type':       str,
                'action_arg':        str,
                'confidence':        float,
                'concern':           str,
                'observation_title': str,
                'observation':       str,
                'step_text':         str,   # raw generated text for this step
                'token_entropies':   list[float],
                'token_offsets':     list[(int,int)],
                'step_correct':      bool,  # retrieval got a supporting fact
            },
            ...
        ],
        'final_answer':      str,
        'trajectory_correct': bool,
        'n_steps':           int,
    }
    """
    history: list[dict] = []
    steps_out: list[dict] = []
    final_answer = ""

    sys_msg = react_system_prompt(max_steps)

    for step_idx in range(max_steps):
        user_msg = react_user_prompt(question, history, step_idx, max_steps)
        prompt_msg = sys_msg + "\n\n" + user_msg

        res = generate_full(
            mdl, tok, prompt_msg,
            temperature=T, max_new_tokens=max_new_per_step,
        )
        step_text = res['full_text']
        thought     = parse_thought(step_text)
        a_type, arg = parse_action(step_text)
        conf        = parse_confidence(step_text)
        concern     = parse_concern(step_text)

        if a_type == 'search':
            ret_title, ret_passage = simulate_retrieve_tool(arg, context)
            observation = f"[{ret_title}] {ret_passage}" if ret_title else ret_passage
            step_correct = step_retrieved_supporting_fact(ret_title, supporting_titles)
        elif a_type == 'finish':
            ret_title    = ""
            ret_passage  = ""
            observation  = f"(finished with answer: {arg})"
            step_correct = (arg.strip().lower() in (gold_answer or "").strip().lower()
                            or (gold_answer or "").strip().lower() in arg.strip().lower())
            final_answer = arg
        else:
            ret_title    = ""
            ret_passage  = ""
            observation  = "(no valid action emitted)"
            step_correct = False

        steps_out.append({
            'thought':           thought,
            'action_type':       a_type,
            'action_arg':        arg,
            'confidence':        conf,
            'concern':           concern,
            'observation_title': ret_title,
            'observation':       observation[:800],
            'step_text':         step_text,
            'token_entropies':   res['token_entropies'],
            'token_offsets':     res['token_offsets'],
            'step_correct':      bool(step_correct),
        })

        history.append({
            'thought':       thought,
            'action_type':   a_type,
            'action_arg':    arg,
            'confidence':    conf,
            'concern':       concern,
            'observation':   observation[:600],
        })

        if a_type == 'finish':
            break

    # If never finished, take the last search query as a degenerate answer
    if not final_answer and steps_out:
        final_answer = steps_out[-1]['action_arg']

    traj_correct = _trajectory_correct(final_answer, gold_answer)

    return {
        'question':           question,
        'gold_answer':        gold_answer,
        'supporting_titles':  list(supporting_titles),
        'steps':              steps_out,
        'final_answer':       final_answer,
        'trajectory_correct': bool(traj_correct),
        'n_steps':            len(steps_out),
    }


def _normalize_answer(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = ''.join(c for c in s if c not in string.punctuation)
    return ' '.join(s.split())


def _trajectory_correct(pred: str, gold: str) -> bool:
    p, g = _normalize_answer(pred), _normalize_answer(gold)
    if not g:
        return False
    return (g in p) or (p in g and len(p) >= 2)


# ── Branching entropy (Step-Boundary Uncertainty Trajectory) ─────────────────

def branching_entropy(token_entropies: list, window: int = 3) -> float:
    """Mean entropy of the first `window` generated tokens of a step.

    This is our gray-box proxy for the Step-Boundary Uncertainty Trajectory
    (SBUT) signal that the CoT survey (Cheng et al. 2025, Zhao et al. 2026)
    identifies as the single most promising new Nadler view: entropy at the
    moment the model begins a new reasoning step is where logical-branch
    commitment happens.

    In our per-step ReAct setup each step is a separate generate() call, so
    token_entropies[0] is literally the first token after the previous step
    ended — exactly the step-boundary token the survey calls out. We take a
    small mean to smooth the deterministic format prefix ("Thought 1:") and
    capture the model's true content-token branching uncertainty.
    """
    if not token_entropies:
        return float('nan')
    import numpy as np
    arr = np.asarray(token_entropies[:max(1, window)], dtype=float)
    if arr.size == 0:
        return float('nan')
    return float(arr.mean())


# ── Spiral-of-Hallucination injection (Direction 4D) ─────────────────────────

def run_spiral_injection_replay(mdl, tok, question: str, context: dict,
                                 supporting_titles: list, gold_answer: str,
                                 original_step1: dict, distractor_passage: str,
                                 T: float = 1.0, max_steps: int = 3,
                                 max_new_per_step: int = 256) -> dict:
    """Replay a trajectory but force step-1's observation to be a distractor.

    Step 1's (thought, action, confidence, concern, token_entropies) are kept
    verbatim from the original trajectory; only the observation is swapped.
    Steps 2..max_steps are re-generated by the model, conditioned on the
    poisoned history. Same return schema as run_react_episode.
    """
    fake_step1 = {
        'thought':           original_step1['thought'],
        'action_type':       original_step1['action_type'],
        'action_arg':        original_step1['action_arg'],
        'confidence':        original_step1['confidence'],
        'concern':           original_step1['concern'],
        'observation_title': '__INJECTED__',
        'observation':       distractor_passage[:800],
        'step_text':         original_step1['step_text'],
        'token_entropies':   original_step1['token_entropies'],
        'token_offsets':     original_step1['token_offsets'],
        'step_correct':      False,
    }
    history = [{
        'thought':     fake_step1['thought'],
        'action_type': fake_step1['action_type'],
        'action_arg':  fake_step1['action_arg'],
        'confidence':  fake_step1['confidence'],
        'concern':     fake_step1['concern'],
        'observation': fake_step1['observation'][:600],
    }]
    steps_out = [fake_step1]
    final_answer = ""
    sys_msg = react_system_prompt(max_steps)

    for step_idx in range(1, max_steps):
        user_msg = react_user_prompt(question, history, step_idx, max_steps)
        prompt_msg = sys_msg + "\n\n" + user_msg

        res = generate_full(
            mdl, tok, prompt_msg,
            temperature=T, max_new_tokens=max_new_per_step,
        )
        step_text = res['full_text']
        thought     = parse_thought(step_text)
        a_type, arg = parse_action(step_text)
        conf        = parse_confidence(step_text)
        concern     = parse_concern(step_text)

        if a_type == 'search':
            ret_title, ret_passage = simulate_retrieve_tool(arg, context)
            observation = f"[{ret_title}] {ret_passage}" if ret_title else ret_passage
            step_correct = step_retrieved_supporting_fact(ret_title, supporting_titles)
        elif a_type == 'finish':
            ret_title    = ""
            ret_passage  = ""
            observation  = f"(finished with answer: {arg})"
            step_correct = (arg.strip().lower() in (gold_answer or "").strip().lower()
                            or (gold_answer or "").strip().lower() in arg.strip().lower())
            final_answer = arg
        else:
            ret_title    = ""
            ret_passage  = ""
            observation  = "(no valid action emitted)"
            step_correct = False

        steps_out.append({
            'thought':           thought,
            'action_type':       a_type,
            'action_arg':        arg,
            'confidence':        conf,
            'concern':           concern,
            'observation_title': ret_title,
            'observation':       observation[:800],
            'step_text':         step_text,
            'token_entropies':   res['token_entropies'],
            'token_offsets':     res['token_offsets'],
            'step_correct':      bool(step_correct),
        })
        history.append({
            'thought':     thought,
            'action_type': a_type,
            'action_arg':  arg,
            'confidence':  conf,
            'concern':     concern,
            'observation': observation[:600],
        })
        if a_type == 'finish':
            break

    if not final_answer and steps_out:
        final_answer = steps_out[-1]['action_arg']
    traj_correct = _trajectory_correct(final_answer, gold_answer)

    return {
        'question':           question,
        'gold_answer':        gold_answer,
        'supporting_titles':  list(supporting_titles),
        'steps':              steps_out,
        'final_answer':       final_answer,
        'trajectory_correct': bool(traj_correct),
        'n_steps':            len(steps_out),
        'injected':           True,
        'distractor_passage': distractor_passage[:600],
    }


# ── Aggregation ───────────────────────────────────────────────────────────────

def aggregate_trajectory(per_step_scores: list, agg: str = 'min') -> float:
    """Φ_min / Φ_avg / Φ_last over a per-step score vector."""
    if not per_step_scores:
        return float('nan')
    import numpy as np
    arr = np.asarray([s for s in per_step_scores if s is not None and not np.isnan(s)],
                     dtype=float)
    if arr.size == 0:
        return float('nan')
    if agg == 'min':
        return float(arr.min())
    if agg == 'avg':
        return float(arr.mean())
    if agg == 'last':
        return float(arr[-1])
    raise ValueError(f"unknown agg: {agg!r}")


def first_incorrect_step_index(traj: dict) -> Optional[int]:
    steps = traj.get('steps', [])
    for idx, step in enumerate(steps):
        if not bool(step.get('step_correct', False)):
            return idx
    if steps and not traj.get('trajectory_correct', False):
        return len(steps) - 1
    return None


def categorize_failure_mode_v2(traj: dict) -> str:
    if traj.get('trajectory_correct'):
        return 'correct'
    steps = traj.get('steps', [])
    if any(s.get('action_type') == 'invalid' for s in steps):
        return 'tool_or_format'
    if not any(s.get('action_type') == 'finish' for s in steps):
        return 'no_finish'

    search_steps = [s for s in steps if s.get('action_type') == 'search']
    if not search_steps:
        return 'planning'

    if not any(bool(s.get('step_correct', False)) for s in search_steps):
        overlaps = [query_support_overlap(s.get('action_arg', ''), traj.get('supporting_titles', []))
                    for s in search_steps]
        return 'retrieval_query' if max(overlaps or [0.0]) < 0.34 else 'retrieval_context'

    if any(s.get('action_type') == 'finish' and not bool(s.get('step_correct', False)) for s in steps):
        return 'reasoning_or_synthesis'

    return 'planning'


def categorize_failure_mode(traj: dict) -> str:
    """Heuristic failure-mode label for an incorrect trajectory.

    Categories:
      - 'planning':  no step ever retrieved a supporting fact
      - 'tool':      retrieved supporting fact in ≥1 step but answer still wrong (synthesis fail)
      - 'invalid':   any step emitted an invalid action
      - 'no_finish': never emitted finish()
      - 'correct':   trajectory_correct == True
    """
    if traj['trajectory_correct']:
        return 'correct'
    if any(s['action_type'] == 'invalid' for s in traj['steps']):
        return 'invalid'
    if not any(s['action_type'] == 'finish' for s in traj['steps']):
        return 'no_finish'
    if not any(s['step_correct'] for s in traj['steps'] if s['action_type'] == 'search'):
        return 'planning'
    return 'tool'


# ── HumanEval code-execution episode ──────────────────────────────────────────

def execute_python_solution(
    full_code: str,
    test_code: str,
    entry_point: str,
    timeout: int = 5,
) -> tuple:
    """Run full_code + test_code in a subprocess. Returns (passed: bool, error: str)."""
    import subprocess, tempfile, os, textwrap

    script = textwrap.dedent(f"""
{full_code}

{test_code}

check({entry_point})
""")
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as fh:
        fh.write(script)
        tmp_path = fh.name

    try:
        result = subprocess.run(
            ['python', tmp_path],
            capture_output=True, text=True, timeout=timeout,
        )
        passed = result.returncode == 0
        error  = (result.stderr or result.stdout or '').strip()
    except subprocess.TimeoutExpired:
        passed = False
        error  = f'TimeoutExpired after {timeout}s'
    except Exception as exc:
        passed = False
        error  = str(exc)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    return passed, error


def run_humaneval_episode(
    mdl,
    tok,
    row: dict,
    T: float = 1.0,
    max_attempts: int = 3,
    max_new: int = 512,
) -> dict:
    """
    Multi-attempt HumanEval episode.

    Each attempt:
      1. Format prompt (with prior error context if retrying).
      2. generate_full → token_entropies.
      3. Concatenate row['prompt'] + generated completion → execute_python_solution.
      4. If pass → done. Else carry error into next attempt.

    Returns:
      {
        'steps':       list of {token_entropies, code, passed, error},
        'any_passed':  bool,
        'n_attempts':  int,
        'task_id':     str,
      }
    """
    from .model_utils import generate_full, token_entropies_from_scores
    from .data_loaders import humaneval_prompt

    steps: list = []
    any_passed   = False
    last_error   = ""

    for attempt_idx in range(max_attempts):
        prompt_text = humaneval_prompt(row, error_context=last_error)

        ids, scores = generate_full(
            mdl, tok,
            prompt=prompt_text,
            temperature=T,
            max_new_tokens=max_new,
        )
        ents = token_entropies_from_scores(scores)

        # Decode the completion and prepend the original function header
        completion = tok.decode(ids[0], skip_special_tokens=True)
        # Strip any echoed prompt prefix the model may have reproduced
        if completion.startswith(prompt_text):
            completion = completion[len(prompt_text):]
        full_code = row['prompt'] + completion

        passed, error = execute_python_solution(
            full_code, row['test'], row['entry_point']
        )

        steps.append({
            'token_entropies': ents,
            'code':            full_code,
            'passed':          passed,
            'error':           error,
            'attempt':         attempt_idx,
        })

        if passed:
            any_passed = True
            break
        last_error = error.splitlines()[-1] if error else ''

    return {
        'steps':      steps,
        'any_passed': any_passed,
        'n_attempts': len(steps),
        'task_id':    row['task_id'],
    }
