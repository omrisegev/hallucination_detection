"""
ALFWorld episode runner for spectral hallucination detection (Phase 11b).

This module is NOT imported by spectral_utils/__init__.py to avoid making
alfworld a hard dependency. Import directly in the notebook:
    from spectral_utils.alfworld_utils import setup_alfworld_env, run_alfworld_episode

Install on Colab (in a dedicated cell, before importing):
    import os
    os.system('pip install -q alfworld pyvirtualdisplay')
    os.system('apt-get install -q -y xvfb')  # virtual display for ALFWorld's renderer
"""

from __future__ import annotations


# ── Prompt helpers ─────────────────────────────────────────────────────────────

_ALFWORLD_SYSTEM = (
    "You are an agent navigating a household environment. "
    "At each step you will receive an observation and a list of valid actions. "
    "Choose exactly ONE action from the list to make progress toward your goal. "
    "Reply with the action text only, nothing else."
)


def alfworld_action_prompt(task_desc: str, history: list[dict], observation: str,
                            admissible: list[str]) -> str:
    """
    Format a ReAct-style turn for ALFWorld.

    Args:
        task_desc:   The goal description (from env.reset()).
        history:     List of prior {action, observation} dicts.
        observation: Current step observation.
        admissible:  List of valid action strings from env.

    Returns:
        Full prompt string for generate_full.
    """
    lines = [f"Task: {task_desc}\n"]
    for i, h in enumerate(history):
        lines.append(f"Step {i+1} — Action: {h['action']}")
        lines.append(f"         Observation: {h['observation']}")
    lines.append(f"\nCurrent observation: {observation}")
    lines.append(f"\nValid actions:\n" + "\n".join(f"  {a}" for a in admissible))
    lines.append("\nYour action:")
    return "\n".join(lines)


def parse_alfworld_action(text: str, admissible: list[str]) -> str:
    """
    Extract an action from model output. Falls back to first admissible action.
    Strips leading/trailing whitespace and takes the first non-empty line.
    """
    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        # Exact match first
        if line in admissible:
            return line
        # Case-insensitive prefix match
        line_lc = line.lower()
        for a in admissible:
            if a.lower().startswith(line_lc[:20]):
                return a
        # Return raw (env will reject → episode terminates cleanly)
        return line
    return admissible[0] if admissible else "look"


# ── Environment setup ──────────────────────────────────────────────────────────

def setup_alfworld_env(task_filter: str = "pick_and_place", data_path: str = ""):
    """
    Create an ALFWorld text environment, optionally filtered to a task type.

    Args:
        task_filter: ALFWorld task prefix, e.g. 'pick_and_place', 'heat_then_place',
                     'cool_then_place', 'examine_in_light', 'pick_two_obj_and_place'.
                     Pass "" to accept all task types.
        data_path:   Path to ALFWorld data directory. Leave "" to use the default
                     (ALFWorld downloads ~3 GB on first run; cached afterward).

    Returns:
        (env, task_descriptions) where task_descriptions is a list of goal strings.
    """
    import alfworld          # deferred import — alfworld not installed by default
    import alfworld.agents   # noqa: F401 — registers envs

    import yaml, os

    # Default config shipped with alfworld package
    cfg_path = os.path.join(os.path.dirname(alfworld.__file__), 'agents', 'config', 'base_config.yaml')
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    if data_path:
        cfg['dataset']['data_path'] = data_path

    # Split to use (val_seen is small enough for a pilot)
    cfg['dataset']['eval_split'] = 'valid_seen'

    import gym
    env = gym.make('AlfredTWEnv-v0', config=cfg, train_eval='eval_out_of_distribution')
    env = env.unwrapped

    # Collect task descriptions across all games
    task_descs: list[str] = []
    env.reset()
    for game in env.gamefiles:
        td = _task_desc_from_gamefile(game)
        if not task_filter or td.startswith(task_filter.replace('_', ' ')):
            task_descs.append(td)

    print(f"ALFWorld env ready — {len(task_descs)} tasks matching '{task_filter or 'all'}'")
    return env, task_descs


def _task_desc_from_gamefile(gamefile: str) -> str:
    """Extract the human-readable task description from the gamefile path."""
    import os
    base = os.path.basename(os.path.dirname(gamefile))
    # e.g. 'pick_and_place_simple-AlarmClock-None-Desk-301'
    parts = base.split('-')
    return parts[0].replace('_', ' ') if parts else base


# ── Episode runner ─────────────────────────────────────────────────────────────

def run_alfworld_episode(
    mdl,
    tok,
    env,
    task_desc: str,
    T: float = 1.0,
    max_steps: int = 20,
    max_new: int = 64,
) -> dict:
    """
    Run one ALFWorld episode, recording per-step token entropy traces.

    Args:
        mdl, tok:    HuggingFace model + tokenizer from load_model().
        env:         ALFWorld TextWorldEnv (from setup_alfworld_env).
        task_desc:   Goal string (from env.reset() or task list).
        T:           Sampling temperature.
        max_steps:   Maximum env steps before forced termination.
        max_new:     Max new tokens per step (actions are short — 20-64 tokens).

    Returns:
        {
          'steps':        list of {token_entropies, action, observation, done},
          'task_success': bool,
          'n_steps':      int,
          'task_desc':    str,
        }
    """
    from .model_utils import generate_full, token_entropies_from_scores

    obs_list, infos = env.reset()
    # ALFWorld may return multiple obs strings; take the first
    obs = obs_list[0] if isinstance(obs_list, list) else obs_list

    history: list[dict] = []
    steps:   list[dict] = []
    done     = False
    success  = False

    for step_idx in range(max_steps):
        admissible = list(infos.get('admissible_commands', [[]])[0])

        prompt_text = alfworld_action_prompt(task_desc, history, obs, admissible)

        ids, scores = generate_full(
            mdl, tok,
            prompt=prompt_text,
            temperature=T,
            max_new_tokens=max_new,
        )
        ents = token_entropies_from_scores(scores)

        completion = tok.decode(ids[0], skip_special_tokens=True)
        if completion.startswith(prompt_text):
            completion = completion[len(prompt_text):]

        action = parse_alfworld_action(completion, admissible)

        obs_list, _, done_list, infos = env.step([action])
        obs  = obs_list[0] if isinstance(obs_list, list) else obs_list
        done = done_list[0] if isinstance(done_list, list) else done_list

        # ALFWorld signals success via the 'won' key in infos
        won = bool(infos.get('won', [False])[0]) if isinstance(infos.get('won'), list) \
              else bool(infos.get('won', False))

        steps.append({
            'token_entropies': ents,
            'action':          action,
            'observation':     obs,
            'done':            done,
            'won':             won,
        })

        history.append({'action': action, 'observation': obs})

        if done or won:
            success = won
            break

    return {
        'steps':        steps,
        'task_success': success,
        'n_steps':      len(steps),
        'task_desc':    task_desc,
    }
