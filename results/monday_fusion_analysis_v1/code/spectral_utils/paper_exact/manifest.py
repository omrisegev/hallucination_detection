"""
RUN_MANIFEST.json — the immutable provenance record for one acquisition run.

Handoff §3.1. Every run directory carries exactly one of these, written before the
first shard and never rewritten with different content. If a resumed job would produce
a manifest that differs from the one on disk in any pinned field, that is a protocol
violation (someone changed the model, prompt, decoding, or dataset mid-run) and
`write_manifest` refuses rather than overwriting the evidence.

The mutable counters (finished/failed trace counts) live in STATUS.json, written by
`shards.ShardWriter`, precisely so that the manifest can stay immutable.

Why this is strict
------------------
Step 244 in this project's own history is the case study: a no-op resume destroyed the
validation evidence it should have preserved. An immutable manifest plus an append-only
shard index makes that class of loss structurally impossible — a resume can only add.
"""
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone

from . import FIDELITY_LABELS, SCHEMA_VERSION

#: Fields that pin *what experiment this is*. A resume may not change any of them.
PINNED_FIELDS = (
    "schema",
    "run_id",
    "paper_title",
    "paper_pdf_sha256",
    "fidelity",
    "dataset_source",
    "dataset_revision",
    "dataset_order_sha256",
    "model_id",
    "model_revision",
    "chat_template_sha256",
    "prompt_sha256",
    "decoding",
    "seed_policy",
    "max_new_tokens",
    "signal_definitions",
    "logits_stage",
)

#: Fields that must be present for the manifest gate to pass.
REQUIRED_FIELDS = PINNED_FIELDS + (
    "created_utc",
    "repo_commit",
    "repo_dirty",
    "container_image",
    "paper_pdf_path",
    "official_code_url",
    "official_code_commit",
    "dataset_example_ids",
    "stop_behavior",
    "hidden_state_layers",
    "expected_traces",
    "shard_index",
    "software",
    "evaluator_revision",
    "declared_deviations",
)

#: Required fields whose empty value is a real, meaningful answer rather than an omission.
#: Emptiness here is informative — "this run captured no hidden states", "this paper
#: published no code", "we deviated from nothing" — and rejecting it would push drivers to
#: invent placeholder text, which is strictly worse provenance than an honest empty.
MAY_BE_EMPTY = frozenset({
    "expected_traces",        # unknown until the driver counts its own N
    "hidden_state_layers",    # [] = no hidden-state capture, the norm for these stages
    "declared_deviations",    # [] = an official-exact run with nothing to declare
    "official_code_url",      # "" = the paper published no code
    "official_code_commit",
})

#: `logits_stage` is a closed vocabulary. Mislabelling post-warper telemetry as raw is
#: the single failure that silently invalidates every DeepConf comparison (handoff §3.2,
#: phase-1 checkpoint §7.12), so it gets its own enum rather than a free-text field.
LOGITS_STAGES = ("raw", "post-warper", "both")


def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    """SHA-256 of a file, streamed (the PDFs and shards are large)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    """SHA-256 of a string, UTF-8, no trailing-newline normalisation.

    Used for prompts and chat templates, where a stray newline genuinely changes the
    tokenisation and therefore the entropy trace — so it must change the hash too.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_order(ids) -> str:
    """Order hash over the ordered example IDs.

    Order matters for REFRAIN (its bandit state crosses questions, so a shuffle changes
    the algorithm — phase-1 checkpoint §7.13) and for reproducibility everywhere else.
    A set hash would silently permit a reshuffle; this does not.
    """
    joined = "\n".join(str(i) for i in ids)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


#: Written by `cluster/sync_code.sh` next to the synced tree. The cluster copy is a tarball
#: with `.git` deliberately excluded, so without this file a cluster manifest could only
#: record `repo_commit="unknown"` — and "a full run's numbers must be traceable to a commit
#: hash" would be an empty claim exactly where it matters most.
SYNC_STAMP = "SYNC_COMMIT.json"


def git_info(repo_root: str) -> dict:
    """Repository commit + dirty flag, from git if present or the sync stamp if not."""
    def _run(args):
        try:
            return subprocess.run(args, cwd=repo_root, capture_output=True, text=True,
                                  timeout=30).stdout.strip()
        except Exception:
            return ""

    if os.path.isdir(os.path.join(repo_root, ".git")):
        commit = _run(["git", "rev-parse", "HEAD"])
        if commit:
            status = _run(["git", "status", "--porcelain", "--untracked-files=no"])
            return {"repo_commit": commit, "repo_dirty": bool(status),
                    "commit_source": "git"}

    stamp = os.path.join(repo_root, SYNC_STAMP)
    if os.path.exists(stamp):
        try:
            with open(stamp) as f:
                d = json.load(f)
            return {"repo_commit": d.get("commit", "unknown"),
                    "repo_dirty": bool(d.get("dirty", True)),
                    "commit_source": f"sync stamp written {d.get('synced_utc', '?')}"}
        except Exception:
            pass
    # No git and no stamp: say so rather than implying a clean unknown tree. `repo_dirty=True`
    # means a full run's manifest gate will refuse, which is the correct outcome — an
    # untraceable tree must not produce a table row.
    return {"repo_commit": "unknown", "repo_dirty": True, "commit_source": "unavailable"}


def software_info() -> dict:
    """Versions that change numerics. Recorded, never asserted equal across machines —
    a B200 reproduction of A100 sampling is protocol-aligned, not bitwise identical
    (phase-1 checkpoint §7.17)."""
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    for mod in ("torch", "transformers", "numpy", "datasets", "vllm", "sentence_transformers"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:
            info[mod] = None
    try:
        import torch
        info["cuda"] = torch.version.cuda
        info["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        info["n_gpu"] = torch.cuda.device_count()
    except Exception:
        info["cuda"] = info["gpu"] = info["n_gpu"] = None
    return info


def build_manifest(
    *,
    run_id: str,
    paper_title: str,
    paper_pdf_path: str,
    fidelity: str,
    dataset_source: str,
    dataset_revision: str,
    dataset_example_ids,
    model_id: str,
    model_revision: str,
    prompt_text: str,
    chat_template: str,
    decoding: dict,
    seed_policy: dict,
    max_new_tokens: int,
    stop_behavior: dict,
    signal_definitions: dict,
    logits_stage: str,
    hidden_state_layers=None,
    official_code_url: str = "",
    official_code_commit: str = "",
    container_image: str = "",
    evaluator_revision: str = "",
    declared_deviations=None,
    repo_root: str = None,
    extra: dict = None,
) -> dict:
    """Assemble a complete `paper_exact_acquisition_v1` manifest.

    `paper_pdf_path` is hashed here rather than taken on trust: `papers/PAPER_EXACT_SOURCES.md`
    pins seven SHA-256 values and the P0 audit checks this hash against that registry, so a
    silently-replaced PDF cannot travel into a run unnoticed.

    `declared_deviations` is a list of {'field', 'paper_says', 'we_do', 'why'} records. It is
    the difference between `paper-specified` and a quiet fabrication: anything the paper does
    not pin and we chose ourselves belongs here, and the fidelity label must then be
    `paper-specified-partial`.
    """
    if fidelity not in FIDELITY_LABELS:
        raise ValueError(f"fidelity {fidelity!r} not in {FIDELITY_LABELS}")
    if logits_stage not in LOGITS_STAGES:
        raise ValueError(f"logits_stage {logits_stage!r} not in {LOGITS_STAGES}")

    ids = list(dataset_example_ids)
    deviations = list(declared_deviations or [])
    if fidelity == "paper-specified" and deviations:
        raise ValueError(
            "fidelity='paper-specified' with declared deviations is contradictory — "
            "a run that fills in constants the paper omits is 'paper-specified-partial'."
        )

    repo_root = repo_root or os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))

    man = {
        "schema": SCHEMA_VERSION,
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "container_image": container_image or os.environ.get("SLURM_CONTAINER_IMAGE", ""),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),

        # paper provenance
        "paper_title": paper_title,
        "paper_pdf_path": paper_pdf_path,
        "paper_pdf_sha256": sha256_file(paper_pdf_path) if os.path.exists(paper_pdf_path) else "",
        "fidelity": fidelity,
        "official_code_url": official_code_url,
        "official_code_commit": official_code_commit,

        # data provenance
        "dataset_source": dataset_source,
        "dataset_revision": dataset_revision,
        "dataset_example_ids": ids,
        "dataset_order_sha256": sha256_order(ids),
        "n_examples": len(ids),

        # model provenance
        "model_id": model_id,
        "model_revision": model_revision,
        "chat_template_sha256": sha256_text(chat_template or ""),
        "prompt_text": prompt_text,
        "prompt_sha256": sha256_text(prompt_text),

        # generation contract
        "decoding": dict(decoding),
        "seed_policy": dict(seed_policy),
        "max_new_tokens": int(max_new_tokens),
        "stop_behavior": dict(stop_behavior),

        # signal contract
        "signal_definitions": dict(signal_definitions),
        "logits_stage": logits_stage,
        "hidden_state_layers": list(hidden_state_layers or []),

        # accounting (expected only; observed counts live in STATUS.json)
        "expected_traces": None,
        "shard_index": "INDEX.jsonl",

        # environment + evaluator
        "software": software_info(),
        "evaluator_revision": evaluator_revision,
        "declared_deviations": deviations,
    }
    man.update(git_info(repo_root))
    if extra:
        man["extra"] = dict(extra)
    return man


def verify_manifest(man: dict, require_clean_tree: bool = False) -> list:
    """Return a list of human-readable problems; empty list means the manifest gate passes.

    `require_clean_tree` is off by default and turned on for `--mode full`. A smoke or pilot
    launched from a dirty tree is normal and healthy — that is what development looks like,
    and the manifest still records `repo_dirty=True` either way. A *full* run is different:
    its numbers go in a table, and "which code produced this" must be answerable by a commit
    hash alone.
    """
    problems = []
    if man.get("schema") != SCHEMA_VERSION:
        problems.append(f"schema is {man.get('schema')!r}, expected {SCHEMA_VERSION!r}")
    for field in REQUIRED_FIELDS:
        if field not in man:
            problems.append(f"missing field: {field}")
        elif man[field] in (None, "", [], {}) and field not in MAY_BE_EMPTY:
            problems.append(f"empty field: {field}")
    if man.get("fidelity") not in FIDELITY_LABELS:
        problems.append(f"bad fidelity: {man.get('fidelity')!r}")
    if man.get("logits_stage") not in LOGITS_STAGES:
        problems.append(f"bad logits_stage: {man.get('logits_stage')!r}")
    if man.get("dataset_example_ids") is not None:
        expect = sha256_order(man["dataset_example_ids"])
        if man.get("dataset_order_sha256") != expect:
            problems.append("dataset_order_sha256 does not match dataset_example_ids")
    if man.get("prompt_text") is not None:
        if man.get("prompt_sha256") != sha256_text(man["prompt_text"]):
            problems.append("prompt_sha256 does not match prompt_text")
    if require_clean_tree and man.get("repo_dirty"):
        problems.append("repo_dirty=True — a full run must be launched from a clean tree")
    return problems


def write_manifest(man: dict, run_dir: str, allow_resume: bool = True) -> str:
    """Write RUN_MANIFEST.json, refusing to change any pinned field on resume.

    On resume the on-disk manifest wins for everything pinned; only the volatile fields
    (created_utc, slurm_job_id, software, expected_traces) are refreshed, and they are
    kept as a `resumes` list rather than overwritten, so a preempted-and-requeued run
    keeps the full record of every attempt.
    """
    os.makedirs(run_dir, exist_ok=True)
    path = os.path.join(run_dir, "RUN_MANIFEST.json")

    if os.path.exists(path):
        with open(path) as f:
            old = json.load(f)
        drift = [k for k in PINNED_FIELDS if old.get(k) != man.get(k)]
        if drift:
            raise ValueError(
                f"RUN_MANIFEST.json at {path} pins different values for {drift}. "
                "This run directory belongs to a different experiment — use a new run_id "
                "instead of overwriting acquisition provenance."
            )
        if not allow_resume:
            raise FileExistsError(f"{path} exists and allow_resume=False")
        old.setdefault("resumes", []).append({
            "utc": man["created_utc"],
            "slurm_job_id": man.get("slurm_job_id", ""),
            "repo_commit": man.get("repo_commit", ""),
            "software": man.get("software", {}),
        })
        old["expected_traces"] = man.get("expected_traces", old.get("expected_traces"))
        man = old

    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(man, f, indent=2, default=str)
    os.replace(tmp, path)
    return path


def load_manifest(run_dir: str) -> dict:
    with open(os.path.join(run_dir, "RUN_MANIFEST.json")) as f:
        return json.load(f)
