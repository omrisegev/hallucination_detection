# Handoff — documentation audit, and merging the Claude and Codex histories

Written 2026-08-19 at commit `58740ce` on `paper-exact/acquisition-v1`.
Two jobs for the next session. They are related: you cannot audit whether the
documentation is trustworthy while two divergent copies of it are being merged,
so settle the merge convention first, then audit the merged result.

---

## Job 1 — merge the two histories without silently corrupting them

Omri is reconciling commits from two repos: this one (Claude) and Codex's. Both
append to `HISTORY.md`, so step numbers will collide.

### The thing to know before choosing a strategy

**Step numbers are already not unique, and were not unique before either side
touched them.** Measured on `58740ce`:

| Measure | Value |
|---|---|
| `### Step N` headings | 259 |
| Number range | 1 … 275 |
| **Duplicated numbers** | **32, 54, 75, 142, 193, 228** |
| Gaps in the sequence | 16, e.g. 75→77, 96→100, 148→151, 257→269 |

Two genuinely different steps already share number 142 (`U-PCR algorithm
correction` at line 4560 and `Add logistic regression oracle` at line 4635) and
number 228 (`Upload the raw dataset cache … via Git LFS` at 10459 and
`atomic-operator premise audit` at 10564). The 257→269 gap is not missing work:
strings matching `Step 26[0-8]` appear 9 times in the file, just not as `### Step
N` headings.

The consequence is the important part: **`Step N` is a label, not an
identifier.** Any merge plan that assumes uniqueness, and any renumbering scheme
that tries to restore it globally, is fighting a battle that was already lost —
and a global renumber would invalidate several hundred cross-references in
`PROGRESS.md`, `Research_Directions.md`, `GLOSSARY.md`, skill files and commit
messages, none of which are mechanically checkable.

### The mechanical hazard

`HISTORY.md` is an append-only log of prose blocks. Git's default text merge
does not know that. When both sides append different blocks at the same offset
it will either interleave them or drop conflict markers **inside a step's
prose**, which then reads as an ordinary paragraph rather than an obvious
breakage. Same for `PROGRESS.md`, whose newest section is prepended at the top —
the single most collision-prone position in the file.

Do not resolve these with `git checkout --ours/--theirs` either: on an
append-only log that silently discards one side's entire contribution.

### Recommended convention — but this is Omri's call, not a default

1. Treat `HISTORY.md` and `PROGRESS.md` as **union-merge** files: take every
   block from both sides, never drop one, never interleave one into another.
   Consider adding to `.gitattributes`:
   `HISTORY.md merge=union` / `PROGRESS.md merge=union` — but verify the result
   by eye, because union merge concatenates at the hunk level and can still
   split a block if both sides edited near the same line.
2. On a number collision, **keep both blocks and disambiguate in the title**
   rather than renumbering: `### Step 274 (Claude) — …` / `### Step 274 (Codex)
   — …`. This matches what the file already does implicitly with 142 and 228, it
   breaks no existing reference, and it makes the two-repo period visible instead
   of hiding it.
3. If Omri prefers renumbering, renumber **only the incoming side's new blocks**
   to the next free numbers, and add one line inside each moved block —
   `(recorded as Step N in the Codex line)` — so any external reference still
   resolves. Never renumber anything that already existed.
4. Whatever is chosen, record it once in `CLAUDE.md` so the next collision is
   resolved the same way.

### What this side contributed, so you can identify it in a merge

Commits `7e771cf`, `b70e80f`, `6481e01`, `2981ae1`, `20d72b4`, `4877799`,
`c730444`, `ca34659`, `58740ce` (plus Codex's own `2c2f5a9`, already in this
branch). In `HISTORY.md` that is exactly two blocks: **Step 274** (protocol
recovery and Step-273 verification) and **Step 275** (paper-exact acquisitions
and their verified backups). Everything else this session touched is outside
`HISTORY.md`: a new head plus two sections in `PROGRESS.md`, one section in
`Research_Directions.md`, a rewritten `fidelity level` entry and a new
`provenance` entry in `spectral_utils/glossary.py` (with `GLOSSARY.md`
regenerated), and sections 8-10 of `HANDOFF_CODEX_2026_08_18.md`.

---

## Job 2 — audit whether the documented claims are actually backed

The question Omri asked, and which was answered honestly rather than
reassuringly: *is every research finding in this repo well documented in the
official documents and in git?* This session could vouch only for its own work.

### Why the audit is justified rather than paranoid

The one gap found today was found **by accident**. `papers/index.md` had listed
the LOS-Net extraction as `extracted` since 2026-07-21, and the file had simply
never been added to git — 71 of 72 extractions were tracked, that one was not. It
surfaced while categorising untracked files, not from any check looking for it.
A repo-only clone, which Step 225 deliberately made the supported case, would
have followed the index to a file that was not there. Fixed in `58740ce`.

Where one gap of that shape exists, others plausibly do.

Note also that an audit of `papers/index.md` itself came back **clean**: all 68
rows with a cached status have their file present, once `extracted` and
`digested` are read as the alternative states the legend defines rather than as
cumulative ones. (The first pass of that audit wrongly required both files and
reported four false positives. Read the legend before encoding a rule.)

### Scope — ask Omri which, do not assume

- **Narrow**: Steps 269-275 (the current campaign) — every number cited in
  `HISTORY.md` / `PROGRESS.md` / `Research_Directions.md` traced to the artifact
  it came from.
- **Medium**: everything from Step 210 onward, i.e. the range Step 225 tracked
  result CSVs for.
- **Full**: all 259 step blocks. Expensive, and most early steps predate the
  artifact-tracking policy, so absence there is expected rather than a finding.

### What the audit should actually check

For each claim, the failure modes worth separating:

1. **Cited artifact does not exist.** A path or filename named in the docs with
   nothing behind it.
2. **Artifact exists but is untracked.** The LOS-Net shape: fine on this machine,
   invisible to a fresh clone. Check against `git ls-files`, not the filesystem.
3. **Number in the prose disagrees with the number in the artifact.** The
   expensive one, and the one that matters most for the advisor-facing tables.
4. **Hash recorded in the docs does not match the file.** Note the two traps this
   session hit: `core.autocrlf` makes a worktree hash differ from the blob hash
   for any file not marked `-text`, and a hash may legitimately be
   machine-specific (see the Step-274 `83529f8d…` case, where the recorded
   per-question hash does not reproduce for a documented and bounded reason).
5. **Index/registry over-claims.** `papers/index.md` is clean; the equivalents
   worth the same treatment are `results/*/README.md`, `PAPER_EXACT_CLAIM_REGISTRY_V1.yaml`,
   and `results/paper_exact/l0/L0_INVENTORY.json`.

### Known open gap, already reported and deliberately not closed

`scripts/build_glossary.py` fails its own coverage check: `a8_lscae`, `a9_dpp`,
`a10_mmdufs` and `a11_rfae_scfs` shipped on 2026-08-05 with no entry in
`spectral_utils/glossary.py`. `GLOSSARY.md` was last generated 2026-08-12 without
them and today's regeneration needed `--allow-gaps` as well. They were not
described from guesswork, because the glossary deliberately duplicates HISTORY's
narrative and a wrong description there propagates into how the work is
described. Whoever built those four selectors should write the entries.

---

## Current state, so nothing is re-derived

**Repository.** `58740ce` on `paper-exact/acquisition-v1`, in sync with origin.
Zero modified tracked files, zero staged. **87 untracked paths, all predating
2026-08-18**: ~9.6 GB of `cache/`, `dataset_cache/`, `data/`; 19 `.bak` CSV
snapshots; 31 loose one-off scripts in the repo root. These are tidy-up debt, not
findings. Deleting or committing them is Omri's call and was left alone.

**Verified this session.** Step 273 reproduces: `STAGE_0_BASELINES.csv/.md`,
`STAGE_1_LOCAL.md`, `STAGE_1_LOCAL_AGGREGATE.csv` and
`STAGE_1_LOCAL_INTERVALS.csv` byte-identical, `STAGE_1_LOCAL_SELECTION.json`
identical in every field but `score_sha256`. Residual is one column (`threshold`,
108/138 rows, max 1.377e-14) plus machine-epsilon diagnostics; no prediction
flipped. The frozen protocol is
`docs/experiments/LOCAL_ONLINE_COMPREHENSIVE_V1.frozen-c921b0d4.md` — **not** the
editable `LOCAL_ONLINE_COMPREHENSIVE_V1.md`, which hashes to `b5991a89…`.

**Acquisitions.** DeepConf K=512 at 15,360/15,360 traces, 0 failed, 24/24 gates;
REFRAIN at 1,000/1,000. Both byte-identical on Drive
(361 / 20,189,077,984 B and 22 / 3,185,291,662 B). `m2_deepconf_full`
(297 / 17,744,439,979 B, K=4096, partial) is kept separately and **must not be
merged with the K=512 pool**.

**Blocked.** `codex/fair-paper-exact-comparisons-v1` is not on our remote and
none of its four named assets exist here, so the approved CPU-first scoring of
the shared 3,400-row table cannot start. It must not be reimplemented from its
description — that would create a second divergent definition of the same table.
Question 3 in `HANDOFF_CODEX_2026_08_18.md`.

**Unblocked and not waiting on anyone.** The offline DeepConf derivation over the
K=512 pool. No GPU, no registry.

**Cluster.** Reachable again; the VPN outage during writing is resolved. The
sync and publish that had failed mid-transfer (`tar: Cannot write: Broken pipe`)
were re-run and verified: `repo_docs/` now carries the stamp of `022d995` and
matches the filtered source at **577 objects / 64,211,939 B**, with `rclone
check` reporting 0 differences. The queue is empty and nothing is running.

One correction worth carrying forward, because it was stated wrongly first:
`repo_docs/` was **not** missing the LOS-Net extraction while it was missing from
git. `cluster/sync_code.sh` tars the *working tree*, not the git index, so an
untracked file syncs and publishes like any other. Drive and git can therefore
disagree in this direction, and only git's view is the one a fresh clone gets —
which is exactly why the audit in Job 2 must check `git ls-files` rather than the
filesystem or Drive.

Treat an empty `bash cluster/upload_docs.sh --status` output as "ssh is down",
not "nothing to report".

---

## Traps this session actually hit — do not re-learn them

- **A background command's exit code is the last command's.** `python … ; echo
  EXIT=$?; tail …` reports `tail`'s success while the script crashed. Put the
  real exit code in the output and read it.
- **`pgrep -f 'X'` matches the shell running it**, because that pattern is in its
  own command line. It reported an upload that did not exist. Write `[X]`.
- **Backslashes are collapsed once** between a heredoc and Python, so `\\n`
  inside a patch script becomes a real newline in the generated source. Build
  such literals with `chr(92)`, and run `ast.parse` after every patch.
- **MSYS `grep` reads in text mode** and reported zero CRs in a file that had one
  per line. Byte counts do not lie; `grep` did.
- **An empty `--status` output means the transport failed**, not that the thing
  being checked is idle.

## Standing constraints that remain in force

Never commit live `submit_*.sbatch` (hardcoded HF_TOKEN). All cluster polling via
`/aircc-status` or the `cluster-ops` sub-agent, never raw ssh loops in the main
context. Work only under `/shared/cycle2_tau_averbuch_prj/omrisegev1`. Never
pip-upgrade torch in the NGC container. Results reach Drive via `rclone` **on the
cluster**, never the Drive MCP tools and never a local hop. Never tune toward a
published number. Ask which method to evaluate — never infer it. No shorthand
labels (R1/R2, channel A/B) in results. Never say "Nadler" (the method is L-SML),
never "MV_EPR", never "recommended" in advisor-facing material.
