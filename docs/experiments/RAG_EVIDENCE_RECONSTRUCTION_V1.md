# RAG evidence reconstruction v1

Status: executable contract implemented; scientific A/B execution is blocked
until independent P0/P1 review approves the frozen source closure.

## Claim boundary

This lane is retrospective application evidence. RAGTruth labels were opened
in earlier project work. It does not produce a single RAG leaderboard and it
never averages across answers, sentences, scorer tokens, examples, claims, or
access regimes.

The release contains four access/estimand families and seven explicit panels:

1. RAGTruth evidence contrast: answer, sentence, and scorer-token ranking are
   three separate panels. The fixed mixed-v2 RAG IU-PCR transform and noctx/LOO
   heads are fitted without labels on development telemetry. LOO is used only
   where registered; Summary rows have full/noctx telemetry only and therefore
   use the noctx head.
2. GASP: sentence ranking on the local class-balanced 400-response cohort. It
   is a protocol reproduction with local IDs and a local splitter, not an exact
   reproduction of the unpublished paper sample. GASP-threshold and the fixed
   RAG head may be contrasted only inside this matched panel.
3. LettuceDetect: exact local example-level reproduction from raw cached model
   predictions. Its supervised access and example F1 remain separate from all
   evidence-contrast ranking metrics.
4. RefChecker: fixed-claim checking only. Three-way NLI accuracy/macro-F1 and
   binary unsupported-claim AUROC/AUPRC are different panels. Accurate, noisy,
   and zero-context settings are evaluated separately and never pooled. Claim
   extraction is out of scope. The binary row is explicitly a cross-scorer
   adaptation: a Qwen2.5/RAGTruth-fitted head is transferred without refit to
   Qwen3 fixed-claim telemetry; it is not a same-scorer reproduction.

The machine-readable authority is
`configs/reconstruction_benchmark_v1/rag_evidence.json`.

## Stage boundary

Preparation rederives all artifacts from the hash-pinned local assets under an
explicit `--source-root`. It writes a sanitized `FIT_INPUT.pkl` beneath the
release tree and a separate `PRIVATE_LABELS.pkl` beneath the private-control
tree. The fit-visible bundle contains neither targets nor source/bootstrap
group identifiers; even Lettuce gold prevalence is retained only in the
private target audit. Recursive forbidden-field validation runs before publish
and again inside the fit worker. Preparation certification reconstructs exact
fit/private bytes and public rosters again from the current registered sources.
The preparation source snapshot explicitly binds the real shared raw adapter,
`spectral_utils/ragtruth_evidence_contrast.py`, and the authenticated-byte
loader in `spectral_utils/reconstruction_benchmark/io.py`; there is no separate
`ragtruth_source_adapter.py`. It also binds every repo-local module reached by
the preparation adapter's module-level imports. A static closure regression test
fails if a future local dependency is imported without joining the certified
source roster.

All four raw adapters parse only an in-memory byte string read from the held
source descriptor and verified against its registered digest. They never hash
one pathname/object and then reopen or stream a second object for parsing; an
in-place modify/restore or pathname ABA therefore cannot authorize different
parsed bytes.

The fit controller copies a data-free code capsule and launches a fresh Python
worker after installing a deny-default audit hook. Registered denial probes
cover the private label sidecar, every raw source, the full registry,
preparation code, and post-freeze evaluation code. The worker emits freshly
computed scores only; historical result paths are not registered inputs. Score
source files are held from snapshot through worker completion, and the capsule
is materialized only from those authenticated snapshot bytes. Its exact file
roster and byte hashes are compared to that snapshot before launch and the held
source bindings plus an independently reopened end snapshot are revalidated
afterward. Score
certification then constructs a third fresh capsule, reruns the restricted
worker, and requires exact score bytes, fit diagnostics, and capsule-tree
identity with both A and B. Coordinated rewrites of both build trees therefore
do not pass merely because A equals B.

Evaluation cannot open the private labels until a PASS score A/B certificate
has been transitively rederived. It uses 20,000 deterministic complete-source
group bootstrap draws. `metrics.csv`, `predictions.csv`, `contrasts.csv`, and
`panel_status.csv` are tidy, byte-comparable reporting artifacts. The only
paired contrast table is the matched GASP sentence panel. Evaluation
certification opens the private labels only after score authentication, then
independently regenerates and byte-compares every reporting table and panel
status against A and B.

These four canonical CSVs are the complete output of this lane, not the final
integrated reporting package. The downstream reporting bridge must emit typed
Parquet plus an explicit column/type schema before claiming dataset/cell-level
reporting readiness. That conversion is outside this scientific lane and may
not alter panel identities, units, access regimes, estimands, or row values.

Every directory stage holds and inode-binds its parent and staging directory,
then publishes with fd-relative `renameat2(RENAME_NOREPLACE)` on Linux or
`renameatx_np(RENAME_EXCL)` on macOS. Sensitive preparation/evaluation writes
are stage-fd-relative, so swapping the pathname parent cannot redirect private
labels. Every ancestor is opened component-by-component with `O_NOFOLLOW`, and
release IDs are restricted to one conservative path component. Certificates
use held-parent fd-relative exclusive create/rename and inode checks, and
cannot replace an existing certificate. A substituted object actually moved
to a final name is atomically quarantined; cleanup never unlinks or recursively
deletes a mutable name, so race evidence and unrelated siblings are preserved.
External fit tools also check the held stage/path binding before and after
execution; as with any portable macOS pathname-based subprocess, the release
parent must have a single trusted writer during that execution window.

Public and private preparation directories carry the same deterministic pair
transaction marker. A crash after publishing only one member is recoverable:
the next identical invocation holds both parents and both available artifact
directories, validates content through those object fds, reasserts the two
canonical-name/inode bindings as one recovery transaction, quarantines the
exact orphan, and reasserts that both canonical names are empty before rebuild.
An already complete, byte-exact pair is an idempotent success only after the
same paired binding reassertion. That reassertion holds and revalidates every
root marker, manifest, nested input directory, fit input, and private-label
file by name, inode, metadata, roster, and exact bytes across validation of both
members. A changed or non-matching member causes both current canonical entries
to be preserved in quarantine and fails closed.

## Scientific command order

The following is a protocol, not authorization to execute it. Replace the
release ID and use the main repository as `SOURCE_ROOT`; the worktree does not
contain the large ignored caches.

```bash
PY=/Users/osegev/Desktop/hallucination_detection/.venv/bin/python
SOURCE_ROOT=/Users/osegev/Desktop/hallucination_detection
RELEASE_ID=<approved-release-id>

$PY scripts/reconstruction_benchmark/prepare_rag_evidence.py --release-id "$RELEASE_ID" --build A --source-root "$SOURCE_ROOT"
$PY scripts/reconstruction_benchmark/prepare_rag_evidence.py --release-id "$RELEASE_ID" --build B --source-root "$SOURCE_ROOT"
$PY scripts/reconstruction_benchmark/verify_rag_evidence_preparation_ab.py --release-id "$RELEASE_ID" --source-root "$SOURCE_ROOT"

$PY scripts/reconstruction_benchmark/run_rag_evidence_methods.py --release-id "$RELEASE_ID" --build A --source-root "$SOURCE_ROOT"
$PY scripts/reconstruction_benchmark/run_rag_evidence_methods.py --release-id "$RELEASE_ID" --build B --source-root "$SOURCE_ROOT"
$PY scripts/reconstruction_benchmark/verify_rag_evidence_ab.py --release-id "$RELEASE_ID" --source-root "$SOURCE_ROOT"

# Post-freeze evaluation runs from a separate clean checkout at the approved
# evaluator commit.  Its trusted, stable interpreter and site-packages
# environment, plus its frozen release trees, are external to that checkout;
# score verification is delegated to the exact clean 4099003 checkout that
# produced the existing score certificate.
EVALUATION_REPO=<clean-evaluator-checkout-at-approved-commit>
SCORE_VERIFIER_REPO=/Users/osegev/Desktop/hallucination_detection/.worktrees/reconstruction-rag-run-v1
RELEASE_ROOT=/Users/osegev/Desktop/hallucination_detection/.worktrees/reconstruction-science-run-v1/results/reconstruction_benchmark_v1/releases
PRIVATE_ROOT=/Users/osegev/Desktop/hallucination_detection/.worktrees/reconstruction-science-run-v1/results/reconstruction_benchmark_v1/private_control

test "$(/usr/bin/git -C "$SCORE_VERIFIER_REPO" rev-parse HEAD)" = 409900332854c0586c4abc7dbc33f10b565b59af
test -z "$(/usr/bin/git -C "$SCORE_VERIFIER_REPO" status --porcelain=v1 --untracked-files=all)"
cd "$EVALUATION_REPO"

$PY scripts/reconstruction_benchmark/evaluate_rag_evidence.py --release-id "$RELEASE_ID" --build A --source-root "$SOURCE_ROOT" --score-verifier-repo "$SCORE_VERIFIER_REPO" --release-root "$RELEASE_ROOT" --private-root "$PRIVATE_ROOT"
$PY scripts/reconstruction_benchmark/evaluate_rag_evidence.py --release-id "$RELEASE_ID" --build B --source-root "$SOURCE_ROOT" --score-verifier-repo "$SCORE_VERIFIER_REPO" --release-root "$RELEASE_ROOT" --private-root "$PRIVATE_ROOT"
$PY scripts/reconstruction_benchmark/verify_rag_evidence_evaluation_ab.py --release-id "$RELEASE_ID" --source-root "$SOURCE_ROOT" --score-verifier-repo "$SCORE_VERIFIER_REPO" --release-root "$RELEASE_ROOT" --private-root "$PRIVATE_ROOT"
```

Debug preparation/fit flags make those artifacts non-scientific. The three
post-freeze evaluation CLIs above are science-only and expose no debug mode.
A scientific certificate rejects debug preparation or fit artifacts. No
command accesses Google Drive, and none downloads a model or dataset.

## Review gates before execution

- P0: target fields and source/bootstrap identities cannot cross the fit
  boundary; denial probes and sticky firewall violations fail closed.
- P0: both preparation builds rederive identical bytes from every registered
  raw asset; a third restricted score execution and a post-certificate private
  re-evaluation authenticate exact score/capsule/table bytes instead of
  trusting A/B equality or self-attested manifests.
- P0: preparation revalidates the complete raw-asset binding after all reads;
  source pathname substitution between the initial hash pass and payload
  construction fails before any preparation publication.
- P0: scorer-token keys must equal the complete registered `(parent,index)`
  lattice exactly once, and every RAGTruth/GASP subgroup task carried by a
  score array must equal its private-label task binding before evaluation.
- P0: no historical score is accepted as an execution substitute.
- P0: parent swaps, staging-entry substitution, symlink parents/targets, and
  target-injection races fail closed without publishing or overwriting the
  replacement tree; wrong-inode outputs are quarantined rather than deleted.
- P0: unsafe release identifiers and symlinked path ancestors are rejected,
  and private/public preparation crash windows have exact-marker recovery.
- P0: no cross-panel macro, cross-unit comparison, RefChecker setting pool, or
  cross-panel paired contrast can enter the certified tidy tables.
- P1: GASP fidelity remains protocol-level; Lettuce is exact local; RefChecker
  is checking-only; RAGTruth evidence-contrast is retrospective.
- P1: the lane's canonical CSV contract is not the integrated reporting
  bridge; typed Parquet and an explicit schema remain mandatory downstream.
- P1: complete source groups, not rows, are the bootstrap units for every
  panel, including scorer tokens and multiple claims from one example.
