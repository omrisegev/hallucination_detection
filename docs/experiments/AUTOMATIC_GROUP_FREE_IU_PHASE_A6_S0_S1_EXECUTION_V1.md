# Automatic group-free IU Phase A6 — S0/S1 execution contract v1

**Status:** `FROZEN_BEFORE_IMPLEMENTATION — INDEPENDENTLY REVIEWED — NO A6
TELEMETRY OR SEALED SIMULATOR SEED HAS BEEN OPENED`

**Review boundary:** the complete pre-freeze body at SHA-256
`5c869db42633d04bf4c46110d95de83891c6ca6b10fdf381653b8a618a750615`
received the independent verdict `NO BLOCKERS`. This status declaration records
that decision; it does not authorize opening telemetry or a sealed simulator
seed. Implemented source and runtime boundaries require a fresh no-edit review
before execution.

This document makes the already frozen A6 protocol mechanically executable for
S0a, S0b, and S1. It does not change the target, estimator, feature contract,
selection gates, or interpretation in
`AUTOMATIC_GROUP_FREE_IU_PHASE_A6_V1.md`. If this document conflicts with that
protocol, the conflict closes the draft and must be resolved before code. No
result may be used to edit this contract.

## 1. Exact model, tokenizer, and audit-model identities

The following repository revisions are immutable for this execution:

| role | repository | revision |
|---|---|---|
| Qwen source scorer 1 | `Qwen/Qwen3-4B` | `1cfa9a7208912126459214e8b04321603b3df60c` |
| Qwen source scorer 2 | `Qwen/Qwen3-8B` | `b968826d9c46dd6066d109eabc6255188de91218` |
| held Llama scorer | `meta-llama/Llama-3.1-8B-Instruct` | `0e9e39f249a16976918f6564b8830bc894c89659` |
| S0b prompt-NLL audit model | `EleutherAI/pythia-410m-deduped` | `c4fc8d586d62df497f1f9b69d66d3ca419992d3e` |

Throughout this contract, every string entering a hash is UTF-8; `\0` is one
zero byte; and every interpolated nonnegative integer is its minimal unsigned
ASCII decimal with no leading zero unless a field explicitly specifies fixed
zero padding such as `jjj`. For SHA-256 digest bytes `h`,
`first64(h)=int.from_bytes(h[:8],"big",signed=False)`. The phrases “first 64
bits” and `first64(SHA256(...))` always mean this definition.

Boundary preparation first resolves each repository at the exact revision.
For tokenizer snapshots it selects only the literal allowlist
`config.json,generation_config.json,tokenizer.json,tokenizer_config.json,
special_tokens_map.json,added_tokens.json,vocab.json,merges.txt,tokenizer.model,
sentencepiece*.model,chat_template*.jinja`; the repository tree and resulting
exact selected path list are stored. It follows ordinary Hugging Face cache
links once, verifies the resolved revision, and copies the bytes into a new
content-addressed boundary-input directory containing regular files only. It
then writes a canonical manifest of **all** files in that directory, relative
paths, sizes, and SHA-256 values before importing Transformers or constructing
a tokenizer. Mutable branch/cache links and a later extra/missing path are
forbidden. S0a execution then sets `HF_HUB_OFFLINE=1` and
`TRANSFORMERS_OFFLINE=1`, uses only the manifested snapshot with
`local_files_only=True`, and fails on any missing, extra, or changed file. Thus
the boundary inputs are predeclared bytes, not a post-hoc list of files that a
particular `AutoTokenizer` happened to open. Both Qwen checkpoints are
manifested and audited separately even if their tokenization proves identical.

The Llama repository is gated; lack of licensed access is
`BLOCKED_TOKENIZER_ACCESS`, not permission to use a proxy. `trust_remote_code`
is exactly `False`, a fast tokenizer with offsets is mandatory, and the
boundary freezes Python, Transformers, Tokenizers, NumPy, SciPy,
scikit-learn, PyTorch, platform, and CPU/BLAS thread settings. Development
proxy-tokenizer results are never boundary evidence.

### 1.1 One contextual scorer-input builder

Every quartet prompt and fixed response is audited in the exact input context
used by teacher forcing. No standalone-response tokenization is admissible.

For Qwen, render with `enable_thinking=False`; for Llama no nonstandard template
argument is passed. For prompt `P` and response `R`:

1. `prefix_text` is `apply_chat_template([{role:user, content:P}],
   tokenize=False, add_generation_prompt=True, ...)`.
2. `full_text` is `apply_chat_template([{role:user, content:P},
   {role:assistant, content:R}], tokenize=False,
   add_generation_prompt=False, ...)`.
3. Require `full_text.startswith(prefix_text)`, require the assistant response
   to begin exactly at `len(prefix_text)`, and require that exact occurrence to
   be the final occurrence of `R` in `full_text`. Otherwise close.
4. Tokenize `full_text` once with `add_special_tokens=False`,
   `padding=False`, `truncation=False`, and `return_offsets_mapping=True`.
   Assert that these token IDs equal the IDs returned by
   `apply_chat_template(..., tokenize=True, add_generation_prompt=False)` with
   the same template arguments.
5. The contextual response span contains every token whose character interval
   intersects `[len(prefix_text), len(prefix_text)+len(R))`. A token crossing
   the prefix/response boundary belongs to the response span. The remaining
   leading and trailing tokens are the contextual prefix and suffix. Offset
   intervals must be monotone, nonnegative, within `full_text`, and cover every
   nonempty response character. No `(0,0)` special-token offset may be assigned
   to the response span.

The complete template argument contract is frozen: Qwen uses only
`enable_thinking=False`; Llama receives no model-specific keyword; both use
`add_generation_prompt` exactly as above, with no tools, system message,
padding, truncation, continuation mode, or custom template. The resolved chat
template text and its SHA-256 are stored in the pre-execution snapshot
manifest.

For each scorer checkpoint, the contextual response-span IDs and suffix IDs for
one fixed response must be identical in all eight prompt/render cells where it
appears. A/B contextual-prefix token counts must be identical for each
rendering. Certificate length is the contextual response-span length and must
be in `[40,80]` separately under all three checkpoint tokenizers. The artifact
stores full/prefix/response/suffix hashes and lengths for all 16 cells. These
checks explicitly include BPE merges across the assistant boundary.

## 2. Exact S0a populations

### 2.1 Quartet populations

The two 900-group populations and the local/global rejection replay follow the
frozen Step-265 construction, with two pre-telemetry schedule clarifications.
Within every
`domain x mutation x grammar` cell, the 50 accepted group slots are assigned
outer fold `within_grammar_index mod 5`. Therefore every outer-held fold has
exactly ten groups in each of the 18 semantic/grammar cells and every outer
training set has exactly forty.

Quartet slots are consumed in this exact order: population
`qwen-source,llama-audit`; domain `arithmetic,relational,finite_logic`;
mutation `value_leaf,relation_operator,constraint_condition`; grammar
`short,certificate`; outer fold `0..4`; and within-fold slot `0..9`. The slot
string is
`<population>:<domain>:<mutation>:<grammar>:fold<fold>:<within_fold:02d>`.
Its attempt seed is the first unsigned big-endian 64 SHA-256 bits of
`"a6-s0-slot-v2\0" || slot_string`, and attempts are contiguous from zero
through 9,999. Exhaustion yields `CLOSE_INVALID_INTERVENTION_BOUNDARY`; a new
slot or namespace is forbidden.

Quartet identities are exactly

```text
source_record_id = population || ":fold" || fold || ":source:" || slot_string
donor_id         = population || ":fold" || fold || ":donor:" || slot_string
template_bank_id = population || ":fold" || fold || ":template-bank:" ||
                   domain || ":" || mutation || ":" || grammar || ":" ||
                   (within_fold mod 5)
template_id      = template_bank_id || ":instance:" || slot_string
```

The five bank labels are fixed pre-telemetry assignment strata; each is owned
by one population/fold and has two groups per semantic/grammar cell. Reuse is
legal only inside that owner. Source, donor, and template-instance IDs are
unique. Both bank and instance IDs are persisted in every public group row.

In every `arithmetic x mutation x grammar x outer-fold` cell, the first two
within-fold slots are the registered derived-answer subdomain. Acceptance
requires that neither reciprocal answer occur as a complete registered answer
atom in either prompt. This freezes 12 derived quartet groups per fold and 60
per population. All other quartet slots are the general subdomain; no
derived/non-derived relabelling is permitted after construction.

For each outer fold and each semantic/grammar cell, sort its forty training
group IDs by

```text
SHA256("a6-s0-inner-v1\0" || outer_fold || "\0" || group_id)
```

as unsigned big-endian bytes and assign the sorted rows round-robin to inner
folds `0..4`. Every inner validation cell has exactly eight groups and every
inner training cell has exactly 32. The complete mapping for all five outer
folds is serialized and hashed; no split is recomputed from row order.

### 2.2 Three 2,000-prompt natural cohorts

The cohort IDs are exactly:

- `qwen3-4b-natural`;
- `qwen3-8b-natural`; and
- `llama31-8b-natural`.

Their `cohort_index` values are exactly `0,1,2` in that order. Each cohort has
five folds of exactly 400 prompt slots. In fold `f`, enumerate
the nine `domain x mutation` cells in canonical domain-major order, rotate the
order left by `(cohort_index + 2*f) mod 9`, assign 45 slots to the first four
cells and 44 to the remaining five, and concatenate in that rotated order.
Thus every cell has 44 or 45 rows per fold and 222 or 223 per cohort.

The canonical construction order is cohort order as listed above, then folds
`0..4`, then the rotated cell order, then `within_cell=0..cell_count-1`.
The exact slot string is
`<cohort_id>:fold<fold>:<domain>:<mutation>:<within_cell:03d>`. “First ten”
below means `within_cell=0..9` in that order.

Inside every `cohort x fold x arithmetic x mutation` cell, the first ten slot
IDs are the registered `derived-answer` subdomain. This freezes 30 derived
natural prompts per fold and 150 per cohort. No relational or finite-logic slot
is claimed as part of this specific non-copy subdomain. A derived slot is accepted only
when neither answer of the mechanically generated reciprocal source pair occurs
as a complete registered answer atom in either canonical prompt. Only one
world, selected by the least significant bit of
`SHA256("a6-natural-side-v1\0" || slot_id)`, becomes the natural prompt: bit
zero selects A and bit one selects B. The
other world and both answers are discarded before the public manifest is
created. The remaining slots are the `general` subdomain and are not required
to be copy or non-copy. The derived gate is evaluated only on these exact
predeclared arithmetic slots, never on a post-hoc subset.

The natural prompt's `mutation_family` records the typed reciprocal generator
bank that produced its well-formed world. It is provenance and a transfer
stratum, not a mutation applied to the eventual on-policy response.

The exact prompt is

```text
Answer the following task with one short answer.
Task: <canonical rendered task>
Answer:
```

Natural attempts use the same typed generator and independent evaluator as the
quartets but do not call any response renderer. The per-slot seed is the first
64 bits, big-endian, of

```text
SHA256("a6-natural-attempt-v1\0" || cohort_id || "\0" || slot_id ||
       "\0" || attempt_index)
```

and attempts `0..9999` are consumed contiguously. Exhaustion has the same hard
closure. Source, donor, and template banks are namespaced by `cohort x fold`;
specifically

```text
item_id          = SHA256("a6-natural-item-v1\0" || slot_id)
source_record_id = cohort_id || ":fold" || fold || ":source:" || slot_id
donor_id         = cohort_id || ":fold" || fold || ":donor:" || slot_id
template_bank_id = cohort_id || ":fold" || fold || ":template-bank:" ||
                   (within_cell mod 20)
template_id      = template_bank_id || ":instance:" || slot_id
complete_prompt_id = SHA256("a6-natural-prompt-v1\0" || cohort_id ||
                            "\0" || prompt_text)
```

No identity crosses a fold or cohort. For a natural slot, both reciprocal
semantic-task hashes and both canonical prompt-content hashes are inserted in
the private collision registry even though only the selected world enters the
public prompt manifest; the discarded world and both answers are otherwise
discarded. This makes replay independent of whether a later slot regenerates
the unused reciprocal world.

The single global construction order is the two quartet populations in the
order above, followed by `qwen3-4b-natural`, `qwen3-8b-natural`, and
`llama31-8b-natural` in the canonical order already defined. One registry spans
that entire order and rejects repeated semantic-task hashes, raw prompt-content
hashes, source IDs, donor IDs, template-instance IDs, or complete prompt IDs.
Template-bank reuse is legal only inside its exact population/cohort and outer
fold owner; a bank crossing an owner is rejected. Control-3 eligibility requires
distinct `template_bank_id`, not merely distinct per-slot instance IDs. Every
local and global reject is serialized before another attempt is examined.
Checkpoint resume restores the exact accepted-hash sets by replaying and
verifying all earlier ledgers; it never reconstructs sets from only the latest
accepted rows.

Every 2,000-row prompt manifest also freezes its future generation contract.
The generation seed is the first unsigned big-endian 64 SHA-256 bits of
`"a6-natural-generation-v1\0" || cohort_id || "\0" || item_id`.

The model input is never raw prompt tokenization. It is exactly the Section 1.1
user-only `prefix_text` with `add_generation_prompt=True`, no system message or
tools, Qwen `enable_thinking=False`, and the default Llama template.
`input_ids` are returned by
`apply_chat_template(...,tokenize=True,add_generation_prompt=True)` and must
equal tokenization of `prefix_text` with
`add_special_tokens=False,padding=False,truncation=False`. Generation uses one
item per batch, `input_ids` shape `(1,T)`, an all-one int64 attention mask of
the same shape, and no padding.

Generation is greedy with `do_sample=False`, `num_beams=1`,
`max_new_tokens=150`, no custom
stopping strings or logits processors, and stops only at the first generated
token in the exact boundary-frozen `eos_token_id` set or at the length limit.
`pad_token_id`, all EOS IDs, generation-config bytes, chat-template bytes, and
model/tokenizer revisions are manifest fields. The manifest stores exact input
IDs, prefix text/ID hashes and lengths, and the attention-mask hash. The response is the new-token
slice decoded with `skip_special_tokens=True` and
`clean_up_tokenization_spaces=False`. The future row schema stores prompt-token
IDs/hash, generation seed, generated-token IDs/hash, stop reason, decoded
response SHA-256, and contextual response-span evidence. Greedy generation is
still assigned a seed so every row has one immutable provenance key; the seed
cannot authorize sampling.

### 2.3 Qwen firewall and Llama future sidecar

The public Qwen natural row type contains only:

```text
item_id, cohort_id, scorer_id, outer_fold, domain, mutation_family,
subdomain, attempt_index, prompt_text, prompt_sha256, semantic_task_sha256,
source_record_id, donor_id, template_bank_id, template_id, complete_prompt_id,
tokenizer_evidence
```

Here `tokenizer_evidence` is a fixed nested scalar/list schema containing only
the manifested tokenizer IDs, prompt-prefix token IDs/hashes/lengths, and
future generation seed/parameters; it is not an opaque mapping and contains no
response-named key, even empty or null. Future response field names exist only
in the separate schema declaration below.

It has no answer, solution, correctness, label, target, response, feature,
sidecar path, or opaque payload field. Its loader reconstructs a new object from
that exact allowlist and rejects target-like keys recursively. Poison objects
whose forbidden properties raise on access must load without those properties
being touched; every Qwen fit API accepts only sanitized row IDs plus numeric
feature matrices and cannot accept the construction object.

S0a creates the identical prompt-only type for Llama plus a separate *schema
declaration*, not a data object, for a future opaque sidecar keyed by
`(cohort_id,item_id,response_sha256)`. It asserts that no Llama response,
feature file, correctness file, or sidecar path exists **inside the exact A6
S0a/S2 namespace, output roots, or manifest objects**. Unrelated pre-existing
Llama caches elsewhere in the repository are outside this assertion and are
never scanned or opened. The post-S2a joint
Llama boundary must later prove 1:1 key/count/hash joins without exposing label
payload to S2b.

### 2.4 PopQA reservation without label access

S0a reserves the A0 confirmation surface as
`akariasai/PopQA` revision
`098765c79ea10a2cb19c828324e33281b8336ec0`, exactly 14,267 test rows. It
creates only the opaque identities `popqa:test:<row_index>` for indices
`0..14266`, the frozen A0 prompt-template hash, dataset revision, expected row
count, and later-validation schema. The literal frozen prompt template is
`Answer the following question with one short answer.\nQuestion: {question}\nAnswer:`
with UTF-8 SHA-256
`97cc05f94fecfc2e30dd3751c2e800039d196b149d538376777aa837c5123963`.
It does not download or parse the TSV and
does not access question text, objects, aliases, or possible answers. S4 must
validate those reserved row indices against the same revision before any
response or label is opened. The `popqa:` namespace is disjoint by construction
from all A6 source/donor/template/item namespaces.

## 3. S0a null strata and partition proofs

For each quartet group define, before telemetry:

```text
length_scalar = mean over 4 renderings and 3 checkpoint tokenizers
                of the contextual prompt-prefix token count
complexity_scalar = AST_node_count + 2 * solution_depth
```

A/B equality is already mandatory, so one value exists per rendering. Within
each fixed `domain x mutation x grammar` cell, stable-rank each scalar by
`(value,group_id)` and assign quartile bins `floor(4*rank/50)`, capped at 3.
The initial null cell is the categorical cell plus `(length_bin,complexity_bin)`.

The required partitions are every outer train/held set and every inner
train/validation set inside every outer fold. Inside one categorical cell,
repeatedly merge initial-bin components until every resulting component has at
least four groups in every required partition:

1. ranks are zero-based and empty initial bin pairs are not components;
2. choose the lexicographically first deficient component by its sorted member
   `(length_bin,complexity_bin)` tuple;
3. among all other occupied components, compute the minimum integer Manhattan
   distance between member-bin pairs; the component(s) at the smallest distance
   are the adjacent occupied bins after empty bins are collapsed;
4. choose the lexicographically first such partner, union the pair, and repeat.

No count-maximizing or centroid rule is used. If no partner exists or the rule
terminates with a count below four, S0a closes. This procedure is run separately
for `qwen-source` and `llama-audit`; population ID is part of every stratum ID.
A final stratum ID is the SHA-256 of population, categorical cell, and sorted
member-bin tuples. The artifact stores zero-based ranks, occupied/empty bins,
every union operation, partition counts, and final IDs. Replay tests reproduce
every merge from the serialized sidecars. Null mappings are shared across
scorers and renderings exactly as in the parent protocol.

## 4. Exact 16-row shortcut table and S0b audit

For each reciprocal group, scorer, rendering, prompt world, and response world,
create one row; there are 16 rows per `group x scorer`. The mechanical target is
`1[prompt_world != response_world]`. Rows carry their indivisible group ID and
the following fixed forbidden-fit columns:

Continuous columns:

```text
prompt_char_length, prompt_word_length,
qwen4_prompt_tokens, qwen8_prompt_tokens, llama_prompt_tokens,
response_char_length, response_word_length,
qwen4_response_span_tokens, qwen8_response_span_tokens,
llama_response_span_tokens,
ast_node_count, solution_depth, changed_node_count,
prompt_levenshtein_distance,
prompt_response_token_jaccard, answer_atom_in_prompt,
numeric_rarity_mean, numeric_rarity_max,
entity_rarity_mean, entity_rarity_max,
pythia_prompt_mean_nll
```

`prompt_levenshtein_distance` is Unicode-code-point Levenshtein distance between
the two same-render prompt strings and is repeated on their four crossed rows.
`prompt_response_token_jaccard` uses sets from NFKC-lowercased
`[A-Za-z0-9_]+|[^\w\s]` tokens; two empty sets score one. `answer_atom_in_prompt`
uses the row's own response-AST canonical answer and the frozen complete-atom
boundary matcher. `changed_node_count` is the number of typed AST paths changed
between the two worlds; `changed_node_type` is their sorted typed-path-kind
tuple. All equations are tested on the four crossed rows, not only diagonal
rows.

Categorical columns, one-hot encoded from boundary-frozen vocabularies, are
`domain, mutation_family, response_grammar, rendering_family,
changed_node_type, template_bank_id, template_id, donor_id`. Unknown held categories
are all-zero; no data-dependent category regrouping is allowed. Vocabularies
are the UTF-8 byte-sorted unique values in the full Qwen S0b construction
population and are stored before any classifier fit. The separate Llama audit
uses this same frozen encoder; Llama-only bank identities map to the registered
all-zero unknown encoding and cannot enlarge the vocabulary after Qwen
inspection.

Pythia is a second pre-resolved offline snapshot. Its boundary manifest covers
every weight shard, index, config, generation config, tokenizer file, and
special-token file at the revision in Section 1, under the same immutable-path,
no-extra-file, and offline-only rules. It runs in `eval()` and
`torch.inference_mode()` on CPU with one thread, deterministic algorithms,
float32 weights/logits, and no autocast; logits are cast to float64 before a
SciPy `logsumexp` subtraction. The boundary freezes exact PyTorch/SciPy
versions, device, dtype, and thread environment.

Pythia input is the UTF-8 rendered task text only, without chat wrapping.
Tokenization uses `add_special_tokens=False`, `padding=False`, and
`truncation=False`. Mean next-token NLL is
`mean_t[-log_softmax(logits[t-1].float64)[input_id[t]]]` for `t=1..T-1`;
empty/one-token input closes S0b. Pythia never sees responses.

For each ridge coefficient `lambda in {0.01,0.1,1,10}`, minimize the explicitly
implemented objective

```text
L(w,b)=sum_i omega_i * log(1+exp(-y_i*(x_i'w+b)))
       + .5*lambda*||w||_2^2,
omega_i = 1/(2*n_{class(i)}), y_i in {-1,+1}.
```

The class weights sum to one, the intercept `b` is unpenalized, and every fit
starts from exact all-zero `w,b`. Analytic objective/gradient are used with
SciPy L-BFGS-B, `maxiter=10000`,
`ftol=gtol=1e-12`. A fit is usable only when finite and the maximum absolute
analytic-gradient component is at most `1e-8`; otherwise S0b closes. The
logistic term is evaluated as stable `numpy.logaddexp(0,-margin)`, never direct
`log(1+exp(...))`. Training-fold mean/std standardization with population
denominator `ddof=0` is used for continuous columns; zero-std columns are fixed
to zero, and the frozen one-hot encoder is unchanged. Evaluate AUROC
inside each held group fold and macro-average the five fold AUROCs. Never rank
concatenated OOF probabilities. Fit and gate two separate shortcut audits: one
on `qwen-source` (the two scorer replicas remain inside the indivisible group)
and one on `llama-audit`. No row, encoder fit, prediction, or bootstrap draw is
pooled across populations. The reported shortcut statistic in each population
is the maximum five-fold macro over the four ridges; ties choose the larger
`lambda` only for artifact naming. The five full training-fold fits per ridge
produce one immutable OOF bundle. Overall/domain/cell/grammar/render statistics
filter rows from those OOF predictions and recompute fold-local AUROCs; they do
not fit restricted models.

Gate names are exactly `overall`, `domain:<domain>`,
`cell:<domain>:<mutation>`, `grammar:<grammar>`, and `render:<render>` using the
canonical literal orders in Section 2. Bootstrap one multiplicity per group inside
`domain x mutation x grammar`, reuse it across scorers/renderings/cells, and
recompute fold-local AUCs and the maximum over ridges. Use exactly 20,000 draws;
seed is the first 64 SHA-256 bits of
`"a6-s0b-shortcut-v1\0" || population_id || "\0" || gate_name`. Initialize one
PCG64 generator from that seed and traverse UTF-8-sorted
`domain x mutation x grammar` strata; for a stratum with `n` sorted group IDs,
the next generator call draws `integers(0,n,size=(20000,n))`; rows correspond
to draws, and each row is converted by `bincount(...,minlength=n)` to group
multiplicities. The fixed stratum arrays are concatenated in that order and
multiplicities are reused across all scorers/renderings/cells and all four ridge bundles. The
two-sided 95% upper endpoint is
ascending order statistic 19,501 in one-based indexing (`method="higher"` at
0.975). The endpoint must be at most 0.60 overall, per domain, in all nine
domain/mutation cells, for each grammar, and for each rendering. All must pass.
The bootstrap resamples the already frozen OOF predictions and does not refit
the shortcut model; it measures grouped evaluation uncertainty. Model-selection
uncertainty is handled conservatively by recomputing the maximum over all four
ridge OOF score bundles in every draw. Qwen and Llama must each pass every
endpoint independently; a pooled diagnostic can never rescue either.

Independently of the classifier, duplicate-row-weighted validity prevalence is
exactly 0.5 for every prompt hash alone and every response hash alone, within
`1e-12`; this is a structural equality, not an estimated classifier score.

### 4.1 Control-3 vector and matching freeze

The S0b matching vector is a Qwen-only group vector. First fit one label-free
full-Qwen-population row-level mean/std standardizer; zero-std continuous
columns are retained as zero. Exclude group-unique source/donor/template-instance
one-hots and template-bank one-hots from distance (raw identities remain
eligibility constraints). Retain the
fixed categorical one-hots for domain, mutation, grammar, rendering, and
changed-node type. For each group, flatten the 32 standardized Qwen row vectors
in the exact order `scorer(qwen4,qwen8) x rendering(canonical,paraphrase,
layout,notation) x prompt(A,B) x response(A,B)`. Euclidean distance is divided
by the square root of this flattened vector dimension.

For the caliper pool, enumerate each unordered eligible Qwen group pair exactly
once over the full population, never once per partition; require a common final
S0a null stratum and distinct source, donor, and template-bank IDs. A pair that co-occurs
in several inner/outer partitions contributes one distance. Each partition
graph filters that single global edge set to endpoints in that partition. The caliper is the global
75th percentile of those finite distances, ascending one-based order statistic
`ceil(0.75*N)`. The exact Fisher-Yates/Hungarian algorithms, 200 seed schedules,
perfect matching proofs, and distinct hash rules are those in the parent
protocol. All are materialized and hashed in S0b before response telemetry.
Unique source/donor/template/group IDs are excluded from the numeric distance
and used only in eligibility. Canonical partition IDs, in order, are for each
outer fold `o=0..4`: `outer:o:train`, `outer:o:held`, followed by inner fold
`i=0..4` IDs `outer:o:inner:i:train` and
`outer:o:inner:i:validation`. Within a partition, final stratum IDs and group
IDs are UTF-8 byte sorted.

For family `c in {2,3}` and draw `j=0..199`, define `seed_u64` as the first
unsigned big-endian 64 bits of
`SHA256("a6-s0b-control-v1\0" || decimal(c) || "\0" || decimal(j))` and
`seed_bytes=seed_u64.to_bytes(8,"big")`. Control 2 handles each
partition/stratum independently. For attempt `a=0..9999`, seed a fresh PCG64
with first64 of

```text
SHA256(seed_bytes || "\0" || partition_id || "\0" || stratum_id ||
       "\0attempt:" || decimal(a)).
```

Starting from sorted group IDs, run in-place Fisher-Yates for
`i=n-1,...,1`, drawing exactly
`k=Generator.integers(0,i+1,dtype=int64)` once per `i` and swapping positions
`i,k`. The first permutation with zero fixed points is used; exhaustion closes
S0b. The complete response block follows that one mapping across scorers and
renderings.

For Control 3, recipient and donor rows are sorted group IDs. For every
eligible directed edge, the exact unsigned 256-bit primary cost key is

```text
int_big_endian(SHA256(seed_bytes || "\0" || partition_id || "\0" ||
                      recipient_id || "\0" || donor_id)).
```

Self/ineligible edges are absent. Any finite-key collision inside a matching
matrix closes S0b. For `n` rows set `B=(n+1)^n`; edge `(row i, donor j)` has
exact Python-integer cost
`primary_key*B + j*(n+1)^(n-1-i)`. A custom shortest-augmenting-path Hungarian
implementation using Python arbitrary-precision integer potentials minimizes
the sum, visiting rows/columns in ascending index order. The secondary term
makes equal-primary-total solutions choose the lexicographically smallest donor
index tuple. Every row/column is used once and fixed points are absent by
eligibility; absence of a perfect assignment closes. Unit tests compare each
solver result against brute-force enumeration on fixed and 1,000 development
graphs of sizes `2..8`, including tied-primary and infeasible cases. No float conversion, PRNG draw, or tie-only jitter is used by
Control 3. Partition names, edge lists,
assignments, and pseudo-contrast schedules are serialized as canonical JSON
with UTF-8, sorted keys, separators `(',',':')`, no NaN/Infinity, and a terminal
newline; SHA-256 is over those exact bytes.

## 5. Exact S1 transformed-coordinate simulator

S1 tests the estimator/selector, not the feature extractor, so it begins in the
nominal 30 named mixed-v2 coordinate system. It uses a simulator-only immutable
`ObservedTransformedTarget` containing exact names, a `2000 x 30` finite
natural-z matrix, its empirical covariance, and an explicitly supplied IU
confidence weight. No `FixedMixedV2Transformer` or
`fit_natural_coordinate_system` is called on these already transformed rows.

Both the real `A6NaturalCoordinateSystem` and simulator object adapt to one
small `TargetAnchorView(names,natural_z,covariance,iu_weight)` protocol. The
factorial moments, nested selector, covariance projection, trust rule, and
affine deployment consume only that view; they cannot branch on adapter type.
Tests compare real and simulator adapters instantiated with identical
transformed arrays/anchors and require byte-identical core outputs. Estimator
functions receive only observed arrays, names, folds, and this anchor view.
Planted bits/directions live in a separate diagnostic object that cannot enter
any fit signature.

### 5.1 Exact RNG, covariance, and coefficient geometry

Development uses namespace `a6-s1-dev-v1`; sealed execution uses
`a6-s1-sealed-v1`. For world integer `w`, repetition integer `r`, and the
literal ASCII subkey `k`, seed NumPy `PCG64` with the first unsigned big-endian
64 bits of

```text
SHA256(namespace || "\0world:" || decimal(w) ||
       "\0rep:" || decimal(r) || "\0" || k).
```

Every normal draw is `Generator(PCG64(seed)).standard_normal(shape,
dtype=float64)` in C order. For covariance `C=L L'`, row normals become
`G @ L.T` using NumPy's lower-triangular `cholesky`. QR is
`numpy.linalg.qr(..., mode="reduced")`; each column is multiplied by the sign
that makes its largest-absolute entry positive, with a tie going to the lowest
coordinate index. All reductions, arrays, and saved scores are float64; the S1
boundary pins NumPy/SciPy/BLAS versions and one thread.

The complete main-repetition subkey table is:

| subkey | draw/operation |
|---|---|
| `covariance_q` | standard normal `(30,30)`, then signed QR |
| `target_raw` | standard normal `(9,30)` |
| `auxiliary_raw` | standard normal `(16,30)` |
| `natural_qwen4`, `natural_qwen8` | standard normal `(2000,30)` each |
| `quartet_b`, `quartet_a`, `quartet_r` | standard normal `(900,2,30)` each |
| `quartet_epsilon` | standard normal `(900,2,2,2,4,30)` |
| `source_render_nll` | standard normal `(900,2,3)` for noncanonical renders |
| `source_nll_epsilon` | standard normal `(900,2,2,2,4)` |
| `eval_epsilon` | standard normal `(18,400,30)` |
| `eval_nll_epsilon` | standard normal `(18,400)` |
| `eval_joint_perm:<environment_id>` | `PCG64(seed).permutation(400)` |

No function may consume one subkey's generator for a second purpose. The 18
environment IDs are canonical
`domain(arithmetic,relational,finite_logic) x mutation(value_leaf,
relation_operator,constraint_condition) x scorer(qwen4,qwen8)` order.

Let `Q` come from `covariance_q`, and
`lambda=exp(linspace(log(.6),log(1.8),30))`. If
`corr(A)=D^(-1/2) A D^(-1/2)` for `D=diag(A)`, define

```text
C_4 = .9*corr(Q diag(lambda) Q') + .1*I
C_8 = .9*corr(Q diag(roll(lambda,7)) Q') + .1*I
Cbar = (C_4+C_8)/2.
```

`normalize_Cs(v)` means `v/sqrt(v'C_s v)` and is invalid when the denominator
is nonfinite or at most `1e-12`. Apply ordered `Cbar` Gram-Schmidt to the nine
`target_raw` rows to obtain, in order, `t0`, three mutation vectors, three
domain vectors, and two target-scorer perturbations. Define coefficient (not
mean-shift) directions

```text
t_common[s] = normalize_Cs(.98*t0 + .10*v_target_scorer[s])
t_family[d,m,s] = normalize_Cs(.90*t0 + .35*v_mutation[m]
                               + .15*v_domain[d]
                               + .10*v_target_scorer[s]).
```

Index 24 is zero-based and is the registered `mean_top1_logprob` coordinate.
For world 8, project the **coefficient** to exact zero there:

```text
t8 = t_family - C_s^{-1}e24 * (e24' t_family)/(e24' C_s^{-1}e24),
then normalize_Cs(t8).
```

For each scorer separately, stack all of its common, family-specific, and
constrained target coefficients used by worlds 1,3,4,5,7,8 and derive their
ordered `C_s`-orthonormal span, dropping only residuals at most `1e-12`. Assign
the 16 `auxiliary_raw` rows as eight for qwen4 followed by eight for qwen8. In
each scorer block, project and normalize sequentially in `C_s` in this order:
three nuisance-render coefficients, three target-by-render interaction
coefficients, one nuisance-scorer coefficient, and one anchor-noise
coefficient. Thus every planted nuisance/interaction/evaluation-nuisance and
anchor-noise coefficient is exactly `C_s`-orthogonal to that scorer's complete
target span; target-scorer perturbations never enter nuisance.

For each world/scorer, `t_anchor` is the equal-environment mean target
coefficient, normalized in `C_s`; worlds 2 and 6 use `t_common` only as a valid
structural anchor. The supplied IU confidence coefficient is

```text
u_s = -normalize_Cs(t_anchor[s] + .50*v_anchor[s]).
```

For every target world and scorer, a deterministic population preflight uses
the oracle `r0=t_anchor` and first forms the raw residual

```text
r_raw = r0 - u_s * (u_s' C_s r0)/(u_s' C_s u_s).
```

The raw structural ratio is `||r_raw||_Cs/||r0||_Cs` and must be at least
`0.30`. Only after that gate, set
`r_perp=r_raw*||u_s||_Cs/||r_raw||_Cs`, exactly as parent Section 5.4. The
preflight then uses this rescaled vector and the analytic held covariances
below. It requires the positive scalar
`cos_Cs(-u_s,t_anchor[s])` to lie in `[0.80,0.95]` and the `C_s` residual-norm
ratio above. For correction risk use `r_perp`; for every final alpha use
`-u_s+alpha*r_perp`. Evaluate the
parent `Pref` over that scorer's nine equal-weight held
`domain x mutation` environments using each environment's analytic `C_e`;
require correction `Pref>=0.10` with positive mean target cosine and at least
one nonzero frozen alpha whose final risk direction has `Pref>=0.10`. Every
scorer must pass separately. An equation, sign, norm, or reconstruction mismatch
is `INVALID_IMPLEMENTATION`. A correctly generated scheduled seed that fails
one of these registered numeric geometry gates yields
`CLOSE_INVALID_SIMULATOR_GEOMETRY`; it is not retried, replaced, or used to
tune the generator.

For each scorer, transform its natural draw with `G @ chol(C_s).T`, subtract
the exact column means, and store the empirical `Z'Z/2000`. The simulator
supplies `u_s` directly; it never fits a second mixed-v2 transform or IU model.

### 5.2 Quartet generator

Use the real 900-group semantic/grammar allocation and frozen outer/inner
folds. In each domain/mutation cell, drop the hash-smallest short group and the
hash-smallest certificate group under
`SHA256(namespace || "\0world:" || w || "\0rep:" || r ||
"\0admission\0" || group_id)`, leaving 98 groups and at least 49 per grammar.

Let `sign(A)=-1`, `sign(B)=+1`, and `y=+1` for off-diagonal/error cells and
`y=-1` for diagonal/correct cells. Directions are risk coefficients, so feature
mean shifts use `C_s` times those coefficients:

```text
z[g,s,p,q,k] = b[g,s] + sign(p)*a[g,s] + sign(q)*r[g,s]
             + .5*y*beta_T*C_s@t[d,m,s]
             + beta_N*C_s@n[k,s] + y*beta_I*C_s@h[k,s]
             + epsilon[g,s,p,q,k].
```

Canonical rendering has `n[0,s]=h[0,s]=0`; noncanonical vectors are the three
registered scorer-specific auxiliary rows. Scale covariance draws as
`b~N(0,.40*C_s)`, `a,r~N(0,.15*C_s)`, and
`epsilon~N(0,.45*C_s)` with the Cholesky rule above. Prompt/response main
effects cancel in the reciprocal contrast.

Mean target-token NLL is a sidecar scalar, not one of the 30 coordinates:

```text
ell_source = A_T*y + A_N*render_noise[g,s,k] + nll_epsilon,
render_noise[...,canonical]=0.
```

The same noncanonical render noise is reused across its four reciprocal cells.
PTNI, IU, transforms, moments, and arm selection signatures cannot receive
either NLL array; only the frozen NLL control can.

### 5.3 Labelled evaluation generated only for diagnostics

Held diagnostic environments are the 18 tuples above. Each has exactly 400
rows. Begin with 100 rows in each lexicographic `(Y,Z)` cell
`(0,0),(0,1),(1,0),(1,1)` and apply that environment's registered joint
permutation. `Y=1` means error and `Z` means nuisance, so their finite-sample
covariance is exactly zero. With coefficient directions, generate

```text
z_eval = epsilon + .5*(2Y-1)*gamma_T*C_s@t_world[d,m,s]
         + .5*(2Z-1)*gamma_N*C_s@n_eval[d,m,s],
epsilon ~ N(0,C_s),
n_eval = normalize_Cs((n[1,s]+n[2,s]+n[3,s])/sqrt(3)
                      + .25*v_nuisance_scorer[s]).
```

The analytic population feature covariance used by `Pref` is

```text
C_e = C_s + .25*gamma_T^2*(C_s@t_world)(C_s@t_world)'
          + .25*gamma_N^2*(C_s@n_eval)(C_s@n_eval)'.
```

No empirical held or label-conditioned covariance is substituted. The planted
target and nuisance risk coefficients gated by `cos_C` are exactly `t_world`
and `n_eval`. AUROC uses risk (`-confidence`) inside each environment and then
equal-macros the 18 values.

The held NLL diagnostic is independently

```text
ell_eval = A_T*(2Y-1) + A_N*(2Z-1) + eval_nll_epsilon.
```

NLL risk is `ell`; NLL confidence is `-ell`. Source and held NLL arrays remain
outside every PTNI/IU signature. Diagnostic truth and directions live in a
separate type unavailable to all fit/selection APIs.

### 5.4 Eight worlds

| world | beta_T | beta_N | beta_I | gamma_T | gamma_N | NLL `A_T/A_N` | target coefficient |
|---|---:|---:|---:|---:|---:|---:|---|
| 1 target only | 1.0 | 0.0 | 0.0 | 1.0 | 0.0 | 0.75 / 0.0 | `t_common[s]` |
| 2 nuisance only | 0.0 | 1.0 | 0.0 | 1.0 | 1.0 | 0.0 / 1.0 | no quartet target; IU remains valid on evaluation |
| 3 equal target+nuisance | 1.0 | 1.0 | 0.20 | 1.0 | 1.0 | 0.75 / 0.75 | `t_common[s]` |
| 4 twice-strong nuisance | 1.0 | 2.0 | 0.40 | 1.0 | 2.0 | 0.75 / 1.5 | `t_common[s]` |
| 5 family-specific target | 1.0 | 1.0 | 0.25 | 1.0 | 1.0 | 0.75 / 0.75 | `t_family[d,m,s]` |
| 6 null/single Gaussian | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 / 0.0 | no target or nuisance signal |
| 7 forced-mismatch NLL | 1.0 | 1.0 | 0.20 | 1.0 | 1.0 | 2.0 / 0.25 | `t_common[s]`; NLL intentionally strongest |
| 8 coherent/no-NLL | 1.0 | 1.5 | 0.30 | 1.0 | 1.5 | 0.0 / 0.25 | `t8[d,m,s]`, coefficient 24 exactly zero |

Worlds 1,3,4,5,7,8 are target/direction-gated worlds; 2 and 6 are fallback
worlds.

### 5.5 Main schedule, controls, and bootstrap

There are exactly 100 repetitions per world, `r=0..99`. Development may use
only `a6-s1-dev-v1`; the sealed schedule is materialized and hashed only in the
reviewed S1 boundary before any result process starts.

Every repetition runs PTNI and matched controls 1--9 from parent Section 6
through their exact outer, inner, final-cross-fit, and final-refit contracts.
Control 10 is an existing retrospective supervised ceiling and is inapplicable
to synthetic coordinates; it is neither run nor replaced. Learned controls
3 and 5--9 must be usable, finite, and deterministic under their own selection
contracts; affine controls reconstruct within `1e-10`, while parameter-free
IU, Family-NRM, and NLL must produce finite scores. Their diagnostic performance
is reported but does not gate PTNI unless a parent gate explicitly names that
control.

In each target world, final and deployed-correction risk directions must pass
the parent's signed `cos_C`/`Pref>=.05` test in at least 90/100 repetitions.
World 1 is nonvacuous because alpha zero has no correction and fails. Worlds 2
and 6 select exact alpha zero in at least 95/100 repetitions. Any unusable main
repetition closes S1; an unexpected exception is `INVALID_IMPLEMENTATION`, not
scientific nonconvergence.

For each target world, bootstrap the 100 paired equal-environment
candidate-minus-IU AUROC differences. Seed PCG64 with the first 64 SHA-256 bits
of `"a6-s1-main-bootstrap-v1\0world:" || w ||
"\0candidate-minus-iu"`, call
`Generator.integers(0,100,size=(20000,100),dtype=int64)` exactly once, and
average the indexed values in each row. The lower endpoint is ascending one-based
order statistic 500 (`method="lower"` at .025) and must be at least zero. Items
are not re-bootstrapped inside this repetition-level gate.

Every inner/final one-SE bootstrap reuses one multiplicity per source group
across scorers/renderings and all arms. Replace all informal placeholders by
one canonical `fit_context_id`. Its allowed bytes are exactly:

```text
main:<procedure>:outer:<o>
main:<procedure>:final
lo-target:<held_mutation>:<procedure>:outer:<o>
lo-target:<held_mutation>:<procedure>:final
lo-render:<held_render>:<procedure>:outer:<o>
lo-render:<held_render>:<procedure>:final
null:<null_family>:draw:<jjj>:ptni:outer:<o>
null:<null_family>:draw:<jjj>:ptni:final
nuisance-as-target:ptni:outer:<o>
nuisance-as-target:ptni:final
stress:<arm>:index:<jjj>:<procedure>:outer:<o>
stress:<arm>:index:<jjj>:<procedure>:final
```

`procedure` is one of `ptni,target-only,diagonal,logistic,unreciprocated,
no-interaction`; `null_family` is one of
`sign,response-block,matched-group`; `jjj` is zero-padded `000..199` for nulls
and `000..099` for stresses. Held names and arm names are their literal
registered identifiers elsewhere in this contract. IU, Family-NRM, NLL, and
single feature do not use a one-SE bootstrap under their parent contracts. `o`
is one ASCII digit `0..4`; `final` is literal. No other context string is legal.
The PCG64 seed is the first 64 bits of

```text
SHA256("a6-s1-selection-bootstrap-v1\0" || namespace || "\0world:" || w ||
       "\0rep:" || r || "\0context:" || fit_context_id).
```

Initialize one PCG64 generator from the fit-context seed, traverse
UTF-8-sorted parent bootstrap strata, sort each stratum's `n` group IDs, and
make the next call `integers(0,n,size=(20000,n))`; convert every row with
`bincount(...,minlength=n)` and concatenate stratum multiplicities in the fixed
stratum order. Each fit context uses a fresh generator from its own hash above,
not a stream shared with another context.
The same 20,000 stored multiplicity vectors evaluate all arms, but only the
pre-tie best arm's SE defines its one-SE set.

### 5.6 Sealed leave-family-out implementation suite

All 100 repetitions of world 5 run three leave-one-target-mutation refits. All
100 repetitions of worlds 4 and 8 run three leave-one-nuisance-render refits.
Held target groups are absent from moments, feasibility, selection, and every
learned control; a held render is absent from every fitted `tau`, `nu`, `iota`,
feasibility quantity, and control. Natural anchor rows remain label-free and
permitted exactly as in the parent.

For a held target mutation, its 48 equal cells are
`scorer(2) x domain(3) x grammar(2) x render(4)`; mutation is fixed. For a held
nuisance render, its 36 equal cells are
`scorer(2) x domain(3) x mutation(3) x grammar(2)`; render is fixed. `J` is the
equal mean of those cell values. The LO bootstrap samples one multiplicity per
group within `domain x grammar` for a fixed held mutation and within
`domain x mutation x grammar` for a fixed held render, reusing it across every
scorer and remaining/evaluated rendering.

For a repetition to pass its LO suite, all three held-family PTNI fits must be
usable, reconstruct within `1e-10`, have `J>=0.60` over the corresponding
48/36-cell macro and `>=0.55` in every one of those cells, and have positive
deployed correction margin in every cell. At least 90/100 repetitions must pass
all three held target families in world 5, and separately all three held
nuisance families in each of worlds 4 and 8. Replacing all held-family fit-side
arrays by a fixed finite poison matrix must leave moments, feasible arms,
selected arm, coefficients, and learned-control artifacts byte-identical; only
held evaluation scores may change. Controls 1--9 execute under their registered
LO contracts and satisfy finite/reconstruction invariants, but their LO
performance is diagnostic.

### 5.7 Null, falsification, and robustness suites

The whole-quartet conditional sign null and placebos 2/3 run separately on
world 4 repetition 0 and world 8 repetition 0. Each reference population must
independently pass every parent threshold, alpha-zero count, distinct schedule
hash, distinct outer-held assignment hash, and unique-realized-J report; pooling
or rescue is forbidden. The sign null alone uses S1-specific high-level seed
`first64(SHA256("a6-s1-sign-null-v1\0world:" || w ||
"\0rep:0\0draw:" || j))`. For each draw, UTF-8-sort all retained source group
IDs once, call
`Generator(PCG64(seed)).integers(0,2,size=n_groups,dtype=int64)` exactly once,
and map zero to `-1`, one to `+1`. That global group-to-sign table is reused in
every inner/outer/final appearance, scorer, and rendering. Placebos 2/3 have
one authority: S1 reads the exact
seeds, algorithms, edge/caliper inputs, and full-population schedule hashes from
the verified S0b completion. After S1's frozen symmetric group drop, it reruns
those same seeds/algorithms on the retained simulator partition exactly as the
parent admission rule requires; it does not derive a second world-specific
seed family. Before any retained refit, every frozen null stratum must still
contain at least four groups in every required partition and all 200 Control-3
perfect matchings must exist. No stratum is remerged and the S0b caliper is
unchanged; failure closes S1. The retained assignment hashes are stored separately for each
reference world because their admitted groups may differ. These are separate
200-refit suites, not refits inside all 800 main repetitions.

The deterministic nuisance-as-target control runs on world 4 repetition 1. It
must first pass every activation gate in parent Section 7; activation failure is
`CLOSE_UNINFORMATIVE_NUISANCE_CONTROL`. After activation, at least one member
of the finite, nonrecursive real-semantic gate set in that section must fail;
otherwise close `CLOSE_NUISANCE_CONFOUNDING`.

Control-4 bootstrap gate names are exactly
`activation-vs-iu`, `semantic-noncopy-j`, `semantic-vs-iu`, and
`semantic-vs-composite`. Each seed is
`first64(SHA256("a6-s1-control4-bootstrap-v1\0" || gate_name))`. Initialize one
PCG64 per gate; traverse UTF-8-sorted `domain x mutation x grammar` strata and
make the exact `(20000,n)` integer/bincount calls from Section 5.5, carrying
each group multiplicity across scorers/renderings. The noncopy gate filters the
same draws to its preregistered groups; there is no refit or alternate seed.

Robustness uses base index `j=0..99`, with pristine main record `(w,r)=(4,j)`
for even `j` and `(w,r)=(8,j)` for odd `j`. Every stress checkpoint stores and
verifies that main record's boundary/record SHA-256 before loading it. Every arm below starts from the same pristine unaugmented base for that
index and reruns complete candidate/control selection independently. Stresses
are never composed:

```text
permutation; exact_duplicate; near_duplicate_rho_0.999;
held_delete_1; held_delete_2; held_delete_3;
source_mcar_0.05; source_mcar_0.15; source_mcar_0.30;
source_domain_block; source_all_missing_coordinate; deterministic_rerun.
```

An arm/subkey seed is the first 64 bits of
`SHA256("a6-s1-stress-v1\0index:" || j || "\0arm:" || arm ||
"\0" || subkey)`. Every name permutation starts from the canonical 30-name
tuple and uses one PCG64 generator with the named subkey, descending
Fisher-Yates `i=29..1`, and exactly one
`integers(0,i+1,dtype=int64)` call per `i`. Arm `permutation` uses subkey
`name_permutation`; each held-deletion arm uses `held_deletion_names` and drops
the first `k` names; `source_domain_block` uses `domain_block_names` and its
first three names. Duplicate source
coordinate is canonical name index `j mod 30`; aliases are stress-only
`<name>__a6_s1_duplicate` and cannot enter a real roster. Exact duplication is
a byte copy in every source/natural/held matrix. Before any fold slice, near
duplication constructs one alias over each full scorer/array family in this
fixed order: qwen4 source tensor, qwen8 source tensor, qwen4 natural matrix,
qwen8 natural matrix, qwen4 nine-environment evaluation matrix, qwen8
nine-environment evaluation matrix. Their literal matrix IDs are
`source_qwen4,source_qwen8,natural_qwen4,natural_qwen8,eval_qwen4,eval_qwen8`.
Flatten only the row axes. For each family, seed PCG64 with subkey
`near_noise:<matrix_id>` and make one
`standard_normal(n_rows,dtype=float64)` call. For source column `x`, let
`xc=x-mean(x)` and draw/center that full-family noise vector;
project noise off `xc`, scale its norm to `||xc||`, and set
`alias=mean(x)+.999*xc+sqrt(1-.999^2)*noise`. Require full-family sample
correlation `.999` within `1e-12` and equal means/centered norms within
`1e-12`. Folds are slices of this one construction; no training-, validation-,
or held-specific alias transform exists.

The stress anchor mapping is exact. A name permutation permutes `u` and every
coefficient before the adapter restores canonical order. An exact duplicate is
collapsed by the registered mean/contrast quotient before **all** candidate and
control fits; expansion splits the original IU and correction coefficient
equally across original and alias so IU and deployed scores are identical. A
near alias is not quotiented: it is inserted immediately after the original in
the stress-only immutable order, receives IU coefficient zero, and inherits the
original Family-NRM family. Single-feature exact ties prefer the original;
ridge logistic includes the near alias at that fixed adjacent position. NLL is
unchanged.

For `held_delete_k`, let `D` be all canonical names except the first `k` names
of the registered `held_deletion_names` permutation. Deletion touches only the
separate 18-by-400 diagnostic evaluation matrices. It touches no source inner
training, inner validation, outer training, outer held/OOF row, and no
natural-anchor row. Rerun the complete candidate/control
selection on the pristine fit inputs with identical seeds; for every arm and
control, all source moments, quotient, directions, feasibility records, OOF
selection scores, selected arm/control identity, and final source artifact must
be byte-identical to the pristine run. This explicit rerun proves that held
deletion cannot adapt selection.

Only after selection, bind the selected artifact to the reduced held target.
For scorer `s`, `C` is that pristine scorer's empirical `Z'Z/2000` natural
covariance from Section 5.1. The reduced label-free IU anchor is
`u_D=solve(C[D,D],C[D,:]@u)` by Cholesky. Restrict the selected source direction
to `r0_D`, then apply the complete parent projection and normalization:

```text
r_raw_D = r0_D - u_D*(u_D'C[D,D]r0_D)/(u_D'C[D,D]u_D)
r_perp_D = r_raw_D * ||u_D||_C[D,D] / ||r_raw_D||_C[D,D].
```

The parent zero-evidence thresholds apply before the second line and return
exact reduced IU. A nonfinite solve/norm is an unusable scheduled stress.
Candidate-style controls use the same restriction, projection, and rescaling;
Family-NRM drops absent coordinates and renormalizes its frozen within-family
weights, the availability-aware single control uses its registered NLL fallback
when needed, and logistic becomes `UNSCORABLE_TARGET_ROSTER` without rescuing
the candidate. For each MCAR arm, initialize PCG64 with subkey `mcar_mask` and
make one `Generator.random((900,2,2,2,4,30),dtype=float64)` call in canonical
group/scorer/prompt/response/render/feature order; mask entries below the arm's
literal rate, then restrict to admitted groups. No natural/evaluation entry is
MCAR-masked. Each inner-training split computes its own medians and
applies them unchanged to its validation/held rows. The domain block chooses
domain `j mod 3` and the first three names of `domain_block_names`, and masks those
coordinates in all source rows of that domain. The all-missing arm masks name
`j mod 30` in every source intervention row and must return exact IU. The
deterministic arm reruns pristine input.

Each arm has 100 scheduled results and is compared only with its pristine base.
Every numerical gate is exactly parent Section 7's arm-specific threshold;
every scheduled unusable arm closes. Exact/near/deletion/MCAR/block/fallback
outcomes are never combined into one score. A simulator-only
`StressRosterAdapter` accepts arbitrary unique permutations and the one
registered alias, canonicalizes by immutable names, and records the inverse
map. Real A6 validators continue to reject aliases and noncanonical rosters.
Unit tests exhaust every permutation for dimensions 1--7 and prove that the
stress adapter cannot be invoked by a real-data loader.

### 5.8 Gate applicability

| gate | worlds/suite |
|---|---|
| final and correction target preference | 1,3,4,5,7,8 |
| candidate-minus-IU AUROC lower endpoint | 1,3,4,5,7,8 |
| exact-IU alpha-zero rate | 2,6 |
| PTNI and controls 1--9 execute under outer/inner/final contracts | all main repetitions; performance gates only where named |
| LO target-family refits | world 5, all 100 repetitions |
| LO nuisance-render refits | worlds 4 and 8, all 100 repetitions |
| sign null and placebos 2/3 | separately: world 4 rep 0 and world 8 rep 0 |
| nuisance-as-target activation/confounding | world 4 rep 1 |
| separated robustness arms | 100 alternating world-4/world-8 bases per arm |

No sign null, nuisance-control activation, LO success, or candidate-improvement
gate applies to worlds 2/6; their scientific gate is exact IU fallback. The
matched controls use mechanical simulator contrasts only. No retrospective or
natural benchmark target, supervised atomic head, response telemetry, or
PopQA field enters S1.

## 6. Append-only stage boundaries

S0a, S0b, and S1 are separate directories and separate reviewed boundaries.
Each boundary hashes this contract, the parent protocol, the complete loaded
local Python import closure, tests, runtime versions, upstream model/dataset
revisions, and prior-stage completion artifact. Preparation requires a new
empty directory and exclusive writes. Status/schema/hash verification occurs
before any model or pickle import.

Long runs write one exclusive canonical-JSON checkpoint per scheduled unit.
Resume verifies every checkpoint against the unchanged boundary and computes
only missing units. Aggregates reproduce exactly from checkpoints; completion
binds the boundary and aggregate hashes. A verifier recomputes the schedule,
summary, and every hash instead of trusting a `gate_pass` field. No later stage
may open unless that recomputation matches byte for byte and the registered
verdict is PASS.

## 7. Freeze order

1. independent no-edit review of this contract;
2. implement S0a plus fail-closed tests;
3. independently review and freeze the S0a source/runtime boundary;
4. execute S0a only;
5. if S0a passes, implement/review/freeze and run S0b;
6. if S0b passes, implement/review/freeze S1 without opening a sealed seed;
7. execute S1; any failure closes A6 and advances to A7.

No response telemetry, natural response, A6 feature row, correctness sidecar,
retrospective target, or PopQA field is accessed in steps 1--4.
