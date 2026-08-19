# Automatic group-free IU Phase A6 — authenticated tokenizer restoration v1

**Status:** `DRAFT_BEFORE_REVIEW — NO RESTORED CACHE OR S0a BOUNDARY EXISTS`

This is a narrow, pre-data execution/provenance addendum to
`AUTOMATIC_GROUP_FREE_IU_PHASE_A6_S0_S1_EXECUTION_V1.md`. It does not change
the A6 target, populations, prompts, tokenizer revisions, estimator, gates,
random schedules, or interpretation. It defines how the already frozen three
tokenizer revisions may be transported from the project's archival Google
Drive cache and an exact official Hugging Face commit into immutable local
regular-file inputs without treating a cache-shaped directory as proof of
authenticity.

No A6 tokenizer restoration, S0a preparation, telemetry, natural response,
correctness sidecar, PopQA field, or sealed simulator seed may be opened under
this draft. The implementation and its source/runtime boundary require a
fresh independent no-edit review before any canonical restoration is created.

## 1. Why this addendum is required

`huggingface_hub.scan_cache_dir` establishes that a directory is internally
shaped like a Hugging Face cache; it does not authenticate the directory name,
repository name, revision directory, link payload, or blob origin. A fabricated
`models--repo/snapshots/<revision>` tree can therefore report the frozen repo
and revision while containing arbitrary bytes. Cache layout is retained only
as transport evidence and never as an authority assertion.

The three frozen identities remain exactly:

| role | repository | revision |
|---|---|---|
| Qwen source scorer 1 | `Qwen/Qwen3-4B` | `1cfa9a7208912126459214e8b04321603b3df60c` |
| Qwen source scorer 2 | `Qwen/Qwen3-8B` | `b968826d9c46dd6066d109eabc6255188de91218` |
| held Llama scorer | `meta-llama/Llama-3.1-8B-Instruct` | `0e9e39f249a16976918f6564b8830bc894c89659` |

The allowed byte transports are deliberately asymmetric, but the official
tree authority is identical for all three roles:

1. The one exact official Hugging Face revision endpoint in Section 3 supplies
   the complete path/object/size tree for Qwen3-4B, Qwen3-8B, and Llama. The
   reviewed fixed projections in this addendum are the expected values; the
   live response must reproduce them before any cache object is trusted.
2. Qwen3-8B and most Llama selected bytes must come from the exact
   standard-cache archival prefixes in Section 3. Every snapshot entry is
   authenticated as an archived raw Drive object, its one-hop target is
   validated, and every selected target blob is content-verified against the
   independent official tree record.
3. Qwen3-4B is not present as a standard archival cache. The exact official
   revision tree supplies its Git/LFS object identities.
   The five byte-identical tokenizer files may be transported from the already
   authenticated Qwen3-8B blobs only after their Qwen3-4B official object
   identities also match. Its unique `config.json` must be obtained at the
   literal official commit and match both the frozen byte SHA-256 and Git-blob
   SHA-1 below.

No flat cache metadata, mutable `main`, user-written revision directory,
filename alone, `scan_cache_dir` result alone, or locally reconstructed
symlink graph is sufficient authority. Failure to authenticate any one of the
three complete selected trees is `BLOCKED_TOKENIZER_ACCESS`; proxies and
partial canonical outputs are forbidden.

## 2. Canonical bytes and cryptographic checks

Canonical JSON in this addendum means UTF-8, sorted keys, separators
`(',',':')`, `ensure_ascii=False`, `allow_nan=False`, and one terminal newline.
Paths are NFC-normal POSIX relative paths. SHA-256 is lowercase hexadecimal
over exact file bytes.

For an ordinary non-LFS Git object with payload `B`, the verifier computes

```text
git_blob_sha1 = SHA1(b"blob " + ASCII_DECIMAL(len(B)) + b"\0" + B)
```

and requires exact equality to the official `blobId`. For an LFS file, the
official `blobId` authenticates the exact Git-LFS pointer bytes, not the
expanded payload. The verifier constructs and authenticates the pointer

```text
version https://git-lfs.github.com/spec/v1
oid sha256:<official lfs.sha256>
size <official lfs.size>
```

There are exactly three LF bytes, including one after the size line, and no CR
or blank line. The verifier requires its Git-blob SHA-1 to equal the official
`blobId`, and separately requires the expanded payload size/SHA-256 to equal
the official `lfs.size`/`lfs.sha256`. For a content-addressed 64-hex cache
blob, the verifier also requires
`SHA256(B)` to equal that filename when the table marks the filename as a
SHA-256 address. A Git-style hash computed over expanded LFS payload bytes is
only diagnostic and is never labelled an official Git object. An Xet identity,
if recorded, is also diagnostic unless present in the frozen official revision
projection and never substitutes for `lfs.sha256`.

Raw standard-cache snapshot entries must be regular archival Drive objects
whose exact bytes are the ASCII string

```text
../../blobs/<registered_blob_name>
```

with no newline. The normalized target must remain inside the same registered
repository cache root, be exactly one hop, and resolve to one regular selected
blob. Absolute targets, empty targets, multi-hop links, `.`/`..` after the
literal two parent components, duplicate path mappings within one role, path traversal,
case-colliding paths, dangling targets, and extra selected paths close the
restoration. Cross-role reuse is legal only for the five aliases in Section 5.

## 3. Exact archival origins and selected file records

### 3.0 One official revision-tree authority

For every role, the only official-tree request is an anonymous HTTPS `GET` to

```text
https://huggingface.co/api/models/<owner>/<model>/revision/<revision>?blobs=true
```

where owner/model/revision are the literal values in Section 1, percent
encoding is forbidden because every character is already URL-safe, the query
contains only `blobs=true`, `Accept: application/json` and
`User-Agent: a6-tokenizer-restore-v1` are the only explicit request headers,
and no cookie, bearer token, redirect, proxy rewrite, mutable ref, or second
endpoint is allowed. TLS certificate and hostname verification use the pinned
Python/OpenSSL/certifi runtime defaults. Require HTTP 200, final URL byte-equal
to the request URL, JSON content type, no `Link`, `next`, pagination, or
truncation indicator, and an EOF-complete JSON object. Network failure,
redirect, authentication challenge, schema drift, or pagination is
`BLOCKED_TOKENIZER_ACCESS`.

The implementation does **not** rely on the website's “verified” badge or
claim a separately verified commit signature. Authority is the official
`huggingface.co` TLS endpoint at the exact immutable revision plus the
pre-data, independently reviewed fixed tree projections in Appendix A.

The parser rejects duplicate JSON keys at every depth. It projects only:

```text
{
  "id": string,
  "sha": 40-lowercase-hex string,
  "gated": false OR "manual",
  "siblings": [
    {"rfilename": string, "blobId": 40-lowercase-hex string,
     "size": nonnegative integer,
     "lfs": absent OR
       {"sha256": 64-lowercase-hex string,
        "size": nonnegative integer, "pointerSize": positive integer}}
  ]
}
```

Unknown top-level response fields are ignored, but any missing/extra field
inside a projected sibling or `lfs` object closes. `id` and `sha` must equal
Section 1; `gated` must be `false,false,"manual"` for Qwen3-4B, Qwen3-8B,
and Llama respectively. Siblings are unique after NFC POSIX-path normalization,
sorted by UTF-8 path bytes, and must equal Appendix A field-for-field. The raw response
bytes/hash and canonical projection bytes/hash are both stored; replay gates
the canonical projection, so dynamic nonprojected popularity metadata cannot
change the tree.

The parent allowlist is expanded against that official projection. Its exact
intersection is six Qwen3-4B paths, six Qwen3-8B paths, and six Llama paths,
including `original/tokenizer.model`. A selected-path count or value different
from those reviewed intersections closes before Drive access.

### 3.0.1 Exact Drive inventory operation

The Google Drive remote is the already configured read-only byte transport
`gdrive:`. For each registered prefix, invoke exactly this argument vector with
`LC_ALL=C`, `LANG=C`, `TZ=UTC`:

```text
[rclone,lsjson,<prefix>,--recursive,--files-only,--hash,--metadata]
```

`rclone` is the absolute binary path recorded in the boundary; its bytes,
SHA-256, version output, and redacted-config fingerprint are frozen. Parse one
EOF-complete JSON array with duplicate-key rejection. Canonicalize each row to
exactly `Path,Name,Size,MimeType,ModTime,IsDir,ID,Hashes,Metadata`, requiring
the source fields to exist with their native JSON types, `IsDir=false`, and
`Name` equal the last NFC-normalized POSIX `Path` component. For these Google
Drive sources, `Hashes` must contain exactly lowercase `md5,sha1,sha256`; a
missing or extra hash closes. `Metadata` requires source keys exactly
`btime,content-type,mtime` among the retained subset and maps them to canonical
artifact keys `btime,content_type,mtime`; access/UI fields such as `owner,starred,
viewed-by-me,copy-requires-writer-permission,writers-can-share` and unknown
metadata are deliberately excluded from the committed projection. Paths must be
relative, unique under both exact bytes and Unicode casefold, and traversal
free. Sort by UTF-8 `Path` bytes and canonical-JSON encode. Raw stdout/stderr,
exit status and canonical projection hashes are stored; raw output order is not
an authority.

Store raw stdout/stderr SHA-256 and exit status but not raw metadata bytes that
may contain account identifiers. Run this inventory immediately before and immediately after every copy batch.
For each copied object require unchanged Path/ID/Size/ModTime/Hashes/Metadata
and re-read the exact bytes into a new exclusive staging file. A changed object
or source tree is `BLOCKED_TOKENIZER_ACCESS`, never a resumable success. Resume
repeats both the official projection and initial/post Drive inventories from
zero before trusting any checkpointed field.

### 3.1 Qwen3-8B standard-cache archive

```text
repository cache root:
  gdrive:hf_cache/hub/models--Qwen--Qwen3-8B
snapshot:
  snapshots/b968826d9c46dd6066d109eabc6255188de91218
reference:
  refs/main
reference exact bytes:
  b968826d9c46dd6066d109eabc6255188de91218
```

| selected path | raw link SHA-256 | registered blob | blob bytes | blob SHA-256 | Git blob / official object |
|---|---|---|---:|---|---|
| `config.json` | `734ad299d853ff19f31e39740a62585f75c01befd179031693f5e2e45408dd9f` | `d46195ac87f837ad233d02b2f80f148bf7c005e0` | 728 | `f7c4eadfbbf522470667b797a3c89be2524832d2d599797248dc304fff447c30` | Git `d46195ac87f837ad233d02b2f80f148bf7c005e0` |
| `generation_config.json` | `faefdd355ec1348af8f0239d2a74f34ed81544633daa5f6b18cd02067f98ddf0` | `20a8a9156fc8c3f25295ca067f61fdf120d517c5` | 239 | `2325da0f15bb848e018c5ae071b7943332e9f871d6b60e2ed22ca97d4cb993d2` | Git `20a8a9156fc8c3f25295ca067f61fdf120d517c5` |
| `merges.txt` | `7be9b77e01bb290341d63588e722aee0aaf0f56da05f3b3cc969be28a36dae0e` | `31349551d90c7606f325fe0f11bbb8bd5fa0d7c7` | 1,671,853 | `8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5` | Git `31349551d90c7606f325fe0f11bbb8bd5fa0d7c7` |
| `tokenizer.json` | `a9c99268f9ee0ab27f82ccb1fc2a9e1d4d6a2ae3ab64a4574184c0ebf87eef92` | `aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4` | 11,422,654 | `aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4` | LFS pointer Git `cd71f61a15a522601badb3dc960d800d9cb3766c`; pointer SHA-256 `9ec507f98e2a5da7ea342682b833d7283b4f0d7661692075ecb048aa083ee203` |
| `tokenizer_config.json` | `e0becfc9eab03ae8a31243d7c4bff0d19f89adcc58c12160b05ccad50ac7dfee` | `417d038a63fa3de29cfde265caedae14d1a58d92` | 9,732 | `d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101` | Git `417d038a63fa3de29cfde265caedae14d1a58d92` |
| `vocab.json` | `6760a07e4510dfd00b607139ade206f596e6fb00454193844f8de51f162a1381` | `4783fe10ac3adce15ac8f358ef5462739852c569` | 2,776,833 | `ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910` | Git `4783fe10ac3adce15ac8f358ef5462739852c569` |

### 3.2 Llama-3.1-8B-Instruct standard-cache archive

```text
repository cache root:
  gdrive:hf_cache/hub/models--meta-llama--Llama-3.1-8B-Instruct
snapshot:
  snapshots/0e9e39f249a16976918f6564b8830bc894c89659
reference:
  refs/main
reference exact bytes:
  0e9e39f249a16976918f6564b8830bc894c89659
```

| selected path | raw link SHA-256 | registered blob | blob bytes | blob SHA-256 | Git blob |
|---|---|---|---:|---|---|
| `config.json` | `194fd3de800e667c240f4a8b39f64765b56db35431b200eebcb84f3c9c9f5fd3` | `0bb6fd75b3ad2fe988565929f329945262c2814e` | 855 | `29e4c210b0d6ac178b16b2a255a568bdb23b581e50ca1ef6a6d071dd85704e6e` | `0bb6fd75b3ad2fe988565929f329945262c2814e` |
| `generation_config.json` | `cec42b4b9bc3c821df0b7fb6ea07d0ae12fcc6cf2e423278fef2ef608cec133f` | `cc7276afd599de091142c6ed3005faf8a74aa257` | 184 | `189fb0c0d7fd8a527db217c0a60a0e013f0394cd8800f9697a666a9e75e5f7fd` | `cc7276afd599de091142c6ed3005faf8a74aa257` |
| `original/tokenizer.model` | no standard-snapshot link; exact flat source and metadata below | `82e9d31979e92ab929cd544440f129d9ecd797b69e327f80f17e1c50d5551b55` | 2,183,982 | `82e9d31979e92ab929cd544440f129d9ecd797b69e327f80f17e1c50d5551b55` | LFS pointer Git `a097ce5a06fce0fa3d685a8cfb175cef243dfde9` |
| `special_tokens_map.json` | `aa5f7ff81b4a75b4d8d03d6eaa55050cd54c310b9dab68c6e35744fc0de09d08` | `02ee80b6196926a5ad790a004d9efd6ab1ba6542` | 296 | `6f38c73729248f6c127296386e3cdde96e254636cc58b4169d3fd32328d9a8ec` | `02ee80b6196926a5ad790a004d9efd6ab1ba6542` |
| `tokenizer.json` | `8faa6731b8e0812cafa463eca756005bbd5d46099b8f05a0f61634c74e72b1ae` | `5cc5f00a5b203e90a27a3bd60d1ec393b07971e8` | 9,085,657 | `79e3e522635f3171300913bb421464a87de6222182a0570b9b2ccba2a964b2b4` | `5cc5f00a5b203e90a27a3bd60d1ec393b07971e8` |
| `tokenizer_config.json` | `5bd47f288d95ba0394524b56f822202ed485272e5d1f22bba0b3f3255fa0a7cf` | `db88166e2bc4c799fd5d1ae643b75e84d03ee70e` | 55,351 | `177c7b61e616fecb84c17ce0591acb92c6c4d60e9ac5ababfb940ff23bbcd424` | `db88166e2bc4c799fd5d1ae643b75e84d03ee70e` |

The flat directory is never repository/revision/tree authority. Its sole
allowed use is transport of the officially registered
`original/tokenizer.model` at

```text
gdrive:hf_cache_flat/meta-llama__Llama-3.1-8B-Instruct/original/tokenizer.model
```

whose expanded payload must match the official LFS SHA-256/size above. Its
archived metadata path is exactly
`.cache/huggingface/download/original/tokenizer.model.metadata`, raw size 124,
SHA-256
`cf5e9fb4186f32441b10e0c5c4a8fb3126cc30ebfab70cddb66f3e9a320771ec`,
and exact three-line values revision `0e9e39f249a16976918f6564b8830bc894c89659`,
etag `82e9d31979e92ab929cd544440f129d9ecd797b69e327f80f17e1c50d5551b55`,
timestamp `1778745152.481112`. Metadata is corroboration only; official tree plus
payload SHA-256 are authority. No other flat-cache file is permitted.

### 3.3 Qwen3-4B official-revision reconstruction

The authority is the exact official revision
[`1cfa9a7208912126459214e8b04321603b3df60c`](https://huggingface.co/Qwen/Qwen3-4B/commit/1cfa9a7208912126459214e8b04321603b3df60c).
It is retrieved and projected only by Section 3.0; there is no alternate Git,
HTML, signed-badge, Xet, or “equivalent” evidence path. The raw official API
response bytes/hash, URL, retrieval timestamp, TLS runtime, and canonical
Appendix-A projection are frozen before materialization. A mutable `main`
response is inadmissible.

The unique config payload uses one additional anonymous no-redirect HTTPS GET,
with the same headers/TLS/error rules, to the exact literal URL

```text
https://huggingface.co/Qwen/Qwen3-4B/raw/1cfa9a7208912126459214e8b04321603b3df60c/config.json
```

Require HTTP 200, content type exactly `text/plain; charset=utf-8`, content
length 726, `ETag="e49eccdc32f36da9c09cfa0e737084f9e0105e5e"`,
`X-Repo-Commit=1cfa9a7208912126459214e8b04321603b3df60c`, exactly 726
response bytes, the table SHA-256, and Git-blob SHA-1 `e49ecc...`. The raw
response bytes/hash and projected stable headers are stored. No `resolve`, CDN,
mutable, browser-HTML, hand-rendered JSON, or second acquisition path is
allowed.

| selected path | byte source | bytes | required SHA-256 | required official object |
|---|---|---:|---|---|
| `config.json` | exact official 4B commit | 726 | `8ba006f74fecfaaeb392872a60f4a480e7ec9860153d2e1b769ec81f9a147f8a` | Git `e49eccdc32f36da9c09cfa0e737084f9e0105e5e` |
| `generation_config.json` | authenticated Qwen3-8B blob `20a8...` | 239 | `2325da0f15bb848e018c5ae071b7943332e9f871d6b60e2ed22ca97d4cb993d2` | Git `20a8a9156fc8c3f25295ca067f61fdf120d517c5` |
| `merges.txt` | authenticated Qwen3-8B blob `3134...` | 1,671,853 | `8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5` | Git `31349551d90c7606f325fe0f11bbb8bd5fa0d7c7` |
| `tokenizer.json` | authenticated Qwen3-8B blob `aeb1...` | 11,422,654 | `aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4` | LFS pointer Git `cd71f61a15a522601badb3dc960d800d9cb3766c`; pointer SHA-256 `9ec507f98e2a5da7ea342682b833d7283b4f0d7661692075ecb048aa083ee203` |
| `tokenizer_config.json` | authenticated Qwen3-8B blob `417d...` | 9,732 | `d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101` | Git `417d038a63fa3de29cfde265caedae14d1a58d92` |
| `vocab.json` | authenticated Qwen3-8B blob `4783...` | 2,776,833 | `ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910` | Git `4783fe10ac3adce15ac8f358ef5462739852c569` |

The official exact-revision tree must prove that this is the complete
intersection of the parent-contract tokenizer allowlist with the Qwen3-4B
commit. A path added to or missing from that intersection closes restoration.
The five reused payloads are copied only after both the 8B archival-source
checks and the independent 4B official-object checks pass.

## 4. Complete source-tree and selected-tree evidence

For each standard-cache source, provenance contains two inventories:

1. `source_tree_inventory`: every raw Drive object beneath `refs/`, the exact
   frozen snapshot directory, and `blobs/`, with path, object ID, size,
   timestamp, and available remote hashes. Snapshot paths outside the exact
   frozen revision are recorded but never selected.
2. `selected_graph_inventory`: the exact allowlist intersection, raw snapshot
   link bytes/hash, normalized one-hop target, selected blob object record,
   blob size/SHA-256, official Git/LFS checks, and output file record.

Nonselected model-weight blobs are inventory evidence, not tokenizer payloads:
their Drive object identities, sizes, remote hashes, and snapshot link targets
are stored, but their multi-gigabyte bytes need not be copied into the tokenizer
boundary. Every selected blob is downloaded and byte-verified. This distinction
is explicit so “complete repository inventory” cannot be misreported as
“all model weight bytes were rehashed.”

The exact allowlist is the parent Section 1 literal glob expansion. Expansion
is performed against the authenticated exact-revision tree, not against files
that happened to download. The result must equal the selected rows in Section
3 for each repository. Tree completeness failures hard-close; no implementation
may infer an absent optional path merely because local loading succeeds.

## 5. Atomic acquisition and materialization

The canonical final root is exactly
`results/automatic_group_free_phase_a6_tokenizer_restore_v1`; its only staging
root is the sibling
`results/.automatic_group_free_phase_a6_tokenizer_restore_v1.staging`. The lock
is outside both at
`results/.automatic_group_free_phase_a6_tokenizer_restore_v1.lock`. A new run
requires all three absent, creates the lock with `O_CREAT|O_EXCL|O_RDWR`, writes
canonical JSON `{addendum_sha256,created_utc,final_root,staging_root}`, fsyncs
it, and holds an exclusive nonblocking `fcntl.flock` on that descriptor for the
entire run. Resume requires final absent, stage and lock present, explicit
`--resume`, exact lock schema/content replay, and successful nonblocking
exclusive flock; a dead process has released the kernel lock even though the
file remains. A second process closes without writing. The final, staging and
lock parents must be the same filesystem. After successful final rename and
parent fsync, unlink the external lock and fsync the parent again; no lock file
ever enters the final root.

Before acquisition it also asserts that no canonical A6 S0a boundary/result
path exists. It does not populate `~/.cache`, `HF_HOME`,
`local_cache/a6_tokenizers`, or a synthetic `models--repo` namespace. It never
automatically deletes or overwrites staging/final data. An invalid stage is
noncanonical evidence to quarantine explicitly after review, not a reason to
start a second history at the same path.

The fixed order is Qwen3-8B source inventory and selected blobs, Llama source
inventory and selected blobs, Qwen3-4B official evidence/config, then all-three
cross-checks. Materialized inputs are ordinary regular-file trees:

```text
materialized/
  qwen3-4b/<selected files>
  qwen3-8b/<selected files>
  llama31-8b/<selected files>
evidence/
  official/{qwen3-4b,qwen3-8b,llama31-8b}.raw.json
  drive/{qwen8-cache,llama-cache,llama-flat-original,
         llama-flat-metadata}.{pre,post}.json
  http/qwen3-4b-config.headers.json
  flat/llama-tokenizer.model.metadata
checkpoints/
  000_official_trees.json
  010_drive_pre.json
  020_qwen8_payloads.json
  030_llama_payloads.json
  040_qwen4_config.json
  050_drive_post.json
  060_materialized.json
CANDIDATE_MANIFEST.json
VERIFICATION_ATTESTATION.json
CACHE_RESTORE_PROVENANCE.json
RESTORE_REPORT.md
```

This is an exhaustive final layout; braces denote the listed literal files,
not directories. Every evidence/checkpoint file is regular, exclusively
created, included in a manifest/tree hash, and semantically replayed. Official
files contain the raw HTTPS JSON response bytes. Drive evidence files contain
the redacted canonical projections, command/status and raw stdout/stderr
hashes, not account-bearing raw output. The HTTP header file contains only the
stable projected headers from Section 3.3. The flat metadata file is the exact
124 raw bytes. The Qwen4 raw config response body is the materialized
`qwen3-4b/config.json` itself and is not duplicated.

No file or directory in the staged or final tree may be a symlink, hard link,
device, FIFO, socket, or path escaping the root. Files are opened with
exclusive creation, flushed, fsynced, re-read, and rehashed. Every created
directory and the staging parent are fsynced after child creation/rename.

Finalization is exactly three-phase and noncircular:

1. The restorer writes immutable `CANDIDATE_MANIFEST.json`, whose status is
   `CANDIDATE_COMPLETE_UNVERIFIED`. It hashes all source evidence,
   materialized files/trees, constant report bytes, restorer source/runtime,
   and checkpoints, but contains no verifier verdict.
2. A separate verifier process accepts only that candidate. It re-enumerates
   the tree from zero, repeats official/Drive/source/object checks, loads all
   three tokenizers offline, and writes immutable
   `VERIFICATION_ATTESTATION.json`. The attestation hashes the candidate bytes,
   materialized aggregate, verifier source/runtime, official projections, and
   its replay result. It cannot modify the candidate or materialized tree.
3. The restorer writes `CACHE_RESTORE_PROVENANCE.json` as a small final
   envelope binding the exact candidate and attestation byte hashes, then a
   fresh read-only verifier process replays the envelope/candidate/attestation
   and tree. Only that final replay may call macOS `renameatx_np` through a
   reviewed ctypes wrapper with flag `RENAME_EXCL=0x00000004`, source=staging,
   destination=final. The pinned runtime must expose that primitive; ordinary
   `rename`, `replace`, or check-then-rename fallback is forbidden. Nonzero
   return closes and preserves staging; `EEXIST` proves the concurrent-target
   test. The parent directory is fsynced after successful rename.

Any interruption leaves a noncanonical staging directory. Resume begins with
official endpoint replay, pre/post Drive inventory replay, and a byte/semantic
reconstruction of every existing checkpoint/candidate/attestation; it never
trusts checkpoint status or a stored hash without recomputing its preimage.

The three scorer trees are one transaction. A missing or invalid Qwen3-4B,
Qwen3-8B, or Llama source yields no canonical restoration. No “two out of
three” candidate, attestation, envelope, or final root exists.

Within one role, selected official path, source object, materialized path, and
file record form a bijection. The only allowed cross-role one-to-many source
aliases are these five exact Qwen3-8B payloads reused at the same Qwen3-4B
relative path: `generation_config.json`, `merges.txt`, `tokenizer.json`,
`tokenizer_config.json`, and `vocab.json`. Their output bytes must be equal.
Qwen3-4B and Qwen3-8B `config.json` must be unequal and match their distinct
official objects. No Llama alias and no other one-to-many/many-to-one mapping
is allowed. These five rows, and only these rows, are the frozen
`cross_role_equalities`.

### 5.1 Llama gated-access criterion

The official projection must report `gated="manual"`. The allowed archived
access basis is exactly the pre-existing private project cache beneath the
user-controlled configured `gdrive:` remote; restoration is local research
use and performs no redistribution or new gated download. Provenance records
the non-secret literal attestation
`preexisting-user-controlled-private-project-archive`, the registered source
prefixes/object IDs, and `redistributed=false`. If the archive is not readable
through that configured project remote, if a public/third-party copy is
substituted, or if the use basis cannot be recorded exactly, return
`BLOCKED_TOKENIZER_ACCESS`; flat metadata alone never satisfies this gate.

## 6. Exact typed artifact schemas and hash preimages

Every JSON loader rejects duplicate keys, NaN/Infinity, booleans where an
integer is required, noncanonical numeric/string types, and unknown/missing
keys. Every artifact is byte-canonical JSON as defined in Section 2. The exact
top-level schemas are:

```text
CANDIDATE_MANIFEST.json = {
  schema_version, status, addendum_sha256, parent_contract_sha256,
  restorer_source_manifest, git_head, git_status_summary, runtime,
  drive_remote_fingerprint_redacted, acquisition_times_utc,
  official_response_records, drive_inventory_records,
  llama_access_attestation, roles, cross_role_equalities,
  evidence_manifest, checkpoint_manifest,
  reconstructed_graph_inventory_sha256,
  materialized_aggregate_sha256, restore_report_sha256
}

VERIFICATION_ATTESTATION.json = {
  schema_version, status, candidate_sha256,
  materialized_aggregate_sha256, verifier_source_manifest,
  verifier_runtime, replayed_official_projection_sha256,
  replayed_drive_inventory_sha256, offline_tokenizer_audits,
  verification_checks, verification_started_utc,
  verification_completed_utc
}

CACHE_RESTORE_PROVENANCE.json = {
  schema_version, status, candidate_sha256, attestation_sha256,
  materialized_aggregate_sha256, addendum_sha256,
  parent_contract_sha256, final_replay_required
}
```

Nested schemas are fixed in implementation constants reviewed before source
access and must implement these exact shapes without extensions:

```text
runtime = {
  python:string, platform:string, filesystem:string, locale:string,
  openssl:string, certifi_version:string, certifi_bundle_sha256:hex64,
  rclone_path:absolute_string, rclone_sha256:hex64, rclone_version:string,
  huggingface_hub:string, transformers:string, tokenizers:string
}

file_record = {path:relative_string, size:uint, sha256:hex64}
source_file_record = file_record + {git_blob_sha1:hex40_or_null}
source_manifest = [source_file_record, ...]

official_response_record = {
  role:role, url:exact_https_string, retrieved_utc:rfc3339_utc,
  http_status:200, content_type:string, raw_response_sha256:hex64,
  canonical_projection_sha256:hex64, tree_row_count:uint,
  allowlist_paths:[relative_string,...]
}

drive_inventory_record = {
  source_id:string, prefix:exact_gdrive_string,
  phase:"pre"|"post", batch_id:string,
  command_args:[string,...], raw_stdout_sha256:hex64,
  raw_stderr_sha256:hex64, exit_code:0,
  canonical_projection_sha256:hex64, object_count:uint
}

drive_object_record = {
  path:relative_string, name:string, size:uint, mime_type:string,
  mod_time:rfc3339_utc, id:string,
  hashes:{md5:hex32,sha1:hex40,sha256:hex64},
  metadata:{btime:string,content_type:string,mtime:string}
}

raw_link_record = {
  path:relative_string, drive_object:drive_object_record,
  raw_sha256:hex64, raw_size:uint, raw_target:string,
  normalized_blob_path:relative_string
}

flat_metadata_record = {
  path:exact_flat_metadata_path, drive_object:drive_object_record,
  raw_size:124, raw_sha256:hex64,
  parsed_revision:exact_llama_revision,
  parsed_etag:exact_llama_lfs_sha256,
  parsed_timestamp:"1778745152.481112"
}

standard_drive_transport = {
  kind:"standard_drive", source_id:"qwen8-cache"|"llama-cache",
  raw_link:raw_link_record, drive_object:drive_object_record
}

flat_drive_transport = {
  kind:"flat_drive", source_id:"llama-flat-original",
  drive_object:drive_object_record,
  metadata:flat_metadata_record
}

official_http_transport = {
  kind:"official_http", url:exact_qwen4_config_url,
  http_status:200, content_type:"text/plain; charset=utf-8",
  content_length:726,
  etag:"e49eccdc32f36da9c09cfa0e737084f9e0105e5e",
  x_repo_commit:exact_qwen4_revision,
  raw_response_sha256:hex64, stable_headers_sha256:hex64
}

selected_blob_record = {
  selected_path:relative_string,
  transport:standard_drive_transport|flat_drive_transport|
            official_http_transport,
  official_blob_id:hex40, official_lfs_sha256:hex64_or_null,
  official_size:uint, git_or_pointer_sha1:hex40,
  pointer_sha256:hex64_or_null, payload_sha256:hex64
}

selected_mapping = {
  selected_path:relative_string, official_tree_index:uint,
  blob:selected_blob_record,
  output:file_record
}

offline_tokenizer_audit = {
  tokenizer_class:string, is_fast:true, vocabulary_size:uint,
  chat_template:string, chat_template_sha256:hex64,
  special_tokens_map_sha256:hex64,
  tokenizer_eos_token_ids:[uint,...], model_config_eos_token_ids:[uint,...],
  generation_config_eos_token_ids:[uint,...],
  effective_generation_eos_token_ids:[uint,...],
  tokenizer_pad_token_id:uint_or_null, model_config_pad_token_id:uint_or_null,
  generation_config_pad_token_id:uint_or_null,
  effective_generation_pad_token_id:uint_or_null,
  known_answer_audit_sha256:hex64
}

role_record = {
  repository:string, revision:hex40, source_kind:string,
  source_prefixes:[string,...], official_projection_sha256:hex64,
  selected_paths:[relative_string,...], mappings:[selected_mapping,...],
  materialized_files:[file_record,...], local_tree_sha256:hex64,
  offline_tokenizer_audit:offline_tokenizer_audit
}

cross_role_equality = {
  left_role:"qwen3-4b", right_role:"qwen3-8b",
  path:one_of_the_five_literal_paths, sha256:hex64
}

checkpoint_record = {
  ordinal:uint, phase:string, path:relative_string,
  bytes_sha256:hex64, semantic_replay_sha256:hex64
}

verification_check = {name:string, passed:true, evidence_sha256:hex64}
```

`runtime`, acquisition times, git status, the Llama attestation and report
records are exact objects too: acquisition times have only `started,completed`;
git status has only `head,branch,tracked_changes,untracked_paths`; Llama access
has only `gated,access_basis,redistributed,source_prefixes`; the redacted Drive
fingerprint has only `remote_name,type,redacted_config_sha256`; and report has
only its SHA-256 in the candidate. `roles` has exactly
`qwen3-4b,qwen3-8b,llama31-8b`; `cross_role_equalities` has exactly the five
Section-5 records. `verification_checks` is the UTF-8-name-sorted complete
literal roster
`all_three_roles,allowed_aliases,candidate_bytes,drive_pre_post_identity,
llama_access,materialized_trees,no_forbidden_paths,official_trees,
offline_tokenizers,report_bytes,role_bijections,selected_payloads,
source_closure,runtime_boundary`; a missing, extra, false, or duplicate name
fails.

A file record is exactly `{path,size,sha256}`. A tree hash is SHA-256 of the
canonical JSON bytes of the UTF-8-path-sorted file-record array. The
materialized aggregate is SHA-256 of canonical JSON over three exact records
`{role,tree_sha256}` in role order. Source, test, and transitive local Python
manifests use the same file-record array/hash rule. The graph inventory hash is
over the canonical role/path/source/object/output mapping array. Candidate and
attestation hashes are over their exact complete file bytes. The envelope has
no self-hash field. `RESTORE_REPORT.md` is a source constant written before the
candidate; its exact SHA-256 is bound in the candidate and it has no authority
beyond that binding.

Candidate status is exactly `CANDIDATE_COMPLETE_UNVERIFIED`; attestation status
is `VERIFIED_CANDIDATE_ALL_THREE`; envelope status is
`AUTHENTICATED_COMPLETE_ALL_THREE`, with `final_replay_required=true`. Every
source/test/transitive local Python file loaded before or during restoration is
hash-bound. `git_status_summary` records tracked changes and untracked paths
without reading unrelated untracked payloads. Credentials, OAuth tokens,
signed URLs, cookies, and Drive secrets are never serialized.

The `offline_tokenizer_audit` records exact tokenizer class, fast-tokenizer
status, resolved chat-template bytes/hash, special-token maps, EOS/pad sources
and effective IDs, vocabulary size, and deterministic known-answer tokenization
checks. It is diagnostic here; the complete contextual quartet audit remains
S0a's job.

## 7. Binding restoration into the S0a boundary

The S0a runner must no longer accept a bare cache directory or a successful
`scan_cache_dir` as provenance. `prepare` accepts only the final restoration
root plus its canonical three-artifact chain. The old `qwen4_source`,
`qwen8_source`, and `llama_source` arguments are removed and rejected. Before
creating the S0a output root or importing Transformers it:

1. performs the fresh final replay of envelope, candidate, attestation,
   addendum, restorer/verifier source hashes, exact statuses, and all-three
   role table;
2. re-enumerates the materialized root and recomputes every file/tree hash;
3. requires the frozen repository/revision/path tables and cross-role byte
   equalities;
4. copies each regular-file role tree into the new S0a boundary-input staging
   directory and rehashes it;
5. embeds exact candidate, attestation and
   `CACHE_RESTORE_PROVENANCE.json` SHA-256 values plus the materialized
   aggregate SHA-256 in `A6_S0A_BOUNDARY.json`; and
6. loads only the copied tree with offline flags and `local_files_only=True`.

The authoritative S0a verifier repeats these checks. A boundary whose source
provenance is missing, changed, unmanifested, hash-only without semantic replay,
or points at a different restoration root fails. The restoration artifact is
an input boundary, not an experimental result, and must be committed/reviewed
before S0a execution just like the S0a source/runtime boundary.

## 8. Mandatory fail-closed tests before acquisition

Unit and reduced end-to-end tests must cover at least:

- altered selected blob bytes, size, SHA-256, Git blob, LFS pointer, or LFS
  expanded-payload record, including pointer/payload confusion;
- altered raw snapshot-entry bytes or Drive object ID;
- wrong repository, wrong revision, mutable `main`, or spoofed revision path;
- dangling, empty, absolute, multi-hop, traversal, case-colliding, or
  root-escaping targets;
- symlink/hard-link/device/FIFO/socket in staged or final trees;
- duplicate source-to-output or output-to-source mappings: the five exact
  cross-Qwen aliases pass and every other duplicate fails;
- missing selected path, extra selected path, incomplete tree, or optional
  path silently inferred from loader success;
- changed/partial Drive inventory, duplicate normalized path/Drive object
  identity, and mutation between pre-inventory, copy, and post-inventory;
- Qwen3-4B official-object mismatch despite byte equality to Qwen3-8B;
- official-tree truncation/pagination/schema drift, a forged API record,
  redirect/mutable URL, or wrong 4B config;
- flat Llama metadata offered as a fallback;
- one scorer missing or invalid, proving that no canonical partial root or S0a
  boundary is produced;
- interruption after every write/finalization phase, deterministic distrustful
  resume, concurrent process/final-root creation, and an existing
  final/staging/canonical-boundary refusal;
- canonical JSON byte tampering and provenance-field/schema poisoning;
- candidate/attestation/envelope/report tampering and unknown nested keys;
- wrong restoration/addendum/source/runtime hash at S0a preparation;
- old bare-cache S0a arguments rejected, and complete provenance replay before
  S0a output-root creation or Transformers import;
- fresh offline prepare reproducing exact tokenizer selected/content/template
  hashes; and
- an independent verifier that derives its verdict from replay rather than
  trusting `status`, a report, or `scan_cache_dir`.

Tests use tiny synthetic repositories and Drive-object fixtures. They may not
write a real repository/revision name unless the bytes are the exact registered
source fixture, preventing a test cache from being mistaken for evidence.

## 9. Hard-stop interpretation

This restoration cannot create evidence for or against PTNI-IU. It only decides
whether the frozen S0a tokenizer inputs can be authenticated. Outcomes are:

- `AUTHENTICATED_COMPLETE_ALL_THREE`: the reviewed restoration may become an
  input to a separately prepared and independently reviewed S0a boundary;
- `BLOCKED_TOKENIZER_ACCESS`: an exact source or licensed artifact cannot be
  obtained or authenticated;
- `INVALID_RESTORE_IMPLEMENTATION`: a code/schema/replay invariant fails before
  source access can be interpreted.

No failure permits changing a model revision, using Qwen2.5 as a proxy,
accepting flat metadata, weakening a hash, hand-authoring a missing file, or
proceeding with a partial scorer set. No restoration outcome permits opening
response telemetry or S1 seeds.

## Appendix A. Frozen official revision-tree projections

These are the complete canonical `siblings` projections defined in Section
3.0. Columns are `path | blobId | size | lfs.sha256 | lfs.size |
lfs.pointerSize`; `-` means the `lfs` field is absent. Row order is UTF-8 path
order. A different row count, order after canonical sorting, value, or optional
field closes restoration.

### A.1 Qwen/Qwen3-4B at `1cfa9a7208912126459214e8b04321603b3df60c`

```text
.gitattributes | 52373fe24473b1aa44333d318f578ae6bf04b49b | 1570 | - | - | -
LICENSE | 6634c8cc3133b3848ec74b9f275acaaa1ea618ab | 11343 | - | - | -
README.md | 2de5ee7eee214bb55ea33ec7505c5838a7adf7f6 | 16857 | - | - | -
config.json | e49eccdc32f36da9c09cfa0e737084f9e0105e5e | 726 | - | - | -
generation_config.json | 20a8a9156fc8c3f25295ca067f61fdf120d517c5 | 239 | - | - | -
merges.txt | 31349551d90c7606f325fe0f11bbb8bd5fa0d7c7 | 1671853 | - | - | -
model-00001-of-00003.safetensors | 546625baab99f4de2b975842100c119fec83919f | 3957900840 | 328a91d3122359d5547f9d79521205bc0a46e1f79a792dfe650e99fc2d651223 | 3957900840 | 135
model-00002-of-00003.safetensors | 3e4d29aad0be87c083e078b50765cd056e793c51 | 3987450520 | 6cd087b316306a68c562436b5492edbcf6e16c6dba3a1308279caa5a58e21ca5 | 3987450520 | 135
model-00003-of-00003.safetensors | 79fb0e18fca00d3a2ab04e5b732a2a740539b913 | 99630640 | e4bf436957184f4eeb86a80e9db394503f1f56446b2e6b7edeac5b81470f4ca1 | 99630640 | 133
model.safetensors.index.json | 95c0a0059df040d75dc6c396b174382cf61d2f91 | 32819 | - | - | -
tokenizer.json | cd71f61a15a522601badb3dc960d800d9cb3766c | 11422654 | aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4 | 11422654 | 133
tokenizer_config.json | 417d038a63fa3de29cfde265caedae14d1a58d92 | 9732 | - | - | -
vocab.json | 4783fe10ac3adce15ac8f358ef5462739852c569 | 2776833 | - | - | -
```

Allowlist intersection, exactly:
`config.json,generation_config.json,merges.txt,tokenizer.json,
tokenizer_config.json,vocab.json`.

### A.2 Qwen/Qwen3-8B at `b968826d9c46dd6066d109eabc6255188de91218`

```text
.gitattributes | 52373fe24473b1aa44333d318f578ae6bf04b49b | 1570 | - | - | -
LICENSE | 6634c8cc3133b3848ec74b9f275acaaa1ea618ab | 11343 | - | - | -
README.md | ecc3ebd0849aa08d9484bd911dddfd5261b10d30 | 16660 | - | - | -
config.json | d46195ac87f837ad233d02b2f80f148bf7c005e0 | 728 | - | - | -
generation_config.json | 20a8a9156fc8c3f25295ca067f61fdf120d517c5 | 239 | - | - | -
merges.txt | 31349551d90c7606f325fe0f11bbb8bd5fa0d7c7 | 1671853 | - | - | -
model-00001-of-00005.safetensors | 8d46cf470aff053c27fd6d956d4d69af9460bd2a | 3996250744 | 31d6a825ae35f11fb85b195b4c42c146c051e446433125a215336abdf95cbf5f | 3996250744 | 135
model-00002-of-00005.safetensors | e726a2bcb3b100cba7ae899689cc029d8a08b20c | 3993160032 | 5991236cea6fe21f3d43cab0f0e84448734fbbe0789816202989f2ddc9d18282 | 3993160032 | 135
model-00003-of-00005.safetensors | c94db38adbcb837246c797986365b8e4603ef3ac | 3959604768 | c5185c4794be2d8a9784d5753c9922db38df478ce11f9ed0b415b7304d896836 | 3959604768 | 135
model-00004-of-00005.safetensors | a6c92dd09e29e1f72271e087f89ee033b321101c | 3187841392 | b5ee7de71fbf17db3d5704e0c8f2bc7d005ca9e1d7ca2aeb19827b0cfcaa917a | 3187841392 | 135
model-00005-of-00005.safetensors | 7bbb0cbc623d70925a3a82b301a99cb52ac9ed5b | 1244659840 | 20c2d6366ab85c90786ccdd829cd2b9e7d30ef3b2ebbb998280e7e4014b542ff | 1244659840 | 135
model.safetensors.index.json | 2b85c00f1b118961cd7a477e2bba0fe197a4ce1a | 32878 | - | - | -
tokenizer.json | cd71f61a15a522601badb3dc960d800d9cb3766c | 11422654 | aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4 | 11422654 | 133
tokenizer_config.json | 417d038a63fa3de29cfde265caedae14d1a58d92 | 9732 | - | - | -
vocab.json | 4783fe10ac3adce15ac8f358ef5462739852c569 | 2776833 | - | - | -
```

Allowlist intersection, exactly:
`config.json,generation_config.json,merges.txt,tokenizer.json,
tokenizer_config.json,vocab.json`.

### A.3 meta-llama/Llama-3.1-8B-Instruct at `0e9e39f249a16976918f6564b8830bc894c89659`

```text
.gitattributes | a6344aac8c09253b3b630fb776ae94478aa0275b | 1519 | - | - | -
LICENSE | a7c3ca16cee30425ed6ad841a809590f2bcbf290 | 7627 | - | - | -
README.md | bbd5630a05b65c1a8b25141bd11ec44844107d58 | 44044 | - | - | -
USE_POLICY.md | 81ebb55902285e8dd5804ccf423d17ffb2a622ee | 4691 | - | - | -
config.json | 0bb6fd75b3ad2fe988565929f329945262c2814e | 855 | - | - | -
generation_config.json | cc7276afd599de091142c6ed3005faf8a74aa257 | 184 | - | - | -
model-00001-of-00004.safetensors | a59dca28867995c0d05384251ac5f4d62461a226 | 4976698672 | 2b1879f356aed350030bb40eb45ad362c89d9891096f79a3ab323d3ba5607668 | 4976698672 | 135
model-00002-of-00004.safetensors | 4ab0726caf5e75dd1e5700c7e7899ac8c0798428 | 4999802720 | 09d433f650646834a83c580877bd60c6d1f88f7755305c12576b5c7058f9af15 | 4999802720 | 135
model-00003-of-00004.safetensors | 5864b15557563e7700a3f95236afdee36f5fdb74 | 4915916176 | fc1cdddd6bfa91128d6e94ee73d0ce62bfcdb7af29e978ddcab30c66ae9ea7fa | 4915916176 | 135
model-00004-of-00004.safetensors | b8babcaa5a888d6efd5c05206361b4685492ad13 | 1168138808 | 92ecfe1a2414458b4821ac8c13cf8cb70aed66b5eea8dc5ad9eeb4ff309d6d7b | 1168138808 | 135
model.safetensors.index.json | 0fd8120f1c6acddc268ebc2583058efaf699a771 | 23950 | - | - | -
original/consolidated.00.pth | 419e2d2c8b3fcbea2955c1177110fe777df351f7 | 16060617592 | ab33d910f405204e5d388bc3521503584800461dc96808e287821dd451c1edac | 16060617592 | 136
original/params.json | f1131204e79d0c09d2bac93f11569a8a655d68ba | 199 | - | - | -
original/tokenizer.model | a097ce5a06fce0fa3d685a8cfb175cef243dfde9 | 2183982 | 82e9d31979e92ab929cd544440f129d9ecd797b69e327f80f17e1c50d5551b55 | 2183982 | 132
special_tokens_map.json | 02ee80b6196926a5ad790a004d9efd6ab1ba6542 | 296 | - | - | -
tokenizer.json | 5cc5f00a5b203e90a27a3bd60d1ec393b07971e8 | 9085657 | - | - | -
tokenizer_config.json | db88166e2bc4c799fd5d1ae643b75e84d03ee70e | 55351 | - | - | -
```

The recursive basename allowlist intersection is exactly:
`config.json,generation_config.json,original/tokenizer.model,
special_tokens_map.json,tokenizer.json,tokenizer_config.json`.
