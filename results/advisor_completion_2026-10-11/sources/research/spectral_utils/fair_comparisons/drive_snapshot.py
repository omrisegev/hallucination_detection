"""Frozen read-only Google Drive metadata observation for comparison package v1.

Only small metadata/status objects were inspected.  The ledger binds every claim to
the Drive-reported SHA-256 and byte size; it does not copy result payloads, mutate
Drive, or turn an incomplete acquisition into an eligible comparison row.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import PurePosixPath
import re
from typing import Any

from .registry import canonical_sha256, is_sha256


OBSERVATION_DATE = "2026-08-18"

L0_INVENTORY = {
    "path": "paper_exact/l0/L0_INVENTORY.json",
    "size_bytes": 61494,
    "sha256": "15e08623eaa30d93da7a6a00457a182815db5353c61a25565e3ed92eac5d5956",
    "modified_utc": "2026-08-17T15:01:08.000Z",
}

S1_METADATA = (
    {
        "path": "paper_exact/s1_refrain_full/BANDIT_STATE.json",
        "size_bytes": 1838,
        "sha256": "0bd1611d69f15a20d7ba24923aa4567a839066ddad1f642392e1179462743117",
    },
    {
        "path": "paper_exact/s1_refrain_full/GATE_S1-refrain-full.json",
        "size_bytes": 572,
        "sha256": "5bead462ee5722cf4afec488eb792438043ebbd39487f746050f2e5a26cd2679",
    },
    {
        "path": "paper_exact/s1_refrain_full/INDEX.jsonl",
        "size_bytes": 37700,
        "sha256": "f38cae596d620d940c7813081c204081cc269fa18363e3d06f960b2825219ba0",
    },
    {
        "path": "paper_exact/s1_refrain_full/RUN_MANIFEST.json",
        "size_bytes": 22115,
        "sha256": "b5cf10ab747fb51b7c87603201d48ae1d8dad7e5b990aa5949637bc233090f4b",
    },
    {
        "path": "paper_exact/s1_refrain_full/STATUS.json",
        "size_bytes": 340,
        "sha256": "e7e34040c798df1dbba9e5b9804cc04be6f1f748e45f2c62f788c420ce299d0b",
    },
)

L1_METADATA = (
    {
        "path": "paper_exact/l1_uprm_judge_full/GATE_L1-uprm-judge-full.json",
        "size_bytes": 598,
        "sha256": "d5dc93943fde859bfabe6f9dbdf86bd7576cfdfa60af6f530cf3819824a2c459",
    },
    {
        "path": "paper_exact/l1_uprm_judge_full/INDEX.jsonl",
        "size_bytes": 180776,
        "sha256": "74b60dab966b80734cb3aa67c2c3551089734af2e59193d7a1b33ae4295844a5",
    },
    {
        "path": "paper_exact/l1_uprm_judge_full/L1_DIAGNOSIS.json",
        "size_bytes": 2202,
        "sha256": "fd84e3de8a4ee96aea552e42eb2082bc088409e7140cd83a099766de456b4053",
    },
    {
        "path": "paper_exact/l1_uprm_judge_full/RUN_MANIFEST.json",
        "size_bytes": 101259,
        "sha256": "68f206dc59d714e4dc2b590214d916f134204770f72f7ab66dde7efbc1a2ffd2",
    },
    {
        "path": "paper_exact/l1_uprm_judge_full/STATUS.json",
        "size_bytes": 339,
        "sha256": "d7f3076d127eaa0988b7d86273ac08e89cf4b94b6871fff146d60c733333cc06",
    },
    {
        "path": "paper_exact/l1_uprm_judge_full/SUMMARY.json",
        "size_bytes": 1251,
        "sha256": "133346aa2ebcf91d1a292e704042e0aa14ec97d9f6190d1bd63283b381999e55",
    },
)

_S2_COMPLETE_METADATA_BY_RUN = {
    "s2_leash_Llama-3.1-8B-Instruct_aqua": (
        ("GATE_S2-leash-full.json", 737, "9455d1b53711d3cb3429ad288d143b408323449a4e60e254312f5a2279493253"),
        ("INDEX.jsonl", 30554, "844422f0d5ae54434737eca917480e10a6425886d43c2b0aef539c5cb0948b87"),
        ("RUN_MANIFEST.json", 8304, "66e3ae93ba5361d5ae04efc4ed761caf1e7976194465d8843f300810f8981351"),
        ("STATUS.json", 356, "edec2287f8f5390fc739a9d36aeeae034e3b83c71964a8c95329210744e32288"),
        ("SUMMARY.json", 1172, "0cd285d7584704276a4ed665952ab814d14e7fe12bf10da57196f6b45e1c9d1c"),
    ),
    "s2_leash_Llama-3.1-8B-Instruct_gsm8k": (
        ("GATE_S2-leash-full.json", 738, "762daa6627a2c0e783fde6d00505be698111c56c8d73e615322e672424531162"),
        ("INDEX.jsonl", 39068, "eb3ccff90acac53070a6f434e30cc2b19e4458685dcdd57709a5994491aaf527"),
        ("RUN_MANIFEST.json", 9481, "3d1c83c6c2c3ded949d2fa14267d9274d09e3f20305cad41bba7955b700641f8"),
        ("STATUS.json", 357, "79759760ca6c0ce560eb89c2ff1bdf128cdcd37c9c501471ec861bc7799dd5e2"),
        ("SUMMARY.json", 1238, "d0b708c5700c0bc2a1832f3ed9c8ffb65d29db6dccd571293bbc1cf126fc5774"),
    ),
    "s2_leash_Phi-3-mini-128k-instruct_aqua": (
        ("GATE_S2-leash-full.json", 740, "56df533e2546355f849b0f43cf3bb5f0613885c19234f0ac35b29468170632d1"),
        ("INDEX.jsonl", 30554, "75c15f83ac95421694e339755cf6c4b806ee6a0bb9500c4de4b7847854ec605c"),
        ("RUN_MANIFEST.json", 8309, "8327dea16c9d524acf30b4abef58bab4ed75f1e631a4ef6b5d24d0ae5b04d113"),
        ("STATUS.json", 359, "8e623240ee2a0186779811bfdd93f50b04583dcc5b9eb52585a5c7e5bfe31691"),
        ("SUMMARY.json", 1174, "790ee99d7280d1998e343810e891a078da760cde57d0db048e1ab3cd1adf4055"),
    ),
    "s2_leash_Phi-3-mini-128k-instruct_gsm8k": (
        ("GATE_S2-leash-full.json", 741, "fdcd9620435f2f715a8d4ebe32afc56450635d66ab7cca00791f7b30e861dbcb"),
        ("INDEX.jsonl", 39068, "a743e6cb23f410b2dabae591c0486ce30f29d6cdfc2acccb4c23fb7a6141cdc6"),
        ("RUN_MANIFEST.json", 9486, "ca7d441e9305eb69f0dc4266df5d15c832476ac433ceea5f6f2587c89e252bae"),
        ("STATUS.json", 360, "612b13e96a859a96bd3973f3c185fefdb547d598c4ceded4f81480674df43ee8"),
        ("SUMMARY.json", 1200, "dda07d688ec01559c0b9eb233256bd35d0d47bfc75409edece5789e43a023c9b"),
    ),
    "s2_leash_Qwen2.5-7B-Instruct_aqua": (
        ("GATE_S2-leash-full.json", 735, "fa88513e68fc0c156eccce6addc777a8515ef7658c54d024301d71a786595980"),
        ("INDEX.jsonl", 30554, "f88e5700f71fed53577ed5ef836d127f2935879c32f97278afca109bfb29ff60"),
        ("RUN_MANIFEST.json", 8294, "5556bbfbd83cf39b6674a0a013b461b09d9725839f9d5d3f0ae49df49abacb16"),
        ("STATUS.json", 354, "f5a48b9ec7686c8ec460ff1c067cbddf86cca87aa9eba190991336f82d8adc0c"),
        ("SUMMARY.json", 1224, "a19de7d31c2257ef165a5aea3fb1b26bb3953d8c30f2e2cd4cf9784f53dba102"),
    ),
    "s2_leash_Qwen2.5-7B-Instruct_gsm8k": (
        ("GATE_S2-leash-full.json", 736, "e22f983d02d818810b6373aa84815177eed6530ef0a6f98074bb93b6e3d422af"),
        ("INDEX.jsonl", 39068, "2962a060c1b49208c767b2ed532d200bd09b3c51759a3d5024c92f3954d42e46"),
        ("RUN_MANIFEST.json", 9471, "35ee412c6fff72dc26f88490a28bf4af94712d5d86b38d66dd4487fef2dc3de3"),
        ("STATUS.json", 355, "0bd0aaaf09c5558dbe8af801e4c284a843cc0b58ad12577ed3e6ee6e19e2ba79"),
        ("SUMMARY.json", 1186, "ebfcd87dda090f90f08f7da5e1d22b371e462336126ccdcc2fd02083646d028f"),
    ),
}

S2_COMPLETE_METADATA = tuple(
    {
        "path": f"paper_exact/{run_id}/{filename}",
        "size_bytes": size_bytes,
        "sha256": sha256,
    }
    for run_id in sorted(_S2_COMPLETE_METADATA_BY_RUN)
    for filename, size_bytes, sha256 in _S2_COMPLETE_METADATA_BY_RUN[run_id]
)

_M2_STATUS_HASHES = (
    "150c985c955491c4ba84eba2f1c6259b7c92a834bdf047b0b5e63e845e0f5edd",
    "e4d2cfcd3b28778ffd2c937136b03ba2834b3e2fa83a064914a839e7797ea12f",
    "e172f0cb6ff62563a01796c0881868859f3690216dad2b701e1977a2fcbc4fab",
    "62f219d2f62ffbe3e9f2f078516e07c7c4b749f0bef508e39a4d094139a2906f",
    "4b4751918e57626b5e355c4e2174e031c7ce569cb7f19fe6aaa09e4619402480",
    "dabea202eb4cad2fd60c54ba09b7fd15758561b6404c011d5c55fa339e1417af",
    "dcd6063aab521a2210d735ee072b2cdf948958a8ca96741e8811711844356e18",
    "d9b2ded188461d748ffcd6cc7a0b2348a3db30cb0ecfe3890215867150fe8a34",
    "a43483a71915717d4783b02f4085dd08b12e640909d6553ab0b34c83db84b703",
    "8c8bc7171f4b88337c9cfb4fca0e88ff3c18c71958df275c335b8f29d0c494f6",
    "61237f699847e2443fbd4642fc0020c4e4d8fb776e25338e11a791c6e8744d11",
    "351bed3f211e411f24ae41e5400c78904ee836879f6478ce043fbfebc89cf723",
    "298977c4886818acdfc1d0a48fe2d53eec5e8f8729095bb70b23cca4562be6cd",
    "ff94ea17a5a22fe53f5467672c0779c1c3d674716b267380c57da8c1deb32ee3",
    "17aa3d322d18e8e8b96d2225b6c6df822bc0cb02bb3b52bd46d2688b3a29f92e",
    "dae5eda45ff3d9dc7d4e4f67c0ab9855cb9865a7d9c11bcd81b4501f9de17e90",
    "1fe2c2d1617eaf5ed335ac6a9a2a84d0a78cdb88d789c94e0c8d445fc1255db4",
    "f0235fc283981e66a3d579f2934e21d38d0385f508be364902c7e3b440ba1018",
    "deb5c93efaceb21769f60d6178cf8e1e37bbfdc23d4fb2fabf3c34b164e37532",
    "8d7c1d1844cc7b1ba4224ee9f0b011a9b878e4206945b9356d5fb1511f80f817",
    "151a2bc23890a6e93391c858d2015c1b38803f873245c312885a9e716db1f846",
    "9ff0131adb3c2d7260959b727782ba36e454623e0ef942543dc1ce96f911ca39",
    "1c2c53a183cfa9da15fc58b7e466133d008807b61da099f7d1f7b0f8b575a8f0",
    "c98a39a589b15fbcae971237af92a99eda366637f41c52c8015a9cf812bfa107",
)

M2_STATUS_METADATA = tuple(
    {
        "path": f"paper_exact/m2_deepconf_full/part_{index:02d}/STATUS.json",
        "size_bytes": 349 if index in {0, 8, 16} else 348,
        "sha256": sha256,
    }
    for index, sha256 in enumerate(_M2_STATUS_HASHES)
)

_FROZEN_SUMMARIES = {
    "l0": {
        "shared_processbench_rows": 3400,
        "shared_table_materialized": False,
    },
    "l1_uprm": {
        "finished": 3400,
        "expected": 3400,
        "failed": 0,
        "shards": 54,
        "output_bytes": 7918993,
        "metadata_members": len(L1_METADATA),
    },
    "s1_refrain": {
        "finished": 512,
        "expected": 1000,
        "failed": 0,
        "summary_present": False,
    },
    "s2_leash_complete": {
        "complete_cells": 6,
        "metadata_members": len(S2_COMPLETE_METADATA),
        "payload_bytes": 405929606,
    },
    "m2_deepconf": {
        "finished": 12370,
        "expected": 122880,
        "failed": 0,
        "status_members": 24,
        "formal_checkpoint_finished": 4608,
        "formal_checkpoint_stale": True,
        "raw_logit_audit_n": 0,
    },
}


class DriveObservationError(ValueError):
    """The frozen Drive metadata observation is internally inconsistent."""


def _validate_member(member: Any, *, section: str, index: int) -> dict[str, Any]:
    if not isinstance(member, Mapping):
        raise DriveObservationError(f"{section}[{index}] must be a mapping")
    required = ("path", "size_bytes", "sha256")
    missing = [field for field in required if field not in member]
    if missing:
        raise DriveObservationError(f"{section}[{index}] missing {missing}")
    normalized = dict(member)
    path = normalized["path"]
    if not isinstance(path, str) or not path or "\\" in path:
        raise DriveObservationError(f"{section}[{index}] has invalid relative Drive path")
    pure = PurePosixPath(path)
    if (
        pure.is_absolute()
        or path != pure.as_posix()
        or any(part in {"", ".", ".."} for part in pure.parts)
        or path.startswith("gdrive:")
    ):
        raise DriveObservationError(
            f"{section}[{index}] path must be remote-prefix-relative and canonical"
        )
    size = normalized["size_bytes"]
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise DriveObservationError(f"{section}[{index}] has invalid size_bytes")
    if not is_sha256(normalized["sha256"]):
        raise DriveObservationError(f"{section}[{index}] has invalid sha256")
    modified = normalized.get("modified_utc")
    if modified is not None and (
        not isinstance(modified, str)
        or re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z", modified)
        is None
    ):
        raise DriveObservationError(f"{section}[{index}] has invalid modified_utc")
    return normalized


def validate_drive_metadata_observation(observation: Mapping[str, Any]) -> dict[str, Any]:
    """Validate exact members, relative paths, counts, sizes, and content hashes."""

    if not isinstance(observation, Mapping):
        raise DriveObservationError("Drive metadata observation must be a mapping")
    required = (
        "schema",
        "observation_date",
        "remote_prefix",
        "access_mode",
        "drive_mutated",
        "metadata_members",
        "metadata_member_count",
        "metadata_total_size_bytes",
        "metadata_members_sha256",
        "summaries",
        "operational_risk",
        "observation_sha256",
    )
    missing = [field for field in required if field not in observation]
    if missing:
        raise DriveObservationError(f"Drive metadata observation missing {missing}")
    normalized = dict(observation)
    if normalized["schema"] != "fair_comparison_drive_metadata_observation_v1":
        raise DriveObservationError("unexpected Drive metadata observation schema")
    if normalized["observation_date"] != OBSERVATION_DATE:
        raise DriveObservationError("Drive metadata observation date drift")
    if normalized["remote_prefix"] != "gdrive:hallucination_detection/cluster_results":
        raise DriveObservationError("Drive remote prefix drift")
    if normalized["drive_mutated"] is not False:
        raise DriveObservationError("Drive observation must remain read-only")
    members = normalized["metadata_members"]
    section_order = (
        "l0",
        "l1_uprm",
        "s1_refrain",
        "s2_leash_complete",
        "m2_deepconf_status",
    )
    if not isinstance(members, Mapping) or set(members) != set(section_order):
        raise DriveObservationError("Drive metadata member sections drift")
    validated_members: dict[str, list[dict[str, Any]]] = {}
    paths: list[str] = []
    for section in section_order:
        values = members[section]
        if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
            raise DriveObservationError(f"Drive metadata section {section} must be a sequence")
        validated_members[section] = [
            _validate_member(member, section=section, index=index)
            for index, member in enumerate(values)
        ]
        paths.extend(member["path"] for member in validated_members[section])
    if len(paths) != len(set(paths)):
        raise DriveObservationError("Drive metadata observation contains duplicate paths")
    expected_counts = {
        "l0": 1,
        "l1_uprm": 6,
        "s1_refrain": 5,
        "s2_leash_complete": 30,
        "m2_deepconf_status": 24,
    }
    observed_counts = {
        section: len(validated_members[section]) for section in expected_counts
    }
    if observed_counts != expected_counts:
        raise DriveObservationError(
            f"Drive metadata member counts drift: {observed_counts}"
        )
    frozen_members = {
        "l0": [dict(L0_INVENTORY)],
        "l1_uprm": [dict(row) for row in L1_METADATA],
        "s1_refrain": [dict(row) for row in S1_METADATA],
        "s2_leash_complete": [dict(row) for row in S2_COMPLETE_METADATA],
        "m2_deepconf_status": [dict(row) for row in M2_STATUS_METADATA],
    }
    if validated_members != frozen_members:
        raise DriveObservationError("exact frozen Drive metadata members drift")
    member_count = sum(observed_counts.values())
    total_size = sum(
        member["size_bytes"]
        for values in validated_members.values()
        for member in values
    )
    if normalized["metadata_member_count"] != member_count:
        raise DriveObservationError("Drive metadata_member_count is inconsistent")
    if normalized["metadata_total_size_bytes"] != total_size:
        raise DriveObservationError("Drive metadata_total_size_bytes is inconsistent")
    member_hash = canonical_sha256(validated_members)
    if normalized["metadata_members_sha256"] != member_hash:
        raise DriveObservationError("Drive metadata_members_sha256 is inconsistent")
    summaries = normalized["summaries"]
    if not isinstance(summaries, Mapping):
        raise DriveObservationError("Drive summaries must be a mapping")
    try:
        if int(summaries["l1_uprm"]["metadata_members"]) != observed_counts[
            "l1_uprm"
        ]:
            raise DriveObservationError("L1 summary metadata_members is inconsistent")
        if int(summaries["s2_leash_complete"]["metadata_members"]) != observed_counts[
            "s2_leash_complete"
        ]:
            raise DriveObservationError("S2 summary metadata_members is inconsistent")
        if int(summaries["s2_leash_complete"]["complete_cells"]) != 6:
            raise DriveObservationError("S2 summary complete_cells is inconsistent")
        if int(summaries["m2_deepconf"]["status_members"]) != observed_counts[
            "m2_deepconf_status"
        ]:
            raise DriveObservationError("M2 summary status_members is inconsistent")
        for section in ("s1_refrain", "m2_deepconf"):
            finished = int(summaries[section]["finished"])
            expected = int(summaries[section]["expected"])
            failed = int(summaries[section]["failed"])
            if min(finished, expected, failed) < 0 or finished > expected:
                raise DriveObservationError(f"{section} summary counts are inconsistent")
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, DriveObservationError):
            raise
        raise DriveObservationError("Drive summaries lack frozen count fields") from exc
    if summaries != _FROZEN_SUMMARIES:
        raise DriveObservationError("exact frozen Drive summaries drift")
    hash_projection = dict(normalized)
    hash_projection.pop("observation_sha256")
    if normalized["observation_sha256"] != canonical_sha256(hash_projection):
        raise DriveObservationError("Drive observation_sha256 is inconsistent")
    normalized["metadata_members"] = validated_members
    return normalized


def build_drive_metadata_observation() -> dict[str, Any]:
    """Return the canonical status ledger and summaries read from those objects."""

    observation = {
        "schema": "fair_comparison_drive_metadata_observation_v1",
        "observation_date": OBSERVATION_DATE,
        "remote_prefix": "gdrive:hallucination_detection/cluster_results",
        "access_mode": "read-only rclone lsd/lsf/lsjson/size/cat",
        "drive_mutated": False,
        "metadata_members": {
            "l0": [dict(L0_INVENTORY)],
            "l1_uprm": [dict(row) for row in L1_METADATA],
            "s1_refrain": [dict(row) for row in S1_METADATA],
            "s2_leash_complete": [dict(row) for row in S2_COMPLETE_METADATA],
            "m2_deepconf_status": [dict(row) for row in M2_STATUS_METADATA],
        },
        "metadata_member_count": (
            1
            + len(L1_METADATA)
            + len(S1_METADATA)
            + len(S2_COMPLETE_METADATA)
            + len(M2_STATUS_METADATA)
        ),
        "metadata_total_size_bytes": sum(
            int(row["size_bytes"])
            for row in (
                L0_INVENTORY,
                *L1_METADATA,
                *S1_METADATA,
                *S2_COMPLETE_METADATA,
                *M2_STATUS_METADATA,
            )
        ),
        "metadata_members_sha256": canonical_sha256(
            {
                "l0": [dict(L0_INVENTORY)],
                "l1_uprm": [dict(row) for row in L1_METADATA],
                "s1_refrain": [dict(row) for row in S1_METADATA],
                "s2_leash_complete": [dict(row) for row in S2_COMPLETE_METADATA],
                "m2_deepconf_status": [dict(row) for row in M2_STATUS_METADATA],
            }
        ),
        "summaries": {
            section: dict(values) for section, values in _FROZEN_SUMMARIES.items()
        },
        "operational_risk": (
            "rclone shared Google Drive client_id is scheduled for retirement during 2026"
        ),
    }
    observation["observation_sha256"] = canonical_sha256(observation)
    return validate_drive_metadata_observation(observation)


__all__ = [
    "L0_INVENTORY",
    "L1_METADATA",
    "DriveObservationError",
    "M2_STATUS_METADATA",
    "OBSERVATION_DATE",
    "S1_METADATA",
    "S2_COMPLETE_METADATA",
    "build_drive_metadata_observation",
    "validate_drive_metadata_observation",
]
