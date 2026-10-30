"""Shared build, protocol and accounting checks for both shard merge paths."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

_BUILD_KEYS = ("sdk_version", "evaluation_kit_commit")


def require_count(value: Any, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def shard_metadata(shard: Mapping[str, Any]) -> Mapping[str, Any]:
    meta = shard.get("shard")
    if not isinstance(meta, Mapping) or "index" not in meta:
        raise ValueError("a result.json carries no shard metadata; it is not a shard run")
    return meta


def validate_shard_set(shards: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not shards:
        raise ValueError("no shard results to merge")
    for shard in shards:
        for field in ("benchmark", "result", "official_result"):
            if not isinstance(shard.get(field), Mapping):
                raise ValueError(f"shard {field} must be an object")
    metas = [shard_metadata(shard) for shard in shards]
    for meta in metas:
        for key in ("count", "full_total_records", "episodes_per_shard"):
            require_count(meta.get(key), key, minimum=1)
        require_count(meta["index"], "index")
        if type(meta.get("explicit_count")) is not bool:
            raise ValueError("shard explicit_count must be a boolean")
    for key in ("count", "full_total_records", "episodes_per_shard", "explicit_count"):
        values = {meta.get(key) for meta in metas}
        if len(values) != 1:
            raise ValueError(f"shards disagree on {key}: {sorted(map(str, values))}")
    count = metas[0]["count"]
    indexes = [meta["index"] for meta in metas]
    if count != len(shards) or sorted(indexes) != list(range(count)):
        raise ValueError(f"shard indexes {indexes} do not cover 0..{count - 1} exactly once")
    for shard in shards:
        if shard.get("result", {}).get("state") != "complete":
            raise ValueError(f"shard {shard_metadata(shard)['index']} did not complete")
    _require_one_build(metas)
    return {
        key: metas[0][key]
        for key in ("count", "full_total_records", "episodes_per_shard", "explicit_count", "build")
    }


def _require_one_build(metas: Sequence[Mapping[str, Any]]) -> None:
    """Shards that ran different SDK or kit builds may partition or score the workload differently."""

    builds = [meta.get("build") for meta in metas]
    for meta, build in zip(metas, builds, strict=True):
        if not isinstance(build, Mapping) or not all(build.get(key) for key in _BUILD_KEYS):
            raise ValueError(f"shard {meta['index']} does not record the SDK build it ran")
    if any(build != builds[0] for build in builds[1:]):
        detail = "; ".join(
            f"shard {meta['index']} sdk={build['sdk_version']} kit={build['evaluation_kit_commit']}"
            for meta, build in zip(metas, builds, strict=True)
        )
        raise ValueError(f"shards ran different SDK builds ({detail})")


def validate_shard_identity(shards: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    identities = [shard["benchmark"] for shard in shards]
    canonical = [json.dumps(identity, sort_keys=True) for identity in identities]
    if any(identity != canonical[0] for identity in canonical[1:]):
        raise ValueError("shards describe different benchmarks, suites or protocols")
    if not identities[0].get("benchmark"):
        raise ValueError("shard results do not name their benchmark")
    return dict(identities[0])


def validate_shard_accounting(shard: Mapping[str, Any]) -> None:
    """Reject invalid local counts even when corrupt shards cancel each other in the sum."""

    result = shard["result"]
    index = shard_metadata(shard)["index"]
    episodes = require_count(result["episodes"], f"shard {index} episodes", minimum=1)
    total = require_count(result["total_episodes"], f"shard {index} total_episodes", minimum=1)
    successes = require_count(result["successes"], f"shard {index} successes")
    failures = require_count(result["failures"], f"shard {index} failures")
    require_count(result["total_steps"], f"shard {index} total_steps")
    require_count(result["completed_items"], f"shard {index} completed_items", minimum=1)
    if episodes != total or successes + failures != episodes:
        raise ValueError(f"shard {index} partition is not complete or its counts are inconsistent")
