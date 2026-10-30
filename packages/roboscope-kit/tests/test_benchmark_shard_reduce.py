# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Sharded runs must reduce to exactly what the unsharded run reports.

The invariant for every benchmark: run the workload once whole, run it again split into N
shards, merge the shard ``result.json`` documents, and the official result must be identical.
The shard documents here are built the way the orchestrator builds them (``aggregate_partial``
when the adapter has one, ``aggregate`` otherwise), so a shard-mode change in an adapter that
breaks mergeability fails these tests.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Sequence
from types import SimpleNamespace
from typing import Any

import pytest

from roboscope_kit.runtime.benchmark.orchestrator import aggregate_standard_results
from roboscope_kit.runtime.benchmark.schema import EvaluationRecord, WorkItem, WorkResult
from roboscope_kit.runtime.benchmark.shard_reduce import merge_shard_results
from roboscope_kit.runtime.benchmark.sharding import select_shard_items, shard_info
from roboscope_kit.runtime.benchmarks.calvin import adapter as calvin
from roboscope_kit.runtime.benchmarks.libero import adapter as libero
from roboscope_kit.runtime.benchmarks.libero_plus import adapter as libero_plus
from roboscope_kit.runtime.benchmarks.metaworld import adapter as metaworld
from roboscope_kit.runtime.benchmarks.robocasa import adapter as robocasa
from roboscope_kit.runtime.benchmarks.robotwin import adapter as robotwin

SHARD_COUNTS = (1, 2, 3, 4)

Workload = tuple[Any, str, list[WorkItem], Callable[[WorkItem], WorkResult]]


def _record(record_id: str, success: bool, **metadata: Any) -> EvaluationRecord:
    # Binary-exact floats so per-shard sums equal the whole-run sum bit for bit.
    return EvaluationRecord(record_id, success, steps=10, elapsed_s=0.25, metadata=metadata)


def _item(ordinal: int, group_id: str, expected: int, **payload: Any) -> WorkItem:
    return WorkItem(
        ordinal=ordinal,
        item_id=f"item-{ordinal:04d}",
        group_id=group_id,
        payload=payload,
        expected_results=expected,
    )


def _libero_like(module: Any, name: str) -> Workload:
    adapter = object.__new__(
        libero.LiberoBenchmarkAdapter if module is libero else libero_plus.LiberoPlusBenchmarkAdapter
    )
    adapter.task_ids = (0, 1, 2)
    # Interleaved like the real adapter, so one task's episodes land on several shards.
    items = [
        _item(chunk * 3 + task, f"task-{task}", 2, task_id=task, chunk=chunk)
        for chunk in range(4)
        for task in range(3)
    ]

    def result(item: WorkItem) -> WorkResult:
        task, chunk = item.payload["task_id"], item.payload["chunk"]
        records = tuple(_record(f"t{task}-c{chunk}-{n}", (task + chunk + n) % 3 != 0) for n in range(2))
        meta = {
            "task_id": task,
            "task_description": f"task {task}",
            "inference_wait_s": 0.5 * (task + 1),
        }
        return WorkResult(item.item_id, item.group_id, 0, records, meta)

    return adapter, name, items, result


def _robotwin_like(cls: Any, name: str, metadata_key: str) -> Workload:
    tasks = ("alpha", "beta", "gamma")
    adapter = object.__new__(cls)
    adapter.config = SimpleNamespace(
        tasks=tasks if name == "robotwin" else tuple((task, f"env-{task}") for task in tasks)
    )
    items = [_item(index, task, 3, task=task) for index, task in enumerate(tasks * 2)]

    def result(item: WorkItem) -> WorkResult:
        task = item.payload["task"]
        records = tuple(_record(f"{item.item_id}-{n}", (item.ordinal + n) % 2 == 0) for n in range(3))
        return WorkResult(item.item_id, item.group_id, 0, records, {metadata_key: task})

    return adapter, name, items, result


def _calvin(chunk_size: int) -> Workload:
    adapter = object.__new__(calvin.CalvinBenchmarkAdapter)
    adapter.config = SimpleNamespace(num_sequences=20, chunk_size=chunk_size)
    items = list(adapter.work_items())

    def result(item: WorkItem) -> WorkResult:
        start = item.payload["start_sequence"]
        count = item.payload["num_sequences"]
        records = []
        for index in range(start, start + count):
            length = (index * 7) % 6  # chained success length, 0..5
            records.append(
                _record(
                    f"sequence-{index:04d}",
                    length == 5,
                    sequence_index=index,
                    chained_success_length=length,
                )
            )
        return WorkResult(item.item_id, item.group_id, 0, tuple(records))

    return adapter, "calvin", items, result


def _metaworld(formal: bool) -> Workload:
    tasks = tuple(metaworld.MT50_TASKS) if formal else tuple(metaworld.MT50_TASKS)[:7]
    per_task = metaworld.MT50_GOALS_PER_TASK if formal else 4
    adapter = object.__new__(metaworld.MetaworldBenchmarkAdapter)
    adapter.config = SimpleNamespace(
        tasks=tasks, episodes_per_task=per_task, formal=formal, episode_horizon=500
    )
    items = list(adapter.work_items())

    def result(item: WorkItem) -> WorkResult:
        slug = item.group_id
        bias = len(slug) % 3
        records = tuple(
            _record(f"{slug}-goal-{goal:02d}", (goal + bias) % 3 != 0, goal_id=goal)
            for goal in range(per_task)
        )
        rate = sum(record.success for record in records) / per_task
        return WorkResult(item.item_id, slug, 0, records, {"upstream_task_success_rate": rate})

    return adapter, "metaworld", items, result


WORKLOADS: dict[str, Callable[[], Workload]] = {
    "libero": lambda: _libero_like(libero, "libero"),
    "libero_plus": lambda: _libero_like(libero_plus, "libero_plus"),
    "robotwin": lambda: _robotwin_like(robotwin.RoboTwinBenchmarkAdapter, "robotwin", "task"),
    "robocasa": lambda: _robotwin_like(robocasa.RoboCasaBenchmarkAdapter, "robocasa", "task_name"),
    "calvin-chunk1": lambda: _calvin(1),
    "calvin-chunk4": lambda: _calvin(4),
    "metaworld-smoke": lambda: _metaworld(False),
    "metaworld-formal": lambda: _metaworld(True),
}


def _shard_document(workload: Workload, index: int, count: int, *, suite: str = "suite-a") -> dict[str, Any]:
    adapter, name, items, make_result = workload
    selected = select_shard_items(items, index, count)
    results = [make_result(item) for item in selected]
    total_records = sum(item.expected_results for item in selected)
    partial = getattr(adapter, "aggregate_partial", None)
    official = partial(results) if partial is not None else adapter.aggregate(results)
    standard = aggregate_standard_results(results, total_records)
    standard["elapsed_s"] = 12.5
    return {
        "sdk_version": 1,
        "shard": shard_info(index, count, sum(item.expected_results for item in items), 500, True, BUILD),
        "benchmark": {
            "adapter": type(adapter).__name__,
            "benchmark": name,
            "suite": suite,
            "protocol": "test-protocol",
        },
        "result": standard,
        "official_result": official,
    }


BUILD = {"sdk_version": "0.7.0b47", "evaluation_kit_commit": "kit-a"}


def _whole_run(workload: Workload) -> dict[str, Any]:
    adapter, _, items, make_result = workload
    return adapter.aggregate([make_result(item) for item in items])


def _documents(workload: Workload, count: int) -> list[dict[str, Any]]:
    return [_shard_document(workload, index, count) for index in range(count)]


@pytest.mark.parametrize("name", WORKLOADS)
@pytest.mark.parametrize("count", SHARD_COUNTS)
def test_merged_official_result_equals_the_unsharded_run(name: str, count: int) -> None:
    workload = WORKLOADS[name]()
    if count > len(workload[2]):
        pytest.skip("more shards than work items")
    merged = merge_shard_results(_documents(workload, count))
    assert merged["official_result"] == _whole_run(workload)


@pytest.mark.parametrize("name", WORKLOADS)
def test_merged_counts_equal_the_whole_workload(name: str) -> None:
    workload = WORKLOADS[name]()
    _, _, items, make_result = workload
    everything = [make_result(item) for item in items]
    whole = aggregate_standard_results(everything, sum(item.expected_results for item in items))
    merged = merge_shard_results(_documents(workload, 3))["result"]
    for key in (
        "episodes",
        "successes",
        "failures",
        "total_steps",
        "total_episodes",
        "success_rate",
        "completed_items",
        "state",
    ):
        assert merged[key] == whole[key], key
    assert "elapsed_s" not in merged and "steps_per_s" not in merged


@pytest.mark.parametrize("name", ["calvin-chunk1", "metaworld-smoke", "metaworld-formal"])
def test_shard_mode_emits_partials_not_final_metrics(name: str) -> None:
    """A subset of sequences or tasks has no official metric; only raw partials may be reported."""

    document = _shard_document(WORKLOADS[name](), 0, 2)
    official = document["official_result"]
    assert official["partial"] is True
    assert not {"avg_seq_len", "success_at_5", "score", "difficulty_rates"} & official.keys()


@pytest.mark.parametrize("name", ["calvin-chunk1", "metaworld-smoke", "metaworld-formal"])
def test_merge_rejects_an_unsharded_official_result(name: str) -> None:
    workload = WORKLOADS[name]()
    documents = _documents(workload, 2)
    documents[0]["official_result"] = _whole_run(workload)
    with pytest.raises(ValueError, match="not a partial aggregate"):
        merge_shard_results(documents)


# ---- shard-set validation -------------------------------------------------------------------


def _libero_documents(count: int = 3) -> list[dict[str, Any]]:
    return _documents(WORKLOADS["libero"](), count)


def test_missing_shard_is_rejected() -> None:
    documents = _libero_documents()
    with pytest.raises(ValueError, match="do not cover 0..2"):
        merge_shard_results(documents[:2])


def test_duplicated_shard_is_rejected() -> None:
    documents = _libero_documents()
    with pytest.raises(ValueError, match="do not cover 0..2"):
        merge_shard_results([documents[0], documents[1], documents[1]])


def test_shards_of_different_sdk_builds_are_rejected_and_named() -> None:
    documents = _libero_documents()
    documents[1]["shard"]["build"] = {"sdk_version": "0.7.0b47", "evaluation_kit_commit": "kit-b"}
    with pytest.raises(ValueError, match="different SDK builds") as raised:
        merge_shard_results(documents)
    assert "shard 0 sdk=0.7.0b47 kit=kit-a" in str(raised.value)
    assert "shard 1 sdk=0.7.0b47 kit=kit-b" in str(raised.value)


@pytest.mark.parametrize("build", [None, {"sdk_version": "0.7.0b47", "evaluation_kit_commit": ""}])
def test_a_shard_that_does_not_record_its_build_is_rejected(build: dict[str, str] | None) -> None:
    documents = _libero_documents()
    if build is None:
        del documents[2]["shard"]["build"]
    else:
        documents[2]["shard"]["build"] = build
    with pytest.raises(ValueError, match="shard 2 does not record the SDK build"):
        merge_shard_results(documents)


def test_the_merged_result_names_the_build_its_shards_ran() -> None:
    assert merge_shard_results(_libero_documents())["shard"]["build"] == BUILD


def test_shard_count_disagreement_is_rejected() -> None:
    documents = _libero_documents()
    documents[1]["shard"]["count"] = 4
    with pytest.raises(ValueError, match="disagree on count"):
        merge_shard_results(documents)


def test_record_total_that_does_not_add_up_is_rejected() -> None:
    documents = _libero_documents()
    documents[2]["result"]["episodes"] -= 1
    with pytest.raises(ValueError, match="partition is not complete"):
        merge_shard_results(documents)


def test_shard_that_ran_the_whole_workload_is_rejected() -> None:
    """A stale image ignoring the shard flags reports every record; the totals expose it."""

    documents = _libero_documents(2)
    workload = WORKLOADS["libero"]()
    everything = _shard_document(workload, 0, 1)
    documents[0]["result"] = everything["result"]
    with pytest.raises(ValueError, match="partition is not complete"):
        merge_shard_results(documents)


def test_incomplete_shard_is_rejected() -> None:
    documents = _libero_documents()
    documents[1]["result"]["state"] = "running"
    with pytest.raises(ValueError, match="did not complete"):
        merge_shard_results(documents)


def test_shards_of_different_suites_are_rejected() -> None:
    workload = WORKLOADS["libero"]()
    documents = [
        _shard_document(workload, 0, 2, suite="suite-a"),
        _shard_document(workload, 1, 2, suite="suite-b"),
    ]
    with pytest.raises(ValueError, match="different benchmarks, suites or protocols"):
        merge_shard_results(documents)


def test_a_result_without_shard_metadata_is_rejected() -> None:
    documents = _libero_documents(2)
    del documents[0]["shard"]
    with pytest.raises(ValueError, match="not a shard run"):
        merge_shard_results(documents)


def test_unknown_benchmark_is_rejected() -> None:
    documents = _libero_documents(2)
    for document in documents:
        document["benchmark"]["benchmark"] = "mystery"
    with pytest.raises(ValueError, match="no shard reducer"):
        merge_shard_results(documents)


def test_empty_input_is_rejected() -> None:
    with pytest.raises(ValueError, match="no shard results"):
        merge_shard_results([])


def test_input_documents_are_not_mutated() -> None:
    documents = _libero_documents()
    before = copy.deepcopy(documents)
    merge_shard_results(documents)
    assert documents == before


# ---- benchmark-specific coverage -------------------------------------------------------------


def test_calvin_sequence_reported_twice_is_rejected() -> None:
    documents = _documents(WORKLOADS["calvin-chunk1"](), 2)
    lengths = documents[1]["official_result"]["sequence_success_lengths_by_index"]
    lengths["0"] = documents[0]["official_result"]["sequence_success_lengths_by_index"]["0"]
    with pytest.raises(ValueError, match="more than one shard"):
        merge_shard_results(documents)


def test_calvin_missing_sequence_is_rejected() -> None:
    documents = _documents(WORKLOADS["calvin-chunk1"](), 2)
    del documents[0]["official_result"]["sequence_success_lengths_by_index"]["0"]
    with pytest.raises(ValueError, match="exactly complete"):
        merge_shard_results(documents)


def test_metaworld_task_reported_twice_is_rejected() -> None:
    documents = _documents(WORKLOADS["metaworld-smoke"](), 2)
    slug, task = next(iter(documents[0]["official_result"]["tasks"].items()))
    documents[1]["official_result"]["tasks"][slug] = task
    with pytest.raises(ValueError, match="more than one shard"):
        merge_shard_results(documents)


def test_metaworld_missing_task_is_rejected() -> None:
    documents = _documents(WORKLOADS["metaworld-smoke"](), 2)
    slug = next(iter(documents[0]["official_result"]["tasks"]))
    del documents[0]["official_result"]["tasks"][slug]
    with pytest.raises(ValueError, match="do not cover tasks"):
        merge_shard_results(documents)


def test_additive_shards_listing_different_tasks_are_rejected() -> None:
    documents = _documents(WORKLOADS["robotwin"](), 2)
    documents[1]["official_result"]["tasks"].pop()
    with pytest.raises(ValueError, match="different task sets"):
        merge_shard_results(documents)


# ---- the aggregate() failures the LIBERO pilot could not reach --------------------------------


def _whole_and_partial_agree(adapter: Any, results: Sequence[WorkResult]) -> None:
    assert adapter.aggregate_partial(results)["partial"] is True


@pytest.mark.parametrize("name", ["calvin-chunk1", "metaworld-smoke", "metaworld-formal"])
def test_full_aggregate_still_demands_complete_coverage(name: str) -> None:
    """Relaxing the shard path must not relax the single-instance guarantee."""

    adapter, _, items, make_result = WORKLOADS[name]()
    subset = [make_result(item) for item in select_shard_items(items, 0, 2)]
    with pytest.raises(RuntimeError):
        adapter.aggregate(subset)
    _whole_and_partial_agree(adapter, subset)
