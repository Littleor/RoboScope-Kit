# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Official CALVIN work manifest and SDK-native evaluator process adapter."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from roboscope_kit.runtime.benchmark import (
    ProcessSpec,
    WorkerContext,
    WorkItem,
    WorkResult,
)
from roboscope_kit.runtime.benchmark.kit_path import kit_import_root

CALVIN_OFFICIAL_SUITES = frozenset({"task_D_D", "task_ABC_D", "task_ABCD_D"})
OFFICIAL_EVAL_SEQUENCE_COUNT = 1000
OFFICIAL_EVAL_SEQUENCES_SHA256 = "c3689911565037d399d779c655e9e943cbd5bfbfbf44045e2b3b1daa8b1a2317"


def summarize_sequence_lengths(lengths: Sequence[int]) -> dict[str, Any]:
    """Derive the official CALVIN metrics from the chained success length of every sequence."""

    count = len(lengths)
    return {
        "num_sequences": count,
        "avg_seq_len": sum(lengths) / count,
        "task_success_rate": {
            f"task_{position}": sum(length >= position for length in lengths) / count
            for position in range(1, 6)
        },
        "success_at_5": sum(length >= 5 for length in lengths) / count,
        "sequence_success_lengths": list(lengths),
    }


def merge_sequence_lengths(partials: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Combine per-shard ``aggregate_partial`` outputs into the full-run official result."""

    if not partials:
        raise ValueError("no CALVIN shard results to merge")
    totals = {int(partial["num_sequences"]) for partial in partials}
    if len(totals) != 1:
        raise ValueError(f"CALVIN shards disagree on num_sequences: {sorted(totals)}")
    num_sequences = totals.pop()
    if num_sequences < 1:
        raise ValueError("CALVIN num_sequences must be positive")
    merged: dict[int, int] = {}
    for partial in partials:
        if partial.get("partial") is not True:
            raise ValueError("CALVIN shard official_result is not a partial aggregate")
        for key, length in partial["sequence_success_lengths_by_index"].items():
            index = int(key)
            if index in merged:
                raise ValueError(f"CALVIN sequence {index} reported by more than one shard")
            if type(length) is not int or not 0 <= length <= 5:
                raise ValueError(f"invalid CALVIN chained success length {length}")
            merged[index] = length
    if sorted(merged) != list(range(num_sequences)):
        raise ValueError(
            f"CALVIN shards cover {len(merged)} of {num_sequences} sequences; "
            "the merged range must be exactly complete"
        )
    return summarize_sequence_lengths([merged[index] for index in range(num_sequences)])


@dataclass(frozen=True)
class CalvinRuntimeConfig:
    python: Path
    calvin_root: Path
    dataset_path: Path
    config_path: Path
    eval_sequences_path: Path
    suite: str
    num_sequences: int = 1000
    chunk_size: int = 1
    cpu_threads: int = 1
    render_backend: str = "egl"

    def __post_init__(self) -> None:
        if self.suite not in CALVIN_OFFICIAL_SUITES:
            raise ValueError(f"unsupported CALVIN suite: {self.suite}")
        if self.num_sequences < 1:
            raise ValueError("CALVIN num_sequences must be positive")
        if self.chunk_size < 1 or self.cpu_threads < 1:
            raise ValueError("CALVIN chunk_size and cpu_threads must be positive")
        if self.render_backend != "egl":
            raise ValueError("CALVIN SDK evaluation currently requires EGL rendering")


class CalvinBenchmarkAdapter:
    """Run the official CALVIN long-horizon protocol against an SDK Policy."""

    def __init__(self, config: CalvinRuntimeConfig) -> None:
        self.config = config
        sequence_bytes = config.eval_sequences_path.read_bytes()
        sequences = json.loads(sequence_bytes)
        self.total_sequences = len(sequences)
        self.eval_sequences_sha256 = hashlib.sha256(sequence_bytes).hexdigest()
        if self.total_sequences != OFFICIAL_EVAL_SEQUENCE_COUNT:
            raise ValueError(
                f"CALVIN manifest contains {self.total_sequences} sequences; "
                f"expected {OFFICIAL_EVAL_SEQUENCE_COUNT}"
            )
        if self.eval_sequences_sha256 != OFFICIAL_EVAL_SEQUENCES_SHA256:
            raise ValueError("CALVIN evaluation manifest does not match the official challenge sequence file")
        if config.num_sequences > self.total_sequences:
            raise ValueError(
                f"CALVIN num_sequences cannot exceed the {self.total_sequences}-sequence manifest"
            )

    def work_items(self) -> Sequence[WorkItem]:
        items: list[WorkItem] = []
        for ordinal, start in enumerate(range(0, self.config.num_sequences, self.config.chunk_size)):
            count = min(self.config.chunk_size, self.config.num_sequences - start)
            items.append(
                WorkItem(
                    ordinal=ordinal,
                    item_id=f"sequences-{start:04d}-{start + count:04d}",
                    group_id=f"sequences-{start:04d}",
                    payload={"start_sequence": start, "num_sequences": count},
                    expected_results=count,
                )
            )
        return items

    def worker_process(self, context: WorkerContext) -> ProcessSpec:
        action_representation = context.execution_contract.action_representation
        command = (
            str(self.config.python),
            "-m",
            "roboscope_kit.runtime.benchmarks.calvin.worker",
            "--worker-id",
            str(context.worker_id),
            "--policy-socket",
            str(context.replica_socket),
            "--dataset-path",
            str(self.config.dataset_path),
            "--asset-root",
            str(self.config.calvin_root / "calvin_env" / "data"),
            "--config-path",
            str(self.config.config_path),
            "--eval-sequences-path",
            str(self.config.eval_sequences_path),
            "--eval-sequences-sha256",
            self.eval_sequences_sha256,
            "--action-representation",
            action_representation,
            "--queue-dir",
            str(context.queue_dir),
            "--worker-state-dir",
            str(context.state_dir),
        )
        threads = str(self.config.cpu_threads)
        env = {
            "PYTHONPATH": ":".join(
                (
                    str(self.config.calvin_root / "calvin_models"),
                    str(self.config.calvin_root / "calvin_env"),
                    str(kit_import_root()),
                    os.environ.get("PYTHONPATH", ""),
                )
            ),
            "CUDA_VISIBLE_DEVICES": context.gpu_device,
            "EGL_VISIBLE_DEVICES": "0",
            "PYOPENGL_PLATFORM": self.config.render_backend,
            "OMP_NUM_THREADS": threads,
            "MKL_NUM_THREADS": threads,
            "OPENBLAS_NUM_THREADS": threads,
            "NUMEXPR_NUM_THREADS": threads,
            "TOKENIZERS_PARALLELISM": "false",
        }
        return ProcessSpec(command=command, env=env)

    def aggregate(self, results: Sequence[WorkResult]) -> dict[str, Any]:
        lengths_by_index = self._sequence_lengths(results)
        expected_indices = list(range(self.config.num_sequences))
        if sorted(lengths_by_index) != expected_indices:
            raise RuntimeError("CALVIN results do not exactly cover the requested sequence range")
        return summarize_sequence_lengths([lengths_by_index[index] for index in expected_indices])

    def aggregate_partial(self, results: Sequence[WorkResult]) -> dict[str, Any]:
        """Report one shard's sequences without the range-coverage check or derived metrics.

        Averages over a subset of sequences are not the official metric, so only the raw per-sequence
        lengths are emitted; ``merge_sequence_lengths`` restores coverage and derives the metrics.
        """

        lengths_by_index = self._sequence_lengths(results)
        return {
            "partial": True,
            "num_sequences": self.config.num_sequences,
            "sequence_success_lengths_by_index": {
                str(index): length for index, length in sorted(lengths_by_index.items())
            },
        }

    @staticmethod
    def _sequence_lengths(results: Sequence[WorkResult]) -> dict[int, int]:
        lengths_by_index: dict[int, int] = {}
        for result in results:
            for record in result.records:
                sequence_index = int(record.metadata["sequence_index"])
                if record.record_id != f"sequence-{sequence_index:04d}":
                    raise RuntimeError(f"unexpected CALVIN record id {record.record_id}")
                if sequence_index in lengths_by_index:
                    raise RuntimeError(f"duplicate CALVIN sequence index {sequence_index}")
                success_length = int(record.metadata["chained_success_length"])
                if not 0 <= success_length <= 5:
                    raise RuntimeError(f"invalid CALVIN chained success length {success_length}")
                if record.success != (success_length == 5):
                    raise RuntimeError(f"inconsistent CALVIN success flag for sequence {sequence_index}")
                lengths_by_index[sequence_index] = success_length
        return lengths_by_index

    def metadata(self) -> dict[str, Any]:
        return {
            "adapter": type(self).__name__,
            "benchmark": "calvin",
            "suite": self.config.suite,
            "protocol": "calvin-long-horizon-1000x5",
            "bound_hint": "gpu_infer",
            "num_sequences": self.config.num_sequences,
            "official_total_sequences": self.total_sequences,
            "chunk_size": self.config.chunk_size,
            "eval_sequences_sha256": self.eval_sequences_sha256,
            "render_backend": self.config.render_backend,
        }


__all__ = [
    "CALVIN_OFFICIAL_SUITES",
    "CalvinBenchmarkAdapter",
    "CalvinRuntimeConfig",
    "merge_sequence_lengths",
    "summarize_sequence_lengths",
]
