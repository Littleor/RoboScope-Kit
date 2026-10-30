# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Structured inference-server metrics and benchmark summaries."""

from __future__ import annotations

import json
import math
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any


class JsonlMetricsRecorder:
    def __init__(self, path: Path | None, replica_id: str) -> None:
        self.path = path
        self.replica_id = replica_id
        self._file = None
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._file = path.open("a", encoding="utf-8")

    def record(self, event_type: str, **values: Any) -> None:
        if self._file is None:
            return
        event = {
            "type": event_type,
            "replica_id": self.replica_id,
            "timestamp_s": time.time(),
            **values,
        }
        self._file.write(json.dumps(event, sort_keys=True) + "\n")
        self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def distribution(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "mean": sum(values) / len(values) if values else None,
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values) if values else None,
    }


def summarize_events(events: Iterable[dict[str, Any]]) -> dict[str, Any]:
    batches = []
    requests = []
    transports = []
    errors = []
    for event in events:
        event_type = event.get("type")
        if event_type == "inference_batch":
            batches.append(event)
        elif event_type == "inference_request":
            requests.append(event)
        elif event_type == "inference_transport":
            transports.append(event)
        elif event_type == "inference_error":
            errors.append(event)

    replicas = sorted({str(event.get("replica_id")) for event in batches + requests})
    batch_sizes = [int(event["batch_size"]) for event in batches]
    max_batch_sizes = [int(event["max_batch_size"]) for event in batches]
    example_count = sum(int(event["example_count"]) for event in batches)
    fill_ratios = [float(event["fill_ratio"]) for event in batches]
    timestamps = [float(event["timestamp_s"]) for event in batches + requests]
    duration_s = max(timestamps) - min(timestamps) if len(timestamps) > 1 else 0.0
    summary = {
        "replicas": replicas,
        "duration_s": duration_s,
        "batch_count": len(batches),
        "request_count": len(requests),
        "error_count": len(errors),
        "example_count": example_count,
        "requests_per_s": len(requests) / duration_s if duration_s > 0 else None,
        "examples_per_s": example_count / duration_s if duration_s > 0 else None,
        "batch_size_counts": {str(size): batch_sizes.count(size) for size in sorted(set(batch_sizes))},
        "batch_fill_ratio": distribution(fill_ratios),
        "queue_wait_ms": distribution([float(event["queue_wait_ms"]) for event in requests]),
        "batch_build_wait_ms": distribution([float(event["batch_build_wait_ms"]) for event in batches]),
        "queue_depth_at_first": distribution([float(event["queue_depth_at_first"]) for event in batches]),
        "queue_depth_at_dispatch": distribution(
            [float(event["queue_depth_at_dispatch"]) for event in batches]
        ),
        "predict_ms": distribution([float(event["predict_ms"]) for event in batches]),
        "response_split_ms": distribution([float(event["response_split_ms"]) for event in batches]),
        "request_total_ms": distribution([float(event["request_total_ms"]) for event in requests]),
        "unpack_ms": distribution([float(event["unpack_ms"]) for event in transports]),
        "pack_ms": distribution([float(event["pack_ms"]) for event in transports]),
        "send_ms": distribution([float(event["send_ms"]) for event in transports]),
        "observed_max_batch_size": max(max_batch_sizes, default=None),
        "preprocess_ms": None,
        "preprocess_available": False,
    }
    summary["per_replica"] = {replica: _replica_summary(replica, batches, requests) for replica in replicas}
    return summary


def _replica_summary(
    replica: str,
    batches: list[dict[str, Any]],
    requests: list[dict[str, Any]],
) -> dict[str, Any]:
    replica_batches = [event for event in batches if str(event.get("replica_id")) == replica]
    replica_requests = [event for event in requests if str(event.get("replica_id")) == replica]
    timestamps = [float(event["timestamp_s"]) for event in replica_batches + replica_requests]
    duration_s = max(timestamps) - min(timestamps) if len(timestamps) > 1 else 0.0
    examples = sum(int(event["example_count"]) for event in replica_batches)
    return {
        "batch_count": len(replica_batches),
        "request_count": len(replica_requests),
        "example_count": examples,
        "requests_per_s": len(replica_requests) / duration_s if duration_s > 0 else None,
        "examples_per_s": examples / duration_s if duration_s > 0 else None,
        "queue_wait_ms": distribution([float(event["queue_wait_ms"]) for event in replica_requests]),
        "predict_ms": distribution([float(event["predict_ms"]) for event in replica_batches]),
        "batch_fill_ratio": distribution([float(event["fill_ratio"]) for event in replica_batches]),
    }


def read_jsonl_since(path: Path, offset: int = 0) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    events = []
    with path.open("r", encoding="utf-8") as file:
        file.seek(offset)
        for line in file:
            if line.strip():
                events.append(json.loads(line))
    return events
