# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Output preservation, retry, and ownership across CLI entry points."""

from __future__ import annotations

import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from roboscope_kit.runtime.benchmark import cli, shard_merge
from roboscope_kit.runtime.benchmark.output import reserve_output_directory


@pytest.fixture
def invocation(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "schema_version": 3,
        "supported_benchmarks": ["libero"],
        "supported_suites": {"libero": ["libero_10"]},
        "protocol_versions": {"libero": "libero-v1"},
        "runtime": {"python": sys.executable, "benchmark_root": str(tmp_path), "config_path": str(manifest)},
        "defaults": {},
        "benchmark_protocols": {},
    }))
    policy = tmp_path / "policy.json"
    policy.write_text(json.dumps({"schema_version": 2, "command": [sys.executable], "cwd": str(tmp_path)}))
    output = tmp_path / "run"
    args = ["--benchmark", "libero", "--suite", "libero_10", "--benchmark-manifest", str(manifest),
            "--policy-config", str(policy), "--checkpoint-source", str(tmp_path), "--output-dir", str(output)]
    monkeypatch.setattr(cli, "detect_gpu_devices", lambda: ("0",))
    # Exercise CLI validation, timing, and output writing without launching GPU processes.
    monkeypatch.setattr(cli, "run_candidate", lambda **kwargs: {"result": {"successes": 1}})
    monkeypatch.setattr(cli, "_autotuned_run", lambda *args, **kwargs: ({"result": {}}, {"k": 1, "g": 1}))
    return args, output


def _snapshot(root):
    return {str(p.relative_to(root)): (p.read_bytes(), p.stat().st_mtime_ns)
            for p in root.rglob("*") if p.is_file()}


@pytest.mark.parametrize("mode", [[], ["--quick-test"], ["--self-test"],
                                   ["--concurrency", "1", "--replicas", "1"],
                                   ["--shard-index", "0", "--shard-count", "1"]])
def test_existing_run_is_preserved(invocation, capsys, mode):
    args, output = invocation
    for name in ("result.json", "errors.jsonl", "events.jsonl", "queue/completed/item.json"):
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)
    before = _snapshot(output)
    assert cli.main(args + mode) == 1
    assert _snapshot(output) == before
    assert not (output / ".run.lock").exists()
    error = capsys.readouterr().err
    assert "Choose a new --output-dir" in error
    assert "Traceback" not in error
    assert len(error.splitlines()) == 1


def test_concurrent_reservation_is_rejected(tmp_path):
    start = threading.Barrier(2)
    rejected = threading.Event()

    def attempt():
        start.wait(timeout=10)
        try:
            with reserve_output_directory(tmp_path):
                assert rejected.wait(timeout=10)
                assert (tmp_path / ".run.lock").exists()
                return "owner"
        except FileExistsError:
            rejected.set()
            return "rejected"

    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(lambda _: attempt(), range(2)))
    assert sorted(results) == ["owner", "rejected"]
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("failure", ["manifest", "suite", "checkpoint", "gpu", "policy"])
def test_preflight_failure_can_be_retried(invocation, monkeypatch, failure):
    args, output = invocation
    bad = args.copy()
    if failure == "suite":
        index = bad.index("--suite")
        del bad[index:index + 2]
    elif failure == "gpu":
        monkeypatch.setattr(cli, "detect_gpu_devices", lambda: ())
    else:
        flag = {"manifest": "--benchmark-manifest", "checkpoint": "--checkpoint-source",
                "policy": "--policy-config"}[failure]
        bad[bad.index(flag) + 1] = str(output.parent / "missing")
    assert cli.main(bad + ["--quick-test"]) == 1
    assert {p.name for p in output.iterdir()} == {"errors.jsonl"}
    previous = (output / "errors.jsonl").read_bytes()
    assert cli.main(bad + ["--quick-test"]) == 1
    assert (output / "errors.jsonl").read_bytes().startswith(previous)
    assert len((output / "errors.jsonl").read_text().splitlines()) == 2
    previous = (output / "errors.jsonl").read_bytes()
    monkeypatch.setattr(cli, "detect_gpu_devices", lambda: ("0",))
    assert cli.main(args + ["--quick-test"]) == 0
    assert (output / "errors.jsonl").read_bytes() == previous
    assert (output / "result.json").is_file()
    assert not (output / ".run.lock").exists()


@pytest.mark.parametrize("mode", [[], ["--quick-test"], ["--self-test"],
                                   ["--concurrency", "1", "--replicas", "1"]])
def test_clean_run_has_no_error_log(invocation, mode):
    args, output = invocation
    assert cli.main(args + mode) == 0
    assert not (output / "errors.jsonl").exists()
    assert not (output / ".run.lock").exists()
    assert (output / ("self-test.json" if "--self-test" in mode else "result.json")).exists()


def test_failed_preflight_cannot_append_to_another_run(invocation):
    args, output = invocation
    with reserve_output_directory(output):
        (output / "errors.jsonl").write_text("owner's log\n")
        before = _snapshot(output)
        args[args.index("--benchmark-manifest") + 1] = str(output / "missing")
        assert cli.main(args) == 1
        assert _snapshot(output) == before
        assert (output / ".run.lock").exists()


@pytest.mark.parametrize("error", [RuntimeError("failed"), KeyboardInterrupt()])
def test_run_failure_releases_lock(invocation, monkeypatch, error):
    args, output = invocation

    def fail(**kwargs):
        raise error

    monkeypatch.setattr(cli, "run_candidate", fail)
    if isinstance(error, KeyboardInterrupt):
        with pytest.raises(KeyboardInterrupt):
            cli.main(args + ["--quick-test"])
    else:
        assert cli.main(args + ["--quick-test"]) == 1
        assert json.loads((output / "errors.jsonl").read_text())["message"] == "failed"
    assert not (output / ".run.lock").exists()
    assert (output / "metrics.json").exists()
    with pytest.raises(FileExistsError), reserve_output_directory(output):
        pytest.fail("partial runs cannot be reused")


@pytest.mark.parametrize("name", [".hidden", "console.log", "result.json"])
def test_unrelated_files_are_preserved(tmp_path, name):
    (tmp_path / name).write_text("keep")
    before = _snapshot(tmp_path)
    with pytest.raises(FileExistsError), reserve_output_directory(tmp_path):
        pytest.fail("occupied directory accepted")
    assert _snapshot(tmp_path) == before


def test_symlinked_error_log_is_not_followed(tmp_path):
    target = tmp_path / "target"
    target.write_text("keep")
    output = tmp_path / "run"
    output.mkdir()
    (output / "errors.jsonl").symlink_to(target)
    with pytest.raises(FileExistsError), reserve_output_directory(output):
        pytest.fail("symlink accepted")
    assert target.read_text() == "keep"
    assert not (output / ".run.lock").exists()


def test_shard_merge_preserves_result_and_releases_lock(tmp_path, monkeypatch, capsys):
    output = tmp_path / "merge"
    args = ["--benchmark", "libero", "--suite", "libero_10", "--benchmark-manifest", "unused",
            "--shard-dir", "unused", "--output-dir", str(output)]
    merged = {"result": {"state": "completed", "episodes": 1, "successes": 1,
                         "total_episodes": 1, "success_rate": 1.0}, "shard": {"count": 1}}
    monkeypatch.setattr(shard_merge, "merge_shards", lambda **kwargs: merged)
    assert shard_merge.main(args) == 0
    assert not (output / "errors.jsonl").exists()
    assert not (output / ".run.lock").exists()
    before = _snapshot(output)
    assert shard_merge.main(args) == 1
    assert _snapshot(output) == before
    assert "Traceback" not in capsys.readouterr().err


def test_shard_merge_failure_can_be_retried(tmp_path, monkeypatch):
    def fail(**kwargs):
        raise ValueError("invalid shards")

    monkeypatch.setattr(shard_merge, "merge_shards", fail)
    args = ["--benchmark", "libero", "--suite", "libero_10", "--benchmark-manifest", "unused",
            "--shard-dir", "unused", "--output-dir", str(tmp_path)]
    assert shard_merge.main(args) == 1
    assert list(tmp_path.iterdir()) == []


def test_metaworld_preflight_does_not_write_config(invocation, monkeypatch):
    args, output = invocation
    manifest_path = cli.Path(args[args.index("--benchmark-manifest") + 1])
    manifest = json.loads(manifest_path.read_text())
    manifest.update(supported_benchmarks=["metaworld"], supported_suites={"metaworld": ["mt50"]},
                    protocol_versions={"metaworld": "metaworld-v1"})
    manifest_path.write_text(json.dumps(manifest))
    args[args.index("--benchmark") + 1] = "metaworld"
    args[args.index("--suite") + 1] = "mt50"
    monkeypatch.setattr(cli, "detect_gpu_devices", lambda: ())
    assert cli.main(args + ["--quick-test"]) == 1
    assert {p.name for p in output.iterdir()} == {"errors.jsonl"}
    monkeypatch.setattr(cli, "detect_gpu_devices", lambda: ("0",))
    assert cli.main(args + ["--quick-test"]) == 0
    assert (output / "evaluation-config.json").is_file()


def test_preexisting_lock_is_never_removed(tmp_path):
    lock = tmp_path / ".run.lock"
    lock.write_text("another process")
    with pytest.raises(FileExistsError), reserve_output_directory(tmp_path):
        pytest.fail("existing lock accepted")
    assert lock.read_text() == "another process"


def test_empty_directory_is_reusable_after_exception(tmp_path):
    with pytest.raises(RuntimeError, match="preflight"), reserve_output_directory(tmp_path):
        raise RuntimeError("preflight")
    assert list(tmp_path.iterdir()) == []
    with reserve_output_directory(tmp_path):
        assert not (tmp_path / "errors.jsonl").exists()
