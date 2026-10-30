"""Keep package metadata and CI aligned with the CPython support policy."""

from __future__ import annotations

import argparse
import json
import re
import tomllib
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIFECYCLE_URL = "https://peps.python.org/api/release-cycle.json"


def version_key(version: str) -> tuple[int, ...]:
    return tuple(map(int, version.split(".")))


def read_policy():
    project = tomllib.loads(
        (ROOT / "packages/roboscope-kit/pyproject.toml").read_text(encoding="utf-8")
    )
    return project, project["tool"]["roboscope"]["python-support"]


def check_policy(project, policy):
    maintained = policy["maintained"]
    versions = maintained
    if not versions or len(set(versions)) != len(versions):
        raise ValueError("maintained versions must be non-empty and unique")
    if "legacy" in policy:
        raise ValueError("legacy Python support is no longer part of the SDK policy")
    if project["project"]["requires-python"] != ">=3.11":
        raise ValueError("the package minimum must be Python 3.11")
    if min(map(version_key, versions)) != (3, 11):
        raise ValueError("the interpreter matrix must start at Python 3.11")
    if project["tool"]["ruff"]["target-version"] != "py311":
        raise ValueError("the shared SDK syntax baseline must be Python 3.11")
    classifiers = {f"Programming Language :: Python :: {version}" for version in versions}
    declared = {
        classifier for classifier in project["project"]["classifiers"]
        if re.fullmatch(r"Programming Language :: Python :: \d+\.\d+", classifier)
    }
    if classifiers != declared:
        raise ValueError("Python classifiers differ from the declared CI/support matrix")
    with urllib.request.urlopen(LIFECYCLE_URL, timeout=30) as response:
        lifecycle = json.load(response)
    upstream = {
        version for version, details in lifecycle.items()
        if details["status"] in {"bugfix", "security"}
    }
    if not upstream:
        raise ValueError("could not read the official CPython lifecycle table")
    if upstream != set(maintained):
        raise ValueError(
            f"CPython lifecycle changed: upstream={sorted(upstream, key=version_key)}, "
            f"declared={maintained}; review package metadata, dependencies and CI together"
        )
    print(f"Maintained: {maintained}; minimum: Python 3.11")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "matrix"))
    args = parser.parse_args()
    project, policy = read_policy()
    if args.command == "check":
        check_policy(project, policy)
    else:
        print(json.dumps({"include": [
            {"python": version, "support": "maintained"}
            for version in policy["maintained"]
        ]}))


if __name__ == "__main__":
    main()
