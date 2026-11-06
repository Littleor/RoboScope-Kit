# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""The pinned robomimic v0.1 model-zoo checkpoints served by ``robomimic_policy.py``.

Proficient-human (PH) low-dimensional BC-RNN agents from
https://robomimic.github.io/docs/model_zoo/robomimic_v0.1.html. The URLs are the ones that
page links; the SHA-256 digests were recorded from those downloads. Standard library only,
so ``download_model_zoo.py`` runs with any Python 3.11+.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

MODEL_ZOO_URL = "http://downloads.cs.stanford.edu/downloads/rt_benchmark/model_zoo"


@dataclass(frozen=True)
class ZooCheckpoint:
    task: str
    env_name: str
    filename: str
    sha256: str
    size: int

    @property
    def url(self) -> str:
        return f"{MODEL_ZOO_URL}/{self.task}/bc_rnn/{self.filename}"


CHECKPOINTS = (
    ZooCheckpoint(
        "lift", "Lift", "lift_ph_low_dim_epoch_1000_succ_100.pth",
        "3ee222cab41f78ba27afca7ef6f70f9f21bd3cc0a37bd79c0b2351dbede51605", 7959022,
    ),
    ZooCheckpoint(
        "can", "PickPlaceCan", "can_ph_low_dim_epoch_1150_succ_100.pth",
        "7a0c9b63d38bc5f792dbd3f26f56a04147d11d65e4d9c468c30261f7d79f5e81", 7984622,
    ),
    ZooCheckpoint(
        "square", "NutAssemblySquare", "square_ph_low_dim_epoch_1850_succ_84.pth",
        "f2143ebfd474e694c7eecbf013ab82308fbc4c6e00abaffd53e5c8eba2613301", 7984686,
    ),
    ZooCheckpoint(
        "transport", "TwoArmTransport", "transport_ph_low_dim_epoch_1000_succ_78.pth",
        "9f603cf5b7a8b32e4371ed09ed02457214b378d84ac7f6c9ded6c077610771dc", 8327662,
    ),
    ZooCheckpoint(
        "tool_hang", "ToolHang", "tool_hang_ph_low_dim_epoch_2000_succ_14.pth",
        "43131a1446da695f0a053e820871f45525a9e39819354787f3fd3d1fd435fdb7", 8176686,
    ),
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()
