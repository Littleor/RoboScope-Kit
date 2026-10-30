# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Model components used by the separately packaged policy HIL runtime."""

from roboscope_kit.runtime.hil.lerobot_policy import (
    LoadedPi05Runtime,
    Pi05RuntimeConfig,
    load_pi05_runtime,
    validate_lerobot_runtime,
)

HIL_CONTRACT_VERSION = "policy_hil_runtime/v1"
HIL_RUNTIME_VERSION = "0.6.5"

__all__ = [
    "HIL_CONTRACT_VERSION",
    "HIL_RUNTIME_VERSION",
    "LoadedPi05Runtime",
    "Pi05RuntimeConfig",
    "load_pi05_runtime",
    "validate_lerobot_runtime",
]
