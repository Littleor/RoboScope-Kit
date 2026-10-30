# Copyright 2026 RoboScope Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Model components used by the separately packaged policy HIL runtime."""

from roboscope_inference_sdk.runtime.hil.lerobot_policy import (
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
