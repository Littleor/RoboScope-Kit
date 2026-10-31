"""StarVLA LIBERO Policy used for the LIBERO-Long results in the RoboScope Kit technical report.

It follows the StarVLA LIBERO client at revision eaa51c4: OpenCV INTER_AREA resizing, normalized actions
clipped to [-1, 1], the gripper binarized before min/max unnormalization, and float32 actions at the
Policy boundary.
"""

from __future__ import annotations

import numpy as np
from starvla_policy import StarVLAPolicy, _build_example, _resize_image_with_opencv

from roboscope_kit import ActionChunk, Observation, PolicyCapabilities, SessionContext, run


def decode_actions(normalized: np.ndarray, statistics: dict) -> np.ndarray:
    mask = np.asarray(statistics.get("mask", np.ones_like(statistics["min"], dtype=bool)))
    high, low = np.asarray(statistics["max"]), np.asarray(statistics["min"])
    values = np.clip(normalized, -1, 1)
    values[:, 6] = np.where(values[:, 6] < 0.5, 0, 1)
    return np.where(mask, 0.5 * (values + 1) * (high - low) + low, values)


class StarVLALiberoPolicy(StarVLAPolicy):
    def load(self, context: SessionContext) -> PolicyCapabilities:
        if context.benchmark.name != "libero":
            raise ValueError("StarVLALiberoPolicy serves LIBERO only")
        capabilities = super().load(context)
        config = self.config
        if config.action_dim != 7 or config.action_horizon != 8 or config.unnorm_key != "franka":
            raise ValueError("StarVLALiberoPolicy expects a 7-D, 8-step franka checkpoint")
        self.image_preprocessor = _resize_image_with_opencv
        return capabilities

    def act(self, observation: Observation) -> ActionChunk:
        if self.framework is None or self.config is None or self.image_preprocessor is None:
            raise RuntimeError("StarVLA LIBERO Policy is not loaded")
        import torch

        example = _build_example(
            observation.features, self.config.inputs, self.norm_processor, self.image_preprocessor
        )
        with torch.inference_mode():
            output = self.framework.predict_action(examples=[example], **self.config.kwargs)
        normalized = np.asarray(output["normalized_actions"])
        if normalized.shape != (1, 8, 7):
            raise ValueError(f"StarVLA returned action shape {normalized.shape}; expected (1, 8, 7)")
        actions = decode_actions(normalized[0], self.framework.norm_stats["franka"]["action"])
        actions = np.array(actions, copy=True, order="C")
        for transform in self.config.transforms:
            transform.apply(actions)
        if not np.isfinite(actions).all():
            raise ValueError("StarVLA returned non-finite actions")
        return ActionChunk(actions.astype(np.float32))


if __name__ == "__main__":
    run(StarVLALiberoPolicy())
