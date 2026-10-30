"""Load complete StarVLA checkpoints for the official LIBERO runtime."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import torch
from accelerate import init_empty_weights
from transformers import AutoConfig, PreTrainedModel


def _metadata_only_from_pretrained(
    model_class: type[PreTrainedModel],
    pretrained_model_name_or_path: str,
    *model_args: object,
    **kwargs: object,
) -> PreTrainedModel:
    if model_args:
        raise TypeError("StarVLA base-model construction does not accept positional model arguments")
    config = kwargs.pop("config", None)
    if config is None:
        config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path,
            trust_remote_code=bool(kwargs.pop("trust_remote_code", False)),
        )
    attention = kwargs.pop("attn_implementation", None)
    kwargs.pop("torch_dtype", None)
    if kwargs:
        names = ", ".join(sorted(kwargs))
        raise TypeError(f"unsupported metadata-only model arguments: {names}")
    if attention is not None:
        config._attn_implementation = attention
    return model_class(config)


@contextmanager
def _metadata_only_transformers_models():
    original = PreTrainedModel.__dict__["from_pretrained"]
    PreTrainedModel.from_pretrained = classmethod(_metadata_only_from_pretrained)
    try:
        yield
    finally:
        PreTrainedModel.from_pretrained = original


def _checkpoint_state_dict(checkpoint: Path) -> dict[str, torch.Tensor]:
    if checkpoint.suffix == ".safetensors":
        from safetensors.torch import load_file

        return load_file(str(checkpoint), device="cpu")
    state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(state_dict, dict):
        raise TypeError(f"StarVLA checkpoint must contain a state dict: {checkpoint}")
    return state_dict


def install_full_checkpoint_loader() -> None:
    from starVLA.model.framework.base_framework import baseframework, build_framework
    from starVLA.model.framework.share_tools import dict_to_namespace, read_mode_config

    def from_full_checkpoint(
        _model_class: type[baseframework],
        pretrained_checkpoint: str,
        **kwargs: object,
    ) -> baseframework:
        if kwargs:
            names = ", ".join(sorted(kwargs))
            raise TypeError(f"unsupported StarVLA checkpoint arguments: {names}")
        checkpoint = Path(pretrained_checkpoint)
        model_config, norm_stats = read_mode_config(checkpoint)
        config = dict_to_namespace(model_config)
        config.trainer.pretrained_checkpoint = None

        with _metadata_only_transformers_models(), init_empty_weights():
            framework = build_framework(cfg=config)

        state_dict = _checkpoint_state_dict(checkpoint)
        framework.load_state_dict(state_dict, strict=True, assign=True)
        framework.norm_stats = norm_stats
        return framework

    baseframework.from_pretrained = classmethod(from_full_checkpoint)
