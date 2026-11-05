# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""NVTX ranges read by Nsight Systems during ``--profile`` runs.

Nothing here does any work unless the process runs under ``nsys``, which sets
``NVTX_INJECTION64_PATH`` to its own injection library. The ranges are written
through that library with ``ctypes``, the way NVTX's header-only C API does, so
profiling needs no package in the simulator environment.
"""

from __future__ import annotations

import ctypes
import os
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from typing import Any

PROFILE_ENV = "ROBOSCOPE_PROFILE"
DOMAIN = "roboscope"
EPISODE_RANGE = "episode"

_INJECTION_ENV = "NVTX_INJECTION64_PATH"
_ETID_CALLBACKS = 1
_ETID_VERSIONINFO = 3
_MODULE_CORE2 = 5
# Callback ids per module (NVTX_CBID_<module>_SIZE); the injection fills every module it knows.
_MODULE_SIZES = {1: 16, 2: 9, 3: 15, 4: 7, _MODULE_CORE2: 16, 6: 7}
_CORE2_RANGE_START = 2
_CORE2_RANGE_END = 3
_CORE2_DOMAIN_CREATE = 12
_NVTX_VERSION = 3
_MESSAGE_ASCII = 1


class _EventAttributes(ctypes.Structure):
    _fields_ = [
        ("version", ctypes.c_uint16),
        ("size", ctypes.c_uint16),
        ("category", ctypes.c_uint32),
        ("color_type", ctypes.c_int32),
        ("color", ctypes.c_uint32),
        ("payload_type", ctypes.c_int32),
        ("reserved0", ctypes.c_int32),
        ("payload", ctypes.c_uint64),
        ("message_type", ctypes.c_int32),
        ("message", ctypes.c_char_p),
    ]


_GetTable = ctypes.CFUNCTYPE(
    ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p)), ctypes.POINTER(ctypes.c_uint)
)
_SetVersion = ctypes.CFUNCTYPE(None, ctypes.c_uint32)
_GetExportTable = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_uint32)


class _Callbacks(ctypes.Structure):
    _fields_ = [("struct_size", ctypes.c_size_t), ("get_module_function_table", _GetTable)]


class _VersionInfo(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_size_t),
        ("version", ctypes.c_uint32),
        ("reserved0", ctypes.c_uint32),
        ("set_injection_nvtx_version", _SetVersion),
    ]


class _Nvtx:
    """The three NVTX calls these ranges need, filled in by the injection library."""

    def __init__(self, library: str) -> None:
        # Like NVTX's globals: per module, an array of pointers to function pointer slots,
        # with null at id 0 and after the last id.
        self._targets = {module: (ctypes.c_void_p * size)() for module, size in _MODULE_SIZES.items()}
        width = ctypes.sizeof(ctypes.c_void_p)
        self._tables = {
            module: (ctypes.c_void_p * (len(targets) + 1))(
                None, *(ctypes.addressof(targets) + i * width for i in range(1, len(targets))), None
            )
            for module, targets in self._targets.items()
        }
        self._get_table = _GetTable(self._module_table)
        self._set_version = _SetVersion(lambda version: None)
        self._callbacks = _Callbacks(ctypes.sizeof(_Callbacks), self._get_table)
        self._version = _VersionInfo(ctypes.sizeof(_VersionInfo), _NVTX_VERSION, 0, self._set_version)
        self._export = _GetExportTable(self._export_table)
        initialize = ctypes.CDLL(library).InitializeInjectionNvtx2
        initialize.restype = ctypes.c_int
        initialize.argtypes = [_GetExportTable]
        if not initialize(self._export):
            raise RuntimeError(f"{library} refused NVTX initialization")
        pointer, uint64 = ctypes.c_void_p, ctypes.c_uint64
        self._range_start = self._function(_CORE2_RANGE_START, uint64, pointer, pointer)
        self._range_end = self._function(_CORE2_RANGE_END, None, pointer, uint64)
        domain_create = self._function(_CORE2_DOMAIN_CREATE, pointer, ctypes.c_char_p)
        self._domain = domain_create(DOMAIN.encode())

    def _export_table(self, table_id: int) -> int | None:
        if table_id == _ETID_CALLBACKS:
            return ctypes.addressof(self._callbacks)
        if table_id == _ETID_VERSIONINFO:
            return ctypes.addressof(self._version)
        return None

    def _module_table(self, module: int, table: Any, size: Any) -> int:
        if module not in self._tables:
            return 0
        table[0] = ctypes.cast(self._tables[module], ctypes.POINTER(ctypes.c_void_p))
        size[0] = _MODULE_SIZES[module]
        return 1

    def _function(self, slot: int, restype: Any, *argtypes: Any) -> Any:
        address = self._targets[_MODULE_CORE2][slot]
        if not address:
            raise RuntimeError(f"NVTX injection did not provide function {slot}")
        return ctypes.CFUNCTYPE(restype, *argtypes)(address)

    def start(self, name: str) -> int:
        message = name.encode()
        attributes = _EventAttributes(
            version=_NVTX_VERSION, size=ctypes.sizeof(_EventAttributes),
            message_type=_MESSAGE_ASCII, message=message,
        )
        return self._range_start(self._domain, ctypes.addressof(attributes))

    def end(self, range_id: int) -> None:
        self._range_end(self._domain, range_id)


def _load() -> _Nvtx | None:
    library = os.environ.get(_INJECTION_ENV)
    if not os.environ.get(PROFILE_ENV) or not library:
        return None
    return _Nvtx(library)


_nvtx = _load()
# True only in a --profile Worker under nsys; callers on the hot path check it first.
ENABLED = _nvtx is not None
_episode: int | None = None


@contextmanager
def _range(name: str) -> Iterator[None]:
    assert _nvtx is not None
    range_id = _nvtx.start(name)
    try:
        yield
    finally:
        _nvtx.end(range_id)


def request(operation: str) -> AbstractContextManager[Any]:
    """Mark one Policy round trip."""
    if _nvtx is None:
        return nullcontext()
    return _range(f"policy.{operation}")


def episode_start() -> None:
    """Open the episode range; an unfinished earlier episode ends here."""
    global _episode
    if _nvtx is None:
        return
    episode_end()
    _episode = _nvtx.start(EPISODE_RANGE)


def episode_end() -> None:
    global _episode
    if _nvtx is not None and _episode is not None:
        _nvtx.end(_episode)
        _episode = None
