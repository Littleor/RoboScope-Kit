# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Framing and safe serialization for the local policy socket.

Two encodings carry the same messages. ``json`` frames are a ``>I`` length and UTF-8 JSON, with arrays and
byte strings base64-encoded inside it. ``msgpack`` frames are the magic ``RSKM``, a ``>I`` length, and a
MessagePack map: byte strings are MessagePack ``bin`` values and an array is the map
``{"__roboscope_type__": "ndarray", "dtype": ..., "shape": [...], "data": <bin>}``. A JSON frame's length is
below :data:`MAX_MESSAGE_BYTES`, so its first byte is never ``R``; a receiver tells the two apart from the
first four bytes and answers in the encoding it received. A client uses MessagePack only after the server's
``health`` reply lists it, so peers without MessagePack keep working over JSON.
"""

from __future__ import annotations

import base64
import json
import socket
import struct
from typing import Any, Literal

import msgpack
import numpy as np

PROTOCOL_VERSION = 1
MAX_MESSAGE_BYTES = 64 * 1024 * 1024
ENCODINGS = ("json", "msgpack")
Encoding = Literal["json", "msgpack"]
_HEADER = struct.Struct(">I")
_MSGPACK_MAGIC = b"RSKM"
_TYPE_KEY = "__roboscope_type__"


class ProtocolError(RuntimeError):
    """Raised when a peer sends an invalid or unsupported message."""


def send_message(connection: socket.socket, message: dict[str, Any], encoding: Encoding = "json") -> None:
    if encoding == "msgpack":
        payload = msgpack.packb(message, default=_msgpack_default, use_bin_type=True)
        prefix = _MSGPACK_MAGIC
    elif encoding == "json":
        payload = json.dumps(_encode(message), separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        prefix = b""
    else:
        raise ValueError(f"unknown protocol encoding {encoding!r}")
    if len(payload) > MAX_MESSAGE_BYTES:
        raise ProtocolError(f"message is {len(payload)} bytes; maximum is {MAX_MESSAGE_BYTES}")
    connection.sendall(prefix + _HEADER.pack(len(payload)) + payload)


def receive_message(connection: socket.socket) -> dict[str, Any] | None:
    received = receive_frame(connection)
    return None if received is None else received[0]


def receive_frame(connection: socket.socket) -> tuple[dict[str, Any], Encoding] | None:
    """Receive one message and the encoding it arrived in; ``None`` when the peer closed cleanly."""

    header = _receive_exact(connection, _HEADER.size)
    if header is None:
        return None
    encoding: Encoding = "json"
    if header == _MSGPACK_MAGIC:
        encoding = "msgpack"
        header = _receive_exact(connection, _HEADER.size)
    if header is None or len(header) != _HEADER.size:
        raise ProtocolError("connection closed inside a message header")
    (size,) = _HEADER.unpack(header)
    if size > MAX_MESSAGE_BYTES:
        raise ProtocolError(f"message declares {size} bytes; maximum is {MAX_MESSAGE_BYTES}")
    payload = _receive_exact(connection, size)
    if payload is None or len(payload) != size:
        raise ProtocolError("connection closed before message body")
    if encoding == "msgpack":
        try:
            decoded = msgpack.unpackb(
                payload, object_hook=_msgpack_object, raw=False, strict_map_key=True,
                max_bin_len=MAX_MESSAGE_BYTES, max_str_len=MAX_MESSAGE_BYTES,
            )
        except ProtocolError:
            raise
        except Exception as exc:
            raise ProtocolError(f"invalid MessagePack message: {exc}") from exc
    else:
        decoded = _decode(json.loads(payload.decode("utf-8")))
    if not isinstance(decoded, dict):
        raise ProtocolError("message root must be an object")
    return decoded, encoding


def validate_version(message: dict[str, Any]) -> None:
    version = message.get("version")
    if version != PROTOCOL_VERSION:
        raise ProtocolError(f"unsupported protocol version {version!r}; expected {PROTOCOL_VERSION}")


def _receive_exact(connection: socket.socket, size: int) -> bytes | None:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = connection.recv(remaining)
        if not chunk:
            return None if remaining == size else b"".join(chunks)
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _msgpack_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError("object arrays are not supported")
        contiguous = np.ascontiguousarray(value)
        return {
            _TYPE_KEY: "ndarray",
            "dtype": contiguous.dtype.str,
            "shape": list(contiguous.shape),
            "data": memoryview(contiguous.reshape(-1).view(np.uint8)),
        }
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"cannot serialize {type(value).__name__} over the policy protocol")


def _msgpack_object(value: dict[str, Any]) -> Any:
    if value.get(_TYPE_KEY) != "ndarray":
        return value
    data = value.get("data")
    if not isinstance(data, bytes):
        raise ProtocolError("array data must be a MessagePack bin value")
    return _array(data, value)


def _array(data: bytes, value: dict[str, Any]) -> np.ndarray:
    dtype = np.dtype(value["dtype"])
    if dtype.hasobject:
        raise ProtocolError("object arrays are not supported")
    shape = tuple(int(dimension) for dimension in value["shape"])
    if any(dimension < 0 for dimension in shape):
        raise ProtocolError("array dimensions cannot be negative")
    expected_size = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
    if len(data) != expected_size:
        raise ProtocolError(
            f"array payload has {len(data)} bytes; shape and dtype require {expected_size}"
        )
    # Copied so the receiver gets a writable array it owns, as with JSON frames.
    return np.frombuffer(data, dtype=dtype).reshape(shape).copy()


def _encode(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        contiguous = np.ascontiguousarray(value)
        return {
            _TYPE_KEY: "ndarray",
            "dtype": contiguous.dtype.str,
            "shape": list(contiguous.shape),
            "data": base64.b64encode(contiguous.tobytes()).decode("ascii"),
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return {_TYPE_KEY: "bytes", "data": base64.b64encode(value).decode("ascii")}
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("protocol object keys must be strings")
        return {key: _encode(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_encode(item) for item in value]
    return value


def _decode(value: Any) -> Any:
    if isinstance(value, list):
        return [_decode(item) for item in value]
    if not isinstance(value, dict):
        return value
    value_type = value.get(_TYPE_KEY)
    if value_type == "bytes":
        return base64.b64decode(value["data"], validate=True)
    if value_type == "ndarray":
        return _array(base64.b64decode(value["data"], validate=True), value)
    return {key: _decode(item) for key, item in value.items()}
