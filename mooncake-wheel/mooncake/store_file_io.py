from __future__ import annotations

import io
import json
import logging
import os
import posixpath
from importlib import import_module
from typing import Any


LOGGER = logging.getLogger(__name__)

INVALID_PARAMS = -600
FILE_NOT_FOUND = -1100
PERSISTENT_FAIL = -1503

_SUPPORTED_KINDS = {"kv_cache", "rl_checkpoint", "model_weights"}


def _normalize_format(format_name: str, file_name: str) -> str:
    if format_name is None:
        format_name = "auto"

    normalized = format_name.lower().replace("-", "_")
    if normalized == "auto":
        if str(file_name).lower().endswith(".safetensors"):
            return "safetensors"
        return "torch"

    if normalized in {"safetensors", "safe_tensors"}:
        return "safetensors"
    if normalized in {"torch", "pt", "pth", "standard", "standard_file"}:
        return "torch"

    raise ValueError(f"Unsupported file format: {format_name}")


def _normalize_filesystem(filesystem: str | None) -> str:
    if filesystem is None:
        return "auto"

    normalized = filesystem.lower().replace("-", "_")
    if normalized in {"local", "localfs", "posix", "file"}:
        return "file"
    return normalized


def _build_target_url(file_name: os.PathLike[str] | str, filesystem: str | None) -> str:
    target = os.fspath(file_name)
    if not target:
        raise ValueError("file_name must not be empty")

    normalized_fs = _normalize_filesystem(filesystem)
    if normalized_fs in {"auto", "file"} or "://" in target:
        return target

    return f"{normalized_fs}://{target}"


def _open_fs_target(
    file_name: os.PathLike[str] | str,
    filesystem: str | None,
    storage_options: dict[str, Any] | None,
):
    import fsspec

    target_url = _build_target_url(file_name, filesystem)
    options = dict(storage_options or {})
    fs, path = fsspec.core.url_to_fs(target_url, **options)
    return fs, path


def _is_dir_like(path: str) -> bool:
    return path.endswith("/") or not path.lower().endswith(".safetensors")


def _write_bytes(
    file_name: os.PathLike[str] | str,
    payload: bytes,
    filesystem: str | None,
    storage_options: dict[str, Any] | None,
) -> None:
    fs, path = _open_fs_target(file_name, filesystem, storage_options)
    parent_dir = posixpath.dirname(path)
    if parent_dir:
        try:
            fs.makedirs(parent_dir, exist_ok=True)
        except (AttributeError, NotImplementedError, OSError):
            pass

    with fs.open(path, "wb") as handle:
        handle.write(payload)


def _write_json(
    file_name: os.PathLike[str] | str,
    data: dict[str, Any],
    filesystem: str | None,
    storage_options: dict[str, Any] | None,
) -> None:
    _write_bytes(
        file_name,
        json.dumps(data, ensure_ascii=True, sort_keys=True).encode("utf-8"),
        filesystem,
        storage_options,
    )


def _read_bytes(
    file_name: os.PathLike[str] | str,
    filesystem: str | None,
    storage_options: dict[str, Any] | None,
) -> bytes:
    fs, path = _open_fs_target(file_name, filesystem, storage_options)
    with fs.open(path, "rb") as handle:
        return handle.read()


def _serialize_obj(obj: Any, format_name: str, fallback_tensor_name: str) -> bytes:
    if format_name == "safetensors":
        from safetensors.torch import save as safetensors_save

        if isinstance(obj, dict):
            if not obj:
                raise ValueError("Cannot serialize an empty dict with safetensors")
            return safetensors_save(obj)

        return safetensors_save({fallback_tensor_name: obj})

    import torch

    buffer = io.BytesIO()
    torch.save(obj, buffer)
    return buffer.getvalue()


def _deserialize_obj(
    payload: bytes,
    format_name: str,
    tensor_name: str | None,
    fallback_name: str | None,
    map_location: Any,
) -> Any:
    if format_name == "safetensors":
        from safetensors.torch import load as safetensors_load

        loaded_tensors = safetensors_load(payload)
        if tensor_name is None and fallback_name is None:
            return loaded_tensors
        return _pick_tensor_entry(loaded_tensors, tensor_name, fallback_name)

    import torch

    return torch.load(io.BytesIO(payload), map_location=map_location)


def _build_artifact_file_path(
    save_path: str,
    rank: int,
    tp_size: int,
    format_name: str,
) -> str:
    if tp_size <= 1 and save_path.lower().endswith(".safetensors"):
        return save_path

    base = save_path if save_path.endswith("/") else f"{save_path}/"
    if format_name == "safetensors":
        return f"{base}rank_{rank:05d}.safetensors"
    return f"{base}rank_{rank:05d}.pt"


def _validate_kind(kind: str) -> None:
    if kind not in _SUPPORTED_KINDS:
        raise ValueError(f"Unsupported artifact kind: {kind}")


def _save_artifact(
    self,
    kind: str,
    name: str,
    obj: Any,
    save_path: os.PathLike[str] | str,
    *,
    format: str = "auto",
    filesystem: str = "auto",
    storage_options: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    tp_rank: int | None = None,
    tp_size: int = 1,
    use_p2p: str = "auto",
) -> dict[str, Any] | None:
    del use_p2p
    try:
        _validate_kind(kind)
        if not name:
            raise ValueError("name must not be empty")

        target_path = os.fspath(save_path)
        if not target_path:
            raise ValueError("save_path must not be empty")
        if tp_size < 1:
            raise ValueError("tp_size must be >= 1")

        rank = 0 if tp_rank is None else int(tp_rank)
        if rank < 0 or rank >= tp_size:
            raise ValueError("tp_rank must satisfy 0 <= tp_rank < tp_size")

        format_name = _normalize_format(format, target_path)
        artifact_file = _build_artifact_file_path(
            target_path, rank, tp_size, format_name
        )
        payload = _serialize_obj(obj, format_name, name)
        _write_bytes(artifact_file, payload, filesystem, storage_options)

        result = {
            "kind": kind,
            "name": name,
            "save_path": target_path,
            "artifact_file": artifact_file,
            "format": format_name,
            "tp_rank": rank,
            "tp_size": tp_size,
            "status": "ok",
        }
        if metadata:
            result["metadata"] = metadata

        if _is_dir_like(target_path) and tp_size > 1:
            manifest_path = posixpath.join(target_path.rstrip("/"), "manifest.json")
            _write_json(
                manifest_path,
                {
                    "kind": kind,
                    "name": name,
                    "format": format_name,
                    "tp_size": tp_size,
                    "metadata": metadata or {},
                },
                filesystem,
                storage_options,
            )
            result["manifest_path"] = manifest_path

        return result
    except Exception:
        LOGGER.exception("Failed to save artifact kind=%s name=%s", kind, name)
        return None


def _load_artifact(
    self,
    kind: str,
    name: str,
    load_path: os.PathLike[str] | str,
    *,
    format: str = "auto",
    filesystem: str = "auto",
    storage_options: dict[str, Any] | None = None,
    map_location: Any = "cpu",
    tp_rank: int | None = None,
    tp_size: int = 1,
    use_p2p: str = "auto",
):
    del use_p2p
    try:
        _validate_kind(kind)
        if not name:
            raise ValueError("name must not be empty")

        target_path = os.fspath(load_path)
        if not target_path:
            raise ValueError("load_path must not be empty")
        if tp_size < 1:
            raise ValueError("tp_size must be >= 1")

        rank = 0 if tp_rank is None else int(tp_rank)
        if rank < 0 or rank >= tp_size:
            raise ValueError("tp_rank must satisfy 0 <= tp_rank < tp_size")

        format_name = _normalize_format(format, target_path)
        artifact_file = _build_artifact_file_path(
            target_path, rank, tp_size, format_name
        )
        payload = _read_bytes(artifact_file, filesystem, storage_options)
        return _deserialize_obj(payload, format_name, None, None, map_location)
    except Exception:
        LOGGER.exception("Failed to load artifact kind=%s name=%s", kind, name)
        return None


def _serialize_tensor(tensor: Any, format_name: str, tensor_name: str) -> bytes:
    if format_name == "safetensors":
        from safetensors.torch import save as safetensors_save

        return safetensors_save({tensor_name: tensor})

    import torch

    buffer = io.BytesIO()
    torch.save(tensor, buffer)
    return buffer.getvalue()


def _pick_tensor_entry(
    loaded_tensors: dict[str, Any],
    preferred_name: str | None,
    fallback_name: str | None,
) -> Any:
    if not loaded_tensors:
        raise ValueError("No tensors found in the file")

    if preferred_name and preferred_name in loaded_tensors:
        return loaded_tensors[preferred_name]

    if fallback_name and fallback_name in loaded_tensors:
        return loaded_tensors[fallback_name]

    first_name = next(iter(loaded_tensors))
    if preferred_name or fallback_name:
        LOGGER.warning(
            "Tensor entry %s was not found in file; using %s instead",
            preferred_name or fallback_name,
            first_name,
        )
    return loaded_tensors[first_name]


def _deserialize_tensor(
    payload: bytes,
    format_name: str,
    tensor_name: str | None,
    store_key: str | None,
    map_location: Any,
) -> Any:
    if format_name == "safetensors":
        from safetensors.torch import load as safetensors_load

        loaded_tensors = safetensors_load(payload)
        return _pick_tensor_entry(loaded_tensors, tensor_name, store_key)

    import torch

    return torch.load(io.BytesIO(payload), map_location=map_location)


def _save_tensor_to_file(
    self,
    key: str,
    file_name: os.PathLike[str] | str | None = None,
    format: str = "auto",
    filesystem: str = "auto",
    storage_options: dict[str, Any] | None = None,
    tensor_name: str | None = None,
) -> int:
    resolved_file_name = key if file_name is None else file_name
    resolved_tensor_name = key if tensor_name is None else tensor_name

    try:
        tensor = self.get_tensor(key)
        if tensor is None:
            LOGGER.error("Failed to fetch tensor for key: %s", key)
            return FILE_NOT_FOUND

        format_name = _normalize_format(format, os.fspath(resolved_file_name))
        payload = _serialize_tensor(tensor, format_name, resolved_tensor_name)
        _write_bytes(resolved_file_name, payload, filesystem, storage_options)
        return 0
    except FileNotFoundError:
        LOGGER.exception("File path not found while saving tensor for key %s", key)
        return FILE_NOT_FOUND
    except (TypeError, ValueError):
        LOGGER.exception("Invalid parameters while saving tensor for key %s", key)
        return INVALID_PARAMS
    except Exception:
        LOGGER.exception("Failed to persist tensor for key %s", key)
        return PERSISTENT_FAIL


def _load_tensor_from_file(
    self,
    key: str | None = None,
    file_name: os.PathLike[str] | str | None = None,
    format: str = "auto",
    filesystem: str = "auto",
    storage_options: dict[str, Any] | None = None,
    tensor_name: str | None = None,
    map_location: Any = None,
):
    if file_name is None:
        LOGGER.error("file_name must be provided when loading a tensor")
        return None

    target_store_key = os.fspath(file_name) if key is None else key

    try:
        payload = _read_bytes(file_name, filesystem, storage_options)
        format_name = _normalize_format(format, os.fspath(file_name))
        tensor = _deserialize_tensor(
            payload, format_name, tensor_name, target_store_key, map_location
        )
        rc = self.put_tensor(target_store_key, tensor)
        if rc != 0:
            LOGGER.error(
                "Failed to store tensor for key %s, rc=%s", target_store_key, rc
            )
            return None
        return tensor
    except FileNotFoundError:
        LOGGER.exception("Tensor file %s was not found", file_name)
        return None
    except Exception:
        LOGGER.exception("Failed to load tensor from file %s", file_name)
        return None


def _save_tensor_to_safetensor(
    self,
    key: str,
    file_name: os.PathLike[str] | str | None = None,
    filesystem: str = "auto",
    storage_options: dict[str, Any] | None = None,
    tensor_name: str | None = None,
) -> int:
    return _save_tensor_to_file(
        self,
        key,
        file_name=file_name,
        format="safetensors",
        filesystem=filesystem,
        storage_options=storage_options,
        tensor_name=tensor_name,
    )


def _load_tensor_from_safetensor(
    self,
    key: str | None = None,
    file_name: os.PathLike[str] | str | None = None,
    filesystem: str = "auto",
    storage_options: dict[str, Any] | None = None,
    tensor_name: str | None = None,
):
    return _load_tensor_from_file(
        self,
        key=key,
        file_name=file_name,
        format="safetensors",
        filesystem=filesystem,
        storage_options=storage_options,
        tensor_name=tensor_name,
    )


def patch_store_file_io_support() -> None:
    try:
        store_module = import_module("mooncake.store")
    except ModuleNotFoundError:
        return

    store_cls = getattr(store_module, "MooncakeDistributedStore", None)
    if store_cls is None:
        return

    if getattr(store_cls, "_mooncake_file_io_patched", False):
        return

    store_cls.save_artifact = _save_artifact
    store_cls.load_artifact = _load_artifact

    store_cls.save_tensor_to_file = _save_tensor_to_file
    store_cls.load_tensor_from_file = _load_tensor_from_file
    store_cls.save_kv_cache_to_file = _save_tensor_to_file
    store_cls.load_kv_cache_from_file = _load_tensor_from_file
    store_cls.save_tensor_to_safetensor = _save_tensor_to_safetensor
    store_cls.load_tensor_from_safetensor = _load_tensor_from_safetensor
    store_cls._mooncake_file_io_patched = True
