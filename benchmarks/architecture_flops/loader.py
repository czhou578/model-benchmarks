"""Find and load the model's config.json: model reference, local file, AutoConfig."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from benchmarks.architecture_flops.config import (
    ConfigLoadResult,
    _as_dict,
    detect_features,
    missing_required_fields,
    normalize_config,
)


def _model_reference(benchmark_config: Mapping[str, Any]) -> str | None:
    for key in ("local_model_path", "model_path", "model_id"):
        value = benchmark_config.get(key)
        if isinstance(value, str) and value:
            return value
    endpoint = benchmark_config.get("endpoint")
    if isinstance(endpoint, Mapping):
        value = endpoint.get("model_name")
        if isinstance(value, str) and value:
            return value
    server = benchmark_config.get("server")
    if isinstance(server, Mapping):
        command = server.get("command")
        if isinstance(command, list):
            try:
                serve_index = command.index("serve")
                value = command[serve_index + 1]
                if isinstance(value, str) and value:
                    return value
            except (ValueError, IndexError):
                pass
    return None


def _local_config_path(benchmark_config: Mapping[str, Any]) -> Path | None:
    explicit = benchmark_config.get("architecture_config_path")
    if isinstance(explicit, str) and explicit:
        path = Path(explicit).expanduser()
        config_dir = benchmark_config.get("_config_dir")
        if not path.is_absolute() and isinstance(config_dir, str):
            path = Path(config_dir) / path
        return path / "config.json" if path.is_dir() else path
    reference = _model_reference(benchmark_config)
    if reference:
        path = Path(reference).expanduser()
        if path.exists():
            return path / "config.json" if path.is_dir() else path
    return None


def _load_local(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"model config at {path} is not a JSON object")
    return value


def _load_auto_config(model_reference: str, trust_remote_code: bool) -> Any:
    try:
        from transformers import AutoConfig
    except ImportError as exc:
        raise RuntimeError(
            "transformers is not installed; install it or set "
            "architecture_config_path to a local config.json"
        ) from exc
    return AutoConfig.from_pretrained(
        model_reference, trust_remote_code=trust_remote_code
    )


def load_architecture_config(
    benchmark_config: Mapping[str, Any],
    *,
    auto_config_loader: Callable[[str], Any] | None = None,
) -> ConfigLoadResult:
    """Load the full architecture config from the first source that fits.

    The sources, in order: a local config file, then Hugging Face (only
    when the local config is not complete). The first complete config wins.
    Without one, the result has the status ``unsupported``. The Hugging Face
    lookup runs remote code only when the benchmark config sets
    ``architecture_trust_remote_code: true``.
    """
    warnings: list[str] = []
    candidates: list[tuple[str, dict[str, Any]]] = []

    local_path = _local_config_path(benchmark_config)
    if local_path is not None:
        try:
            candidates.append((f"local:{local_path}", _load_local(local_path)))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            warnings.append(f"could not load local model config {local_path}: {exc}")

    model_reference = _model_reference(benchmark_config)
    local_complete = False
    if candidates:
        first_norm = normalize_config(candidates[0][1], config_source=candidates[0][0])
        local_complete = not missing_required_fields(first_norm)
    if model_reference and not local_complete:
        trust_remote_code = benchmark_config.get("architecture_trust_remote_code") is True
        try:
            if auto_config_loader is None:
                loaded = _load_auto_config(model_reference, trust_remote_code)
            else:
                loaded = auto_config_loader(model_reference)
            candidates.append(("huggingface", _as_dict(loaded)))
        except Exception as exc:  # dependency, connectivity, or remote config error
            warnings.append(
                f"could not load Hugging Face config for {model_reference}: {exc}"
            )

    for source, raw in candidates:
        normalized = normalize_config(raw, config_source=source)
        missing = missing_required_fields(normalized)
        if not missing:
            features = detect_features(normalized)
            return ConfigLoadResult(
                status="exact_from_config",
                config_source=source,
                raw_config=raw,
                normalized_config=normalized,
                features=features,
                warnings=tuple(warnings),
            )
        warnings.append(f"{source} config is incomplete: missing {', '.join(missing)}")

    return ConfigLoadResult(
        status="unsupported",
        config_source=None,
        raw_config=None,
        normalized_config=None,
        features=None,
        warnings=tuple(warnings or ("no model configuration source was available",)),
    )
