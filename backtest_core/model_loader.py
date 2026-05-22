"""Loading and validating pluggable backtest model files."""

import importlib.util
import logging
import os
import sys
from pathlib import Path
from types import ModuleType

from . import runtime
from .config import MODEL_CONFIG_DIR, MODEL_CONFIG_REQUIRED, MODEL_DIR, MODEL_FILE, MODEL_FILES, MODEL_SELECTION

log = logging.getLogger(__name__)

def _validate_model_filename(model_file: str) -> None:
    model_path = Path(model_file)
    if not model_file or model_path.name != model_file or model_path.suffix != ".py":
        raise ValueError(
            f"Invalid model file {model_file!r}. Use a plain Python filename like pullback_bounce_fundamental_v1.py"
        )


def select_model_files() -> list[str]:
    if MODEL_SELECTION not in {"single", "multi", "all"}:
        raise ValueError("MODEL_SELECTION must be one of: single, multi, all")

    model_dir = Path(MODEL_DIR)
    if MODEL_SELECTION == "single":
        selected = [MODEL_FILE]
    elif MODEL_SELECTION == "multi":
        if not MODEL_FILES:
            raise ValueError("MODEL_SELECTION=multi requires MODEL_FILES with a comma-separated file list")
        selected = MODEL_FILES
    else:
        if not model_dir.is_dir():
            raise FileNotFoundError(f"MODEL_DIR not found: {model_dir}")
        selected = sorted(
            path.name
            for path in model_dir.glob("*.py")
            if not path.name.startswith("_") and path.name != "__init__.py"
        )
        if not selected:
            raise FileNotFoundError(f"MODEL_SELECTION=all found no model files in {model_dir}")

    deduped: list[str] = []
    seen: set[str] = set()
    for model_file in selected:
        _validate_model_filename(model_file)
        if model_file not in seen:
            deduped.append(model_file)
            seen.add(model_file)
    return deduped


def load_model_module(model_file: str) -> ModuleType:
    """Load one configured backtesting model from backtest_models/<model_file>."""
    _validate_model_filename(model_file)
    model_path = Path(model_file)
    full_path = Path(MODEL_DIR) / model_file
    if not full_path.is_file():
        raise FileNotFoundError(f"Backtesting model file not found: {full_path}")

    module_name = f"backtest_model_{model_path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, full_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load backtesting model spec from {full_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    required_attrs = [
        "SignalConfig",
        "signal_config_from_env",
        "compute_long_signal",
        "compute_short_signal",
    ]
    missing = [name for name in required_attrs if not hasattr(module, name)]
    if missing:
        raise RuntimeError(
            f"Backtesting model {model_file} is missing required symbols: {', '.join(missing)}"
        )

    log.info("Loaded backtesting model %s path %s", model_file, full_path)
    return module


def _parse_env_value(raw: str) -> str:
    value = raw.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def load_model_config_env(model_file: str) -> Path | None:
    """Load backtest_model_configs/<model_stem>.env into this worker process."""
    _validate_model_filename(model_file)
    config_path = Path(MODEL_CONFIG_DIR) / f"{Path(model_file).stem}.env"
    if not config_path.is_file():
        if MODEL_CONFIG_REQUIRED:
            raise FileNotFoundError(f"Model config file not found: {config_path}")
        log.info("No model config file found model %s path %s", model_file, config_path)
        return None

    loaded = 0
    for line_no, raw_line in enumerate(config_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            raise ValueError(f"Invalid model config line {config_path}:{line_no}: expected KEY=VALUE")
        key, raw_value = line.split("=", 1)
        key = key.strip()
        if not key or not key.replace("_", "A").isalnum() or key[0].isdigit():
            raise ValueError(f"Invalid model config key {config_path}:{line_no}: {key!r}")
        os.environ[key] = _parse_env_value(raw_value)
        loaded += 1

    log.info("Loaded model config model %s path %s variables %d", model_file, config_path, loaded)
    return config_path


def get_model_module() -> ModuleType:
    if runtime.MODEL_MODULE is None:
        raise RuntimeError("Backtesting model has not been loaded yet")
    return runtime.MODEL_MODULE


def set_model_module(module: ModuleType) -> None:
    runtime.MODEL_MODULE = module
