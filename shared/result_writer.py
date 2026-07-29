"""Persistência atômica dos resultados de benchmark."""

import datetime
import json
import os
import re
import tempfile
from typing import Any


def _safe_prefix(value: str) -> str:
    prefix = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(value)).strip("._-")
    return prefix[:120] or "benchmark"


def save_result(output_dir: str, prefix: str, result: Any) -> str:
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    target = os.path.join(output_dir, f"{_safe_prefix(prefix)}_{timestamp}.json")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=output_dir,
                prefix=".result-", suffix=".tmp", delete=False) as handle:
            temporary = handle.name
            json.dump(result, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o644)
        os.replace(temporary, target)
        return target
    except Exception:
        if temporary:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
        raise
