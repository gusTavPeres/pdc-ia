"""Download e cache de artefatos de modelo."""

import fcntl
import hashlib
import os
import time
from urllib.parse import urlparse

try:
    import requests
    _REQUESTS = True
except ImportError:
    _REQUESTS = False


def _safe_filename(value: str) -> str:
    filename = os.path.basename(str(value or "").strip())
    if not filename or filename in (".", ".."):
        raise ValueError("nome de arquivo do modelo inválido")
    return filename


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _valid_cache(path: str, expected_sha256: str = "") -> bool:
    if not os.path.isfile(path) or os.path.getsize(path) <= 0:
        return False
    return not expected_sha256 or _sha256(path).lower() == expected_sha256.lower()


def get_or_download(model_config: dict, cache_dir: str = "/app/data/models",
                    artifact_label: str = "modelo") -> str:
    """Retorna um artefato completo do cache ou faz o download."""
    os.makedirs(cache_dir, exist_ok=True)

    url = str(model_config.get("url") or model_config.get("model_url") or "").strip()
    parsed_url = urlparse(url)
    if parsed_url.scheme not in ("http", "https") or not parsed_url.netloc:
        raise ValueError(f"a URL do {artifact_label} deve usar HTTP ou HTTPS")

    filename = model_config.get("filename")
    if not filename:
        filename = os.path.basename(parsed_url.path)
    if not filename:
        filename = hashlib.sha256(url.encode()).hexdigest()[:16] + ".bin"
    filename = _safe_filename(filename)

    expected_sha256 = str(model_config.get("sha256") or "").strip()
    if expected_sha256 and (
            len(expected_sha256) != 64 or any(c not in "0123456789abcdefABCDEF"
                                              for c in expected_sha256)):
        raise ValueError("sha256 deve conter 64 caracteres hexadecimais")

    local_path = os.path.abspath(os.path.join(cache_dir, filename))
    lock_path = local_path + ".lock"
    with open(lock_path, "a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if _valid_cache(local_path, expected_sha256):
            size_mb = os.path.getsize(local_path) / (1024 ** 2)
            print(f"[download] Cache de {artifact_label}: {filename} ({size_mb:.1f} MB)")
            return local_path
        if os.path.exists(local_path):
            os.remove(local_path)

        if not _REQUESTS:
            raise ImportError("requests não está instalado")

        temporary_path = f"{local_path}.part.{os.getpid()}"
        last_error = None
        for attempt in range(3):
            if attempt:
                time.sleep(2 ** (attempt - 1))
            try:
                print(f"[download] {artifact_label} {attempt + 1}/3: {filename}")
                with requests.get(
                        url,
                        stream=True,
                        timeout=(15, 300),
                        headers={"User-Agent": "hardware-tester-ai/1.0"},
                ) as response:
                    response.raise_for_status()
                    content_type = response.headers.get("content-type", "").lower()
                    if "text/html" in content_type:
                        raise ValueError(
                            f"a URL retornou HTML em vez do arquivo do {artifact_label}")
                    expected_bytes = int(response.headers.get("content-length", 0) or 0)
                    downloaded = 0
                    with open(temporary_path, "wb") as output:
                        for chunk in response.iter_content(chunk_size=1024 * 1024):
                            if not chunk:
                                continue
                            output.write(chunk)
                            downloaded += len(chunk)
                        output.flush()
                        os.fsync(output.fileno())
                    if downloaded <= 0:
                        raise ValueError("o download retornou um arquivo vazio")
                    if expected_bytes and not response.headers.get("content-encoding") \
                            and downloaded != expected_bytes:
                        raise ValueError(
                            f"download incompleto: {downloaded} de {expected_bytes} bytes")
                if expected_sha256 and _sha256(temporary_path).lower() != expected_sha256.lower():
                    raise ValueError(f"SHA-256 do {artifact_label} não confere")
                os.replace(temporary_path, local_path)
                size_mb = os.path.getsize(local_path) / (1024 ** 2)
                print(f"[download] {artifact_label} salvo: {filename} ({size_mb:.1f} MB)")
                return local_path
            except Exception as exc:
                last_error = exc
                try:
                    os.remove(temporary_path)
                except FileNotFoundError:
                    pass
                print(f"[download] Falha no {artifact_label}: {exc}")

    raise RuntimeError(f"download de {filename} falhou após 3 tentativas: {last_error}")
