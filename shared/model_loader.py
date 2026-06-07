"""Download e cache de modelos."""

import os
import time
import hashlib
from typing import Optional

try:
    import requests
    _REQUESTS = True
except ImportError:
    _REQUESTS = False


def get_or_download(model_config: dict, cache_dir: str = "/app/data/models") -> str:
    """Retorna o caminho local do modelo, baixando quando necessário."""
    os.makedirs(cache_dir, exist_ok=True)

    url = model_config.get("url") or model_config.get("model_url")
    if not url:
        raise ValueError("model_config deve conter 'url' ou 'model_url'")

    filename = model_config.get("filename")
    if not filename:
        filename = url.split("/")[-1].split("?")[0]
        if not filename:
            filename = hashlib.md5(url.encode()).hexdigest()[:12] + ".bin"

    local_path = os.path.join(cache_dir, filename)

    if os.path.exists(local_path):
        size_mb = os.path.getsize(local_path) / (1024**2)
        print(f"[model_loader] Cache hit: {filename} ({size_mb:.1f} MB)")
        return os.path.abspath(local_path)

    print(f"[model_loader] Baixando: {filename}")
    print(f"[model_loader] URL: {url}")

    if not _REQUESTS:
        raise ImportError("requests não instalado. Execute: pip install requests")

    max_retries = 3
    last_error = None

    for attempt in range(max_retries):
        if attempt > 0:
            wait_sec = 2 ** (attempt - 1)
            print(f"[model_loader] Tentativa {attempt + 1}/{max_retries} em {wait_sec}s...")
            time.sleep(wait_sec)

        try:
            response = requests.get(url, stream=True, timeout=300,
                                    headers={"User-Agent": "pdc-ia/1.0"})
            response.raise_for_status()

            total_bytes = int(response.headers.get("content-length", 0))
            downloaded = 0
            last_print = 0

            tmp_path = local_path + ".tmp"

            with open(tmp_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)

                        if total_bytes > 0:
                            pct = downloaded / total_bytes * 100
                            if pct - last_print >= 10:
                                print(f"[model_loader] {pct:.0f}% ({downloaded/(1024**2):.1f}/{total_bytes/(1024**2):.1f} MB)")
                                last_print = pct
                        else:
                            if downloaded - last_print >= 50 * 1024 * 1024:  # 50MB
                                print(f"[model_loader] {downloaded/(1024**2):.1f} MB baixados...")
                                last_print = downloaded

            os.rename(tmp_path, local_path)
            final_mb = os.path.getsize(local_path) / (1024**2)
            print(f"[model_loader] Download completo: {filename} ({final_mb:.1f} MB)")
            return os.path.abspath(local_path)

        except Exception as e:
            last_error = e
            print(f"[model_loader] Erro na tentativa {attempt + 1}: {e}")
            if os.path.exists(local_path + ".tmp"):
                os.remove(local_path + ".tmp")

    raise RuntimeError(
        f"Download falhou após {max_retries} tentativas. Último erro: {last_error}"
    )
