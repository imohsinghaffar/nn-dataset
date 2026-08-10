"""
Minimal, robust HuggingFace model downloader.
"""
import os
import logging
from huggingface_hub import snapshot_download

os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)

def ensure_hf_model(repo_id: str) -> str:
    """
    Ensure an HF model/tokenizer repo is available locally without network requests.
    """
    if os.path.exists(repo_id):
        return repo_id

    user_repo = repo_id.replace("/", "--")
    cache_base = os.path.expanduser(f"~/.cache/huggingface/hub/models--{user_repo}/snapshots")
    if os.path.exists(cache_base):
        snaps = [os.path.join(cache_base, s) for s in os.listdir(cache_base) if os.path.isdir(os.path.join(cache_base, s))]
        if snaps:
            return snaps[0]

    os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        return snapshot_download(repo_id, local_files_only=True)
    except Exception:
        os.environ.pop("HF_HUB_OFFLINE", None)
        return snapshot_download(repo_id)
