"""HF Hub checkpoint sync for ephemeral/capped GPU sessions (Colab, etc.).

A Colab session's disk is wiped when the VM is released and there's a 24h cap,
so a multi-session run must persist checkpoints off-VM. This pushes each new
`{prefix}_step_*.pt` to a private HF repo as it's saved, and pulls the latest
back on start — so the resume-scan in run_pretrain_extend chains across sessions.

Used by lightning_midtrain3.py via --hf-repo. Needs HF_TOKEN in the env.
"""
from __future__ import annotations

import glob
import os
import re
import threading
import time


def _step_of(path: str) -> int:
    m = re.search(r"step_(\d+)\.pt$", path)
    return int(m.group(1)) if m else -1


def pull_latest(repo_id: str, ckpt_dir: str, prefix: str,
                base_name: str | None = None) -> None:
    """Download the highest {prefix}_step_*.pt from the repo into ckpt_dir (so
    the resume-scan finds it). Also fetch base_name if given and absent."""
    from huggingface_hub import HfApi, hf_hub_download

    os.makedirs(ckpt_dir, exist_ok=True)
    api = HfApi()
    try:
        files = api.list_repo_files(repo_id, repo_type="model")
    except Exception as e:  # noqa: BLE001 — fresh/empty repo is fine
        print(f"[hf-sync] no repo yet ({type(e).__name__}); starting clean",
              flush=True)
        files = []

    if base_name and not os.path.exists(os.path.join(ckpt_dir, base_name)):
        if base_name in files:
            print(f"[hf-sync] pulling base {base_name}...", flush=True)
            hf_hub_download(repo_id, base_name, repo_type="model",
                            local_dir=ckpt_dir)

    steps = [f for f in files if re.match(rf"{prefix}_step_\d+\.pt$", f)]
    if steps:
        latest = max(steps, key=_step_of)
        print(f"[hf-sync] resuming: pulling {latest}...", flush=True)
        hf_hub_download(repo_id, latest, repo_type="model", local_dir=ckpt_dir)
    else:
        print("[hf-sync] no prior checkpoints in repo — starting from base",
              flush=True)


def start_push_daemon(repo_id: str, ckpt_dir: str, prefix: str,
                      interval: int = 60, keep_remote: int = 3) -> None:
    """Background thread: upload new {prefix}_step_*.pt as they appear, and
    prune the repo to the newest `keep_remote` (HF is generous but 4.9GB each
    adds up). Fire-and-forget; daemon dies with the process."""
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id, repo_type="model", private=True, exist_ok=True)
    pushed: set[str] = set()

    def _loop() -> None:
        while True:
            try:
                local = sorted(
                    glob.glob(os.path.join(ckpt_dir, f"{prefix}_step_*.pt")),
                    key=_step_of)
                for path in local:
                    name = os.path.basename(path)
                    if name in pushed:
                        continue
                    print(f"[hf-sync] uploading {name}...", flush=True)
                    api.upload_file(path_or_fileobj=path, path_in_repo=name,
                                    repo_id=repo_id, repo_type="model")
                    pushed.add(name)
                # prune remote to newest keep_remote
                remote = [f for f in api.list_repo_files(repo_id, repo_type="model")
                          if re.match(rf"{prefix}_step_\d+\.pt$", f)]
                for old in sorted(remote, key=_step_of)[:-keep_remote]:
                    api.delete_file(old, repo_id, repo_type="model")
                    print(f"[hf-sync] pruned remote {old}", flush=True)
            except Exception as e:  # noqa: BLE001 — never kill training on a sync hiccup
                print(f"[hf-sync] WARN {type(e).__name__}: {str(e)[:120]}",
                      flush=True)
            time.sleep(interval)

    threading.Thread(target=_loop, daemon=True).start()
    print(f"[hf-sync] push daemon started → {repo_id} (every {interval}s, "
          f"keep {keep_remote})", flush=True)
