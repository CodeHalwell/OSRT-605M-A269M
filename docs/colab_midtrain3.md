# midtrain3 on Colab H100 — paste-in cells

Colab gives the **same H100** as Modal, so `MidtrainExtend3Config` runs as-is
(batch 6, seq 4096, grad-ckpt, ~28s/step, ~10k tok/s). ~550 units ≈ ~35 hr ≈
**~1.2B tokens** in one burst → base 2.2B → ~3.4B.

The run **auto-resumes** across Colab disconnects: `run_pretrain_extend` scans
`--ckpt-dir` for the highest `osrt_v5_midtrain3_step_*.pt`. Keep that dir on
**Google Drive** so checkpoints survive a dropped session.

> **Drive space gotcha (free = 15GB):** each checkpoint is ~4.9GB. Over a 35hr
> burst at `ckpt_interval=500` that's ~8 checkpoints = ~39GB — over the free
> quota. Use the **prune cell** (keeps only the latest 2 ≈ 10GB), or paid Drive,
> or push to a private HF repo.

---

### Cell 1 — confirm the GPU is an H100
```python
!nvidia-smi --query-gpu=name,memory.total --format=csv
```

### Cell 2 — mount Drive + set the checkpoint dir
```python
from google.colab import drive
drive.mount('/content/drive')
CKPT_DIR = "/content/drive/MyDrive/osrt/ckpt"
!mkdir -p {CKPT_DIR}
```

### Cell 3 — clone the repo (re-run each session; code isn't persisted)
```python
!git clone https://github.com/CodeHalwell/OSRT-605M-A269M.git /content/osrt || (cd /content/osrt && git pull)
%cd /content/osrt
```

### Cell 4 — install deps (Colab's torch is recent enough; install the rest)
```python
!pip install -q "transformers==5.3.0" "datasets==4.6.1" "tokenizers==0.22.2" \
    "safetensors==0.7.0" "wandb==0.25.1" "lion-pytorch==0.2.4" sentencepiece
# If you hit a torch API error, pin it: 
#   !pip install -q torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
```

### Cell 5 — stage the base checkpoint + tokenizer into CKPT_DIR (ONE TIME)
Upload from your Mac to `MyDrive/osrt/ckpt/` once (via drive.google.com or the
Drive desktop app):
- `osrt_v5_midtrain2_step_1750.pt`  (4.9GB — the base)
- the `v6_tokenizer_export/` folder → put it at `/content/osrt/v6_tokenizer_export` (it's small; or copy from Drive)
```python
# tokenizer is tiny — copy from Drive if you uploaded it, else it's in the repo already
import os; print("base present:", os.path.exists(f"{CKPT_DIR}/osrt_v5_midtrain2_step_1750.pt"))
!ls -la v6_tokenizer_export
```

### Cell 6 — secrets (add HF_TOKEN + WANDB_API_KEY in Colab's 🔑 panel first)
```python
import os
from google.colab import userdata
os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
os.environ["WANDB_API_KEY"] = userdata.get("WANDB_API_KEY")
```

### Cell 7 — SANITY GATE (30 steps, ~5 min) — run once before the burst
```python
!PYTHONPATH=src python scripts/lightning_midtrain3.py --sanity --ckpt-dir {CKPT_DIR}
```
Expect: `CUDA device: … H100`, clean load of step_1750, mix streams, ~30 steps,
no crash.

### Cell 8 — (optional) background prune: keep only the latest 2 checkpoints
```python
import glob, os, re, time, threading
def prune():
    while True:
        cks = glob.glob(f"{CKPT_DIR}/osrt_v5_midtrain3_step_*.pt")
        cks.sort(key=lambda p: int(re.search(r'step_(\d+)', p).group(1)))
        for old in cks[:-2]:
            os.remove(old); print("pruned", os.path.basename(old))
        time.sleep(300)
threading.Thread(target=prune, daemon=True).start()
```

### Cell 9 — FULL BURST (resumes automatically on every reconnect)
```python
!PYTHONPATH=src python scripts/lightning_midtrain3.py --ckpt-dir {CKPT_DIR} --ckpt-interval 500
```
On reconnect after a disconnect: re-run cells 1–6, then re-run **Cell 9** — it
prints `Found midtrain3 checkpoint at step N / Resumed at step N+1` and
continues the same cosine. Watch the `osrt-v6-midtrain3` W&B run for the ppl
trend across sessions.

---

**When the 550 units run out** (~step 1200–1300 from step_1750, i.e. ~1.2B more
tokens): the latest `midtrain3_step_*.pt` is on Drive. Continue on the monthly
Modal drip (`modal volume put` it, `modal run --detach app.py::run_midtrain3`)
or the next Colab credits — same resume-scan. Toward ~1× Chinchilla, re-run
SFT v2 (now with the EOS fix) on the strengthened base.
