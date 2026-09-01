# v6 archive

Everything the v6 lineage needed, moved here when the repo focused on v7.
**Reference only — not importable from these paths.**

| | |
|---|---|
| `src/` | SFT pipeline (`sft_train`, `sft_eval`, `sft_data`), RL (`grpo_train`, `rewards`), `lm_eval_wrapper`, the SFT `RolloutDataset`, the 31 v6 stage configs, and the dead `run_pretrain_extend` / `run_rollout_eval` paths |
| `tests/` | their tests |
| `scripts/` | the v6 runners — `lightning_*`, `colab_grpo`, `build_sft_v*`, `collect_*`, `smoke_*`, `local_sft_*`, `probe_sft_*` |
| `app.py` | the ~20-stage v6 Modal app, replaced by a lean v7 one |
| `notebooks/` `rollouts/` `review/` `paper/` `configs/` | v6 artefacts |
| `v6_tokenizer_export/` | the 65,536 BPE, superseded at gate G2 |
| `docs/` | v6 handoff and Colab recipe |

## Two things worth knowing before reusing any of it

**The GRPO reward is on record as harmful.** Wave 1 made the model measurably
worse — soup − step100 θ = +8.00pp, p=0.002, with the reasoning-off control
unchanged (p=0.745). v7's post-training is redesigned around a verifier that
does not exist yet. Do not lift `rewards.py` without reading roadmap §7.6.

**The v6 tokenizer made 1–3 digit numbers atomic** (100 / 100 / 96.7%
single-token) at 75% context consistency. That is what motivated gate G2 and
why v7 uses the OSRT-Ostinato vocabulary instead (roadmap §16).

Full v3–v6 history is in the git log; `archive/v3` and `archive/v4` predate this.
