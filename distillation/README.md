# Knowledge Distillation

Distill a domain-knowledge-enhanced **Qwen2.5-32B** teacher into a **Qwen2.5-7B** student for offline automotive fault diagnosis.

Back to the [root README](../README.md).

## Layout

```
distillation/
├── config.py              # Paths, models, SFT / PPO hyperparameters
├── data/
│   └── prepare_data.py    # Long-tail balancing + train/val/benchmark splits
├── train_sft.py           # LoRA SFT
├── teacher/
│   ├── vllm_server.py     # Teacher serving
│   └── teacher_agent.py   # Teacher + JSON GT cache (Neo4j optional)
├── distill/
│   ├── reward.py          # Reverse KL + DK / ECU rewards
│   ├── sampler.py         # Online sampling
│   └── trainer.py         # PPO distillation
└── scripts/               # Pipeline wrappers
```

## Pipeline

| Stage | What it does |
|-------|----------------|
| **Data prep** | Square-root sampling on high-frequency `RepairMeasures`, convert to SFT format, split train / val / benchmark |
| **SFT** | LoRA fine-tune Qwen2.5-7B to emit JSON `{FaultDescription, RepairMeasures}` |
| **Teacher** | vLLM service; GT lookup from `Integrated_Data.json` (Neo4j fallback off by default) |
| **Distillation** | PPO with reverse KL vs teacher logits, plus optional DK / ECU consistency rewards |

Default checkpoints (override in `config.py` or via env):

```python
MODEL_CONFIG = {
    "teacher_model_name": "Qwen/Qwen2.5-32B-Instruct-AWQ",
    "student_model_name": "Qwen/Qwen2.5-7B-Instruct",
}
```

## Quick Start

```bash
pip install -r requirements.txt

# Full pipeline (teacher must be started in a second terminal when prompted)
bash scripts/run_all.sh

# Or stage by stage
bash scripts/1_prepare_data.sh
bash scripts/2_train_sft.sh
bash scripts/3_start_teacher.sh    # keep running
bash scripts/4_train_distill.sh
```

`scripts/4_train_distill.sh` checks `http://127.0.0.1:8000/health` and exits if the teacher is down.

Processed files written by step 1:

| File | Role |
|------|------|
| `data/processed/sft_balanced.json` | Balanced full set |
| `data/processed/train.json` | Training split |
| `data/processed/val.json` | Validation split |
| `data/processed/test_benchmark.json` | 500-instance held-out benchmark |
| `data/processed/test_benchmark_subset.json` | 100-instance eval subset |

## Hardware

| Stage | GPU memory |
|-------|------------|
| SFT (LoRA, 7B) | 24 GB+ |
| Teacher (AWQ 32B) | 20 GB+ |
| Distillation | 80 GB (A100-class) |

Teacher GT retrieval uses the JSON cache and does **not** need Neo4j unless you set `TEACHER_GT_CONFIG["use_neo4j_fallback"] = True`.

Optional env vars:

```bash
export VLLM_MODEL=Qwen/Qwen2.5-32B-Instruct-AWQ
export NEO4J_PASSWORD=your_password   # Neo4j fallback only
```

## References

- [MiniLLM: Knowledge Distillation of Large Language Models](https://arxiv.org/abs/2306.08543)
- [vLLM Documentation](https://docs.vllm.ai/)
