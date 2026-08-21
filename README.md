# AutoRepairBench: Benchmarking Long-Tailed, Safety-Critical Automotive Repair Reasoning

[![CIKM](https://img.shields.io/badge/CIKM-2026-1f4e79.svg)](https://cikm2026.diag.uniroma1.it/)
[![DOI](https://img.shields.io/badge/DOI-10.1145%2F3799682.3840599-blue.svg)](https://doi.org/10.1145/3799682.3840599)
[![Paper](https://img.shields.io/badge/Paper-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Code](https://img.shields.io/badge/Code-MIT-green.svg)](LICENSE)
[![Git LFS](https://img.shields.io/badge/Git-LFS-orange.svg)](https://git-lfs.com)

Official implementation of **AutoRepairBench**, our **CIKM '26** paper on knowledge distillation of large language models for long-tailed, safety-critical automotive repair reasoning.

Chenyu Zuo<sup>1</sup>, Xuechen Zhou<sup>2</sup>, Yongqi Chen<sup>3</sup>, Lei Xue<sup>3</sup>, Shan Jiang<sup>2</sup>, [Jiaxing Shen](mailto:jiaxingshen@ln.edu.hk)<sup>1✉</sup>, Chengzu Dong<sup>1</sup>

<sup>1</sup> Lingnan University &nbsp;&nbsp; <sup>2</sup> Sun Yat-sen University (Zhuhai) &nbsp;&nbsp; <sup>3</sup> Sun Yat-sen University (Shenzhen)

**AutoRepairBench: Benchmarking Long-Tailed, Safety-Critical Automotive Repair Reasoning.**  
*Proceedings of the 35th ACM International Conference on Information and Knowledge Management (CIKM '26)*, November 07–11, 2026, Rome, Italy.  
DOI: [10.1145/3799682.3840599](https://doi.org/10.1145/3799682.3840599) · ISBN: 979-8-4007-2539-5

We distill a domain-knowledge-enhanced **Qwen2.5-32B** teacher into a compact **Qwen2.5-7B** student, so the student can run offline in vehicle diagnostic settings while retaining teacher-level diagnostic structure.

---

## Highlights

- **Teacher–student distillation with PPO.** Reverse-KL logit matching plus PPO, following the MiniLLM-style objective.
- **Domain-knowledge rewards.** Optional knowledge-graph / GT validation and ECU-consistency rewards to reduce hallucinated control-unit or DTC content.
- **Long-tail diagnostic data.** Square-root sampling over high-frequency repair texts, plus focal loss, to avoid collapsing onto a few common repair templates.
- **Gated evaluation.** Multi-stage scoring with high-confidence pass, ECU/DTC entity checks, logic-conflict detection, and semantic arbitration.

## Method Overview

```
Integrated_Data.json
        │
        ▼
 Data prep (balance + train/val/benchmark split)
        │
        ▼
 SFT on Qwen2.5-7B (LoRA, JSON FaultDescription / RepairMeasures)
        │
        ├──────────────────────────────┐
        ▼                              ▼
 Student (SFT checkpoint)     Teacher (vLLM, Qwen2.5-32B-AWQ + DK cache)
        │                              │
        └────────── PPO distill ───────┘
                    reverse KL + DK/ECU reward
                          │
                          ▼
              Compact offline 7B diagnostic model
                          │
                          ▼
         Gated evaluator (LLM score + entity + arbitration)
```

Default models in `distillation/config.py`:

| Role | Default checkpoint |
|------|--------------------|
| Teacher | `Qwen/Qwen2.5-32B-Instruct-AWQ` |
| Student | `Qwen/Qwen2.5-7B-Instruct` |

These paths are configurable. Some script comments still mention Qwen3; the runnable defaults above are the ones used unless you override them.

## Repository Structure

```
├── Integrated_Data.json      # Diagnostic dataset (Git LFS, ~125 MB)
├── distillation/             # Training pipeline
│   ├── data/                 # Balancing, formatting, splits
│   ├── teacher/              # vLLM teacher + DK/GT lookup
│   ├── distill/              # Reward, sampler, PPO trainer
│   ├── scripts/              # End-to-end and step-wise bash scripts
│   └── train_sft.py          # Student SFT
└── eval/                     # Accuracy evaluation
    ├── evaluators/           # Gated multi-stage evaluator
    └── benchmark_test/       # Additional benchmark runners
```

More detail: [distillation/README.md](distillation/README.md) · [eval/README.md](eval/README.md)

## Dataset

`Integrated_Data.json` contains **117,465** ECU/DTC diagnostic pairs.

| Statistic | Count |
|-----------|------:|
| Samples | 117,465 |
| Unique ECU IDs | 505 |
| Unique ECU variants | 499 |
| Unique DTC codes | 50,660 |

Each record is a diagnosis request and a structured repair answer:

```json
{
  "input": "ECUID: 0087, ECU_Variant: lrr_60, DTC: 5D0C, Trigger: A fault in the control unit was detected., TimeCondition: 5 s",
  "output": {
    "FaultDescription": "...",
    "RepairMeasures": "..."
  }
}
```

The file is stored with **Git LFS**. Clone with LFS enabled:

```bash
git lfs install
git clone https://github.com/AutoRepairBench/Automotive-Fault-Diagnosis-via-Knowledge-Distilled-LLM.git
```

Data is released for **research use**. It contains OEM-style diagnostic wording (ECU names, DTC texts, service measures). Redistribute or use it only in line with applicable terms.

## Requirements

- Python 3.10+
- NVIDIA GPU (see hardware table)
- [Git LFS](https://git-lfs.com)
- Linux or WSL is recommended: training/eval scripts are bash

| Stage | Typical GPU memory |
|-------|--------------------|
| SFT (LoRA, 7B) | 24 GB+ |
| Teacher (AWQ 32B) | 20 GB+ |
| Distillation (student + teacher logits) | 80 GB (A100-class) |

## Quick Start

### 1. Install

```bash
pip install -r distillation/requirements.txt
```

Evaluation additionally needs an OpenRouter (or compatible) API key and a sentence-embedding model. See [eval/README.md](eval/README.md).

Optional environment variables:

```bash
export OPENROUTER_API_KEY=your_key          # evaluation LLM
export NEO4J_PASSWORD=your_password         # only if you enable Neo4j fallback
export VLLM_MODEL=Qwen/Qwen2.5-32B-Instruct-AWQ
export SEMANTIC_MODEL_PATH=/path/to/e5-mistral-7b-instruct
```

Teacher GT lookup defaults to the JSON cache (`Integrated_Data.json`). Neo4j is optional.

### 2. Train

```bash
cd distillation
bash scripts/run_all.sh
```

Or run stages separately:

```bash
bash scripts/1_prepare_data.sh    # balance + split
bash scripts/2_train_sft.sh       # SFT
bash scripts/3_start_teacher.sh   # vLLM teacher (keep this terminal open)
bash scripts/4_train_distill.sh   # PPO distillation
```

### 3. Evaluate

```bash
cd eval
python run_eval.py --model_path ../distillation/outputs/minillm/final_model --num_samples 100
```

## Citation

If you use this repository or the paper, please cite:

```bibtex
@inproceedings{zuo2026autorepairbench,
  title     = {{AutoRepairBench}: Benchmarking Long-Tailed, Safety-Critical Automotive Repair Reasoning},
  author    = {Zuo, Chenyu and Zhou, Xuechen and Chen, Yongqi and Xue, Lei and Jiang, Shan and Shen, Jiaxing and Dong, Chengzu},
  booktitle = {Proceedings of the 35th ACM International Conference on Information and Knowledge Management},
  series    = {CIKM '26},
  year      = {2026},
  isbn      = {979-8-4007-2539-5},
  publisher = {Association for Computing Machinery},
  address   = {New York, NY, USA},
  location  = {Rome, Italy},
  doi       = {10.1145/3799682.3840599},
  url       = {https://doi.org/10.1145/3799682.3840599}
}
```

A `CITATION.cff` file is included so GitHub can generate the same citation.

## Contact

Corresponding author: Jiaxing Shen ([jiaxingshen@ln.edu.hk](mailto:jiaxingshen@ln.edu.hk)).

## License

The paper is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) (ACM copyright). Code in this repository is released under the [MIT License](LICENSE). The diagnostic dataset is for research use only; see the [Dataset](#dataset) section.

## Acknowledgements

Training follows the knowledge-distillation setup of [MiniLLM](https://arxiv.org/abs/2306.08543). Teacher serving uses [vLLM](https://docs.vllm.ai/).
