# Model Evaluation

Gated evaluation of SFT and distilled models on automotive fault diagnosis.

Back to the [root README](../README.md).

## Pipeline

Multi-stage scoring (see `evaluators/multiagent_evaluator.py`):

1. **Step 1 — technical precision.** An evaluator LLM scores the prediction.
2. **Gate 1.** High-confidence pass when the score is ≥ 92.
3. **Gate 2.** Hard entity match on ECU / DTC mentions.
4. **Gate 3.** Logic-conflict / unsafe-repair checks.
5. **Step 2.** Structured extraction plus semantic / search arbitration when earlier gates do not decide.

## Setup

Install the training extras plus evaluation dependencies:

```bash
pip install -r ../distillation/requirements.txt
pip install sentence-transformers
```

Set an API key for the scoring / arbitration LLMs (OpenRouter by default in `config.py`):

```bash
export OPENROUTER_API_KEY=your_key
```

Optionally point at a local embedding checkpoint (otherwise the evaluator falls back to ModelScope / Hugging Face `intfloat/e5-mistral-7b-instruct`):

```bash
export SEMANTIC_MODEL_PATH=/path/to/e5-mistral-7b-instruct
```

Edit `config.py` to change:

- `LLM_EVALUATOR_CONFIG` — scoring LLM (DeepSeek-V3 / GPT-4o class recommended)
- `LLM_SAME_CONFIG` — arbitration LLM
- `DEFAULT_TEST_DATA_PATH` — defaults to `distillation/data/processed/val.json`

## Quick Start

```bash
cd eval

# SFT model
python run_eval.py --model_path ../distillation/outputs/sft/merged_model --num_samples 100

# Distilled model
python run_eval.py --model_path ../distillation/outputs/minillm/final_model --num_samples 100

# Custom test file
python run_eval.py --model_path <MODEL_PATH> --test_data /path/to/test.json
```

Run data prep first (`distillation/scripts/1_prepare_data.sh`) if `val.json` is not present. For the 100-instance subset use `--test_data ../distillation/data/processed/test_benchmark_subset.json`.

## Arguments

| Argument | Description |
|----------|-------------|
| `--model_path` | Model directory (required) |
| `--test_data` | Test JSON; default `val.json` |
| `--num_samples` | Number of samples; default all |
| `--eval_field` | `FaultDescription` / `ServiceMeasures` / `both` |
| `--use_vllm` | Infer via a running vLLM server |
| `--vllm_url` | vLLM chat-completions URL |

`--eval_field ServiceMeasures` is the CLI name for the dataset field `RepairMeasures`.

## Output

Results go to `results/eval_results_<timestamp>/`:

```json
{
  "summary": {
    "FaultDescription": {"total": 100, "pass": 87, "accuracy": 87.0},
    "ServiceMeasures": {"total": 100, "pass": 82, "accuracy": 82.0},
    "overall": {"accuracy": 84.5}
  }
}
```

The JSON numbers above are format examples, not reported paper results.
