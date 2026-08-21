# Model Evaluation

Evaluation module for assessing SFT and distilled model accuracy on automotive fault diagnosis.

## Evaluation Pipeline

Multi-stage evaluation with threshold calibration:

1. **Gate 1**: High-confidence fast track (Score ≥ 90, Similarity ≥ 0.8)
2. **Gate 2**: Hard entity matching (DTC/ECU validation)
3. **Gate 3**: Logic contradiction detection
4. **Gate 4**: Safety net fallback
5. **Step 2**: Structured extraction + online arbitration

## Quick Start

```bash
cd eval

# Evaluate SFT model
python run_eval.py --model_path ../distillation/outputs/sft/merged_model --num_samples 100

# Evaluate distilled model
python run_eval.py --model_path ../distillation/outputs/minillm/final_model --num_samples 100

# Custom test data
python run_eval.py --model_path <MODEL_PATH> --test_data /path/to/test.json
```

## Arguments

| Argument | Description |
|----------|-------------|
| `--model_path` | Model directory path (required) |
| `--test_data` | Test data JSON file |
| `--num_samples` | Number of test samples |
| `--eval_field` | Field to evaluate: `FaultDescription` / `ServiceMeasures` / `both` |
| `--use_vllm` | Use vLLM API for inference |

## Configuration

Edit `config.py`:

- `LLM_EVALUATOR_CONFIG`: Scoring LLM (DeepSeek-V3 / GPT-4o recommended)
- `SEMANTIC_MODEL_NAME`: Local embedding model path

## Output

Results saved to `results/eval_results_<timestamp>/`:

```json
{
  "summary": {
    "FaultDescription": {"total": 100, "pass": 87, "accuracy": 87.0},
    "ServiceMeasures": {"total": 100, "pass": 82, "accuracy": 82.0},
    "overall": {"accuracy": 84.5}
  }
}
```
