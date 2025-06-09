# Generate synthetic data using Agent Pipeline

```bash
source .env
uv run -m src.3_evals.2_synthetic_data.synthesize_data \
--source_dataset hf://junzhang1207/search-dataset@2349ba4:train \
--langfuse_dataset_name search-dataset-synthetic-20250609 \
--limit 18
```

## Run Evaluation on synthetic data

```bash
source .env
uv run -m src.3_evals.1_llm_judge.run_eval \
--langfuse_dataset_name search-dataset-synthetic-20250609 \
--run_name enwiki_elasticsearch \
--limit 18
```