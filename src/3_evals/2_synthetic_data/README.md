# Generate synthetic data using Agent Pipeline

```bash
uv run --env-file .env -m src.3_evals.2_synthetic_data.synthesize_data \
--source_dataset hf://vector-institute/hotpotqa@d997ecf:train \
--langfuse_dataset_name search-dataset-synthetic-20250609 \
--limit 18
```

## Evaluate diversity of synthetic data

```bash
# Baseline: "Real" dataset
uv run \
--env-file .env \
-m src.3_evals.2_synthetic_data.annotate_diversity \
--langfuse_dataset_name search-dataset \
--run_name cosine_similarity_bge_m3

# Synthetic dataset
uv run \
--env-file .env \
-m src.3_evals.2_synthetic_data.annotate_diversity \
--langfuse_dataset_name search-dataset-synthetic-20250609 \
--run_name cosine_similarity_bge_m3
```

Example Output:

```bash
# baseline
Items to process: 18
Embedding ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:02 0:00:00
Cosine similarity of search-dataset
 count    18.000000
mean      0.376153
std       0.027593
min       0.337244
25%       0.355240
50%       0.368091
75%       0.397940
max       0.421926
dtype: float64
Uploading scores... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:00

# synthetic, default temperature, etc.
Items to process: 80
Embedding ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:09 0:00:00
Cosine similarity of search-dataset-synthetic-20250609
 count    80.000000
mean      0.350789
std       0.027978
min       0.275784
25%       0.330807
50%       0.351904
75%       0.371099
max       0.409278
dtype: float64
Uploading scores... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:00
```

## Run Evaluation on synthetic data

```bash
uv run \
--env-file .env \
-m src.3_evals.1_llm_judge.run_eval \
--langfuse_dataset_name search-dataset-synthetic-20250609 \
--run_name enwiki_weaviate \
--limit 18
```
