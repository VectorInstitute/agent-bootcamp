# 3.1 LLM-as-a-Judge

## Overview

Run in the following steps:

- Create Langfuse "dataset" and upload test data to Langfuse
- Run each agent variation on the test dataset, linking the traces to the dataset run.


## Create and Populate Dataset

```bash
source .env
uv run -m src.3_evals.1_llm_judge.upload_data \
--source_dataset hf://junzhang1207/search-dataset@2349ba4:train \
--langfuse_dataset_name search-dataset \
--limit 18
```

##
