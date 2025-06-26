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

Example data:

```
                                            question                                    expected_answer            category       area
0                  steve jobs statue location budapst  The Steve Jobs statue is located in Budapest, ...                Arts  Knowledge
1   Why is the Battle of Stalingrad considered a t...  The Battle of Stalingrad is considered a turni...        General News       News
2   In what year did 'The Birth of a Nation' surpa...  This question is based on a false premise. 'Th...       Entertainment       News
3   How many Russian soldiers surrendered to AFU i...  About 300 Russian soldiers surrendered to the ...        General News       News
4    What event led to the creation of Google Images?  Jennifer Lopez's appearance in a green Versace...          Technology       News
5                  steve jobs statue location budapst  The Steve Jobs statue is located in Budapest, ...                Arts  Knowledge
6   Why is the Battle of Stalingrad considered a t...  The Battle of Stalingrad is considered a turni...        General News       News
```

## Run LLM-as-a-Judge Evaluation

```bash
source .env
uv run -m src.3_evals.1_llm_judge.run_eval \
--langfuse_dataset_name search-dataset \
--run_name enwiki_weaviate
```

Example output:

```bash
...
Running agent and evaluating ━━━━━━━━━━━━━━━━━━━━━━╺━━━━━━━━━━━━━━━━━  56% 0:00:05
             OpenAI-Agent-Trace
               OpenAI Agents trace: Agent workflow
                 Agent run: 'Wikipedia Agent'
21:22:24.963       Function: search_knowledgebase
21:22:24.964         search
21:22:24.988 OpenAI Agents trace: Agent workflow
21:22:24.989   Agent run: 'Evaluator Agent'
21:22:24.990     Responses API with 'gpt-4o'
             OpenAI-Agent-Trace
               OpenAI Agents trace: Agent workflow
                 Agent run: 'Wikipedia Agent'
21:22:25.037       Function: search_knowledgebase
Running agent and evaluating ━━━━━━━━━━━━━━━━━━━━━━╺━━━━━━━━━━━━━━━━━  56% 0:00:05
21:22:25.039         search
...
21:22:34.278     Responses API with 'gpt-4o'
Running agent and evaluating ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:00
Uploading scores ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:01
```
