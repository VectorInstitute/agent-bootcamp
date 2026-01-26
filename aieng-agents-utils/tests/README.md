# Unit tests

```bash
uv run --env-file .env pytest -sv aieng-agents-utils/tests/data/test_load_hf.py
uv run --env-file .env pytest -sv aieng-agents-utils/tests/tools/test_weaviate.py
uv run --env-file .env pytest -sv aieng-agents-utils/tests/tools/test_code_interpreter.py
uv run --env-file .env pytest -sv aieng-agents-utils/tests/tools/test_gemini_grounding.py
uv run --env-file .env pytest -sv aieng-agents-utils/tests/tools/test_get_news_events.py
uv run --env-file .env pytest -sv aieng-agents-utils/tests/web_search_test_web_search_auth.py
```
