"""Unit tests for the hitachione multi-agent financial intelligence system.

These tests use **mocks** for all external dependencies (LLM, Weaviate, network)
so they run fast, offline, and deterministically.

Run all tests::

    uv run --env-file .env pytest -sv tests/test_hitachione/

Run a single file::

    uv run --env-file .env pytest -sv tests/test_hitachione/test_orchestrator.py

Organisation
------------
- ``test_schemas.py``          – data-model invariants
- ``test_knowledge_retrieval.py`` – KB agent (mocked Weaviate)
- ``test_researcher.py``       – researcher parallel fan-out (mocked tools)
- ``test_synthesizer.py``      – synthesizer (mocked LLM)
- ``test_reviewer.py``         – reviewer quality gate (pure logic)
- ``test_orchestrator.py``     – end-to-end orchestrator workflow (all mocked)
"""
