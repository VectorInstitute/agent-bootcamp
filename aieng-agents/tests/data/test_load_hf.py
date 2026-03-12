"""Test Loading HuggingFace datasets."""

import pandas as pd
from aieng.agents.data import get_dataset, get_dataset_url_hash


def test_load_from_hub_unspecified_subset() -> None:
    """Test loading dataset from hub, no subset specified."""
    url = "hf://vector-institute/hotpotqa@d997ecf:train"
    rows_limit = 18
    dataset = get_dataset(url, limit=rows_limit)
    print(url, get_dataset_url_hash(url), rows_limit, dataset)
    assert isinstance(dataset, pd.DataFrame), type(dataset)
    assert len(dataset) == rows_limit


def test_load_from_hub_named_subset() -> None:
    """Test loading dataset from hub, no subset specified."""
    url = "hf://vector-institute/hotpotqa@d997ecf:train"
    rows_limit = 18
    dataset = get_dataset(url, limit=rows_limit)
    print(url, get_dataset_url_hash(url), rows_limit, dataset)
    assert isinstance(dataset, pd.DataFrame), type(dataset)
    assert len(dataset) == rows_limit
