"""Test Loading HuggingFace datasets."""

import pandas as pd

from src.utils.data import get_dataset, get_dataset_url_hash


def test_load_from_hub_unspecified_subset():
    """Test loading dataset from hub, no subset specified."""
    url = "hf://junzhang1207/search-dataset@2349ba4:train"
    rows_limit = 18
    dataset = get_dataset(url, limit=rows_limit)
    print(url, get_dataset_url_hash(url), rows_limit, dataset)
    assert isinstance(dataset, pd.DataFrame), type(dataset)
    assert len(dataset) == rows_limit


def test_load_from_hub_named_subset():
    """Test loading dataset from hub, no subset specified."""
    url = "hf://ComplexDataLab/Misinfo_Datasets@9f1ca17[liar_new]:test"
    rows_limit = 18
    dataset = get_dataset(url, limit=rows_limit)
    print(url, get_dataset_url_hash(url), rows_limit, dataset)
    assert isinstance(dataset, pd.DataFrame), type(dataset)
    assert len(dataset) == rows_limit
