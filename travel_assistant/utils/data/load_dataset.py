"""Logic for loading datasets."""

import hashlib
import os.path
import re

import datasets
import pandas as pd
import pydantic


PATTERN = re.compile(
    r"(?P<provider>[^:]+)://"
    r"(?P<repo>[^:@]+)?"
    r"(@(?P<version>[a-f\d]+))?"
    r"(\[(?P<subset>\w+)\])?"
    r"(:(?P<split>\w+))?$"
)


class _SourceInfo(pydantic.BaseModel):
    provider: str
    repo: str
    version: str | None = None
    subset: str | None = None
    split: str = "train"

    @staticmethod
    def _from_url(dataset_url: str) -> "_SourceInfo":
        """Parse URL."""
        url_match = PATTERN.match(dataset_url)
        dataset_info = _SourceInfo(**url_match.groupdict()) if url_match else None
        if dataset_info is None:
            raise ValueError(
                "Invalid URL pattern. Should be {provider}://{path}[@{commit}]:{split}"
            )

        return dataset_info


def get_dataset(dataset_url: str, limit: int | None = None) -> pd.DataFrame:
    """Load dataset from the given URL.

    Params
    ------
        dataset_url: in the following format:
            {provider}://{path}[@{commit}][[subset]]:{split}
        limit: optional; max number of items to include.

    Returns
    -------
        Huggingface dataset instance.
    """
    dataset_info = _SourceInfo._from_url(dataset_url)
    if dataset_info.provider == "hf":
        return _load_hf(dataset_info, limit=limit).to_pandas()  # type: ignore

    raise ValueError(
        f"Dataset provider not supported: {dataset_info.provider}. Available options: hf"
    )


def get_dataset_url_hash(dataset_url: str) -> str:
    """Hash dataset url for attribution."""
    return hashlib.sha256(dataset_url.encode()).hexdigest()[:6]


def _load_hf(dataset_info: _SourceInfo, limit: int | None = None) -> datasets.Dataset:
    """Load HF dataset."""
    # Prefer load_from_disk locally.
    # If not possible, load from Hub or local snapshot using load_dataset.
    if (dataset_info.version is None) and (os.path.exists(dataset_info.repo)):
        try:
            dataset_or_dict = datasets.load_from_disk(dataset_info.repo)
            if dataset_info.split is not None:
                return dataset_or_dict[dataset_info.split]  # type: ignore

            return dataset_or_dict  # type: ignore

        except FileNotFoundError:
            pass  # type: ignore

    split_name = dataset_info.split
    if limit is not None:
        split_name += f"[0:{limit}]"

    return datasets.load_dataset(
        dataset_info.repo, name=dataset_info.subset, split=split_name
    )  # type: ignore
