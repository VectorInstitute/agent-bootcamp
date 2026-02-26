"""Script to load and explore Bloomberg Financial News dataset.

This dataset consists of 446,762 financial news articles from Bloomberg (2006-2013).
Dataset: danidanou/Bloomberg_Financial_News
"""

import os
import pandas as pd


def load_bloomberg_data(
    limit: int | None = None,
    split: str = "train"
) -> pd.DataFrame:
    """Load Bloomberg Financial News dataset from HuggingFace.
    
    Args:
        limit: Optional maximum number of articles to load
        split: Dataset split to load (default: "train")
    
    Returns:
        DataFrame with financial news articles
    """
    print(f"Loading Bloomberg Financial News dataset (split: {split})...")
    
    # Try direct parquet reading approach
    try:
        print("Method 1: Direct download and read as parquet...")
        from huggingface_hub import hf_hub_download
        import pyarrow.parquet as pq
        
        print("Starting download of parquet file...")
        # Download the file
        file_path = hf_hub_download(
            repo_id="danidanou/Bloomberg_Financial_News",
            filename="bloomberg_financial_data.parquet.gzip",
            repo_type="dataset"
        )
        print(f"Download complete. File path: {file_path}")
        
        print("Starting to read parquet file...")
        table = pq.read_table(file_path)
        print(f"Parquet table read successfully. Table shape: {table.shape}")
        
        print("Converting table to pandas DataFrame...")
        df = table.to_pandas()
        print(f"Conversion to pandas complete. DataFrame shape: {df.shape}")
        
        print(f"Successfully loaded {len(df)} articles")
        
        # Apply limit if specified
        if limit is not None:
            df = df.head(limit)
            print(f"Limited to {limit} articles")
        
        print(f"Columns: {list(df.columns)}")
        return df
        
    except Exception as e:
        print(f"Method 1 failed: {e}")
        print("\nMethod 2: Trying datasets library with no verification...")
        
    # Fallback: datasets library with verification disabled
    try:
        from datasets import load_dataset
        
        dataset = load_dataset(
            "danidanou/Bloomberg_Financial_News",
            split=split,
            verification_mode="no_checks",
            trust_remote_code=False
        )
        
        df = dataset.to_pandas()
        print(f"Successfully loaded {len(df)} articles")
        
        if limit is not None:
            df = df.head(limit)
            print(f"Limited to {limit} articles")
        
        print(f"Columns: {list(df.columns)}")
        return df
        
    except Exception as e:
        print(f"Method 2 failed: {e}")
        raise RuntimeError(
            f"Unable to load Bloomberg dataset. Both methods failed.\n"
            f"This dataset has known compression format issues.\n"
            f"Please report this issue or try manually downloading from:\n"
            f"https://huggingface.co/datasets/danidanou/Bloomberg_Financial_News"
        )


def explore_dataset(df: pd.DataFrame) -> None:
    """Print summary statistics about the dataset.
    
    Args:
        df: DataFrame containing Bloomberg articles
    """
    print("\n" + "="*60)
    print("DATASET EXPLORATION")
    print("="*60)
    
    print(f"\nTotal articles: {len(df):,}")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nData types:\n{df.dtypes}")
    
    print("\n" + "-"*60)
    print("SAMPLE RECORD:")
    print("-"*60)
    print(df.iloc[0].to_dict())
    
    # Check for text columns
    text_columns = [col for col in df.columns if 'text' in col.lower() or 'content' in col.lower()]
    if text_columns:
        print(f"\nText columns found: {text_columns}")
        for col in text_columns:
            print(f"\nSample from '{col}':")
            print(df[col].iloc[0][:500] + "..." if len(str(df[col].iloc[0])) > 500 else df[col].iloc[0])
    
    # Check for missing values
    print("\n" + "-"*60)
    print("MISSING VALUES:")
    print("-"*60)
    print(df.isnull().sum())
    
    # Text length statistics if text column exists
    if text_columns:
        for col in text_columns:
            df[f'{col}_length'] = df[col].astype(str).str.len()
            print(f"\n{col} length statistics:")
            print(df[f'{col}_length'].describe())


if __name__ == "__main__":
    # Load a sample of the dataset for exploration
    df = load_bloomberg_data(limit=1000)
    
    # Explore the dataset
    explore_dataset(df)
    
    print("\n" + "="*60)
    print("To load the full dataset, use:")
    print("  df = load_bloomberg_data()")
    print("="*60)
