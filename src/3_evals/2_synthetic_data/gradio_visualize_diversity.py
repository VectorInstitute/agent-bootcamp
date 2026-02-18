"""Visualize embedding diversity of the given LangFuse dataset.

Usage:

uv run \
--env-file .env \
gradio src/3_evals/2_synthetic_data/gradio_visualize_diversity.py
"""

from typing import List

import gradio as gr
import numpy as np
import plotly.express as px
from openai import AsyncOpenAI
from plotly.graph_objs import Figure
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from src.utils import Configs, create_batches, gather_with_progress
from src.utils.langfuse.shared_client import langfuse_client


def reduce_dimensions(
    embeddings: np.ndarray, method: str = "tsne", n_components: int = 2
) -> np.ndarray:
    """
    Reduces the dimensionality of the given embeddings to 2D using the specified method.

    Args:
        embeddings (np.ndarray): The input embeddings of shape (n_samples, n_features).
        method (str): The dimensionality reduction method to use ('tsne' or 'pca').
        n_components (int): Number of dimensions to reduce to (default is 2).

    Returns
    -------
        np.ndarray: Reduced 2D embeddings of shape (n_samples, 2).
    """
    if method == "tsne":
        reducer = TSNE(
            n_components=n_components,
            perplexity=min(embeddings.shape[0] - 1, 30),
            random_state=42,
        )
    elif method == "pca":
        reducer = PCA(n_components=n_components)
    else:
        raise ValueError("Method must be 'tsne' or 'pca'")

    return reducer.fit_transform(embeddings)


def plot_embeddings_2d(
    reduced_embeddings: np.ndarray, texts: List[str], dataset_title: str | None = None
) -> Figure:
    """
    Plot 2D embeddings using Plotly, displaying text on hover.

    Args:
        reduced_embeddings (np.ndarray): 2D embeddings of shape (n_samples, 2).
        texts (List[str]): List of text snippets for hover information.
    """
    fig = px.scatter(
        x=reduced_embeddings[:, 0],
        y=reduced_embeddings[:, 1],
        hover_name=texts,
        title=f"Text Embeddings for {dataset_title}"
        "<sup>Note: Axis are not interpretible</sup>",
        labels={"x": "Latent Component 1", "y": "Latent Component 2"},
    )
    fig.update_traces(marker={"size": 8, "opacity": 0.7})
    return fig


async def get_projection_plot(
    dataset_name: str,
    projection_method: str,
    limit: int | None = None,
    embedding_batch_size: int = 16,
) -> Figure:
    """Obtain projection plot for the given dataset up to `limit` items."""
    lf_dataset_items = langfuse_client.get_dataset(dataset_name.strip()).items

    # Generate embeddings
    configs = Configs()
    embedding_client = AsyncOpenAI(
        api_key=configs.embedding_api_key,
        base_url=configs.embedding_base_url,
        max_retries=5,
    )

    texts = [_item.input["text"] for _item in lf_dataset_items]
    text_batches = create_batches(
        texts,
        batch_size=embedding_batch_size,
        limit=int(limit) if limit else None,
        keep_trailing=True,
    )
    embed_coros = [
        embedding_client.embeddings.create(
            input=_batch, model=configs.embedding_model_name
        )
        for _batch in text_batches
    ]
    batched_embed_results = await gather_with_progress(
        embed_coros, description=f"Generating {len(texts)} embeddings"
    )
    embeddings = [
        _data.embedding for _result in batched_embed_results for _data in _result.data
    ]  # unpacked
    embeddings_np = np.asarray(embeddings)

    # Reduce dimensions
    num_texts = min(int(limit), len(texts)) if limit else len(texts)
    assert embeddings_np.shape[0] == num_texts, (embeddings_np.shape, num_texts)
    embeddings_reduced = reduce_dimensions(embeddings_np, method=projection_method)

    # Create plot
    return plot_embeddings_2d(
        reduced_embeddings=embeddings_reduced,
        texts=texts[:num_texts],
        dataset_title=dataset_name,
    )


if __name__ == "__main__":
    viewer = gr.Interface(
        fn=get_projection_plot,
        inputs=[
            gr.Textbox(label="Dataset name"),
            gr.Radio(["tsne", "pca"], label="Dimensionality Reduction Method"),
            gr.Number(value=18, label="Number of rows to plot", minimum=1),
        ],
        outputs=gr.Plot(label="2D Embedding Plot"),
        title="3.2 Text Embedding Visualizer",
        description="Select a method to visualize 256-D embeddings of text snippets.",
    )

    viewer.launch(share=True)
