"""Convert PDF files into a HuggingFace dataset with OpenAI-compatible OCR.

This script renders PDF pages to images, sends each page to a multimodal model
via an OpenAI-compatible API, and then chunks the resulting text into a
HuggingFace dataset with a ``text`` column.

Requirements
------------
- ``OPENAI_API_KEY`` must be set in your environment.
- ``OPENAI_BASE_URL`` is optional if you use a non-default endpoint.
- ``pymupdf`` is required for PDF rendering.

Examples
--------
Transcribe a single PDF and save to ``hf_dataset``:
    uv run --env-file .env src/utils/data/pdf_to_hf_dataset.py \
        --input-path ./docs/example.pdf

Transcribe a folder recursively with a smaller DPI and a custom output:
    uv run --env-file .env src/utils/data/pdf_to_hf_dataset.py \
        --input-path ./docs --recursive --dpi 150 --output-dir ./out_dataset

Notes
-----
- Pages can be skipped using regex patterns or by skipping front/back matter.
- Chunking is token-based using the tokenizer for the target embedding model.
- Use ``--structured-ocr`` to request JSON blocks for more reliable headings.
"""

import base64
import os
import re
import time
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import click
import datasets
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from transformers import AutoTokenizer


DEFAULT_SKIP_PATTERNS = (
    r"^table of contents\b",
    r"^contents\b",
    r"^appendix\b",
    r"^appendices\b",
    r"^references\b",
    r"^bibliography\b",
    r"^glossary\b",
    r"^index\b",
    r"^acknowledg(e)?ments\b",
    r"^foreword\b",
    r"^preface\b",
)

DEFAULT_PROMPT = (
    "Transcribe all readable text from this page in natural reading order. "
    "Return plain text only. Do not summarize or add commentary. "
    "If images, charts, or other visualizations are referenced in the text, "
    "transcribe their content as part of the flow. "
    "Ignore headers, footers, page numbers, and decorative elements."
)

STRUCTURED_PROMPT = (
    "Extract the page content and return JSON only with this schema: "
    '{"blocks":[{"type":"heading|paragraph|list|table|figure|caption",'
    '"text":"...","level":1}]}. '
    "Use type=heading for section titles. If you can infer hierarchy from "
    "numbering, set level (1-4); otherwise omit or set null. Preserve the "
    "reading order and keep text verbatim. Do not add commentary. If there is "
    'no readable text, return {"blocks":[]}. Ignore page numbers, headers, footers, '
    "and decorative elements. For figures (images, charts, schematics, etc.) "
    "describe the content in text in as much detail as possible."
)


@dataclass
class Segment:
    """Chunking segment derived from a page."""

    text: str
    title: str | None = None
    level: int | None = None


class BlockType(str, Enum):
    """Allowed block types for structured OCR."""

    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST = "list"
    TABLE = "table"
    FIGURE = "figure"
    CAPTION = "caption"


class Block(BaseModel):
    """Structured OCR block schema for response_format."""

    model_config = ConfigDict(extra="ignore")

    type: BlockType = Field(
        default=BlockType.PARAGRAPH,
        description="Block type (heading, paragraph, list, table).",
    )
    text: str = Field(..., description="Verbatim block text.")
    level: int | None = Field(default=None, description="Heading level if available.")

    @field_validator("type", mode="before")
    @classmethod
    def _normalize_type(cls, value: object) -> BlockType:
        if isinstance(value, BlockType):
            return value
        if isinstance(value, str):
            cleaned = value.strip().lower()
            try:
                return BlockType(cleaned)
            except ValueError:
                return BlockType.PARAGRAPH
        return BlockType.PARAGRAPH


class Page(BaseModel):
    """Structured OCR output for a page."""

    model_config = ConfigDict(extra="ignore")

    blocks: list[Block] = Field(default_factory=list)


def _load_pymupdf() -> object:
    """Import PyMuPDF lazily and raise a clear error if missing."""
    try:
        import pymupdf  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - import guard
        raise SystemExit(
            "PyMuPDF is required to read PDFs. Install it with `pip install pymupdf`."
        ) from exc
    return pymupdf


def _resolve_pdf_paths(input_path: Path, recursive: bool) -> list[Path]:
    """Return a sorted list of PDF files from a file or directory input."""
    if input_path.is_file():
        if input_path.suffix.lower() != ".pdf":
            raise ValueError("input_path must point to a PDF file.")
        return [input_path]

    if not input_path.is_dir():
        raise ValueError("input_path must be a PDF file or directory.")

    if recursive:
        candidates = [p for p in input_path.rglob("*") if p.is_file()]
    else:
        candidates = [p for p in input_path.iterdir() if p.is_file()]

    return sorted([p for p in candidates if p.suffix.lower() == ".pdf"])


def _looks_like_toc(text: str) -> bool:
    """Heuristic for detecting table-of-contents style pages."""
    # look for repeated "title .... page" patterns across many lines.
    # TOCs are usually noisy for retrieval and waste embedding budget.
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 6:
        return False

    dotted = sum(1 for line in lines if re.search(r"\.{2,}\s*\d{1,4}$", line))
    numbered = sum(1 for line in lines if re.search(r"\s\d{1,4}$", line))
    score = max(dotted, numbered) / len(lines)
    return score >= 0.3


def _should_skip_page(
    text: str,
    *,
    min_page_characters: int,
    min_page_words: int,
    skip_patterns: list[re.Pattern[str]],
    skip_toc_detection: bool,
) -> bool:
    """Decide whether to drop a page based on text heuristics."""
    # front/back matter and sparse pages add noise to RAG indexes.
    # reject pages that are too short or match known filler patterns.
    stripped = text.strip()
    if not stripped:
        return True

    normalized = " ".join(stripped.split())
    if min_page_characters and len(normalized) < min_page_characters:
        return True
    if min_page_words and len(normalized.split()) < min_page_words:
        return True

    first_line = ""
    for line_text in stripped.splitlines():
        stripped_line = line_text.strip()
        if stripped_line:
            first_line = stripped_line.lower()
            break

    if first_line and any(pattern.search(first_line) for pattern in skip_patterns):
        return True

    return bool(skip_toc_detection and _looks_like_toc(stripped))


def _chunk_text(
    text: str,
    tokenizer: AutoTokenizer,
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    """Split text into overlapping token windows for embedding."""
    # the embedding model has a hard context window we must respect.
    # let the tokenizer produce overlapping windows capped at chunk_size.
    if not text.strip():
        return []
    if chunk_size <= 0:
        return [text.strip()]
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size.")

    # some tokenizers can panic on unexpected characters; keep the pipeline moving.
    # try tokenization, then sanitize/retry, then use char windows.
    try:
        encoding = tokenizer(
            text,
            add_special_tokens=False,
            max_length=chunk_size,
            truncation=True,
            stride=chunk_overlap,
            return_overflowing_tokens=True,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
    except Exception:
        cleaned = _sanitize_text(text)
        if cleaned and cleaned != text:
            try:
                encoding = tokenizer(
                    cleaned,
                    add_special_tokens=False,
                    max_length=chunk_size,
                    truncation=True,
                    stride=chunk_overlap,
                    return_overflowing_tokens=True,
                    return_attention_mask=False,
                    return_token_type_ids=False,
                )
                text = cleaned
            except Exception:
                return _fallback_chunk_text(cleaned, chunk_size, chunk_overlap)
        else:
            return _fallback_chunk_text(cleaned, chunk_size, chunk_overlap)

    input_ids = encoding.get("input_ids", [])
    if not input_ids:
        return []
    if isinstance(input_ids[0], int):
        input_ids = [input_ids]

    chunks: list[str] = []
    for chunk_ids in input_ids:
        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True).strip()
        if chunk_text:
            chunks.append(chunk_text)

    return chunks


def _sanitize_text(text: str) -> str:
    """Remove invalid unicode sequences that can crash some tokenizers."""
    # OCR sometimes yields unpaired surrogates or invalid UTF-8 sequences.
    # round-trip through UTF-8 with "ignore" to drop problematic bytes.
    return text.encode("utf-8", "ignore").decode("utf-8", "ignore")


def _fallback_chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Chunk by character window when tokenization is unavailable."""
    # if the tokenizer panics, approximate chunking still produces usable text.
    # treat 1 token ~= 4 chars to create overlapping character windows.
    if not text.strip():
        return []
    if chunk_size <= 0:
        return [text.strip()]

    char_limit = max(chunk_size * 4, 1)
    char_overlap = max(chunk_overlap * 4, 0)
    step = max(char_limit - char_overlap, 1)

    chunks: list[str] = []
    for start in range(0, len(text), step):
        chunk = text[start : start + char_limit].strip()
        if chunk:
            chunks.append(chunk)
        if start + char_limit >= len(text):
            break
    return chunks


def _parse_structured_page(response_text: str) -> Page | None:
    """Parse structured OCR output into a Pydantic model."""
    # some providers return raw JSON even with response_format enabled.
    # attempt a strict JSON parse into the Page schema.
    try:
        return Page.model_validate_json(response_text)
    except ValidationError:
        return None


def _token_count(tokenizer: AutoTokenizer, text: str) -> int:
    """Return the number of tokens for a text string."""
    # we need approximate token sizes to merge blocks for RAG.
    # try tokenization, then sanitize/retry, then approximate by chars.
    try:
        return len(tokenizer.encode(text, add_special_tokens=False))
    except Exception:
        cleaned = _sanitize_text(text)
        if cleaned and cleaned != text:
            try:
                return len(tokenizer.encode(cleaned, add_special_tokens=False))
            except Exception:
                pass
        approx = len(cleaned) if cleaned else len(text)
        return max(1, approx // 4)


def _segments_from_blocks(
    blocks: list[Block],
    tokenizer: AutoTokenizer,
    *,
    max_tokens: int,
    min_tokens: int,
) -> list[Segment]:
    """Split blocks into heading-scoped segments with size limits."""
    # RAG works better with coherent sections than isolated blocks.
    # collect blocks under the latest heading until a size budget is hit.
    if not blocks:
        return []

    segments: list[Segment] = []
    current_parts: list[str] = []
    active_title: str | None = None
    active_level: int | None = None
    current_tokens = 0

    def render_segment_text(body: str) -> str:
        # headings are useful retrieval context but should not stand alone.
        # prefix the heading once if the body does not already include it.
        if active_title and not body.lower().startswith(active_title.lower()):
            return f"{active_title}\n{body}".strip()
        return body

    def flush() -> None:
        nonlocal current_parts, current_tokens
        body = "\n".join(part for part in current_parts if part.strip()).strip()
        if body:
            segments.append(
                Segment(
                    text=render_segment_text(body),
                    title=active_title,
                    level=active_level,
                )
            )
        current_parts = []
        current_tokens = 0

    for block in blocks:
        text = block.text.strip()
        if not text:
            continue

        if block.type == BlockType.HEADING:
            flush()
            active_title = text
            active_level = block.level
            continue

        block_tokens = _token_count(tokenizer, text) if max_tokens > 0 else 0
        if not current_parts:
            current_parts = [text]
            current_tokens = block_tokens
            continue

        if (
            max_tokens > 0
            and current_tokens + block_tokens > max_tokens
            and current_tokens >= min_tokens
        ):
            flush()
            current_parts = [text]
            current_tokens = block_tokens
        else:
            current_parts.append(text)
            current_tokens += block_tokens

    flush()
    return segments


def _transcribe_page(
    client: object,
    *,
    model: str,
    image_bytes: bytes,
    prompt: str,
    temperature: float,
    max_output_tokens: int | None,
    seed: int | None,
    max_retries: int,
    retry_base_seconds: float,
    response_format: type[BaseModel] | None,
) -> tuple[str, Page | None]:
    """Call the multimodal model to OCR a single page image."""
    # the OCR model expects a prompt plus an image payload.
    # send a chat message with the prompt and base64-encoded PNG.
    image_b64 = base64.b64encode(image_bytes).decode("ascii")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                },
            ],
        }
    ]

    for attempt in range(max_retries + 1):
        try:
            request_kwargs: dict[str, object] = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
            }
            if max_output_tokens is not None:
                request_kwargs["max_tokens"] = max_output_tokens
            if seed is not None:
                request_kwargs["seed"] = seed
            if response_format is not None:
                request_kwargs["response_format"] = response_format

            # structured OCR returns parsed content when response_format is set.
            # use parse() to capture message.parsed without extra JSON parsing.
            response = client.chat.completions.parse(**request_kwargs)
            message = response.choices[0].message
            content = message.content if message else ""
            parsed_page: Page | None = None
            if message is not None and hasattr(message, "parsed"):
                parsed = message.parsed
                if isinstance(parsed, Page):
                    parsed_page = parsed
                elif isinstance(parsed, BaseModel):
                    parsed_page = None
                elif isinstance(parsed, dict) and response_format is not None:
                    try:
                        parsed_page = response_format.model_validate(parsed)
                    except ValidationError:
                        parsed_page = None
            return (content.strip() if content else ""), parsed_page
        except Exception as exc:  # pragma: no cover - network errors
            if attempt >= max_retries:
                raise exc
            time.sleep(retry_base_seconds * (2**attempt))

    return "", None


@contextmanager
def _page_indices(page_limit: int, show_progress: bool, label: str):
    """Yield page indices, optionally wrapped in a progress bar."""
    indices = range(page_limit)
    if show_progress:
        with click.progressbar(indices, label=label) as progress:
            yield progress
    else:
        yield indices


def _compile_skip_patterns(
    skip_pattern: tuple[str, ...],
    use_default_skip_patterns: bool,
) -> list[re.Pattern[str]]:
    """Compile regex patterns that identify filler pages."""
    # reusable regexes are faster and keep skip logic centralized.
    # merge default and custom patterns, then compile with IGNORECASE.
    patterns: list[str] = list(skip_pattern)
    if use_default_skip_patterns:
        patterns = list(DEFAULT_SKIP_PATTERNS) + patterns
    return [re.compile(pattern, re.IGNORECASE) for pattern in patterns]


def _resolve_openai_api_key() -> str:
    """Load the API key from environment variables."""
    # keys should be sourced from env to avoid shell history leaks.
    # check common env var names used for OpenAI/Gemini-compatible keys.
    openai_api_key = (
        os.getenv("OPENAI_API_KEY")
        or os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
    )
    if openai_api_key is None:
        raise ValueError(
            "API key not found. Set OPENAI_API_KEY (or GEMINI_API_KEY/GOOGLE_API_KEY)."
        )
    return openai_api_key


def _get_openai_client(api_key: str, base_url: str | None) -> object:
    """Instantiate an OpenAI-compatible client."""
    client_kwargs = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
    return OpenAI(**client_kwargs)


def _collect_records(
    pdf_paths: list[Path],
    pymupdf: object,
    client: object,
    tokenizer: AutoTokenizer,
    *,
    chunk_size: int,
    chunk_overlap: int,
    model: str,
    prompt: str,
    temperature: float,
    max_output_tokens: int | None,
    seed: int | None,
    max_retries: int,
    retry_base_seconds: float,
    dpi: int,
    max_pages_per_doc: int | None,
    skip_front_pages: int,
    skip_back_pages: int,
    min_page_characters: int,
    min_page_words: int,
    skip_patterns: list[re.Pattern[str]],
    skip_toc_detection: bool,
    show_progress: bool,
    structured_ocr: bool,
    source_root: Path,
) -> list[dict[str, object]]:
    """Process PDFs into a list of dataset records."""
    records: list[dict[str, object]] = []
    # segment sizes should roughly align with final chunk sizes for RAG.
    # derive a max/min token budget from chunk_size.
    segment_max_tokens = chunk_size if chunk_size > 0 else 0
    segment_min_tokens = max(64, chunk_size // 4) if chunk_size > 0 else 0

    for pdf_path in pdf_paths:
        try:
            doc = pymupdf.open(pdf_path)
        except Exception as exc:  # pragma: no cover - backend errors
            click.echo(f"Skipping {pdf_path.name}: failed to open PDF ({exc}).")
            continue

        with doc:
            page_count = doc.page_count
            max_pages = max_pages_per_doc or page_count
            page_limit = min(page_count, max_pages)
            source = pdf_path.relative_to(source_root).as_posix()

            with _page_indices(
                page_limit, show_progress, f"OCR {pdf_path.name}"
            ) as page_iter:
                for page_index in page_iter:
                    if skip_front_pages and page_index < skip_front_pages:
                        continue
                    if skip_back_pages and page_index >= page_limit - skip_back_pages:
                        continue
                    try:
                        # render each page as an image to feed the OCR model.
                        # rasterize the page into PNG bytes for the API call.
                        page = doc.load_page(page_index)
                        pixmap = page.get_pixmap(dpi=dpi)
                        image_bytes = pixmap.tobytes("png")

                        response_text, structured_page = _transcribe_page(
                            client,
                            model=model,
                            image_bytes=image_bytes,
                            prompt=prompt,
                            temperature=temperature,
                            max_output_tokens=max_output_tokens,
                            seed=seed,
                            max_retries=max_retries,
                            retry_base_seconds=retry_base_seconds,
                            response_format=(Page if structured_ocr else None),
                        )
                    except Exception as exc:  # pragma: no cover - backend errors
                        click.echo(
                            f"Skipping {pdf_path.name} page {page_index + 1}: {exc}"
                        )
                        continue

                    # some providers return JSON as plain text even with parsing.
                    # parse the raw response into the Page schema when needed.
                    if structured_ocr and structured_page is None and response_text:
                        structured_page = _parse_structured_page(response_text)
                    page_text = response_text
                    if structured_page is not None and structured_page.blocks:
                        page_text = "\n".join(
                            block.text for block in structured_page.blocks if block.text
                        )

                    if _should_skip_page(
                        page_text,
                        min_page_characters=min_page_characters,
                        min_page_words=min_page_words,
                        skip_patterns=skip_patterns,
                        skip_toc_detection=skip_toc_detection,
                    ):
                        continue

                    segments = [Segment(text=page_text)]
                    if structured_page is not None and structured_page.blocks:
                        segments = _segments_from_blocks(
                            structured_page.blocks,
                            tokenizer,
                            max_tokens=segment_max_tokens,
                            min_tokens=segment_min_tokens,
                        )

                    for segment_index, segment in enumerate(segments):
                        # final chunks must fit the embedding context window.
                        # split each segment into token-sized windows.
                        chunks = _chunk_text(
                            segment.text, tokenizer, chunk_size, chunk_overlap
                        )
                        for chunk_index, chunk in enumerate(chunks):
                            records.append(
                                {
                                    "text": chunk,
                                    "source": source,
                                    "page_index": page_index + 1,
                                    "segment_index": segment_index,
                                    "chunk_index": chunk_index,
                                    "section_title": segment.title,
                                    "section_level": segment.level,
                                }
                            )

    return records


def _save_dataset(
    records: list[dict[str, object]], output_dir: Path
) -> datasets.DatasetDict:
    """Save records to disk as a HuggingFace DatasetDict."""
    dataset = datasets.Dataset.from_list(records)
    dataset_dict = datasets.DatasetDict({"train": dataset})
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_dict.save_to_disk(output_dir)
    click.echo(f"Saved dataset with {len(dataset)} chunks to {output_dir}.")
    return dataset_dict


@click.command()
@click.option(
    "--input-path",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to a PDF file or directory of PDFs.",
)
@click.option(
    "--output-dir",
    default=Path("hf_dataset"),
    type=click.Path(path_type=Path),
    show_default=True,
    help="Directory to save the HuggingFace dataset.",
)
@click.option(
    "--recursive/--no-recursive",
    default=False,
    show_default=True,
    help="Recursively scan directories for PDFs.",
)
@click.option(
    "--tokenizer-name",
    default="BAAI/bge-m3",
    show_default=True,
    help="Tokenizer used for chunking.",
)
@click.option(
    "--chunk-size",
    default=512,
    show_default=True,
    help="Max tokens per chunk.",
)
@click.option(
    "--chunk-overlap",
    default=64,
    show_default=True,
    help="Token overlap between chunks.",
)
@click.option(
    "--model",
    default="gemini-2.5-flash",
    show_default=True,
    help="Model used for OCR (OpenAI-compatible endpoint).",
)
@click.option(
    "--openai-base-url",
    default=None,
    help="Overrides OPENAI_BASE_URL for the OpenAI-compatible endpoint.",
)
@click.option(
    "--prompt",
    default=DEFAULT_PROMPT,
    show_default=False,
    help="Prompt passed to the OCR model.",
)
@click.option("--temperature", default=0.0, show_default=True, type=float)
@click.option("--max-output-tokens", default=4096, show_default=True, type=int)
@click.option("--seed", default=None, type=int)
@click.option("--dpi", default=300, show_default=True, type=int)
@click.option("--max-pages-per-doc", default=None, type=int)
@click.option("--skip-front-pages", default=0, show_default=True, type=int)
@click.option("--skip-back-pages", default=0, show_default=True, type=int)
@click.option("--min-page-characters", default=200, show_default=True, type=int)
@click.option("--min-page-words", default=0, show_default=True, type=int)
@click.option(
    "--skip-toc-detection/--no-skip-toc-detection",
    default=True,
    show_default=True,
)
@click.option(
    "--use-default-skip-patterns/--no-default-skip-patterns",
    default=True,
    show_default=True,
)
@click.option(
    "--skip-pattern",
    multiple=True,
    help="Regex pattern to skip pages if it matches the first line.",
)
@click.option("--max-retries", default=3, show_default=True, type=int)
@click.option("--retry-base-seconds", default=2.0, show_default=True, type=float)
@click.option(
    "--show-progress/--no-show-progress",
    default=True,
    show_default=True,
    help="Show a progress indicator while OCR runs.",
)
@click.option(
    "--structured-ocr/--no-structured-ocr",
    default=False,
    show_default=True,
    help="Request structured JSON blocks from the OCR model (overrides --prompt).",
)
@click.option("--save-to-hub", is_flag=True)
@click.option("--hub-repo-id", default=None)
def main(
    input_path: Path,
    output_dir: Path,
    recursive: bool,
    tokenizer_name: str,
    chunk_size: int,
    chunk_overlap: int,
    model: str,
    openai_base_url: str | None,
    prompt: str,
    temperature: float,
    max_output_tokens: int,
    seed: int | None,
    dpi: int,
    max_pages_per_doc: int | None,
    skip_front_pages: int,
    skip_back_pages: int,
    min_page_characters: int,
    min_page_words: int,
    skip_toc_detection: bool,
    use_default_skip_patterns: bool,
    skip_pattern: tuple[str, ...],
    max_retries: int,
    retry_base_seconds: float,
    show_progress: bool,
    structured_ocr: bool,
    save_to_hub: bool,
    hub_repo_id: str | None,
) -> None:
    """Convert PDFs to a chunked HuggingFace dataset."""
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")

    pdf_paths = _resolve_pdf_paths(input_path, recursive)
    if not pdf_paths:
        raise ValueError("No PDF files found to process.")

    compiled_patterns = _compile_skip_patterns(skip_pattern, use_default_skip_patterns)
    openai_api_key = _resolve_openai_api_key()
    prompt_text = STRUCTURED_PROMPT if structured_ocr else prompt

    max_output = max_output_tokens if max_output_tokens > 0 else None

    pymupdf = _load_pymupdf()
    client = _get_openai_client(openai_api_key, openai_base_url)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

    source_root = input_path if input_path.is_dir() else input_path.parent

    records = _collect_records(
        pdf_paths,
        pymupdf,
        client,
        tokenizer,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        model=model,
        prompt=prompt_text,
        temperature=temperature,
        max_output_tokens=max_output,
        seed=seed,
        max_retries=max_retries,
        retry_base_seconds=retry_base_seconds,
        dpi=dpi,
        max_pages_per_doc=max_pages_per_doc,
        skip_front_pages=skip_front_pages,
        skip_back_pages=skip_back_pages,
        min_page_characters=min_page_characters,
        min_page_words=min_page_words,
        skip_patterns=compiled_patterns,
        skip_toc_detection=skip_toc_detection,
        show_progress=show_progress,
        structured_ocr=structured_ocr,
        source_root=source_root,
    )

    if not records:
        raise ValueError("No text chunks were produced.")

    dataset_dict = _save_dataset(records, output_dir)

    if save_to_hub:
        if not hub_repo_id:
            raise ValueError("hub_repo_id must be provided when save_to_hub is True.")
        dataset_dict.push_to_hub(hub_repo_id, private=False)


if __name__ == "__main__":
    load_dotenv()

    main()
