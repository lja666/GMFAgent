# -*- coding: utf-8 -*-
"""RAG retrieval from LanceDB with fallback when search fails."""
from pathlib import Path
from typing import Optional

try:
    from config import (
        LANCEDB_PATH,
        LANCEDB_TABLE,
        RAG_FALLBACK_TOKEN_LIMIT,
        RAG_CONTENT_COLUMN,
        RAG_RERANK_WEIGHT,
    )
except ImportError:
    LANCEDB_PATH = "./my_lancedb"
    LANCEDB_TABLE = "rag_table"
    RAG_FALLBACK_TOKEN_LIMIT = 2000
    RAG_CONTENT_COLUMN = "content"
    RAG_RERANK_WEIGHT = 0.7


def _count_tokens(text: str) -> int:
    """Approximate token count. Uses tiktoken if available, else ~3 chars per token."""
    if not text:
        return 0
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except ImportError:
        return max(1, len(text) // 3)


def _get_fallback_content(
    table,
    token_limit: int = RAG_FALLBACK_TOKEN_LIMIT,
    content_column: str = RAG_CONTENT_COLUMN,
    max_rows: int = 100,
) -> str:
    """
    When search fails: return content from first rows until token limit.
    """
    try:
        df = table.to_pandas()
        if len(df) > max_rows:
            df = df.head(max_rows)
    except Exception:
        return ""

    if df.empty:
        return ""

    col = content_column
    if col not in df.columns:
        candidates = ["content", "text", "body", "chunk"]
        for c in candidates:
            if c in df.columns:
                col = c
                break
        else:
            col = df.columns[0]

    parts = []
    total_tokens = 0
    for _, row in df.iterrows():
        val = row.get(col)
        if val is None or (isinstance(val, float) and str(val) == "nan"):
            continue
        s = str(val).strip()
        if not s:
            continue
        tokens = _count_tokens(s)
        if total_tokens + tokens > token_limit:
            remaining = token_limit - total_tokens
            if remaining > 50:
                approx_chars = remaining * 3
                s = s[:approx_chars] + "..." if len(s) > approx_chars else s
                parts.append(s)
            break
        parts.append(s)
        total_tokens += tokens

    return "\n\n---\n\n".join(parts) if parts else ""


def rag_retrieve(
    query: str,
    db_path: Optional[str] = None,
    table_name: Optional[str] = None,
    limit: int = 10,
    fallback_token_limit: Optional[int] = None,
    content_column: Optional[str] = None,
    rerank_weight: Optional[float] = None,
) -> str:
    """
    Retrieve from LanceDB. On search error, return front rows up to token limit.

    Args:
        query: Search query string.
        db_path: LanceDB path (default from config).
        table_name: Table name (default from config).
        limit: Max results for normal search.
        fallback_token_limit: Max tokens for fallback content (default from config).
        content_column: Column name for content (default from config).
        rerank_weight: Reranker weight 0~1, 0.7=mostly semantic (default from config).

    Returns:
        Concatenated content string for LLM context.
    """
    import warnings

    db_path = db_path or LANCEDB_PATH
    table_name = table_name or LANCEDB_TABLE
    fallback_token_limit = fallback_token_limit if fallback_token_limit is not None else RAG_FALLBACK_TOKEN_LIMIT
    content_column = content_column or RAG_CONTENT_COLUMN
    rerank_weight = rerank_weight if rerank_weight is not None else RAG_RERANK_WEIGHT

    try:
        import lancedb
    except ImportError as e:
        warnings.warn(f"lancedb not installed: {e}. RAG disabled.", stacklevel=2)
        return ""

    db_path = Path(db_path)
    if not db_path.exists():
        warnings.warn(f"LanceDB path not found: {db_path}. RAG disabled.", stacklevel=2)
        return ""

    try:
        db = lancedb.connect(str(db_path))
        table = db.open_table(table_name)
    except Exception as e:
        warnings.warn(f"LanceDB connect/open_table failed: {e}. RAG disabled.", stacklevel=2)
        return ""

    try:
        try:
            from lancedb.rerankers import LinearCombinationReranker
            reranker = LinearCombinationReranker(weight=rerank_weight)
            df_results = (
                table.search(query, query_type="hybrid")
                .rerank(reranker)
                .limit(limit)
                .to_pandas()
            )
        except Exception as e:
            warnings.warn(f"LanceDB search failed ({e}). RAG disabled.", stacklevel=2)
            return ""

        if df_results is None or df_results.empty:
            return ""

        col = content_column
        if col not in df_results.columns:
            for c in ["content", "text", "body", "chunk"]:
                if c in df_results.columns:
                    col = c
                    break
            else:
                col = df_results.columns[0]

        parts = []
        total_tokens = 0
        token_limit = fallback_token_limit * 2
        for _, row in df_results.iterrows():
            val = row.get(col)
            if val is None or (isinstance(val, float) and str(val) == "nan"):
                continue
            s = str(val).strip()
            if not s:
                continue
            tokens = _count_tokens(s)
            if total_tokens + tokens > token_limit:
                break
            parts.append(s)
            total_tokens += tokens

        return "\n\n---\n\n".join(parts) if parts else ""

    except Exception as e:
        warnings.warn(f"RAG retrieve error: {e}. RAG disabled.", stacklevel=2)
        return ""


if __name__ == "__main__":
    q = "AbrahamsonEtAl2018 ParkerEtAl2020 ZhaoEtAl2006 ZhaoEtAl2016"
    out = rag_retrieve(q, fallback_token_limit=1500)
    print(f"Retrieved {len(out)} chars, ~{_count_tokens(out)} tokens")
    print(out[:500] + "..." if len(out) > 500 else out)
