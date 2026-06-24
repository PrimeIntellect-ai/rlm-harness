"""Built-in ``search`` skill — web search via Serper.

Enabled via ``RLM_SKILLS``; pre-imported into the IPython kernel so the agent calls
``await search(query="...")``. Needs ``SERPER_API_KEY``. Ported from the Serper ``websearch``
skill in research-environments/rlm_browsecomp.
"""

from __future__ import annotations

import asyncio
import os

import httpx

SERPER_URL = "https://google.serper.dev/search"


def format_results(results, query: str) -> str:
    sections: list[str] = []
    for i, result in enumerate(results, 1):
        title = (result.get("title") or "").strip() or "Untitled"
        lines = [f"Result {i}: {title}"]
        link = (result.get("link") or "").strip()
        if link:
            lines.append(f"URL: {link}")
        snippet = (result.get("snippet") or "").strip()
        if snippet:
            lines.append(f"  - {snippet}")
        sections.append("\n".join(lines))
    if not sections:
        return f"No results returned for query: {query}"
    return "\n\n---\n\n".join(sections)


def search(query: str, num_results: int = 5) -> str:
    """Run a synchronous Serper web search and return formatted results."""
    api_key = os.environ.get("SERPER_API_KEY", "")
    if not api_key:
        return "Error: SERPER_API_KEY environment variable is not set"
    response = httpx.post(
        SERPER_URL,
        json={"q": query},
        headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
        timeout=45,
    )
    response.raise_for_status()
    organic = response.json().get("organic") or []
    return format_results(organic[:num_results], query)


async def run(query: str, *, num_results: int = 5) -> str:
    """Run a web search via Serper and return formatted results.

    Args:
        query: Web search query.
        num_results: Number of results to return.

    Returns:
        Formatted results (title, URL, snippet).
    """
    return await asyncio.to_thread(search, query, num_results)
