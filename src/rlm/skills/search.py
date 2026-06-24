"""Built-in ``search`` skill — web search via Exa.

Enabled via ``RLM_SKILLS``; pre-imported into the IPython kernel so the agent calls
``await search(query="...")``. Needs ``EXA_API_KEY``. Ported from the Exa ``websearch``
skill in research-environments/rlm_browsecomp.
"""

from __future__ import annotations

import asyncio
import os

from exa_py import Exa


def format_results(results, query: str) -> str:
    sections: list[str] = []
    for i, result in enumerate(results, 1):
        lines = [f"Result {i}: {getattr(result, 'title', '') or 'Untitled'}"]
        url = getattr(result, "url", "")
        if url:
            lines.append(f"URL: {url}")
        for highlight in getattr(result, "highlights", None) or []:
            clean = " ".join(str(highlight).split())
            if clean:
                lines.append(f"  - {clean}")
        sections.append("\n".join(lines))
    if not sections:
        return f"No results returned for query: {query}"
    return "\n\n---\n\n".join(sections)


def search(query: str, num_results: int = 5) -> str:
    """Run a synchronous Exa web search and return formatted results."""
    api_key = os.environ.get("EXA_API_KEY", "")
    if not api_key:
        return "Error: EXA_API_KEY environment variable is not set"
    response = Exa(api_key=api_key).search_and_contents(
        query, num_results=num_results, highlights=True
    )
    return format_results(response.results, query)


async def run(query: str, *, num_results: int = 5) -> str:
    """Run a web search via Exa and return formatted results.

    Args:
        query: Web search query.
        num_results: Number of results to return.

    Returns:
        Formatted results (title, URL, highlights).
    """
    return await asyncio.to_thread(search, query, num_results)
