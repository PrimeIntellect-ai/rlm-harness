"""Built-in ``search`` skill — web search via Exa.

Enabled via ``RLM_SKILLS``; pre-imported into the IPython kernel so the agent calls
``await search(queries=["..."])``. Needs ``EXA_API_KEY``. Ported from the Exa ``websearch``
skill in research-environments/rlm_browsecomp.
"""

from __future__ import annotations

import asyncio
import os

from exa_py import Exa


def _format_results(results, query: str) -> str:
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


def _search_one(exa: Exa, query: str, num_results: int) -> str:
    response = exa.search_and_contents(query, num_results=num_results, highlights=True)
    return _format_results(response.results, query)


async def run(
    queries: list[str],
    *,
    num_results: int | None = None,
    max_output: int = 8192,
) -> str:
    """Run web searches via Exa in parallel and return formatted results.

    Pass multiple queries to search different angles at once.

    Args:
        queries: Web search queries.
        num_results: Results per query. Defaults to ``$RLM_SEARCH_NUM_RESULTS`` or 5.
        max_output: Truncate the combined output to this many chars.

    Returns:
        Formatted results (title, URL, highlights) concatenated across queries.
    """
    api_key = os.environ.get("EXA_API_KEY", "")
    if not api_key:
        return "Error: EXA_API_KEY environment variable is not set"

    if num_results is None:
        num_results = int(os.environ.get("RLM_SEARCH_NUM_RESULTS", "5"))
    max_concurrent = int(os.environ.get("RLM_SEARCH_MAX_CONCURRENT", "10"))
    if max_concurrent < 1:
        raise ValueError(
            f"RLM_SEARCH_MAX_CONCURRENT must be >= 1, got {max_concurrent}"
        )

    queries = queries[:max_concurrent]
    exa = Exa(api_key=api_key)

    async def _run_query(query: str) -> str:
        try:
            result = await asyncio.to_thread(_search_one, exa, query, num_results)
        except Exception as e:
            result = f"Error searching for '{query}': {e}"
        return f'Results for query "{query}":\n\n{result}'

    parts = await asyncio.gather(*[_run_query(query) for query in queries])
    output = "\n\n---\n\n".join(parts)

    if len(output) > max_output:
        half = max_output // 2
        output = (
            output[:half]
            + f"\n... [output truncated, {len(output)} chars total] ...\n"
            + output[-half:]
        )
    return output
