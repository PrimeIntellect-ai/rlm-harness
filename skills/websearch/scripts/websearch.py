"""Websearch tool — Google search via Serper API."""

import json
import os
import urllib.error
import urllib.request

PARAMETERS = {
    "type": "object",
    "properties": {
        "queries": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "maxItems": 10,
            "description": (
                "Google search queries (up to 10). "
                "Use multiple queries to search different angles in parallel."
            ),
        }
    },
    "required": ["queries"],
}


def _format_serper_results(data: dict, query: str, num_results: int = 5) -> str:
    """Format a Serper API response into readable text."""
    sections: list[str] = []

    kg = data.get("knowledgeGraph")
    if kg:
        kg_lines: list[str] = []
        title = (kg.get("title") or "").strip()
        if title:
            kg_lines.append(f"Knowledge Graph: {title}")
        description = (kg.get("description") or "").strip()
        if description:
            kg_lines.append(description)
        for key, value in (kg.get("attributes") or {}).items():
            text = str(value).strip()
            if text:
                kg_lines.append(f"{key}: {text}")
        if kg_lines:
            sections.append("\n".join(kg_lines))

    for i, result in enumerate((data.get("organic") or [])[:num_results]):
        title = (result.get("title") or "").strip() or "Untitled"
        lines = [f"Result {i}: {title}"]
        link = (result.get("link") or "").strip()
        if link:
            lines.append(f"URL: {link}")
        snippet = (result.get("snippet") or "").strip()
        if snippet:
            lines.append(snippet)
        sections.append("\n".join(lines))

    people_also_ask = data.get("peopleAlsoAsk") or []
    if people_also_ask:
        max_q = max(1, min(3, len(people_also_ask)))
        questions: list[str] = []
        for item in people_also_ask[:max_q]:
            question = (item.get("question") or "").strip()
            if not question:
                continue
            entry = f"Q: {question}"
            answer = (item.get("snippet") or "").strip()
            if answer:
                entry += f"\nA: {answer}"
            questions.append(entry)
        if questions:
            sections.append("People Also Ask:\n" + "\n".join(questions))

    if not sections:
        return f"No results returned for query: {query}"

    return "\n\n---\n\n".join(sections)


def _fetch_serper(
    query: str, api_key: str, timeout: int = 45, num_results: int = 5
) -> str:
    """Execute a single Serper API search."""
    req = urllib.request.Request(
        "https://google.serper.dev/search",
        data=json.dumps({"q": query}).encode(),
        headers={
            "X-API-KEY": api_key,
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        body = e.read().decode() if e.fp else ""
        raise RuntimeError(f"Serper search error ({e.code}): {body}") from e

    return _format_serper_results(data, query, num_results=num_results)


def run(
    queries: list[str],
    *,
    max_output: int = 8192,
    timeout: int | None = None,
    num_results: int | None = None,
    **_,
) -> str:
    """Run up to 10 Google searches via Serper and return formatted results."""
    api_key = os.environ.get("SERPER_API_KEY", "")
    if not api_key:
        return "Error: SERPER_API_KEY environment variable is not set"

    if timeout is None:
        timeout = int(os.environ.get("RLM_WEBSEARCH_TIMEOUT", "45"))
    if num_results is None:
        num_results = int(os.environ.get("RLM_WEBSEARCH_NUM_RESULTS", "5"))

    queries = queries[:10]
    parts: list[str] = []
    for query in queries:
        try:
            result = _fetch_serper(
                query, api_key, timeout=timeout, num_results=num_results
            )
        except Exception as e:
            result = f"Error searching for '{query}': {e}"
        parts.append(f'Results for query "{query}":\n\n{result}')

    output = "\n\n---\n\n".join(parts)

    if len(output) > max_output:
        half = max_output // 2
        total = len(output)
        output = (
            output[:half]
            + f"\n... [output truncated, {total} chars total] ...\n"
            + output[-half:]
        )

    return output


def main():
    import argparse

    parser = argparse.ArgumentParser(prog="websearch")
    parser.add_argument("queries", nargs="+", help="Google search queries (up to 10).")
    parser.add_argument(
        "--max-output",
        type=int,
        default=8192,
        help="Truncate output to this many chars.",
    )
    parser.add_argument(
        "--timeout", type=int, default=None, help="Per-query HTTP timeout in seconds."
    )
    parser.add_argument(
        "--num-results", type=int, default=None, help="Organic results per query."
    )
    args = parser.parse_args()

    print(
        run(
            args.queries,
            max_output=args.max_output,
            timeout=args.timeout,
            num_results=args.num_results,
        )
    )


if __name__ == "__main__":
    main()
