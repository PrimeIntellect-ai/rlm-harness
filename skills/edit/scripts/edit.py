"""Edit tool — safe single-occurrence string replacement."""

from pathlib import Path

SCHEMA = {
    "type": "function",
    "function": {
        "name": "edit",
        "description": (
            "Replace a unique string in a file. "
            "old_str must appear exactly once in the file."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path to edit."},
                "old_str": {
                    "type": "string",
                    "description": "The exact string to find (must be unique).",
                },
                "new_str": {"type": "string", "description": "The replacement string."},
            },
            "required": ["path", "old_str", "new_str"],
        },
    },
}


def run(
    path: str,
    old_str: str,
    new_str: str,
    *,
    cwd: str,
    **_,
) -> str:
    """Safe single-occurrence string replacement."""
    filepath = Path(cwd) / path
    if not filepath.exists():
        return f"Error: {path} not found"
    try:
        content = filepath.read_text()
    except Exception as e:
        return f"Error reading {path}: {e}"

    count = content.count(old_str)
    if count == 0:
        return f"Error: string not found in {path}"
    if count > 1:
        return f"Error: found {count} occurrences, need exactly 1"

    filepath.write_text(content.replace(old_str, new_str, 1))
    return f"Edited {path}"
