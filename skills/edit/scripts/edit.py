"""Edit tool — safe single-occurrence string replacement."""

import asyncio
from pathlib import Path

PARAMETERS = {
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
}


async def run(
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


def main():
    import argparse
    import os

    parser = argparse.ArgumentParser(prog="edit")
    parser.add_argument("path", help="File path to edit.")
    parser.add_argument("old_str", help="The exact string to find (must be unique).")
    parser.add_argument("new_str", help="The replacement string.")
    parser.add_argument("--cwd", default=os.getcwd(), help="Working directory.")
    args = parser.parse_args()

    print(asyncio.run(run(args.path, args.old_str, args.new_str, cwd=args.cwd)))


if __name__ == "__main__":
    main()
