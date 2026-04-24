# AGENTS.md

## Writing code

- **Minimal try/except**: let errors propagate — silent failures hide bugs. Only catch for intentional fault tolerance (retries, robustness).
- **Targeted comments**: don't explain your work process or reference old code. Use comments sparingly to clarify ambiguous logic or non-obvious constraints.
- **No "this replaced that" comments**: never write comments or docstrings that refer to code that used to exist (e.g. "no more `PARAMETERS` dict", "gone — no hand-written `main()`", "replaces the old argparse wrapper", "used to do X, now does Y"). They make sense only in the PR that introduced the change; outside that context they're noise that confuses readers who never saw the old code. Describe what the code does now, not what it isn't anymore.

## Running code

- **Always use uv**: run code with `uv run`, never raw `python`.
- **Adding dependencies**: add to `pyproject.toml` and run `uv sync` (or `uv sync --group dev` for dev-only) to install and lock.

## Testing

- Run the suite with `uv run pytest tests/`.
- Write tests as plain functions; don't use class-based tests.
- **Conservative test additions**: don't add new tests unless the user asks or it's clearly necessary. Editing existing tests is fine.
- **Test what matters**: only test code with clear, isolated logic. If you need to patch everything to make it testable, it's probably not worth testing.

## Git

- **Branch prefixes**: use `feat/`, `fix/`, `chore/`, `docs/`, `tests/`.

## GitHub

- **Draft PRs**: always create PRs as drafts (`gh pr create --draft`) to avoid triggering CI unnecessarily.
- **Pull requests**: do not include a "test plan" section unless you actually ran tests or the user explicitly asked for one.
