#!/usr/bin/env bash
set -eo pipefail

# Ensure curl is available (Multi-SWE-RL images may lack it)
command -v curl >/dev/null 2>&1 || { apt-get update -qq && apt-get install -y -qq curl; }

# Always install latest uv (sandbox images may have stale versions)
curl -LsSf https://astral.sh/uv/install.sh | sh
[ -f "$HOME/.local/bin/env" ] && . "$HOME/.local/bin/env" || export PATH="$HOME/.local/bin:$PATH"

# Install ripgrep if missing
command -v rg >/dev/null 2>&1 || { apt-get update -qq && apt-get install -y -qq ripgrep; }

# Clone the repo
RLM_REPO_URL="${RLM_REPO_URL:-github.com/PrimeIntellect-ai/rlm.git}"
RLM_REPO_BRANCH="${RLM_REPO_BRANCH:-main}"
RLM_CHECKOUT="${RLM_CHECKOUT_PATH:-/tmp/rlm-checkout}"
rm -rf "$RLM_CHECKOUT"
case "$RLM_REPO_URL" in
    https://*|http://*)
        CLONE_URL="$RLM_REPO_URL"
        ;;
    github.com/*)
        CLONE_URL="https://${GH_TOKEN:+${GH_TOKEN}@}${RLM_REPO_URL}"
        ;;
    *)
        CLONE_URL="$RLM_REPO_URL"
        ;;
esac
git clone --depth 1 --branch "$RLM_REPO_BRANCH" "$CLONE_URL" "$RLM_CHECKOUT"

# Install rlm as an isolated CLI tool (separate venv, on PATH).
# Discover workspace skill packages and include them with their
# executables so that skills are both importable and on PATH.
SKILL_ARGS=""
for skill_dir in "$RLM_CHECKOUT"/skills/*/; do
    [ -f "$skill_dir/pyproject.toml" ] || continue
    skill_name=$(grep '^name' "$skill_dir/pyproject.toml" | head -1 | sed 's/.*"\(.*\)".*/\1/')
    SKILL_ARGS="$SKILL_ARGS --with-editable $skill_dir --with-executables-from $skill_name"
done
uv tool install --python 3.10 --editable "$RLM_CHECKOUT" $SKILL_ARGS
