#!/usr/bin/env bash
set -eo pipefail

# Ensure curl is available (Multi-SWE-RL images may lack it)
command -v curl >/dev/null 2>&1 || { apt-get update -qq && apt-get install -y -qq curl; }

# Install uv if missing
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    [ -f "$HOME/.local/bin/env" ] && . "$HOME/.local/bin/env" || export PATH="$HOME/.local/bin:$PATH"
fi

# Install ripgrep if missing
command -v rg >/dev/null 2>&1 || { apt-get update -qq && apt-get install -y -qq ripgrep; }

# Clone the repo
RLM_CHECKOUT="${RLM_CHECKOUT_PATH:-/tmp/rlm-checkout}"
rm -rf "$RLM_CHECKOUT"
git clone --depth 1 "https://${GH_TOKEN:+${GH_TOKEN}@}github.com/PrimeIntellect-ai/rlm.git" "$RLM_CHECKOUT"

# Install rlm as an isolated CLI tool (separate venv, on PATH).
# Discover workspace skill packages and include them so that
# get_installed_skills() finds them via importlib.metadata.
SKILL_ARGS=""
for skill_dir in "$RLM_CHECKOUT"/skills/*/; do
    [ -f "$skill_dir/pyproject.toml" ] && SKILL_ARGS="$SKILL_ARGS --with-editable $skill_dir"
done
eval uv tool install --python 3.10 --editable "$RLM_CHECKOUT" $SKILL_ARGS
