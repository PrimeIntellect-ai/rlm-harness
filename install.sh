#!/usr/bin/env bash
set -eo pipefail

# Ensure curl is available (Multi-SWE-RL images may lack it)
command -v curl >/dev/null 2>&1 || { apt-get update -qq && apt-get install -y -qq curl; }

# Always install latest uv (sandbox images may have stale versions). Pin
# UV_INSTALL_DIR (uv installer target) and UV_TOOL_BIN_DIR (`uv tool install`
# shim target) to the same directory, and make the PATH export agree — on
# images that set XDG_DATA_HOME independent of HOME (e.g. SWE-rebench-V2
# python images with HOME=/root, XDG_DATA_HOME=/workspace/.local/share),
# both uv and its tool shims otherwise resolve via $XDG_DATA_HOME/../bin
# while a $HOME-based PATH export points at an empty dir — leaving `uv`
# and any `uv tool install`-ed executables (e.g. `rlm`) off PATH.
export UV_INSTALL_DIR="${UV_INSTALL_DIR:-$HOME/.local/bin}"
export UV_TOOL_BIN_DIR="${UV_TOOL_BIN_DIR:-$UV_INSTALL_DIR}"
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$UV_INSTALL_DIR:$PATH"

# Install ripgrep if missing
command -v rg >/dev/null 2>&1 || { apt-get update -qq && apt-get install -y -qq ripgrep; }

# Clone the repo
RLM_REPO_URL="${RLM_REPO_URL:-github.com/PrimeIntellect-ai/rlm.git}"
RLM_REPO_BRANCH="${RLM_REPO_BRANCH:-main}"
RLM_CHECKOUT="${RLM_CHECKOUT_PATH:-/tmp/rlm-checkout}"
if ! git -C "$RLM_CHECKOUT" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
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
    rm -rf "$RLM_CHECKOUT"
    git clone --depth 1 --branch "$RLM_REPO_BRANCH" "$CLONE_URL" "$RLM_CHECKOUT"
fi

# Install rlm as an isolated CLI tool (separate venv, on PATH).
# Skills are owned by the environment (e.g. ComposableEnv uploads them to
# /task/rlm-skills before this script runs).  Discover and install any
# that are present so they're both importable and on PATH.
SKILL_ARGS=""
if [ -d /task/rlm-skills ]; then
    for skill_dir in /task/rlm-skills/*/; do
        [ -f "$skill_dir/pyproject.toml" ] || continue
        skill_name=$(grep '^name' "$skill_dir/pyproject.toml" | head -1 | sed 's/.*"\(.*\)".*/\1/')
        SKILL_ARGS="$SKILL_ARGS --with-editable $skill_dir --with-executables-from $skill_name"
    done
fi
uv tool install --python 3.10 --editable "$RLM_CHECKOUT" $SKILL_ARGS
