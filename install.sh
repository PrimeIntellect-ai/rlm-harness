#!/usr/bin/env bash
set -e
command -v uv >/dev/null 2>&1 || {
    command -v curl >/dev/null 2>&1 || { apt-get update -qq && apt-get install -y -qq curl; }
    curl -LsSf https://astral.sh/uv/install.sh | sh
    [ -f "$HOME/.local/bin/env" ] && . "$HOME/.local/bin/env" || export PATH="$HOME/.local/bin:$PATH"
}
command -v rg >/dev/null 2>&1 || { apt-get update -qq && apt-get install -y -qq ripgrep; }
RLM_REPO_REF="${RLM_REPO_REF:-main}"
RLM_INSTALL_REF_SUFFIX="@${RLM_REPO_REF}"
uv tool install --python 3.11 "rlm @ git+https://x-access-token:${GH_TOKEN}@github.com/PrimeIntellect-ai/rlm.git${RLM_INSTALL_REF_SUFFIX}"
