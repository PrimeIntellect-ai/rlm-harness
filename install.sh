#!/usr/bin/env bash
set -e
command -v uv >/dev/null 2>&1 || {
    command -v curl >/dev/null 2>&1 || { apt-get update -qq && apt-get install -y -qq curl; }
    curl -LsSf https://astral.sh/uv/install.sh | sh
    [ -f "$HOME/.local/bin/env" ] && . "$HOME/.local/bin/env" || export PATH="$HOME/.local/bin:$PATH"
}
command -v rg >/dev/null 2>&1 || { apt-get update -qq && apt-get install -y -qq ripgrep; }
uv tool install --python 3.11 "rlm @ git+https://${GH_TOKEN}@github.com/PrimeIntellect-ai/rlm.git"
