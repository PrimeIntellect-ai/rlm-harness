#!/usr/bin/env bash
set -e
command -v uv >/dev/null 2>&1 || { curl -LsSf https://astral.sh/uv/install.sh | sh; source "$HOME/.local/bin/env"; }
command -v rg >/dev/null 2>&1 || { apt-get update -qq && apt-get install -y -qq ripgrep; }
uv tool install --python 3.11 "rlm @ git+https://${GH_TOKEN}@github.com/PrimeIntellect-ai/rlm.git"
