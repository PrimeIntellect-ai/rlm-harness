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
if [ ! -f "$RLM_CHECKOUT/install.sh" ] || [ ! -f "$RLM_CHECKOUT/pyproject.toml" ]; then
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
SKILL_TOOL_NAMES=""
if [ -d /task/rlm-skills ]; then
    for skill_dir in /task/rlm-skills/*/; do
        [ -f "$skill_dir/pyproject.toml" ] || continue
        skill_name=$(grep '^name' "$skill_dir/pyproject.toml" | head -1 | sed 's/.*"\(.*\)".*/\1/')
        SKILL_ARGS="$SKILL_ARGS --with-editable $skill_dir --with-executables-from $skill_name"
        for candidate in "$skill_dir"/src/*; do
            [ -d "$candidate" ] || continue
            [ "$(basename "$candidate")" = "__pycache__" ] && continue
            SKILL_TOOL_NAMES="$SKILL_TOOL_NAMES $(basename "$candidate")"
            break
        done
    done
fi

# Forward caller-supplied `uv tool install` extras (e.g. "--with numpy
# --with sympy") when set, so environments can inject extra pip deps into
# the rlm tool venv without patching this script. Tokens are word-split by
# the shell, so avoid shell metacharacters (e.g. `>=`) inside package specs.
EXTRA_UV_ARGS=""
if [ -n "${RLM_EXTRA_UV_ARGS:-}" ]; then
    EXTRA_UV_ARGS="$RLM_EXTRA_UV_ARGS"
fi

uv tool install --python 3.10 --editable "$RLM_CHECKOUT" $SKILL_ARGS $EXTRA_UV_ARGS

REAL_TOOL_DIR="$UV_TOOL_BIN_DIR/.rlm-real-tools"
mkdir -p "$REAL_TOOL_DIR"

wrap_skill_cli() {
    local tool_name="$1"
    local wrapper_path="$UV_TOOL_BIN_DIR/$tool_name"
    local real_path="$REAL_TOOL_DIR/$tool_name"

    if [ -x "$wrapper_path" ] && [ ! -x "$real_path" ]; then
        mv "$wrapper_path" "$real_path"
    fi
    [ -x "$real_path" ] || return 0

    cat > "$wrapper_path" <<EOF
#!/usr/bin/env bash
set -eo pipefail
REAL_TOOL="$real_path"
TOOL_NAME="$tool_name"
SOURCE="\${RLM_TOOL_CALL_SOURCE:-bash}"
if [ -n "\${RLM_SESSION_DIR:-}" ]; then
  printf '{"tool":"%s","source":"%s","timestamp":%s}\n' \\
      "\$TOOL_NAME" "\$SOURCE" "\$(date +%s.%N)" \\
      >> "\${RLM_SESSION_DIR}/programmatic_tool_calls.jsonl" 2>/dev/null || true
fi
exec "\$REAL_TOOL" "\$@"
EOF
    chmod +x "$wrapper_path"
}

for tool_name in $SKILL_TOOL_NAMES; do
    wrap_skill_cli "$tool_name"
done
