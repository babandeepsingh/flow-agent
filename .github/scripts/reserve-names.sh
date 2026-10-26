#!/usr/bin/env bash
# Publishes a minimal 0.0.1 placeholder for each flow-agent crate name that is
# not on crates.io yet. Safe to re-run: existing names are skipped.
# Set DRY_RUN=1 to package without uploading.
set -euo pipefail

CRATES=(
  flow-agent-core
  flow-agent-models
  flow-agent-graph
  flow-agent-checkpoint
  flow-agent-prebuilt
  flow-agent
)
REPO_URL="https://github.com/babandeepsingh/flow-agent"
USER_AGENT="flow-agent-reserve-names (${REPO_URL})"
WORK_DIR="$(mktemp -d)"
trap 'rm -rf "$WORK_DIR"' EXIT

exists_on_crates_io() {
  local code
  code="$(curl -s -o /dev/null -w '%{http_code}' -A "$USER_AGENT" "https://crates.io/api/v1/crates/$1")"
  [ "$code" = "200" ]
}

make_placeholder() {
  local name="$1" dir="$WORK_DIR/$1"
  mkdir -p "$dir/src"
  cat > "$dir/Cargo.toml" <<TOML
[package]
name = "$name"
version = "0.0.1"
edition = "2021"
description = "Placeholder for $name, part of flow-agent: LangChain/LangGraph-style LLM chains, agents and graphs for Rust. Real release coming as 0.1.0."
license = "MIT OR Apache-2.0"
repository = "$REPO_URL"
homepage = "$REPO_URL"
readme = "README.md"
keywords = ["llm", "agent", "langchain", "langgraph", "ai"]
categories = ["asynchronous"]
TOML
  cat > "$dir/README.md" <<MD
# $name

Part of [flow-agent]($REPO_URL), a Rust library for LLM chains, tool-using
agents and stateful graphs inspired by LangChain and LangGraph.

This \`0.0.1\` release is a placeholder with no functionality. The first real
release will be \`0.1.0\`. Follow progress at $REPO_URL.
MD
  cat > "$dir/src/lib.rs" <<RS
//! Placeholder for \`$name\`, part of [flow-agent]($REPO_URL).
//!
//! This release contains no functionality. The first real release will be \`0.1.0\`.
#![forbid(unsafe_code)]
RS
  echo "$dir"
}

publish() {
  local dir="$1"
  if [ "${DRY_RUN:-0}" = "1" ]; then
    cargo publish --dry-run --allow-dirty --manifest-path "$dir/Cargo.toml"
    return
  fi
  # crates.io limits how fast new crates can be created; wait and retry on rate limits.
  local attempt
  for attempt in 1 2 3; do
    if cargo publish --allow-dirty --manifest-path "$dir/Cargo.toml"; then
      return 0
    fi
    if [ "$attempt" -lt 3 ]; then
      echo "Publish failed (attempt $attempt), likely rate limited; retrying in 11 minutes..."
      sleep 660
    fi
  done
  return 1
}

if [ "${DRY_RUN:-0}" != "1" ] && [ -z "${CARGO_REGISTRY_TOKEN:-}" ]; then
  echo "::error::CARGO_REGISTRY_TOKEN secret is not set. Add it under Settings > Secrets and variables > Actions."
  exit 1
fi

for name in "${CRATES[@]}"; do
  if exists_on_crates_io "$name"; then
    echo "✔ $name already exists on crates.io, skipping"
    continue
  fi
  echo "→ Reserving $name"
  publish "$(make_placeholder "$name")"
  if [ "${DRY_RUN:-0}" = "1" ]; then echo "✔ $name packaged (dry run)"; else echo "✔ $name reserved: https://crates.io/crates/$name"; fi
done
