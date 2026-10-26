#!/usr/bin/env bash
# Publish every crate to crates.io in dependency order.
# Versions already on crates.io are skipped, so a failed release can be re-run.
# DRY_RUN=1 packages and verifies everything without uploading.
set -euo pipefail

CRATES=(
  flow-agent-core
  flow-agent-models
  flow-agent-graph
  flow-agent-checkpoint
  flow-agent-prebuilt
  flow-agent
)
VERSION="$(sed -n 's/^version = "\(.*\)"$/\1/p' Cargo.toml | head -n1)"
USER_AGENT="flow-agent-release (https://github.com/babandeepsingh/flow-agent)"
echo "Releasing version $VERSION"

if [ "${DRY_RUN:-0}" = "1" ]; then
  cargo publish --workspace --dry-run --locked
  exit 0
fi

if [ -z "${CARGO_REGISTRY_TOKEN:-}" ]; then
  echo "::error::CARGO_REGISTRY_TOKEN is not set."
  exit 1
fi

for crate in "${CRATES[@]}"; do
  code="$(curl -s -o /dev/null -w '%{http_code}' -A "$USER_AGENT" \
    "https://crates.io/api/v1/crates/$crate/$VERSION")"
  if [ "$code" = "200" ]; then
    echo "✔ $crate $VERSION already published, skipping"
    continue
  fi
  echo "→ Publishing $crate $VERSION"
  # cargo waits until the new version is in the index before returning,
  # so the next crate can depend on it.
  cargo publish -p "$crate" --locked
done
echo "✔ Released $VERSION"
