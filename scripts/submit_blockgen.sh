#!/usr/bin/env bash
# Legacy alias — use ./scripts/submit_block.sh
exec "$(cd "$(dirname "$0")" && pwd)/submit_block.sh" "$@"
