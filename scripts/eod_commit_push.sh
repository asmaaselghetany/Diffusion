#!/usr/bin/env bash
# End-of-day: commit tracked+safe work and push (no Cursor co-author, no git config).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

AUTHOR_NAME="${EOD_GIT_AUTHOR_NAME:-Asmaa Elsayed}"
AUTHOR_EMAIL="${EOD_GIT_AUTHOR_EMAIL:-asmaaselghetany@gmail.com}"
BRANCH="$(git branch --show-current)"
REMOTE="${EOD_GIT_REMOTE:-origin}"

if [[ -z "${BRANCH}" ]]; then
  echo "EOD: detached HEAD — refuse" >&2
  exit 1
fi

# Never stage secrets / heavy / local-only paths (even if not gitignored).
EXCLUDES=(
  '.venv'
  '.venv/'
  'venv/'
  'wandb/'
  'wandb_sync_staging/'
  'outputs/'
  'out/'
  'third_party/'
  '*.ckpt'
  '*.pt'
  '*.pth'
  '*.safetensors'
  '.env'
  'credentials.json'
  '*.pptx'
  'docs/extension_guides/*.pptx'
)

is_excluded() {
  local f="$1"
  local pat
  for pat in "${EXCLUDES[@]}"; do
    # shellcheck disable=SC2254
    case "$f" in
      $pat|*/$pat|$pat/*|*/$pat/*) return 0 ;;
    esac
    if [[ "$f" == $pat ]]; then
      return 0
    fi
  done
  return 1
}

echo "EOD: repo=${REPO_ROOT} branch=${BRANCH} author=${AUTHOR_EMAIL}"

# Stage modifications + new files selectively.
mapfile -t CHANGED < <(git status --porcelain -u --untracked-files=all | sed 's/^...//')
staged=0
skipped=0
for f in "${CHANGED[@]+"${CHANGED[@]}"}"; do
  [[ -z "$f" ]] && continue
  # Unquote paths with spaces from porcelain if needed — rare here.
  f="${f#\"}"
  f="${f%\"}"
  if is_excluded "$f"; then
    echo "EOD: skip $f"
    skipped=$((skipped + 1))
    continue
  fi
  git add -A -- "$f" 2>/dev/null || git add -- "$f" || true
  staged=$((staged + 1))
done

if git diff --cached --quiet; then
  echo "EOD: nothing to commit"
else
  msg="$(cat <<EOF
EOD snapshot $(date -u +%Y-%m-%d): save day's AR→block / eval work.

Automated end-of-day sync on ${BRANCH}.
EOF
)"
  GIT_AUTHOR_NAME="${AUTHOR_NAME}" \
  GIT_AUTHOR_EMAIL="${AUTHOR_EMAIL}" \
  GIT_COMMITTER_NAME="${AUTHOR_NAME}" \
  GIT_COMMITTER_EMAIL="${AUTHOR_EMAIL}" \
    git commit -m "$msg"
  echo "EOD: committed (staged≈${staged}, skipped=${skipped})"
fi

# Refuse force-push. If histories diverged (e.g. pending authorship rewrite), stop.
if ! git rev-parse --verify "${REMOTE}/${BRANCH}" >/dev/null 2>&1; then
  echo "EOD: no ${REMOTE}/${BRANCH} — push -u"
  GIT_TERMINAL_PROMPT=0 git push -u "${REMOTE}" "HEAD:${BRANCH}"
  exit 0
fi

git fetch "${REMOTE}" "${BRANCH}" 2>/dev/null || true
ahead="$(git rev-list --count "${REMOTE}/${BRANCH}..HEAD" 2>/dev/null || echo 0)"
behind="$(git rev-list --count "HEAD..${REMOTE}/${BRANCH}" 2>/dev/null || echo 0)"
echo "EOD: ahead=${ahead} behind=${behind}"

if [[ "${behind}" != "0" ]]; then
  echo "EOD: remote is ahead/diverged (behind=${behind}). Refusing non-ff push." >&2
  echo "EOD: resolve (merge/rebase) or finish authorship force-with-lease first." >&2
  exit 2
fi

if [[ "${ahead}" == "0" ]]; then
  echo "EOD: already synced with ${REMOTE}/${BRANCH}"
  exit 0
fi

GIT_TERMINAL_PROMPT=0 git push "${REMOTE}" "HEAD:${BRANCH}"
echo "EOD: pushed ${BRANCH} → ${REMOTE}"
