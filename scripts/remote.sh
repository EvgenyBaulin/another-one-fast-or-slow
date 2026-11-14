#!/bin/bash
# Sync code to the team server and run a command there.
#   scripts/remote.sh "<command>"        run inside ~/hacklab/fos with the uv venv on PATH
#   scripts/remote.sh --pull <path>...   copy result files from the server back to this project
# Host alias `team06` comes from ~/.ssh/config (BindInterface en0 bypasses the VPN).
set -euo pipefail
cd "$(dirname "$0")/.."
REMOTE=team06
RDIR=hacklab/fos
SSH="ssh -o BatchMode=yes -o ConnectTimeout=15 -o LogLevel=ERROR"
if [ "${1:-}" = "--pull" ]; then
  shift
  for f in "$@"; do mkdir -p "$(dirname "$f")"; rsync -az -e "$SSH" "$REMOTE:$RDIR/$f" "$f"; done
  exit 0
fi
rsync -az --delete -e "$SSH" --exclude __pycache__ scripts/ "$REMOTE:$RDIR/scripts/"
rsync -az -e "$SSH" pyproject.toml uv.lock "$REMOTE:$RDIR/"
$SSH "$REMOTE" "cd ~/$RDIR && export PATH=\$HOME/.local/bin:\$PWD/.venv/bin:\$PATH && $*"
