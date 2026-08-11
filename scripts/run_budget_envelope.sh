#!/usr/bin/env bash
# Run one SSL method's 800-epoch budget cells across the data envelope,
# SEQUENTIALLY. Each cell already runs its three seeds concurrently, so a
# second cell in parallel would oversubscribe a dataloader-bound job and
# slow both. Waits for any in-flight pretrain to clear first, for the same
# reason.
set -u
cd "$(dirname "$0")/.."
LIST=${LIST:?set LIST=path/to/cell-list}
LOG=${LOG:-logs/budget_envelope.log}
mkdir -p "$(dirname "$LOG")"
while pgrep -f 'simclr_pretrain\.py|simsiam_pretrain\.py' > /dev/null; do sleep 120; done
echo "$(date -Is) start: $(grep -cvE '^\s*(#|$)' "$LIST") cells" >> "$LOG"
while IFS= read -r cmd <&3; do
  case "$cmd" in ''|\#*) continue;; esac
  s=$(date +%s)
  eval "$cmd" >/dev/null 2>&1 </dev/null
  echo "$(date -Is) rc=$? $(( ($(date +%s)-s)/60 ))min :: $cmd" >> "$LOG"
done 3< "$LIST"
echo "$(date -Is) ENVELOPE_COMPLETE" >> "$LOG"
