#!/bin/bash
# Keep BSC's grid workers topped up (cron, 2026-07-22 user decision).
#
# Each ms_grid job = 1 node = 4 H100s, walltime 5h45. The user chose 32
# concurrent GPUs = 8 jobs; this holds 8 in flight (running + pending) so a
# finishing job is replaced without a gap, and STOPS on its own once the
# shared worklist is drained (counter >= N) or a stop-file is dropped.
#
# Runs on the LOCAL machine (which has the BSC ssh key). Idempotent and safe
# to fire every 15 min: it only ever tops UP to the target, never past it.
set -uo pipefail

BSC="ub881905@alogin1.bsc.es"
MS=/gpfs/scratch/ub234/momentstem
TARGET=8                                   # 8 jobs x 4 GPUs = 32 concurrent
LOG=$HOME/projects/MomentsCNNEncoder/logs/bsc_keeper.log
STAMP=$(date -u +%Y-%m-%dT%H:%M:%SZ)

log() { echo "$STAMP $*" >> "$LOG"; }

# One remote round-trip: report drained?/count, and submit the shortfall.
out=$(timeout 120 ssh -o BatchMode=yes -o ConnectTimeout=30 "$BSC" bash -s <<REMOTE 2>&1
set -uo pipefail
MS=$MS
TARGET=$TARGET
[ -f "\$MS/STOP_KEEPER" ] && { echo "STATE stopped-by-file"; exit 0; }
N=\$(wc -l < "\$MS/worklist.bsc")
CTR=\$(cat "\$MS/queue.counter" 2>/dev/null || echo 0)
if [ "\$CTR" -ge "\$N" ]; then echo "STATE drained \$CTR/\$N"; exit 0; fi
CUR=\$(squeue -u ub881905 -h -n ms_grid -t R,PD -o "%i" | wc -l)
NEED=\$(( TARGET - CUR ))
echo "STATE feeding ctr=\$CTR/\$N running_or_pending=\$CUR need=\$NEED"
[ "\$NEED" -le 0 ] && exit 0
cd "\$MS"
for i in \$(seq 1 "\$NEED"); do
    sbatch "\$MS/bsc_worker.sbatch" 2>&1 | sed 's/^/SUBMIT /'
done
REMOTE
)
rc=$?
log "rc=$rc :: $(echo "$out" | tr '\n' '|')"
echo "$out"
exit $rc
