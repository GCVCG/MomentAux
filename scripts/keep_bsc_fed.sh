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
# big lane: hold ONE 24h ms_big job in flight while worklist.big has
# unclaimed tasks (cnx >=25%, pathmnist/food101 @100%)
NBIG=\$(wc -l < "\$MS/worklist.big" 2>/dev/null || echo 0)
CBIG=\$(cat "\$MS/queue.counter.big" 2>/dev/null || echo 0)
BIGJOBS=\$(squeue -u ub881905 -h -n ms_big -t R,PD -o "%i" | wc -l)
if [ "\$CBIG" -lt "\$NBIG" ] && [ "\$BIGJOBS" -lt 2 ]; then
    for i in \$(seq 1 \$((2 - BIGJOBS))); do
        sbatch "\$MS/bsc_big.sbatch" 2>&1 | sed 's/^/SUBMIT-BIG /'
    done
fi
CUR=\$(squeue -u ub881905 -h -n ms_grid -t R,PD -o "%i" | wc -l)
if [ "\$CTR" -ge "\$N" ]; then
    # Queue counter exhausted. RECONCILE: the counter advances on CLAIM with no
    # retry, so cells lost to walltime expiry are gaps. Regenerate the worklist
    # from what has NO final.json and start a fresh pass over just those. Only
    # do this when NO worker is live (CUR==0) so we never swap the file under a
    # worker mid-read; while workers drain we just wait.
    if [ "\$CUR" -gt 0 ]; then echo "STATE draining \$CTR/\$N, \$CUR still live"; exit 0; fi
    cd "\$MS/repo"                          # scripts + configs/grid live here
    # cnx cells at >=25% exceed the 5h45 worker walltime and live in the
    # dedicated 24h big-cell lane (worklist.big / bsc_big.sbatch) -- keep
    # them OUT of the normal reconcile pass or they churn forever.
    # exclude every config that lives in the big lane (data-driven: the
    # big worklist itself is the source of truth, no pattern list to rot)
    grep -o "configs/[^ ]*\.yaml" "\$MS/worklist.big" | sort -u > "\$MS/.bigcfgs"
    # --split all: BSC owns the ENTIRE grid since 2026-07-23 (turing's queue
    # lane retired; its 2 GPUs run dedicated waves only)
    OUT="\$MS/runs" python scripts/make_missing_worklist.py --split all 2>/dev/null \
        | grep -vFf "\$MS/.bigcfgs" > "\$MS/worklist.bsc.new"
    M=\$(grep -c . "\$MS/worklist.bsc.new" || echo 0)
    if [ "\$M" -eq 0 ]; then
        rm -f "\$MS/worklist.bsc.new"; touch "\$MS/GRID_COMPLETE"
        echo "STATE complete: 0 cells missing"; exit 0
    fi
    mv "\$MS/worklist.bsc.new" "\$MS/worklist.bsc"
    echo 0 > "\$MS/queue.counter"           # fresh pass over the gaps
    echo "STATE reconciled: \$M cells missing -> new pass"
    N=\$M; CTR=0
fi
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
