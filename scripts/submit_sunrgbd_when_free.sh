#!/bin/bash
# Poll the BSC account submit cap (MaxSubmitPA=20) and submit block H's
# smoke-gated sunrgbd lane as soon as there is room: smoke + >=1 worker node
# (2 nodes when room allows). Idempotent: refuses if ms_sunrgbd already queued.
HOST=ub881905@alogin2.bsc.es
for i in $(seq 1 200); do
  OUT=$(ssh -o BatchMode=yes -o ConnectTimeout=30 $HOST '
    MS=/gpfs/scratch/ub234/momentstem
    if squeue -A ub234 -h -o "%j" | grep -q "^ms_sunrgbd$"; then echo ALREADY; exit 0; fi
    N=$(squeue -A ub234 -h | wc -l)
    FREE=$((20 - N))
    if [ "$FREE" -lt 2 ]; then echo "WAIT free=$FREE"; exit 0; fi
    cd $MS/repo
    SM=$(sbatch --parsable slurm/sunrgbd_smoke.sbatch) || { echo "ERR smoke"; exit 0; }
    J1=$(sbatch --parsable --dependency=afterok:$SM slurm/bsc_sunrgbd.sbatch) || { echo "SMOKE_ONLY $SM"; exit 0; }
    if [ "$FREE" -ge 3 ]; then
      J2=$(sbatch --parsable --dependency=afterok:$SM slurm/bsc_sunrgbd.sbatch) && echo "SUBMITTED smoke=$SM lanes=$J1,$J2" && exit 0
    fi
    echo "SUBMITTED smoke=$SM lanes=$J1"
  ' 2>/dev/null)
  echo "$(date -Is) $OUT"
  case "$OUT" in *SUBMITTED*|*ALREADY*) exit 0;; esac
  sleep 600
done
echo GAVE_UP
