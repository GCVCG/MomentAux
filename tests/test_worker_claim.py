"""The lane worker's claim must be bounded by the LIVE worklist.

Two silent failures came from an unconditional increment (2026-08-26):

  * an idling worker burned one index per poll, so the shared counter ran
    away from the file (observed at 396 against a 299-line worklist) and
    every APPENDED task landed below the counter and was never claimed;
  * a claim taken just before an append was DISCARDED rather than run,
    losing one task per live worker, in a perfect odd/even pattern.

Both are the 2026-08-03 family: claims are atomic, execution is not. This
file pins the fix statically (every lane uses the bounded form) and
dynamically (a four-worker simulation runs every appended task exactly once).
"""
import glob
import os
import subprocess
import textwrap



_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _lane_templates():
    out = []
    for f in sorted(glob.glob(os.path.join(_ROOT, "slurm", "*.sbatch"))):
        s = open(f).read()
        if 'flock "$LOCK"' in s and "IDLE_MAX" in s:
            out.append(f)
    return out


def test_every_lane_claims_within_the_live_worklist():
    lanes = _lane_templates()
    assert lanes, "no lane templates found"
    bad = []
    for f in lanes:
        s = open(f).read()
        # The claim must consult the worklist length and refuse when drained.
        if "DRAINED" not in s or "wc -l < '$WORKLIST'" not in s:
            bad.append(os.path.basename(f))
    assert not bad, (
        "these lanes increment the counter without checking the live "
        f"worklist, so idling workers burn indices: {bad}")


def test_no_lane_discards_a_claim_on_growth():
    """The old code did `N=$NEWN; continue`, throwing the claim away."""
    bad = []
    for f in _lane_templates():
        s = open(f).read()
        if "worklist grew" in s and "DRAINED" not in s:
            bad.append(os.path.basename(f))
    assert not bad, f"claim discarded on worklist growth in: {bad}"



def test_appended_tasks_all_run_exactly_once(tmp_path):
    """Four workers idle on a 5-line list, 10 lines are appended mid-run.

    Under the old logic this executed 5 of 15 and left the counter at ~41.
    """
    sim = tmp_path / "sim.sh"
    sim.write_text(textwrap.dedent(r"""
        set -u
        WL=$PWD/wl; CTR=$PWD/ctr; LOCK=$PWD/lock; DONE=$PWD/done
        IDLE_MAX=8
        echo 0 > $CTR; : > $LOCK; : > $DONE
        worker () {
            IDLE=0
            while :; do
                i=$(flock "$LOCK" bash -c "
                    v=\$(cat '$CTR' 2>/dev/null)
                    case \"\$v\" in ''|*[!0-9]*) exit 1 ;; esac
                    n=\$(wc -l < '$WL' 2>/dev/null || echo 0)
                    if [ \"\$v\" -ge \"\$n\" ]; then echo DRAINED; exit 0; fi
                    echo \$((v+1)) > '$CTR'
                    echo \$v")
                if [ "$i" = DRAINED ]; then
                    IDLE=$((IDLE+1)); [ "$IDLE" -ge "$IDLE_MAX" ] && break
                    sleep 1; continue
                fi
                i=$((i+1)); IDLE=0
                cmd=$(sed -n "${i}p" "$WL"); [ -z "$cmd" ] && continue
                flock "$LOCK" bash -c "echo $i >> $DONE"
            done
        }
        for w in 1 2 3 4; do worker & done
        sleep 3
        seq 6 15 | sed 's/^/task/' >> "$WL"
        wait
    """))
    (tmp_path / "wl").write_text("".join(f"task{i}\n" for i in range(1, 6)))
    subprocess.run(["bash", str(sim)], cwd=tmp_path, check=True,
                   capture_output=True, timeout=180)
    done = sorted(int(x) for x in
                  (tmp_path / "done").read_text().split())
    assert done == list(range(1, 16)), (
        f"appended tasks were skipped or repeated: {done}")
    assert (tmp_path / "ctr").read_text().strip() == "15", (
        "counter ran past the worklist, which is what strands appended tasks")
