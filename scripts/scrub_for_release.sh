#!/usr/bin/env bash
# Pre-publication scrub. Run ONCE, immediately before the repository is made
# public, and not before: it rewrites the cluster scripts to read their host
# and account from environment variables, which would break the live cron
# keeper while the campaign is still feeding.
#
# What it removes and why:
#   - the private workstation's IP address: a machine on the university
#     network, publishing it invites unsolicited scanning and it carries no
#     reproducibility value
#   - absolute home paths: they document one person's filesystem, not the
#     experiment
#   - cluster account names: the institutions are already named in the
#     acknowledgments, the accounts need not be
#
# What it deliberately KEEPS: the cluster hostnames and the scheduler
# directives. A reader reproducing this needs to see that the campaign ran
# on two named public HPC systems under real queue limits.
#
# Verify with --check first; it reports what would change and exits.
set -euo pipefail
cd "$(dirname "$0")/.."
CHECK=${1:-}

files=$(git ls-files | grep -E '^(slurm|scripts)/' || true)
pat_ip='161\.116\.84\.52'
pat_user='amughrabi'

if [ "$CHECK" = "--check" ]; then
  echo "files that would be rewritten:"
  grep -lE "$pat_ip|$pat_user" $files 2>/dev/null || echo "  (none)"
  exit 0
fi

for f in $files; do
  [ -f "$f" ] || continue
  sed -i -E \
    -e "s#$pat_ip#\${WORKSTATION_HOST:?set WORKSTATION_HOST}#g" \
    -e "s#/home/$pat_user#\${HOME}#g" \
    -e "s#/media/HDD_4TB/$pat_user#\${WORKSTATION_ROOT}#g" \
    -e "s#/mnt/beegfs/$pat_user#\${CLUSTER_SCRATCH}#g" \
    -e "s#\b$pat_user@#\${CLUSTER_USER}@#g" \
    "$f"
done
echo "scrubbed. Now re-read the diff by hand: a sed pass is not a review."
grep -rlE "$pat_ip|$pat_user" $files 2>/dev/null && echo "REMAINING (fix by hand)" || echo "no identifiers remain in slurm/ or scripts/"
