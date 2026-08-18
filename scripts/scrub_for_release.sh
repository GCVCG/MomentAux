#!/usr/bin/env bash
# Pre-publication scrub. Run ONCE, immediately before the repository is made
# public, and not before: it rewrites the cluster scripts to read their host
# and account from environment variables, which would break the live cron
# keeper while the campaign is still feeding. (The keeper was disabled on
# 2026-08-18 when the grid completed, so that constraint no longer binds.)
#
# What it removes and why:
#   - the private workstation's IP address: a machine on the university
#     network, publishing it invites unsolicited scanning and it carries no
#     reproducibility value
#   - absolute home and scratch paths: they document one person's filesystem,
#     not the experiment
#   - cluster account names: the institutions are already named in the
#     acknowledgments, the accounts need not be
#
# What it deliberately KEEPS: the cluster hostnames and the scheduler
# directives. A reader reproducing this needs to see that the campaign ran
# on two named public HPC systems under real queue limits.
#
# THE RULE WHERE THOSE TWO CLAUSES COLLIDE, and `#SBATCH --account=` is
# exactly where they do: KEEP THE DIRECTIVE, PARAMETERIZE THE VALUE. A reader
# needs to know an account directive is required, not which account. That is
# what this script already does for ssh, where ${CLUSTER_USER}@ preserves the
# command and drops the user, and it is applied to the account directive too.
#
# ---------------------------------------------------------------------------
# READ THIS BEFORE TRUSTING A CLEAN RUN.
#
# A SCRUB KEYED ON AN ENUMERATED PATTERN LIST CANNOT ENFORCE A POLICY STATED
# IN CATEGORIES. The comment above promises to remove three CATEGORIES; the
# code below matches a LIST. Those are not the same thing, and during the
# 2026-08-17/18 artifact pass the gap opened four separate times, each caught
# only because somebody searched instead of re-running this script:
#
#   1. SCOPE. The file list was slurm/ and scripts/ only, so 65 lines of home
#      path sat in logs/ and shipped inside logs.tar.gz.
#   2. A MANGLED FORM. "-home-amughrabi-projects-..." inside a scratch path
#      is a home path with the slashes replaced by dashes, which "s#/home/..#"
#      cannot match.
#   3. A BARE USERNAME. Captured `ls -l` output carries the user as owner and
#      group with no path around it at all, matching no path pattern.
#   4. AN ENTIRE CATEGORY. The BSC accounts (a personal login, a group
#      allocation and their /gpfs/scratch paths) appeared in 34 tracked files
#      and 722 lines of logs.tar.gz. The policy above named "cluster account
#      names" from the day it was written; the pattern list never named them.
#
# So: DO NOT treat a clean exit from this script as proof the tree is clean.
# The check that actually works is to search for the CATEGORY the policy names
# and classify every hit, e.g.
#
#     grep -ria '<identifier>' <every tree that ships>
#     tar -xzOf dist/<asset>.tar.gz | grep -ac '<identifier>'
#
# and then LOOK at what comes back rather than counting it: a residual
# "/media/HDD_16TB/momentstem_data" is a mount with no user in it and is fine,
# while a residual "user user" in an ls listing is not. Verify the built
# assets, not the working tree, because the assets are what is published.
# ---------------------------------------------------------------------------
#
# Verify with --check first; it reports what would change and exits.
set -euo pipefail
cd "$(dirname "$0")/.."
CHECK=${1:-}

# Scope. logs/ was added after it escaped this script twice. Note that the
# release collector walks the FILESYSTEM while this list comes from git, so an
# untracked log still ships and is still not seen here -- another reason the
# search above is the real check.
#
# This script EXCLUDES ITSELF, and that is load-bearing rather than tidy: the
# account patterns below are bare identifiers, so a pass over this file would
# rewrite the pattern variables into ${CLUSTER_USER} and leave a scrub that
# matches nothing while still exiting 0. The older patterns all required a
# path prefix and so were self-safe by accident, which is why this only
# became a hazard when the account category was added.
files=$(git ls-files | grep -E '^(slurm|scripts|logs)/' \
        | grep -v '^scripts/scrub_for_release.sh$' || true)

pat_ip='161\.116\.84\.52'
pat_user='amughrabi'
pat_bsc_login='ub881905'   # personal login: removed everywhere, no exceptions
pat_bsc_group='ub234'      # group allocation: parameterized, directive kept
ALL="$pat_ip|$pat_user|$pat_bsc_login|$pat_bsc_group"

if [ "$CHECK" = "--check" ]; then
  echo "files that would be rewritten:"
  grep -lE "$ALL" $files 2>/dev/null || echo "  (none)"
  exit 0
fi

for f in $files; do
  [ -f "$f" ] || continue
  sed -i -E \
    -e "s#$pat_ip#\${WORKSTATION_HOST:?set WORKSTATION_HOST}#g" \
    -e "s#/home/$pat_user#\${HOME}#g" \
    -e "s#/media/HDD_4TB/$pat_user#\${WORKSTATION_ROOT}#g" \
    -e "s#/media/$pat_user/HDD_4TB_1#\${WORKSTATION_ROOT2}#g" \
    -e "s#/mnt/beegfs/$pat_user#\${CLUSTER_SCRATCH}#g" \
    -e "s#-home-$pat_user-#-home-user-#g" \
    -e "s#\b$pat_user@#\${CLUSTER_USER}@#g" \
    -e "s#\b$pat_user $pat_user\b#user user#g" \
    -e "s#/gpfs/scratch/$pat_bsc_group#\${CLUSTER_SCRATCH}#g" \
    -e "s#(--account=)$pat_bsc_group#\1\${CLUSTER_ACCOUNT:?set CLUSTER_ACCOUNT}#g" \
    -e "s#\b$pat_bsc_group\b#\${CLUSTER_ACCOUNT}#g" \
    -e "s#\b$pat_bsc_login\b#\${CLUSTER_USER}#g" \
    "$f"
done
echo "scrubbed. Now re-read the diff by hand: a sed pass is not a review,"
echo "and see the header: a clean exit here is not evidence the tree is clean."
grep -rlE "$ALL" $files 2>/dev/null && echo "REMAINING (fix by hand)" \
  || echo "no enumerated identifier remains in slurm/, scripts/ or logs/"
