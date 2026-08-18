#!/usr/bin/env python3
"""Kill training processes matching a config substring, safely.

Written after `pkill -f`/`pgrep -f` matched THIS SESSION'S OWN SHELL three
times and killed it: the shell's command line contains the pattern being
searched for. Matching argv from /proc and excluding our own ancestor chain is
the only form that cannot self-match.

Usage: kill_wave.py <config-substring> [...]
"""
import os, signal, sys


def ancestors(pid):
    """Our own process chain -- never kill anything in it."""
    out, p = set(), pid
    while p > 1:
        out.add(p)
        try:
            p = int(open(f"/proc/{p}/stat").read().split(")")[-1].split()[1])
        except Exception:
            break
    return out


def main(patterns):
    if not patterns:
        print("usage: kill_wave.py <config-substring> [...]")
        return 2
    safe = ancestors(os.getpid())
    killed = 0
    for d in os.listdir("/proc"):
        if not d.isdigit() or int(d) in safe:
            continue
        try:
            argv = [a.decode("utf8", "replace")
                    for a in open(f"/proc/{d}/cmdline", "rb").read().split(b"\0") if a]
        except Exception:
            continue
        if len(argv) < 3 or not argv[0].endswith("bin/python") or argv[1] != "train.py":
            continue
        if not any(p in a for p in patterns for a in argv[2:]):
            continue
        try:
            os.kill(int(d), signal.SIGKILL)
            killed += 1
            print(f"  killed {d} {' '.join(argv[2:4])}")
        except Exception as e:
            print(f"  {d}: {e}")
    print(f"{killed} killed")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
