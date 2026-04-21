#!/usr/bin/env python3
"""Helper for managing pending/done dataset assignments during debugging.

Usage:
    python3 _debug_helper.py next   <Name>          # Print the next pending dataset
    python3 _debug_helper.py done   <Name> <dataset> # Move dataset from pending to done
    python3 _debug_helper.py status <Name>           # Show pending/done counts
"""
import json
import sys
import os

# Ensure LF-only output so git-bash consumers (mapfile / read loops) don't get
# trailing \r on every line when this runs under a native Windows Python.
try:
    sys.stdout.reconfigure(newline="\n")
except Exception:
    pass

try:
    import fcntl  # Unix-only; missing on Windows.
    _HAS_FCNTL = True
except ImportError:
    import msvcrt  # Windows fallback.
    _HAS_FCNTL = False

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PENDING_FILE = os.path.join(SCRIPT_DIR, "debug_assignments_pending.json")
DONE_FILE = os.path.join(SCRIPT_DIR, "debug_assignments_done.json")
LOCK_FILE = os.path.join(SCRIPT_DIR, ".assignments.lock")


def _load(path):
    with open(path) as f:
        return json.load(f)


def _save(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def _with_lock(fn):
    """File-lock so concurrent scripts don't corrupt the JSON."""
    os.makedirs(SCRIPT_DIR, exist_ok=True)
    with open(LOCK_FILE, "w") as lf:
        if _HAS_FCNTL:
            fcntl.flock(lf, fcntl.LOCK_EX)
            try:
                return fn()
            finally:
                fcntl.flock(lf, fcntl.LOCK_UN)
        else:
            # Windows: LK_LOCK blocks until the lock is available (retries internally).
            # Lock a single byte at offset 0 (msvcrt.locking locks relative to current pos).
            lf.write("\0")
            lf.flush()
            lf.seek(0)
            msvcrt.locking(lf.fileno(), msvcrt.LK_LOCK, 1)
            try:
                return fn()
            finally:
                lf.seek(0)
                msvcrt.locking(lf.fileno(), msvcrt.LK_UNLCK, 1)


def cmd_next(name):
    def _inner():
        pending = _load(PENDING_FILE)
        datasets = pending.get(name, [])
        if not datasets:
            return None
        return datasets[0]
    result = _with_lock(_inner)
    if result is None:
        print("__EMPTY__")
    else:
        print(result)


def cmd_done(name, dataset):
    def _inner():
        pending = _load(PENDING_FILE)
        done = _load(DONE_FILE)

        p_list = pending.get(name, [])
        d_list = done.get(name, [])

        if dataset in p_list:
            p_list.remove(dataset)
        if dataset not in d_list:
            d_list.append(dataset)

        pending[name] = p_list
        done[name] = d_list

        _save(PENDING_FILE, pending)
        _save(DONE_FILE, done)

    _with_lock(_inner)
    print(f"Moved '{dataset}' to done for {name}")


def cmd_status(name):
    def _inner():
        pending = _load(PENDING_FILE)
        done = _load(DONE_FILE)
        p = len(pending.get(name, []))
        d = len(done.get(name, []))
        return p, d
    p, d = _with_lock(_inner)
    print(f"{name}: {d} done, {p} pending, {p + d} total")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    action = sys.argv[1]
    name = sys.argv[2]

    if action == "next":
        cmd_next(name)
    elif action == "done":
        if len(sys.argv) < 4:
            print("Usage: _debug_helper.py done <Name> <dataset>")
            sys.exit(1)
        cmd_done(name, sys.argv[3])
    elif action == "status":
        cmd_status(name)
    else:
        print(f"Unknown action: {action}")
        sys.exit(1)
