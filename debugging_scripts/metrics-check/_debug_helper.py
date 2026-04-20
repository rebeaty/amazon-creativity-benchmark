#!/usr/bin/env python3
"""Helper for managing pending/done dataset assignments during debugging.

Usage:
    python3 _debug_helper.py next    <Name>            # Print the next pending dataset
    python3 _debug_helper.py done    <Name> <dataset>  # Move dataset from pending to done
    python3 _debug_helper.py pending <Name> <dataset>  # Move dataset from done back to pending
    python3 _debug_helper.py status  <Name>            # Show pending/done counts
"""
import json
import sys
import os
import fcntl

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
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            return fn()
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)


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


def cmd_pending(name, dataset):
    def _inner():
        pending = _load(PENDING_FILE)
        done = _load(DONE_FILE)

        p_list = pending.get(name, [])
        d_list = done.get(name, [])

        if dataset in d_list:
            d_list.remove(dataset)
        if dataset not in p_list:
            p_list.append(dataset)

        pending[name] = p_list
        done[name] = d_list

        _save(PENDING_FILE, pending)
        _save(DONE_FILE, done)

    _with_lock(_inner)
    print(f"Moved '{dataset}' back to pending for {name}")


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
    elif action == "pending":
        if len(sys.argv) < 4:
            print("Usage: _debug_helper.py pending <Name> <dataset>")
            sys.exit(1)
        cmd_pending(name, sys.argv[3])
    elif action == "status":
        cmd_status(name)
    else:
        print(f"Unknown action: {action}")
        sys.exit(1)
