#!/usr/bin/env python3
"""
glazaxly.py

Master launcher for the space_sim viewer. This script locates the default
`data/universe.json` and starts the cluster-first viewer with wiki integrated.

Usage:
    python glazaxly.py

It requires the repository layout to have `data/universe.json` next to this
script (as in the workspace). If you want a different universe file, pass its
path as the first argument.
"""
import os
import sys

from orbit_sandbox_cluster_first import ClusterViewer


def find_universe_json():
    here = os.path.dirname(__file__)
    candidates = [
        os.path.join(here, "data", "universe.json"),
        os.path.join(here, "../data", "universe.json"),
        os.path.join(here, "../space_sim", "data", "universe.json"),
    ]
    for c in candidates:
        p = os.path.abspath(c)
        if os.path.exists(p):
            return p
    return None


def main():
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = find_universe_json()
        if not path:
            print("Could not find data/universe.json. Provide a path as argument.")
            sys.exit(1)

    viewer = ClusterViewer(path)
    viewer.run()


if __name__ == '__main__':
    main()
