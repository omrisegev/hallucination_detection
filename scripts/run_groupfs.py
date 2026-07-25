"""
run_groupfs.py — Helper runner for GroupFS sweep.
"""

import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (REPO, os.path.join(REPO, "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from scripts.sweep_dufs_groupfs import main

if __name__ == '__main__':
    sys.argv = ['sweep_dufs_groupfs.py', '--arm', 'groupfs']
    main()
