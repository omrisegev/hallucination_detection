"""I/O helpers: pickle cache load/save."""
import os
import pickle


def load_cache(path: str) -> dict:
    """Load a pickle cache file. Returns empty dict if file does not exist."""
    return pickle.load(open(path, "rb")) if os.path.exists(path) else {}


def save_cache(obj, path: str) -> None:
    """Save obj to a pickle file."""
    pickle.dump(obj, open(path, "wb"))


def save_cache_atomic(obj, path: str) -> None:
    """
    Atomic pickle save: write to path.tmp then os.replace.

    A SIGKILL mid-write (e.g. Slurm preemption after the 15-min grace period)
    can never leave a truncated cache behind — the previous complete file
    survives until the replace.
    """
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(obj, f)
    os.replace(tmp, path)
