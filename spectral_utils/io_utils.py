"""I/O helpers: pickle cache load/save."""
import os
import pickle


def load_cache(path: str) -> dict:
    """Load a pickle cache file. Returns empty dict if file does not exist."""
    return pickle.load(open(path, "rb")) if os.path.exists(path) else {}


def save_cache(obj, path: str) -> None:
    """Save obj to a pickle file."""
    pickle.dump(obj, open(path, "wb"))
