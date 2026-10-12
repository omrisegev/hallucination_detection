import pickle
import os
import sys

# Configure output to support UTF-8 encoding
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

cache_dir = r"C:\Users\omris\TAU\hallucination_detection\local_cache"

with open(os.path.join(cache_dir, "trace_cells.pkl"), "rb") as f:
    data = pickle.load(f)

for k, val in data.items():
    print(f"Key: {k}, type: {type(val)}")
    if isinstance(val, tuple):
        print(f"  Tuple length: {len(val)}")
        for idx, item in enumerate(val):
            print(f"    Item {idx}: type {type(item)}, size/length: {len(item) if hasattr(item, '__len__') else 'N/A'}")
            if isinstance(item, list) and len(item) > 0:
                print(f"      First element preview: {str(item[0])[:150]}")
            elif isinstance(item, dict) and len(item) > 0:
                first_k = list(item.keys())[0]
                print(f"      First dict key: {first_k}, type: {type(item[first_k])}")
