import pickle
import os
import sys

# Configure output to support UTF-8 encoding
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

cache_dir = r"C:\Users\omris\TAU\hallucination_detection\local_cache"

def inspect_file(filename):
    path = os.path.join(cache_dir, filename)
    print("="*60)
    print(f"INSPECTING: {filename}")
    if not os.path.exists(path):
        print("  File does not exist!")
        return
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
        print(f"Type: {type(data)}")
        if isinstance(data, dict):
            print(f"Keys: {list(data.keys())[:20]}")
            # Print structure of first item
            first_key = list(data.keys())[0]
            val = data[first_key]
            print(f"First item key: {first_key}, type: {type(val)}")
            if isinstance(val, dict):
                print(f"  First item sub-keys: {list(val.keys())}")
                for sk in list(val.keys())[:5]:
                    sv = val[sk]
                    print(f"    sub-key '{sk}': type {type(sv)}, value preview/length: {str(sv)[:100]}")
            elif isinstance(val, list):
                print(f"  First item list length: {len(val)}, first item in list: {str(val[0])[:100]}")
        elif isinstance(data, list):
            print(f"Length: {len(data)}")
            print(f"First item type: {type(data[0])}")
            print(f"First item: {str(data[0])[:200]}")
    except Exception as e:
        print(f"Error inspecting file: {e}")
    print("="*60)

inspect_file("trace_cells.pkl")
inspect_file("math500_qwen7b_T1.0_run0.pkl")
