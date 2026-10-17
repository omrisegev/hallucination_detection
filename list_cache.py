import os
import sys

# Configure output to support UTF-8 encoding
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

local_cache_dir = r"C:\Users\omris\TAU\hallucination_detection\local_cache"
print("Files in local_cache:")
if os.path.exists(local_cache_dir):
    for f in os.listdir(local_cache_dir):
        fp = os.path.join(local_cache_dir, f)
        print(f"  {f} - {os.path.getsize(fp)} bytes")
else:
    print("  local_cache directory does not exist!")
