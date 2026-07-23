#!/usr/bin/env python3
"""
Recover ubaseline_scores.csv from git history + current version.

Step 0e of the plan: the current 6-row file is missing 13 rows that were
silently dropped by score_ubaselines.py's unconditional open("w") + writerows.
Recover losslessly from git commit 4df18aa (pre-overwrite 18-row version).

Usage:
    python scripts/recover_ubaseline_csv.py
"""
import subprocess
import pandas as pd
import sys

def main():
    # Get pre-overwrite version from git
    try:
        old_csv_str = subprocess.check_output(
            ['git', 'show', '4df18aa:results/repgrid/ubaseline_scores.csv'],
            cwd='c:\\Users\\omris\\TAU\\hallucination_detection',
            text=True
        )
        old_df = pd.read_csv(pd.io.common.StringIO(old_csv_str))
    except Exception as e:
        print(f"ERROR: Failed to recover old CSV from git: {e}", file=sys.stderr)
        sys.exit(1)

    # Read current version
    try:
        curr_df = pd.read_csv('results/repgrid/ubaseline_scores.csv')
    except Exception as e:
        print(f"ERROR: Failed to read current CSV: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Old version: {len(old_df)} rows")
    print(f"Current version: {len(curr_df)} rows")

    # Merge: keep all old rows, overwrite with current rows where they overlap (by 'cell')
    merged = pd.concat([old_df, curr_df], ignore_index=False).drop_duplicates(subset=['cell'], keep='last')
    merged = merged.sort_values('cell').reset_index(drop=True)

    print(f"Merged version: {len(merged)} rows")

    # Write back
    out_path = 'results/repgrid/ubaseline_scores.csv'
    merged.to_csv(out_path, index=False)
    print(f"Wrote {out_path} with {len(merged)} rows")

    # Verify the lost rows are back
    lost_cells = {
        'epr_triviaqa_mistral24b', 'inside_coqa_llama7b', 'lapeigvals_gsm8k_llama8b',
        'lapeigvals_gsm8k_phi35', 'losnet_hotpotqa_mistral7b', 'sciq_llama8b',
        'se_squad_v2_llama8b', 'seiclr_triviaqa_opt30b', 'semenergy_triviaqa_qwen3_8b',
        'spilled_triviaqa_llama8b', 'truthfulqa_llama8b'
    }
    recovered = lost_cells & set(merged['cell'].unique())
    print(f"Recovered lost cells: {len(recovered)} / {len(lost_cells)}")
    if recovered:
        print(f"  {', '.join(sorted(recovered))}")

if __name__ == '__main__':
    main()
