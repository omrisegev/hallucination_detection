import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def print_file_lines(filepath, start, end):
    print("="*60)
    print(f"FILE: {filepath} (Lines {start}-{end})")
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
        for i in range(start-1, min(end, len(lines))):
            print(f"{i+1}: {lines[i]}", end="")
    except Exception as e:
        print(f"Error reading file: {e}")
    print("="*60)

print_file_lines(r"C:\Users\omris\TAU\hallucination_detection\RESEARCH_PROPOSAL_JUNE2026.md", 50, 95)
print_file_lines(r"C:\Users\omris\TAU\hallucination_detection\Research_Directions.md", 320, 360)
