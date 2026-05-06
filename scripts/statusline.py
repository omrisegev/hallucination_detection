import sys, json, io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

try:
    data = json.load(sys.stdin)
    used = data.get('context_window', {}).get('used_percentage')
    remaining = data.get('context_window', {}).get('remaining_percentage')
    model = data.get('model', {}).get('display_name', '')
    cwd = data.get('cwd', data.get('workspace', {}).get('current_dir', ''))

    if len(cwd) > 40:
        parts = cwd.replace('\\', '/').split('/')
        cwd = '.../' + '/'.join(parts[-2:])

    if used is not None and remaining is not None:
        filled = int(used / 10)
        bar = '=' * filled + '-' * (10 - filled)
        sys.stdout.write(f'\033[2m{model}  |  {cwd}  |  [{bar}] {used:.0f}% ctx\033[0m')
    else:
        sys.stdout.write(f'\033[2m{model}  |  {cwd}\033[0m')
    sys.stdout.flush()
except Exception:
    pass
