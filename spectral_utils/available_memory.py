"""Available host RAM, bounded by the current Linux cgroup when present."""
from __future__ import annotations

import ctypes
from pathlib import Path
import sys


def cgroup_remaining(root):
    """Read a v2 or v1 limit. Unlimited cgroups impose no extra bound."""
    root = Path(root)
    candidates = [('memory.max', 'memory.current'),
                  ('memory.limit_in_bytes', 'memory.usage_in_bytes')]
    for limit_name, usage_name in candidates:
        limit_path, usage_path = root / limit_name, root / usage_name
        if not limit_path.exists() or not usage_path.exists():
            continue
        limit = limit_path.read_text().strip()
        if limit == 'max':
            continue
        limit = int(limit)
        if limit >= 2**60:
            continue
        return max(0, limit - int(usage_path.read_text().strip()))
    return None


def available_gib():
    if sys.platform == 'win32':
        class Memory(ctypes.Structure):
            _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong)] + [
                (n, ctypes.c_ulonglong) for n in ('total', 'available', 'total_page',
                'available_page', 'total_virtual', 'available_virtual', 'extended')]
        m = Memory(); m.length = ctypes.sizeof(m)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
            raise OSError('cannot check available RAM')
        return m.available / 2**30
    if sys.platform != 'linux':
        raise OSError('unsupported memory accounting platform')
    lines = Path('/proc/meminfo').read_text().splitlines()
    values = {line.split(':', 1)[0]: line.split(':', 1)[1].strip() for line in lines}
    available = int(values['MemAvailable'].split()[0]) * 1024
    roots = {Path('/sys/fs/cgroup'), Path('/sys/fs/cgroup/memory')}
    for line in Path('/proc/self/cgroup').read_text().splitlines():
        _, controllers, relative = line.split(':', 2)
        if controllers == '' or 'memory' in controllers.split(','):
            base = Path('/sys/fs/cgroup') if controllers == '' else Path('/sys/fs/cgroup/memory')
            # Containers may namespace the mount at the current cgroup root.
            path = base / relative.lstrip('/')
            if '..' not in path.parts:
                roots.update([path, *[p for p in path.parents if p == base or base in p.parents]])
    for root in roots:
        remaining = cgroup_remaining(root)
        if remaining is not None:
            available = min(available, remaining)
    return available / 2**30
