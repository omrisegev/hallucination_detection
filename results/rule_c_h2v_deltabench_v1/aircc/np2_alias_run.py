"""Run a script so that pickles written under numpy 2 (module path numpy._core.*) load under the container's numpy 1.26
(numpy.core.*). Used ONLY for the stored C cells (built 2026-10-26 in the native numpy-2 pyenv): the H2V builder parity
step and the DeltaBench C extract_steps step. The script itself runs unchanged (runpy, __main__).
    python np2_alias_run.py SCRIPT [args...]"""
import importlib, runpy, sys
import numpy
if not numpy.__version__.startswith('1.'):
    pass                                                   # numpy 2: nothing to alias
else:
    import numpy.core
    sys.modules.setdefault('numpy._core', numpy.core)
    for sub in ('multiarray', 'numeric', '_multiarray_umath', 'fromnumeric', 'umath', 'records', 'overrides', 'arrayprint', 'shape_base'):
        try:
            sys.modules['numpy._core.' + sub] = importlib.import_module('numpy.core.' + sub)
        except ImportError:
            pass
script = sys.argv[1]; sys.argv = sys.argv[1:]
runpy.run_path(script, run_name='__main__')
