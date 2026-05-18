import sys as _sys
try:
    from .amni_kernels import *
    from . import amni_kernels as _ak_ext
    __doc__=_ak_ext.__doc__
    if hasattr(_ak_ext,"__all__"):__all__=_ak_ext.__all__
except ImportError as _e:
    _py=f'{_sys.version_info.major}.{_sys.version_info.minor}'
    raise ImportError(f"amni_kernels native extension failed to import on Python {_py}. The repo currently ships a prebuilt cp313-win_amd64 .pyd; for any other Python version or platform you need to build from source:\n  cd amni_kernels\n  pip install maturin\n  maturin develop --release\nOriginal error: {type(_e).__name__}: {_e}") from _e
