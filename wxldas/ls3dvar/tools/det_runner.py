"""
deterministic runner (same as the one ls3dvar/tools/gate.py writes; used by
scripts/cycle_ls3dvar.sh with DETERMINISTIC=1)

    python _det_runner.py MODULE [args...]      ==   python -m MODULE [args...]

TF has no deterministic GPU kernel for the gradient of nearest-neighbour resize, which the
LS-3DVar decoder uses. For EXACT integer up-sampling (the decoder: 34->68->136->272),
nearest resize is a pure copy, so tf.repeat gives a bit-identical forward result and a
deterministic gradient. This runner swaps it in, checks the swap against the original
resize once per shape (stops on any mismatch), then runs MODULE as __main__.
Other resizes are passed through unchanged.
"""
import os
import runpy
import sys

sys.path[0] = os.getcwd()                         # like python -m: repo root first
mod = sys.argv[1]
sys.argv = [mod] + sys.argv[2:]

import tensorflow as tf  # noqa: E402

_orig = tf.image.resize
_checked = set()


def _resize(images, size, method="bilinear", preserve_aspect_ratio=False, antialias=False, name=None):
    kw = dict(method=method, preserve_aspect_ratio=preserve_aspect_ratio, antialias=antialias, name=name)
    if str(method).lower() == "nearest" and not preserve_aspect_ratio and images.shape.rank == 4:
        h, w = images.shape[1], images.shape[2]
        try:
            H, W = int(size[0]), int(size[1])
        except (TypeError, ValueError):
            H = W = None
        if h and w and H and W and H % h == 0 and W % w == 0:
            out = tf.repeat(tf.repeat(images, H // h, axis=1), W // w, axis=2)
            key = (tuple(images.shape), H, W, str(images.dtype))
            if key not in _checked:
                ref = _orig(images, size, **kw)
                if ref.dtype != out.dtype or ref.shape != out.shape or not bool(tf.reduce_all(tf.equal(ref, out))):
                    raise RuntimeError(f"gate det runner: repeat != nearest resize for {key}")
                _checked.add(key)
                print(f"[gate det] nearest {h}x{w} -> {H}x{W}: tf.repeat, forward bit-identical", file=sys.stderr)
            return out
    return _orig(images, size, **kw)


tf.image.resize = _resize
runpy.run_module(mod, run_name="__main__", alter_sys=True)
