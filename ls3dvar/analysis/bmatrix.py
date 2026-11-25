# ls3dvar/analysis/bmatrix.py
"""
Background-error structures for LS-3DVar control vectors.

HybridControlRuntime: Gaussian-correlated state-space blocks (one flat control vector)

    v = [ v_1 (1, H, W, 1) | v_2 ... ]

    x(v)  = x_b + sum_i e_{c_i} * sigma_i * G_{L_i}(v_i)
    Jb    = 1/2 |v|^2

blocks: list of (channel_index, sigma_normalized, corr_length_cells, name[, mask]).
Each block is a variance-preserving Gaussian-correlated field on the model grid added to one
channel (e.g. PRES, T2M, D2M, U10, V10, REFC). Optional mask (H, W) in [0, 1] multiplies the
block's standard deviation (e.g. land fraction).
"""
from __future__ import annotations

import numpy as np
import tensorflow as tf


def gaussian_kernel_1d(L):
    if L <= 0:
        return np.ones(1, np.float32)
    r = int(np.ceil(3 * L))
    x = np.arange(-r, r + 1, dtype=np.float64)
    k = np.exp(-0.5 * (x / L) ** 2)
    k /= np.sqrt(np.sum(k ** 2))          # 1D: sum k^2 = 1; 2D separable -> also 1
    return k.astype(np.float32)


def gaussian_smooth(v, k1):
    """Separable Gaussian along y and x of a (1, ny, nx, k) field.
    k1: 1D kernel (normalized sum k^2 = 1 -> variance preserving)."""
    r = (len(k1) - 1) // 2
    if r == 0:
        return v
    ky = tf.constant(np.asarray(k1, np.float32).reshape(-1, 1, 1, 1))
    kx = tf.constant(np.asarray(k1, np.float32).reshape(1, -1, 1, 1))
    f = tf.transpose(v, [3, 1, 2, 0])                      # (k, ny, nx, 1)
    f = tf.pad(f, [[0, 0], [r, r], [0, 0], [0, 0]], mode="SYMMETRIC")
    f = tf.nn.conv2d(f, ky, strides=1, padding="VALID")
    f = tf.pad(f, [[0, 0], [0, 0], [r, r], [0, 0]], mode="SYMMETRIC")
    f = tf.nn.conv2d(f, kx, strides=1, padding="VALID")
    return tf.transpose(f, [3, 1, 2, 0])


class HybridControlRuntime:
    """state-space block control; the latent state stays at the background (bmat must be None)"""

    def __init__(self, base_runtime, bmat, latent_b, sigma_b, blocks,
                 grid_shape, n_channels):
        if bmat is not None:
            raise ValueError("this release has no latent B: use --control blocks")
        self.base = base_runtime
        self.bmat = None
        self.latent_b = tf.convert_to_tensor(latent_b, tf.float32)
        self.sigma_b = float(sigma_b)
        self.H, self.W = int(grid_shape[0]), int(grid_shape[1])
        self.blocks = []
        for blk in blocks:
            ch, sig, L, name = blk[:4]
            mask = blk[4] if len(blk) > 4 else None
            e = np.zeros((1, 1, 1, n_channels), np.float32)
            e[..., ch] = 1.0
            amp = float(sig) if mask is None else tf.constant(
                (float(sig) * np.asarray(mask, np.float32))[None, :, :, None])
            self.blocks.append(dict(ch=ch, sigma=amp, k1=gaussian_kernel_1d(float(L)),
                                    e=tf.constant(e), L=float(L), name=name,
                                    masked=mask is not None))
        self.n_z = 0
        self.n_blk = self.H * self.W
        self.size = self.n_blk * len(self.blocks)

    @property
    def control_shape(self):
        return (self.size,)

    def _parts(self, v):
        v = tf.reshape(tf.convert_to_tensor(v, tf.float32), [-1])
        return None, [tf.reshape(v[i * self.n_blk:(i + 1) * self.n_blk], (1, self.H, self.W, 1))
                      for i in range(len(self.blocks))]

    def to_latent(self, v):
        return self.latent_b

    def block_increment(self, v):
        """sum of direct channel increments, normalized units (1, H, W, C)"""
        _, fields = self._parts(v)
        inc = 0.0
        for b, f in zip(self.blocks, fields):
            inc = inc + b["sigma"] * gaussian_smooth(f, b["k1"]) * b["e"]
        return inc

    def forecast_only(self, v, state):
        x = self.base.analysis_state(self.latent_b, state)
        return x + self.block_increment(v) if self.blocks else x

    analysis_state = forecast_only
