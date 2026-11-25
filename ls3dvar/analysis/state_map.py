# ls3dvar/analysis/state_map.py  (extracted from mantari/ls3dvar/analysis_state_runtime_tf.py)
"""
mantari/ls3dvar/analysis_state_runtime_tf.py

Latent -> analysis-time state map for LS-3DVar (fully differentiable).

    x(z) = x_b + unpad( refine(z) - refine(z_b) )

    refine(z) = DecoderRuntimeTF(z, skip1, skip0, processed, time_vec)

No processor, no forecast.  Anchored, so x(z_b) == x_b exactly and the
first innovations are the true O-B.  Skips / processed / time_vec stay at
their background values (from extract_latent on the analysis input).

Drop-in for FullLatentForecastRuntimeTF: exposes forecast_only(latent, state)
so the existing objective_* functions work unchanged, but what it returns is
the ANALYSIS-TIME state, not a forecast.
"""
from __future__ import annotations

import tensorflow as tf

from ls3dvar.hrrrcast.decoder import DecoderRuntimeTF
from ls3dvar.hrrrcast.tail import TailModelRuntimeTF


class AnalysisStateRuntimeTF:

    def __init__(self, model, x_b, latent_b, state):
        self.decoder = DecoderRuntimeTF(model)
        self.tail = TailModelRuntimeTF()
        self.x_b = tf.convert_to_tensor(x_b, tf.float32)
        self.refine_b = self._refine(tf.convert_to_tensor(latent_b, tf.float32), state)

    def _refine(self, latent, state):
        return self.decoder.decode_features(
            latent,
            state["skip1"],
            state["skip0"],
            state["processed"],
            state["time_vec"],
        )["refine"]

    def analysis_state(self, latent, state):
        return self.x_b + self.tail.unpad(self._refine(latent, state) - self.refine_b)

    # drop-in name used by objective_t2m_d2m_ps.compute_cost_t2m_d2m_ps
    forecast_only = analysis_state

    # ------------------------------------------------------------------
    # checks
    # ------------------------------------------------------------------
    def anchor_error(self, latent_b, state):
        x0 = self.analysis_state(tf.convert_to_tensor(latent_b, tf.float32), state)
        return float(tf.reduce_max(tf.abs(x0 - self.x_b)))

    def gradient_norm(self, latent_b, state, seed=0):
        """|d<w, x(z)>/dz| at z_b for random w: must be > 0."""
        z = tf.Variable(tf.convert_to_tensor(latent_b, tf.float32))
        w = tf.random.stateless_normal(self.x_b.shape, seed=[seed, 1])
        with tf.GradientTape() as tape:
            s = tf.reduce_sum(w * self.analysis_state(z, state))
        g = tape.gradient(s, z)
        return None if g is None else float(tf.norm(g))
