# ls3dvar/analysis/solver.py  (extracted from mantari/ls3dvar/solver_multi.py)
"""
mantari/ls3dvar/solver_multi.py

L-BFGS-B LS-3DVar solver for any number of observation groups.

    J(c) = 1/2 |(c - c_b) / sigma_b|^2 + sum_k 1/2 |(H_k x(c) - y_k) / sigma_k|^2
    x(c) = runtime.forecast_only(c, state)      (analysis-time state, anchored decode)

groups: dict  name -> {"y", "s", "iy", "ix", "op", "n"}   (as built by the driver;
groups with n == 0 are skipped)

Same behaviour as solver_t2m_d2m_ps (first guess separate from background,
best-J tracking, early stop, Jo gradient checked on its own), but the history
records one J column per group (J_t2m, J_u10, ...).
"""
import numpy as np
import tensorflow as tf
from scipy.optimize import minimize


class LS3DVARMultiSolver:

    def __init__(self, runtime):
        self.runtime = runtime

    def cost(self, c, c_b, state, groups, sigma_b):
        x = self.runtime.forecast_only(c, state)
        Jb = 0.5 * tf.reduce_sum(tf.square((c - c_b) / sigma_b))
        Jk, innov = {}, []
        for name, g in groups.items():
            d = g["op"](x, g["iy"], g["ix"]) - g["y"]
            r = d / g["s"]
            Jk[name] = 0.5 * tf.reduce_sum(tf.square(r))
            innov.append(r)
        return Jb, Jk, tf.concat(innov, 0)

    def solve(self, latent_background, state, groups, sigma_b, max_iters,
              latent_first_guess=None, grad_tol=1.0e-3, cost_tol=1.0e-4,
              stop_rel=None, stop_window=5, stop_reduction=None, j_ref=None):
        """
        Stopping (all off by default = the original behaviour):
          stop_rel        stall: stop when the best J improved by less than stop_rel (relative)
                          over the last stop_window evaluations, e.g. 5e-3
          stop_reduction  stop when the best J <= (1 - stop_reduction) * j_ref
                          (j_ref: J at the first guess of outer loop 0; default this loop's first J)
        result.message tells which rule stopped ("stall", "reduction", "early stop", or scipy's).
        """
        groups = {k: g for k, g in groups.items() if g["n"] > 0}
        shape = latent_background.shape
        c_b = tf.convert_to_tensor(latent_background, tf.float32)
        first = c_b if latent_first_guess is None else latent_first_guess
        x0 = np.asarray(first, np.float64).reshape(-1)
        sigma_b = float(sigma_b)

        history, it, prev = [], [0], [None]
        best = {"J": np.inf, "x": x0.copy()}
        best_hist, why = [], ["early stop"]

        def objective(xv):
            c = tf.reshape(tf.convert_to_tensor(xv, tf.float32), shape)
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(c)
                Jb, Jk, r = self.cost(c, c_b, state, groups, sigma_b)
                Jo = tf.add_n(list(Jk.values()))
                J = Jb + Jo
            g_obs = tape.gradient(Jo, c)
            g = tape.gradient(J, c)
            del tape
            if g_obs is None:
                raise RuntimeError("Jo has no gradient w.r.t. the control: the runtime breaks "
                                   "the TF graph (numpy conversion).")
            grad = tf.reshape(g, [-1]).numpy().astype(np.float64)
            gn, gon = float(np.linalg.norm(grad)), float(tf.norm(g_obs))
            it[0] += 1
            Jf = float(J)
            if Jf < best["J"]:
                best.update(J=Jf, x=xv.copy())
            dJ = np.nan if prev[0] is None else prev[0] - Jf
            prev[0] = Jf
            row = {"iter": it[0], "J": Jf, "Jb": float(Jb), "Jobs": float(Jo),
                   "innov_rms_norm": float(tf.sqrt(tf.reduce_mean(r ** 2))),
                   "grad_norm": gn, "grad_obs_norm": gon, "dJ": dJ}
            row.update({f"J_{k}": float(v) for k, v in Jk.items()})
            history.append(row)
            print(f"ITER={it[0]:03d} J={Jf:12.4f} Jb={float(Jb):10.4f} "
                  + " ".join(f"J{k}={float(v):10.4f}" for k, v in Jk.items())
                  + f" |g|={gn:10.3e} |g_obs|={gon:10.3e}")
            if it[0] == 1 and gon == 0.0:
                raise RuntimeError("grad Jo == 0 at the first guess: no obs sensitivity.")
            if gn < grad_tol and it[0] > 5:
                print(f"EARLY STOP: |g|={gn:.3e}")
                raise StopIteration()
            if not np.isnan(dJ) and abs(dJ) < cost_tol and it[0] > 5:
                print(f"EARLY STOP: dJ={dJ:.3e}")
                raise StopIteration()
            best_hist.append(best["J"])
            if stop_reduction is not None:
                ref = j_ref if j_ref is not None else history[0]["J"]
                if best["J"] <= (1.0 - stop_reduction) * ref and it[0] > 5:
                    print(f"EARLY STOP: J reduced {100 * (1 - best['J'] / ref):.1f}% "
                          f"(>= {100 * stop_reduction:g}% of {ref:.4g})")
                    why[0] = "reduction"
                    raise StopIteration()
            if stop_rel is not None and len(best_hist) > stop_window:
                old = best_hist[-1 - stop_window]
                gain = (old - best["J"]) / max(abs(best["J"]), 1e-30)
                if gain < stop_rel:
                    print(f"EARLY STOP: stall, best J improved {100 * gain:.3f}% over the last "
                          f"{stop_window} evaluations (< {100 * stop_rel:g}%)")
                    why[0] = "stall"
                    raise StopIteration()
            return Jf, grad

        try:
            result = minimize(objective, x0, jac=True, method="L-BFGS-B",
                              options={"maxiter": max_iters})
        except StopIteration:
            class Result:
                pass
            result = Result()
            result.nit, result.success, result.message = it[0], True, why[0]

        c_a = tf.reshape(tf.convert_to_tensor(best["x"], tf.float32), shape)
        return c_a, result, history
