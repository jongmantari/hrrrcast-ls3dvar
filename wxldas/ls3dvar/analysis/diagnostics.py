# ls3dvar/analysis/diagnostics.py  (extracted from mantari/ls3dvar/diagnostics.py)
"""
mantari/ls3dvar/diagnostics.py

Convergence plots and innovation summaries for LS-3DVar.

plot_convergence works with the history of either solver:
    solver_multi       J, Jb, Jobs, J_<group>..., innov_rms_norm, grad_norm, grad_obs_norm, dJ
    solver_t2m_d2m_ps  J, Jb, Jobs, JT2M, JD2M, JPS, innov_rms, grad_norm, ...
It writes
    cost_convergence_outer_XX.png        J, Jb, Jobs
    observation_costs_outer_XX.png       one line per observation group
    innovation_gradient_outer_XX.png     innovation rms, |grad J|, |grad Jo|
    lbfgs_history_outer_XX.csv           the raw history (same content the driver writes)
"""
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

LEGACY_GROUPS = ("JT2M", "JD2M", "JPS")


def _group_columns(keys):
    """observation-group cost columns: J_<name> (solver_multi) or JT2M/JD2M/JPS (legacy)"""
    cols = [k for k in keys if k.startswith("J_")]
    return cols or [k for k in LEGACY_GROUPS if k in keys]


def _plot(iters, series, title, ylabel, path, log=False):
    plt.figure(figsize=(10, 6))
    for label, values, kw in series:
        plt.plot(iters, values, label=label, **kw)
    if log:
        plt.yscale("log")
    plt.xlabel("Iteration")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _positive(values):
    """log axes cannot show values <= 0: replace them with NaN (gap in the line)"""
    return [v if v is not None and v > 0 else float("nan") for v in values]


def plot_convergence(history, output_dir, outer_loop):
    """Plot one outer loop's L-BFGS history; does nothing for an empty history."""
    if len(history) == 0:
        return

    keys = list(history[0].keys())
    iters = [h["iter"] for h in history]
    col = lambda k: [h.get(k) for h in history]

    _plot(iters,
          [("J", col("J"), {"linewidth": 3})]
          + [(k, col(k), {}) for k in ("Jb", "Jobs") if k in keys],
          f"LS3DVar Cost Convergence (Outer {outer_loop})", "Cost",
          output_dir / f"cost_convergence_outer_{outer_loop:02d}.png")

    groups = _group_columns(keys)
    if groups:
        _plot(iters,
              [(k[2:] if k.startswith("J_") else k, _positive(col(k)), {}) for k in groups],
              f"Observation Cost Components (Outer {outer_loop})", "Observation Cost",
              output_dir / f"observation_costs_outer_{outer_loop:02d}.png", log=True)

    innov = "innov_rms_norm" if "innov_rms_norm" in keys else "innov_rms"
    series = []
    if innov in keys:
        series.append(("Innovation RMS" + (" (normalized)" if innov.endswith("norm") else ""),
                       _positive(col(innov)), {}))
    if "grad_norm" in keys:
        series.append(("Gradient Norm", _positive(col("grad_norm")), {}))
    if "grad_obs_norm" in keys:
        series.append(("Obs Gradient Norm", _positive(col("grad_obs_norm")), {}))
    if series:
        _plot(iters, series, f"Innovation and Gradient (Outer {outer_loop})", "Magnitude",
              output_dir / f"innovation_gradient_outer_{outer_loop:02d}.png", log=True)

    pd.DataFrame(history).to_csv(
        output_dir / f"lbfgs_history_outer_{outer_loop:02d}.csv", index=False)


def rms(x):
    return float(tf.sqrt(tf.reduce_mean(x ** 2)))


def summarize_innovations(innovation_b, innovation_a):
    """O-B vs O-A statistics of two innovation vectors (same obs, same order)."""
    bg_rms = rms(innovation_b)
    an_rms = rms(innovation_a)
    return {
        "background_rms": bg_rms,
        "analysis_rms": an_rms,
        "background_mae": float(tf.reduce_mean(tf.abs(innovation_b))),
        "analysis_mae": float(tf.reduce_mean(tf.abs(innovation_a))),
        "background_bias": float(tf.reduce_mean(innovation_b)),
        "analysis_bias": float(tf.reduce_mean(innovation_a)),
        "improvement_pct": 100.0 * (bg_rms - an_rms) / bg_rms,
    }
