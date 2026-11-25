# ls3dvar/obs/operators.py  (extracted from mantari/ls3dvar/operators.py)
import numpy as np
import tensorflow as tf

#
# Forecast channels
#
PRES_CHANNEL = 120
MSLMA_CHANNEL = 121

T2M_CHANNEL = 123
D2M_CHANNEL = 128

#
# Normalization statistics
# net-diffusion/normalize-stats.nc
#
PRES_MEAN = 96069.5
PRES_STD = 571.6757

MSLMA_MEAN = 99621.234
MSLMA_STD = 617.72125

T2M_MEAN = 288.07220458984375
T2M_STD = 8.021387

D2M_MEAN = 281.51956
D2M_STD = 8.593886

#
# Hydrostatic constants
#
G = 9.80665          # m s-2
RD = 287.05          # J kg-1 K-1
LAPSE = 0.0065       # K m-1, standard atmosphere


def gather_points(
    field,
    obs_y,
    obs_x,
):
    """
    Sample model field at observation locations.
    """

    return tf.gather_nd(
        field[0],
        tf.stack(
            [
                obs_y,
                obs_x,
            ],
            axis=1,
        ),
    )


def t2m_observation_operator(
    forecast,
    obs_y,
    obs_x,
):
    """
    2-m temperature (K)
    """

    t2m = (
        forecast[..., T2M_CHANNEL]
        * T2M_STD
        + T2M_MEAN
    )

    return gather_points(
        t2m,
        obs_y,
        obs_x,
    )


def d2m_observation_operator(
    forecast,
    obs_y,
    obs_x,
):
    """
    2-m dewpoint (K)
    """

    d2m = (
        forecast[..., D2M_CHANNEL]
        * D2M_STD
        + D2M_MEAN
    )

    return gather_points(
        d2m,
        obs_y,
        obs_x,
    )


def pres_observation_operator(
    forecast,
    obs_y,
    obs_x,
):
    """
    Surface pressure (Pa) at the MODEL surface, no height correction.
    """

    pres = (
        forecast[..., PRES_CHANNEL]
        * PRES_STD
        + PRES_MEAN
    )

    return gather_points(
        pres,
        obs_y,
        obs_x,
    )


def mslp_observation_operator(
    forecast,
    obs_y,
    obs_x,
):
    """
    Mean sea-level pressure (Pa)
    """

    mslp = (
        forecast[..., MSLMA_CHANNEL]
        * MSLMA_STD
        + MSLMA_MEAN
    )

    return gather_points(
        mslp,
        obs_y,
        obs_x,
    )


# =====================================================
# STATION-HEIGHT CORRECTED PRESSURE
# =====================================================

def station_height_offsets(observations, obs_y, obs_x, orog):
    """
    dz = station elevation - model terrain (m), one value per observation.

    observations : list of SurfaceObservation (elevation may be None)
    orog         : (H, W) model terrain height [m] (hrrr npz OROG_raw)

    Missing station elevation -> dz = 0 (no correction).
    """
    iy = np.asarray(obs_y)
    ix = np.asarray(obs_x)
    z_model = np.asarray(orog, np.float64)[iy, ix]
    z_stn = np.array(
        [np.nan if o.elevation is None else o.elevation for o in observations],
        np.float64,
    )
    missing = ~np.isfinite(z_stn)
    dz = np.where(missing, 0.0, z_stn - z_model)
    return dz.astype(np.float32), int(missing.sum())


def make_station_pressure_operator(dz):
    """
    Returns H(forecast, obs_y, obs_x) giving model pressure at the STATION
    elevation:

        p_stn = p_sfc * exp( -g dz / (Rd Tv) )

        Tv = (T2M - LAPSE * dz/2) (1 + 0.608 q),  q from model D2M and p_sfc

    dz (n_obs,) must be in the same order as obs_y / obs_x. Fully
    differentiable (p_sfc, T2M, D2M all come from the state).
    """
    dz = tf.constant(np.asarray(dz, np.float32))

    def operator(forecast, obs_y, obs_x):
        p = gather_points(forecast[..., PRES_CHANNEL] * PRES_STD + PRES_MEAN, obs_y, obs_x)
        t = gather_points(forecast[..., T2M_CHANNEL] * T2M_STD + T2M_MEAN, obs_y, obs_x)
        td = gather_points(forecast[..., D2M_CHANNEL] * D2M_STD + D2M_MEAN, obs_y, obs_x)

        td_c = td - 273.15
        e = 611.2 * tf.exp(17.67 * td_c / (td_c + 243.5))      # Pa
        q = 0.622 * e / (p - 0.378 * e)

        t_mid = t - LAPSE * 0.5 * dz
        tv = t_mid * (1.0 + 0.608 * q)

        return p * tf.exp(-G * dz / (RD * tv))

    return operator


# =====================================================
# GENERIC CHANNEL + 10-M WIND
# =====================================================

def make_channel_operator(channel, mean, std):
    """H(x) = x[..., channel] * std + mean at the obs points (physical units)."""
    channel, mean, std = int(channel), float(mean), float(std)

    def operator(forecast, obs_y, obs_x):
        return gather_points(forecast[..., channel] * std + mean, obs_y, obs_x)

    return operator


def make_wind_operator(component, iu, iv, stats_u, stats_v, cos_t=None, sin_t=None):
    """
    10-m wind component in EARTH-relative m/s (obs: windEastwardAt10M / windNorthwardAt10M).

    component : "u" or "v"
    iu, iv    : channel indices of the model U10 / V10
    stats_*   : (mean, std) normalization of those channels
    cos_t/sin_t (H, W): grid rotation (channels.grid_rotation); None = model winds
                are already earth-relative.

        u_e = u_g cos - v_g sin ,   v_e = u_g sin + v_g cos
    """
    iu, iv = int(iu), int(iv)
    mu_u, sd_u = map(float, stats_u)
    mu_v, sd_v = map(float, stats_v)
    rotate = cos_t is not None
    if rotate:
        c_f = tf.constant(np.asarray(cos_t, np.float32)[None])
        s_f = tf.constant(np.asarray(sin_t, np.float32)[None])

    def operator(forecast, obs_y, obs_x):
        u = gather_points(forecast[..., iu] * sd_u + mu_u, obs_y, obs_x)
        v = gather_points(forecast[..., iv] * sd_v + mu_v, obs_y, obs_x)
        if not rotate:
            return u if component == "u" else v
        c = gather_points(c_f, obs_y, obs_x)
        s = gather_points(s_f, obs_y, obs_x)
        return u * c - v * s if component == "u" else u * s + v * c

    return operator


# =====================================================
# T2M / D2M STATION-HEIGHT (LAPSE-RATE) CORRECTION
# =====================================================

def make_lapse_operator(base_operator, dz, lapse_k_per_m):
    """
    H(x) at the station elevation:  T_stn = T_model - lapse * (z_stn - z_model)

    base_operator : t2m / d2m operator (model value at the grid point, K)
    dz            : (n_obs,) z_stn - z_model [m], same order as obs_y / obs_x
    lapse_k_per_m : 0.0065 for T2M, ~0.0018 for D2M (positive = cooler aloft)
    """
    corr = tf.constant(-float(lapse_k_per_m) * np.asarray(dz, np.float32), tf.float32)

    def operator(forecast, obs_y, obs_x):
        return base_operator(forecast, obs_y, obs_x) + corr

    return operator


def lapse_settings(cfg):
    """{'t2m': K/m, 'd2m': K/m} from observations.temperature_height_correction (or {})"""
    t = (cfg.get("observations", {}) or {}).get("temperature_height_correction") or {}
    if not t.get("enabled", False):
        return {}
    return {"t2m": float(t.get("t2m_lapse_k_per_km", 6.5)) / 1000.0,
            "d2m": float(t.get("d2m_lapse_k_per_km", 1.8)) / 1000.0}


# ----------------------------------------------------------------------------
# gridded pseudo-observations (superobs of a reference field, e.g. HRRR analysis)
# ----------------------------------------------------------------------------
def superob_boxes(H, W, box):
    """(by, bx) indices of all complete box x box tiles (row-major), as int32 arrays"""
    ny, nx = H // int(box), W // int(box)
    by, bx = np.mgrid[0:ny, 0:nx]
    return by.ravel().astype(np.int32), bx.ravel().astype(np.int32)


def make_superob_operator(channel, mean, std, box, floor=None, leak=0.2):
    """
    H(x) = box mean of g(x[..., channel] * std + mean) over box x box tiles, sampled at
    tile indices (by, bx). Apply the SAME operator to the reference field to get y,
    so obs and model are aggregated identically.

    floor (physical units, e.g. 0 dBZ for REFC): values below the floor are compressed,
        g(f) = f                              f >= floor
             = floor + leak * (f - floor)     f <  floor
      so "no echo" at -10 or -30 dBZ counts (almost) the same, while the gradient stays
      non-zero (leak) and the analysis can still create an echo where the model has none.
    """
    channel, mean, std, box = int(channel), float(mean), float(std), int(box)

    def operator(forecast, obs_y, obs_x):
        f = forecast[..., channel] * std + mean                       # (1, H, W)
        if floor is not None:
            d = f - float(floor)
            f = float(floor) + tf.maximum(d, 0.0) + float(leak) * tf.minimum(d, 0.0)
        p = tf.nn.avg_pool2d(f[..., None], ksize=box, strides=box, padding="VALID")[..., 0]
        return gather_points(p, obs_y, obs_x)

    return operator
