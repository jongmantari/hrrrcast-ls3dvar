# ls3dvar/obs/gross_filter.py  (extracted from mantari/ls3dvar/observation_filter.py)
import numpy as np


class ObservationFilter:

    def __init__(
        self,
        cfg,
    ):
        self.cfg = cfg

        self.filter_cfg = (
            cfg
            .get("ls3dvar", {})
            .get(
                "observation_filter",
                {},
            )
        )

    # =================================================
    # ENABLED?
    # =================================================

    @property
    def enabled(self):

        return self.filter_cfg.get(
            "enabled",
            False,
        )

    # =================================================
    # APPLY
    # =================================================

    def apply(
        self,
        variable,
        observations,
        innovations,
    ):

        if not self.enabled:

            return observations

        filter_type = (
            self.filter_cfg.get(
                "type",
                "innovation",
            )
        )

        if filter_type == "innovation":

            return self._innovation_filter(
                variable,
                observations,
                innovations,
            )

        if filter_type == "percentile":

            return self._percentile_filter(
                observations,
                innovations,
            )

        return observations

    # =================================================
    # INNOVATION THRESHOLD
    # =================================================

    def _innovation_filter(
        self,
        variable,
        observations,
        innovations,
    ):

        thresholds = (
            self.filter_cfg.get(
                "thresholds",
                {},
            )
        )

        threshold = thresholds.get(
            variable,
            None,
        )

        if threshold is None:

            return observations

        filtered = []

        for obs, innov in zip(
            observations,
            innovations,
        ):

            if (
                np.abs(innov)
                <= threshold
            ):
                filtered.append(
                    obs
                )

        print(
            f"[QC] {variable}: "
            f"{len(observations)} -> "
            f"{len(filtered)}"
        )

        return filtered

    # =================================================
    # PERCENTILE FILTER
    # =================================================

    def _percentile_filter(
        self,
        observations,
        innovations,
    ):

        percentile = (
            self.filter_cfg.get(
                "percentile",
                99,
            )
        )

        cutoff = np.percentile(
            np.abs(innovations),
            percentile,
        )

        filtered = []

        for obs, innov in zip(
            observations,
            innovations,
        ):

            if (
                np.abs(innov)
                <= cutoff
            ):
                filtered.append(
                    obs
                )

        print(
            f"[QC] percentile="
            f"{percentile} "
            f"cutoff={cutoff:.3f} "
            f"{len(observations)} -> "
            f"{len(filtered)}"
        )

        return filtered

    # =================================================
    # DIAGNOSTICS
    # =================================================

    @staticmethod
    def innovation_stats(
        innovations,
    ):

        innovations = np.asarray(
            innovations
        )

        # no observations of this variable in the window (e.g. a network without dew
        # point): report nan instead of failing in np.percentile on an empty array
        if innovations.size == 0:
            return {"count": 0, **{k: float("nan") for k in
                                   ("mean", "rms", "p50", "p75", "p90", "p95", "p99")}}

        return {
            "count":
                len(innovations),

            "mean":
                float(
                    np.mean(
                        innovations
                    )
                ),

            "rms":
                float(
                    np.sqrt(
                        np.mean(
                            innovations
                            ** 2
                        )
                    )
                ),

            "p50":
                float(
                    np.percentile(
                        np.abs(
                            innovations
                        ),
                        50,
                    )
                ),

            "p75":
                float(
                    np.percentile(
                        np.abs(
                            innovations
                        ),
                        75,
                    )
                ),

            "p90":
                float(
                    np.percentile(
                        np.abs(
                            innovations
                        ),
                        90,
                    )
                ),

            "p95":
                float(
                    np.percentile(
                        np.abs(
                            innovations
                        ),
                        95,
                    )
                ),

            "p99":
                float(
                    np.percentile(
                        np.abs(
                            innovations
                        ),
                        99,
                    )
                ),
        }
