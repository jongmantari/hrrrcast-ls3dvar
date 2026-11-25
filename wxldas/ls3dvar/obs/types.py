# ls3dvar/obs/types.py  (extracted from mantari/observations.py)
from dataclasses import dataclass
from typing import Optional

import numpy as np


# =====================================================
# OBSERVATION
# =====================================================

@dataclass
class SurfaceObservation:

    station: str

    lat: float

    lon: float

    variable: str

    value: float

    error: float = 1.0

    source: str = "unknown"

    #
    # observation valid time, numpy datetime64[s] (None if not in file)
    #
    time: Optional[np.datetime64] = None

    #
    # station elevation [m] (None if not in file); for pressure height correction
    #
    elevation: Optional[float] = None


# =====================================================
# OBS SET
# =====================================================

class SurfaceObservationSet:

    def __init__(
        self,
        observations,
    ):

        self.observations = observations

    def values(self):
        return np.array([obs.value for obs in self.observations], dtype=np.float32)

    def errors(self):
        return np.array([obs.error for obs in self.observations], dtype=np.float32)

    def latitudes(self):
        return np.array([obs.lat for obs in self.observations], dtype=np.float32)

    def longitudes(self):
        return np.array([obs.lon for obs in self.observations], dtype=np.float32)

    def stations(self):
        return [obs.station for obs in self.observations]

    def times(self):
        return np.array(
            [obs.time if obs.time is not None else np.datetime64("NaT")
             for obs in self.observations],
            dtype="datetime64[s]",
        )

    def filter_variable(
        self,
        variable,
    ):
        return SurfaceObservationSet(
            [obs for obs in self.observations if obs.variable == variable]
        )

    def count(self):
        return len(self.observations)

    def variable_counts(self):
        counts = {}
        for obs in self.observations:
            counts[obs.variable] = counts.get(obs.variable, 0) + 1
        return counts

    def summary(self):
        print()
        print("=" * 80)
        print("SURFACE OBSERVATIONS")
        print("=" * 80)
        print("count:", self.count())
        print()
        for variable, count in sorted(self.variable_counts().items()):
            print(f"{variable:30s} {count:8d}")
        print()
