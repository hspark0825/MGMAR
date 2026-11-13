"""Normalized metal artifact reduction (NMAR): the measured sinogram is divided by a prior sinogram,
interpolated across the metal trace and multiplied back. Sinograms are [view, detector]."""
import numpy as np
from scipy.interpolate import PchipInterpolator


def nmar_parallel(sino, prior, trace):
    """NMAR with linear interpolation, parallel geometry. trace: True on metal bins."""
    sino, prior = sino.astype(np.float64), prior.astype(np.float64)
    ratio = (sino + 1) / (prior + 1)
    ratio[np.isinf(ratio)] = 0
    bins = np.arange(sino.shape[1])
    for v in range(sino.shape[0]):
        keep = np.flatnonzero(~trace[v])
        ratio[v] = _interp_linear(keep, ratio[v, keep], bins)
    return ratio * prior


def nmar_fan(sino, prior, trace):
    """NMAR with monotone cubic interpolation, fan geometry. trace: True on metal bins."""
    ratio = (sino + 1.0) / (prior + 1.0)
    for v in np.flatnonzero(trace.any(axis=1)):
        keep, metal = np.flatnonzero(~trace[v]), np.flatnonzero(trace[v])
        ratio[v, metal] = PchipInterpolator(keep, ratio[v, keep])(metal)
    return ratio * (prior + 1.0) - 1.0


def _interp_linear(x, y, xq):
    i = np.clip(np.searchsorted(x, xq, side='right') - 1, 0, len(x) - 2)
    w = (xq - x[i]) / (x[i + 1] - x[i])
    out = y[i] * (1 - w) + y[i + 1] * w
    out[(xq < x[0]) | (xq > x[-1])] = np.nan
    return out
