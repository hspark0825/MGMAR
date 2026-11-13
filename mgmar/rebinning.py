"""Fan-beam <-> parallel-beam rebinning of AAPM CT-MAR sinograms.

fan2para, para2fan and fanbeam reproduce the MATLAB Image Processing Toolbox functions of the same
name (arc detector). Public functions take and return sinograms as [view, detector].
"""
import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator

N_VIEWS = 1000
VIEW_STEP = 360.0 / N_VIEWS

# detector-offset compensation (channels) applied when rebinning to / from the parallel geometry
SHIFT_TO_PARALLEL = {500: 1.7, 360: 1.25 * 512 / 360, 355: 1.25 * 512 / 355, 650: 1.7}
SHIFT_TO_FAN = {500: 1.7, 360: 1.25 * 512 / 360, 355: 2.0, 650: 1.7}


def fan_to_parallel(sino, fov):
    """Measured fan-beam sinogram [1000, 900] -> parallel-beam sinogram [1000, n_bins]."""
    d, dgamma = _geometry(fov)
    fan = _reverse_views(sino.T.astype(np.float64))
    return np.ascontiguousarray(_fan2para(_fshift(fan, -SHIFT_TO_PARALLEL[fov]), d, dgamma).T)


def parallel_to_fan(sino, fov):
    """Parallel-beam sinogram [1000, n_bins] -> fan-beam sinogram [1000, 900] in the measured geometry."""
    d, dgamma = _geometry(fov)
    fan = _sensors(_para2fan(sino.T, d, dgamma, full_cycle=True, kind='spline'))
    return np.ascontiguousarray(_reverse_views(_fshift(fan, SHIFT_TO_FAN[fov])).T)


def parallel_trace(mask, fov):
    """Metal trace [1000, n_bins] of a 512 x 512 metal mask in the parallel geometry."""
    d, dgamma = _geometry(fov)
    proj = _fan2para(_sensors(_fanbeam(mask[::-1] > 0.5, d, dgamma)), d, dgamma)
    return np.ascontiguousarray((proj > 0.9).T)


def _geometry(fov):
    """Source-isocentre distance (pixels) and fan sensor spacing (degrees) on the 512-pixel grid."""
    sad, sdd, du = 550.0 * 512 / fov, 950.0 * 512 / fov, 1.0 * 512 / fov
    return sad, du / sdd * 180 / np.pi


def _reverse_views(x):
    return np.hstack([x[:, :1], x[:, :0:-1]])


def _sensors(x):
    """The 900 sensors (index -449..450 about the central ray) of a fan sinogram."""
    c = (x.shape[0] - 1) // 2
    return x[c - 449:c + 451]


def _fshift(x, s):
    """Circular fractional shift by s samples along axis 0 (FFT phase ramp)."""
    n = x.shape[0]
    f = (np.arange(1, n + 1) - (n // 2 + 1)) / (n / 2)
    p = np.exp(-1j * s * np.pi * f)
    if n % 2 == 0:
        p[0] = p[0].real
    return np.fft.ifft(np.fft.fft(x, axis=0) * np.fft.ifftshift(p)[:, None], axis=0).real


def _interp(x, y, xq, kind, axis=0):
    f = CubicSpline(x, y, axis=axis) if kind == 'spline' else PchipInterpolator(x, y, axis=axis)
    return f(xq)


def _interp_rows(x, y, xq, kind):
    return np.stack([_interp(x, y[i], xq[i], kind) for i in range(y.shape[0])])


def _pad_to_odd(x):
    zero = np.zeros((1, x.shape[1]))
    if x.shape[0] % 2 == 0:
        x = np.vstack([zero, x])
    return np.vstack([zero, x, zero])


def _centered_grid(step, lo, hi):
    return step * np.arange(-int(np.floor(-lo / step)), int(np.floor(hi / step)) + 1)


def _radon(image):
    """MATLAB radon(image, 0:0.5:179.5)."""
    m, n = image.shape
    xo, yo = (n - 1) // 2, (m - 1) // 2
    r_last = int(np.ceil(np.sqrt((m - 1 - yo) ** 2 + (n - 1 - xo) ** 2))) + 1
    rows, cols = np.nonzero(image)
    val = image[rows, cols] * 0.25
    x, y = (cols - xo).astype(float), (yo - rows).astype(float)
    theta = np.arange(360) * 0.5 * np.pi / 180
    out = np.zeros((2 * r_last + 1, theta.size))
    for k in range(theta.size):
        c, s = np.cos(theta[k]), np.sin(theta[k])
        col = np.zeros(2 * r_last + 2)
        for dx, dy in ((-0.25, 0.25), (0.25, 0.25), (-0.25, -0.25), (0.25, -0.25)):
            r = (x + dx) * c + (y + dy) * s + r_last
            r0 = r.astype(np.int64)
            w = r - r0
            np.add.at(col, r0, val * (1 - w))
            np.add.at(col, r0 + 1, val * w)
        out[:, k] = col[:-1]
    return out


def _para2fan(par, d, dgamma, full_cycle, kind):
    """F(gamma, beta) = P(t = d sin(gamma), theta = beta - gamma). par covers 360 (full_cycle) or 180 degrees."""
    par = _pad_to_odd(np.asarray(par, float))
    m, n = par.shape
    t = np.arange(-((m - 1) // 2), m // 2 + 1)
    gamma = _centered_grid(np.deg2rad(dgamma), np.arcsin(t.min() / d), np.arcsin(t.max() / d))
    gdeg = np.rad2deg(gamma)
    theta = (360.0 if full_cycle else 180.0) / n * np.arange(n)
    beta = np.arange(0, 360 - VIEW_STEP / 2, VIEW_STEP)

    par = _interp(t, par, d * np.sin(gamma), kind)
    if not full_cycle:
        par, theta = np.hstack([par, par[::-1]]), np.concatenate([theta, theta + 180])
    span = gdeg.max() - gdeg.min()
    lo, hi = theta >= 360 - span, theta <= span
    par = np.hstack([par[:, lo], par, par[:, hi]])
    theta = np.concatenate([theta[lo] - 360, theta, theta[hi] + 360])

    fan = _interp_rows(theta, par, beta[None] - gdeg[:, None], kind)
    if full_cycle:                       # average with the conjugate rays P(-t, theta + 180)
        beta2 = np.mod(beta + 180, 360) - 180
        fan = (fan + _interp_rows(theta - 180, par[::-1], beta2[None] - gdeg[:, None], kind)) / 2
    fan[np.isnan(fan)] = 0
    return fan


def _fan2para(fan, d, dgamma):
    """P(t, theta) = F(gamma: d sin(gamma) = t, beta = theta + gamma), spline interpolation, 360 degrees."""
    fan = _pad_to_odd(np.asarray(fan, float))
    m, n = fan.shape
    gdeg = np.arange(-((m - 1) // 2), m // 2 + 1) * dgamma
    gamma = np.deg2rad(gdeg)
    step = 360.0 / n
    beta = np.arange(n) * 360.0 / n
    t = _centered_grid(1.0, d * np.sin(gamma.min()), d * np.sin(gamma.max()))
    theta = np.arange(0, 180 - step / 2, step)
    wrap = int(np.ceil(n / 4))
    fan = np.hstack([fan[:, n - wrap:], fan])
    beta = np.concatenate([beta[n - wrap:] - 360, beta])
    par = _interp(d * np.sin(gamma), _interp_rows(beta, fan, theta[None] + gdeg[:, None], 'spline'), t, 'spline')
    par[np.isnan(par)] = 0
    return np.hstack([par, par[::-1]])


def _fanbeam(image, d, dgamma):
    """MATLAB fanbeam(image, d, 'FanSensorSpacing', dgamma, 'FanRotationIncrement', 0.36)."""
    return _para2fan(_radon(image.astype(float)), d, dgamma, full_cycle=False, kind='cubic')
