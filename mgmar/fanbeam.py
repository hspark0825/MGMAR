"""Fan-beam forward projection and FBP in the AAPM CT-MAR geometry (wrapper of libfanbeam.so)."""
import ctypes
from ctypes import c_float, c_int
from pathlib import Path

import numpy as np
from numpy.ctypeslib import ndpointer
from scipy.interpolate import interp1d

DSO, DSD = 550.0, 950.0                  # source-isocentre and source-detector distance (mm)
N_VIEWS, N_CHANNELS = 1000, 900
CHANNEL_OFFSET = 1.25
N = 512                                  # image size
FILTER_SIZE = 2048

_array = ndpointer(c_float, flags='C_CONTIGUOUS')
_lib = ctypes.cdll.LoadLibrary(str(Path(__file__).resolve().with_name('libfanbeam.so')))
_lib.fan_project.argtypes = [c_int, c_int, c_float, c_float, _array, _array, _array, _array, c_int, c_int, _array]
_lib.fan_project.restype = None
_lib.fan_backproject.argtypes = [c_float, c_float, c_int, c_int, c_float, _array, c_int, c_int, c_float, c_float,
                                 _array]
_lib.fan_backproject.restype = None


class FanBeam:
    def __init__(self, fov):
        self.dx = fov / N
        self.theta = np.linspace(0, 2 * np.pi, N_VIEWS, endpoint=False, dtype=np.float32)
        alpha = (np.arange(N_CHANNELS) - (N_CHANNELS - 1) * 0.5 - CHANNEL_OFFSET) * 2. * np.arctan2(0.5, DSD)
        self.xds = np.float32(DSD * np.sin(alpha) / self.dx)
        self.yds = np.float32((DSO - DSD * np.cos(alpha)) / self.dx)

        fan_angle = 2 * np.arctan2(0.5, DSD, dtype=np.float32) * N_CHANNELS
        self.dfan = fan_angle / N_CHANNELS
        channel = np.arange(N_CHANNELS, dtype=np.float64) - (N_CHANNELS - 1) * 0.5
        self.cosine = np.cos(channel * self.dfan).astype(np.float32)[None]

        # apodized ramp filter
        window = interp1d([0, 0.25, 0.5, 0.75, 1], [1, 0.9338, 0.7441, 0.4425, 0.0531], kind='quadratic')
        p = np.arange(FILTER_SIZE)
        kf = window(p / FILTER_SIZE) * 0.997 * (p + 0.003) / (2 * FILTER_SIZE)
        kb = window(p / FILTER_SIZE) * 0.997 * (p + 1.003) / (2 * FILTER_SIZE)
        self.kernel = np.concatenate([kf, kb[::-1]])[None] * np.csingle(1j)

    def project(self, image):
        """Line integrals of a 512 x 512 image in pixel units, [view, channel]."""
        sino = np.zeros((N_VIEWS, N_CHANNELS), np.float32)
        _lib.fan_project(N_CHANNELS, N_VIEWS, 0.0, DSO / self.dx, self.xds, self.yds, self.theta, sino,
                         N, N, np.ascontiguousarray(image, np.float32))
        return sino

    def fbp(self, sino):
        """Filtered backprojection of a [view, channel] sinogram to a 512 x 512 image."""
        weighted = np.multiply(sino, self.cosine).astype(np.float32)
        filtered = np.fft.ifft(np.fft.fft(weighted, 2 * FILTER_SIZE, axis=1) * self.kernel, axis=1)
        filtered = filtered[:, :N_CHANNELS].imag.astype(np.float32) / self.dfan
        image = np.zeros((N, N), np.float32)
        _lib.fan_backproject(DSD, DSO, N_CHANNELS, N_VIEWS, CHANNEL_OFFSET, filtered.ravel(),
                             N, N, self.dx, self.dx, image)
        return np.ascontiguousarray((image * (DSO * np.pi / N_VIEWS)).T)
