"""
Software package for two-photon holographic optogenetics
Copyright (C) 2014-2017  Joseph Donovan, Max Planck Institute of Neurobiology

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np
from math import sin, sqrt
from collections import namedtuple
from numpy.fft import fft2, ifft2, fftshift, ifftshift

pi = np.pi


class GSFresult(
    namedtuple('GSFresult', ['phase', 'target_fields', 'lenses', 'algorithm', 'errors', 'correlations'])):
    """
    Storage class for GSF results
    Namedtuple doesn't support default args :(
    """

    def __new__(cls, phase, target_fields=None, lenses=None, algorithm=None, errors=None, correlations=None):
        return super(GSFresult, cls).__new__(cls, phase, target_fields, lenses, algorithm, errors, correlations)


def GS_FFT(target_amplitude, iterations=30, replace_middle=True):
    # simple GS FFT propagator
    assert target_amplitude.shape[0] == target_amplitude.shape[1]
    ini_amplitude = np.random.rand(*target_amplitude.shape)
    slm_field = ini_amplitude

    corrs = []
    for i in range(iterations):
        target_field = fftshift(fft2(fftshift(slm_field)))

        x = np.abs(target_field)
        corrs.append(np.corrcoef(x.ravel(), target_amplitude.ravel())[0, 1])
        if i == iterations - 1:
            export_target_field = np.abs(target_field.copy())
        target_field = np.abs(target_amplitude) * np.exp(1j * np.angle(target_field))
        slm_field = fftshift(ifft2(fftshift(target_field)))
        slm_field = ini_amplitude * np.exp(1j * np.angle(slm_field))

    if replace_middle:  # replace middle of export field, there's always a high intensity pixel there from the fft
        export_target_field[int(export_target_field.shape[0] / 2), int(export_target_field.shape[1] / 2)] = 0

    return GSFresult(phase=np.angle(slm_field) + pi, target_fields=[export_target_field], correlations=corrs,
                     algorithm='GSF_2D')


def GS_new(target_amplitude, iterations=30, replace_middle=True):
    # simple GS FFT propagator
    assert target_amplitude.shape[0] == target_amplitude.shape[1]
    ini_amplitude = np.random.rand(*target_amplitude.shape)
    slm_field = ini_amplitude
    holo_phase = np.random.rand(*target_amplitude.shape)

    corrs = []
    for i in range(iterations):
        # target_field = fftshift(fft2(fftshift(slm_field)))
        # x = np.abs(target_field)
        # corrs.append(np.corrcoef(x.ravel(), target_amplitude.ravel())[0, 1])
        # if i == iterations - 1:
        #     export_target_field = np.abs(target_field.copy())
        # target_field = np.abs(target_amplitude) * np.exp(1j * np.angle(target_field))
        # slm_field = fftshift(ifft2(fftshift(target_field)))
        # slm_field = ini_amplitude * np.exp(1j * np.angle(slm_field))


        hologram = ini_amplitude *  np.exp(1j * holo_phase)
        targ_approx = fftshift(fft2(hologram))
        if i == iterations - 1:
            export_target_field = np.abs(targ_approx.copy())
        targ_approx = target_amplitude * np.exp(1j * np.angle(targ_approx))
        holo_approx = ifftshift(ifft2(fftshift(targ_approx)))
        holo_phase = np.angle(holo_approx)

    if replace_middle:  # replace middle of export field, there's always a high intensity pixel there from the fft
        export_target_field[int(export_target_field.shape[0] / 2), int(export_target_field.shape[1] / 2)] = 0

    return GSFresult(phase=np.angle(slm_field) + pi, target_fields=[export_target_field], correlations=corrs,
                     algorithm='GSF_2D')


# def GS_FFT_AA(target_amplitude, iterations=30, replace_middle=True):
#     # simple GS FFT propagator
#     assert target_amplitude.shape[0] == target_amplitude.shape[1]
#     ini_amplitude = np.random.rand(*target_amplitude.shape)
#     slm_field = ini_amplitude
#     a = .6
#
#     norm_ta = np.abs(target_amplitude) / np.abs(target_amplitude).mean()
#     corrs = []
#     for i in range(iterations):
#         target_field = fft2(slm_field)
#
#         x = np.abs(target_field)
#         corrs.append(np.corrcoef(x.ravel(), target_amplitude.ravel())[0, 1])
#         if i == iterations - 1:
#             export_target_field = np.abs(target_field.copy())
#         amp_tf = np.abs(target_field)
#         target_field = (a * norm_ta + (1 - a) * amp_tf / amp_tf.mean()) * np.exp(1j * np.angle(target_field))
#         slm_field = ifft2(target_field)
#         slm_field = ini_amplitude * np.exp(1j * np.angle(slm_field))
#
#     if replace_middle:  # replace middle of export field, there's always a high intensity pixel there from the fft
#         export_target_field[int(export_target_field.shape[0] / 2), int(export_target_field.shape[1] / 2)] = 0
#
#     return GSFresult(phase=np.angle(slm_field) + pi, target_fields=[export_target_field], correlations=corrs,
#                      algorithm='GSF_2D')


def GSF_Zlens(target_amplitude, Z=0, wavelength=960, *args, **kwargs):
    res = GS_FFT(target_amplitude, *args, **kwargs)
    if Z != 0:
        lens1 = lens(res.phase.shape, res.phase.shape[1] / 2, Z, wavelength)
        res = res._replace(phase=(res.phase + lens1) % (2 * pi))
        res = res._replace(lenses=[lens1])
    return res


def GSF_2D_compatibility_wrapper(target_amplitudes, Zs=None, *args, **kwargs):
    """
    Provides a wrapper with the same interface as the 3D GSF, to simplify calling
    :param target_amplitudes: list with a single target_pattern
    :param Zs: list with a single target Z
    """
    if Zs is None:
        Zs = [0]
    assert len(target_amplitudes) == 1 and len(Zs) == 1
    return GSF_Zlens(target_amplitudes[0], Zs[0], *args, **kwargs)


def lens(ini_field_shape, side_length, z, wavelength, focal_length=1, m=1):
    # m = refractive index
    # focal length should come from system, in um
    # wavelength in nm
    # wavelength /= 1000. #to convert to um
    z = z / 10  # previous there was a factor 10 scaling applied for Z calibration, moving that here instead
    dx = side_length / ini_field_shape[1]  # should it be 1 or 0 of the shape?
    x = np.linspace(-side_length / 2, side_length / 2 - dx, ini_field_shape[1])
    xp, yp = np.meshgrid(x, x)

    clens = - (pi * m) / (wavelength * focal_length ** 2) * -z * (xp ** 2 + yp ** 2)  # minus Z to match the system
    return np.fmod(clens, 2 * pi)


def lens_zernicke(ini_field_shape, side_length, z, wavelength):
    n = 1.33  # refractive index
    alpha = 0.743259997  # lens angle, radians
    z = z * 1000  # and convert nm to um
    k = 2 * pi / wavelength
    c4 = n * k * z * sin(alpha) ** 2 / (8 * pi * sqrt(3)) * (
        1 + 1 / 4 * sin(alpha) ** 2 + 9 / 80 * sin(alpha) ** 4 + 1 / 16 * sin(alpha) ** 6)  # defocus
    c11 = n * k * z * sin(alpha) ** 4 / (96 * pi * sqrt(5)) * (
        1 + 3 / 4 * sin(alpha) ** 2 + 15 / 18 * sin(alpha) ** 4)  # 1st spherical
    c22 = n * k * z * sin(alpha) ** 6 / (640 * pi * sqrt(7)) * (1 + 5 / 4 * sin(alpha) ** 2)  # 2nd spherical

    dx = side_length / ini_field_shape[1]  # should it be 1 or 0 of the shape?
    # x = np.linspace(-side_length / 2, side_length / 2 - dx, ini_field_shape[1])
    x = np.linspace(-1, 1, ini_field_shape[1])
    xp, yp = np.meshgrid(x, x)

    r = np.sqrt(xp ** 2 + yp ** 2)
    z4 = np.sqrt(3) * (2 * r ** 2 - 1)
    z11 = np.sqrt(5) * (6 * r ** 4 - 6 * r ** 2 + 1)
    z22 = np.sqrt(7) * (20 * r ** 6 - 30 * r ** 4 + 12 * r ** 2 - 1)

    return np.remainder((c4 * z4 + c11 * z11 + c22 * z22), 1) * 2 * pi
