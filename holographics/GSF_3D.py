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

from __future__ import print_function, division
import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from GSF import lens_zernicke as lens, GSFresult

pi = np.pi


def normedplanes(targets, amps):
    """takes iterable, with pairs of lists"""
    res = [np.dot(np.int64(t.ravel()), a.ravel()) for t, a in zip(targets, amps)]
    tts, aas = zip(*[((t.ravel() ** 2).sum(), (a.ravel() ** 2).sum()) for t, a in zip(targets, amps)])
    denom = sum(tts) ** .5 * sum(aas) ** .5
    # print ("res,denom", res, denom)
    return np.asarray(res) / denom


def GS_3D(target_amplitudes, target_Zs, wavelength=960, iterations=30, replace_middle=True):
    """
    :param target_amplitudes: list of target arrays
    :param target_Zs: list of floats/ints
    :return:
    """

    assert len(target_amplitudes) == len(target_Zs)
    assert len(list(set([t.shape for t in target_amplitudes]))) == 1, "All target amplitudes must be the same shape"
    assert target_amplitudes[0].shape[0] == target_amplitudes[0].shape[1], "Target amplitudes should be square!"

    target_amplitudes = [t**.5 for t in target_amplitudes]

    target_ratios = normedplanes(target_amplitudes, target_amplitudes)
    target_ratios /= target_ratios.sum()
    field_ratios = target_ratios.copy()

    print ("Target ratios: ", target_ratios)
    # ini_amplitudes = [np.random.rand(*target_amplitudes[0].shape) for i in target_amplitudes]
    ini_amplitudes = [np.random.rand(*target_amplitudes[0].shape),] * len(target_amplitudes)
    unified_slm_field = ini_amplitudes[0]

    lenses = [lens(ini_amplitudes[0].shape, ini_amplitudes[0].shape[1] / 2, Z, wavelength) for Z in target_Zs]

    export_target_fields = []
    corrs = []
    for i in range(iterations):
        slm_fields = []
        export_target_fields = []
        for planenum, plane_lens in enumerate(lenses):
            slm_field = ini_amplitudes[planenum] * np.exp(1j * (unified_slm_field - plane_lens))
            target_field = fftshift(fft2((slm_field)))

            export_target_field = np.abs(target_field)**2
            if replace_middle:  # replace middle of export field, there's always a high intensity pixel there from the fft
                export_target_field[
                    int(export_target_field.shape[0] / 2), int(export_target_field.shape[1] / 2)] = 0
            export_target_fields.append(export_target_field)

            target_field = np.abs(target_amplitudes[planenum]) * np.exp(1j * np.angle(target_field))

            slm_field = fftshift(ifft2(fftshift(target_field)))
            slm_field = (np.angle(slm_field) + plane_lens)
            slm_field = ini_amplitudes[planenum] * np.exp(1j * slm_field)
            slm_fields.append(slm_field)

        corrs.append(normedplanes(target_amplitudes, export_target_fields))

        c = corrs[-1]
        c = np.asarray(c) / sum(c)

        if i > 1:
            field_ratios += (target_ratios - c) / 2.
        slm_fields = [s * field_ratios[ii] for ii, s in enumerate(slm_fields)]
        unified_slm_field = np.angle(np.dstack(slm_fields).sum(2)) % (2 * pi)

    nr = normedplanes(target_amplitudes, export_target_fields)
    print ("Targets: ", target_ratios, nr/nr.sum())
    return GSFresult(unified_slm_field, export_target_fields, correlations=corrs, algorithm='GS_3D')
