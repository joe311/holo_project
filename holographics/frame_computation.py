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

import math
import os

import numpy as np
import scipy.ndimage
from holographics import svg_util
from holographics.GSF_3D import GS_3D as GS
from holographics.frame import Frame, target_size, compute_size, svg_target_size_x, svg_target_size_y
from joblib import Memory
from scipy.misc import imresize
from diffraction_efficiency import diff3d as diffractioneff3d

cachedir = './gsf_cache'
memory = Memory(cachedir=cachedir, verbose=0)

if not os.path.exists(cachedir):
    os.mkdir(cachedir)
memory.clear()


def computehologram(frames, wavelength, *args, **kwargs):
    target_amplitudes, Zs = zip(*[[f.raster, f.Zlevel] for f in frames])

    # drop blank frames, if all are blank, use a large blank circle
    non_blank = [(t, Z) for t, Z in zip(target_amplitudes, Zs) if t.sum() != 0]
    if non_blank:
        target_amplitudes, Zs = zip(*non_blank)
    else:
        print('Frames blank, showing default large circle')
        f = Frame(svg=svg_util.generate_circle_svg(0, 0, 80))
        f.rasterize()
        target_amplitudes = [f.raster]
        Zs = [0]

    res = GS(target_amplitudes, Zs, wavelength, *args, **kwargs)

    hologram = (res.phase % (2 * math.pi)) * 255 / (2 * math.pi)

    # resize to match the SLM size
    resized = imresize(hologram, (max(target_size),) * 2)
    resized = resized[:, resized.shape[1] / 2 - target_size[1] / 2:resized.shape[1] / 2 + target_size[1] / 2].astype(
        'uint8')
    return (resized)


def computemultipatternhologram(frames, wavelength, *args, **kwargs):
    holo = memory.cache(computehologram)(frames, wavelength, *args, **kwargs)
    return [holo]


def simple_generate_frames(frames):
    """
    Only for testing purposes
    """
    for f in frames:
        f.holograms = computemultipatternhologram([f.raster], [f.Zlevel], npatterns=1)


def frame_diffraction_effs(frames):
    """
    Correct frames to compensate for diffraction efficiency
    """
    lx = np.linspace(-svg_target_size_x / 2, svg_target_size_x / 2, compute_size[0])
    ly = np.linspace(-svg_target_size_y / 2, svg_target_size_y / 2, compute_size[1])
    x, y = np.meshgrid(lx, ly)

    for f in frames:
        diff = diffractioneff3d(x, y, f.Zlevel)
        f.raster = f.raster / diff
    fmax = max([f.raster.max() for f in frames])
    for f in frames:
        f.raster = f.raster * 255 / fmax
        f.raster = f.raster.astype(np.uint8)
