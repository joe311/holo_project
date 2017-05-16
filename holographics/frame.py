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

import copy

import numpy as np
from PIL import Image

import svg_util
from svg_util import add_background, svg_to_np, set_svg_bounds

compute_size = (792, 792)

target_size = (792, 600)
svg_target_size_x = 400  # um
svg_target_size_y = int(svg_target_size_x * float(compute_size[0]) / compute_size[1])


class Frame(object):
    def __init__(self, svg=None, raster=None, holograms=None, Zlevel=0, frame_num=None, duration=0):
        self.svg = svg
        self.raster = raster
        self.Zlevel = Zlevel
        self.frame_num = frame_num
        self.duration = duration
        self.holograms = holograms
        self.computedpattern = None

    def rasterize(self):
        assert self.svg
        self.set_svg_bounds()
        dpi = (72 * svg_target_size_x / float(compute_size[0]))
        raster = svg_to_np(self.svg, dpi)
        assert np.diff(raster[:, :, :3]).sum() == 0, 'All svg color channels should be the same'
        self.raster = raster[:, :, 0]

    def apply_deformation_correction(self, SLM_correction, *args, **kwargs):
        self.holograms = [SLM_correction.apply_deformation_pattern(holo, *args, **kwargs) for holo in self.holograms]

    def apply_LUT_correction(self, SLM_correction, *args, **kwargs):
        self.holograms = [SLM_correction.apply_LUT(holo, *args, **kwargs) for holo in self.holograms]

    def apply_factor_correction(self, factor):
        self.holograms = [(holo * factor).astype('uint8') for holo in self.holograms]

    @classmethod
    def from_svg_path(self, svgpath, *args, **kwargs):
        with open(svgpath) as f:
            svg = f.read()
        return self(svg, *args, **kwargs)

    @classmethod
    def from_raster_path(self, rasterpath, *args, **kwargs):
        raster = np.asarray(Image.open(rasterpath))
        return self(raster=raster, *args, **kwargs)

    @classmethod
    def blankframe(cls, *args, **kwargs):
        svg = svg_util.empty_svg()
        raster = np.zeros(target_size)
        return cls(svg=svg, raster=raster, *args, **kwargs)

    def copy(self):
        return copy.deepcopy(self)

    def set_svg_bounds(self):
        self.svg = add_background(
            set_svg_bounds(self.svg, -svg_target_size_x / 2, -svg_target_size_y / 2, svg_target_size_x,
                           svg_target_size_y))
