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
import os
import re
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy import misc


class SLM_correction(object):
    def __init__(self, deformation_corrections_path='../static/deformation_correction_pattern',
                 LUT_path='../static/SLM_LUT/LUT_info.ini'):
        self.load_deformation_corrections(deformation_corrections_path)
        self.load_LUT_corrections(LUT_path)

    def load_deformation_corrections(self, deformation_corrections_path):
        bmps = [f for f in os.listdir(deformation_corrections_path) if f.endswith('.bmp')]
        self.deformation_wavelengths = [int(re.search('\d{3,4}(?=nm\.bmp)', f).group()) for f in bmps]

        self.deformation_corrections = dict(zip(self.deformation_wavelengths, [
            misc.imread(os.path.join(deformation_corrections_path, f), flatten=True).T for f in bmps]))

    def load_LUT_corrections(self, LUT_path):
        with open(LUT_path, 'rU') as LUT_file:
            start = re.compile('\[WaveLength\]')
            end = re.compile('\[Temperature\]')

            wavelengths = []
            corrections = []
            started, ended = False, False
            for line in LUT_file.readlines():
                if started:
                    ended = end.match(line)
                    if ended:
                        break
                    wavedata = re.match('\s*\d+=\d+,\s*\d+.\d+e\+\d+', line)  # not tdatas, not whitespace
                    if wavedata:
                        wavelengths.append(int(re.search('(?<==)\d+(?=,)', wavedata.group(0)).group(0)))
                        corrections.append(float(re.search('(?<=,)\s*\d+.\d+e\+\d+', wavedata.group(0)).group(0)))
                if not started:
                    started = start.match(line)
        wavelengths.reverse()
        corrections.reverse()
        self.wavelength_LUT = InterpolatedUnivariateSpline(wavelengths, corrections, k=2)

    def get_deformation_pattern(self, target_wavelength):
        wavelength = min(self.deformation_wavelengths, key=lambda x: abs(target_wavelength - x))
        # print "Nearest wavelength is %d" % wavelength
        return self.deformation_corrections[wavelength]

    def apply_deformation_pattern(self, hologram, wavelength):
        deformation_pattern = self.get_deformation_pattern(wavelength).astype('uint8')
        assert hologram.shape == deformation_pattern.shape
        hologram = hologram.astype('uint16')
        hologram += self.get_deformation_pattern(wavelength).astype('uint16')
        res = hologram % 256

        return res.astype('uint8')

    def get_LUT_correction(self, wavelength):
        return 255. / (293. * self.wavelength_LUT(wavelength))
        # 255 is there since its 8 bit, the 293 comes from LUT file
        # (which matches the numbers given in the inspection sheet from hamamatsu for input signal)

    def apply_LUT(self, hologram, wavelength, correction_factor=.8):  # SLM table is wrong!
        # print "wavelength %d correction val %f" % (wavelength, self.get_LUT_correction(wavelength))
        # return (hologram * self.get_LUT_correction(wavelength)*correction_factor).astype('uint8')
        return ((hologram * correction_factor) % 255).astype('uint8')

