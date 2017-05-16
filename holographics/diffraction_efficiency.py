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

from __future__ import division
import numpy as np
import scipy.special

Si = lambda x: scipy.special.sici(x)[0]
sinc = np.sinc
pi = np.pi
cos = np.cos


def diff3d(x, y, z, xyscale=.006, zscale=.002):
    """
    Smaller number for scale decreases the amount of correction
    """
    return sinc(xyscale * x / 2) ** 2 * sinc(xyscale * y / 2) ** 2 * sinc(zscale * z / 2) ** 2

