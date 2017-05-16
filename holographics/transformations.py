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
from scipy import optimize


def applytrans(mat, trans):
    """
    :param mat: 2xN np array (x as column 1, y as column 2)
    :param trans: 3x3 transformation matrix
    """
    amat = np.hstack((mat, np.ones((mat.shape[0], 1))))
    return np.dot(trans, amat.T)[:2, :].T


def pad_trans(trans):
    return np.vstack((trans, [0, 0, 1]))


def reprojection_error(data1, data2, trans):
    return rms(applytrans(data1, pad_trans(trans)), data2)


def fit_trans(data1, data2):
    def do_trans(trans):
        trans = trans.reshape((2, 3))
        return reprojection_error(data1, data2, trans)

    res = optimize.minimize(do_trans, np.identity(3)[:2, :])
    return res.x.reshape((2, 3)), do_trans(res.x)


def rms(data1, data2):
    return ((np.asarray(data1) - np.asarray(data2)) ** 2).mean() ** .5
