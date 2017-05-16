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

from __future__ import print_function
import zmq as zmq
import numpy as np
import time
import holoclient
from frame import Frame
from svg_util import generate_circle_svg, generate_circles_svg


@holoclient.holodec
def test():
    frames = []

    rs = [10, 5, 20, 5, 10, 15]
    xs, ys = np.linspace(-70, 70, 6), [30, ] * 6
    zlevels = np.linspace(-20, 20, 6)
    for x, y, z, r in zip(xs, ys, zlevels, rs):
        frames.append(Frame(svg=str(generate_circles_svg([x], [y], [r])), Zlevel=z, frame_num=0, duration=6.0))
    requests = []
    requests += [holoclient.Generate(frames, 920, correction_factor=.83)]
    requests += [holoclient.Play()]
    return requests


print(test())
