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
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.widgets import Button


def pickimage(images, labels=None):
    gs = gridspec.GridSpec(2, len(images), wspace=.04, right=.98, left=.02)

    index = [None, ]  # TODO using a class is prettier than a mutable

    def index_func(num):
        def inner(x):
            index[0] = num
            plt.close()

        return inner

    bs = []
    for i, img in enumerate(images):
        plt.subplot(gs[0, i])
        plt.imshow(img, 'gray')
        if labels is not None:
            plt.xlabel('%.3f' % labels[i])

        ax = plt.subplot(gs[1, i])
        b = Button(ax, 'Choose image\n above')
        b.on_clicked(index_func(i))
        bs.append(b)
    plt.show()
    return index[0]

if __name__ == '__main__':
    images = [np.random.rand(256, 256), ] * 5
    print(pickimage(images))

