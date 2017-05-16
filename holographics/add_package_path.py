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

import sys
import os


if not os.path.abspath("..") in sys.path:
    sys.path.append(os.path.abspath(".."))

if __name__ == "__main__" and __package__ is None:
    __package__ = "holographics"
