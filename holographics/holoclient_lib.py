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
from holoclient import *


class Holoclient(object):
    def __init__(self, address="tcp://localhost:51233"):
        print("Connecting to holo server at %s" % address)
        context = zmq.Context()
        self.socket = context.socket(zmq.REQ)
        self.socket.connect(address)

    def status(self):
        return Status().send(self.socket)

    def play(self):
        return Play().send(self.socket)

    def generate(self, *args, **kwargs):
        return Generate(*args, **kwargs).send(self.socket)
