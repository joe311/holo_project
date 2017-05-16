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

from frame import Frame


def serialize(cmd, frameimages=None):
    """ Takes in a ProtoBuf cmd, and a list of svg images, and returns a list
    that can be fed into ZMQ send_multipart
    """
    msgframes = [cmd.SerializeToString()]

    if frameimages:
        for img in frameimages:
            msgframes += [img]
    return msgframes


def unserialize(msg, cmdtype):
    """Reverse of the serialization process, takes in a multipart msg from ZMQ, and
    a ProtoBuf cmdtype, and returns the cmd and a list of frames 
    """
    cmdtype.ParseFromString(msg.pop(0))

    framelist = []
    for image_meta in cmdtype.image_meta:
        frame = Frame(Zlevel=image_meta.Zlevel, frame_num=image_meta.frame_num, duration=image_meta.duration)
        framelist += [frame]

    assert len(framelist) == len(msg)

    for i, frame in enumerate(msg):
        framelist[i].svg = frame

    return cmdtype, framelist
