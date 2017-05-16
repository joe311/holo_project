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

from __future__ import division, print_function

import time
from functools import wraps

import zmq as zmq

import holo_msg_pb2
import serializer


class Message(object):
    def __init__(self):
        self.cmd = holo_msg_pb2.StandardCommand()
        self.frames = []

    def send(self, socket):
        msg = serializer.serialize(self.cmd, self.frames)
        socket.send_multipart(msg)
        reply = socket.recv_multipart()
        return serializer.unserialize(reply, holo_msg_pb2.StandardReply())

    def print_msg(self):
        print(self.cmd)


class Status(Message):
    def __init__(self):
        super(Status, self).__init__()
        self.cmd.cmd = holo_msg_pb2.StandardCommand.STATUS


class Play(Message):
    def __init__(self):
        super(Play, self).__init__()
        self.cmd.cmd = holo_msg_pb2.StandardCommand.PLAY


class Generate(Message):
    def __init__(self, frames, wavelength=None, correction_factor=None):
        super(Generate, self).__init__()
        self.cmd.cmd = holo_msg_pb2.StandardCommand.GENERATE
        if wavelength:
            self.cmd.wavelength = wavelength
        if correction_factor:
            self.cmd.correction_factor = correction_factor
        self.frames = [frame.svg for frame in frames]
        for frame in frames:
            frame_meta = self.cmd.image_meta.add()
            frame_meta.Zlevel = frame.Zlevel
            frame_meta.frame_num = frame.frame_num
            frame_meta.duration = frame.duration


class Calibrate_Background(Message):
    def __init__(self):
        super(Calibrate_Background, self).__init__()
        self.cmd.cmd = holo_msg_pb2.StandardCommand.CALIBRATE_BACKGROUND


class Calibrate_Circle(Message):
    def __init__(self, position_x, position_y):
        super(Calibrate_Circle, self).__init__()
        self.cmd.cmd = holo_msg_pb2.StandardCommand.CALIBRATE_CIRCLE
        self.cmd.calibration_circle_x = position_x
        self.cmd.calibration_circle_y = position_y


class Calibrate_Run(Message):
    def __init__(self):
        super(Calibrate_Run, self).__init__()
        self.cmd.cmd = holo_msg_pb2.StandardCommand.CALIBRATE_RUN


def holodec(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print("Connecting to holo server")
        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        socket.connect("tcp://localhost:51233")

        t1 = time.time()

        requests = func(*args, **kwargs)

        if len(requests) == 0:
            raise NotImplemented("Don't return requests yet!")

        for request in requests:
            print("Sending request: ", end="")
            request.print_msg()
            msg = request.send(socket)
            print("Received reply: ", msg[0])
        print("done in: ", time.time() - t1)

    return wrapper

