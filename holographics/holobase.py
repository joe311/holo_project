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

import ConfigParser
import os.path
import time

import numpy as np
import zmq.green as zmq
from joblib import Parallel, delayed
from scipy.misc import imsave

import holo_msg_pb2
import serializer
from SLM_correction import SLM_correction
from calibration2 import CorrectionFactorCalibrator, XYCalibrator, ZCalibrator, CameraHandle
from frame import Frame
from holographics.frame_computation import computemultipatternhologram, frame_diffraction_effs
from playframes import Frameplayer


def checkframes(frames):
    frame_nums = np.asarray([f.frame_num for f in frames])
    diffs = np.diff(np.asarray(frame_nums))
    assert all((diffs == 0) | (diffs == 1))
    # framenumbers should be adjacent and increasing

    frame_idx_list = [np.where(frame_nums == fn)[0].tolist() for fn in set(frame_nums)]  # list of lists by frame_num

    for frame_idxs in frame_idx_list:
        durations = [frames[frame_idx].duration for frame_idx in frame_idxs]
        assert len(set(durations)) == 1
        # durations should all be the same

        Zs = [frames[frame_idx].Zlevel for frame_idx in frame_idxs]
        assert len(set(Zs)) == len(Zs)
        # Zs should all be different (only one SVG per frame)


class Holobase(object):
    def __init__(self):
        self.config = ConfigParser.ConfigParser()
        try:
            self.config.read('holo_config.cfg')
        except ConfigParser.Error as e:
            raise Exception('Failed to read config file!')

        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        address = "tcp://*:" + self.config.get('IP', 'ZMQport')
        print("binding to: ", address)
        self.socket.bind(address)

        self.frameplayer = Frameplayer(fullscreen='auto')
        self.SLM_correction = SLM_correction()

        self.CameraHandle = CameraHandle()
        self.CorrectionFactorCalibrator = CorrectionFactorCalibrator(self.CameraHandle, self)
        self.XYCalibrator = XYCalibrator(self.CameraHandle, self)
        self.ZCalibrator = ZCalibrator(self.CameraHandle, self)

        self.wavelength = float(self.config.get('holo', 'wavelength'))
        self.correction_factor = float(self.config.get('holo', 'correction_factor'))

        self.pre_frames = None
        self.postgsf_frames = None

        self.run()

    def generate_frames(self, frames, npatterns=1, correction_factor=None):
        frame_nums = np.asarray([f.frame_num for f in frames])
        frame_idxss = [np.where(frame_nums == fn)[0].tolist() for fn in set(frame_nums)]  # list of lists by frame_num

        durations = [frames[frame_idx[0]].duration for frame_idx in frame_idxss]

        holos = Parallel()(
            [delayed(computemultipatternhologram)([frames[fi] for fi in frame_idxs], self.wavelength) for frame_idxs in frame_idxss])

        # holos = Parallel()(
        #     [delayed(computemultipatternhologram)(t, Z, self.wavelength, npatterns) for t, Z in zip(targets, Zs)])
        postgsf_frames = [Frame(holograms=fs, duration=d, frame_num=i) for i, (fs, d) in
                          enumerate(zip(holos, durations))]

        [frame.apply_deformation_correction(self.SLM_correction, self.wavelength) for frame in postgsf_frames]
        if correction_factor is not None:
            [frame.apply_factor_correction(correction_factor) for frame in postgsf_frames]
        else:
            [frame.apply_factor_correction(self.correction_factor) for frame in postgsf_frames]
        self.frameplayer.loadframes(postgsf_frames)
        self.postgsf_frames = postgsf_frames

    def run(self):
        print("Holobase running...")
        while True:
            try:
                msg = self.socket.recv_multipart()
                try:
                    msg, frames = serializer.unserialize(msg, holo_msg_pb2.StandardCommand())

                except AssertionError:
                    replymsg = holo_msg_pb2.StandardReply()
                    replymsg.reply = holo_msg_pb2.StandardReply.ERROR
                    replymsg.error = holo_msg_pb2.StandardReply.BAD_REQUEST
                    replymsg.error_message = "Error, number of image_metas in command and number of holographic frames need to match!"

                else:
                    print(msg, ' #frames attached:', len(frames))
                    print('')

                    if msg.cmd == holo_msg_pb2.StandardCommand.STATUS:
                        replymsg = holo_msg_pb2.StandardReply()
                        replymsg.reply = holo_msg_pb2.StandardReply.OK

                    elif msg.cmd == holo_msg_pb2.StandardCommand.GENERATE:
                        if msg.wavelength:
                            if msg.wavelength != self.wavelength:
                                self.wavelength = msg.wavelength
                                print("Setting wavelength to %d" % self.wavelength)
                        if msg.correction_factor:
                            if msg.correction_factor != self.correction_factor:
                                self.correction_factor = msg.correction_factor
                                print("Setting correction factor to %.3f" % self.correction_factor)

                        if len(frames) == 0:
                            replymsg = holo_msg_pb2.StandardReply()
                            replymsg.reply = holo_msg_pb2.StandardReply.ERROR
                            replymsg.error = holo_msg_pb2.StandardReply.BAD_REQUEST
                            replymsg.error_message = "Error, need to send frames to generate!"
                        else:
                            try:
                                checkframes(frames)
                            except AssertionError:
                                replymsg = holo_msg_pb2.StandardReply()
                                replymsg.reply = holo_msg_pb2.StandardReply.ERROR
                                replymsg.error = holo_msg_pb2.StandardReply.BAD_REQUEST
                                replymsg.error_message = "Error, frames incorrectly specified!"

                            self.pre_frames = [frame.copy() for frame in frames]
                            for frame in frames:
                                # print ("frame before calib and bounding", frame.svg)
                                frame.svg = self.XYCalibrator.apply(frame.svg)
                                self.ZCalibrator.apply(frame)
                                frame.rasterize()
                                print("frame after calib and bounding", frame.svg)

                            frame_diffraction_effs(frames)
                            self.generate_frames(frames)
                            replymsg = holo_msg_pb2.StandardReply()
                            replymsg.reply = holo_msg_pb2.StandardReply.OK

                    elif msg.cmd == holo_msg_pb2.StandardCommand.PLAY:
                        if self.postgsf_frames is None:
                            replymsg = holo_msg_pb2.StandardReply()
                            replymsg.reply = holo_msg_pb2.StandardReply.ERROR
                            replymsg.error = holo_msg_pb2.StandardReply.BAD_REQUEST
                            replymsg.error_message = "Error, haven't sent any frames yet!"

                        else:
                            timestamp = time.localtime()
                            self.frameplayer.playframes()
                            print("Done playing frames")
                            print("")

                            directory = './_raw_svgs'
                            if not os.path.exists(directory):
                                os.mkdir(directory)
                            for i, frame in enumerate(self.pre_frames):
                                filename = time.strftime("%Y_%m_%d__%H-%M-%S_frame", timestamp) + str(
                                    i) + "_%d" % frame.frame_num + '.svg'
                                if os.path.exists(directory) and frame.svg:
                                    filepath = os.path.join(directory, filename)
                                    with open(filepath, 'wb') as f:
                                        f.write(frame.svg)
                                else:
                                    print("error saving svg log data!!!!!!!!!!!!!!!!!!!")

                            # directory = './_rasters'
                            # if not os.path.exists(directory):
                            #     os.mkdir(directory)
                            # for i, frame in enumerate(self.postgsf_frames):
                            #     filename = time.strftime("%Y_%m_%d__%H-%M-%S_frame", timestamp) + str(
                            #         i) + "_%d" % frame.frame_num + '.'
                            #     if os.path.exists(directory) and frame.raster:
                            #         imsave(os.path.join(directory, filename), frame.raster)
                            #     else:
                            #         print ("error saving raster log data!!!!!!!!!!!!!!!!!!!")

                            directory = './_raw_postgsfs'
                            if not os.path.exists(directory):
                                os.mkdir(directory)
                            for i, frame in enumerate(self.postgsf_frames):
                                if os.path.exists(directory) and frame.holograms is not None:
                                    for j, holoframe in enumerate(frame.holograms):
                                        filename = time.strftime("%Y_%m_%d__%H-%M-%S_frame", timestamp) + str(i) + str(
                                            j) + "_%.2f" % frame.frame_num + '.bmp'
                                        filepath = os.path.join(directory, filename)
                                        imsave(filepath, holoframe.T)
                                else:
                                    print("error saving postgsf log data!!!!!!!!!!!!!!!!!!!")

                            replymsg = holo_msg_pb2.StandardReply()
                            replymsg.reply = holo_msg_pb2.StandardReply.OK

                    elif msg.cmd == holo_msg_pb2.StandardCommand.CALIBRATE_BACKGROUND:
                        self.XYCalibrator.grab_background()
                        replymsg = holo_msg_pb2.StandardReply()
                        replymsg.reply = holo_msg_pb2.StandardReply.OK

                    elif msg.cmd == holo_msg_pb2.StandardCommand.CALIBRATE_CIRCLE:
                        self.XYCalibrator.grab_circle_image((msg.calibration_circle_x, msg.calibration_circle_y))
                        replymsg = holo_msg_pb2.StandardReply()
                        replymsg.reply = holo_msg_pb2.StandardReply.OK

                    elif msg.cmd == holo_msg_pb2.StandardCommand.CALIBRATE_RUN:
                        self.XYCalibrator.run()
                        replymsg = holo_msg_pb2.StandardReply()
                        replymsg.reply = holo_msg_pb2.StandardReply.OK

                    elif msg.cmd == holo_msg_pb2.StandardCommand.CALIBRATE_CORRECTION_FACTOR:
                        self.correction_factor = self.CorrectionFactorCalibrator.calibrate()
                        print("Correction factor calibrated to: ", self.correction_factor)
                        replymsg = holo_msg_pb2.StandardReply()
                        replymsg.reply = holo_msg_pb2.StandardReply.OK

                    elif msg.cmd == holo_msg_pb2.StandardCommand.CALIBRATE_Z:
                        self.ZCalibrator.calibrateZ(msg.calibration_Z_level)
                        replymsg = holo_msg_pb2.StandardReply()
                        replymsg.reply = holo_msg_pb2.StandardReply.OK

                    elif msg.cmd == holo_msg_pb2.StandardCommand.CALIBRATE_Z_OBJ:
                        if msg.objectiveZlevel is None:
                            replymsg = holo_msg_pb2.StandardReply()
                            replymsg.reply = holo_msg_pb2.StandardReply.ERROR
                            replymsg.error = holo_msg_pb2.StandardReply.BAD_REQUEST
                            replymsg.error_message = "Error, need to provide the objective Z level"
                        else:
                            self.ZCalibrator.setobjectiveZlevel(msg.objectiveZlevel)
                            replymsg = holo_msg_pb2.StandardReply()
                            replymsg.reply = holo_msg_pb2.StandardReply.OK

                    elif msg.cmd == holo_msg_pb2.StandardCommand.CALIBRATE_Z_RUN:
                        self.ZCalibrator.run()
                        replymsg = holo_msg_pb2.StandardReply()
                        replymsg.reply = holo_msg_pb2.StandardReply.OK

                    elif msg.cmd == holo_msg_pb2.StandardCommand.CALIBRATE_RELEASE:
                        self.CameraHandle.release_cam()
                        replymsg = holo_msg_pb2.StandardReply()
                        replymsg.reply = holo_msg_pb2.StandardReply.OK

                    else:
                        replymsg = holo_msg_pb2.StandardReply()
                        replymsg.reply = holo_msg_pb2.StandardReply.ERROR
                        replymsg.error = holo_msg_pb2.StandardReply.BAD_REQUEST
                        replymsg.error_message = "Received an unknown message type!"

                self.socket.send_multipart(serializer.serialize(replymsg))

            except Exception as e:
                replymsg = holo_msg_pb2.StandardReply()
                replymsg.reply = holo_msg_pb2.StandardReply.ERROR
                replymsg.error = holo_msg_pb2.StandardReply.SOFTWARE
                replymsg.error_message = "Unhandled error inside our software - quitting now!"
                self.socket.send_multipart(serializer.serialize(replymsg))
                time.sleep(1)
                raise

    def quit(self):
        self.context.term()


if __name__ == '__main__':
    holo = Holobase()
