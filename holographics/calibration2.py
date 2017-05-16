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
import os.path
from collections import namedtuple
import pickle
import warnings

import numpy as np
import cv2
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.ndimage
import scipy.stats

import svg_util
from frame import Frame
from pickimages import pickimage
import transformations


def pad_zeroes(array):
    return np.c_[array, np.zeros((array.shape[0], 1), 'float32')]


Zscalefactor = 1 #Default scaling for Z before calibration is performed

Calib = namedtuple('Calib', ['rms', 'camera_matrix', 'dist_coefs', 'rvecs', 'tvecs'])


class CameraHandle(object):
    """
    Holds handle to the camera, uses openCV
    """
    def __init__(self, video_src=0):
        self.video_src = video_src
        self.cam = None

    def grab_image(self):
        """
        Grabs an image.
        Connects to the camera is necessary
        """
        if not self.cam or not self.cam.isOpened():
            self.start_cam()

        self.cam.read()  # seems like sometimes some buffer is stuck with an old image, throw away a grab to clear
        ret, frame = self.cam.read()
        if ret:
            return frame[:, :, 0]
        else:
            raise Exception("Couldn't read from camera properly")

    def grab_image_quick(self):
        """
        Grabs an image.
        Doesn't open camera handle, and doesn't double grab to prevent the buffering issue
        """
        ret, frame = self.cam.read()
        if ret:
            return frame[:, :, 0]
        else:
            raise Exception("Couldn't read from camera properly")

    def start_cam(self):
        self.cam = cv2.VideoCapture(self.video_src)
        if self.cam is None or not self.cam.isOpened():
            raise Exception("Couldn't open camera for calibration!")
        self.cam.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 1280)
        self.cam.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 1024)

    def release_cam(self):
        self.cam.release()


class Calibrator(object):
    def __init__(self, camerahandle, holobase):
        self.background = None
        self.camerahandle = camerahandle
        self.holobase = holobase
        self.holo_image = None  # for use with callback during playing

    def grab_point(self, x, y, z, size=20, correction_factor=None):
        return self.grab_svg(svg_util.generate_circle_svg(x, y, size), z, correction_factor)

    def grab_svg(self, svg, z, correction_factor=None):
        frame = Frame(svg=svg, Zlevel=z, duration=1, frame_num=0)
        frame.set_svg_bounds()
        frame.rasterize()
        self.holobase.generate_frames([frame], npatterns=1, correction_factor=correction_factor)
        self.holobase.frameplayer.playframes_with_callback(self.grab_holo_image, 0.4)
        return self.holo_image

    def findcenter(self, img, blurRadius=211):
        assert blurRadius % 2 == 1, "blurRadius must be odd!"

        img = img.copy()
        if self.background is not None:
            img = img.astype('int32')
            img -= self.background
            img = np.clip(img, 0, 2 ** 8)
            img = img.astype('uint8')

        img = cv2.GaussianBlur(img, (blurRadius,) * 2, 0, 0)
        centroid_int = np.unravel_index(img.argmax(), img.shape)
        centroid_int = (centroid_int[1], centroid_int[0])

        return centroid_int

    def findcenter2(self, img):
        circ = cv2.HoughCircles(img, method=cv2.cv.CV_HOUGH_GRADIENT, dp=1.2, minDist=100, minRadius=1, maxRadius=200,
                                param2=10)
        return circ[0, 0, 0:2]

    def grab_background(self):
        self.background = self.grab_image()

    def grab_image(self):
        return self.camerahandle.grab_image()

    def grab_holo_image(self):
        self.holo_image = self.grab_image()
        self.image_ok(self.holo_image)

    def load(self):
        assert self.filepath is not None
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, 'rb') as f:
                    return pickle.load(f)
            except:
                print("COULDN'T LOAD CALIBRATION! - SHV opened but failed")
                return None
        else:
            print("COULDN'T LOAD CALIBRATION! ", self.filepath)
            return None

    def save(self, data):
        with open(self.filepath, 'wb') as f:
            pickle.dump(data, f)

    def image_ok(self, img):
        assert img.dtype == np.dtype('uint8')
        overexposure = (img > 252).mean()
        underexposure = (img < 15).mean()

        print("Image over: %.3f and under %.3f exposure" % (overexposure, underexposure))
        return True


class CorrectionFactorCalibrator(Calibrator):
    def __init__(self, *args, **kwargs):
        self.correction_factor_range = .25
        self.correction_factor_middle = .75
        self.nspots = 10
        self.max_iterations = 1
        super(CorrectionFactorCalibrator, self).__init__(*args, **kwargs)

    def calibrate(self):
        correction_factor_middle = self.correction_factor_middle
        correction_factor_range = self.correction_factor_range
        for i in range(self.max_iterations):
            correction_factors = np.linspace(correction_factor_middle - correction_factor_range / 2.,
                                             correction_factor_middle + correction_factor_range / 2, self.nspots)
            factor_images = [self.grab_point(30, 30, 0, 24, factor) for factor in correction_factors]
            factor_images = [scipy.ndimage.interpolation.zoom(img, .5) for img in
                             factor_images]  # don't need full size images

            index = pickimage(factor_images, correction_factors)
            return correction_factors[index]


class XYCalibrator(Calibrator):
    def __init__(self, *args, **kwargs):
        self.filepath = 'holoXYcal.pkl'
        self.circle_images = []
        self.circle_positions = []
        super(XYCalibrator, self).__init__(*args, **kwargs)
        self.transformation_matrix = self.load()

    def grab_circle_image(self, position):
        self.circle_images.append(self.grab_image())
        self.circle_positions.append(position)

    def show_imgs(self, imgs, centers=None, title=None):
        gs = gridspec.GridSpec(2, 2)
        for i, img in enumerate(imgs):
            plt.subplot(gs[i])
            plt.imshow(img, 'gray')
            if centers is not None:
                plt.scatter(*centers[i], c='r', s=10)
        if title is not None:
            plt.title(title)
        plt.show()

    def run(self):
        circle_centers = [self.findcenter(img, 155) for img in self.circle_images]
        holo_imgs = [self.grab_point(pos[0], pos[1], 0) for pos in self.circle_positions]

        holo_centers = [self.findcenter(img, 431) for img in holo_imgs]

        calib1, err1 = transformations.fit_trans(np.asarray(self.circle_positions), np.asarray(circle_centers))
        calib2, err2 = transformations.fit_trans(np.asarray(holo_centers), np.asarray(self.circle_positions))

        print("Reprojection errors: ", err1, err2)
        calib = np.dot(transformations.pad_trans(calib2), transformations.pad_trans(calib1))

        cal_svgs = [self.apply(svg_util.generate_circle_svg(pos[0], pos[1], 20), calib) for pos in self.circle_positions]
        calibrated_holo_imgs = [self.grab_svg(svg, 0) for svg in cal_svgs]
        calibrated_holo_centers = [self.findcenter(img, 431) for img in calibrated_holo_imgs]

        print("Target circle centers", circle_centers)
        print("Uncalibrated circle centers", holo_centers)
        print("Calibrated circle centers", calibrated_holo_centers)

        um_per_pix = (np.abs(1 / calib1[0, 0]) + np.abs(1 / calib1[1, 1])) / 2.
        print('scaling um/pix %.4f avg, %.4f %.4f' % (
            um_per_pix, 1 / calib1[0, 0], 1 / calib1[1, 1]))
        rms_err_string = "RMS point mismatch %.3f (um)" % (
            transformations.rms(calibrated_holo_centers, circle_centers) * um_per_pix)
        print(rms_err_string)

        if 1:  # plot
            gs = gridspec.GridSpec(2, 2, top=.92, hspace=.1, wspace=.06, left=.02, right=.98)
            for i, pos in enumerate(self.circle_positions):
                plt.subplot(gs[i])
                plt.imshow(np.dstack(
                    [img / float(img.max()) for img in (self.circle_images[i], holo_imgs[i], calibrated_holo_imgs[i])]))
                plt.xlim([0, self.circle_images[i].shape[1]])
                plt.ylim([0, self.circle_images[i].shape[0]])

                for color, center in zip(('r', 'g', 'b'),
                                         (circle_centers[i], holo_centers[i], calibrated_holo_centers[i])):
                    plt.scatter(*center, c=color, s=100, alpha=1)

                plt.xlabel('Distance of imaging to calibrated %.2f (um)' % (
                    um_per_pix * transformations.rms(circle_centers[i], calibrated_holo_centers[i])))

            plt.figtext(.35, .95, 'Imaging points', color='red', ha='center')
            plt.figtext(.5, .95, 'Uncalibrated holo', color='green', ha='center')
            plt.figtext(.65, .95, 'Calibrated holo', color='blue', ha='center')
            plt.figtext(.5, .08, rms_err_string, ha='center')
            plt.suptitle('XY calibration report - close to continue')
            plt.show()  # eventually save as pdf? with datetimename

        self.transformation_matrix = calib
        self.save(self.transformation_matrix)

    def apply(self, svg, transform=None):
        if transform is not None:
            return svg_util.insertTransform(svg, transform)
        else:
            if self.transformation_matrix is None:
                print("Can't apply XYCalibration, not calibrated yet!")
                return svg
            return svg_util.insertTransform(svg, self.transformation_matrix)


class ZCalibrator(Calibrator):
    def __init__(self, *args, **kwargs):
        self.filepath = 'holoZcal.pkl'
        self.Zlevels = []
        self.Zobjs = []
        super(ZCalibrator, self).__init__(*args, **kwargs)
        self.coefs = self.load()
        self.minz, self.maxz = 0, 0

    def calibrateZ(self, Zlevel):
        """
        Plays a point at the given Z level, the user then focuses on the pattern, and the imaging software should send
        a message with the objective Z level.
        This should be called at multiple different Z levels
        """

        npoints = 7
        l = np.linspace(0, np.pi, npoints)
        xl = 20*np.cos(l)
        yl = 20*np.sin(l)
        rl = (5,)*npoints
        zsvg = svg_util.generate_circles_svg(xl, yl, rl)
        self.grab_svg(zsvg, Zlevel)

        key = -1
        while key == -1:
            winname = 'Press any key when in focus'
            cv2.imshow(winname, self.camerahandle.grab_image_quick())
            key = cv2.waitKey(20)
        cv2.destroyWindow(winname)

        self.Zlevels.append(Zlevel / Zscalefactor)

    def setobjectiveZlevel(self, Zobj):
        """Stores the Zlevel of the objective, to be called right after calibrate Z"""
        self.Zobjs.append(Zobj)

    def run(self):
        # have Zlevels which is lens positions
        # and Zobjs which is true objective positions

        Zlevels = np.asarray(self.Zlevels)
        Zobjs = np.asarray(self.Zobjs)
        coefs = np.polyfit(Zobjs, Zlevels, deg=5)
        zmin = Zobjs.min() - .1 * (Zobjs.max() - Zobjs.min())
        zmax = Zobjs.max() + .1 * (Zobjs.max() - Zobjs.min())
        x = np.linspace(zmin, zmax, 500)
        plt.scatter(Zobjs, Zlevels)
        plt.plot(x, np.polyval(coefs, x))
        plt.xlabel("Z obj")
        plt.ylabel("Z level")
        plt.xlim([zmin, zmax])
        plt.ylim([zmin, zmax])
        plt.gca().set_aspect('equal')
        plt.plot(x, x, c='k')

        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(Zobjs, Zlevels)
        print("linear fit r-value: {0}".format(r_value))
        plt.plot(x, slope * x + intercept, 'g-')

        plt.show()

        self.coefs = coefs
        self.save(self.coefs)

        self.minz = min(Zobjs)
        self.maxz = max(Zobjs)

    def apply(self, frame):
        if self.coefs is not None:
            if self.maxz < frame.Zlevel < self.minz:
                warnings.warn("Trying to play a pattern outside of the Z calibration range")
            frame.Zlevel = np.polyval(self.coefs, frame.Zlevel)
        return frame
