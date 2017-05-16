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
import pyglet
from pyglet.gl import *
from itertools import chain
import time
import gevent
import cStringIO
from PIL import Image

pattern_time = 1 / 12.0 #When using multiple subpatterns, the time for each pattern


class Frameplayer(pyglet.window.Window):
    def __init__(self, width=800, height=600, fullscreen=True, screen_num=None):
        platform = pyglet.window.get_platform()
        display = platform.get_default_display()

        screen_h = None
        if fullscreen == 'auto':
            # if 800x600 display attached, use that and fullscreen = True
            for i, screen in enumerate(display.get_screens()):
                if screen.width == 800 and screen.height == 600:
                    screen_h = screen
                    fullscreen = True
                    break
            if fullscreen == 'auto': #didn't find a 800x600 screen
                fullscreen = False

        if screen_h is None:
            screen_h = display.get_screens()[-1]

        if fullscreen is True:
            super(Frameplayer, self).__init__(vsync=True, screen=screen_h, fullscreen=True)
        else:
            super(Frameplayer, self).__init__(width=width, height=height, vsync=True, screen=screen_h, fullscreen=fullscreen)
        self.set_caption("Frameplayer")

        pyglet.clock.schedule_interval(self.update, 1 / 60.0)

        self.nframes = None
        self.frames = None
        pyglet.clock.set_fps_limit(60)
        self.fps_display = pyglet.clock.ClockDisplay()

        self.starttime = None
        self.patternnum = 0

    def loadframes(self, frames):
        self.frames = [[self.to_texture(holo) for holo in fr.holograms] for fr in frames]
        self.durations = [fr.duration for fr in frames]
        self.currentframe = 0

    def to_texture(self, framedata):
        temp = cStringIO.StringIO()
        Image.fromarray(framedata.T).save(temp, format='png')
        return pyglet.image.load('.png', file=temp).get_texture()

    def update(self, dt):
        gevent.sleep(.001)

    def on_draw(self):
        pyglet.gl.glClearColor(0, 0, 0, 0)
        self.clear()

        if time.time() - self.starttime >= self.durations[self.currentframe]:
            # print 'new frame', time.time() - self.starttime, self.durations[self.currentframe]
            self.currentframe += 1
            self.starttime = time.time()

            if self.currentframe == len(self.frames):
                pyglet.app.event_loop.has_exit = True
                self.currentframe = 0
                self.patternnum = 0

        self.frames[self.currentframe][int(self.patternnum)].blit(0, 0, 0)

        self.patternnum = (self.patternnum + pattern_time) % len(self.frames[self.currentframe])
        #only relevant when using multiple patterns per frame

    def playframes(self):
        self.starttime = time.time()
        pyglet.app.run()

    def playframes_nonblocking(self):
        self.starttime = time.time()
        return gevent.spawn(pyglet.app.run)

    def playframes_with_callback(self, callback, calltime):
        self.starttime = time.time()

        def dummy(dt):
            callback()

        pyglet.clock.schedule_once(dummy, calltime)
        pyglet.app.run()

