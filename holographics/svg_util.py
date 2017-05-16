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
from svgfig import SVG, canvas
from xml.etree import ElementTree as ET
import cStringIO
import numpy as np
import cairosvg
from svgfig import load_stream
from scipy.misc import imread

ET.register_namespace('', "http://www.w3.org/2000/svg")


def insertTransform(svg, transform_matrix):
    c = cStringIO.StringIO()
    c.write(svg)
    c.seek(0)
    p = ET.parse(c)
    root = p.getroot()

    newroot = root.copy()
    elems = [elem for elem in newroot]
    for elem in elems:
        newroot.remove(elem)

    matrix_string = "matrix(%f, %f, %f, %f, %f, %f)" % tuple(transform_matrix[:2, :].ravel()[[0, 3, 1, 4, 2, 5]])
    g = ET.SubElement(newroot, 'g', transform=matrix_string)

    [g.append(elem) for elem in elems]
    return ET.tostring(newroot)


def generate_circle_svg(x=60, y=60, r=30, width=792, height=600, colorhex="ff"):
    """
    x: x coord of circle center
    y: y coord of circle center
    r: circle radius
    """
    c = SVG("circle", cx=x, cy=y, r=r, fill="#%s%s%s" % (colorhex, colorhex, colorhex))
    g = canvas(c, width="%dpx" % width, height="%dpx" % height, viewBox="0 0 %d %d" % (width, height),
               style="background-color:black")
    return g.standalone_xml()


def generate_circles_svg(xlist, ylist, rlist, width=792, height=600, colorhexes=["ff"]):
    if len(colorhexes) < len(xlist):
        colorhexes = colorhexes * len(xlist)
    assert len(xlist) == len(ylist) == len(rlist) == len(colorhexes)
    circles = []
    for x, y, r, colorhex in zip(xlist, ylist, rlist, colorhexes):
        circles.append(SVG("circle", cx=x, cy=y, r=r, fill="#%s%s%s" % (colorhex, colorhex, colorhex)))
    g = canvas(*circles, width="%dpx" % width, height="%dpx" % height, viewBox="0 0 %d %d" % (width, height),
               style="background-color:black")
    return g.standalone_xml()


def generate_ellipse_svg(x=60, y=60, r=30, width=792, height=600, a=1, colorhex="ff"):
    c = SVG("ellipse", cx=x, cy=y, r=r, ry=a * r, fill="#%s%s%s" % (colorhex, colorhex, colorhex))
    g = canvas(c, width="%dpx" % width, height="%dpx" % height, viewBox="0 0 %d %d" % (width, height),
               style="background-color:black")
    return g.standalone_xml()


def empty_svg(width=792, height=600):
    return "<svg></svg>"


def add_background(svg):
    c = cStringIO.StringIO()
    c.write(svg)
    c.seek(0)
    p = ET.parse(c)
    root = p.getroot()
    root.set('style', "background-color:black")
    return ET.tostring(root)


def surface_to_np(surface):
    """ Transforms a Cairo surface into a numpy array. """
    temp_buf = cStringIO.StringIO()
    surface.cairo.write_to_png(temp_buf)
    surface.finish()
    temp_buf.seek(0)
    return imread(temp_buf)  # WxHx4 - RGBA


def svg_to_np(svg_bytestring, dpi):
    """ Renders a svg bytestring as a RGB image in a numpy array """
    tree = cairosvg.parser.Tree(bytestring=svg_bytestring)
    surf = cairosvg.surface.SVGSurface(tree, output=None, dpi=dpi)
    return surface_to_np(surf)


def set_svg_bounds(svg, x, y, w, h):
    c = cStringIO.StringIO()
    c.write(svg)
    c.seek(0)
    svg = load_stream(c)

    svg['viewBox'] = u'%.4f %.4f %.4f %.4f' % (x, y, w, h)
    svg['height'] = u'100%'
    svg['width'] = u'100%'
    return svg.standalone_xml()


if __name__ == '__main__':
    svg = generate_circle_svg(20, 20)

    filepath = 'sample_images/two_dots.svg'
    with open(filepath, 'rb') as f:
        svg = f.read()
    print(svg)

    from os import path

    print(add_background(svg))
    svgout = insertTransform(svg, None)
