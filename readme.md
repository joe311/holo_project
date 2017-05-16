# Holographics package

**Paper link is pending**

## Holographic software package, for two-photon light shaping to drive optogenetics, as a companion to [Dal Maschio and Donovan et al., Neuron 2017](*insert link to paper here*)

This software requires the proper hardware setup to use (see the paper for hardware implementation details).
It also requires some simple integration into the software running the microscope to calibrate the imaging path relative to the holographic system.
  
#### Hardware assumptions:
* DVI-connected SLM with 792x600 active pixels (in our case, a Hamamatsu SLM). Calibration files for your specific SLM should be placed in the static folder.
* For calibration, an OpenCV-compatiable camera placed after the objective (with appropriate coupling objective).

For more details about the optical setup, see the paper and 3d CAD model.
   
#### Software model
The system is setup as a single server - multiple client model.  The server is connected to SLM, and handles calibration and computation. The clients request desired computations patterns, or provide information for calibration.
   
#### Message format:
The server and clients communicate using ZeroMQ messages, which are serialized using Google Protocol buffers and SVGs.

The client initiates every request, and the server replies to each request with a single message.
The messages are serialized using Protocol buffers.  The desired patterns are passed inside the messages using the SVG vector format, using μm units.

Two crucial message types are Generate (for generating a desired set of patterns), and Play (play the last pattern generated).

The server caches its computations, so previously requested patterns are generated nearly instantly.
   
#### Frame format:
A generate message can have multiple frames.  All frames to be played simultaneously should have the same `frame_num` message parameter.
Increasing `frame_num` indicates multiple frames to be played in order.
Each separate Z-level should have a separate frame in the message. The duration is specified in the message, which should be the same for all frames for each `frame_num`.
   
#### XY Calibration:
To calibrate the system in XY, a client (generally the scanning software) should send several calibrations messages with a corresponding hardware states.

The states are:

* background image, no beams active (`CALIBRATE_BACKGROUND`),

* calibration point, galvo scanners positioned at a point in the focal plane, with the shutter open; this should be called multiple times with different points (`CALIBRATE_CIRCLE`),

Finally call `CALIBRATE_RUN`.

#### Z Calibration:
To calibrate the system in Z, a client (generally the scanning software) should send `CALIBRATE_Z` messages, with the current Z-level. The holographic beam should be enabled (ie: shutter open). The user then has to adjust the objective level to focus the displayed pattern. After all Z-levels have been entered, `CALIBRATE_Z_RUN` should be called.

#### Power control:
The SLM system shapes the light through phase shifts, rather than directly controlling the amplitude, so proper control of the laser power entering the system is critical.  The software always normalizes the power within each contemporaneous frame set to the maximum, and the laser power should be adjusted accordingly using a Pockel's cell or similar. The system computes the excitation power - that is that it doesn't take into effect the two-photon effect, or differences in expression. 

#### Software and libraries:
Tested on windows 7 - Windows 10 *seems* to work, other operating systems are possible but may require small adjustments.

Python 2.7 (32 vs 64bit shouldn't matter as long as all the libraries are the correct version).

The package path should be added to the system or python path. 

See the package requirements file for a detailed list of libraries; starting from an installation of [Anaconda](https://www.continuum.io/downloads) is likely faster than installing all libraries manually.

SVGfig-1.x can be found here: [SVGFIG](https://github.com/jpivarski/svgfig/tree/master/svgfig-1.x)
    
#### Software organization:
`holobase.py` is the main block of code for the sever.

`holoclient.py` provides the client interface, with the option of using a convenience class or function decorator.
    
#### License
The software is released under a AGPL v3 license - for the full license please see the LICENSE text. 
In general terms, this license requires projects distributing this code (or modifications thereof) to release their source code.
For questions about licensing, please email us or consult with a lawyer specializing in software licensing. 
    
   
    
    