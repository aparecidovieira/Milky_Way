#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  20 02:07:13 2019

@author: prabhakar
"""
# import necessary argumnets 
import gi
import cv2, torch
import argparse
from fractions import Fraction

import numpy as np
import gstreamer.utils as utils
from NellieJay.NellieJay import NellieJay
from gstreamer import GstContext, GstPipeline, GstApp, Gst, GstVideo, GLib, GstVideoSink

# import required library like Gstreamer and GstreamerRtspServer
gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
from gi.repository import Gst, GstRtspServer, GObject


def fraction_to_str(fraction: Fraction) -> str:
    """Converts fraction to str"""
    return '{}/{}'.format(fraction.numerator, fraction.denominator)


def parse_caps(pipeline: str) -> dict:
    """Parses appsrc's caps from pipeline string into a dict

    :param pipeline: "appsrc caps=video/x-raw,format=RGB,width=640,height=480 ! videoconvert ! autovideosink"

    Result Example:
        {
            "width": "1080",
            "height": "720"
            "format": "RGB",
            "fps": "30/1",
            ...
        }
    """

    try:
        # typ.List[typ.Tuple[str, str]]
        caps = [prop for prop in pipeline.split(
            "!")[0].split(" ") if "caps" in prop][0]
        return dict([p.split('=') for p in caps.split(',') if "=" in p])
    except IndexError as err:
        return None

VIDEO_FORMAT = "RGB"
WIDTH, HEIGHT = 1080, 720
FPS = Fraction(30)
GST_VIDEO_FORMAT = GstVideo.VideoFormat.from_string(VIDEO_FORMAT)


CHANNELS = utils.get_num_channels(GST_VIDEO_FORMAT)
DTYPE = utils.get_np_dtype(GST_VIDEO_FORMAT)

FPS_STR = fraction_to_str(FPS)
CAPS = "video/x-raw,format={VIDEO_FORMAT},width={WIDTH},height={HEIGHT},framerate={FPS_STR}".format(**locals())



FPS_STR = fraction_to_str(FPS)
DEFAULT_CAPS = "video/x-raw,format={VIDEO_FORMAT},width={WIDTH},height={HEIGHT},framerate={FPS_STR}".format(**locals())
# Sensor Factory class which inherits the GstRtspServer base class and add
# properties to it.
class SensorFactory(GstRtspServer.RTSPMediaFactory):
    def __init__(self, **properties):
        super(SensorFactory, self).__init__(**properties)
        self.cap = cv2.VideoCapture(opt.device_id)
        self.number_frames = 0
        self.fps = opt.fps
        self.duration = 1 / self.fps * Gst.SECOND  # duration of a frame in nanoseconds
        self.launch_string = 'appsrc name=source is-live=true block=true format=GST_FORMAT_TIME ' \
                             'caps=video/x-raw,format=BGR,width={},height={},framerate={}/1 ' \
                             '! videoconvert ! video/x-raw,format=I420 ' \
                             '! x264enc speed-preset=ultrafast tune=zerolatency ' \
                             '! rtph264pay config-interval=1 name=pay0 pt=96' \
                             .format(opt.image_width, opt.image_height, self.fps)
    # method to capture the video feed from the camera and push it to the
    # streaming buffer.
        # Path to torch weights
        self.weight = opt.weight
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=self.weight, force_reload=True)
        self.model.conf = 0.5
        self.my_smart_cow = NellieJay(height=opt.image_height, width=opt.image_width, max_cows=6, delay=1)

    def on_need_data(self, src, length):
        # if self.cap.isOpened():
        with GstContext():  # create GstContext (hides MainLoop)
            command = opt.pipeline

            pipeline = GstPipeline(command)

            def on_pipeline_init(self):
                """Setup AppSrc element"""
                appsrc = self.get_by_cls(GstApp.AppSrc)[0]  # get AppSrc

                # instructs appsrc that we will be dealing with timed buffer
                appsrc.set_property("format", Gst.Format.TIME)

                # instructs appsrc to block pushing buffers until ones in queue are preprocessed
                # allows to avoid huge queue internal queue size in appsrc
                appsrc.set_property("block", True)

                # set input format (caps)
                appsrc.set_caps(Gst.Caps.from_string(CAPS))

            # override on_pipeline_init to set specific properties before launching pipeline
            pipeline._on_pipeline_init = on_pipeline_init.__get__(pipeline)
            
            # my_smart_cow = NellieJay(height=HEIGHT, width=WIDTH, max_cows=6, delay=1)
            print('Muuuuuuuuuuhhhhhhh')
            try:
                pipeline.startup()
                appsrc = pipeline.get_by_cls(GstApp.AppSrc)[0]  # GstApp.AppSrc
                answer=0
                pts = 0  # buffers presentation timestamp
                duration = 10**9 / (FPS.numerator / FPS.denominator) * 0.8  # frame duration
                # for _ in range(NUM_BUFFERS):
                while True:

                    # create random np.ndarray
                    
                    frame, cow_count = self.my_smart_cow.generate_frame()
                    frame = np.array(frame)[:, :, :3]
                    results = self.model(frame)  # inference


                    res = (results.pandas().xyxy[0].confidence)
                    answer = len(res) # Your cow count should go here

                    # results.imgs # array of original images (as np array) passed to model for inference
                    results.render()
                    frame = self.my_smart_cow.print_scores(results.imgs[0], cow_count, answer)
                    # pred_img = np.array(results.imgs[0])
                    gst_buffer = utils.ndarray_to_gst_buffer(frame)

                    # set pts and duration to be able to record video, calculate fps
                    pts += duration  # Increase pts by duration
                    # gst_buffer.pts = pts
                    # gst_buffer.duration = duration
                    data = frame.tostring()
                    buf = Gst.Buffer.new_allocate(None, len(data), None)
                    buf.fill(0, data)
                    buf.duration = self.duration
                    timestamp = self.number_frames * self.duration
                    buf.pts = buf.dts = int(timestamp)
                    buf.offset = timestamp
                    self.number_frames += 1
                    retval = src.emit('push-buffer', gst_buffer)
                    print('pushed buffer, frame {}, duration {} ns, durations {} s'.format(self.number_frames,
                                                                                        self.duration,
                                                                                        self.duration / Gst.SECOND))
                    if retval != Gst.FlowReturn.OK:
                        print(retval)
                        # emit <push-buffer> event with Gst.Buffer
                        # src.emit("push-buffer", gst_buffer)

                # emit <end-of-stream> event
                appsrc.emit("end-of-stream")

                while not pipeline.is_done:
                    time.sleep(.1)
            except Exception as e:
                print("Error: ", e)
            finally:
                pipeline.shutdown()

    # attach the launch string to the override method
    def do_create_element(self, url):
        return Gst.parse_launch(self.launch_string)
    
    # attaching the source element to the rtsp media
    def do_configure(self, rtsp_media):
        self.number_frames = 0
        appsrc = rtsp_media.get_element().get_child_by_name('source')
        appsrc.connect('need-data', self.on_need_data)

# Rtsp server implementation where we attach the factory sensor with the stream uri
class GstServer(GstRtspServer.RTSPServer):
    def __init__(self, **properties):
        super(GstServer, self).__init__(**properties)
        self.factory = SensorFactory()
        self.factory.set_shared(True)
        self.set_service(str(opt.port))
        self.get_mount_points().add_factory(opt.stream_uri, self.factory)
        self.attach(None)
        
DEFAULT_PIPELINE = utils.to_gst_string([
    "appsrc emit-signals=True is-live=True caps={DEFAULT_CAPS}".format(**locals()),
    "queue",
    "videoconvert",
    "autovideosink"
])


def fraction_to_str(fraction: Fraction) -> str:
    """Converts fraction to str"""
    return '{}/{}'.format(fraction.numerator, fraction.denominator)


def parse_caps(pipeline: str) -> dict:
    """Parses appsrc's caps from pipeline string into a dict

    :param pipeline: "appsrc caps=video/x-raw,format=RGB,width=640,height=480 ! videoconvert ! autovideosink"

    Result Example:
        {
            "width": "1080",
            "height": "720"
            "format": "RGB",
            "fps": "30/1",
            ...
        }
    """

    try:
        # typ.List[typ.Tuple[str, str]]
        caps = [prop for prop in pipeline.split(
            "!")[0].split(" ") if "caps" in prop][0]
        return dict([p.split('=') for p in caps.split(',') if "=" in p])
    except IndexError as err:
        return None
from gstreamer import GstContext, GstPipeline, GstApp, Gst, GstVideo, GLib, GstVideoSink
import gstreamer.utils as utils


# getting the required information from the user 
parser = argparse.ArgumentParser()
parser.add_argument("--device_id", required=True, help="device id for the \
                video device or video file location")
parser.add_argument("--fps", required=True, help="fps of the camera", type = int)
parser.add_argument("--image_width", default=1080, help="video frame width", type = int)
parser.add_argument("--image_height", default=720, help="video frame height", type = int)
parser.add_argument("--port", default=8554, help="port to stream video", type = int)
parser.add_argument("--stream_uri", default = "/video_stream", help="rtsp video stream uri")
parser.add_argument("--weight", default='best.pt', help="path to torch weights", type = str)
parser.add_argument("-p", "--pipeline", required=False,
                default=DEFAULT_PIPELINE, help="Gstreamer pipeline without gst-launch")
opt = parser.parse_args()

try:
    opt.device_id = int(opt.device_id)
except ValueError:
    pass

# initializing the threads and running the stream on loop.
GObject.threads_init()
Gst.init(None)
server = GstServer()
loop = GObject.MainLoop()
loop.run()