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
import numpy as np
import gstreamer.utils as utils
from NellieJay.NellieJay import NellieJay

# import required library like Gstreamer and GstreamerRtspServer
gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
from gi.repository import Gst, GstRtspServer, GObject

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
        if self.cap.isOpened():
        # with GstContext():  # create GstContext (hides MainLoop)

            # pipeline = GstPipeline(command)
        # while True:
            ret, frame = self.cap.read()
            # while True:
            if ret:
                # It is better to change the resolution of the camera 
                # instead of changing the image shape as it affects the image quality.
                frame, cow_count = self.my_smart_cow.generate_frame()
                frame = np.array(frame)[:, :, :3]
                results = self.model(frame)  # inference

                res = (results.pandas().xyxy[0].confidence)
                answer = len(res) # Your cow count should go here

                results.render()
                frame = self.my_smart_cow.print_scores(results.imgs[0], cow_count, answer)
                gst_buffer = utils.ndarray_to_gst_buffer(frame)

                frame = cv2.resize(frame, (opt.image_width, opt.image_height), \
                    interpolation = cv2.INTER_LINEAR)

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