import sys
import traceback
import argparse
import typing as typ
import random
import time, sys
import torch, cv2
import numpy as np
from fractions import Fraction
sys.path.append('../')
from NellieJay.NellieJay import NellieJay
# from gi.repository import Gdk


from gstreamer import GstContext, GstPipeline, GstApp, Gst, GstVideo, GLib, GstVideoSink
import gstreamer.utils as utils

VIDEO_FORMAT = "RGB"
WIDTH, HEIGHT = 1080, 720
FPS = Fraction(30)
GST_VIDEO_FORMAT = GstVideo.VideoFormat.from_string(VIDEO_FORMAT)


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


FPS_STR = fraction_to_str(FPS)
DEFAULT_CAPS = "video/x-raw,format={VIDEO_FORMAT},width={WIDTH},height={HEIGHT},framerate={FPS_STR}".format(**locals())

# Converts list of plugins to gst-launch string
# ['plugin_1', 'plugin_2', 'plugin_3'] => plugin_1 ! plugin_2 ! plugin_3
DEFAULT_PIPELINE = utils.to_gst_string([
    "appsrc emit-signals=True is-live=True caps={DEFAULT_CAPS}".format(**locals()),
    "queue",
    "videoconvert",
    "autovideosink"
])


ap = argparse.ArgumentParser()
ap.add_argument("-p", "--pipeline", required=False,
                default=DEFAULT_PIPELINE, help="Gstreamer pipeline without gst-launch")

ap.add_argument("-n", "--num_buffers", required=False,
                default=100, help="Num buffers to pass")

args = vars(ap.parse_args())

command = args["pipeline"]

args_caps = parse_caps(command)
NUM_BUFFERS = int(args['num_buffers'])

WIDTH = int(args_caps.get("width", WIDTH))
HEIGHT = int(args_caps.get("height", HEIGHT))
FPS = Fraction(args_caps.get("framerate", FPS))

GST_VIDEO_FORMAT = GstVideo.VideoFormat.from_string(
    args_caps.get("format", VIDEO_FORMAT))
CHANNELS = utils.get_num_channels(GST_VIDEO_FORMAT)
DTYPE = utils.get_np_dtype(GST_VIDEO_FORMAT)

FPS_STR = fraction_to_str(FPS)
CAPS = "video/x-raw,format={VIDEO_FORMAT},width={WIDTH},height={HEIGHT},framerate={FPS_STR}".format(**locals())

path = '/home/aparecido/Desktop/Take_Home_Computer_Vision/best.pt'
model = torch.hub.load('ultralytics/yolov5', 'custom', path=path, force_reload=True)
model.conf = 0.5

with GstContext():  # create GstContext (hides MainLoop)

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
    
    my_smart_cow = NellieJay(height=HEIGHT, width=WIDTH, max_cows=6, delay=1)
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
            
            frame, cow_count = my_smart_cow.generate_frame()
            # frame = np.array(frame)[:, :, :3]
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model(frame)  # inference


            res = (results.pandas().xyxy[0].confidence)
            answer = len(res) # Your cow count should go here

            # results.imgs # array of original images (as np array) passed to model for inference
            results.render()
            frame = my_smart_cow.print_scores(results.imgs[0], cow_count, answer)
            # pred_img = np.array(results.imgs[0])
            gst_buffer = utils.ndarray_to_gst_buffer(frame)

            # set pts and duration to be able to record video, calculate fps
            pts += duration  # Increase pts by duration
            # gst_buffer.pts = pts
            # gst_buffer.duration = duration

            # emit <push-buffer> event with Gst.Buffer
            appsrc.emit("push-buffer", gst_buffer)

        # emit <end-of-stream> event
        appsrc.emit("end-of-stream")

        while not pipeline.is_done:
            time.sleep(.1)
    except Exception as e:
        print("Error: ", e)
    finally:
        pipeline.shutdown()
