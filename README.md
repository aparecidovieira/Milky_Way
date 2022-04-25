# Milky_Way

Create a environment
Python=3.6.6

### Installation

change to gst-python-tutorials folder and run
pip install --upgrade wheel pip setuptools
pip install --upgrade --requirement requirements.txt
pip install torch==1.7.1+cpu torchvision==0.8.2+cpu torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.htm

pip install -r environment_milky.txt

This implementation has been developed and tested on Ubuntu 18.04 and 20.04. So the installation steps are specific to debian based linux distros

### Run the application with

{ $ python main.py }

### Push images (np.ndarray) to any Gstreamer pipeline

python gst-python-tutorials/launch_pipeline/run_appsrc_test.py -p "appsrc emit-signals=True is-live=True caps=video/x-raw ! queue ! videoconvert ! autovideosink"  -n 1000


### Usage to start the rtsp server

python gst-python-tutorials/launch_pipeline/muh.py --device_id 0 --fps 30  --port 8554 --stream_uri /video_stream


### Visualization

You can view the video feed on rtsp://server-ip-address:8554/stream_uri

e.g: rtsp://192.168.100.7:8554/video_stream
