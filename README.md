# Milky_Way

Create a environment
Python=3.6.6

### Installation

pip install -r environment_cesar.txt

### Push images (np.ndarray) to any Gstreamer pipeline

python launch_pipeline/run_appsrc.py -p "appsrc emit-signals=True is-live=True caps=video/x-raw,format=RGB,width=640,height=480 ! queue ! videoconvert ! autovideosink"  -n 1000
