#!/usr/bin/env python3

#
# Video inference script
#

# Importing Libraries
import queue
import threading
from threading import Lock
import cv2 as cv
import pytz
import os
import numpy as np
import base64
import logging
import datetime
import time
import random
import argparse
import re
import pprint
import yaml
from dotenv import dotenv_values
import tritonclient.http as httpclient
from tritonclient.utils import triton_to_np_dtype
from PIL import Image
import io
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from time import perf_counter
import json
from json import JSONEncoder

# DEBUG flags
DEBUG     = 0
DEBUG_CAM = 0

ap = argparse.ArgumentParser()
ap.add_argument("--source", "-s", default=0, help="Path to video file or video stream.")
args = vars(ap.parse_args())

source = args["source"]
if source == 0:
    print("Must specify a source.")
    exit(1)

env_file_path = "../shared/.env"

# Load environment variables as a dictionary
try:
    f = open(env_file_path)
    f.close()
except FileNotFoundError:
    print("File '.env' does not exist!")

env_vars = dotenv_values(env_file_path)

# Access environment variables from the dictionary
MQTT_PASSWORD = env_vars.get("MQTT_PASSWORD")
if MQTT_PASSWORD == None:
    print("MQTT_PASSWORD not found!")
    exit(1)

REDIS_PASSWORD = env_vars.get("REDIS_PASSWORD")
if REDIS_PASSWORD == None:
    print("REDIS_PASSWORD not found!")
    exit(1)

class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyArrayEncoder, self).default(obj)

def load_conf_file(config_file):
   with open(config_file, "r") as f:
       config = yaml.safe_load(f)
       triton_conf = config["triton"]
       mqtt_conf = config["mqtt"]
       redis_conf = config["redis"]
       locations_conf = config["locations"]
       for location in config.get('locations', []):
           location_code = location["code"]
           if location["feeds"]:
               for feed_url in location["feeds"]:
                   if source == feed_url:
                      break; 
   return triton_conf, mqtt_conf, redis_conf, locations_conf, location_code, feed_url

config_file_path = "../shared/config.yaml"
triton_conf, mqtt_conf, redis_conf, locations_conf, location_code, feed_url = load_conf_file(config_file_path)

print("\nConfig read successful.\n")

if DEBUG == 1:
    print("MQTT_CONFIG:\n " + str(mqtt_conf))
    print("\n")
    print("LOCATIONS_CONFIG:\n " + str(locations_conf))
    print("\n")
    print("LOCATION_CODE: " + str(location_code))
    print("FEED_URL: " + str(feed_url))

# Triton connection
TRITON_SERVER_IP        = triton_conf["server_ip"]
TRITON_SERVER_PORT      = triton_conf["server_port"]
TRITON_SERVER_URL       = TRITON_SERVER_IP + ":" + str(TRITON_SERVER_PORT)

stream_title = feed_url;

if DEBUG == 1:
    print("MQTT_BROKER_DNS: " + MQTT_BROKER_DNS);
    print("MQTT_BROKER_PORT: " + str(MQTT_BROKER_PORT));
    print("MQTT_USER: " + str(MQTT_USER));
    print("MQTT_PASSWORD: " + str(MQTT_PASSWORD));
    print("REDIS_HOST: " + str(REDIS_HOST));
    print("REDIS_PORT: " + str(REDIS_PORT));
    print("REDIS_PASSWORD: " + str(MQTT_PASSWORD));

FIRST_RECONNECT_DELAY = 1 
RECONNECT_RATE = 2 
MAX_RECONNECT_COUNT = 12
MAX_RECONNECT_DELAY = 60

# XXX Need this setting for rtsp feeds!!! XXX
# https://mpolinowski.github.io/docs/IoT-and-Machine-Learning/ML/2021-12-02--opencv-with-videos/2021-12-02/
#os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

try:
    triton_client = httpclient.InferenceServerClient(url=TRITON_SERVER_URL)
except Exception as ex:
    print('Error:', ex)
    exit('Failed to connect to triton server! Terminating.')

category_index = label_map_util.create_category_index_from_labelmap("./labels.txt", use_display_name=True)
if DEBUG == 1:
    print("CATEGORY INDEX")
    print(category_index)

def infer(image_np, frame_timestamp):

    if DEBUG > 0:
        print("Inside infer")
        print(image_np)
    try:
        image = Image.open(io.BytesIO(image_np))
    except Exception as e:
        print(e)
        sys.exit(1)

    #image_resized = image.resize((600,400))
    image_resized = image.resize((1024,600))

    image_resized_height = image_resized.height
    image_resized_width = image_resized.width
    if DEBUG > 0:
        print(image.info)
        print("Height: " + str(image_resized_height))
        print("Width: " + str(image_resized_width))

    if DEBUG > 0:
        print("Image")
        print(image_resized)

    image_np = np.array(image_resized)

    image_np_org = image_np

    # Add a batch dimension
    image_np = np.expand_dims(image_np, axis=0)
    if DEBUG == 1:
        print("IMAGE_NP_ORG:")
        print(image_np_org)
        print("IMAGE_NP_ORG_SHAPE:")
        print(image_np_org.shape)
        print("IMAGE_NP:")
        print(image_np)
        print("IMAGE_NP_SHAPE:")
        print(image_np.shape)

    inputs = []
    outputs = []

    # Set some values we may want to change 
    #model_name = "faster_rcnn_inception_v2"
    model_name = "object_detection"
    model_version = "1"
    max_boxes_to_draw = 50
    min_score_thresh = .20

    input = httpclient.InferInput("image_tensor", image_np.shape, "UINT8")
    inputs.append(input)
    inputs[0].set_data_from_numpy(image_np)
    
    if DEBUG == 1:
        print("INPUTS:")
        print(inputs)
    
    outputs.append(httpclient.InferRequestedOutput("num_detections", binary_data=False))
    outputs.append(httpclient.InferRequestedOutput("detection_classes", binary_data=False))
    outputs.append(httpclient.InferRequestedOutput("detection_boxes", binary_data=False))
    outputs.append(httpclient.InferRequestedOutput("detection_scores", binary_data=False))
   
    if DEBUG == 1:
        print("OUTPUTS:")
        print(outputs)
    

    #npimg = np.frombuffer(image_np, dtype=np.uint8)
    
    # record start time
    time_start = perf_counter()

    if DEBUG > 0:
        print("Call triton")

    # DO REMOTE INFERENCE
    try:
        response = triton_client.infer(
            model_name,
            inputs=inputs,
            outputs=outputs
        )
    except Exception as e:
        print("Triton client creation failed: " + str(e))
        exit(1)

    if DEBUG > 0:
        print("Done w/ Triton")

    # record end time
    time_end = perf_counter()
    # calculate the duration
    time_duration = time_end - time_start
    # report the duration
    if DEBUG > 0:
        print(f'ONLY INFERENCE: Took {time_duration} seconds')

    result = response.get_response()
    if DEBUG == 1:
        print("RESULT")
        print(result)
    
    num_detections    = response.as_numpy("num_detections")
    detection_classes = response.as_numpy("detection_classes")
    detection_boxes   = response.as_numpy("detection_boxes")
    detection_scores  = response.as_numpy("detection_scores")
    if DEBUG == 1:
        print("Detection scores:")
        print(detection_scores)
    
    label_id_offset = 1
    
    image_np_with_detections = image_np_org.copy()
    if DEBUG == 1:
        print("\nBEFORE image_np_with_detections: ")
        print(image_np_with_detections)
        print("\nDETECTION_SCORES: ")
        print(detection_scores[0])
    
    # record start time
    time_start = perf_counter()

    viz_utils.visualize_boxes_and_labels_on_image_array(
          image_np_with_detections,
          detection_boxes[0],
          (detection_classes[0] + label_id_offset).astype(int),
          detection_scores[0],
          category_index,
          use_normalized_coordinates=True,
          max_boxes_to_draw=max_boxes_to_draw,
          min_score_thresh=min_score_thresh,
          agnostic_mode=False)
    
    # record end time
    time_end = perf_counter()
    # calculate the duration
    time_duration = time_end - time_start
    # report the duration
    if DEBUG > 0: 
        print(f'ONLY viz_utils: Took {time_duration} seconds')

    if DEBUG == 1:
        print("\nAFTER image_np_with_detections: ")
        print(image_np_with_detections)
    frame_details = {}
    frame_details["height"] = image_resized_height
    frame_details["width"] = image_resized_width
    frame_details["timestamp"] = frame_timestamp

    if DEBUG > 0:
        print(frame_timestamp)

    json_frame_details = json.dumps({"frame_details": frame_details})
    if DEBUG == 2:
        print(json_frame_details)
        print("\n")

    numpy_classes = {"detection_classes": detection_classes[0]}
    numpy_boxes = {"detection_boxes": detection_boxes[0]}
    numpy_scores = {"detection_scores": detection_scores[0]}
    json_classes = json.dumps(numpy_classes, cls=NumpyArrayEncoder)
    json_boxes = json.dumps(numpy_boxes, cls=NumpyArrayEncoder)
    json_scores = json.dumps(numpy_scores, cls=NumpyArrayEncoder)

    # Front-end prefers ints instead of floats. 
    # We may need to undo this one day, if the floats prove to be useful from another model.
    # Note: These values can be correlated to the labels.txt file in this directory.
    json_classes_ints = re.sub(r"\.\d+", "", json_classes)

    merged_json_string = merge_json_strings([json_frame_details, json_classes_ints, json_boxes, json_scores])

    if DEBUG == 2:
        print(json_frame_details)
        print("\n")
        print(numpy_boxes)
        print(json_classes)
        print("\n")
        print(json_classes_ints)
        print("\n")
        print(json_boxes)
        print("\n")
        print(json_scores)
        print("\n")
        print(merged_json_string)
        print("\n")

    # record end time
    time_end = perf_counter()
    # calculate the duration
    time_duration = time_end - time_start
    # report the duration
    if DEBUG > 0:  
        print(f'ONLY PUBLISHING: Took {time_duration} seconds')

    return image_np_with_detections, merged_json_string

# Clean up the data for the front-end
def merge_json_strings(json_strings):
    merged_data = "{"
    for json_string in json_strings:
        json_string = re.sub(r"^{", "", json_string)
        json_string = re.sub(r"}$", "", json_string)
        if DEBUG == 2:
            print("json_string: " + str(json_string))
        merged_data = merged_data + str(json_string) + ", ";
    merged_data = re.sub(r", $", "", merged_data)
    merged_data = merged_data + "}"

    return (merged_data)

def time_it(start_time):
    end_time = time.time()
    time_delta = end_time - start_time

    return str(time_delta)


def get_time_local():
    return str(datetime.datetime.now())

def get_time_eastern():
    eastern_tz = pytz.timezone('US/Eastern')
    return str(datetime.datetime.now(tz=eastern_tz))

loop_cnt = 0 
sub_cnt = 0 

#display_frame_wait = 20
display_frame_wait = 1

stream_title = "LIVE: " + source

cv.namedWindow(stream_title)
cv.moveWindow(stream_title, 40,30)

cap = cv.VideoCapture(source)

if DEBUG_CAM == 1:
    print("\n")
    print("Time (Eastern) : " + get_time_eastern())
    print("Time (Local)   : " + get_time_local())
    print("\n")
    print("Current camera configuration:")
    print("Backend      : " + str(cap.getBackendName()))
    print("FPS          : " + str(cap.get(cv.CAP_PROP_FPS)))
    print("Frame Width  : " + str(cap.get(cv.CAP_PROP_FRAME_WIDTH)))
    print("Frame Height : " + str(cap.get(cv.CAP_PROP_FRAME_HEIGHT)))
    print("Codec        : " + str(cap.get(cv.CAP_PROP_FOURCC)))
    print("Codec Format : " + str(cap.get(cv.CAP_PROP_CODEC_PIXEL_FORMAT)))
    print("Format       : " + str(cap.get(cv.CAP_PROP_FORMAT)))
    print("Mode         : " + str(cap.get(cv.CAP_PROP_MODE)))
    print("GUID         : " + str(cap.get(cv.CAP_PROP_GUID)))
    print("Bitrate      : " + str(cap.get(cv.CAP_PROP_BITRATE)))
    print("Open Timeout : " + str(cap.get(cv.CAP_PROP_OPEN_TIMEOUT_MSEC)))
    print("Read Timeout : " + str(cap.get(cv.CAP_PROP_READ_TIMEOUT_MSEC)))
    print("\n")

cap_fps = cap.get(cv.CAP_PROP_FPS)

try:
    while(cap.isOpened()):
        start = time.time()
        frame_timestamp_raw = datetime.datetime.now()
        frame_timestamp = str(frame_timestamp_raw)
        unix_timestamp = datetime.datetime.timestamp(frame_timestamp_raw)

        # Read Frame
        time_it_start = time.time()

        ret, frame = cap.read()

        return_imencode, buffer = cv.imencode('.jpg', frame)

        if DEBUG > 0:  
            print("* imencoding: " + time_it(time_it_start))
    
        # converting into numpy array from buffer
        npimg = np.frombuffer(buffer, dtype=np.uint8)

        if DEBUG == 1:
            print ("NPIMG: ")
            print (npimg)
    
        # record start time
        infer_start = perf_counter()
    
        # Do the inference
        if DEBUG > 0:
            print("Running inference ...")

        npimg_with_detections, boxes_json_string = infer(npimg, frame_timestamp)

        if DEBUG > 0:
            print("Done inferencing ...")

        if DEBUG == 1:
            print("BEFORE FRAMING: ")
            print(npimg_with_detections)
    
        # record end time
        infer_end = perf_counter()

        # calculate the duration
        infer_duration = infer_end - infer_start

        # report the duration
        print(f'    TOTAL INFERENCE TIME: Took {infer_duration} seconds')
    
        frame = npimg_with_detections

        if DEBUG == 1:
            print ("FRAME: ")
            print (frame)
    
        cv.imshow(stream_title, frame)

        end = time.time()
        frame_duration = end - start
        print(f'TOTAL FRAME TIME: Took {frame_duration} seconds')

        if cv.waitKey(display_frame_wait) & 0xFF == ord('q'):
            break
except:
    cap.release()
    cv.destroyAllWindows()
