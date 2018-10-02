'''
    This script is runs an inference using Tensorflow's Object Detection API  on the Raspberry Pi 3. Works.
    See README.md in root dir for instructions. Enjoy!
    It saves the bounding boxes that have a confidence score > 90%, and saves the coordinates + raw image in pascal voc format for retrainig
    images could be found in retraining/

    Filename: retrain_inference_voc.py
    Author: Miguel Perez
    Date created: 9/01/2018
    Date last modified: 10/02/2018
    Python Version: 3.5
'''

import os
import sys
import cv2
import time
import math

from picamera.array import PiRGBArray
from picamera import PiCamera
from PIL import ImageFont
from PIL import Image

import numpy as np
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from pascal_voc_writer import Writer

#set a variable so that we can log events
i = 0

#load camera
cam_width = 640
cam_height = 480

#append path
sys.path.append('...')

#path to frozen graph
PATH_TO_CKPT = '../my_models/thumbs_up_01/frozen_inference_graph.pb'
PATH_TO_LABELS = os.path.join('../training', 'object-detection.pbtxt')
NUM_CLASSES = 1

#load frozen tensorflow model into memory

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

#load label map

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)
np.set_printoptions(threshold='nan')

#saves the image and the xml string for image annotation
def save_data(img, box):
    #gets the shape of the image
    height, width, channels = img.shape
    #TO DO: double check which values represent xmin, ymin, xmax, ymax
    y1 = math.floor(box[0][0][0] * height)
    x1 = math.floor(box[0][0][1] * width)
    y2 = math.floor(box[0][0][2] * height)
    x2 = math.floor(box[0][0][3] * width)
    res = x1, y1, x2, y2

    print('INFO[*] Thumb Detected at Coordinates: ' + str(res))

    filename = 'retraining/' + str(time.strftime("%Y%m%d-%H%M%S")) + '.jpg'
    cv2.imwrite(filename,img)

    writer = Writer(filename, width, height)
    writer.addObject('thumbs_up', x1, y1, x2, y2)
    writer.save('retraining/img.xml')

    time.sleep(1)

def is_active(bool):
    if(bool):
        i+=1
        print('[INFO] Detections: ' + str(i))
        time.sleep(2)

with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:

        camera = PiCamera()
        camera.resolution = (cam_width,cam_height)
        camera.framerate = 10
        rawCapture = PiRGBArray(camera, size=(cam_width,cam_height))

        for frame in camera.capture_continuous(rawCapture, format='bgr', use_video_port=True):
            image_np = frame.array
            image_np.setflags(write=1)
            image_np_expanded = np.expand_dims(image_np, axis=0)

            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            #actual detections
            (boxes, scores, classes, num_detections) = sess.run(
                                                               [boxes, scores, classes, num_detections],
                                                               feed_dict={image_tensor: image_np_expanded})

            #visualization of the results of a detection
            test, thumb_is_up = vis_util.visualize_boxes_and_labels_on_image_array(image_np,
                                                               np.squeeze(boxes),
                                                               np.squeeze(classes).astype(np.int32),
                                                               np.squeeze(scores),
                                                               category_index,
                                                               use_normalized_coordinates=True,
                                                               min_score_thresh=.5,
                                                               line_thickness=3)


            num_thumbs_up = len([category_index.get(value) for index,value in enumerate(classes[0]) if scores[0,index] > 0.9])

            if(num_thumbs_up > 0):
                print('thumb detected')
                save_data(image_np, boxes)

            cv2.imshow('thumbs up detection', image_np)

            rawCapture.truncate(0)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break

