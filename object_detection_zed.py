#!/usr/bin/env python3
import numpy as np
import os
import urllib.request
import sys
import tensorflow as tf


import collections
import statistics
import math
import tarfile
import os.path

from threading import Lock, Thread
from time import sleep

import cv2

# ZED imports
import pyzed.sl as sl

sys.path.append('utils')

# ## Object detection imports
from object_detection.utils import config_util
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder

print("Tensorflow version: ", tf.__version__)

# This main thread will run the object detection, the capture thread is loaded later
DATA_DIR = os.path.join(os.getcwd(), 'data')
MODELS_DIR = os.path.join(DATA_DIR, 'models')
for dir in [DATA_DIR, MODELS_DIR]:
	if not os.path.exists(dir):
		os.mkdir(dir)

# What model to download and load
#MODEL_NAME = 'ssd_mobilenet_v1_coco_2018_01_28'
#MODEL_NAME = 'ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03'
#MODEL_NAME = 'ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03'
#MODEL_NAME = 'ssd_mobilenet_v1_coco_2018_01_28'
#MODEL_NAME = 'faster_rcnn_nas_coco_2018_01_28' # Accurate but heavy
#MODEL_NAME = 'centernet_resnet50_v2_512x512_coco17_tpu-8'


# Download and extract model
MODEL_DATE = '20200711'
MODEL_NAME = 'centernet_resnet101_v1_fpn_512x512_coco17_tpu-8'
MODEL_TAR_FILENAME = MODEL_NAME + '.tar.gz'
MODELS_DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/tf2/'
MODEL_DOWNLOAD_LINK = MODELS_DOWNLOAD_BASE + MODEL_DATE + '/' + MODEL_TAR_FILENAME
PATH_TO_MODEL_TAR = os.path.join(MODELS_DIR, MODEL_TAR_FILENAME)
PATH_TO_CKPT = os.path.join(MODELS_DIR, os.path.join(MODEL_NAME, 'checkpoint/'))
PATH_TO_CFG = os.path.join(MODELS_DIR, os.path.join(MODEL_NAME, 'pipeline.config'))
if not os.path.exists(PATH_TO_CKPT):
	print('Downloading model. This may take a while... ', end='')
	urllib.request.urlretrieve(MODEL_DOWNLOAD_LINK, PATH_TO_MODEL_TAR)
	tar_file = tarfile.open(PATH_TO_MODEL_TAR)
	tar_file.extractall(MODELS_DIR)
	tar_file.close()
	os.remove(PATH_TO_MODEL_TAR)
	print('Done')

# Download labels file
LABEL_FILENAME = 'mscoco_label_map.pbtxt'
LABELS_DOWNLOAD_BASE = \
		'https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/data/'
PATH_TO_LABELS = os.path.join(MODELS_DIR, os.path.join(MODEL_NAME, LABEL_FILENAME))
if not os.path.exists(PATH_TO_LABELS):
	print('Downloading label file... ', end='')
	urllib.request.urlretrieve(LABELS_DOWNLOAD_BASE + LABEL_FILENAME, PATH_TO_LABELS)
	print('Done')

# Enable GPU dynamic memory allocation
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
	try:
		print("gpus: ",gpus)
		for gpu in gpus:
			#tf.config.experimental.set_memory_growth(gpu, True)
			tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)])	
	except RuntimeError as e:
		print(e)
		
#Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(PATH_TO_CFG)
model_config = configs['model']
detection_model = model_builder.build(model_config=model_config, is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(PATH_TO_CKPT, 'ckpt-0')).expect_partial()	

category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
																	use_display_name=True)

def load_image_into_numpy_array(image):
	ar = image.get_data()
	ar = ar[:, :, 0:3]
	(im_height, im_width, channels) = image.get_data().shape
	return np.array(ar).reshape((im_height, im_width, 3)).astype(np.uint8)


def load_depth_into_numpy_array(depth):
	ar = depth.get_data()
	ar = ar[:, :, 0:4]
	(im_height, im_width, channels) = depth.get_data().shape
	return np.array(ar).reshape((im_height, im_width, channels)).astype(np.float32)

@tf.function
def detect_fn(image):
	"""Detect objects in image."""

	image, shapes = detection_model.preprocess(image)
	prediction_dict = detection_model.predict(image, shapes)
	detections = detection_model.postprocess(prediction_dict, shapes)

	return detections, prediction_dict, tf.reshape(shapes, [-1])



lock = Lock()
width = 704
height = 416
confidence = 0.35

image_np_global = np.zeros([width, height, 3], dtype=np.uint8)
depth_np_global = np.zeros([width, height, 4], dtype=np.float)

exit_signal = False
new_data = False


# ZED image capture thread function
def capture_thread_func(svo_filepath=None):
	global image_np_global, depth_np_global, exit_signal, new_data

	zed = sl.Camera()

	# Create a InitParameters object and set configuration parameters
	input_type = sl.InputType()
	if svo_filepath is not None:
		input_type.set_from_svo_file(svo_filepath)

	init_params = sl.InitParameters(input_t=input_type)
	init_params.camera_resolution = sl.RESOLUTION.HD720
	init_params.camera_fps = 30
	init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE
	init_params.coordinate_units = sl.UNIT.METER
	init_params.svo_real_time_mode = False


	# Open the camera
	err = zed.open(init_params)
	print(err)
	while err != sl.ERROR_CODE.SUCCESS:
		err = zed.open(init_params)
		print(err)
		sleep(1)

	image_mat = sl.Mat()
	depth_mat = sl.Mat()
	runtime_parameters = sl.RuntimeParameters()
	image_size = sl.Resolution(width, height)

	while not exit_signal:
		if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
			zed.retrieve_image(image_mat, sl.VIEW.LEFT, resolution=image_size)
			zed.retrieve_measure(depth_mat, sl.MEASURE.XYZRGBA, resolution=image_size)
			lock.acquire()
			image_np_global = load_image_into_numpy_array(image_mat)
			depth_np_global = load_depth_into_numpy_array(depth_mat)
			new_data = True
			lock.release()

		sleep(0.001)

	zed.close()


def display_objects_distances(image_np, depth_np, num_detections, boxes_, classes_, scores_, category_index):
	box_to_display_str_map = collections.defaultdict(list)
	box_to_color_map = collections.defaultdict(str)

	research_distance_box = 30

	for i in range(num_detections):
		if scores_[i] > confidence:
			box = tuple(boxes_[i].tolist())
			if classes_[i] in category_index.keys():
				class_name = category_index[classes_[i]]['name']
			display_str = str(class_name)
			if not display_str:
				display_str = '{}%'.format(int(100 * scores_[i]))
			else:
				display_str = '{}: {}%'.format(display_str, int(100 * scores_[i]))

			# Find object distance
			ymin, xmin, ymax, xmax = box
			x_center = int(xmin * width + (xmax - xmin) * width * 0.5)
			y_center = int(ymin * height + (ymax - ymin) * height * 0.5)
			x_vect = []
			y_vect = []
			z_vect = []

			min_y_r = max(int(ymin * height), int(y_center - research_distance_box))
			min_x_r = max(int(xmin * width), int(x_center - research_distance_box))
			max_y_r = min(int(ymax * height), int(y_center + research_distance_box))
			max_x_r = min(int(xmax * width), int(x_center + research_distance_box))

			if min_y_r < 0: min_y_r = 0
			if min_x_r < 0: min_x_r = 0
			if max_y_r > height: max_y_r = height
			if max_x_r > width: max_x_r = width

			for j_ in range(min_y_r, max_y_r):
				for i_ in range(min_x_r, max_x_r):
					z = depth_np[j_, i_, 2]
					if not np.isnan(z) and not np.isinf(z):
						x_vect.append(depth_np[j_, i_, 0])
						y_vect.append(depth_np[j_, i_, 1])
						z_vect.append(z)

			if len(x_vect) > 0:
				x = statistics.median(x_vect)
				y = statistics.median(y_vect)
				z = statistics.median(z_vect)
				
				distance = math.sqrt(x * x + y * y + z * z)

				display_str = display_str + " " + str('% 6.2f' % distance) + " m "
				box_to_display_str_map[box].append(display_str)
				box_to_color_map[box] = vis_util.STANDARD_COLORS[classes_[i] % len(vis_util.STANDARD_COLORS)]

	for box, color in box_to_color_map.items():
		ymin, xmin, ymax, xmax = box

		vis_util.draw_bounding_box_on_image_array(
			image_np,
			ymin,
			xmin,
			ymax,
			xmax,
			color=color,
			thickness=4,
			display_str_list=box_to_display_str_map[box],
			use_normalized_coordinates=True)

	return image_np


def main(args):
	svo_filepath = None
	if len(args) > 1:
		svo_filepath = args[1]


	# %%
	# Load label map data (for plotting)
	# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	# Label maps correspond index numbers to category names, so that when our convolution network
	# predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility
	# functions, but anything that returns a dictionary mapping integers to appropriate string labels
	# would be fine.




	# Start the capture thread with the ZED input
	print("Starting the ZED")
	capture_thread = Thread(target=capture_thread_func, kwargs={'svo_filepath': svo_filepath})
	capture_thread.start()
	# Shared resources
	global image_np_global, depth_np_global, new_data, exit_signal

	# Load a (frozen) Tensorflow model into memory.

	# Detection
	while not exit_signal:
		# Expand dimensions since the model expects images to have shape: [1, None, None, 3]
		if new_data:
			lock.acquire()
			image_np = np.copy(image_np_global)
			depth_np = np.copy(depth_np_global)
			new_data = False
			lock.release()

			image_np_expanded = np.expand_dims(image_np, axis=0)

			input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
			detections, predictions_dict, shapes = detect_fn(input_tensor)

			label_id_offset = 1
			image_np_with_detections = image_np.copy()

			viz_utils.visualize_boxes_and_labels_on_image_array(
				image_np_with_detections,
				detections['detection_boxes'][0].numpy(),
				(detections['detection_classes'][0].numpy() + label_id_offset).astype(int),
				detections['detection_scores'][0].numpy(),
				category_index,
				use_normalized_coordinates=True,
				max_boxes_to_draw=200,
				min_score_thresh=.30,
				agnostic_mode=False)

			# Display output
			cv2.imshow('object detection', cv2.resize(image_np_with_detections, (width,height)))

			if cv2.waitKey(25) & 0xFF == ord('q'):
				break

			"""
					# Visualization of the results of a detection.
					image_np = display_objects_distances(
						image_np,
						depth_np,
						num_detections_,
						np.squeeze(boxes),
						np.squeeze(classes).astype(np.int32),
						np.squeeze(scores),
						category_index)

					cv2.imshow('ZED object detection', cv2.resize(image_np, (width, height)))
					if cv2.waitKey(10) & 0xFF == ord('q'):
						cv2.destroyAllWindows()
						exit_signal = True
			"""            
		else:
			sleep(0.01)

			



	exit_signal = True
	capture_thread.join()


if __name__ == '__main__':
	main(sys.argv)
