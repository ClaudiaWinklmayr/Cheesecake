import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("/home/dlrc1/Documents/models/research/")
from object_detection.utils import ops as utils_ops


from object_detection.utils import label_map_util

from object_detection.utils import visualization_utils as vis_util

from toolbox import kinematic_mapping as km


def create_category_index(PATH_TO_LABELS):
    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
    return category_index

def download_model(DOWNLOAD_BASE, MODEL_FILE ):
    opener = urllib.request.URLopener()
    opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
    tar_file = tarfile.open(MODEL_FILE)
    for file in tar_file.getmembers():
        file_name = os.path.basename(file.name)
        if 'frozen_inference_graph.pb' in file_name:
            tar_file.extract(file, os.getcwd())


def load_model(PATH_TO_FROZEN_GRAPH, detection_graph):
    #detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')


def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


def create_tensor_dict(arr):
    ops = tf.get_default_graph().get_operations()
    all_tensor_names = {output.name for op in ops for output in op.outputs}
    tensor_dict = {}
    for key in arr:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
            tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
    if 'detection_masks' in tensor_dict:
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(detection_masks, detection_boxes, imrgb.shape[0], imrgb.shape[1])
        detection_masks_reframed = tf.cast(tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        tensor_dict['detection_masks'] = tf.expand_dims(detection_masks_reframed, 0)
    return tensor_dict


def run_inference_for_single_image(image, graph):
  with graph.as_default():

    with tf.Session() as sess:
    #with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
      # Get handles to input and output tensors
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)
      if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

      # Run inference
      output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: np.expand_dims(image, 0)})

      # all outputs are float32 numpy arrays, so convert types as appropriate
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.uint8)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
  return output_dict



def run_inference_for_single_image_test(image, tensor_dict, sess):
    image_tensor = sess.graph.get_tensor_by_name('image_tensor:0')
    output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: np.expand_dims(image, 0)})
    # all outputs are float32 numpy arrays, so convert types as appropriate
    output_dict['num_detections'] = int(output_dict['num_detections'][0])
    output_dict['detection_classes'] = output_dict[
        'detection_classes'][0].astype(np.uint8)
    output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
    output_dict['detection_scores'] = output_dict['detection_scores'][0]
    if 'detection_masks' in output_dict:
      output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict



def visualize_pred_for_single_img(image, output_dict, category_index):
    IMAGE_SIZE = (12, 8)
    #image_np = load_image_into_numpy_array(image)
    image_np = np.zeros(image.shape)
    np.copyto(image_np, image)
    image_np = image_np.astype('uint8')
    #image_np_expanded = np.expand_dims(image_np, axis=0)
    #output_dict = run_inference_for_single_image(image_np, detection_graph)
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        instance_masks=output_dict.get('detection_masks'),
        use_normalized_coordinates=True,
        line_thickness=8)
    plt.figure(figsize=IMAGE_SIZE)
    plt.imshow(image_np)
    return image_np


def Camera(Q, Point):
        """Point is an np array of [x, y, z] with ONLY ONE POINT
        x, y and z MUST BE IN METERS """
        df=km.init_params()
        df['theta'].values[0:7]=Q
        TF_Base=km.Transform2Base(df)[7]
        TC_F=km.Rz((1/4)*np.pi) #Fixed

        #PCamera_F=np.array([0,-0.05,0.08, 1])
        PCamera_F=np.array([0,-0.1, 0.13, 1])

        TC_F[:,3]=PCamera_F
        TC_Base=np.dot(TF_Base,TC_F)
        Point=np.append(Point,1)
        print("Point with respect to camera: ", Point)
        print("Point with respect to system F: ", km.ExecuteTransform(TC_F, Point))
        print("Point with respect to Base: ", km.ExecuteTransform(TC_Base, Point))
        print("Location of the camera: ", km.ExecuteTransform(TC_Base, np.array([0,0,0,1])))
        print("Location of the coordinate Camerax: ", km.ExecuteTransform(TC_Base, np.array([0.1,0,0,1])))

        print("Location of the coordinate Cameray: ", km.ExecuteTransform(TC_Base, np.array([0,0.1,0,1])))
        print("Location of the system F: ", km.ExecuteTransform(TF_Base, np.array([0,0,0,1])))


        return km.ExecuteTransform(TC_Base, Point)



def get_single_detection_box_on_world_coordinates(rect, depth_img, Q, is_center):
    fx = 615.607
    fy = 615.607

    cx = 317.79
    cy = 239.907
    scaling_factor = 1000
    
    rect = rect.astype('int')
    
    
    if is_center:
        x = int((rect[1] + rect[3])/2)
        y = int((rect[0] + rect[2])/2)
        z = depth_img[y,x]
    else:
        x = np.array([rect[1], rect[3], rect[3], rect[1]])
        y = np.array([rect[0], rect[0], rect[2], rect[2]])
        z = depth_img[y,x]
    
    
    x = (x - cx) / (fx)
    y = (y - cy) / (fy)
    
    
    
    x =np.multiply(x,z)/scaling_factor
    y =np.multiply(y,z)/scaling_factor
    z = z/scaling_factor

    
    xyz = np.stack((x,y,z)).T
        
    if is_center:
        temp = Camera(Q, xyz)
        xyz = np.array([temp[0], temp[1], temp[2]])
    else:
        for j in range(len(xyz)):
            temp = Camera(Q, xyz[j])
            xyz[j] = np.array([temp[0], temp[1], temp[2]])

    return xyz



def get_detection_boxes_on_world_coordinates(boxlist, depth_img, Q, is_center=True):
    if is_center:
        world_boxes = np.zeros((len(boxlist),1,3))
    else:
        world_boxes = np.zeros((len(boxlist),4,3))
        
        
    for i in range(len(boxlist)):
        rect = boxlist[i]
        xyz = get_single_detection_box_on_world_coordinates(rect, depth_img, Q, is_center)        
        world_boxes[i,:,:] = xyz    
    return world_boxes



def get_boxes_with_label_and_threshold(label, threshold, output_dict, imdepth, Q, category_index, is_center = True):
    boxes = output_dict['detection_boxes']
    classes = output_dict['detection_classes']
    scores = output_dict['detection_scores']

    idx = [ val for key,val in category_index.items() if val['name']==label ][0]['id']
    not_equal_get_indexes = lambda xs, x: [i for (y, i) in zip(xs, range(len(xs))) if x != y]
    less_than_get_indexes = lambda xs, x: [i for (y, i) in zip(xs, range(len(xs))) if x > y]


    not_labels = not_equal_get_indexes(classes, idx)
    classes = np.delete(classes, not_labels)
    boxes = np.delete(boxes, not_labels, axis = 0)
    scores = np.delete(scores, not_labels)

    less_than = less_than_get_indexes(scores, threshold)
    scores = np.delete(scores, less_than)
    classes = np.delete(classes, less_than)
    boxes = np.delete(boxes, less_than, axis=0)
    im_height = imdepth.shape[0]
    im_width = imdepth.shape[1]
    scales = np.array([im_height, im_width, im_height, im_width])
    boxes = np.multiply(boxes, scales)
    result = get_detection_boxes_on_world_coordinates(boxes, imdepth, Q, is_center)
    return (result, boxes)
