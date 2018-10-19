import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import cv2

from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("/home/dlrc1/Documents/models/research/")
from object_detection.utils import ops as utils_ops


from object_detection.utils import label_map_util

from object_detection.utils import visualization_utils as vis_util

from toolbox import kinematic_mapping as km


class object_classifier:
    def __init__(self):
        self.fx = 615.607
        self.fy = 615.607

        self.cx = 317.79
        self.cy = 239.907
        self.scaling_factor = 1000
        self.is_center = True
        self.threshold = 0.5
        self.boxlist = []
        self.worldCoords = []
        self.tensor_dict = {}
        
        self.graph = tf.Graph()
        self.sess = tf.Session(graph = self.graph)
        
        self.load_model()
        self.image = np.zeros((299,299,3))
        
        
        
        
    def download_model(self, 
                       DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/', 
                       MODEL_FILE = 'faster_rcnn_inception_v2_coco_2018_01_28.tar.gz' ):
        opener = urllib.request.URLopener()
        opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
        tar_file = tarfile.open(MODEL_FILE)
        for file in tar_file.getmembers():
            file_name = os.path.basename(file.name)
            if 'frozen_inference_graph.pb' in file_name:
                tar_file.extract(file, os.getcwd())
                
    def load_model(self, 
                   PATH_TO_FROZEN_GRAPH = '/home/dlrc1/Dcouments/cheesecake/RobotMapping/faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb',
                  PATH_TO_LABELS="/home/dlrc1/Documents/models/research/object_detection/data/mscoco_label_map.pbtxt"):
        self.sess.close()
        self.graph = tf.Graph()
        with self.graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        self.sess = tf.Session(graph = self.graph)
        names = [n.name for n in self.graph.as_graph_def().node]
        if 'detection_classes' in names:
            self.create_tensor_dict(['num_detections', 'detection_boxes', 'detection_scores', 'detection_classes', 'detection_masks'])
        else:
            self.create_tensor_dict(['final_result'])
        self.create_category_index(PATH_TO_LABELS)
                
    def load_image_into_numpy_array(self, image):
        (im_width, im_height) = image.size
        self.image = np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)
        return self.image
    
    def create_category_index(self, PATH_TO_LABELS):
        self.category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
    
    def create_tensor_dict(self, arr):
        ops = self.graph.get_operations()
        all_tensor_names = {output.name for op in ops for output in op.outputs}
        self.tensor_dict = {}
        for key in arr:
            tensor_name = key + ':0'
            if tensor_name in all_tensor_names:
                self.tensor_dict[key] = self.graph.get_tensor_by_name(tensor_name)
        if 'detection_masks' in self.tensor_dict:
            detection_boxes = tf.squeeze(self.tensor_dict['detection_boxes'], [0])
            detection_masks = tf.squeeze(self.tensor_dict['detection_masks'], [0])
            real_num_detection = tf.cast(self.tensor_dict['num_detections'][0], tf.int32)
            detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
            detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
            detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(detection_masks, detection_boxes, imrgb.shape[0], imrgb.shape[1])
            detection_masks_reframed = tf.cast(tf.greater(detection_masks_reframed, 0.5), tf.uint8)
            self.tensor_dict['detection_masks'] = tf.expand_dims(detection_masks_reframed, 0)
        return self.tensor_dict
    
    def run_inference_for_single_image(self):
        names = [n.name for n in self.graph.as_graph_def().node]
        if 'detection_classes' in names:
            image_tensor = self.sess.graph.get_tensor_by_name('image_tensor:0')
        else:
            image_tensor = self.sess.graph.get_tensor_by_name('Placeholder:0')
            
        self.output_dict = self.sess.run(self.tensor_dict,
                                     feed_dict={image_tensor: np.expand_dims(self.image, 0)})

        if 'num_detections' in self.tensor_dict:
            self.output_dict['num_detections'] = int(self.output_dict['num_detections'][0])
            num_detections = self.output_dict['num_detections']
        if 'detection_classes' in self.tensor_dict:
            self.output_dict['detection_classes'] = self.output_dict['detection_classes'][0].astype(np.uint8)
            self.output_dict['detection_classes'] = self.output_dict['detection_classes'][0:num_detections]
        if 'detection_boxes' in self.tensor_dict:
            self.output_dict['detection_boxes'] = self.output_dict['detection_boxes'][0]
            self.output_dict['detection_boxes'] = self.output_dict['detection_boxes'][0:num_detections]
        if 'detection_scores' in self.tensor_dict:
            self.output_dict['detection_scores'] = self.output_dict['detection_scores'][0]
            self.output_dict['detection_scores'] = self.output_dict['detection_scores'][0:num_detections]
        if 'detection_masks' in self.output_dict:
            self.output_dict['detection_masks'] = self.output_dict['detection_masks'][0]
            self.output_dict['detection_masks'] = self.output_dict['detection_masks'][0:num_detections]
            
        if 'final_result' in self.tensor_dict:
            results = np.squeeze(self.output_dict['final_result'])
            self.output_dict = {'detection_classes':[], 'detection_scores':[]}
            top_k = results.argsort()[-5:][::-1]
            for i in top_k:
                self.output_dict['detection_classes'].append(self.category_index[i+1]['id'])
                self.output_dict['detection_scores'].append(results[i])
        return self.output_dict
    
    def visualize_pred_for_single_img(self):
        if 'detection_boxes' not in self.tensor_dict:
            print('The model is not detecting bounding boxes')
            return None
        IMAGE_SIZE = (12, 8)
        image_np = np.zeros(self.image.shape)
        np.copyto(image_np, self.image)
        image_np = image_np.astype('uint8')
        #image_np_expanded = np.expand_dims(image_np, axis=0)
        #output_dict = run_inference_for_single_image(image_np, detection_graph)
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            self.output_dict['detection_boxes'],
            self.output_dict['detection_classes'],
            self.output_dict['detection_scores'],
            self.category_index,
            instance_masks=self.output_dict.get('detection_masks'),
            use_normalized_coordinates=True,
            line_thickness=8)
        plt.figure(figsize=IMAGE_SIZE)
        plt.imshow(image_np)
        return image_np
        
        
    def Camera(self, Q, Point):
        """Point is an np array of [x, y, z] with ONLY ONE POINT
        x, y and z MUST BE IN METERS """
        df=km.init_params()
        df['theta'].values[0:7]=Q
        TF_Base=km.Transform2Base(df)[7]
        TC_F=km.Rz((1/4)*np.pi) #Fixed
        
        PCamera_F=np.array([0,-0.1, 0.13, 1])

        TC_F[:,3]=PCamera_F
        TC_Base=np.dot(TF_Base,TC_F)
        Point=np.append(Point,1)
        return km.ExecuteTransform(TC_Base, Point)
    
    def setDepthImage(self, img):
        if 'final_result' in self.tensor_dict:
            img = cv2.resize(img, (299,299))
            img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        self.depth_img = img
        
    def setRGBImg(self, img):
        if 'final_result' in self.tensor_dict:
            img = cv2.resize(img, (299,299))
            img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        self.image = img
        
    def set_is_center(self, var):
        self.is_center = var
        
    def setThreshold(self, var):
        self.threshold = var
    
    
    def get_single_detection_box_on_world_coordinates(self, rect, Q):
        rect = rect.astype('int')
        if self.is_center:
            x = int((rect[1] + rect[3])/2)
            y = int((rect[0] + rect[2])/2)
            z = self.depth_img[y,x]
        else:
            x = np.array([rect[1], rect[3], rect[3], rect[1]])
            y = np.array([rect[0], rect[0], rect[2], rect[2]])
            z = self.depth_img[y,x]


        x = (x - self.cx) / (self.fx)
        y = (y - self.cy) / (self.fy)



        x =np.multiply(x,z)/self.scaling_factor
        y =np.multiply(y,z)/self.scaling_factor
        z = z/self.scaling_factor


        xyz = np.stack((x,y,z)).T

        if self.is_center:
            temp = self.Camera(Q, xyz)
            xyz = np.array([temp[0], temp[1], temp[2]])
        else:
            for j in range(len(xyz)):
                temp = self.Camera(Q, xyz[j])
                xyz[j] = np.array([temp[0], temp[1], temp[2]])

        return xyz



    def get_detection_boxes_on_world_coordinates(self, Q):
        if self.is_center:
            world_boxes = np.zeros((len(self.boxlist),1,3))
        else:
            world_boxes = np.zeros((len(self.boxlist),4,3))


        for i in range(len(self.boxlist)):
            rect = self.boxlist[i]
            xyz = self.get_single_detection_box_on_world_coordinates(rect, Q)        
            world_boxes[i,:,:] = xyz    
        return world_boxes



    def get_boxes_with_label_and_threshold(self, label, Q):
        if 'detection_boxes' not in self.tensor_dict:
            print('The model is not detecting bounding boxes')
            return None
        
        boxes = self.output_dict['detection_boxes']
        classes = self.output_dict['detection_classes']
        scores = self.output_dict['detection_scores']

        idx = [ val for key,val in self.category_index.items() if val['name']==label ][0]['id']
        not_equal_get_indexes = lambda xs, x: [i for (y, i) in zip(xs, range(len(xs))) if x != y]
        less_than_get_indexes = lambda xs, x: [i for (y, i) in zip(xs, range(len(xs))) if x > y]


        not_labels = not_equal_get_indexes(classes, idx)
        classes = np.delete(classes, not_labels)
        boxes = np.delete(boxes, not_labels, axis = 0)
        scores = np.delete(scores, not_labels)

        less_than = less_than_get_indexes(scores, self.threshold)
        scores = np.delete(scores, less_than)
        classes = np.delete(classes, less_than)
        boxes = np.delete(boxes, less_than, axis=0)
        
        im_height = self.image.shape[0]
        im_width = self.image.shape[1]
        scales = np.array([im_height, im_width, im_height, im_width])
        boxes = np.multiply(boxes, scales)
        self.boxlist = boxes
        self.worldCoords = self.get_detection_boxes_on_world_coordinates(Q)
        return self.worldCoords
    
    def getHighestConfidenceObservation(self):
        if 'detection_classes' not in self.tensor_dict and 'detection_scores' not in self.tensor_dict:
            print('the network is different than tensorflow api zoo models, checking other models...')
            ops = self.graph.get_operations()
            all_tensor_names = [output.name for op in ops for output in op.outputs]
            if 'final_result:0' not in all_tensor_names:
                print('No known network is assigned!')
                return None
            else:
                image_tensor = self.sess.graph.get_tensor_by_name('Placeholder:0')
                final_layer = self.sess.graph.get_tensor_by_name('final_result:0')
                results = self.sess.run(final_layer, feed_dict={image_tensor: np.expand_dims(self.image, 0)})
                results = np.squeeze(results)
                top = results.argsort()[-1:][0]
                return (self.category_index[top+1]['name'], results[top]) 
        else:
            if not self.output_dict:
                self.run_inference_for_single_image()
            
            return (self.output_dict['detection_classes'][0], self.output_dict['detection_scores'][0])
        
        
    def searchForTarget(self, target):
        if not self.output_dict:
            self.run_inference_for_single_image()
            
        classes = self.output_dict['detection_classes']
        scores = self.output_dict['detection_scores']
        
        
        idx = [ val for key,val in self.category_index.items() if val['name']==target ]
        if not idx:
            return (-1, -1)
        
        idx = idx[0]['id']
        not_equal_get_indexes = lambda xs, x: [i for (y, i) in zip(xs, range(len(xs))) if x != y]
        less_than_get_indexes = lambda xs, x: [i for (y, i) in zip(xs, range(len(xs))) if x > y]
        

        not_labels = not_equal_get_indexes(classes, idx)
        classes = np.delete(classes, not_labels)
        scores = np.delete(scores, not_labels)
        
        if not classes or not scores:
            return (-1, -1)
        
        return (classes, scores)
