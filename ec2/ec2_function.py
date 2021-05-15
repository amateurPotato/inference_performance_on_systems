import io
import os
import json
import boto3
import ntpath
import base64
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2

# Import utilites
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


MODEL_MAP = {'faster-rcnn-resnet50': 'faster_rcnn_resnet50_coco_2018_01_28', 
             'faster-rcnn-inception': 'faster_rcnn_inception_v2_coco_2018_01_28',
             'ssd-mobilenet-v2': 'ssdlite_mobilenet_v2_coco_2018_05_09',
             'ssd-mobilenet-v1': 'ssd_mobilenet_v1_0.75_depth_300x300_coco14_sync_2018_07_03',
             'faster-rcnn-resnet101': 'faster_rcnn_resnet101_coco_2018_01_28'}

def download_from_s3(BUCKET_NAME=None,KEY=None,img_filename=None):
    s3 = boto3.resource('s3')
    s3.Bucket(BUCKET_NAME).download_file(KEY, img_filename)
    
def od_get_boxes(CWD_PATH=os.getcwd(), NUM_CLASSES=90,min_thresh=0.6,IMAGE_NAME = 'image1.jpg', PICK_FROM_S3=False): 
    # Path to frozen detection graph .pb file and label map file, which contains the model that is used
    # for object detection.   
    if PICK_FROM_S3:
        PATH_TO_CKPT = os.path.join(CWD_PATH,'frozen_inference_graph.pb')
        PATH_TO_LABELS = os.path.join(CWD_PATH,'label_map.pbtxt')
        
    else:
        PATH_TO_CKPT = os.path.join(CWD_PATH,'inference_graph','frozen_inference_graph.pb')
        PATH_TO_LABELS = os.path.join(CWD_PATH,'training','label_map.pbtxt')

    # Path to image
    PATH_TO_IMAGE = os.path.join("/home/ubuntu/project",IMAGE_NAME)

    # Number of classes the object detector can identify
    NUM_CLASSES = NUM_CLASSES

    # Load the label map.
    # returns a dictionary mapping integers to string labels
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    # Load the Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef()
        with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        sess =tf.compat.v1.Session(graph=detection_graph)


    # Input tensor is the image
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    # Output tensors are the detection boxes, scores, and classes
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    # Each score represents level of confidence for each of the objects.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

    # Number of objects detected
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    # Load image using OpenCV and
    # expand image dimensions to have shape: [1, None, None, 3]
    image = cv2.imread(PATH_TO_IMAGE)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_expanded = np.expand_dims(image_rgb, axis=0)

    # Perform the actual detection by running the model with the image as input
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: image_expanded})
    
     
    #Draw box in the image
    vis_util.visualize_boxes_and_labels_on_image_array(
    image,
    np.squeeze(boxes),
    np.squeeze(classes).astype(np.int32),
    np.squeeze(scores),
    category_index,
    use_normalized_coordinates=True,
    line_thickness=8,
    min_score_thresh=min_thresh)

    return image, boxes, scores, classes, num

def ec2_handler(event):
    min_thresh = event['min_thresh']
    body_image64 = event['body64'].encode("utf-8")
    num_classes = 90
    
    MODEL_BUCKET = "object-detection-model-fastrcnn"
    img_path = "saved_img.png"
    IMAGE_NAME = ntpath.basename(img_path)
    # Decode & save inp image
    with open(img_path, "wb") as f:
        f.write(base64.b64decode(body_image64))

    # get model
    if event.get('model_from_s3', False):
        model_path = f"/home/ubuntu/project/s3/{MODEL_MAP[event['model']]}"
        download_from_s3(BUCKET_NAME=MODEL_BUCKET,KEY=os.path.join(MODEL_MAP[event['model']],'inference_graph','frozen_inference_graph.pb'),
                     img_filename= model_path + "/frozen_inference_graph.pb")
        download_from_s3(BUCKET_NAME=MODEL_BUCKET,KEY=os.path.join(MODEL_MAP[event['model']],'training','label_map.pbtxt'),
                     img_filename=model_path + "/label_map.pbtxt")
        image, boxes, scores, classes, num = od_get_boxes(CWD_PATH=model_path, NUM_CLASSES=num_classes,
                                                   min_thresh=min_thresh,IMAGE_NAME = IMAGE_NAME,PICK_FROM_S3=True)
        
    else:
        model_path = f"/home/ubuntu/project/{MODEL_MAP[event['model']]}"
        image, boxes, scores, classes, num = od_get_boxes(CWD_PATH=model_path, NUM_CLASSES=num_classes,
                                                   min_thresh=min_thresh,IMAGE_NAME = IMAGE_NAME)
       
        
    img_Pil = Image.fromarray(image, 'RGB')
    buffer = io.BytesIO()
    img_Pil.save(buffer,format="PNG") 
    img_out = buffer.getvalue()

    BUCKET = "object-detection-inference-output-images"

    # if s3 option is true, save to s3 else save on ec2
    if event['key']:
        file_to_save = f"ec2-{event['key']}.jpg"
        with open(file_to_save, "wb") as fh:
            fh.write(img_out)

        s3 = boto3.resource('s3')
        
        s3.Bucket(BUCKET).upload_file(file_to_save, file_to_save)
    else:
        file_to_save = "ec2-jmeter.jpg"
        with open(file_to_save, "wb") as fh:
            fh.write(img_out)

    event["image"] = base64.b64encode(img_out).decode("utf-8")
    
    # remove input image from json
    del event['body64']
    
    return event
