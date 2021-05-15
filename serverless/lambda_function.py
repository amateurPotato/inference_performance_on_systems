import io, os, json, ntpath, boto3, base64
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

LAMBDA = True if os.environ.get("AWS_EXECUTION_ENV") else False

    
def od_get_boxes(CWD_PATH=os.getcwd(), NUM_CLASSES=90,min_thresh=0.6,IMAGE_NAME = 'image1.jpg'):


    CWD_PATH = CWD_PATH
    
    if LAMBDA:
    	#get artifacts
        PATH_TO_CKPT = os.path.join(CWD_PATH,'frozen_inference_graph.pb')
        PATH_TO_LABELS = os.path.join(CWD_PATH,'label_map.pbtxt')
        PATH_TO_IMAGE = os.path.join("/tmp/",IMAGE_NAME)
        
    else:
        PATH_TO_CKPT = os.path.join(CWD_PATH,'inference_graph','frozen_inference_graph.pb')
        PATH_TO_LABELS = os.path.join(CWD_PATH,'training','label_map.pbtxt')
        PATH_TO_IMAGE = os.path.join("examples",IMAGE_NAME)

    NUM_CLASSES = NUM_CLASSES

    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    # 
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        sess = tf.Session(graph=detection_graph)

    # Define input and output tensors (i.e. data) for the object detection classifier
    # Input tensor is the image
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    # Output tensors are the detection boxes, scores, and classes
    # Each box represents a part of the image where a particular object was detected
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    # Each score represents level of confidence for each of the objects.
    # The score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

    # Number of objects detected
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    # Load image using OpenCV and
    # expand image dimensions to have shape: [1, None, None, 3]
    # i.e. a single-column array, where each item in the column has the pixel RGB value
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

def download_from_s3(BUCKET_NAME=None,KEY=None,img_filename=None):
    s3 = boto3.resource('s3')
    s3.Bucket(BUCKET_NAME).download_file(KEY, img_filename)

def lambda_handler(event, context):
    

    min_thresh = event['min_thresh']
    body_image64 = event['body64'].encode("utf-8")

    # env_variables
    BUCKET_model = os.environ['BUCKET_model']
    MODEL_PATH = os.environ['MODEL_PATH']
    num_classes = int(os.environ['num_classes'])

    
    
    img_path = "/tmp/saved_img.png"
    IMAGE_NAME = ntpath.basename(img_path)
    # Decode & save inputt image to /tmp
    with open(img_path, "wb") as f:
        f.write(base64.b64decode(body_image64))
    print()

    #download label map and model 
    download_from_s3(BUCKET_NAME=BUCKET_model,KEY=os.path.join(MODEL_PATH,'inference_graph','frozen_inference_graph.pb'),
                 img_filename="/tmp/frozen_inference_graph.pb")
    download_from_s3(BUCKET_NAME=BUCKET_model,KEY=os.path.join(MODEL_PATH,'training','label_map.pbtxt'),
                 img_filename="/tmp/label_map.pbtxt")

    # get inference 
    inference_start = datetime.now()
    image, boxes, scores, classes, num = od_get_boxes(CWD_PATH="/tmp/", NUM_CLASSES=categories,
                                               min_thresh=min_thresh,IMAGE_NAME = IMAGE_NAME)
    
    img_Pil = Image.fromarray(image, 'RGB')
    buffer = io.BytesIO()
    img_Pil.save(buffer,format="PNG") 
    img_out = buffer.getvalue()                     
    inference_stop = datetime.now()

    c = inference_stop - inference_start
    total_time = c.total_seconds()
    event["inference_time"] = total_time
    event["image"] = base64.b64encode(img_out).decode("utf-8")  
    # remove input image from json
    del event['body64']

    img_data = event["image"]
    img_data = img_data.encode("utf-8")
    with open("/tmp/" + filename, "wb") as fh:
        fh.write(base64.decodebytes(img_data))

    bucket = 'object-detection-inference-output-images'
    file_name = "/tmp/" + filename
    key_name = filename
    s3 = boto3.client('s3')
    s3.upload_file(file_name, bucket, key_name)
    
    return event
