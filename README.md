# inference_performance_on_systems

# Goal:  
Our goal is to compare the inference performance of an object detection model in two different paradigms: server and serverless platforms (AWS EC2 and AWS Lambda).For a detailed description, please refer to the Project Report.pdf present in the repo.  

The project is divided into 5 components:  
1) ec2: contains all ec2 api files.  
2) serverless: contains lambda api files.  
3) object_detection: to be put in the same folder as the ec2/serverless folder contents.  
4) serverless.postman_collection.json - API calls examples for serverless setup.  
5) dlproject-experiments.postman_collection.json - API calls examples for EC2 setup.  

 
# Follow the following steps for EC2:
1) Create EC2 instance (t2.small, Ubuntu AMI)
2) Attach static IP
3) Run the following commands on by logging in to the remote server

```
curl -O https://bootstrap.pypa.io/get-pip.py
python3 get-pip.py --user
ls -a ~
vim .profile
export PATH=~/.local/bin:$PATH
pip --version
source ~/.profile
pip install tensorflow

pip install opencv-python
pip install pillow
pip install numpy

pip install boto3
sudo apt-get update
sudo apt-get install -y python3-opencv
pip install opencv-python
pip install psutil
```
3) Keep the 'object_detection' folder in the same directory as the 'ec2' folder contents.
4) Run the following commands to start up the flask server:
```
export FLASK_APP=myapp.py
flask run --host=0.0.0.0 --port=8080
```
5) Use one of the postman calls to test the inference. (dlproject-experiments.postman_collection.json)
6) Don't forget to change the IP address on the call to the one attached to your EC2 instance.  


# Follow the following steps for Serverless setup: 
1) Create a bucket for storing model and label map. 
2) Create a bucket for input images. 
3) Create a bucket for output images. 
4) Create a bucket for artifacts. 
5) Upload model and label files in for each model in separate folder. 
6) Upload lambda code in a zip to artifacts bucket. 
7) Upload requirements.zip file to artifact bucket. 
8) Create IAM Role: lambda_inference_role with following access for lambda function:  
```
create cloudwatch logs
access above s3 buckets
invoke other lambda function
```

9) Create aws layer with serverless/requirements.zip. 

10) Create 5 lambda functions:  
```
Runtime: Python 3.6
Role: lambda_inference_role
Environment Variable: 
BUCKET_NAME: <use the unique bucket name for model>
MODEL_FOLDER: < use the folder name for model as per the lambda function>
TOTAL_CATEGORIES: 90
```

Lambda | Timeout | Memory 
--- | --- | --- 
ssdlite_mobilenet_v2_coco | 720 | 20 
faster-rcnn-inception-v2 | 1000 | 30
ssd_mobilenet_v1 | 750 | 18 
faster_rcnn_resnet50_coco | 1500 | 40
faster_rcnn_resnet101_coco | 1800 | 60

11) Create API gateway with 5 resources with POST method on each, and attach corresponding lambda functions

# Result:
<img width="1090" alt="Screen Shot 2021-05-15 at 10 13 59 AM" src="https://user-images.githubusercontent.com/5769303/118364516-b8ffde00-b566-11eb-8cd0-18c0aa259ed1.png">

