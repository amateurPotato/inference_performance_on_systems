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

# Result:
<img width="1090" alt="Screen Shot 2021-05-15 at 10 13 59 AM" src="https://user-images.githubusercontent.com/5769303/118364516-b8ffde00-b566-11eb-8cd0-18c0aa259ed1.png">

