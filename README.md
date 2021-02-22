# Face Recognition for NVIDIA Jetson (Nano) using Deepstream
Face recognition with [Google FaceNet](https://arxiv.org/abs/1503.03832)
architecture. The Deepstream pipeline uses YOLO Face as the primary detector and FaceNet for the recognition of the face. This Repo consists of implementation of the Deepstream app in C++ and Python. 


## Dependencies
cuda 10.2 + cudnn 8.0 <br> TensorRT 7.x <br> OpenCV 4.1.1 <br>
TensorFlow 1.15.4 <br> numpy 1.16.1 <br>onnx 1.6.0 <br> onnx-tf 1.3.0
 <br> onnxruntime 1.6.0 <br> DeepStream SDK 5.0.1
 

## Steps
1. Installation
2. Prepare PGIE (YOLO Face)
3. Prepare SGIE (FaceNet)
4. Implement in Python
5. Implement in C++


## Installation
#### 1. Install Cuda, CudNN, TensorRT, and TensorFlow for Python 
Installing DeepStream is fairly simple, all the steps are mentioned in the Nvidia's [Docs](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Quickstart.html). In case you are coming from previous versions of DeepStreamSDK, they have guides to upgrade to the latest version as well. Before proceeding, please do test the installation by running a sample app by running the following command in the shell

```bash
# Test deepstream installation:
deepstream-app -c /opt/nvidia/deepstream/deepstream-5.0/samples/configs/deepstream-app/source8_1080p_dec_infer-resnet_tracker_tiled_display_fp16_nano.txt
```
If this app runs fine, we can proceed to next steps, In case you are facing issues while running the sample app please refer [Deepstream SDK forums](https://forums.developer.nvidia.com/c/accelerated-computing/intelligent-video-analytics/deepstream-sdk/15)

Follow [this](https://github.com/NVIDIA-AI-IOT/deepstream_python_apps/blob/master/HOWTO.md) guide to install DeepStream-Python-Apps 

#### 2. Install Tensorflow
The following shows the steps to install Tensorflow for Jetpack 4.4. This was copied from the official [NVIDIA documentation](https://docs.nvidia.com/deeplearning/frameworks/install-tf-jetson-platform/index.html). I'm assuming you don't need to install it in a virtual environment. If yes, please refer to the documentation linked above. If you are not installing this on a jetson, please refer to the official tensorflow documentation.

```bash
# Install system packages required by TensorFlow:
sudo apt update
sudo apt install libhdf5-serial-dev hdf5-tools libhdf5-dev zlib1g-dev zip libjpeg8-dev liblapack-dev libblas-dev gfortran
sudo apt-get install libprotobuf-dev protobuf-compiler

# Install and upgrade pip3
sudo apt install python3-pip
sudo pip3 install -U pip testresources setuptools

#Installing ONNX and its dependancies

sudo pip3 install onnx==1.6.0 onnx-tf 1.3.0 onnxruntime==1.6.0

#Note that ONNX will try to install Numpy 1.19 let it install, once the installation of ONNX is complete please revert back to Numpy == 1.16 as the latest version has conflits with tensorflow 1.15. All the ONNX operations are supported with Numpy 1.16 as well.

# Install the Python package dependencies
sudo pip3 install -U numpy==1.16.1 future==0.18.2 mock==3.0.5 h5py==2.10.0 keras_preprocessing==1.1.1 keras_applications==1.0.8 gast==0.2.2 futures protobuf pybind11

# Install TensorFlow using the pip3 command. This command will install the latest version of TensorFlow compatible with JetPack 4.4.
sudo pip3 install --pre --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v44 'tensorflow<2'

```


## Prepare Face detector
* First, we make some changes so that we can use YOLO with DeepStream. Either provide the deepstream folder the read and write permissions or use the commangs with sudo
```bash
#navigating to target folder
cd /opt/nvidia/deepstream/deepstream-5.0/sources/objectDetector_Yolo

#making a backup folders to revert the changes
sudo mkdir bkp

#moving the folder to backup folder
mv ./nvdsinfer_custom_impl_Yolo ./bkp

#cloning the repo which has made all the changes already, Thanks to  @marcoslucianops 
git clone https://github.com/marcoslucianops/DeepStream-Yolo.git

cd /DeepStream-Yolo/native
cp -r ./nvdsinfer_custom_impl_Yolo /opt/nvidia/deepstream/deepstream-5.0/sources/objectDetector_Yolo

#compile the changes made
cd /opt/nvidia/deepstream/deepstream-5.0/sources/objectDetector_Yolo

CUDA_VER=10.2 make -C nvdsinfer_custom_impl_Yolo

```


* Download the weights, cfg and labels files from [this](https://github.com/lthquy/Yolov3-tiny-Face-weights) repo. place the .weights and .cfg file in /opt/nvidia/deepstream/deepstream-5.0/sources/objectDetector_Yolo, after that make a labels.txt file consisting of one class 'Face'.

* if the below mentioned config files are already existing in your workspace please move them to /bkp and move these files in /objectDetector_Yolo
```bash
cd /opt/nvidia/deepstream/deepstream-5.0/sources/objectDetector_Yolo

#clone develop branch of this repo
git clone -b develop --single-branch https://github.com/shubham-shahh/mtcnn_facenet_cpp_tensorRT.git

#copy config files for reference
cd ./mtcnn_facenet_cpp_tensorRT/YoloApp
cp ./deepstream_app_config_yoloV3_tiny.txt /opt/nvidia/deepstream/deepstream-5.0/sources/objectDetector_Yolo
cp ./config_infer_primary_yoloV3_tiny.txt /opt/nvidia/deepstream/deepstream-5.0/sources/objectDetector_Yolo
cp ./dstest2_pgie_config.txt /opt/nvidia/deepstream/deepstream-5.0/sources/objectDetector_Yolo

```
* add names or paths of the respective files based on your config [here](https://github.com/shubham-shahh/mtcnn_facenet_cpp_tensorRT/blob/81a3cad4efa76eea9f98e96dfd5540f341107068/YoloApp/config_infer_primary_yoloV3_tiny.txt#L65-L68) once everything is in place we can test our app.

```bash
#Test if we properly configured YOLO to run with Deepstream
cd /opt/nvidia/deepstream/deepstream-5.0/sources/objectDetector_Yolo
deepstream-app -c ./deepstream_app_config_yoloV3_tiny.txt
```
* change the [source](https://github.com/shubham-shahh/mtcnn_facenet_cpp_tensorRT/blob/81a3cad4efa76eea9f98e96dfd5540f341107068/YoloApp/deepstream_app_config_yoloV3_tiny.txt#L47) in case you want to test on some other video

## Prepare FaceNet
* This step relies on [this]() repo. I couldn't make things work using this repo. for more info follow [this](https://github.com/riotu-lab/deepstream-facenet/issues/5) issue hence I've made some modifications to keep things rolling.


* First, download the keras model of Facenet from [this](https://github.com/nyoki-mtl/keras-facenet) fork and place it in /mtcnn_facenet_cpp_tensorRT/ModelConverion/kerasmodel

```bash
#Test if we properly configured YOLO to run with Deepstream
cd ./mtcnn_facenet_cpp_tensorRT/ModelConversion
python3 h5topb.py --input_path ./kerasmodel/facenet_keras_128.h5 --output_path ./tensorflowmodel/facenet.pb

#Convert Tensorflow model to an ONNX model. This will take approx 50 mins and this has to be done on the host device
python3 -m tf2onnx.convert --input ./tensorflowmodel/facenet.pb --inputs input_1:0[1,160,160,3] --inputs-as-nchw input_1:0 --outputs Bottleneck_BatchNorm/batchnorm_1/add_1:0 --output onnxmodel/facenetconv.onnx

#Convert ONNX model to a model that can take dynamic input size
python3 dynamic_conv.py --input_path ./onnxmodel/facenetconv.onnx --output_path ./dynamiconnxmodel/dynamicfacenetmodel.onnx

```

## Python Implementation
* Once we have the dynamic onnx model copy it to /mtcnn_facenet_cpp_tensorRT/facenet-python/facenetmodel
```bash
cd ./mtcnn_facenet_cpp_tensorRT/ModelConversion/dynamiconnxmodel/
cp ./dynamicfacenetmodel.onnx  /path/to/mtcnn_facenet_cpp_tensorRT/facenet-python/facenetmodel

#Now, copy the facenet-python dir to /opt/nvidia/deepstream/deepstream-5.0/sources/deepstream_python_apps/apps
cd ./mtcnn_facenet_cpp_tensorRT
cp -r ./facenet-python /opt/nvidia/deepstream/deepstream-5.0/sources/deepstream_python_apps/apps

```
* Now, change the [paths](https://github.com/shubham-shahh/mtcnn_facenet_cpp_tensorRT/blob/baac7f037a0767f7061c075556937c1655fe0db8/facenet-python/deepstream_facenet.py#L326-L327) of required files based on your environment to start the app.

* [This](https://github.com/shubham-shahh/mtcnn_facenet_cpp_tensorRT/blob/361a1682c9a01ab0f8b974f3af486f12bbb0a96f/facenet-python/dstest2_sgie1_config.txt#L63) as well.

```bash
#Start the app
python3 deepstream_facenet.py ./testvideo.264

```

#### Output

```ini
shape:  128
128d tensor [-1.4865984  -1.544318    0.89401746 -0.8994489  -0.04300085  0.52093357
 -0.8916136   2.220441    0.8929736  -1.4589194   0.7189616  -0.6165253
  0.6620405   0.15181659 -1.2555692  -0.31640643  0.80598235 -0.64922345
 -0.32095382 -0.3466192   0.41663417 -0.31778926 -0.571472   -1.2705638
 -1.65505    -0.7065828  -0.24981631  1.6484964   0.5794762   1.2870077
 -1.315695    1.3969455  -2.1019044  -0.6780747   0.6097226   0.01213232
 -1.1178402  -0.66908723 -0.22235823  0.33602542  0.20067582 -0.83650416
 -0.69822705 -0.95611566  2.2626438  -0.7318402   1.4129282   0.9443468
  0.9543168   1.0541947  -2.1260738   1.7032665   0.14758769  0.95614445
 -2.9374108  -0.57548594 -0.312395    0.34166384 -1.8268638   0.9515188
 -1.4756488   0.40981948  0.3032776   2.4384832   0.288767    0.34942645
  1.5571721   0.25892335  1.2773193  -1.8331543  -0.06099812  0.8007121
 -0.02881579  1.3058838   0.06461819 -1.0351236  -0.05685008 -0.38556075
 -0.1014444  -0.12031181 -1.0232524  -1.2787757   0.57911897 -2.3480034
  0.24407476 -1.072583    0.3505367  -1.0544084  -1.1268206   0.2771215
  0.12124684  1.1293007  -1.2275653  -0.297546    0.7775245   2.8364015
 -1.1730605   0.7696141  -0.51775104 -0.00422744  1.1396134   1.8077573
  0.9840826  -0.24143988 -0.8211156   2.7129369   0.13263687 -1.031245
  0.23503277 -1.6717556   0.67445654 -0.2286004  -0.29565084 -0.14208874
  0.74103457  0.9470074   1.4978198   0.54924846 -2.0620053  -0.46697372
 -0.92504185  0.9490336   0.9313675  -0.52098674 -0.95104074  0.36103418
 -0.47283262 -0.3241306 ]
```

* [This](https://github.com/shubham-shahh/mtcnn_facenet_cpp_tensorRT/blob/baac7f037a0767f7061c075556937c1655fe0db8/facenet-python/deepstream_facenet.py#L223) line prints the 128d vector which is the output of the Facenet model. you can compare this vector with the the embeddings of known people and can infer if there is a match or not. 

* [This](https://www.pyimagesearch.com/2018/06/18/face-recognition-with-opencv-python-and-deep-learning/) tutorial shows how to make a pickle file of known embeddings and compare them with the ones recived from the model in real time.

* The current implementation supports .H264 files, if you want to use .mp4, RTSP, or USB webcams refer the deepstream-python-apps, in different test apps, different Gstream pipelines are demonstrated. This example is based on deepstream test-app-2


## C++ Implementation

* For running FaceNet with C++ Deepstream app, First test the sample app in its default configuration

```bash
#Move to sample app directory
cd /opt/nvidia/deepstream/deepstream-5.0/sources/apps/sample_apps/deepstream-infer-tensor-meta-test

#build
CUDA_VER = 10.2 make

#Run test app
./deepstream-infer-tensor-meta-app -t infer /opt/nvidia/deepstream/deepstream-5.0/samples/streams/sample_720p.mp4

```
* If the App runs fine, we can proceed with the next steps

```bash
#Make sure we are in the same directory
cd /opt/nvidia/deepstream/deepstream-5.0/sources/apps/sample_apps/deepstream-infer-tensor-meta-test

#create a backup folder
mkdir ./bkp

#move the sample app to the backup folder
mv ./deepstream_infer_tensor_meta_test.cpp ./bkp

#copy the custom app to this directory
cp /path/to/mtcnn_facenet_cpp_tensorRT/facenet-cpp/deepstream_infer_tensor_meta_test.cpp /opt/nvidia/deepstream/deepstream-5.0/sources/apps/sample_apps/deepstream-infer-tensor-meta-test

#copy the custom sgie file to the directory
cp /path/to/mtcnn_facenet_cpp_tensorRT/facenet-python/dstest2_sgie1_config.txt /opt/nvidia/deepstream/deepstream-5.0/sources/apps/sample_apps/deepstream-infer-tensor-meta-test

#move to target folder
cd
/opt/nvidia/deepstream/deepstream-5.0/sources/apps/sample_apps/deepstream-infer-tensor-meta-test

#Build with new changes
CUDA_VER = 10.2 make

#Run the App

./deepstream-infer-tensor-meta-app file:///home/ubuntu/video1.mp4

```
#### Output

```ini
Shape 128
128d Tensor
 0.267679 -0.251554 -0.260311 -0.223737 0.319461 0.119369 -0.0708687 -0.221111 0.267715 -0.0990431 0.141647 -0.297575 -0.142546 -0.526927 0.163986 -0.0697501 0.595039 -0.607409 0.244937 -0.193974 -0.268592 0.0620347 -0.845466 0.308341 0.0138658 -0.0437872 0.511274 0.0900756 -0.24564 0.108228 0.408589 0.172266 0.120605 -0.149561 0.227818 -0.182969 0.160609 0.0315109 0.545435 0.293824 0.155773 0.25595 -0.0702008 -0.366996 -0.782633 -0.414467 -0.046183 0.0648381 0.299499 0.0560513 0.326161 0.13931 0.529288 -0.562848 0.376559 0.338937 0.164113 -0.272101 -0.0265516 -0.552251 -0.194472 -0.344293 0.149408 0.960968 0.282589 0.415575 0.12719 0.00664066 0.0597391 -0.0656708 0.254978 -0.845805 0.623282 -0.711394 -0.207582 0.21915 0.0530824 0.540761 -0.331404 -0.380826 -0.508088 -0.118071 0.025195 0.608881 0.216518 -0.262486 0.522583 -0.626702 -0.64167 0.615544 0.364192 0.103187 -0.506685 -0.0775325 0.406819 -0.00199352 0.256417 -0.283961 -0.615214 0.268594 -0.229041 0.60271 0.223027 0.0621645 0.132152 -0.00342648 -0.176054 -0.0365441 -0.267923 -0.77549 0.175019 -0.0819169 0.398986 0.301 -0.029393 0.063587 -0.306683 0.719321 0.56422 0.190449 -0.0865007 -0.501071 0.14652 0.157381 0.187478 -0.450476 0.0641025 -0.0103367

```

* This app can support .Mp4 format and RTSP streams check the pipelines of other sample Deepstream Apps to run USB webcams.
```bash
#Run .MP4
./deepstream-test3-app file:///home/ubuntu/video1.mp4 file:///home/ubuntu/video2.mp4

#Run RTSP streams
./deepstream-test3-app rtsp://127.0.0.1/video1 rtsp://127.0.0.1/video2
```




