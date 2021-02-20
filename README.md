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
First, we make some changes so that we can use YOLO with DeepStream. Either provide the deepstream folder the read and write permissions or use the commangs with sudo
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


Download the weights, cfg and labels files from [this](https://github.com/lthquy/Yolov3-tiny-Face-weights) repo. place the .weights and .cfg file in /opt/nvidia/deepstream/deepstream-5.0/sources/objectDetector_Yolo, after that make a labels.txt file consisting of one class 'Face'.

if the below mentioned files are already existing in your workspace please move them to /bkp and move these files in /objectDetector_Yolo
```bash
cd /opt/nvidia/deepstream/deepstream-5.0/sources/objectDetector_Yolo

#clone develop branch of this repo
git clone -b develop --single-branch https://github.com/shubham-shahh/mtcnn_facenet_cpp_tensorRT.git

#copy config files for reference
cd ./mtcnn_facenet_cpp_tensorRT/YoloApp
cp ./deepstream_app_config_yoloV3_tiny.txt /opt/nvidia/deepstream/deepstream-5.0/sources/objectDetector_Yolo
cp ./config_infer_primary_yoloV3_tiny.txt /opt/nvidia/deepstream/deepstream-5.0/sources/objectDetector_Yolo

```
add names or paths of the respective files based on your config [here](https://github.com/shubham-shahh/mtcnn_facenet_cpp_tensorRT/blob/81a3cad4efa76eea9f98e96dfd5540f341107068/YoloApp/config_infer_primary_yoloV3_tiny.txt#L65-L68) once everything is in place we can test our app.

```bash
#Test if we properly configured YOLO to run with Deepstream
cd /opt/nvidia/deepstream/deepstream-5.0/sources/objectDetector_Yolo
deepstream-app -c ./deepstream_app_config_yoloV3_tiny.txt
```
change the [source](https://github.com/shubham-shahh/mtcnn_facenet_cpp_tensorRT/blob/81a3cad4efa76eea9f98e96dfd5540f341107068/YoloApp/deepstream_app_config_yoloV3_tiny.txt#L47) in case you want to test on some other video

## Prepare FaceNet
This step relies on [this]() repo. I couldn't make things work using this repo. for more info follow [this](https://github.com/riotu-lab/deepstream-facenet/issues/5) issue hence I've made some modifications to keep things rolling.


First, download the keras model of Facenet from [this](https://github.com/nyoki-mtl/keras-facenet) fork and place it in /mtcnn_facenet_cpp_tensorRT/ModelConverion/KerasModel

```bash
#Test if we properly configured YOLO to run with Deepstream
cd ./mtcnn_facenet_cpp_tensorRT/ModelConversion
python3 h5topb.py --input_path ./kerasmodel/facenet_keras_128.h5 --output_path ./tensorflowmodel/facenet.pb

#Convert Tensorflow model to an ONNX model. This will take approx 50 mins and this has to be done on the host device
python3 -m tf2onnx.convert --input ./tensorflowmodel/facenet_freezed.pb --inputs input_1:0[1,160,160,3] --inputs-as-nchw input_1:0 --outputs Bottleneck_BatchNorm/batchnorm_1/add_1:0 --output onnxmodel/facenetconv.onnx

#Convert ONNX model to a model that can take dynamic input size
python3 dynamic_conv.py --input_path ./onnxmodel/facenetconv.onnx --output_path ./dynamiconnxmodel/dynamicfacenetmodel.onnx

```
Once we have the dynamic onnx model copy it to /mtcnn_facenet_cpp_tensorRT/facenet-python/facenetmodel
```bash
cd ./mtcnn_facenet_cpp_tensorRT/ModelConversion/dynamiconnxmodel/
cp ./dynamicfacenetmodel.onnx  /path/to/mtcnn_facenet_cpp_tensorRT/facenet-python/facenetmodel

#Now, copy the facenet-python model to /opt/nvidia/deepstream/deepstream-5.0/sources/deepstream_python_apps/apps
cd ./mtcnn_facenet_cpp_tensorRT
cp -r ./facenet-python /opt/nvidia/deepstream/deepstream-5.0/sources/deepstream_python_apps/apps

```









page under maintainence

cp -r /opt/nvidia/deepstream/deepstream-5.0/sources/deepstream_python_apps/apps/tf2trt_with_onnx/test /opt/nvidia/deepstream/deepstream-5.0/sources/deepstream_python_apps/apps
