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

#### 2. Install Tensorflow
The following shows the steps to install Tensorflow for Jetpack 4.4. This was copied from the official [NVIDIA documentation](https://docs.nvidia.com/deeplearning/frameworks/install-tf-jetson-platform/index.html). I'm assuming you don't need to install it in a virtual environment. If yes, please refer to the documentation linked above. If you are not installing this on a jetson, please refer to the official tensorflow documentation.

```bash
# Install system packages required by TensorFlow:
sudo apt update
sudo apt install libhdf5-serial-dev hdf5-tools libhdf5-dev zlib1g-dev zip libjpeg8-dev liblapack-dev libblas-dev gfortran

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


#### 3. Prepare Face detector
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


#Download the weights, cfg and labels files from [this](https://github.com/lthquy/Yolov3-tiny-Face-weights) repo. place the .weights and .cfg file in /opt/nvidia/deepstream/deepstream-5.0/sources/objectDetector_Yolo, after that make a labels.txt file consisting of one class 'Face'.

#if these files are already existing in your workspace please move them to /bkp and move these files in /objectDetector_Yolo
cd /opt/nvidia/deepstream/deepstream-5.0/sources/objectDetector_Yolo
git clone -b develop --single-branch https://github.com/shubham-shahh/mtcnn_facenet_cpp_tensorRT.git
cd ./mtcnn_facenet_cpp_tensorRT
cp ./deepstream_app_config_yoloV3_tiny.txt /opt/nvidia/deepstream/deepstream-5.0/sources/objectDetector_Yolo
cp ./config_infer_primary_yoloV3_tiny.txt /opt/nvidia/deepstream/deepstream-5.0/sources/objectDetector_Yolo







```




page under maintainence