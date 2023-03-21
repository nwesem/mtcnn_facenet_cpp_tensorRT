# Face Recognition for NVIDIA Jetson AGX Orin using TensorRT
- This project is based on the implementation of this repo:
[Face Recognition for NVIDIA Jetson (Nano) using TensorRT](https://github.com/nwesem/mtcnn_facenet_cpp_tensorRT). Since the original author is no longer updating his content, and many of the original content cannot be applied to the new Jetpack version and the new Jetson device. Therefore, I have modified the original author's content slightly to make it work for face recognition on the Jetson AGX Orin.
- Face recognition with [Google FaceNet](https://arxiv.org/abs/1503.03832) architecture and retrained model by David Sandberg ([github.com/davidsandberg/facenet](https://github.com/davidsandberg/facenet)) using TensorRT and OpenCV.
- Moreover, this project uses an adapted version of [PKUZHOU's implementation](https://github.com/PKUZHOU/MTCNN_FaceDetection_TensorRT)
of the mtCNN for face detection. More info below.

## Hardware
- Nvidia Jetson AGX Orin DVK
- Logitech C922 Pro HD Stream Webcam

If you want to use a CSI camera instead of USB Camera, set the boolean _isCSICam_ to true in [main.cpp](./src/main.cpp).


## Dependencies
- JetPack 5.1
- CUDA 11.4.19 + cuDNN 8.6.0
- TensorRT 8.5.2
- OpenCV 4.5.4
- Tensorflow 2.11


## Installation

#### 1. Install Tensorflow
The following shows the steps to install Tensorflow for Jetpack 5.1. This was copied from the official [NVIDIA documentation](https://docs.nvidia.com/deeplearning/frameworks/install-tf-jetson-platform/index.html). I'm assuming you don't need to install it in a virtual environment. If yes, please refer to the documentation linked above. If you are not installing this on a jetson, please refer to the official tensorflow documentation.

```bash
# Install system packages required by TensorFlow:
sudo apt update
sudo apt install libhdf5-serial-dev hdf5-tools libhdf5-dev zlib1g-dev zip libjpeg8-dev liblapack-dev libblas-dev gfortran

# Install and upgrade pip3
sudo apt install python3-pip
sudo python3 -m pip install --upgrade pip
sudo pip3 install -U testresources setuptools==65.5.0

# Install the Python package dependencies
sudo pip3 install -U numpy==1.22 future==0.18.2 mock==3.0.5 keras_preprocessing==1.1.2 keras_applications==1.0.8 gast==0.4.0 protobuf pybind11 cython pkgconfig packaging h5py==3.6.0

# Install TensorFlow using the pip3 command. This command will install the latest version of TensorFlow compatible with JetPack 5.1.
sudo pip3 install --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v51 tensorflow==2.11.0+nv23.01
```


#### 3. Prune and freeze TensorFlow model or get frozen model in the link
The inputs to the original model are an input tensor consisting of a
single or multiple faces and a phase train tensor telling all batch
normalisation layers that model is not in train mode. Batch
normalisation uses a switch layer to decide if the model is currently
trained or just used for inference. This switch layer cannot be
processed in TensorRT which is why it needs to be removed. Apparently
this can be done using freeze_graph from TensorFlow, but here is a link
to model where the phase train tensor has already been removed from the
saved model
[github.com/apollo-time/facenet/raw/master/model/resnet/facenet.pb](https://github.com/apollo-time/facenet/raw/master/model/resnet/facenet.pb)

#### 4. Convert frozen protobuf (.pb) model to UFF
Use the convert-to-uff tool which is installed with tensorflow
installation to convert the *.pb model to *.uff. The script will replace
unsupported layers with custom layers implemented by
[github.com/r7vme/tensorrt_l2norm_helper](https://github.com/r7vme/tensorrt_l2norm_helper).
Please check the file for the user defined values and update them if
needed. Do not worry if there are a few warnings about the
TRT_L2NORM_HELPER plugin.
```bash
cd path/to/project
python3 step01_pb_to_uff.py
```
You should now have a facenet.uff file in the [facenetModels folder](./facenetModels) which will be used as the input model to TensorRT. <br>


#### 4. Get mtCNN models
This repo uses an [implementation by PKUZHOU](https://github.com/PKUZHOU/MTCNN_FaceDetection_TensorRT)
of the [multi-task Cascaded Convolutional Neural Network (mtCNN)](https://arxiv.org/pdf/1604.02878.pdf)
for face detection. The original implementation was adapted to return the bounding boxes such that it
can be used as input to my FaceNet TensorRT implementation.
You will need all models from the repo in the [mtCNNModels](./mtCNNModels) folder so please do this 
to download them:
```bash
# go to one above project,
cd path/to/project/..
# clone PKUZHOUs repo,
git clone https://github.com/PKUZHOU/MTCNN_FaceDetection_TensorRT
# and move models into mtCNNModels folder
mv MTCNN_FaceDetection_TensorRT/det* path/to/project/mtCNNModels
```
After doing so you should have the following files in your [mtCNNModels](./mtCNNModels) folder:<br>
* det1_relu.caffemodel
* det1_relu.prototxt
* det2_relu.caffemodel
* det2_relu.prototxt
* det3_relu.caffemodel
* det3_relu.prototxt
* README.md

Done you are ready to build the project!

#### 5. Build the project
```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j${nproc}
```
If **not** run on Jetson platform set the path to your CUDA and TensorRT installation
using _-DCUDA_TOOLKIT_ROOTDIR=path/to/cuda_ and _-DTENSORRT_ROOT=path/to/tensorRT_.

## NOTE
**.uff and .engine files are GPU specific**, so if you use want to run
this project on a different GPU or on another machine, always start over
at step **3.** above.

## Usage
Put images of people in the imgs folder. Please only use images that contain one face.<br>
**NEW FEATURE**:You can now add faces while the algorithm is running. When you see
the OpenCV GUI, press "**N**" on your keyboard to add a new face. The camera input will stop until
you have opened your terminal and put in the name of the person you want to add.
```bash
./face_recogition_tensorRT
```
Press "**Q**" to quit and to show the stats (fps).

_NOTE:_ This step might take a while when done the first time. TensorRT
now parses and serializes the model from .uff to a runtime engine
(.engine file). 

## Performance
Performance on **NVIDIA Jetson AGX Orin**
* ~24ms for face detection using mtCNN
* ~4ms per face for facenet inference
* **Total:** ~30fps
  
## License
Please respect all licenses of OpenCV and the data the machine learning models (mtCNN and Google FaceNet)
were trained on.

