# Face Recognition for NVIDIA Jetson (Nano) using TensorRT
Face recognition with [Google FaceNet](https://arxiv.org/abs/1503.03832)
architecture and retrained model by David Sandberg
([github.com/davidsandberg/facenet](https://github.com/davidsandberg/facenet))
using TensorRT and OpenCV. <br> This project is based on the
implementation of l2norm helper functions which are needed in the output
layer of the FaceNet model. Link to the repo:
[github.com/r7vme/tensorrt_l2norm_helper](https://github.com/r7vme/tensorrt_l2norm_helper). <br>
Moreover, this project uses an adapted version of [PKUZHOU's implementation](https://github.com/PKUZHOU/MTCNN_FaceDetection_TensorRT)
of the mtCNN for face detection. More info below.

## Hardware
* NVIDIA Jetson Nano
* Raspberry Pi v2 camera 

If you want to use a USB camera instead of Raspi Camera set the boolean _isCSICam_ to false in [main.cpp](./src/main.cpp).


## Dependencies
cuda 10.0 + cudnn 7.5 <br> TensorRT 5.1.x <br> OpenCV 3.x <br>
TensorFlow r1.14 (for Python to convert model from .pb to .uff)

## Installation
#### 1. Install Cuda, CudNN, TensorRT, and TensorFlow for Python 
You can check [NVIDIA website](https://developer.nvidia.com/) for help.
Installation procedures are very well documented.<br><br>**If you are
using NVIDIA Jetson (Nano, TX1/2, Xavier) with Jetpack 4.2.2**, all needed packages
should be installed if the Jetson was correctly flashed using SDK
Manager, you will only need to install cmake and openblas:
```bash
sudo apt-get install cmake libopenblas-dev
```

#### 2. Prune and freeze TensorFlow model or get frozen model in the link
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

#### 3. Convert frozen protobuf (.pb) model to UFF
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
You should now have a facenet.uff (or similar) file which will be used
as the input model to TensorRT. <br>
The path to model is hardcoded, so please put the __facenet.uff__ in the
[facenetModels](./facenetModels) directory.


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
./mtcnn_facenet_cpp_tensorRT
```
Press "**Q**" to quit and to show the stats (fps).

_NOTE:_ This step might take a while when done the first time. TensorRT
now parses and serializes the model from .uff to a runtime engine
(.engine file). 

## Performance
Performance on **NVIDIA Jetson Nano**
* ~60ms +/- 20ms for face detection using mtCNN
* ~22ms +/- 2ms per face for facenet inference
* **Total:** ~15fps

Performance on **NVIDIA Jetson AGX Xavier**:
* ~40ms +/- 20ms for mtCNN 
* ~9ms +/- 1ms per face for inference of facenet
* **Total:** ~22fps
  
## License
Please respect all licenses of OpenCV and the data the machine learning models (mtCNN and Google FaceNet)
were trained on.



## Info
Niclas Wesemann <br>
[niclaswesemann@gmail.com](mailto:niclas.wesemann@gmail.com) <br>
