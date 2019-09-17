# mtcnn_facenet_cpp_tensorRT
Face recognition with [Google FaceNet](https://arxiv.org/abs/1503.03832)
architecture and retrained model by David Sandberg
([github.com/davidsandberg/facenet](https://github.com/davidsandberg/facenet))
using TensorRT and OpenCV. <br> This project is based on the
implementation of l2norm helper functions which are needed in the output
layer of the FaceNet model. Link to the repo:
[github.com/r7vme/tensorrt_l2norm_helper](https://github.com/r7vme/tensorrt_l2norm_helper)

## Dependencies
cuda 10.0 + cudnn 7.5 <br> TensorRT 5.1.x <br> OpenCV 3.x <br>
TensorFlow r1.14 (for Python to convert model from .pb to .uff)

## Installation
#### 1. Install Cuda, CudNN, TensorRT, and TensorFlow for Python 
You can check [NVIDIA website](https://developer.nvidia.com/) for help.
Installation procedures are very well documented.<br><br>**If you are
using NVIDIA Jetson AGX Xavier with Jetpack 4.2.2**, all needed packages
should be installed if the Xavier was correctly flashed using SDK
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
saved model <br>
[https://github.com/apollo-time/facenet/raw/master/model/resnet/facenet.pb]

#### 3. Convert frozen protobuf (.pb) model to UFF
Use the convert-to-uff tool which is installed with tensorflow
installation to convert the *.pb model to *.uff. The script will replace
unsupported layers with custom layers implemented by
[github.com/r7vme/tensorrt_l2norm_helper](https://github.com/r7vme/tensorrt_l2norm_helper).
Please check the file for the user defined values and update them if
needed. Do not worry if there are a few warnings about the
TRT_L2NORM_HELPER plugin.
```bash
cd /path/to/project
python3 step01_pb_to_uff.py
```
You should now have a facenet.uff (or similar) file which will be used
as the input model to TensorRT.

#### 4. Build the project
WARNING: This step might take a while when done the first time. TensorRT
now parses and serializes the model from .uff to a runtime engine
(.engine file). 
```bash
mkdir build && cd build
cmake \
-DCMAKE_BUILD_TYPE=Release \
-DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-10.0/ \
-DTENSORRT_ROOT=/usr/ ..
make -j${nproc}
```
If **not** run on Xavier update the path to your TensorRT installation.

## NOTE
**.uff and .engine files are GPU specific**, so if you use want to run
this project on a different GPU or on another machine, always start over
at step **3.** above.

## Usage
 
```bash
./mtcnn_facenet_cpp_tensorRT
```


## Notes
**Performance** on NVIDIA Jetson Xavier:
* ~40ms +/- 20ms for mtCNN 
* ~9ms +/- 1ms per face for inference of facenet <br><br> **TOTAL:**
  ~22fps with ~13% GPU usage

## ToDo
*
* how to get acquainted to new people while algorithm is running
* database of embeddings not the actual pictures

## Info
Niclas Wesemann <br>
[niclas.wesemann@gmail.com](mailto:niclas.wesemann@gmail.com) <br>
August 2019