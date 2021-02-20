# tf2trt_wtih_onnx
This repo documnet how to convert Tensorflow / Keras model to TRT engine using ONNX.  

## Deprication of Caffe Parser and UFF Parser in TensorRT 7
Note this quote from the [official TensorRT Release Notes](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-700/tensorrt-release-notes/tensorrt-7.html#rel_7-0-0):

> Deprecation of Caffe Parser and UFF Parser - We are deprecating Caffe Parser and UFF Parser in TensorRT 7. They will be tested and functional in the next major release of TensorRT 8, but we plan to remove the support in the subsequent major release. Plan to migrate your workflow to use tf2onnx, keras2onnx or TensorFlow-TensorRT (TF-TRT) for deployment.

In this repository, we will use [tf2onnx](https://github.com/onnx/tensorflow-onnx) to convert Keras model to TRT engine.  

## ONNX Workflow 

![ONNX-workflow-1024x195](https://user-images.githubusercontent.com/13350394/96366303-8d8f4f00-114f-11eb-9a41-994807beb1aa.jpg)
*ONNX Workflow - [image source](https://developer.nvidia.com/blog/speeding-up-deep-learning-inference-using-tensorflow-onnx-and-tensorrt/)*.   

1. Convert the TensorFlow/Keras model to a .pb file.
2. Convert the .pb file to the ONNX format.
3. Create a TensorRT engine.
4. Run inference from the TensorRT engine.

## Jupyter Notebook
The steps are documented in [this Jupyter notebook](convert_tf_keras_model_to_trt_using_onnx.ipynb).

## Known Issues
**Protobuf compiler not found while installing tf2onnx tool**  
![image](https://user-images.githubusercontent.com/13350394/98991122-72401580-253c-11eb-9905-4512f1ad393b.png)

**Solution**  
`sudo apt-get install  libprotobuf-dev protobuf-compiler`
