# Deepstream Facenet
This is demo is build on top of Python sample app [deepstream-test2](https://github.com/NVIDIA-AI-IOT/deepstream_python_apps/tree/master/apps/deepstream-test2) 
 - Replaced the PGIE with Peoplenet TLT model, and the SGIE1 with Facenet engine. 
 - removed the other two sgies. 
 - no changes regarding the tracker.

## Steps to run the demo:

- Generate the engine file for Facenet 
  - facenet_keras.h5 can be found in the models folder. The model is taken from [nyoki-mtl/keras-facenet](https://github.com/nyoki-mtl/keras-facenet)
  - Convert facenet model to TensorRT engine using [this jupyter notebook](https://github.com/riotu-lab/tf2trt_with_onnx). The steps in the jupyter notebook is taken  from [Nvidia official tutorial](https://developer.nvidia.com/blog/speeding-up-deep-learning-inference-using-tensorflow-onnx-and-tensorrt/).  
  **Note: converting to ONNX step must be done on the same machine that the model will work on**
  - when converting pb file to onnx use below command instead:
  `python -m tf2onnx.convert --input facenet.pb --inputs input_1:0[1,160,160,3] --inputs-as-nchw input_1:0 --outputs Bottleneck_BatchNorm/batchnorm_1/add_1:0 --output facenet.onnx`
- change the model-engine-file path to the your facenet engine file in `dstest2_sgie1_config.txt`.
- `python3 deepstream_test_2.py <h264_elementary_stream_contains_faces`
