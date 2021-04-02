from onnx_to_trt import create_engine

ONNX_PATH = './facenet.onnx'
TRT_ENGINE_PATH = './facenet_engine.plan'

create_engine(ONNX_PATH, TRT_ENGINE_PATH)
