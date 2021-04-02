import onnx

import argparse


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input_path", type=str, default = "./onnxmodel/facenetconv.onnx",
	help="path to input model")
ap.add_argument("-o", "--output_path", type=str, default = "./dynamiconnxmodel/dynamiconnxmodel.onnx",
	help="path to input model")
args = vars(ap.parse_args())


model = onnx.load(args["input_path"])
model.graph.input[0].type.tensor_type.shape.dim[0].dim_param = '?'
onnx.save(model, args["output_path"])
