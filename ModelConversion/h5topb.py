from keras_to_pb_tf2  import keras_to_pb
from keras.models import load_model
import argparse


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input_path", type=str, default = "./kerasmodel/facenet_keras_128.h5",
	help="path to input model")
ap.add_argument("-o", "--output_path", type=str, default = "./tensorflowmodel/facenet.pb",
	help="path to input model")
args = vars(ap.parse_args())

model = load_model(args["input_path"])
input_name, output_node_names = keras_to_pb(model, args["output_path"], None)






