import tensorflow as tf
from tensorflow.compat.v1 import graph_util
from tensorflow.keras.models import Model
from tensorflow.python.keras import backend as K
from tensorflow.keras.models import load_model
import argparse

tf.compat.v1.disable_eager_execution()
K.set_learning_phase(0)

def keras_to_pb(model, output_filename, output_node_names):

   """
   This is the function to convert the Keras model to pb.

   Args:
      model: The Keras model.
      output_filename: The output .pb file name.
      output_node_names: The output nodes of the network. If None, then
      the function gets the last layer name as the output node.
   """

   # Get the names of the input and output nodes.
   in_name = model.layers[0].get_output_at(0).name.split(':')[0]

   if output_node_names is None:
       output_node_names = [model.layers[-1].get_output_at(0).name.split(':')[0]]

   sess = K.get_session()

   # The TensorFlow freeze_graph expects a comma-separated string of output node names.
   output_node_names_tf = ','.join(output_node_names)

   frozen_graph_def = graph_util.convert_variables_to_constants(
       sess,
       sess.graph.as_graph_def(),
       output_node_names)

   sess.close()
   wkdir = ''
   tf.io.write_graph(frozen_graph_def, wkdir, output_filename, as_text=False)

   return in_name, output_node_names

def main(args):
    # load ResNet50 model pre-trained on imagenet
    model = load_model(args.model_path)

    # Convert keras ResNet50 model to .bp file
    in_tensor_name, out_tensor_names = keras_to_pb(model, args.output_pb_file , None) 

if __name__ == 'main':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='facenet_keras.h5')
    parser.add_argument('--output_pb_file', type=str, default='facenet.pb')
    args=parser.parse_args()
    main(args)
