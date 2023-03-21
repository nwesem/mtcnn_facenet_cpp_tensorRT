#!/usr/bin/env python3
import graphsurgeon as gs
import tensorflow as tf
import uff

if __name__ == "__main__":
  # USER DEFINED VALUES
  output_nodes = ["Bottleneck/BatchNorm/batchnorm/add_1"]
  input_node   = "input"
  pb_file      = "./facenet.pb"
  uff_file     = "./facenetModels/facenet.uff"
  # END USER DEFINED VALUES

  # read tensorflow graph
  dynamic_graph = gs.DynamicGraph(pb_file)
  # write UFF to file
  uff_model = uff.from_tensorflow(dynamic_graph.as_graph_def(), output_nodes=output_nodes, output_filename=uff_file, text=False)