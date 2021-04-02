import graphsurgeon
sg = graphsurgeon.StaticGraph('/home/jetson-tx2/code/onnx/models/facenet.pb')
print(sg.graph_inputs)
