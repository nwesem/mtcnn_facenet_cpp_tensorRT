from PIL import Image
import numpy as np
from mtcnn.mtcnn import MTCNN

import engine as eng
import inference as inf
import keras
import tensorrt as trt 
from utils import *

TRT_LOGGER = trt.Logger(trt.Logger.INTERNAL_ERROR)
trt_runtime = trt.Runtime(TRT_LOGGER)

engine_path = "./facenet_engine.plan"
input_file_path = 'pat/to/your/testimage.jpg'
dataset_embeddings_path = '/path/to/your/dataset'
HEIGHT = 160
WIDTH = 160


# load dataset embeddings
#dataset_embeddings = np.load(dataset_embeddings_path)
#faces_embeddings, labels = dataset_embeddings['arr_0'], dataset_embeddings['arr_1']
# Normalize dataset embeddings
#faces_embeddings = normalize_vectors(faces_embeddings)

detector = MTCNN()

face_array = extract_face_from_image(input_file_path, detector)

face_pixels = face_array
# scale pixel values
face_pixels = face_pixels.astype('float32')
# standardize pixel values across channels (global)
mean, std = face_pixels.mean(), face_pixels.std()
face_pixels = (face_pixels - mean) / std
# transform face into one sample
samples = np.expand_dims(face_pixels, axis=0)
# make prediction to get embedding


engine = eng.load_engine(trt_runtime, engine_path)

h_input, d_input, h_output, d_output, stream = inf.allocate_buffers(engine, 1, trt.float32)
yhat = inf.do_inference(engine, samples, h_input, d_input, h_output, d_output, stream, 1, HEIGHT, WIDTH)

print(yhat.shape)

face_to_predict_embedding = normalize_vectors(yhat)
#result = predict_using_min_l2_distance(faces_embeddings, labels, face_to_predict_embedding)
print(face_to_predict_embedding)
print('Predicted name: %s' % (str(result).title()))


