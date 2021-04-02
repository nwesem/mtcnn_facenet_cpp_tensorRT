from sklearn.preprocessing import Normalizer
from PIL import Image
import numpy as np

def normalize_vectors(vectors):
    # normalize input vectors
    normalizer = Normalizer(norm='l2')
    vectors = normalizer.transform(vectors)

    return vectors

def predict_using_min_l2_distance(faces_embeddings, labels, face_to_predict_embedding):

    #Given a list of face encodings, compare them to a known face encoding and get a euclidean distance
    #for each comparison face. The distance tells you how similar the faces are.
    face_distance = np.linalg.norm(faces_embeddings - face_to_predict_embedding, axis=1)
    
    #print(face_distance.shape)
    index = np.argmin(face_distance)
    #print(str(index), ' prdicted name: ',  labels[index])

    return labels[index]
    
def extract_face_from_image(input_file_path, detector):
	# load image from file
	image = Image.open(input_file_path)
	# convert to RGB, if needed
	image = image.convert('RGB')
	# convert to array
	pixels = np.asarray(image)
	# detect faces in the image
	results = detector.detect_faces(pixels)

	# extract the bounding box from the first face
	x1, y1, width, height = results[0]['box']
	# bug fix
	x1, y1 = abs(x1), abs(y1)
	x2, y2 = x1 + width, y1 + height
	# extract the face
	face = pixels[y1:y2, x1:x2]
	# resize pixels to the model size
	image = Image.fromarray(face)
	image = image.resize((160, 160))
	face_array = np.asarray(image)
	
	return face_array
