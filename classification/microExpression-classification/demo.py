#--encoding:utf-8--#
import cv2
import numpy as np
import sys
import torch
import PIL.Image as Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.nn import functional as F
import Classifier
from torch.autograd import Variable
#from model import *

def format_face(frame):
	# convert image to grey since our train data is grey pictures with 48*48 shape
	image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	# detect all the faces, and save the faces' coordinate, size in vector
	# scaleFactor is an hyperparameter, suggest to use 1.1
	faces = cascade_classifier.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5)
	
	# if no faces, return None
	if not len(faces) > 0:
		return None, None
	# initialize max area face
	max_area_face = faces[0]
	for face in faces:
		if face[2]*face[3] > max_area_face[2] * max_area_face[3]:
			max_area_face = face
	
	# cut face_image
	face_image = image[max_area_face[1]:(max_area_face[1] + max_area_face[2]),
				  max_area_face[0]:(max_area_face[0] + max_area_face[3])]
	try:
		face_image = cv2.resize(face_image, (48, 48), interpolation = cv2.INTER_CUBIC)
	except Exception:
		print("resize error")
		return None, None
	return face_image, max_area_face

def demo(model, EMOTIONS, feeling_faces, showBox):
	# get video capture
	# '0' means open the built-in camera of laptop, 'video file path' means open the video
	video_captor = cv2.VideoCapture(0)
	result = None
	while True:
		# get camera's picture per frame
		# if get successfully, ret = True.
		ret, frame = video_captor.read()
		detected_face, face_coor = format_face(frame)
		if showBox:
			# get face coordinate, use rectangel box to show it
			if face_coor is not None:
				[x, y, w, h] = face_coor
				cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)
			
		# update per 10ms, and capture the frame when press 'a'
		# since we run in a 64-bit system, we must use 0xFF == ord('a)
		if cv2.waitKey(10) & 0xFF == ord('a'):
			if detected_face is not None:
				print('get picture successfully')
				
				# transfer face picture to tensor
				detected_face = detected_face.astype(np.uint8)
				 
				# transfer image
				test_transform = transforms.Compose([
								transforms.ToPILImage(),
								transforms.ToTensor(),
							])
				
				detected_face_tensor = test_transform(detected_face)
				# ATTENTION here: since the model needs a 4-dimension input which the first dimension indicates to batch size
				# When predict, we only have 1 picture as input. so we need to add one dimension by unsqueeze after transfer the picture to tensor
				detected_face_tensor = torch.unsqueeze(detected_face_tensor, 0)
				model.eval()
				
				with torch.no_grad():
					result = model(detected_face_tensor)
				_, predicted = torch.max(result, 1) # get the classification result
				print('result:', result)
				p_result = F.softmax(result, dim=1)
				print('softmax:', p_result)
		if result is not None:
			for index, emotion in enumerate(EMOTIONS):
				# add emotion labels to frame
				cv2.putText(frame, emotion, (10, index*20 + 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
				# add predicted probability under emotion labels
				cv2.rectangle(frame, (130, index*20 + 10), (130+int(p_result[0][index] *100), (index+1)*20+4), (255, 0, 0), -1)
			# get corresponding emoji face
			classIndex = predicted[0]
			emotion = EMOTIONS[classIndex]
			#print('emotion:',emotion)
			emoji_face = feeling_faces[classIndex]
			
			# add emoji face to frame
			frame[200:320, 10:130, :] = emoji_face[:, :, :]
		
		cv2.imshow('face', frame)
		if cv2.waitKey(10) & 0xFF == ord('q'):
			break
		
	video_captor.release()
	cv2.destroyAllWindows()
		
	

	
if __name__=='__main__':
	# load opencv face recognizer
	CASC_PATH = '/Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages/cv2/data/haarcascade_frontalface_alt2.xml'
	cascade_classifier = cv2.CascadeClassifier(CASC_PATH)
	
	# load model
	#model = torch.load('model_best.pkl')
	model = Classifier.Classifier()
	model.load_state_dict(torch.load('model_best_para.pkl',map_location=torch.device('cpu')))
	
	# y labels
	EMOTIONS = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']
	
	# load emoji faces for expressing y label
	feeling_faces = []
	for emotion in EMOTIONS:
		img = cv2.imread('Microexpression_recognition-master/data/emojis/'+ emotion + '.png', 1)
		feeling_faces.append(cv2.resize(img, (120, 120)))
	
	#
	print('yes')
	demo(model, EMOTIONS, feeling_faces, True)
