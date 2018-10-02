import numpy as np
import time
from tensorface.mtcnn import detect_face, create_mtcnn
import tensorflow as tf

from tensorface.model import Face
from scipy import misc
from skimage import transform as trans
import cv2
from matplotlib import pyplot as plt

face_crop_margin = 32
face_crop_size = 160

model_path = '/home/ubuntu/insightface/face-recognition-evaluation/models'
def _setup_mtcnn():
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.2  
    config.gpu_options.allow_growth = True  
    with tf.Graph().as_default():
        sess = tf.Session(config=config)
        with sess.as_default():
            return create_mtcnn(sess,model_path)

pnet, rnet, onet = _setup_mtcnn()

def alignment(img,bb,landmark,image_size):
  M = None
  if landmark is not None:
    src = np.array([
      [30.2946, 51.6963],
      [65.5318, 51.5014],
      [48.0252, 71.7366],
      [33.5493, 92.3655],
      [62.7299, 92.2041] ], dtype=np.float32 )
    if image_size[1]==112:
      src[:,0] += 8.0
    dst = landmark.astype(np.float32)

    tform = trans.SimilarityTransform()
    tform.estimate(dst, src)
    M = tform.params[0:2,:]

  if M is None:
     ret = img[bb[1]:bb[3],bb[0]:bb[2],:]
     if len(image_size)>0:
        ret = cv2.resize(ret, (image_size[1], image_size[0]))
     return ret
  else:
     warped = cv2.warpAffine(img,M,(image_size[1],image_size[0]), borderValue = 0.0)
     return warped


def get_faces(image, threshold=0.5, minsize=20):
    threshold = [0.6, 0.7, 0.7] 
    factor = 0.709  
    faces = []
    idx = 0 
    bounding_boxes, points = detect_face(image, minsize, pnet, rnet, onet, threshold, factor)
    for bb in bounding_boxes:
        landmark = points[:, idx].reshape((2, 5)).T
        bbox = bb[0:4]
        img = alignment(image, bbox, landmark, (112,112))
        if face_crop_size != 112:
            img = misc.imresize(img, (face_crop_size, face_crop_size), interp='bilinear')
        faces.append(Face(*bb, img))
        idx += 1
    return faces
