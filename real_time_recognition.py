#coding=utf-8
import face_model
import argparse
import cv2
import sys
import numpy as np
import time
from cv2 import VideoWriter_fourcc

parser = argparse.ArgumentParser(description='face model test')
# general
parser.add_argument('--image-size', default='112,112', help='')
parser.add_argument('--model', default='/home/ubuntu/insightface/models/model-r100-ii/model,00', help='path to load model.')
parser.add_argument('--ga-model', default='', help='path to load model.')
parser.add_argument('--gpu', default=0, type=int, help='gpu id')
parser.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from begining')
parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')
args = parser.parse_args()

model = face_model.FaceModel(args)
video_capture = cv2.VideoCapture("video/1.mp4")
fourcc = VideoWriter_fourcc(*"XVID")
ret, frame = video_capture.read() 
out = cv2.VideoWriter("a.mp4", fourcc,10, (720,1280),True)

start = time.time()
embed_list = []

while ret:
    try:
       aligned = model.get_input(frame)
       embeds = model.get_feature(aligned)
       ret, frame = video_capture.read() 

       dist_euclidean = 0
       if embeds != None:
          for embed in embeds:
              embed_list.extend(embed.embedding)
              if len(embed_list) > 2:
                 dist_euclidean = np.sum((embed_list[-1] - embed_list[-2])**2)  
              
       if dist_euclidean < 0.7:
          print(dist_euclidean)
          out.write(frame)
       else:
          out = cv2.VideoWriter("b.mp4", fourcc,10, (720,1280),True)
          

    except:
       break
    if cv2.waitKey(1) & 0xFF == ord('q'):    
       break

end = time.time() - start
print("end-time"+":"+str(end))
video_capture.release()
cv2.destroyAllWindows()
