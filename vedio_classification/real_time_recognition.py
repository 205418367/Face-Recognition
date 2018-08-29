#coding=utf-8
import face_model
import argparse
import cv2
import sys,os
import numpy as np
import time
from cv2 import VideoWriter_fourcc
import uuid

parser = argparse.ArgumentParser(description='face model test')
# general
parser.add_argument('--image-size', default='112,112', help='')
parser.add_argument('--model', default='/home/ubuntu/insightface/models/model-r100-ii/model,00', help='path to load model.')
parser.add_argument('--gpu', default=0, type=int, help='gpu id')
parser.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from begining')
parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')
args = parser.parse_args()

model = face_model.FaceModel(args)
video_capture = cv2.VideoCapture("video/1.mp4")
fourcc = VideoWriter_fourcc(*"XVID")
ret, frame = video_capture.read() 
size = frame.shape[1],frame.shape[0]
out = cv2.VideoWriter("0.avi", fourcc,10, size,True)

start = time.time()
embed_list = []
dist = 0

while ret:
    try:
       aligned = model.get_input(frame) 
       faces = model.get_feature(aligned)
       ret, frame = video_capture.read() 

       for face in faces:
           if face.dist:
              dist = face.dist
              if dist < 0.4:
                 label = face.label
                 if os.path.isdir("video/result/"+str(label)):
                    print("new!@@@@@@")
                    out = cv2.VideoWriter("{0}/{1}.avi".format("video/result/"+str(label), str(uuid.uuid1())), fourcc,10, size,True)
                 else:
                    os.makedirs("video/result/"+str(label))
                    print("new!$$$$$$")
                    out = cv2.VideoWriter("{0}/{1}.avi".format("video/result/"+str(label), str(uuid.uuid1())), fourcc,10, size,True)

       if dist > 0:
          out.write(frame)

    except:
       break
    if cv2.waitKey(1) & 0xFF == ord('q'):    
       break

end = time.time() - start
print("end-time"+":"+str(end))
video_capture.release()
cv2.destroyAllWindows()
