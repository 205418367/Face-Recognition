import numpy as np
import detection
import cv2
from insightface import Embedding

embedding = Embedding('models/model-r100-ii/model',0)

def identify(img):
    faces = detection.get_faces(img, 0.6, 50)
    face = faces[0]
    #选择最大人脸
    maxbox = 0
    for temp in faces:
        img_size = np.asarray(img.shape)[0:2]
        temp.x1 = np.maximum(temp.x1, 0)
        temp.y1 = np.maximum(temp.y1, 0)
        temp.x2 = np.minimum(temp.x2, img_size[1])
        temp.y2 = np.minimum(temp.y2, img_size[0])
        boxarea = (temp.y2 - temp.y1) * (temp.x2 - temp.x1)
        if boxarea > maxbox:
            maxbox = boxarea
            face = temp
    f = embedding.get(face.img)
    face.embedding = f 
    # 提取所有人脸特征
    # for face in faces:
    #     f = embedding.get(face.img)
    #     face.embedding = f
    return [face,]
