# coding=utf-8
"""实施人脸距离匹配--"""
#加入人脸对齐的人脸检测
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
import detect_face 
import facenet
import argparse
import math,pickle,cPickle
from scipy import misc
from six.moves import xrange
import os,sys,h5py, time, glob 
import face_preprocess
import cv2


facenet_model_checkpoint = "/home/ubuntu/Additive-Margin-Softmax-master/cbir/model_check_points/20170511-185253.pb"
mtcnn_model_checkpoint = "/home/ubuntu/Additive/cbir/model_check_points"


class Face:
    def __init__(self):
        self.label = None
        self.dist = None
        self.bounding_box = None
        self.container_image = None
        self.embedding = None 
        self.score = None
        self.img_face = None

class Recognition:
    def __init__(self):
        self.detect = Detection()
        self.encoder = Encoder()  

    def identify(self,image,feats,labels):
        if image.ndim == 2:
           image = facenet.to_rgb(image)
        image = image[:,:,0:3] 
        faces = self.detect.find_faces(image)
        for face in faces:
            embedding = self.encoder.generate_embedding(face)
            face.dist,face.label = self.encoder.compute_euclideanDistance(embedding,feats,labels)  
        return faces

#特征提取类
class Encoder:
    def __init__(self):
        self.sess = tf.Session()
        with self.sess.as_default():
             facenet.load_model(facenet_model_checkpoint)

    def compute_euclideanDistance(self,embedding,feats,labels):        
        dists = ((embedding - feats)**2).sum(axis=1)
        idx = np.argsort(dists)[:1]
        rank_dists = dists[idx]
        rank_labels = [labels[k] for k in idx]
        return rank_dists,rank_labels
   
    def compute_yuxianDistance(self,embedding,feats,labels,names):   
        dot = np.sum(np.multiply(embedding, feats), axis=1)
        norm = np.linalg.norm(embedding) * np.linalg.norm(feats, axis=1)
        similarity = dot / norm
        dists = np.arccos(similarity) / math.pi
        idx = np.argsort(dists)[:]
        rank_dists = dists[idx]
        rank_labels = [labels[k] for k in idx]
        rank_names = [names[k] for k in idx]
        return rank_dists,rank_labels,rank_names

    #提取特征      
    def generate_embedding(self, face):
        # Get input and output tensors
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        
        #face.image = facenet.left_right(face.image)
        prewhiten_face = facenet.prewhiten(face.image)
        #Run forward pass to calculate embeddings
        feed_dict = {images_placeholder: [prewhiten_face], phase_train_placeholder: False}
        return self.sess.run(embeddings, feed_dict=feed_dict)[0]
 
#人脸检测类
class Detection:  
    minsize = 20  
    threshold = [0.6, 0.7, 0.7]
    factor = 0.709 
 
    def __init__(self, image_size ='112,112',margin = 44):
        self.pnet, self.rnet, self.onet = self._setup_mtcnn()
        self.image_size = image_size
        self.margin = margin

    def _setup_mtcnn(self):    
        with tf.Graph().as_default():
            sess = tf.Session()
            with sess.as_default():
                return detect_face.create_mtcnn(sess, mtcnn_model_checkpoint)
 
    def find_faces(self, img): 
        faces = []

        bounding_boxes,points = detect_face.detect_face(img, self.minsize,self.pnet,self.rnet,self.onet,self.threshold,self.factor)
        nrof_faces = bounding_boxes.shape[0]
        if nrof_faces>0:
          detect_multiple_faces = True
          det = bounding_boxes[:,0:4]
          det_arr = []
          img_size = np.asarray(img.shape)[0:2]

          if nrof_faces>1:
             if detect_multiple_faces:
                for i in range(nrof_faces):
                    det_arr.append(np.squeeze(det[i]))
             else:
                bounding_box_size = (det[:,2]-det[:,0])*(det[:,3]-det[:,1])
                img_center = img_size / 2
                offsets = np.vstack([(det[:,0]+det[:,2])/2-img_center[1], (det[:,1]+det[:,3])/2-img_center[0] ])
                offset_dist_squared = np.sum(np.power(offsets,2.0),0)
                index = np.argmax(bounding_box_size-offset_dist_squared*2.0)
                det_arr.append(det[index,:])
          else:
                det_arr.append(np.squeeze(det))

          for i,det in enumerate(det_arr): 
              face = Face() 
              face.bounding_box = np.zeros(4, dtype=np.int32)  

              det = np.squeeze(det)
              bb = np.zeros(4, dtype=np.int32)
              bb[0] = np.maximum(det[0]-self.margin/2, 0)
              bb[1] = np.maximum(det[1]-self.margin/2, 0)
              bb[2] = np.minimum(det[2]+self.margin/2, img_size[1])
              bb[3] = np.minimum(det[3]+self.margin/2, img_size[0])
              face.bounding_box[0] = det[0]
              face.bounding_box[1] = det[1]
              face.bounding_box[2] = det[2]
              face.bounding_box[3] = det[3]

              if nrof_faces == 1:
                 _landmark = points[:,0].reshape((2,5)).T
                 img_face = face_preprocess.preprocess(img,det,_landmark,self.image_size)  
              else:
                 crop = img[bb[1]:bb[3], bb[0]:bb[2],:] 
                 bounding,po = detect_face.detect_face(crop, self.minsize,
                                                           self.pnet,self.rnet,self.onet,
                                                           self.threshold,self.factor)
                 num_faces = bounding.shape[0]

                 if num_faces > 0:
                    bound = bounding[:,0:4]
                    bindex = 0
                    if num_faces > 1:
                       bounding_box_size = (bound[:,2]-bound[:,0])*(bound[:,3]-bound[:,1])
                       img_center = img_size / 2
                       offsets = np.vstack([(bound[:,0]+bound[:,2])/2-img_center[1], (bound[:,1]+bound[:,3])/2-img_center[0]])
                       offset_dist_squared = np.sum(np.power(offsets,2.0),0)
                       bindex = np.argmax(bounding_box_size-offset_dist_squared*2.0)
 
                    _landmark = po[:, bindex].reshape((2,5)).T
                    img_face = face_preprocess.preprocess(crop,bounding,_landmark,self.image_size)

              face.img_face = img_face
              faces.append(face)
        return faces

