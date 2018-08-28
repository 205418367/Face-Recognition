#coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import sys
import os
import argparse
import numpy as np
import mxnet as mx
import random
import cv2
import sklearn
from sklearn.decomposition import PCA
from time import sleep
from easydict import EasyDict as edict
from mtcnn_detector import MtcnnDetector
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'common'))
import face_image
import face_preprocess
import tensorflow as tf
import detect_face
from scipy import misc

mtcnn_model_checkpoint = "/home/ubuntu/insightface/deploy/mtcnn-model"
class Face:
    def __init__(self):
        self.img_face = None
        self.yaw = None

class Embed:
    def __init__(self):
        self.embedding = None

def get_model(ctx, image_size, model_str, layer):
    _vec = model_str.split(',')
    assert len(_vec)==2
    prefix = _vec[0]
    epoch = int(_vec[1])
    print('loading',prefix, epoch)
    sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
    all_layers = sym.get_internals()
    sym = all_layers[layer+'_output']
    model = mx.mod.Module(symbol=sym, context=ctx, label_names = None)
    model.bind(data_shapes=[('data', (1, 3, image_size[0], image_size[1]))])
    model.set_params(arg_params, aux_params)
    return model

class FaceModel:
  def __init__(self, args):
      self.args = args
      ctx = mx.cpu()
      _vec = args.image_size.split(',')
      image_size = (int(_vec[0]), int(_vec[1]))
      self.model = None
      self.model = get_model(ctx, image_size, args.model, 'fc1')
      self.threshold = args.threshold
      self.minsize = 60
      self.threshold = [0.6,0.7,0.8]
      self.margin = 44
      self.pnet, self.rnet, self.onet = self._setup_mtcnn()
      self.factor = 0.709

  def _setup_mtcnn(self):    
      with tf.Graph().as_default():
          sess = tf.Session()
          with sess.as_default():
              return detect_face.create_mtcnn(sess, mtcnn_model_checkpoint)

  def get_input(self, img):
      faces = []
      bounding_boxes,points = detect_face.detect_face(img, self.minsize,
                                                           self.pnet,self.rnet,self.onet,
                                                           self.threshold,self.factor )
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
                offsets = np.vstack([ (det[:,0]+det[:,2])/2-img_center[1], (det[:,1]+det[:,3])/2-img_center[0] ])
                offset_dist_squared = np.sum(np.power(offsets,2.0),0)
                index = np.argmax(bounding_box_size-offset_dist_squared*2.0)
                det_arr.append(det[index,:])
          else:
                det_arr.append(np.squeeze(det))


          for i,det in enumerate(det_arr):         
              face = Face()
              det = np.squeeze(det)
              bb = np.zeros(4, dtype=np.int32)
              bb[0] = np.maximum(det[0]-self.margin/2, 0)
              bb[1] = np.maximum(det[1]-self.margin/2, 0)
              bb[2] = np.minimum(det[2]+self.margin/2, img_size[1])
              bb[3] = np.minimum(det[3]+self.margin/2, img_size[0])

              if nrof_faces == 1:
                 crop = img[bb[1]:bb[3], bb[0]:bb[2],:]
                 _landmark = points[:,0].reshape((2,5)).T
                 img_face,yaw = face_preprocess.preprocess(crop,det,_landmark,self.image_size)  
              else:
                 crop = img[bb[1]:bb[3], bb[0]:bb[2],:] 
                 bounding,po = detect_face.detect_face(crop, self.minsize,
                                                           self.pnet,self.rnet,self.onet,
                                                           self.threshold,self.factor )

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
                    img_face,yaw = face_preprocess.preprocess(crop,bounding,_landmark,self.image_size) 

              face.img_face = img_face
              face.yaw = yaw
              faces.append(face)
      return faces

  def get_feature(self, aligned):
      embeds = []
      embed = Embed()
      for align in aligned:
          if align.yaw < 30:
             aligned = np.transpose(align.img_face, (2,0,1))
             input_blob = np.expand_dims(aligned, axis=0)
             data = mx.nd.array(input_blob)
             db = mx.io.DataBatch(data=(data,))
             self.model.forward(db, is_train=False)
             embedding = self.model.get_outputs()[0].asnumpy()
             embedding = sklearn.preprocessing.normalize(embedding).flatten()
             embed.embedding = embedding
             embeds.append(embed)
      if len(embeds) > 0:
         return embeds
      else:
         return None

  def get_ga(self, aligned):
      input_blob = np.expand_dims(aligned, axis=0)
      data = mx.nd.array(input_blob)
      db = mx.io.DataBatch(data=(data,))
      self.ga_model.forward(db, is_train=False)
      ret = self.ga_model.get_outputs()[0].asnumpy()
      g = ret[:,0:2].flatten()
      gender = np.argmax(g)
      a = ret[:,2:202].reshape( (100,2) )
      a = np.argmax(a, axis=1)
      age = int(sum(a))
      return gender, age

