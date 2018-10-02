import cv2
import numpy as np
import sys
import mxnet as mx
import datetime
from skimage import transform as trans
from sklearn import preprocessing


class Embedding:
    def __init__(self, prefix, epoch, ctx_id=0):
        print('loading', prefix, epoch)
        ctx = mx.gpu(ctx_id)
        sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
        all_layers = sym.get_internals()
        sym = all_layers['fc1_output']
        image_size = (112, 112)
        self.image_size = image_size
        model = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
        # model = mx.mod.Module(symbol=sym, context=ctx)
        model.bind(for_training=False, data_shapes=[('data', (1, 3, image_size[0], image_size[1]))])
        model.set_params(arg_params, aux_params)
        self.model = model

    def get(self, img):
        img = np.transpose(img, (2, 0, 1))  # 3*112*112, RGB
        input_blob = np.zeros((1, 3, self.image_size[1], self.image_size[0]), dtype=np.uint8)
        input_blob[0] = img
        data = mx.nd.array(input_blob)
        db = mx.io.DataBatch(data=(data,))
        self.model.forward(db, is_train=False)
        embedding = self.model.get_outputs()[0].asnumpy()
        embedding = preprocessing.normalize(embedding).flatten()
        return embedding
