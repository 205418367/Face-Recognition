from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys,os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import time
from scipy import misc
import tensorflow as tf
import numpy as np
import argparse
import facenet
import detect_face
import glob,h5py
import mxnet as mx
from six.moves import xrange
import sklearn

def main(args):
    train_set = facenet.get_dataset(args.data_dir)
    image_list, label_list = facenet.get_image_paths_and_labels(train_set)
    # fetch the classes (labels as strings) exactly as it's done in get_dataset
    path_exp = os.path.expanduser(args.data_dir)
    classes = [path for path in os.listdir(path_exp) if os.path.isdir(os.path.join(path_exp, path))]
    classes.sort()
    # get the label strings
    label_strings = [name for name in classes if os.path.isdir(os.path.join(path_exp, name))]
   
    model = get_model(mx.cpu(), (112,112), '/home/ubuntu/insightface/models/model-r100-ii/model,00', 'fc1')
    
    # Run forward pass to calculate embeddings
    nrof_images = len(image_list)
    print('Number of images: ', nrof_images)
    batch_size = args.image_batch
    if nrof_images % batch_size == 0:
        nrof_batches = nrof_images // batch_size
    else:
        nrof_batches = (nrof_images // batch_size) + 1

    print('Number of batches: ', nrof_batches)
    emb_array = np.zeros((nrof_images, 512))
    start_time = time.time()

    for i in range(nrof_batches):
        if i == nrof_batches -1:
           n = nrof_images
        else:
           n = i*batch_size + batch_size
        
        db = facenet.load_data(image_list[i*batch_size:n], False, False, args.image_size)
        model.forward(db, is_train=False)
        embed = model.get_outputs()[0].asnumpy()
        embed = sklearn.preprocessing.normalize(embed).flatten()
        emb_array[i*batch_size:n, :] = embed
        print('Completed batch', i+1, 'of', nrof_batches)

    run_time = time.time() - start_time
    print('Run time: ', run_time)

    out = '/home/ubuntu/insightface/video_classification/feats/2.h5'
    h5f = h5py.File(out, 'w')
 
    #export emedings and labels
    label_list  = np.array(label_list)  
    image_list = np.array(image_list)
    h5f['feats']  = emb_array
    h5f['lables'] = label_list
    h5f['names'] = image_list
    h5f.close()

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

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str,
        help='Directory containing images. If images are not already aligned and cropped include --is_aligned False.')
    parser.add_argument('--is_aligned', type=str,
        help='Is the data directory already aligned and cropped?', default=True)
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--margin', type=int,
        help='Margin for the crop around the bounding box (height, width) in pixels.',
        default=44)
    parser.add_argument('--gpu_memory_fraction', type=float,
        help='Upper bound on the amount of GPU memory that will be used by the process.',
        default=1.0)
    parser.add_argument('--image_batch', type=int,
        help='Number of images stored in memory at a time. Default 500.',
        default=500)

    #numpy file Names
    parser.add_argument('--embeddings_name', type=str,
        help='Enter string of which the embeddings numpy array is saved as.',
        default='embeddings.npy')
    parser.add_argument('--labels_name', type=str,
        help='Enter string of which the labels numpy array is saved as.',
        default='labels.npy')
    parser.add_argument('--labels_strings_name', type=str,
        help='Enter string of which the labels as strings numpy array is saved as.',
        default='label_strings.npy')
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
