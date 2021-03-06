#coding=utf-8
import numpy as np
from tensorface.facenet128 import Embedding
from tensorface import detection
import math
import cv2
from matplotlib import pyplot as plt
import os,time

#测试facenet的脚本，需要事先准备训练集（有标签）用于建立特征库 ，测试集（有标签）一般是应用场景数据用于评测
#facenet提取人脸特征前需要对图片进行归一化
embedding = Embedding('models/20170511-185253/20170511-185253.pb')
usr_dir = '/home/ubuntu/insightface/face-recognition-evaluation/test_images/users'
img_dir = '/home/ubuntu/insightface/face-recognition-evaluation/test_images/images'
save_dir = '/home/ubuntu/insightface/face-recognition-evaluation/test_images/facenet_128/'

# get users features
def extract_users_feature():
    users_feature = np.zeros((1, 128))
    users_id = []
    for dirname in os.listdir(usr_dir):
        u_id = int(dirname)
        dir_path = usr_dir + '/' + dirname
        for filename in os.listdir(dir_path):
            img_path = dir_path + '/' + filename
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)[..., ::-1]
            faces = detection.get_faces(img, 0.6)
            face = faces[0]
            users_id.append(u_id)
            f = embedding.get(face.img)
            users_feature = np.row_stack((users_feature, f))

    users_feature = np.delete(users_feature, 0, 0)
    np.save(save_dir + 'users_id.npy', np.array(users_id))
    np.save(save_dir + 'users_feature.npy', users_feature)


# get test data features
def extract_images_feature():
    for dirname in os.listdir(img_dir):
        u_id = int(dirname)
        dir_path = img_dir + '/' + dirname
        feature_out = save_dir + 'feature_out_data/'
        result = np.zeros((1, 128))
        path_txt = feature_out + dirname + '.txt'

        with open(path_txt, mode='w') as pf:
            for filename in os.listdir(dir_path):
                img_path = dir_path + '/' + filename
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)[..., ::-1]
                faces = detection.get_faces(img, 0.6)
                if len(faces) == 0:
                    continue
                face = faces[0]
                maxbox = 0
                #选择最大人脸
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
                result = np.row_stack((result, f))
                print(img_path, file=pf)
        result = np.delete(result, 0, 0)
        np.save(feature_out + dirname + '.npy', result)
 
def loadtxt(txtpath):
    result = []
    with open(txtpath, 'r') as f:
        for line in f:
            result.append(list(line.strip('\n').split(',')))
    return result
 
def total_images():
    total_num = 0
    for dirname in os.listdir(img_dir):
        dirname = img_dir +"/"+dirname
        for image in os.listdir(dirname):
            total_num += 1
    return total_num

def evaluate():
    users_id = np.load(save_dir + 'users_id.npy')
    users_feature = np.load(save_dir + 'users_feature.npy')
 
    acc_list = []
    far_list = []
    thred_list = []
    wrong_img_list = []
    total_num = total_images()
    for thred in np.arange(0.3, 1.0, 0.01):
        correct = 0
        false_accepted_num = 0
        for dirname in os.listdir(img_dir):
            u_id = int(dirname)
            feature_out_dir = save_dir + 'feature_out_data/'
            feature_path = feature_out_dir + dirname + '.npy'
            txt_path = feature_out_dir + dirname + '.txt'
            features = np.load(feature_path)
            img_path_list = loadtxt(txt_path)

            for i in range(len(img_path_list)):
                ff = features[i, :]
                #从测试集中取出一个特征与特征库中所有特征计算相似度
                #欧式相似度
                # dists = np.sum((ff - users_feature) ** 2, axis=1)
                # similarity = 1 / (dists + 1)
                #余弦相似度并不等于余弦距离
                dot = np.sum(np.multiply(ff, users_feature), axis=1)
                similarity = dot
                index = np.argsort(-similarity)

                if u_id == 0:
                    if similarity[index[0]] < thred:
                        correct = correct + 1
                    else:
                        false_accepted_num = false_accepted_num + 1
                        wrong_img_list.append((thred, users_id[index[0]], img_path_list[i], similarity[index[0]], u_id))
                else:
                    #u_id是测试样本真实标签；users_id[index[0]]是预测标签；similarity[index[0]]是相似度
                    if u_id == users_id[index[0]] and similarity[index[0]] >= thred:
                        correct = correct + 1
                    if u_id == users_id[index[0]] and similarity[index[0]] < thred:
                        wrong_img_list.append((thred, users_id[index[0]], img_path_list[i], similarity[index[0]], u_id))
                    if u_id != users_id[index[0]] and similarity[index[0]] >= thred:
                        false_accepted_num = false_accepted_num + 1
                    if u_id != users_id[index[0]]:
                        wrong_img_list.append((thred, users_id[index[0]], img_path_list[i], similarity[index[0]], u_id))
 
        print('-----------------------------------------')
        print('correct:%d, total:%d, thredshold:%f' % (correct, total_num, thred))
        acc = float(correct) / float(total_num)
        print("%%%%%%%%%%%%%: "+str(float(correct)) +" "+str(float(total_num)))
        far = float(false_accepted_num) / float(total_num)
        acc_list.append(acc)
        far_list.append(far)
        thred_list.append(thred)
        #accuracy:通过率，即标签相同并且相似度高于阈值
        #false_rate:误识率，即标签不同并且相似度高于阈值
        print('accuracy: %f, false_rate: %f' % (acc, far))
 
    x = np.array(thred_list)
    y = np.array(acc_list)
    plt.plot(x, y)
    plt.show()
    y = np.array(far_list)
    plt.plot(x, y)
    plt.show()
    x = np.array(far_list)
    y = np.array(acc_list)
    plt.plot(x, y)
    plt.show()
 
    print('----------------------------------')
    idx = np.argmax(acc_list)
    print('best accuracy: %f, best threshold:%f' % (acc_list[idx], thred_list[idx]))
 
    for i in range(len(wrong_img_list)):
        if abs(wrong_img_list[i][0] - thred_list[idx]) < 0.001:
            print(wrong_img_list[i][2])
            print(wrong_img_list[i][1], wrong_img_list[i][3], wrong_img_list[i][4])
 
# --------------------------------------------------------------------------------------------------------------
extract_users_feature()
extract_images_feature()
evaluate()
