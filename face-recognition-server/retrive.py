# -*- coding:utf-8 -*-
from flask import Flask,render_template,request
import recognition
from Queue import Queue
import threading
from scipy import ndimage
import numpy as np
import math
import base64
import sys,os
import cv2
import demjson
import xml.etree.cElementTree as et


app = Flask(__name__)
#网页可视化
#@app.route('/')
#def index():
     #return render_template("index.html")
#创建两个队列
q1 = Queue()
q2 = Queue()
fileName = None
flage = True

#主函数
@app.route('/predict/', methods=['GET','POST'])
def predict():
    #获取json数据
    recv = demjson.decode(request.get_data())
    global fileName
    fileName = recv["fileName"]
    base64Data = recv["base64Data"]
    image_data = base64.b64decode(base64Data)
    image_data = np.fromstring(image_data, np.uint8)
    x = recv["X"]
    y = recv["Y"]
    z = recv["Z"]

    try:
        image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)    
        
        if (x == "0" and y == "0" and z == "0"):
           global flage
           flage = False
     
        if flage:
           if (float(z) > -0.9 and float(z) < 0.9):
              angle = math.atan2(float(y),float(x))
              angle = angle*(180/math.pi) + 90
              angle = -angle
              print("~~~~~~~~~~~~~图片旋转%d度~~~~~~~~~~"%angle)
              if not (angle > -30 and angle< 30): 
                 image = ndimage.rotate(image,angle)
        flage = True     
    except:
        print("提示：读取发生异常！")
        data = {"fileName":fileName,"isResult":"false","resultNum":"0","base64Data":"{}"}
        data = demjson.encode(data)
        q2.put(data)
    else:  
        q1.put(image)  
        print("~~~~~~~~~~~~~数据已经加入队列q1~~~~~~~~~~")                                
    return q2.get()
 
#客户端函数    
def client():
    tree=et.parse("config/conf.xml")
    root=tree.getroot()
    port=root.find('port').text
    host=root.find('host').text
    port = int(os.environ.get("PORT", port))
    app.run(host=host, port=port)
                      
if __name__ == '__main__':
    #建立客户端连接线程
    parseImage_thread = threading.Thread(target=client)
    parseImage_thread.start()

    while True:
        while not q1.empty():
            value = q1.get()
            print("从队列q1中取出数据")
            try:
                print("~~~~~~~~~~~~~0~~~~~~~~~~~~~~~")
                faces = recognition.identify(value)
                print("~~~~~~~~~~~~~1~~~~~~~~~~~~~~~")
            except:
                print("提示：识别发生异常！")
                data = {"fileName":fileName,"isResult":"false","resultNum":"0","base64Data":"{}"}
                data = demjson.encode(data)
                q2.put(data)
            else:
                if faces:
                    base64_dic = []
                    num = len(faces)
                    for one_face in faces:
                        bb_0 = one_face.x1
                        bb_1 = one_face.y1
                        bb_2 = one_face.x2
                        bb_3 = one_face.y2

                        embedding = one_face.embedding
                        embedding = embedding.flatten()
                        embedding = embedding.tolist()
                        base64Data = {"featueData":embedding,"left":bb_0,"top":bb_1,"bottom":bb_2,"right":bb_3}
                        base64_dic.append(base64Data)

                        data = {"fileName":fileName,"isResult":"true","resultNum":str(num),"base64Data":str(base64_dic)}
                        data = demjson.encode(data)
                    q2.put(data)
                else:
                    data = {"fileName":fileName,"isResult":"false","resultNum":"0","base64Data":"{}"}
                    data = demjson.encode(data)
                    q2.put(data)
                    print("检测数据已经放入队列q2-2")


