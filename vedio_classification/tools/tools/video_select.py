#coding=utf-8
import cv2
import os

root_dir = "/home/ubuntu/insightface/video_classification/video/result/31"
#遍历所有mp4格式的视频文件
for filename in os.listdir(root_dir):
    if filename.split('.')[1] != 'avi':
       continue
    video_name = os.path.join(root_dir, filename)
    
    try:
        video_capture = cv2.VideoCapture(video_name)
        ret, frame = video_capture.read()
        cv.imshow("11",frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):    
           break
    except:
        print(video_name)
        os.remove(video_name)
