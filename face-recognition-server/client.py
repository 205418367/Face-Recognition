#coding=utf-8
import urllib2
import json
from socket import *
import os,time
import threading
from Queue import Queue
from scipy import misc
import demjson
import base64
 
def http_post():  
        url = "http://172.16.145.214:5000/predict/" 
        f = open("/home/ubuntu/Desktop/timg.jpeg")
        base64_data = base64.b64encode(f.read())
 
        data = {"fileName":"0.png","base64Data":str(base64_data),"X":"0","Y":"0","Z":"0"}
        #postData = {"fileName":"0.png","base64Data":str("base64_data"),"X":"-0.958637","Y":"0.00766414455","Z":"-0.1388637"}
        postData = demjson.encode(data)
 
        req = urllib2.Request(url, postData)  
        req.add_header('Content-Type', 'application/json')  
        response = urllib2.urlopen(req)  
        result = json.loads(response.read())  
        print result    
 
http_post()
