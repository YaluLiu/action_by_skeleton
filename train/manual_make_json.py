# -*- coding: UTF-8 -*-
import json
import requests
import threading
import time
import os
import cv2
import numpy as np
from utils import read_json,get_all_images

#获取接口地址
def get_url_alphapose():
    global act_cfg
    #端口号
    port = 7008
    #主机地址
    host = act_cfg["host"]
    cam_id = 2
    api = "api_detect_climb"
    detect_url = "http://{}:{}/{}/{}".format(host,port, api, cam_id)
    return detect_url

#发送图片给检测服务器docker
def post_image(post_url,frame):
    frame_encoded = cv2.imencode(".jpg", frame)[1]
    Send_file = {'image': frame_encoded.tostring()}
    jsondata = requests.post(post_url,files=Send_file)

    #output
    if jsondata.status_code == requests.codes.ok:
        return jsondata.json()
    else:
        print("Error")

def resize_frame(frame):
    height,width,_ = frame.shape
    size = (int(width*0.5), int(height*0.5))
    frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
    return frame

def show_rectangle(img,rec):
    x0,y0,w,h = np.array(rec,dtype=np.int32)
    print(x0,y0,w,h)
    cv2.rectangle(img,(x0,y0),(x0+w,y0+h),color = (255,255,0),thickness=2)

def show_rectangle(frame,data):
    for person in data:
        frame_show = np.copy(frame)
        person.pop('climb', None)
        x0,y0,w,h = np.array(person["box"],dtype=np.int32)
        cv2.rectangle(frame_show,(x0,y0),(x0+w,y0+h),color = (255,255,0),thickness=2)
        cv2.imshow("aa",frame_show)
        key = cv2.waitKey(0)
        person["attack"] = True
        if key == ord('a'):
            person["attack"] = False
        print(person["attack"])

def make_label(data,target_label,act_cfg):
    for person in data:
        person.pop('climb', None)
        for act_name,act_id in act_cfg.items():
            if act_name == target_label:
                person["act_id"] = act_id
    
def save_data(data,file_path):
    with open(file_path, 'w') as outfile:
        json.dump(data, outfile)
    print("save {} success!".format(file_path))

if __name__ == '__main__':
    act_cfg = read_json("../config/action_space.json")
    url = get_url_alphapose()

    img_path = "../imgs/1445.jpg"
    frame = cv2.imread(img_path)
    data = post_image(url,frame)
    make_label(data,"cross",act_cfg)
    print(json.dumps(data,indent=4))

            
            


