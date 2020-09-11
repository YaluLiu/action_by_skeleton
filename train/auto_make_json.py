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
    #端口号
    port = 7008
    #主机地址
    host = "192.168.200.233"
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

def add_act_id(data,target_act):
    global act_cfg
    for person in data:
        person.pop('climb', None)
        for act_name,act_id in act_cfg.items():
            if target_act == act_name:
                person["act_id"] = act_id
                person["act_name"] = act_name
    
def save_data(data,file_path):
    with open(file_path, 'w') as outfile:
        json.dump(data, outfile)
    print("save {} success!".format(file_path))

def solve_one_image(image_path,act_name):
    global url,act_cfg
    json_path = image_path.replace(".jpg",".json")

    frame = cv2.imread(image_path)
    data = post_image(url,frame)
    add_act_id(data,act_name)

    # print(json.dumps(data,indent=4))
    save_data(data,json_path)

if __name__ == '__main__':
    act_cfg = read_json("../config/action_space.json")
    url = get_url_alphapose()
    # solve_one_image("../imgs/1452.jpg","cross")

    imgs = get_all_images("../imgs")
    for img_path in imgs:
        solve_one_image(img_path,"cross")
    

    

            
            


