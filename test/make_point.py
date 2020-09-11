# -*- coding: UTF-8 -*-
import json
import requests
import threading
import time
import os
import cv2
import numpy as np

def get_file_list(file_path):
    dir_list = os.listdir(file_path)
    if not dir_list:
        return
    else:
        # 注意，这里使用lambda表达式，将文件按照最后修改时间顺序升序排列
        # os.path.getmtime() 函数是获取文件最后修改时间
        # os.path.getctime() 函数是获取文件最后创建时间
        dir_list = sorted(dir_list,  key=lambda x: os.path.getmtime(os.path.join(file_path, x)))
        #dir_list = sorted(dir_list,  key=lambda x: os.path.getmtime(os.path.join(file_path, x)))
        # print(dir_list)
        return dir_list

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
        # print(jsondata.json())
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

def make_tag(frame,data):
    for person in data:
        person.pop('climb', None)
        person["other"] = True
    
def save_data(data,file_path):
    with open(file_path, 'w') as outfile:
        json.dump(data, outfile)
    print("save {} success!".format(file_path))

if __name__ == '__main__':
    url = get_url_alphapose()
    dir_path = "train_data/space"

    files = get_file_list(dir_path)

    for file in files:
        json_file = file.replace('.jpg',".json")
        json_file = os.path.join(dir_path,json_file)
        file = os.path.join(dir_path,file)
        frame = cv2.imread(file)
        frame = resize_frame(frame)
        data = post_image(url,frame)

        print(json.dumps(data, indent=4, sort_keys=True))
        break
        # show_rectangle(frame,data)
        # make_tag(frame,data)
        # save_data(data,json_file)

            
            


