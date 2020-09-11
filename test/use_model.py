import json
import numpy as np
import cv2
import os
from joblib import load
import requests
import time

def preprocess(ann):
    kpts = ann['keypoints']
    bbox = ann['box']
    
    kpts = np.array(kpts).reshape(-1,3)[:,:2]
    cpts = (kpts[11] + kpts[12]) / 2
    kpts = kpts - cpts

    x, y, w, h = bbox
    kpts[:,0] = kpts[:,0]/w
    kpts[:,1] = kpts[:,1]/h
    
    return kpts.reshape(1,-1)

#发送图片给检测服务器docker
def post_to_alphapose(frame):
    cam_id = 0
    port = 7008
    interface = "api_detect_climb"
    post_url = "http://localhost:{}/{}/{}".format(port, interface, cam_id)



    #frame to image_file
    frame_encoded = cv2.imencode(".jpg", frame)[1]
    Send_file = {'image': frame_encoded.tostring()}
    # Send_file = Send_file = {'image': open("test.jpg", 'rb')}
    jsondata = requests.post(post_url,files=Send_file)

    if(jsondata.status_code == 200):
        return jsondata.json()
    else:
        return None

def predict_action(action_model,person):
    kpts = preprocess(person)
    result = action_model.predict(kpts)[0]
    act_id = 0
    if sum(result) == 0:
        act_id = 0
    elif sum(result) == 1:
        act_id = np.where(result == 1)[0]
    elif sum(result) > 1:
        print(sum_result)
        act_id = None

    return int(act_id)

def predict_action_score(action_model,person):
    kpts = preprocess(person)
    result = action_model.predict_proba(kpts)
    result = np.array(result)
    return result

def workon_frame_score(action_model,frame):
    act_names = ["normal", "cross","throw","aim","handgun","attack"]

    start = time.time()
    data = post_to_alphapose(frame)
    alpha_pose_cost = time.time() - start
    print("alpha_pose_cost:",alpha_pose_cost)

    for person in data:
        start = time.time()
        score = predict_action_score(action_model,person)
        parse_act_cost = time.time() - start
        print("parse_act_cost:",parse_act_cost)
        for idx in range(score.shape[0]):
            person[act_names[idx]] = round(score[idx][0][1],2)
    return data

def workon_frame(action_model,frame):
    act_names = ["normal", "cross","throw","aim","handgun","attack"]

    data = post_to_alphapose(frame)
    
    for person in data:
        act_id = predict_action(action_model,person)
        person["act"] = act_names[act_id]
    return data

if __name__ == '__main__':
    
    action_model = load("action_model.joblib")
    img_name = "1445.jpg"
    frame = cv2.imread(img_name)

    data = workon_frame(action_model,frame)
    print(data)




