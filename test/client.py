import json
import requests
import threading
import time
import numpy as np
import cv2
import os
from pprint import pprint


#发送图片给检测服务器docker
def post_image(frame):
    cam_id = 0
    port = 7008
    interface = "api_post_frame"
    post_url = "http://localhost:{}/{}/{}".format(port, interface, cam_id)
    #frame to image_file
    frame_encoded = cv2.imencode(".jpg", frame)[1]
    Send_file = {'image': frame_encoded.tostring()}
    jsondata = requests.post(post_url,files=Send_file)

    if(jsondata.status_code == 200):
        return jsondata.json()
    else:
        return None

def predict_act():
    cam_id = 0
    port = 7008
    interface = "api_detect_act"
    post_url = "http://localhost:{}/{}/{}".format(port, interface, cam_id)


    jsondata = requests.get(post_url)

    if(jsondata.status_code == 200):
        return jsondata.json()
    else:
        return None

def show_rectangle(img,person):
    x0,y0,w,h = np.array(person["box"],dtype=np.int32)
    cv2.rectangle(img,(x0,y0),(x0+w,y0+h),color = (255,255,0),thickness=2)
    act_names = ["normal", "cross","throw","aim","handgun","attack"]
    show_str = person['act']
    show_text(frame,show_str,(x0,y0))

def show_score(img,person):
    x0,y0,w,h = np.array(person["box"],dtype=np.int32)
    cv2.rectangle(img,(x0,y0),(x0+w,y0+h),color = (255,255,0),thickness=2)
    act_names = ["normal", "cross","throw","aim","handgun","attack"]
    # show_str = person['act']
    # show_text(frame,show_str,(x0,y0))
    score_strs = []
    for idx,act_name in enumerate(act_names):
        if person[act_name] > 0.2:
            score_str = "{}:{}".format(act_name,person[act_name])
            score_strs.append(score_str)
    
    for idx,score_str in enumerate(score_strs):
        show_text(img,score_str,(x0,y0-30*idx))


def show_text(frame, out_text, org = (50, 50)):
    # font
    font = cv2.FONT_HERSHEY_SIMPLEX 
    
    # org
    org = org
    
    # fontScale
    fontScale = 1
    
    # Blue color in BGR
    color = (255, 0, 0)
    
    # Line thickness of 2 px
    thickness = 2
    
    # Using cv2.putText() method
    image = cv2.putText(frame, out_text, org, font,  
                    fontScale, color, thickness, cv2.LINE_AA) 

def get_all_imgs(dir_path):
    file_lst = os.listdir(dir_path)
    file_lst = sorted(file_lst,  key=lambda x: os.path.getmtime(os.path.join(dir_path, x)))
    fns = []
    for fn in file_lst:
        if fn.endswith(".jpg"): 
            fn = os.path.join(dir_path,fn)
            fns.append(fn)
    return fns


def get_video_writer(video_reader,file_name):
    fps = video_reader.get(cv2.CAP_PROP_FPS)
    size = (int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter(file_name,fourcc,fps, size)
    return video_writer

def solve_frame():
    persons = predict_act()
    for person_idx in range(len(persons)):
        person = persons[person_idx]
        # show_score(frame,person)
        print(person)
    # cv2.imshow("img",frame)
    # cv2.waitKey(0)

def solve_dir():
    dir_path = "attack/one_attack"
    imgs = get_all_imgs(dir_path)
    cost_all = 0
    
    img_num = 1
    for img_path in imgs[:10]:
        frame = cv2.imread(img_path)
        start = time.time()  
        solve_frame()
        cost_all += time.time() - start
    print("all:",cost_all, "\nsingle:",cost_all/img_num)

def resize_frame(frame):
    height,width,_ = frame.shape
    size = (int(width*0.25), int(height*0.25))
    frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
    return frame

if __name__ == '__main__':  
    frame = cv2.imread("1445.jpg")

    start = time.time() 
    for idx in range(5):
        post_image(frame)
    solve_frame()
    cost = time.time() - start
    print("cost:",cost)

        
        

# if __name__ == '__main__':
#     video = "D:/action_image/origin_video/attack_2.mp4"
#     out_video = "D:/action_image/origin_video/attack_2_mark.mp4"
#     video_reader = cv2.VideoCapture(video)
#     video_writer = get_video_writer(video_reader,out_video)
#     frame_idx = 0
#     while True:
#         valid,frame = video_reader.read()
#         if not valid:
#             break
#         else:
#             frame_idx += 1
        
#         persons = post_image(frame)
#         for person_idx in range(len(persons)):
#             rect_person = persons[person_idx]
#             show_rectangle(frame,rect_person)
#         # cv2.imshow("img",frame)
#         # cv2.waitKey(1)
#         video_writer.write(frame)
#     video_reader.release()
#     video_writer.release()



