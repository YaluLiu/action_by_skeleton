# make kps data and label data
import numpy as np 
import json
import os
from utils import read_json,get_all_images,get_all_jsons

#Train
import joblib
from sklearn import ensemble
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def preprocess(ann):
    kpts = ann['keypoints']
    bbox = ann['box']
    
    kpts = np.array(kpts).reshape(-1,3)[:,:2]
    cpts = (kpts[11] + kpts[12]) / 2
    kpts = kpts - cpts

    x, y, w, h = bbox
    kpts[:,0] = kpts[:,0]/w
    kpts[:,1] = kpts[:,1]/h
    
    return kpts.reshape(-1)

def parse_kps(kps):
    kps = np.array(kps,dtype = np.int64)
    kps = kps.reshape(-1,3)[:,:2].reshape(-1)
    return kps

def make_train_data(root_dir):
    all_jsons = get_all_jsons(root_dir)
    kps_data = []
    labels = []
    for json_path in  all_jsons:
        persons = read_json(json_path)
        for person in persons:
            kps = preprocess(person)
            kps_data.append(kps)
            labels.append(person['act_id'])
    kps_data = np.array(kps_data,dtype = np.float64)
    data_num = len(labels)
    label_data = np.zeros((data_num,6),dtype=np.int64)
    for idx in range(len(labels)):
        tmp_lable = labels[idx]
        label_data[idx][tmp_lable] = 1
    return kps_data,label_data

    # print(kps_data.shape)   #(persons_num,skeleton_poins_num)
    # print(label_data.shape) #(persons_num,acts_num)
    # np.save("feature_throw.npy", kps_data)
    # np.save("label_throw.npy", label_data)

def train_model(x_train,y_train):
    clf = ensemble.RandomForestClassifier(max_depth=20)
    clf.fit(x_train, y_train)
    return clf
    

if __name__ == '__main__':
    act_cfg = read_json("../config/action_space.json")
    train_data_path = act_cfg["train_data_path"]
    kps_data,label_data = make_train_data(train_data_path)
    x = kps_data
    y = label_data
    x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                        test_size=0.1, 
                                                        random_state=40)

    model = train_model(x_train,  y_train)

    y_pred = model.predict(x_test)
    y_pred = np.array(y_pred)


    
    target_names = list(act_cfg.keys())
    print(classification_report(y_test, y_pred, target_names=target_names))

    # save Model
    # joblib.dump(clf, model_path) #"stand_sit_model.joblib"






