import cv2
import numpy as np
import json
import joblib
import time

#flask lib
import flask
from flask import Flask,url_for,render_template,request,redirect,send_file,jsonify

app = Flask(__name__)

from use_model import workon_frame,workon_frame_score

global args,action_model

@app.route('/api_detect_act/<id>' , methods=['POST'])
def api_detect_act(id):
    global args,action_model

    start = time.time()
    request_file = request.files['image']
    read_cost = time.time() - start
    print("read_cost:",read_cost)

    start = time.time()
    npimg = np.fromfile(request_file, np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    decode_cost = time.time() - start
    print("decode_cost:",decode_cost)

    start = time.time()
    cv2.imwrite("xx.jpg",frame)
    frame = cv2.imread("xx.jpg")
    rw_cost = time.time() - start
    print("rw_cost:",rw_cost)

    start = time.time()
    result = workon_frame_score(action_model,frame)
    work_cost = time.time() - start
    print("work_cost:",work_cost)
    return jsonify(result)


# @app.route('/api_test/<id>' , methods=['POST'])
# def api_test(id):
#     return None

if __name__ == "__main__":
    global args,action_model
    action_model = joblib.load("action_model.joblib")
    app.run(debug=False, host='0.0.0.0', port=7011,threaded=True)
