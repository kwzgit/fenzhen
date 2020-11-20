#!/usr/bin/python3
# -*- coding: utf-8 -*-
# --------------------------------
# Name triage_server
# Author DELL
# Date  2020/11/19
# flask
# -------------------------------
import json
import sys
import traceback
from flask import Flask, request, Response

sys.path.append('..')
from config import IP, PORT
from structures import copy_response_structure,copy_response_structure_pre
# from triage_service import wrapper, Service, wrap_extracted_words
from triage_service import Service,wrapper


app = Flask(__name__)
triage_service_obj = Service()


@app.after_request
def cors(environ):  # 防止跨域问题
    environ.headers['Access-Control-Allow-Origin'] = '*'
    environ.headers['Access-Control-Allow-Method'] = '*'
    environ.headers['Access-Control-Allow-Headers'] = 'x-requested-with,content-type'
    return environ


def get_request_json():
    """
    :return: dict, {medical_text_type_1:content_1, ...}
    """
    json_request = {}
    try:
        if request.get_data():
            params = json.loads(request.get_data())  # 获取json
            json_request = params
    except:
        traceback.print_exc()
        print('-- 请求参数获取错误')

    return json_request


@app.route('/api/push', methods=['POST', 'GET'])
def predict():
    """预测
    """
    response_json = copy_response_structure()
    try:
        request_param = get_request_json()

        wrapper(service_obj=triage_service_obj, request_param=request_param, response_json=response_json)

    except:
        traceback.print_exc()

    finally:
        return Response(json.dumps(response_json, ensure_ascii=False), mimetype='application/json')


if __name__ == '__main__':
    # 获取本机电脑名和ip
    ip = IP
    port = PORT
    if len(sys.argv) >= 2:
        port = sys.argv[1]
    app.run(host=ip, port=port, debug=True)