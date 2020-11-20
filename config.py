#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
"""
import os

PROJECT_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = os.path.join(PROJECT_PATH, 'data')
NER_MODEL_PATH = os.path.join(PROJECT_PATH, 'medical_entity_recognition', 'bilstm_crf')

IP = '192.168.3.180'
PORT = '8877'

GRAPH_URL = 'http://192.168.2.123:7472'
GRAPH_USER = 'neo4j'
GRAPH_PWD = '12345678'


# match (n:`疾病`{name:'主动脉狭窄'})-[r:`疾病相关科室`]->(w:`科室`) where r.rank in ['1','2'] return n,r,w