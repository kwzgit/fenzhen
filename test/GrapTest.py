#!/usr/bin/python3
# -*- coding: utf-8 -*-
# --------------------------------
# Name GrapTest
# Author DELL
# Date  2020/11/18
# 测试图谱
# -------------------------------
from graph import GraphObj
from config import DATA_PATH, GRAPH_PWD, GRAPH_URL, GRAPH_USER
class Service(object):
    def __init__(self):
        self._graph = GraphObj(GRAPH_URL, GRAPH_USER, GRAPH_PWD)

    def search_diag_sex(self):
        disease_sex = self._graph.disease_sex()
        for key,value in disease_sex.items():
            print(key,value)


if __name__ == '__main__':
    service = Service()
    service.search_diag_sex()
