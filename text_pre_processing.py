#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""文本预处理
"""


class TextPreProcessing():
    """编写并注册文本预处理方法
    """

    def __init__(self):
        """
        """
        pass

    @classmethod
    def processing(cls, string):
        """数据预处理接口
        :param string:
        :return: string
        """
        processing_functions = []  # 注册预处理方法的地址
        for func in processing_functions:
            string = func(string)
        return string