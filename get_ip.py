#!/usr/bin/python3
# -*- coding: utf-8 -*-
# --------------------------------
# Name get_ip
# Author DELL
# Date  2020/11/20

# -------------------------------
import socket

def get_host_ip():
    """
    查询本机ip地址
    :return:
    """
    try:
        s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
        s.connect(('8.8.8.8',80))
        ip=s.getsockname()[0]
    finally:
        s.close()

    return ip
