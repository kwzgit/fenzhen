#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""安装帮助
    1- 删除无关的文件
    2- 做出依赖文件requirements.txt
    3- 下载所需文件到python_packages文件夹下
"""

import os
import shutil
import sys

sys.path.append('..')

from config import PROJECT_PATH

def get_requirements():
    """生成依赖文件
    """
    if os.path.exists('requirements.txt'):
        os.remove('requirements.txt')
    command_line = 'pipreqs ./ --encoding=utf-8'
    os.system(command_line)


def download_requirements():
    """下载安装包
    """

    command_line = 'pip download -d {} -r requirements.txt'.format(os.path.join(PROJECT_PATH, 'python_packages'))
    command_line += ' -i https://pypi.tuna.tsinghua.edu.cn/simple'
    os.system(command_line)


def remove_pipeline_data():
    """移除流水线文件
    """
    if os.path.exists('pipeline_data'):
        shutil.rmtree('pipeline_data')
    os.mkdir('pipeline_data')


def main():
    """
    """
    print(os.system('python -V'))

    result = None
    while result not in ('yes', 'no'):
        result = input('持否重新查询依赖，yes or no:')

    if result.strip() == 'yes':
        get_requirements()

    to_download_requirements = None
    while to_download_requirements not in ('yes', 'no'):
        to_download_requirements = input('是否下载依赖项，yes or no:')

    if to_download_requirements == 'yes':
        clear_packages = None
        while clear_packages not in ('yes', 'no'):
            clear_packages = input('是否清空python_packages文件夹，yes or no:')

        if clear_packages.strip() == 'yes':
            if os.path.exists('python_packages'):
                shutil.rmtree('python_packages')
            os.mkdir('python_packages')
        else:
            if not os.path.exists('python_packages'):
                os.mkdir('python_packages')

        download_requirements() # 下载依赖项


if __name__ == '__main__':
    main()