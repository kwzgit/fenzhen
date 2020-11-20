#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""工具类
"""
import json
import math
import os
import pickle
import shutil
import logging
import argparse
import pandas as pd
import numpy as np


PAD = '[PAD]'
SEP = '[SEP]'
UKN = '[UKN]'
MASK = '[MASK]'
CLS = '[CLS]'


def generate_word2id_dict(sentences, rate=0.99):
    """
    :param sentences:
    :param rate:
    :return: {token:id, }
    """
    word_cnt_dict = {}
    for sentence in sentences:
        for token in sentence:
            word_cnt_dict.setdefault(token, 0)
            word_cnt_dict[token] += 1

    word_cnt_sorted = sorted(word_cnt_dict.items(), key=lambda x: x[-1], reverse=True)
    word2id_dict = {PAD: 0, SEP: 1, UKN: 2, MASK: 3, CLS: 4}
    for word_cnt in word_cnt_sorted[:int(len(word_cnt_sorted) * rate)]:
        word2id_dict[word_cnt[0]] = len(word2id_dict)
    return word2id_dict


def generate_idf_dict(documents):
    """
    :param documents:
    :return: {token:idf_value}
    """
    N = len(documents)
    idf_dict = {}
    # compute number of document a token appear nt
    for i, document in enumerate(documents):
        for token in document:
            idf_dict.setdefault(token, set())
            idf_dict[token].add(i)

    # compute token's idf
    for token, nt in idf_dict.items():
        idf_dict[token] = math.log((N + 1 ) / (len(nt) + 1)) + 1 # 加平滑

    return idf_dict


def compute_tf(document):
    """compute normalized term frequency
    :param document:
    :return: {token: tf_value}
    """
    _len = len(document)
    tf_dict = {}
    for token in document:
        tf_dict.setdefault(token, 0.0)
        tf_dict[token] += 1 / _len

    return tf_dict


def tf_idf(tf_dict, idf_dict, word2id_dict):
    """
    :param tf_dict:
    :param idf_dict:
    :param word2id_dict:
    :return: vector
    """
    vector = [0] * len(word2id_dict)
    for token, tf in tf_dict.items():
        idf = idf_dict.get(token)
        if idf is not None:
            tf_idf_value = tf / idf
            index = word2id_dict.get(token)
            if index is not None:
                vector[index] = tf_idf_value
    return vector


def compute_tf_idf(document, idf_dict, word2id_dict):
    """
    :param document:
    :param idf_dict:
    :param word2id_dict:
    :return:
    """
    tf_dict = compute_tf(document)
    return tf_idf(tf_dict, idf_dict, word2id_dict)


def tokens2ids(tokens, word2id, max_len):
    """token转id
    :param tokens: list
    :param word2id: dict
    :param max_len: int
    :return: list of integer
    """
    vector = [word2id[PAD]] * max(max_len, len(tokens))
    for i, token in enumerate(tokens):
        vector[i] = word2id.get(token, word2id[UKN])
    return vector[:max_len]


def write_lines_to_file(line_data, file_path, add_line_break=False):
    """逐行写入到文件
    :param line_data:  行数据，需要自己制定换行符
    :param file_path:  文件路径
    :param add_line_break:  添加换行符
    :return:
    """
    if add_line_break:
        line_data = [str(line) + '\n' for line in line_data]
        line_data[-1] = line_data[-1].strip()  # 最后无空白行
    with open(file_path, 'w', encoding='UTF-8', newline='') as f:
        f.writelines(line_data)


def write_lines_to_file_append(line_data, file_path, add_line_break=False):
    """逐行写入到文件
    :param line_data:  行数据，需要自己制定换行符
    :param file_path:  文件路径
    :param add_line_break:  添加换行符
    :return:
    """
    if add_line_break:
        line_data = [str(line) + '\n' for line in line_data]
    with open(file_path, 'a+', encoding='UTF-8', newline='') as f:
        f.writelines(line_data)


def white_char_replace(origin_char):
    """
    :param origin_char:
    :return:
    """
    char_replace_dict = {' ': '$', '\u3000': '₳', '\t': '€'}
    reversed_char_replace_dict = dict([(value, key) for key, value in char_replace_dict.items()])
    if char_replace_dict.get(origin_char) is not None:
        return char_replace_dict.get(origin_char)
    if reversed_char_replace_dict.get(origin_char) is not None:
        return reversed_char_replace_dict.get(origin_char)
    return origin_char


def white_char_replace_content(origin_content):
    """空白字符替换
    :param origin_content: 原始文本
    :return: string
    """
    char_list = list(origin_content)
    for i, char in enumerate(char_list):
        char_list[i] = white_char_replace(char)
    return ''.join(char_list)


def is_txt_bio_match(txt_file, bio_file):
    """txt文件和bio文件是否一致
    :param txt_file:
    :param bio_file:
    :return: bool
    """
    with open(txt_file, 'r', encoding='utf-8', newline='') as txt:
        txt_content = ''.join(txt.readlines())
        txt_content = txt_content.replace('\r\n', '\n')

    with open(bio_file, 'r', encoding='utf-8', newline='') as f:
        bio_content = []
        for line in f.readlines():
            line = line.strip()
            if line:
                bio_content.append(white_char_replace(line.split(' ')[0]))
            else:
                bio_content.append('\n')
        if bio_content[-1] == '\n':  # 除去最后的空格
            bio_content = bio_content[:-1]
        bio_content = ''.join(bio_content)

    return txt_content == bio_content


def write_pickle_file(data, file_path):
    """保存pickle数据
    :param data:
    :param file_path:
    """
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)


def read_pickle_file(file_path):
    """保存pickle数据
    :param data:
    :param file_path:
    """
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def write_json_file(obj, file_path):
    """写json文件
    :param obj: 对象
    :param file_path: 路径
    """
    with open(file_path.encode('UTF-8'), 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False)


def read_json_file(file_path):
    """读取json文件
    :param file_path:
    :return: json object
    """
    with open(file_path, 'r', encoding='UTF-8') as f:
        return json.load(f)


def copy_files_dir_to_dir(from_dir, to_dir, overwrite=True):
    """复制一个文件夹下的所有文件到另一个文件夹中
    :param from_dir:
    :param to_dir:
    :param overwrite:是否覆盖
    """
    from_file_names = os.listdir(from_dir)
    to_file_names = set(os.listdir(to_dir))
    for from_file_name in from_file_names:
        if overwrite:
            from_file_path = os.path.join(from_dir, from_file_name)
            to_file_path = os.path.join(to_dir, from_file_name)
            shutil.copyfile(from_file_path, to_file_path)
        else:
            if from_file_name not in to_file_names:
                from_file_path = os.path.join(from_dir, from_file_name)
                to_file_path = os.path.join(to_dir, from_file_name)
                shutil.copyfile(from_file_path, to_file_path)


def full_text_search_in_line(string, file_dir, file_type='.py'):
    """统计file_dir目录下，搜索指定字符串
    :param string: 指定字符串
    :param file_dir: 目录
    :param file_type: 文件类型
    :return: int
    """
    queue = [file_dir]
    while queue:
        path = queue.pop(0)
        if os.path.isfile(path):
            if len(path) > len(file_type) and path[-len(file_type):] == file_type:
                with open(path, 'r', encoding='UTF-8') as f:
                    file_path = os.path.abspath(path)
                    find_lines = []
                    for i, line in enumerate(f.readlines()):
                        if line.find(string) > -1:
                            find_lines.append((i, line))
                    if find_lines:
                        print(file_path, '----->')
                        for pair in find_lines:
                            print('    行号，内容：', pair[0], pair[1])
        elif os.path.isdir(path):
            file_names = os.listdir(path)
            for file_name in file_names:
                queue.append(os.path.join(path, file_name))
        else:
            print('error path?', os.path.abspath(path))


def copy_file_directory_tree(origin_dir_root, new_dir_root):
    """复制文件目录
    :param origin_dir_root:
    :param new_dir_root:
    """
    origin_dir_list = []
    queue = [origin_dir_root]
    while queue:
        path = queue.pop(0)
        if os.path.isdir(path):
            queue.extend([os.path.join(path, sub_path) for sub_path in os.listdir(path)])
            origin_dir_list.append(path.replace(origin_dir_root, new_dir_root))

    for path in origin_dir_list:
        os.mkdir(path)


def create_dir_if_not_exist(dir_path):
    """创建文件夹
    :param dir_path:
    """
    dir_list = []
    dir_name = dir_path
    while not os.path.exists(dir_name):
        dir_list.insert(0, dir_name)
        dir_name = os.path.dirname(dir_name)

    for dir_ in dir_list:
        os.mkdir(dir_)


def copy_dir_to_dir_linux(from_dir, to_dir):
    """复制文件夹到文件夹
    :param from_dir:
    :param to_dir:
    """
    command_line = 'cp -r {} {}'.format(from_dir, to_dir)
    os.system(command_line)


def move_all_file_from_a_dir_to_another_dir_linux(from_dir, to_dir):
    """移动一个文件夹下的所有文件到另一个文件夹
    :param from_dir:
    :param to_dir:
    """
    if from_dir[-1] != ['/']:
        from_dir += '/'
    if to_dir[-1] != '/':
        to_dir += '/'
    from_dir += '*'
    command_line = 'mv {} {}'.format(from_dir, to_dir)
    os.system(command_line)


def remove_test_train_files_to_their_parent_dir(data_pool_path):
    """移动测试文件和训练文件到父目录
    :param data_pool_path:
    :return:
    """
    test_path = os.path.join(data_pool_path, 'labeled_ann_txt', 'test')
    train_path = os.path.join(data_pool_path, 'labeled_ann_txt', 'train')
    labeled_ann_txt_path = os.path.join(data_pool_path, 'labeled_ann_txt')
    move_all_file_from_a_dir_to_another_dir_linux(test_path, labeled_ann_txt_path)
    move_all_file_from_a_dir_to_another_dir_linux(train_path, labeled_ann_txt_path)


def text_file_empty(file_path):
    """文本文件是否为空
    :param file_path:
    :return: bool
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
        if lines:
            return False
    return True


def read_line_data(file_path, clear_empty_line=True, strip=True):
    """读取文本文件行数据
    :param file_path:
    :param clear_empty_line: 是否删除空白行
    :param strip: 是否做strip
    :return: list of string
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        if strip:
            lines = [line.strip() for line in lines]
        if clear_empty_line:
            lines = [line for line in lines if line]
        return lines


def str2bool(v):
    # copy from StackOverflow
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_entity_keys(tag_seq, char_seq, keys):
    # entity = get_entity_one_(tag_seq, char_seq)
    # return entity
    entities = []
    for key in keys:
        entities.append(get_entity_key(tag_seq, char_seq, key))
    return entities


def get_entity_key(tag_seq, char_seq, key):
    entities = []
    entity = ''
    for (char, tag) in zip(char_seq, tag_seq):
        if tag == 'B-' + key or tag == 'I-' + key or tag == 'E-' + key:
            entity += char
        else:
            if len(entity) != 0:
                entities.append(entity)
                entity = ''
    if len(entity) != 0:
        entities.append(entity)
    return entities


# 将实体提取出来
def get_entity_one_(tag_seq, char_seq):
    sequence = []
    seq = ''
    for i, tag in enumerate(tag_seq):
        if tag == 'B' or tag == 'I':
            seq += char_seq[i]
        else:
            if len(seq) != 0:
                sequence.append(seq)
                seq = ''
    if len(seq) != 0:
        sequence.append(seq)
    return sequence


def get_logger(filename):
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(format='%(message)s', level=logging.DEBUG)
    handler = logging.FileHandler(filename)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
    logging.getLogger().addHandler(handler)
    return logger



def read_two_row_excel_data_to_dict(file_path, strip=True):
    """读取两行数据为字典
    :param file_path:
    :param strip:
    :return: pre_entity_list, target_entity_list
    """
    f = pd.read_excel(file_path)
    data = np.array(f)
    result = {}
    for item in data:
        if strip:
            result[item[0].strip()] = item[1].strip()
        else:
            result[item[0]] = item[1]
    return result