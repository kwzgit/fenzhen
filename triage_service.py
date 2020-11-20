#!/usr/bin/python3
# -*- coding: utf-8 -*-
# --------------------------------
# Name triage_service
# Author DELL
# Date  2020/11/18
# -------------------------------
import os
import re
import time
import traceback
import operator
from ac_automate import ACAutomaton
from terminology_mapping_model import TerminologyMappingModelInterface
from common import generate_symptom_to_graph_symptom_dict,read_one_symptom_to_disease_matrix
from before_processing import BeforeProcessing
from after_processing import AfterProcessing
from text_pre_processing import TextPreProcessing
from graph import GraphObj
from config import DATA_PATH, GRAPH_PWD, GRAPH_URL, GRAPH_USER
from utils import read_line_data, read_two_row_excel_data_to_dict
from key_words_extraction import KeyWordsExtractor
from medical_entity_recognition.bilstm_crf.ner_interface import NERModel

def response_status_related_setting_before(request_param, response_json):
    """响应对象状态设置[前]
    :param request_param:
    :param response_json:
    :return:
    """
    if request_param.get('length') is None:
        response_json['ret'] = -2  # token丢失
    elif request_param.get('length') <= 0:
        response_json['ret'] = -1  # 参数错误

    if request_param.get('featureType') is None:
        response_json['ret'] = -2  # token丢失
    else:
        feature_types = request_param['featureType']
        feature_types = re.split('[,，]', feature_types)
        for feature_type in feature_types:
            valid = False
            # 推送类型(多选必填)，1：症状，4：查体结果，42：查体指标，5：检验，6：检查，7：诊断， 8：治疗方案，22：核心指标、危机值、开单合理项
            if feature_type in {'1', '7'}:  # {'1', '4', '42', '5', '6', '7', '8', '22'} 逐个增加
                valid = True
            if not valid:
                response_json['ret'] = -1

    chief = request_param.get('chief')
    symptom = request_param.get('symptom')

    if chief:
        chief = chief.strip()
    if symptom:
        symptom = symptom.strip()

    if chief is None and symptom is None:  # 主诉、现病史皆为空
        response_json['ret'] = -2  # token丢失
    elif not (chief or symptom):  # 参数皆为空
        response_json['ret'] = -1  # 参数错误

    response_json['startTime'] = int(round(time.time() * 1000))  # 开始时间


def response_status_related_setting_after(response_json, predict_info):
    """响应对象状态设置[后]
    :param response_json:
    :param predict_info:
    :return:
    """
    response_json['endTime'] = int(round(time.time() * 1000))  # 结束时间
    response_json['msg'] = predict_info['msg']
    response_json['ret'] = predict_info['ret']
    response_json['data']['symptom'].extend(predict_info['symptom'])  # 症状添加
    response_json['data']['dis'].extend(predict_info['dis'])  # 疾病添加

class Service(object):

    def __init__(self, small_big_merge=True):
        """
        """
        self._small_big_merge = small_big_merge
        self._graph = GraphObj(GRAPH_URL, GRAPH_USER, GRAPH_PWD)
        file_path = os.path.join(DATA_PATH, 'symptoms.txt')
        standard_key_words = set(read_line_data(file_path))
        automaton = ACAutomaton(standard_key_words, [])
        terminology_mapping_model = TerminologyMappingModelInterface()
        ner_model = NERModel()
        symptom_to_graph_symptom_dict = generate_symptom_to_graph_symptom_dict(self._graph)  # 词到图谱词的映射
        # 文本中抽取词
        self.key_words_extractor = KeyWordsExtractor(automaton, ner_model, terminology_mapping_model,
                                                     symptom_to_graph_symptom_dict)
        disease_big_class_code_dict = read_two_row_excel_data_to_dict(
            os.path.join(DATA_PATH, 'disease_big_class_icd_10_code.xlsx'))  # 疾病大小类映射
        # 一个症状推送一个疾病
        one_symptom_to_disease_matrix = read_one_symptom_to_disease_matrix(
            os.path.join(DATA_PATH, 'one_symptom_to_disease.xlsx'))
        # 双症状不推疾病
        self.double_symptom_noPush_disease = read_one_symptom_to_disease_matrix(
            os.path.join(DATA_PATH, 'double_symptom_no_push.xlsx'))

        self._before_processor = BeforeProcessing(one_symptom_to_disease_matrix, disease_big_class_code_dict)

        # 加载后处理模块
        frequency_dict = self._graph.disease_distribution()  # 疾病频率
        disease_department = self._graph.disease_department()  # 疾病科室
        disease_icd_dict = self._graph.disease_icd()  # 疾病编码
        disease_sex = self._graph.disease_sex()  # 疾病性别
        disease_age = self._graph.disease_age()  # 疾病年龄

        self._after_processor = AfterProcessing(disease_big_class_code_dict=disease_big_class_code_dict,
                                                frequency_dict=frequency_dict,
                                                disease_department_dict=disease_department,
                                                disease_icd_dict=disease_icd_dict,
                                                disease_sex_dict=disease_sex,
                                                disease_age_dict=disease_age
                                                )

    def _extract_key_words(self, free_text, structured_words):
        """
        :param free_text:
        :param structured_words:
        :return:
        """
        free_text = TextPreProcessing.processing(free_text)
        origin_standard_words = self.key_words_extractor.extract_key_words(free_text, structured_words)  # [(原词，标准词), ...]
        return origin_standard_words


    def _query_and_processing_diseases(self, symptoms, length, sex, age):
        """
        :param symptoms:
        :return:
        """
        disease_dict = self._graph.query_symptoms_to_diseases(symptoms)
        self._after_processor.disease_filter_by_sex(disease_dict, sex)
        self._after_processor.disease_filter_by_age(disease_dict, age)
        diseases, _ = self._after_processor.sub_set_intersection_union(disease_dict, length)
        return diseases


    def _get_disease_sort_by_department_vote(self, symptoms, length, sex, age):
        """获取疾病，按科室投票排名
        :param symptoms:
        :param length:
        :param sex:
        :return:
        """
        disease_obj_list = []
        diseases = self._query_and_processing_diseases(symptoms, length, sex, age)  # 最多1000个疾病
        groups = self._after_processor.sort_by_department_vote(diseases)
        groups = [self._after_processor.sorted_by_frequency(items) for items in groups]
        for group in groups:
            disease_obj_list.extend(group)
            if len(disease_obj_list) >= length:
                break
        disease_obj_list = [self._after_processor.wrap_disease(disease, self._small_big_merge) for disease in
                            disease_obj_list[:length]]
        return disease_obj_list


    def disease_push(self, param):
        """疾病推送
        :param param: 参数
        :return: disease set
        """
        free_text = param.get('free_text')
        structured_words = param.get('structured_words')
        length = param.get('length')
        sex = param.get('sex')
        age = param.get('age')
        symptoms = set(
            [origin_standard[-1] for origin_standard in self._extract_key_words(free_text, structured_words)])
        disease_obj_list = []
        if len(symptoms) == 1:
            print('一个症状，症状为',symptoms)
            disease_obj_list = self._before_processor.symptoms_to_disease(symptoms)  # 一个症状到疾病
            print('推送的疾病为：',disease_obj_list)
            if len(disease_obj_list) > 0:
                return disease_obj_list
        # 双症状符合，不推送疾病
        if len(symptoms) == 2:
            reverse_symptoms = list(symptoms)[::-1] # 反转
            for row in self.double_symptom_noPush_disease:
                row_ = str(row[0])
                split = row_.split('，')
                if operator.eq(split, list(symptoms)) or operator.eq(split, reverse_symptoms):
                    return disease_obj_list
        if not disease_obj_list:  # 科室投票排名
            disease_obj_list = self._get_disease_sort_by_department_vote(symptoms, length, sex, age)

        return disease_obj_list  # 最多推length个


    def symptom_push(self, param):
        """症状推送
        :param param: 推送个数
        :return: disease set
        """
        free_text = param.get('free_text')
        structured_words = param.get('structured_words')
        length = param.get('length')
        sex = param.get('sex')
        age = param.get('age')
        symptoms = set([origin_standard[-1] for origin_standard in self._extract_key_words(free_text, structured_words)])
        diseases = self._query_and_processing_diseases(symptoms, length, sex, age)
        diseases = self._after_processor.sorted_by_frequency(diseases)  # 按频率排序

        symptoms_query = self._graph.query_diseases_to_symptoms(diseases)
        symptoms_query -= symptoms  # 去除一开始带来的症状

        symptom_obj_list = []
        for symptom in symptoms_query:
            symptom_obj_list.append(self._after_processor.wrap_symptom(symptom))
            if len(symptom_obj_list) >= length:  # 最多推length个
                break

        return symptom_obj_list





def wrapper(service_obj: Service, request_param, response_json):
    """ 包装预测结果
    :param service_obj: 服务对象
    :param request_param: 请求对象
    :param response_json: 请求对象
    :return:
    """
    response_status_related_setting_before(request_param, response_json)

    info = {'ret': 0, 'symptom': [], 'dis': [], 'msg': 'MSG_SUCCESS'}
    if response_json['ret'] not in (-1, -2):  # 如果数据有效
        length = request_param['length']
        chief = request_param.get('chief', '')
        present = request_param.get('symptom', '')
        chief += present
        sex = request_param.get('sex', 'A')
        age = request_param.get('age',0)
        param = {'free_text': chief,
                 'structured_words': set(),
                 'length': length,
                 'sex': sex,
                 'age': age
                 }
        try:
            feature_types = request_param['featureType']
            feature_types = re.split('[,，]', feature_types)
            for feature_type in feature_types:
                if feature_type == '1':
                    symptoms = service_obj.symptom_push(param)
                    info['symptom'].extend(symptoms)
                if feature_type == '7':
                    diseases = service_obj.disease_push(param)
                    info['dis'].extend(diseases)
        except:
            traceback.print_exc()
            info['ret'] = 1
            info['msg'] = 'MSG_FALURE'

    response_status_related_setting_after(response_json, info)
