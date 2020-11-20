#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""后处理
"""
from itertools import combinations
from common import Icd10BigDiseaseClass
from structures import copy_feature_rate, copy_extracted_word


class AfterProcessing(object):
    """
    """

    def __init__(self, disease_big_class_code_dict, frequency_dict, disease_department_dict, disease_icd_dict,
                 disease_sex_dict,disease_age_dict,
                 stop_rule='length'):
        """
        """
        self._icd_10_big_disease = Icd10BigDiseaseClass(disease_big_class_code_dict)
        self._disease_department_dict = disease_department_dict
        self._frequency_dict = frequency_dict
        self._stop_rule = stop_rule
        self._disease_icd_dict = disease_icd_dict
        self._disease_sex_dict = disease_sex_dict
        self._disease_age_dict = disease_age_dict
        assert self._stop_rule in ('length', 'not_empty')  # length 满足长度，not_empty非空即返回

    def disease_filter_by_sex(self, node_related_nodes_dict, sex):
        """疾病过滤，用新街
        :param node_related_nodes_dict: {'症状':[疾病s], ...}
        :param sex:
        :return:
        """
        if sex == 'M':
            sex = '男'
        elif sex == 'F':
            sex = '女'
        else:
            sex = None

        if sex:
            for symptom, disease_list in node_related_nodes_dict.items():
                remove_diseases = set()
                for disease in disease_list:
                    _disease_sex = self._disease_sex_dict.get(disease)
                    if _disease_sex and _disease_sex != sex:
                        remove_diseases.add(disease)
                for disease in remove_diseases:
                    disease_list.remove(disease)

    def disease_filter_by_age(self, node_related_nodes_dict, age):
        """疾病过滤，用新街
        :param node_related_nodes_dict: {'症状':[疾病s], ...}
        :param age:
        :return:
        """
        if age:
            for symptom, disease_list in node_related_nodes_dict.items():
                remove_diseases = set()
                for disease in disease_list:
                    _disease_age = self._disease_age_dict.get(disease)
                    if _disease_age:
                        split_age = str(_disease_age).split('-')
                        if age < int(split_age[0]) or age > int(split_age[1]):
                            remove_diseases.add(disease)
                for disease in remove_diseases:
                    disease_list.remove(disease)


    def sub_set_intersection_union(self, node_related_nodes_dict, length):
        """
            从大到小，做出所有key的子集，计算出它们value的交集，如果交集非空且stop_until_not_empty==True就返回
            否则，继续求子集的交集
        :param node_related_nodes_dict: {'症状':[疾病s], ...}
        :param length: 返回元素个数
        :return:
        """

        all_keys = list(node_related_nodes_dict.keys())
        result = set()
        data = []  # [len(sub_keys), sub_keys, values]
        for i in range(len(all_keys), 0, -1):
            sub_keys = combinations(all_keys, i)
            for keys in sub_keys:  # 同一等级的子集内(元素个数一样)求交集
                temp_set = set(node_related_nodes_dict[keys[0]])
                for key in keys[1:]:
                    temp_set &= set(node_related_nodes_dict[key])
                result |= temp_set  # 同一等级的子集间(元素个数一样)求并集
                data.append((len(keys), keys, temp_set))
            if self._stop_rule == 'not_empty' and result:
                return result, data
            if self._stop_rule == 'length' and len(result) >= length:
                return result, data

        return result, data

    def sorted_by_frequency(self, items):
        """按频率排序
        :param items:
        :return:
        """
        items_with_frequency = []
        for item in items:
            items_with_frequency.append((item, self._frequency_dict.get(item, 0.0)))
        items_with_frequency.sort(key=lambda x: x[-1], reverse=True)

        return [item_with_frequency[0] for item_with_frequency in items_with_frequency]


    def sort_by_department_vote(self, items):
        """按科室投票排序
        :param items:
        :return:
        """
        department_disease_dict = {}
        for item in items:
            department = self._disease_department_dict.get(item, '空白科室')
            department_disease_dict.setdefault(department, [])
            department_disease_dict[department].append(item)
        department_disease_list = sorted(department_disease_dict.values(), key=lambda x:len(x), reverse=True)
        return department_disease_list

    def get_big_class_for_disease(self, disease_name):
        """获取大类名称
        :param disease_name:
        :return: string
        """
        icd_10_code = self._disease_icd_dict.get(disease_name)
        return self._icd_10_big_disease.search_big_class(icd_10_code)

    def wrap_disease(self, disease_name, add_big_class=True):
        """包装疾病
        :param disease_name:
        :param add_big_class: 是否添加大类
        :return: featureRate 对象，来自于structures模块
        """
        feature_rate = copy_feature_rate()
        feature_rate['featureName'] = disease_name
        feature_rate['desc'] = '{\"可能诊断\":\"\"}'
        if add_big_class:
            feature_rate['bigClass'] = self.get_big_class_for_disease(disease_name)
        feature_rate['extraProperty'] = self._disease_department_dict.get(disease_name, '全科')
        return feature_rate

    def wrap_symptom(self, symptom_name):
        """ 包装症状
        :param symptom_name:
        :return: featureRate 对象，来自于structures模块
        """
        feature_rate = copy_feature_rate()
        feature_rate['featureName'] = symptom_name
        return feature_rate

    def wrap_extracted_words(self, origin_standard, type_value=1):
        """
        :param origin_standard:
        :param type_value:
        :return:
        """
        extract_words_obj = copy_extracted_word()
        extract_words_obj['standard'] = origin_standard[1]
        extract_words_obj['origin'] = origin_standard[0]
        extract_words_obj['type'] = type_value
        return extract_words_obj