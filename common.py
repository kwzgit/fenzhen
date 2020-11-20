#!/usr/bin/python3
# -*- coding: utf-8 -*-
# --------------------------------
# Name common
# Author DELL
# Date  2020/11/19

# -------------------------------
import os
import numpy as np
import pandas as pd
from config import DATA_PATH
from utils import read_two_row_excel_data_to_dict

class Icd10BigDiseaseClass(object):
    """icd-10疾病大类
    """

    def __init__(self, disease_big_class_code_dict):
        """
        :param disease_big_class_code_dict:
        """
        self._disease_big_class_code_dict = disease_big_class_code_dict

        self._generate_search_structure()

    def _generate_search_structure(self):
        """
        """
        code_search_structure_dict = {}
        for code_range, disease_big_class in self._disease_big_class_code_dict.items():
            first_char = code_range[0]
            code_search_structure_dict.setdefault(first_char, set())
            code_range = code_range.split('-')
            code_search_structure_dict[first_char].add((code_range[0], code_range[1], disease_big_class))
        self._code_search_structure_dict = code_search_structure_dict

    def search_big_class(self, icd_10_code):
        """
        :param icd_10_code:
        :return: disease_big_class, a string
        """
        if icd_10_code:
            first_char = icd_10_code[0]
            code_search_structure_sub_set = self._code_search_structure_dict.get(first_char)
            if code_search_structure_sub_set is not None:
                for code_search_structure in code_search_structure_sub_set:
                    if code_search_structure[0] <= icd_10_code <= code_search_structure[1]:
                        return code_search_structure[-1]
        return None

# 同义词映射到标准词
def generate_symptom_to_graph_symptom_dict(graph):
    """生成症状到图症状映射字典
    :return:
    """
    #图谱中的症状
    graph_symptoms = set(graph.symptom_fetch())
    #标准词---同义词
    symptom_synonym_dict = read_symptom_synonym()
    symptom_to_graph_symptom_dict = {}
    for standard_symptom, symptoms in symptom_synonym_dict.items():
        if standard_symptom not in graph_symptoms:
            # print(standard_symptom)
            pass
        else:
            for symptom in symptoms:
                symptom_to_graph_symptom_dict[symptom] = standard_symptom  # 同义词到标准词
    for symptom in graph_symptoms:
        symptom_to_graph_symptom_dict[symptom] = symptom  # 图谱词到自身的映射

    return symptom_to_graph_symptom_dict


def read_symptom_synonym(file_path=os.path.join(DATA_PATH, 'symptom_synonym.xlsx')):
    """
    :param file_path:
    :return:
    """
    symptom_synonym_dict = read_two_row_excel_data_to_dict(file_path)
    for standard_symptom, symptoms in symptom_synonym_dict.items():
        symptoms = symptoms.split('，')
        symptoms = [symptom.strip() for symptom in symptoms if symptom.strip()]
        symptom_synonym_dict[standard_symptom] = set(symptoms)

    return symptom_synonym_dict


def read_one_symptom_to_disease_matrix(file_path):
    """
    :param file_path:
    :return:
    """
    f = pd.read_excel(file_path)
    f = f.fillna(0) # 把pandas中的NAN填充0
    data = np.array(f)
    return data




if __name__ == '__main__':
    read_symptom_synonym()