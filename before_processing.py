#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""前处理
"""
import copy
from common import Icd10BigDiseaseClass
from structures import copy_feature_rate



class BeforeProcessing(object):
    """
    """

    def __init__(self, one_symptom_to_disease_matrix, disease_big_class_code_dict):
        """
        """
        self._one_symptom_to_disease_matrix = one_symptom_to_disease_matrix
        self._icd_10_big_disease = Icd10BigDiseaseClass(disease_big_class_code_dict)

        self._generate_data()

    def _generate_data(self):
        """
        """
        one_symptom_to_disease_dict = {}
        for row in self._one_symptom_to_disease_matrix:
            feature_rate = copy_feature_rate()
            feature_rate['featureName'] = row[1]
            feature_rate['bigClass'] = self._icd_10_big_disease.search_big_class(row[2])
            feature_rate['extraProperty'] = row[3]
            feature_rate['desc'] = '{\"可能诊断\":\"\"}'
            one_symptom_to_disease_dict[row[0]] = feature_rate

        self._one_symptom_to_disease_dict = one_symptom_to_disease_dict

    def symptoms_to_disease(self, symptoms):
        """
        :param symptoms:
        :return: disease, string
        """
        if len(symptoms) == 1:  # 只处理一个症状的情况
            disease_obj = self._one_symptom_to_disease_dict.get(list(symptoms)[0])
            if disease_obj:
                return [copy.deepcopy(disease_obj)]

        return []


