#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""一些结构
"""
import copy

AdverseReactionDetail = {
    'name': '',
    'value': ''
}

Medicition = {
    'medicitionName': '',
    'isShow': '',
    'forbidden': '',
    'showInfo': '',
    'rate': '',
    'type': ''
}

MedicitionClass = {
    'showInfo': '',
    'drugsForbidden': '',
    'bigdrugsName': '',
    'bigdrugsType': '',
    'subdrugsName': '',
    'subdrugsType': '',
    'medicitionList': ''
}

MeditionDetail = {
    'description': '',
    'treatment': []  # MedicitionClass
}

AdverseReaction = {
    'name': '',
    'type': '',
    'showInfo': '',
    'controlType': '',
    'details': [],  # AdverseReactionDetail
}

TreatmentPlan = {
    'title': '',
    'meditionDetails': []  # MeditionDetail
}

MedicalIndicationDetail = {
    'type': '',  # 类型
    'content': ''  # 详细内容
}

Treat = {
    'diseaseName': '',
    'disType': '',
    'treatmentPlan': [],  # TreatmentPlan
    'adverseEvent': [],  # AdverseReaction
}

MedicalIndication = {
    'name': '',  # 名称
    'details': []  # 明细, MedicalIndicationDetail
}

FeatureRate = {
    'featureName': None,  # 名称
    'extraProperty': None,  # 附加属性
    'bigClass': None,  # 附加属性
    'desc': None,  # 描述
    'rate': None  # 概率
}

ResponseData = {
    'symptom': [],  # 伴随症状, FeatureRate
    'history': [],  # 病史, FeatureRate
    'vitals': [],  # 推荐查体, FeatureRate
    'labs': [],  # 推荐化验, FeatureRate
    'pacs': [],  # 推荐检查, FeatureRate
    'dis': [],  # 推荐诊断, FeatureRate
    'hasIndications': '0',  # 有无病情提示标识
    'medicalIndications': [],  # 病情提示MedicalIndication
    'treat': Treat,  # 治疗方案
    'scale': None,  # 量表
    'managementEvaluation': None  # 管理评估
}

PushResponse = {
    'status': 'PENDING',  # 状态
    'msg': 'MSG_FALURE',  # 消息，MSG_SUCCESS(操作成功), MSG_FALURE(操作失败)
    'startTime': '',  # 开始时间
    'endTime': '',  # 结束时间
    'version': '',  # 版本
    'ret': '1',  # 返回结构标识, 0：成功，1：失败，-1：参数错误，-2：token丢失
    'data': ResponseData,  # 返回对象
}

precData = {
    'controlType':0,
    'name' :'',
    'content':'',
    'questionMapping':[]
}

PrecResponse = {
    'code': '0',  # 状态
    'msg': 'MSG_FALURE',  # 消息，MSG_SUCCESS(操作成功), MSG_FALURE(操作失败)
    'data': precData,  # 返回对象
}



extracted_words = {
    'standard': None,  # 标准词
    'type': None,  # 1- 症状
    'origin': None  # 原始词
}


def copy_extracted_word():
    """
    """
    return copy.deepcopy(extracted_words)


def copy_response_structure():
    """
    """
    return copy.deepcopy(PushResponse)

def copy_response_structure_pre():
    """
    """
    return copy.deepcopy(PrecResponse)


def copy_feature_rate():
    """
    :return:
    """
    return copy.deepcopy(FeatureRate)
