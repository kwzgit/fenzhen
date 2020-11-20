#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
"""
from py2neo import Node, Relationship, Graph, NodeMatch


class GraphObj(object):
    """
    """

    def __init__(self, url, username, password):
        """
        :param url:
        :param username:
        :param password:
        """
        self._url = url
        self._username = username
        self._password = password

        self._build()

    def _disease_filter(self, disease):
        """严重疾病过滤
        :param disease:
        :return:
        """
        if disease.find('恶性') > -1 and disease.find('肿') > -1:
            return True
        if disease.find('艾滋病') > -1:
            return True
        if disease.find('癌') > -1:
            return True
        if disease.find('肿瘤') > -1:
            return True

        return False

    def _build(self):
        """
        """
        try:
            self._graph = Graph(
                self._url,
                username=self._username,
                password=self._password
            )
        except:
            print("无法连接数据库，请确认数据库是否正常运行！")

    def query_symptoms_to_diseases(self, symptoms):
        """
        :param symptoms:
        :return:
        """
        disease_dict = {}
        # 过滤掉皮肤科和血液内科疾病
        # cql = 'MATCH (n:`症状`)<-[:`疾病相关症状`]-(m:`疾病`) WHERE n.name=' + "'{}'" + '  return m.name'
        cql = 'MATCH (n:`症状`)<-[:`疾病相关症状`]-(m:`疾病`) ,(m:`疾病`)-[r:疾病相关标准科室]->(d) ' \
              'WHERE n.name=' + "'{}'" + ' and not d.name in ["皮肤科","血液内科"]  return m.name'
        for symptom in symptoms:
            diseases = self._graph.run(cql.format(symptom)).data()
            dis_set = []
            for dic in diseases:
                disease = dic['m.name']
                if not self._disease_filter(disease):
                    dis_set.append(disease)
            disease_dict[symptom] = dis_set

        return disease_dict

    def query_symptoms_relation_value(self,symptom):
        relation_value = {}
        cql = 'MATCH(d:症状)-[r]->(v) where d.name="{}" return type(r),collect(v.name)'
        rows_data = self._graph.run(cql.format(symptom)).data()
        for row in rows_data:
            relation_value[row['type(r)']] = row['collect(v.name)']

        return relation_value

    def query_diseases_to_symptoms(self, diseases, top_n=2):
        """
        :param diseases:
        :param top_n:
        :return: set()
        """
        symptom_set = set()
        cql = 'MATCH (n:`症状`)<-[:`疾病相关症状`]-(m:`疾病`) WHERE m.name=' + "'{}'" + '  return n.name'
        for disease in diseases[:top_n]:
            symptoms = self._graph.run(cql.format(disease)).data()
            sym_set = set()
            for sym in symptoms:
                sym_set.add(sym['n.name'])
            symptom_set |= sym_set

        return symptom_set


    def disease_distribution(self):
        """
        :return: dict {disease: frequency or count, ...}
        """
        cql = 'MATCH (n:`发病率`)<-[:`发病情况`]-(m:`疾病`) return n.name, m.name'
        distribution = {}
        data = self._graph.run(cql).data()
        for item in data:
            distribution[item['m.name']] = float(item['n.name'])
        return distribution

    def disease_icd(self):
        """
        :return: dict {disease: frequency or count, ...}
        """
        cql = 'MATCH (n:`编码`)<-[:`疾病相关编码`]-(m:`疾病`) return n.name, m.name'
        distribution = {}
        data = self._graph.run(cql).data()
        for item in data:
            distribution[item['m.name']] = item['n.name']
        return distribution

    def disease_department(self):
        """
        :return: dict {disease: department, ...}
        """
        cql = 'match (n:`疾病`)-[r:`疾病相关标准科室`]->(w:`科室`) return n.name, w.name'
        department_dict = {}
        data = self._graph.run(cql).data()
        for item in data:
            department_dict[item['n.name']] = item['w.name']
        return department_dict


    def disease_sex(self):
        """
        :return: dict {disease: department, ...}
        """
        cql = 'match (n:`疾病`)-[r:`疾病相关性别`]->(w:`性别`) return n.name, w.name'
        department_dict = {}
        data = self._graph.run(cql).data()
        for item in data:
            department_dict[item['n.name']] = item['w.name']
        return department_dict


    def disease_age(self):
        """
        :return: dict {disease: department, ...}
        """
        cql = 'match (n:`疾病`)-[r:`疾病相关年龄`]->(w:`年龄`) return n.name, w.name'
        department_dict = {}
        data = self._graph.run(cql).data()
        for item in data:
            department_dict[item['n.name']] = item['w.name']
        return department_dict


    def symptom_fetch(self):
        """获取症状
        :return:
        """
        cql = 'MATCH (n:`症状`) RETURN n.name'
        all_symptoms = []
        data = self._graph.run(cql).data()
        for item in data:
            all_symptoms.append(item['n.name'])
        return all_symptoms