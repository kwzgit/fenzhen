#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""关键词抽取
"""


class KeyWordsExtractor(object):
    """
    """

    def __init__(self, automaton, ner_model, terminology_mapping_model, word_to_graph_word_dict):
        """
        """
        self._automaton = automaton
        self._ner_model = ner_model
        self._terminology_mapping_model = terminology_mapping_model
        self._word_to_graph_word_dict = word_to_graph_word_dict

    def extract_key_words(self, free_text, structured_words=set()):
        """提取关键词
        :param free_text: 自由文本
        :param structured_words: 结构化关键词集合
        :return: key words set , set of string
        """
        automaton_key_words = self._automaton.match(free_text)
        ner_key_words = self._ner_model.extract(free_text)
        # 并集
        candidate_key_words = ner_key_words | structured_words | automaton_key_words  # 自动机词、结构化词、ner词可能有非标准词
        candidate_key_words = self._remove_covered_words(candidate_key_words)  # 删除被覆盖的词
        origin_standard_key_words, other_key_words = self.check_in(candidate_key_words)
        origin_mapped_key_words = set()
        if len(other_key_words) >0:
            origin_mapped_key_words = set([(i,i) for i in other_key_words])

        # origin_mapped_key_words = self._terminology_mapping_model.mapping(other_key_words)
        origin_standard_key_words |= origin_mapped_key_words

        return origin_standard_key_words

    def _remove_covered_words(self, candidate_key_words):
        """过滤被覆盖的词，比如：[剑突下疼痛、疼痛]  则删除'疼痛'
        :param candidate_key_words:
        :return:
        """
        remove_list = set()
        for word in candidate_key_words:
            for word_2 in candidate_key_words:
                if word_2 != word and word_2.find(word) > -1:
                    remove_list.add(word)
        left_words = set()
        for word in candidate_key_words:
            if word not in remove_list:
                left_words.add(word)

        return left_words

    def check_in(self, key_words):
        """分成在图谱中（包括同义词在图谱中）和不在两部分
        :param key_words:
        :return:
        """
        _in_set = set()
        _not_in_set = set()
        for key_word in key_words:
            if key_word in self._word_to_graph_word_dict:
                _in_set.add((key_word, self._word_to_graph_word_dict.get(key_word)))
            else:
                _not_in_set.add(key_word)
        return _in_set, _not_in_set
