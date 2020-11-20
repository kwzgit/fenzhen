#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""ac自动机
"""
import ahocorasick

class ACAutomaton(object):
    """
    """
    def __init__(self, words, stop_words):
        """
        :param words:
        :param stop_words:
        """
        self._words = words
        self._stop_words = stop_words

        self._build()

    def _build(self):
        """生成自动机
        """
        automaton = ahocorasick.Automaton()
        for word in self._words:
            automaton.add_word(word, word)
        automaton.make_automaton()
        self.automaton = automaton


    def match(self, string):
        """匹配
        :param string:
        :return:
        """
        name_list = set()
        for item in self.automaton.iter(string):  # 将AC_KEY中的每一项与content内容作对比，若匹配则返回
            name_list.add(item[1])
        name_list = list(name_list)
        final_wds = set()
        if len(name_list) > 0:
            stop_wds = []
            target_wds = name_list
            for wd1 in target_wds:
                for wd2 in target_wds:
                    if wd1 in wd2 and wd1 != wd2:
                        stop_wds.append(wd1)
            final_wds = [i for i in target_wds if i not in stop_wds]

        return set(final_wds)  # 最终匹配的词语

