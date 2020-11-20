#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""术语映射接口
"""


class TerminologyMappingModelInterface(object):
    """
    """

    def __init__(self):
        """
        """
        pass

    def mapping(self, terminology_sets):
        """
        :param terminology_sets:
        :return: terminology set, set of string
        """
        return set()  # [(原词，标准词), ...]

