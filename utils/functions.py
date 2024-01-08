#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2023/4/23 上午11:13
# @Author  : PeiP Liu
# @FileName: functions.py
# @Software: PyCharm

def dict2str(src_str):
    dst_str = ''
    for key in src_str.keys():
        dst_str += " %s: %.4f " % (key, src_str[key]) # 将字典转换成为字符串，并在value值保留４位数值
    return dst_str

class Storage(dict):
    """
    将字典扩充为类似于静态函数的样式，直接使用字典类对象访问key的value信息
    """
    def __getattr__(self, key):
        try:
            return self[key] if key in self else False
        except KeyError as k:
            raise AttributeError(k)

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]

        except KeyError as k:
            raise AttributeError(k)

    def __str__(self):
        return '<' + self.__class__.__name__ + dict.__repr__(self) + '>'
