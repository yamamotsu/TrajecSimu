#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 19:42:10 2017

@author: shugo
"""

# testtest


class testA():
    def __init__(self):
        self.x = 10

class testB(testA):
    def __init__(self):
        testA.__init__(self)   # 追記
        self.y = self.x + 10  # self.x で testA のインスタンス変数にアクセスできる
        print('x,y',self.x, self.y)

A = testA()
B = testB()
