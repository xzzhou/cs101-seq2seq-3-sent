#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 17:49:04 2018

@author: Xinghao
"""

a = [1.2324,2.454,3.4324,4.4342,5.6546,6.6546]

with open('./result/a.txt', 'w', encoding = 'utf-8') as f:
    for i in range(len(a)):
        f.write(str(a[i]) + '\n')
