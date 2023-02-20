# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 00:56:53 2022

@author: bushra
"""
import pickle
with open('D:/New Studies Plan/GLAD (EMNLP-2022) 786/adv_text.pkl', "rb") as tokens:
                 adv_text=pickle.load(tokens)
                 print(adv_text)
