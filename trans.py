# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 12:14:53 2020

@author: bushra
"""

# -*- coding: utf-8 -*-
"""
Created on Sun May 10 20:28:52 2020

@author: bushra
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 08:04:34 2020

@author: bushra
"""

import os
import tarfile
import re
from nltk.tokenize import word_tokenize
import collections
import pandas as pd
import pickle
import numpy as np
import codecs
codecs.register_error("strict", codecs.ignore_errors)
vocapath='/fast/users/a1735399/project2/Datasets/vocab/'
TEST_PATH=''
def isnan(value):
    try:
        import math
        return math.isnan(float(value))
    except:
        return False


def clean_str(text):
    if(isnan(text)):
           text=''
    text = re.sub(r"[^A-Za-z0-9(),!?\'\`\"]", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = text.strip().lower()
    
    return text

def build_word_dict(data):
    
    with open(vocapath+data+"word_dict.pickle", "rb") as f:
            word_dict = pickle.load(f)

    return word_dict

def build_word_dataset(step, word_dict, document_max_len,attack,data):
    TEST_PATH='/fast/users/a1735399/project2/Datasets/'+attack+'/'+data+'.csv'
    df =  pd.read_csv(TEST_PATH,encoding="utf-8")
    df=df.fillna('')
    x = list(map(lambda d: word_tokenize(clean_str(d)), df["adversarial"]))
    x = list(map(lambda d: list(map(lambda w: word_dict.get(w, word_dict["<unk>"]), d)), x))
    x = list(map(lambda d: d + [word_dict["<eos>"]], x))
    x = list(map(lambda d: d[:document_max_len], x))
    x = list(map(lambda d: d + (document_max_len - len(d)) * [word_dict["<pad>"]], x))
    df=df[pd.notnull(df['originallabel'])]
    labels=df["originallabel"]
    y = list(map(lambda d: int(d) - 1, list(labels)))
    x=x[:len(y)]
    #print("Length y ",len(y))
    #print("Length x ",len(x))
    return x, y

     
    return x, y


def build_char_dataset(step, model, document_max_len,attack,data):
    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:’'\"/|_#$%ˆ&*˜‘+=<>()[]{} "
    TEST_PATH='/fast/users/a1735399/project2/Datasets/'+attack+'/'+data+'.csv'
    df =  pd.read_csv(TEST_PATH,encoding="utf-8")
    df=df.fillna('')
    # Shuffle dataframe
    df = df.sample(frac=1)
    char_dict = dict()
    char_dict["<pad>"] = 0
    char_dict["<unk>"] = 1
    for c in alphabet:
        char_dict[c] = len(char_dict)

    alphabet_size = len(alphabet) + 2

    x = list(map(lambda content: list(map(lambda d: char_dict.get(str(d), char_dict["<unk>"]), content.lower())), df["adversarial"]))
    x = list(map(lambda d: d[:document_max_len], x))
    x = list(map(lambda d: d + (document_max_len - len(d)) * [char_dict["<pad>"]], x))
    labels=df["originallabel"]
    y = list(map(lambda d: int(d) - 1 , list(labels)))
    x=x[:len(y)]
    return x, y, alphabet_size


def batch_iter(inputs, outputs, batch_size, num_epochs):
    inputs = np.array(inputs)
    outputs = np.array(outputs)

    num_batches_per_epoch = (len(inputs) - 1) // batch_size + 1
    for epoch in range(num_epochs):
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, len(inputs))
            yield inputs[start_index:end_index], outputs[start_index:end_index]
