# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 03:53:08 2020

@author: bushra
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 00:40:43 2020

@author: bushra
"""


# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 06:42:41 2020

@author: bushra
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 14:40:58 2020

@author: bushra
"""

   # -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 17:33:58 2020

@author: bushra
#"""
import matplotlib.pyplot as plt
from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines.common.noise import AdaptiveParamNoiseSpec, NormalActionNoise
from stable_baselines.common.callbacks import BaseCallback

from stable_baselines import PPO2
import random
import numpy as np
from collections import deque
import numpy as np
import pandas as pd
import time
import os,spacy
from nltk.tokenize import word_tokenize 
import pandas as pd
import gensim
from spacy import displacy
from collections import Counter
from gensim.models import KeyedVectors
from random import randint
import codecs,re
codecs.register_error("strict", codecs.ignore_errors)
from collections import deque
#from keras.models import Sequential
#from keras.optimizers import Adam
import pandas as pd
import pickle
import numpy as np
from word2number import w2n
from random import choice, randint,  randrange
from string import punctuation
from spacy import displacy
from collections import Counter
import tensorflow_hub as hub
import language_tool_python
tool = language_tool_python.LanguageTool('en-US')
import nltk,pickle
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
import csv
from sklearn.feature_extraction.text import CountVectorizer
_digits = re.compile('d')
import re
import numpy as np
import os
from utility_functions import *
emb_path='EMBEDDINGs PATH'
word2vec_output_file =emb_path+'glove.word2vec'
word_vectors = KeyedVectors.load_word2vec_format(word2vec_output_file)
nlp = spacy.load('en_core_web_lg') 
path='' #Where you want to safe
import pkg_resources
from symspellpy import SymSpell, Verbosity
sym_spell = SymSpell(max_dictionary_edit_distance=0, prefix_length=7)
dictionary_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
bigram_path = pkg_resources.resource_filename("symspellpy", "frequency_bigramdictionary_en_243_342.txt")
### term_index is the column of the term and count_index is the
### column of the term frequency
sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
sym_spell.load_bigram_dictionary(bigram_path, term_index=0, count_index=2)
sym_spell2 = SymSpell(5, 15)
sym_spell2.load_dictionary(dictionary_path, term_index=0, count_index=1)
sym_spell2.load_bigram_dictionary(bigram_path, term_index=0, count_index=2)
from nltk.corpus import stopwords 
nltk.download('stopwords')
nltk.download('punkt')
stopwords2=stopwords.words('english')

stopwords = set(
        ["http","https","ftp","www","com","net","because","a", "about", "above", "across", "after","get","using","we","will","very","us","www","so", "afterwards", "again", "against", "ain", "all", "almost", "alone", "along", "already", "also", "although", "am", "among", "amongst", "an", "and", "another", "any", "anyhow", "anyone", "anything", "anyway", "anywhere", "are", "aren", "aren't", "around", "as", "at", "back", "been", "before", "beforehand", "behind", "being", "below", "beside", "besides", "between", "beyond", "both", "but", "by", "can", "cannot", "could", "couldn", "couldn't", "d", "didn", "didn't", "doesn", "doesn't", "don", "don't", "down", "due", "during", "either", "else", "elsewhere", "empty", "enough", "even", "ever", "everyone", "everything", "everywhere", "except", "first", "for", "former", "formerly", "from", "hadn", "hadn't", "hasn", "hasn't", "haven", "haven't", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", "however", "hundred", "i", "if", "in", "indeed", "into", "is", "isn", "isn't", "it", "it's", "its", "itself", "just", "latter", "latterly", "least", "ll", "may", "me", "meanwhile", "mightn", "mightn't", "mine", "more", "moreover", "most", "mostly", "must", "mustn", "mustn't", "my", "myself", "namely", "needn", "needn't", "neither", "never", "nevertheless", "next", "no", "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere", "o", "of", "off", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "per", "please", "s", "same", "shan", "shan't", "she", "she's", "should've", "shouldn", "shouldn't", "somehow", "something", "sometime", "somewhere", "such", "t", "than", "that", "that'll", "the", "their", "theirs", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they", "this", "those", "through", "throughout", "thru", "thus", "to", "too", "toward", "towards", "under", "unless", "until", "up", "upon", "used", "ve", "was", "wasn", "wasn't", "we", "were", "weren", "weren't", "what", "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "with", "within", "without", "won", "won't", "would", "wouldn", "wouldn't", "y", "yet", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"]
    )

from scipy import spatial
embeddings_dict = {}
with open(emb_path+"counter-fitted-vectors.txt", 'r') as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], "float32")
        embeddings_dict[word] = vector
import nltk
nltk.download('wordnet')
   
#url=re.compile("(http[s]?://[^/]+/(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+)|(ftp?://[^/]+/(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+)|(www.+(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+)")
#url=re.compile("(http[s]?://[^/]+/(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+)|(ftp?://[^/]+/(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+)|(www.+(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+)")
import gym
from gym import spaces
url=re.compile(r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''')
import tensorflow as tf
################################################################################
##Preprocess
###############################################################################
################################################################################
##Preprocess
###############################################################################
def generate_tokens(doc):
             if(task=='Email'):
                 tokens=email_preprocessor(doc)
             elif(task=='Twitter'):
                 tokens=twitter_preprocessor(doc)
             else:
                 tokens=sms_preprocessor(doc)
             ####SAVEEE ITTT ##################################################
             return tokens
     
         #return doctokens

def get_doctokens():
        with open(path+task+"tokens.pickle", "rb") as tokens:
                 doctokens = pickle.load(tokens)
                 return doctokens
def contains_digits(d):
         return bool(_digits.search(d))

def save_replace(d,flag,name):
             value=flag.findall(d)
             count=1
             for x in value:
                     
                     try:
                         d=re.sub(str(x),name,str(d))
                     except:
                         d=str(d).replace(str(x),name)
                     count+=1
         
             return value,d
def tokenparts(words):
        finalwords=[]
        spelling_mistakes=0
        for w in words:
            w=w.lower()
            if(w not in word_vectors.vocab and w not in nlp.vocab):
                 spelling_mistakes+=1
           
            if(w not in set(stopwords) and w not in stopwords2 and len(w)>2):
               if(w!=' ' and w!=''):
                    if((w not in nlp.vocab) and (w not in word_vectors.vocab) and not (contains_digits(w)) and re.findall('[$0-9+]',w)==[]):
                       subwords=sym_spell.word_segmentation(w).corrected_string
                       if(len(subwords.split(' '))>1): 
                              for s in (subwords.split(' ')):
                                    if(s!=''):
                                        if ((s in nlp.vocab or s in word_vectors.vocab) and (s not in stopwords) and s not in stopwords2 and len(s)>2) :
                                             finalwords.append(wordnet_lemmatizer.lemmatize(s))
                                        else:
                                             
                                             if(len(s)>2):
                                                 finalwords.append(wordnet_lemmatizer.lemmatize(s))
                    else:
                        finalwords.append(wordnet_lemmatizer.lemmatize(w))
               
        if(len(words)!=0):
            spelling_mistakes=spelling_mistakes/len(words)
        return (finalwords,spelling_mistakes)     
def preprocessText(sent):
        
        filtered_sentence=''
        if(sent!=float('nan') and sent!='nan'and sent!=None):
            #sent = re.sub("'", " ", sent)  # remove single quotes
            words=word_tokenize(clean_str(sent))
            filtered_sentence,spellerrors=tokenparts(words)
                            
        return filtered_sentence,spellerrors
def texttokeys(text,Immutable):
        #url=re.compile("(http://[^/]+/(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+)")
        emailaddress=re.compile('S+@S+')
       
        Im_flags=dict()
        
        Im_flags[Immutable[0]],text=save_replace(text,url,Immutable[0])
        Im_flags[Immutable[1]],text=save_replace(text,emailaddress,Immutable[1])
        
        doc = nlp(text)
       
        for i in range(2,len(Immutable)):
            count=1
            entities=[]
            for X in doc.ents:
                 if(X.label_ ==Immutable[i]):
                    entities.append(X.text)
                    text=text.replace(X.text,Immutable[i]+' ')
                    count+=1
            Im_flags[Immutable[i]]=entities
        return Im_flags,text
def spellingerrors(sent):
         spellbee=0
         words=word_tokenize(clean_str(sent))
         for newword in words: 
             if(newword not in word_vectors.vocab and newword not in nlp.vocab):
                 spellbee+=1
         if(len(words)>0):
             return spellbee/len(words) 
         else:
             return 0  
      ##############################DATA SPECIFIC PREPROCESSOR
def twitter_preprocessor(d):
         org=d
         global identities
         spelling_error=0
         grammer_errors=0
         Immutable=['USERNAME','URL','HASTTAG']
         #url=re.compile("(http://[^/]+/(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+)")
         username=re.compile('@(?i)[a-z0-9_]+')
         hashtag=re.compile('#(w+)')
         #for d in data:
         grammer_errors=len(tool.check(d))
         immutablewords=[]
         Im_flags=dict()
         d=re.sub("\s*(\W)\s*",r"\1",d)
         Im_flags[Immutable[0]],d=save_replace(d,username,Immutable[0])
         Im_flags[Immutable[1]],d=save_replace(d,url,Immutable[1])
         Im_flags[Immutable[2]],d=save_replace(d,hashtag,Immutable[2])
         for key in Im_flags.keys():
             for values in Im_flags[key]:
#                if(key=='URL'):
#                    info = tldextract.extract(str(values))
#                    if(info.subdomain not in stopwords):
#                        stopwords.add(info.subdomain)
#                    if(info.domain not in stopwords):
#                        stopwords.add(info.domain)
#                    if(info.suffix not in stopwords):
#                        stopwords.add(info.suffix)
#                else:  
                    
                    Imwords=(word_tokenize(clean_str(values)))
                    for im in Imwords:
                         immutablewords.append(im)
         
         tokens,spelling_errors=preprocessText(d)
         DM.writerow([identities,org,Im_flags,immutablewords,tokens,spelling_errors,grammer_errors])
         doctokens.append(tokens)
         identities+=1
         #print(identities)
         
         return tokens
            
def email_preprocessor(d):
         global identities
         Immutable=['URL','EmailAddress','PERSON','ORG']
         spelling_error=0
         grammer_errors=0
         
         try: 
             grammer_errors=len(tool.check(d))
         except:
             grammer_errors=10
             pass
         org=d
         d=re.sub("\s*(\W)\s*",r"\1",d)
         Im_flags,d=texttokeys(d,Immutable)
         immutablewords=[]
         for key in Im_flags.keys():
             for values in Im_flags[key]:
#                if(key=='URL'):
#                    info = tldextract.extract(str(values))
#                    if(info.subdomain not in stopwords):
#                        stopwords.add(info.subdomain)
#                    if(info.domain not in stopwords):
#                        stopwords.add(info.domain)
#                    if(info.suffix not in stopwords):
#                        stopwords.add(info.suffix)
#                else:    
                    
                    Imwords=(word_tokenize(clean_str(values)))
                    for im in Imwords:
                        if(im not in stopwords):
                            immutablewords.append(im)
         
         tokens,spelling_errors=preprocessText(d)
         DM.writerow([identities,org,Im_flags,immutablewords,tokens,spelling_errors,grammer_errors])
         doctokens.append(tokens)
         identities+=1
        # print(identities)
         return tokens 
def sms_preprocessor(d):
         global identities
         spelling_error=0
         grammer_errors=0
         Immutable=['URL']
         org=d
         grammer_errors=len(tool.check(d))
         Im_flags=dict()
         d=re.sub("\s*(\W)\s*",r"\1",d)
         Im_flags[Immutable[0]],d=save_replace(d,url,Immutable[0])
         immutablewords=[]
         for key in Im_flags.keys():
             for values in Im_flags[key]:
#                if(key=='URL'):
#                    info = tldextract.extract(values)
#                    if(info.subdomain not in stopwords):
#                        stopwords.add(info.subdomain)
#                    if(info.domain not in stopwords):
#                        stopwords.add(info.domain)
#                    if(info.suffix not in stopwords):
#                        stopwords.add(info.suffix)
#                else:    
                    
                    Imwords=(word_tokenize(clean_str(values)))
                    for im in Imwords:
                        if(im not in stopwords):
                            immutablewords.append(im)
         tokens,spelling_errors=preprocessText(d)
         DM.writerow([identities,org,Im_flags,immutablewords,tokens,spelling_errors,grammer_errors])
         doctokens.append(tokens)
         identities+=1
         return tokens
########################################################################################
              ##Utilities and Transformations
###########################################################################################
def find_closest_embeddings(embedding):
    return sorted(embeddings_dict.keys(), key=lambda word: spatial.distance.euclidean(embeddings_dict[word], embedding))

###ACTIONS STARTED

def get_semantic_related(word,n=10):

    words=[]
    if(word in word_vectors.vocab):
        pb=word_vectors.most_similar(word,topn=30)
        for i,p in enumerate(pb):
            if(p[0].lower() not in words):
                words.append(p[0].lower()+' '+ word)
                
            
    else:
       words=[]
    #print(words)  
    return words 
def get_semantic_related1(word,n=10):
    words=[]
    if(word in word_vectors.vocab):
        pb=word_vectors.most_similar(word,topn=30)
        for i,p in enumerate(pb):
            if(p[0].lower() not in words):
                
                words.append(word+' '+ p[0].lower())
            
    else:
       words=[]
    #print(words)  
    return words 

def get_syntactic_related(word,n=10):
        words=[]
        max_edit_distance_lookup = 5
        suggestion_verbosity = n # TOP, CLOSEST, ALL
        suggestions = sym_spell2.lookup(word, suggestion_verbosity,
                                           max_edit_distance_lookup)
        for i, p in enumerate(suggestions):
                if(i<=n):
                    words.append(p.term.lower())
        
        return words
# action=['semanticreplace','delete_characters','generatehom','swap_letters','insert_punctuation','insert_duplicate_characters','delete_characters','num_to_word']
_digits = re.compile('d')
import homoglyphs as hg
homoglyphs =hg.Homoglyphs(languages={'en'},
            strategy=hg.STRATEGY_LOAD,
            ascii_strategy=hg.STRATEGY_REMOVE
        )
def generatehom(domain):
        array = np.zeros((100,len(domain)),'U1')  
        for i,each in enumerate(domain):
           hum=hg.Homoglyphs()
           listofoptions=(hum.get_combinations(each))
        
           for j,l in enumerate(listofoptions):
               
                        array[j][i]=l
                        if(i>0and j>0):
                            for k in range(i-1,-1,-1):
                                if(array[j][k]==''):
                                        ind=random.randint(0,j-1)
                                        array[j][k]=array[ind][k]
                                    
                    
           l=len(listofoptions)   
           if(j>1):
               while(array[l][i-1]!=''):
                      ind=random.randint(1,j-1)
                      array[l][i]=array[ind][i]
                      l=l+1
        domains=[]    
        
        
#       
        for row in array:
               newdm=''.join(row)
#               
#               if(newdm!='' and newdm!=domain):
#                  domains.append(newdm)
               #print(newdm)
               try:
                   newdm=homoglyphs.to_ascii(newdm)
               except:
                   pass
               for dm in newdm:
                   if(dm!='' and dm!=domain and dm not in domains):
                       domains.append(dm)
                     
        if(domain in domains):
            domains.remove(domain)
        #print('hi',domains)
        return domains  
         
qwerty = {
		'1': '2q', '2': '3wq1', '3': '4ew2', '4': '5re3', '5': '6tr4', '6': '7yt5', '7': '8uy6', '8': '9iu7', '9': '0oi8', '0': 'po9',
		'q': '12wa', 'w': '3esaq2', 'e': '4rdsw3', 'r': '5tfde4', 't': '6ygfr5', 'y': '7uhgt6', 'u': '8ijhy7', 'i': '9okju8', 'o': '0plki9', 'p': 'lo0',
		'a': 'qwsz', 's': 'edxzaw', 'd': 'rfcxse', 'f': 'tgvcdr', 'g': 'yhbvft', 'h': 'ujnbgy', 'j': 'ikmnhu', 'k': 'olmji', 'l': 'kop',
		'z': 'asx', 'x': 'zsdc', 'c': 'xdfv', 'v': 'cfgb', 'b': 'vghn', 'n': 'bhjm', 'm': 'njk'
		}
qwertz = {
		'1': '2q', '2': '3wq1', '3': '4ew2', '4': '5re3', '5': '6tr4', '6': '7zt5', '7': '8uz6', '8': '9iu7', '9': '0oi8', '0': 'po9',
		'q': '12wa', 'w': '3esaq2', 'e': '4rdsw3', 'r': '5tfde4', 't': '6zgfr5', 'z': '7uhgt6', 'u': '8ijhz7', 'i': '9okju8', 'o': '0plki9', 'p': 'lo0',
		'a': 'qwsy', 's': 'edxyaw', 'd': 'rfcxse', 'f': 'tgvcdr', 'g': 'zhbvft', 'h': 'ujnbgz', 'j': 'ikmnhu', 'k': 'olmji', 'l': 'kop',
		'y': 'asx', 'x': 'ysdc', 'c': 'xdfv', 'v': 'cfgb', 'b': 'vghn', 'n': 'bhjm', 'm': 'njk'
		}
azerty = {
		'1': '2a', '2': '3za1', '3': '4ez2', '4': '5re3', '5': '6tr4', '6': '7yt5', '7': '8uy6', '8': '9iu7', '9': '0oi8', '0': 'po9',
		'a': '2zq1', 'z': '3esqa2', 'e': '4rdsz3', 'r': '5tfde4', 't': '6ygfr5', 'y': '7uhgt6', 'u': '8ijhy7', 'i': '9okju8', 'o': '0plki9', 'p': 'lo0m',
		'q': 'zswa', 's': 'edxwqz', 'd': 'rfcxse', 'f': 'tgvcdr', 'g': 'yhbvft', 'h': 'ujnbgy', 'j': 'iknhu', 'k': 'olji', 'l': 'kopm', 'm': 'lp',
		'w': 'sxq', 'x': 'wsdc', 'c': 'xdfv', 'v': 'cfgb', 'b': 'vghn', 'n': 'bhj'
		}
keyboards = [ qwerty, qwertz, azerty ]
def bitsquatting(domain):
    result = []
    if(type(domain)!=str):
        return []
    masks = [1, 2, 4, 8, 16, 32, 64, 128]
    for i in range(0, len(domain)):
        c = domain[i]
        for j in range(0, len(masks)):
            b = chr(ord(c) ^ masks[j])
            o = ord(b)
            if (o >= 48 and o <= 57) or (o >= 97 and o <= 122) or o == 45:
                result.append(domain[:i] + b + domain[i+1:])
    return result
def insertion(domain):
        result = []
        if(type(domain)==str):
            for i in range(1, len(domain)-1):
                for keys in keyboards:
                    if domain[i] in keys:
                        for c in keys[domain[i]]:
                            result.append(domain[:i] + c + domain[i] + domain[i+1:])
            return list(set(result))
        else:
            return result
def omission(domain):
         result = []
         if(type(domain)==str):
             if(len(domain)>2):
                 for i in range(0, len(domain)):
                 			result.append(domain[:i] + domain[i+1:])
                 n = re.sub(r'(.)1+', r'1', domain)
            
                 if n not in result and n != domain:
                 	result.append(n)
                 return list(set(result))
             else:
                 return [domain]
         else:
             return []
def repetition(domain):
    result = []
    if(type(domain)==str):
        for i in range(0, len(domain)):
            if domain[i].isalpha():
                result.append(domain[:i] + domain[i] + domain[i] + domain[i+1:])
            return list(set(result))
    else:
        return []
def vowel_swap(domain): #drop it
    vowels = 'aeiou'
    result = []
    if(type(domain)!=str):
        return []
    for i in range(0, len(domain)):
        for vowel in vowels:
            if domain[i] in vowels:
                result.append(domain[:i] + vowel + domain[i+1:])
    return (result)
def vowel_swap_cons(domain):
    vowels = 'bcdfghjklmnpqrstvwxyz'
    result = []
    if(type(domain)!=str):
        return []
    for i in range(0, len(domain)):
        for vowel in vowels:
            if domain[i] in vowels:
                result.append(domain[:i] + vowel + domain[i+1:])
    return (result)
def addition(domain):
    result = []
    if(type(domain)!=str):
        return []
    for i in range(97, 123):
        result.append(domain + chr(i))
    result.append(domain + chr(32))
    return result

def transposition(domain): #drop it
    result = []
    if(type(domain)!=str):
        return []
    for i in range(0, len(domain)-1):
        if domain[i+1] != domain[i]:
            result.append(domain[:i] + domain[i+1] + domain[i] + domain[i+2:])
    return result    

import string
def checkwordtype(word,action):
    chars = re.escape(string.punctuation)
    if(action=='num_to_word'):
        return word.isnumeric()
    
    elif(action=='word_to_num'):
        try:
           w2n.word_to_num(word)
           return True
        except:
            return False
    elif(action=='word_month'):
        months_master = ['january','february','march','april','may','june','july','august','september','october','november','december']
        months_abb= ['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec']
        if(word in months_master or word in months_abb):
            return True
        else:
            return False
    elif(action=='word_weekday'):
        day_master = ['monday','tuesday','wednesday','thursday','friday','saturday','sunday']
        day_abb= ['mon','tue','wed','thur','fri','sat','sun']
        if(word in day_master or word in day_abb):
            return True
        else:
            return False
    elif(action=='get_semantic_related' or action=='get_semantic_related1'):
             number=re.sub(r'['+chars+']','',word)
             if(word in word_vectors.vocab and  number.isnumeric()==False):
                 return True
             else:
                 return False
    elif(action=='get_synonyms'):
             number=re.sub(r'['+chars+']','',word)
             if(word in nlp.vocab and   number.isnumeric()==False):
                 return True
             else:
                 return False
    elif(action=='get_syntactic_related'):
             number=re.sub(r'['+chars+']','',word)
             if(word not in nlp.vocab and  number.isnumeric()==False):
                 return True
             else:
                 return False
    else:
        return True

    
def word_to_num(word):
    try:
        return w2n.word_to_num(word)
    except:
        return ''
    
def num_to_word(number):
        replacement=[]
        ones = ("", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine")
        tens = ("", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety")
        teens = ("ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen")
        levels = ("", "thousand", "million", "billion", "trillion", "quadrillion", "quintillion", "sextillion", "septillion", "octillion", "nonillion")
    
        word = ""
        #number will now be the reverse of the string form of it
        num = reversed(str(number))
        number = ""
        for x in num:
            number += x
        del num
        if len(number) % 3 == 1: number += "0"
        x = 0
        for digit in number:
            if x % 3 == 0:
                if(int(x / 3)<len(levels)-1):
                    word = levels[int(x / 3)] + ", " + word
                    n = int(digit)
            elif x % 3 == 1:
                if digit == "1":
                    num = teens[n]
                else:
                    num = tens[int(digit)]
                    if n:
                        if num:
                            num += "-" + ones[n]
                        else:
                            num = ones[n]
                word = num + " " + word
            elif x % 3 == 2:
                if digit != "0":
                    word = ones[int(digit)] + " hundred " + word
            x += 1
        replacement.append((word.strip(", ")))
        
    
        return replacement
def currency_to_word(word):
    newOptions=[]
    if(type(word)==str):
        #path='E:/Code_Spam_Fooler/listofcurrencies.csv'
        path='path//TextualDatasets//listofcurrencies.csv'
        currenciesDataset = pd.read_csv(path,encoding="utf-8")
        currency=list(currenciesDataset['Currency'])
        Alpha=list(currenciesDataset['Alpha'])
        Symbols=list(currenciesDataset['Symbol'])
        word=word.lower()
        for i in range(0,len(currenciesDataset)):
            if(str(currency[i]).lower()==word):
                #Replace it with Alpha and Symbols
                newOptions.append(Alpha[i])
                newOptions.append(Symbols[i])
            elif(str(Alpha[i]).lower()==word):
                newOptions.append(currency[i])
                newOptions.append(Symbols[i])
            elif(str(Symbols[i]).lower()==word):
                org_word=word
                newword=word.replace(str(Symbols[i]).lower(),currency[i])
                newOptions.append(newword)
                newword=org_word.replace(str(Symbols[i]).lower(),Alpha[i])
                newOptions.append(newword)
        if(newOptions==[]):
            newOptions.append(word)
        
    return newOptions

def word_month(word):
    months_master = ['january','february','march','april','may','june','july','august','september','october','november','december']
    months_abb= ['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec']
    newword=''
    if(word in months_master):
        newword=months_abb[months_master.index(word)]
    elif(word in months_abb):
        newword=months_master[months_abb.index(word)]
    return newword   
        
def word_weekday(word):
    day_master = ['monday','tuesday','wednesday','thursday','friday','saturday','sunday']
    day_abb= ['mon','tue','wed','thur','fri','sat','sun']
    newword=''
    if(word in day_master):
        newword=day_abb[day_master.index(word)]
    elif(word in day_abb):
        newword=day_master[day_abb.index(word)]
    return newword                        
from nltk.corpus import wordnet
def get_synonyms(word):
#    synonyms=[]
    try:
        synonyms=find_closest_embeddings(embeddings_dict[word])[:30]
    except:
        synonyms=[]        
    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            if(l.name!=word):
                synonyms.append(l.name())
    
    return list(set(synonyms))
########################################################################################
    #Evaluate Semantic Similarity
########################################################################################
class USE(object):
    def __init__(self):
       
        module_url = emb_path+"universal-sentence-encoder-large_5//"
        self.embed = hub.load(module_url)
        self.sess = tf.Session(config=config)
        self.build_graph()
        self.sess.run([tf.global_variables_initializer(), tf.tables_initializer()])

    def build_graph(self):
        self.sts_input1 = tf.placeholder(tf.string, shape=(None))
        self.sts_input2 = tf.placeholder(tf.string, shape=(None))

        sts_encode1 = tf.nn.l2_normalize(self.embed(self.sts_input1), axis=1)
        sts_encode2 = tf.nn.l2_normalize(self.embed(self.sts_input2), axis=1)
        self.cosine_similarities = tf.reduce_sum(tf.multiply(sts_encode1, sts_encode2), axis=1)
        clip_cosine_similarities = tf.clip_by_value(self.cosine_similarities, -1.0, 1.0)
        self.sim_scores = 1.0 - tf.acos(clip_cosine_similarities)

    def semantic_sim(self, sents1, sents2):
        scores = self.sess.run(
            [self.sim_scores],
            feed_dict={
                self.sts_input1: sents1,
                self.sts_input2: sents2,
            })
        return scores
import itertools
##################################################################################
#TARGET CLASSIFIER
##################################################################################
class DiscriminatorEnvironment:
      def __init__(self,model,name):
          self.modelname='ModelsPath//TrainedModel//'+name+'//'+name+'_'+model
          self.mymodel=model
         # print(self.modelname)
          self.name=name
      def get_score(self,corpus):
          IndivAccuracies=[]
          allpredictions=[]
          allscores=[]
          BATCH_SIZE = 300
          WORD_MAX_LEN = 300
          CHAR_MAX_LEN = 1014
          if self.mymodel == "char_cnn":
            test_x, test_y, alphabet_size = build_char_dataset("test",CHAR_MAX_LEN,corpus)
          else:
            word_dict = build_word_dict(self.name)
            test_x, test_y = build_word_dataset("test", word_dict, WORD_MAX_LEN,corpus) #Last input is corpus
          checkpoint_file = tf.train.latest_checkpoint(self.modelname)
          graph = tf.Graph()
          with graph.as_default():
            with tf.Session(config=config) as sess:
                saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
                saver.restore(sess, checkpoint_file)
                x = graph.get_operation_by_name("x").outputs[0]
                y = graph.get_operation_by_name("y").outputs[0]
                is_training = graph.get_operation_by_name("is_training").outputs[0]
                accuracy = graph.get_operation_by_name("accuracy/accuracy").outputs[0]
                predictions = graph.get_operation_by_name("output/predictions").outputs[0]
                scores = graph.get_operation_by_name("output/scores").outputs[0]
                batches = batch_iter(test_x, test_y, BATCH_SIZE, 1)
                sum_accuracy, cnt = 0, 0
                for batch_x, batch_y in batches:
                    feed_dict = {
                        x: batch_x,
                        y: batch_y,
                        is_training: False
                    }
                    preds, s = sess.run([predictions, scores], feed_dict)
                    accuracy_out = sess.run(accuracy, feed_dict=feed_dict)
                    sum_accuracy += accuracy_out
                    cnt += 1
                    allpredictions.append(preds)
                    allscores.append(s)
                    IndivAccuracies.append(accuracy_out)
                   
                if(cnt>0):
                    
                    return allpredictions,allscores,(sum_accuracy / cnt)
                else:
                    return allpredictions,allscores,sum_accuracy    
##################################################################################
        ##Word Environment
###################################################################################
class WordEnvironment: #path,task,self.corpus,step)
    def __init__(self,path,task,data,step,replacements=20):
        self.path=path
        self.task=task
        self.step=step
        self.corpus=data
        self.maxreplacements=replacements
        if(step=='train'):
            
            if (not os.path.exists(path+"vectorizer.pickle")):
                self.vectorizer=CountVectorizer(tokenizer=generate_tokens,min_df=mindf, max_df=maxdf)
                mapp = vectorizer.fit_transform(self.corpus['text'])
                
                with open(path+step+"tokens.pickle", "wb") as f:
                    pickle.dump(doctokens, f)  
                with open(path+"vectorizer.pickle", "wb") as vec:
                        pickle.dump(self.vectorizer,vec) 
                with open(path+step+"mapping.pickle", "wb") as vec:
                        pickle.dump(mapp,vec) 
            else:
                  with open(path+"vectorizer.pickle", "rb") as f:
                      self.vectorizer=pickle.load(f)
                     
                  with open(path+step+"mapping.pickle", "rb") as vec:
                      mapp=pickle.load(vec)
        else:
                 with open(path+"vectorizer.pickle", "rb") as f:
                                  self.vectorizer=pickle.load(f)
                                  mapp=self.vectorizer.transform(self.corpus['text'])
        dmad.close()                        
        self.words=self.vectorizer.get_feature_names()  
        self.frequencies=mapp.sum(axis=0).tolist()[0]  
        self.mapping=np.transpose(mapp.toarray()) 
        #self.get_important_words()
        self.DEFINEDACTIONS=['get_synonyms','get_semantic_related','get_semantic_related1','get_syntactic_related','generatehom','bitsquatting',
         'insertion','omission','repetition','addition','num_to_word','currency_to_word','word_to_num','word_weekday','word_month']
        self.special=['num_to_word','currency_to_word','word_to_num','word_weekday','word_month']
        self.specialword=['get_synonyms','get_semantic_related','get_semantic_related1']
        #self.wordhistory=self.word_history()
        self.worddict=self.wordtoindex()
        print("Total Words in Vocabulary ",len(self.worddict))
        self.get_important_words()
        self.state=list(self.importantwords.keys())
        self.state_size=len(self.state)
        print('State Size', self.state_size)
        self.generate_actions_dict()
        self.action_size=len(self.action_space)
        
    def get_important_words(self):
        if(not os.path.exists(path+"importantwords.pickle")):
            self.importantwords=dict()
            target_classifier=DiscriminatorEnvironment(model,name)
            originalpredictions,allscores,sum_accuracy =target_classifier.get_score(self.corpus)
            oldscores=list(itertools.chain(*allscores)) 
            originalpredictions=list(itertools.chain(*originalpredictions)) 
            lab=dict()
            for k,word in enumerate(self.words):
                #kth word is in Exampleindicies
                Exampleindices=self.get_examples_for_word(k)
                
                df = pd.DataFrame(index=Exampleindices,
                data=self.corpus,
                columns=['text','label'])
                df['text']=df['text'].str.replace(word,'')
                # CORPUS WITH WORD K
                #newcorpus=pd.DataFrame(list(zip(newExamples.values(),lab.values())), columns =['text', 'label']) 
                newpredictions,newscores,new_accuracy=target_classifier.get_score(df)
                newscores=list(itertools.chain(*newscores))
                Impact_on_decision=0
                count=0
                for i in Exampleindices:
                    pscores=oldscores[i] # Old 
                    newscore=newscores[count]
                    if(df.loc[i]['label']==1):
                            previousscore=(pscores[0])
                            updatedscore=(newscore[0])
                    else:
                            previousscore=(pscores[1])
                            updatedscore=(newscore[1])
                    if(previousscore>0.4):        
                            Impact_on_decision+=(previousscore-updatedscore)/previousscore
                if(Impact_on_decision/len(Exampleindices)>=0):
                    self.importantwords[k]=[Impact_on_decision,self.frequencies[k],word]
                                
                       
            #print("STATE LENGTH########################################################",len(self.importantwords))      
            with open(path+"importantwords.pickle", "wb") as f:
                     pickle.dump(self.importantwords, f) 
            with open(path+"originalscores.pickle", "wb") as f:
                     pickle.dump(oldscores, f)  
            with open(path+"originalpredictions.pickle", "wb") as f:
                     pickle.dump(originalpredictions, f)  
        else:
                with open(path+"importantwords.pickle", "rb") as f:
                   self.importantwords=pickle.load(f)  
            
    def reset(self):
        
#        if(self.step=='train'):
#            with open(path+"vectorizer.pickle", "rb") as f:
#                     vectorizer=pickle.load(f)
#            with open(path+self.step+"mapping.pickle", "rb") as vec:
#                          mapping=pickle.load(vec)
#        else:
#              
#                dmad=open(filename, 'w' ,encoding='utf-8',newline='')
#                DM= csv.writer(dmad)
#                global identities
#                identities=1
#                DM.writerow(['id','orignal_text','immutables','tokens','spelling_errors','grammer_errors'])                 
#                with open(path+"vectorizer.pickle", "rb") as f:
#                                  vectorizer=pickle.load(f)
#                                  mapping=vectorizer.transform(self.corpus['text'])
#                dmad.close()
        self.words=self.vectorizer.get_feature_names()   
        #self.frequencies=mapping.sum(axis=0).tolist()[0]  
        #self.mapping=np.transpose(mapping.toarray()) 
        self.get_important_words()
        #self.wordhistory=self.word_history()
        self.worddict=self.wordtoindex()
        self.state=list(self.importantwords.keys())
        self.state_size=len(self.state)
        
    def wordtoindex(self):
        word_dict=dict()
        for word in self.words:
            word_dict[len(word_dict)]=word
   
        return word_dict
        
    def word_history(self):
        word_dict = dict()
        for word in self.words:
            word_dict[word] = []
        return word_dict      
    def updatewordhistory(self,key,newtoken,act):
        self.wordhistory[key].append((newtoken,act))
    
    def get_state(self):
        return np.array(self.state)
    def get_state_size(self):
        return self.state_size
    def get_action_size(self):
        return self.action_size
    def get_action_space(self):
        return self.action_space,self.Word_Action_Mapping
#    def get_word_action_mapping(self):
#        return self.Word_Action_Mapping
    def get_value(self,val):
          for key, value in self.worddict.items(): 
              if val == value: 
                  return key 
    def update_state(self,wordtochange,newtoken):
        
        self.words[wordtochange]=newtoken
        index=self.state.index(wordtochange)
        #print('index is', index)
        if(newtoken not in self.worddict.values()):
                val=len(self.worddict)
                self.worddict[val]=newtoken
                #print('val is', val, 'index is', index)
                #print('self.state ', self.state[index])
                self.state[index]=val
                #self.state[index]+=1
        else:
                self.state[index]=self.get_value(newtoken)
                #self.state[index]+=1
    def get_examples_for_word(self,index): # Search Example to be updated
        Example_Word_Map=self.mapping[index]
        Examplesindex=[]
        for i,e in enumerate(Example_Word_Map):
            if(e>0):
                Examplesindex.append(i)
        return Examplesindex                
    
    ####DEFINE ACTION SPACE
    def perform_action(self,wordtochange,actiontoperform,replacementindex):
         oldword=self.worddict[wordtochange]
         newword=self.find_replacement(actiontoperform,oldword,replacementindex)
         if(newword==oldword):
             newtoken=''
         else:
             newtoken=newword
         return oldword,newtoken
    def generate_actions_dict(self):
        if(not os.path.exists(path+"actionspace.pickle")):
           #self.action_space,self.Word_Action_Mapping=
            self.action_space=dict()
            self.Word_Action_Mapping=dict()
            count=0
            for word_index in self.state:
                  Action_for_word=[]
                  word=self.worddict[word_index]
                  #if(round(impact,2)>0):
                  if(word not in stopwords):
                      for act in self.DEFINEDACTIONS:
                              if(checkwordtype(word,act)): # if the action is valid against the word
#                                  if(act in self.special):
#                                      maxreplacements=1
#                                  elif(act in self.specialword):
#                                      maxreplacements=10
#                                  else:
#                                      maxreplacements=self.maxreplacements
                                  replacements=set(self.find_replacement(act,word))
                                  
                                  for k,replace in enumerate(replacements):
                                      
#                                      if(k>maxreplacements):
#                                          break
                                      if(replace!=word and replace!=''):
                                          print(word," ",act," ",replace)
                                          self.action_space['action'+str(count)]=[word_index,act,replace]
                                          Action_for_word.append(count)
                                          count+=1
                      print(len(Action_for_word))
                      self.Word_Action_Mapping[word_index]=Action_for_word
            
            with open(path+"actionspace.pickle", "wb") as f:
                     pickle.dump(self.action_space, f) 
            with open(path+"Word_Action_Mapping.pickle", "wb") as f:
                     pickle.dump(self.Word_Action_Mapping, f)  
        else:
            with open(path+"actionspace.pickle", "rb") as f:
                   self.action_space=pickle.load(f)  
            with open(path+"Word_Action_Mapping.pickle", "rb") as f:
                   self.Word_Action_Mapping=pickle.load(f) 
            
#        return Actions,Word_Action_Mapping
    
    
    def find_replacement(self,actiontoperform,wordtochange):
        listofreplacement=[]
        if(not checkwordtype(wordtochange,actiontoperform)):
            return ''
        elif(actiontoperform=='get_synonyms'):
            listofreplacement=get_synonyms(wordtochange)
        elif(actiontoperform=='get_semantic_related'):
            listofreplacement=get_semantic_related(wordtochange)
        elif(actiontoperform=='get_syntactic_related'):
            listofreplacement=get_syntactic_related(wordtochange)
        elif(actiontoperform=='get_semantic_related1'):
            listofreplacement=get_semantic_related1(wordtochange)
        ##Character Perturbation
        elif(actiontoperform=='generatehom'):
             listofreplacement=generatehom(wordtochange)
        elif(actiontoperform=='bitsquatting'):
            listofreplacement=bitsquatting(wordtochange)
        elif(actiontoperform=='insertion'):
            listofreplacement=insertion(wordtochange)
        elif(actiontoperform=='omission'):
            listofreplacement=omission(wordtochange)
        elif(actiontoperform=='repetition'):
            listofreplacement=repetition(wordtochange)
        elif(actiontoperform=='vowel_swap'):
            listofreplacement=vowel_swap(wordtochange)
        elif(actiontoperform=='vowel_swap_cons'):
            listofreplacement=vowel_swap_cons(wordtochange)
        elif(actiontoperform=='addition'):
            listofreplacement=addition(wordtochange)
        elif(actiontoperform=='transposition'):
            listofreplacement=transposition(wordtochange)
        # Name Entity Perturbation
        elif(actiontoperform=='num_to_word'):
            listofreplacement=num_to_word(wordtochange)
        elif(actiontoperform=='currency_to_word'):
            listofreplacement=currency_to_word(wordtochange)
        elif(actiontoperform=='word_to_num'):
            listofreplacement=[word_to_num(wordtochange)]
        elif(actiontoperform=='word_weekday'):
            listofreplacement=[word_weekday(wordtochange)]
        elif(actiontoperform=='word_month'):
            listofreplacement=[word_month(wordtochange)]
#        if(len(listofreplacement)>1):
#            select_random=randint(0,len(listofreplacement)-1)
#            return [listofreplacement[select_random]]
#        else:
        #print(len(listofreplacement),replacementindex)
        if((type(listofreplacement)==list)):
            
            return listofreplacement
        elif(type(listofreplacement)!=list and listofreplacement!=''):
            return [listofreplacement]
        else:
            return []
######################################################################################################################################################
                 
             ##ATTACK
from gym.utils import seeding                 
             ##ATTACK
class Attacker_Corpus(gym.Env):
     def __init__(self,task,seedfile,filename,model,step,name):
            super(Attacker_Corpus, self).__init__()
            # Path of the Corpus
            self.corpus = pd.read_csv(seedfile,encoding="utf-8")
            self.corpus=self.corpus.dropna().reset_index(drop=True)
           
            self.miss=0
            self.reward_range=(-100, 100) 
            self.num_envs=8
            #dataset=dataset.dropna()
            self.name=name
            #'id','orignal_text','immutables','tokens','spelling_errors','grammer_errors'
            self.task=task # The type of data the corpus represent
            
            #Initialize the word environment
            self.word_env=WordEnvironment(path,task,self.corpus,step,5)
            self.actions,self.Word_Action_Mapping=self.word_env.get_action_space()
            ########According to Agent
            self.action_space=spaces.Discrete(len(self.actions))
            self.word_state=self.word_env.get_state()
            
            self.state=self.word_state.copy() # No word is changed
            self.state_flag=np.zeros(len(self.word_state),dtype=np.int32)
            self.observation_space=spaces.Box(low=0, high=20000,
                                        shape=(1,len(self.state)), dtype=np.bool)
            #Initialize the Discriminator environment
            self.indexofalreadyAdv=[]
            self.original=self.corpus.copy()
            self.target_classifier=DiscriminatorEnvironment(model,name)
            if(not os.path.exists(path+step+"originalscores.pickle") or step!='train'):
                self.originalpredictions,allscores,sum_accuracy =self.target_classifier.get_score(self.corpus)
                allscores=list(itertools.chain(*allscores))
                self.originalpredictions=list(itertools.chain(*self.originalpredictions))
                with open(path+step+"originalscores.pickle", "wb") as f:
                     pickle.dump(allscores, f)  
                with open(path+mode+"originalpredictions.pickle", "wb") as f:
                     pickle.dump(self.originalpredictions, f)  
            else:
                with open(path+step+"originalscores.pickle", "rb") as f:
                     allscores=pickle.load(f)
                with open(path+step+"originalpredictions.pickle", "rb") as f:
                     self.originalpredictions=pickle.load(f)

            self.score=self.Spam_score_dict(allscores)
            self.original_score=self.score.copy() # CONFIDENCE OF CLASSIFIER ON DECISION 
            self.queries=1
            totalexamples=list(range(0,len(self.score)))
            self.word_changes=dict.fromkeys(totalexamples, 0)
            self.wordreplacements=dict.fromkeys(totalexamples, [])
            self.replacementactions=dict.fromkeys(totalexamples, [])
            #self.averageSpellingErrors=#functio
            ###Post Constraints
            self.sim_predictor = USE()
            self.sim_score_threshold=0.60
            self.original_performance_score=self.spamCorpusScore()
            self.valid_actions=np.ones(len(self.actions),dtype=bool)
            ###Post Constraints
            dataset = pd.read_csv(filename,encoding="utf-8")
            self.spellings=list(dataset['spelling_errors'])
            self.grammers=list(dataset['grammer_errors'])
            self.doctokens=list(dataset['tokens'])
            self.immutalbe_doctokens=list(dataset['immutable_tokens'].str.replace('[^\w\s]',''))
            del dataset
            #self.averageGrammerErrors=max(self.grammers)#sum(list(dataset['grammer_errors']))/len(list(dataset['grammer_errors'])) #Threshold
            #self.averageSpellingErrors=max(self.spellings)#sum(list(dataset['spelling_errors']))/len(list(dataset['spelling_errors']))
            #print("Avg Grammer", self.averageGrammerErrors, "Avg Spelling:", self.averageSpellingErrors)
        
            
        #GET STATE that agent can observe according to the goal if the goal is benign observe the spam score i.e., confidence value
     def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
     def observe_state(self):
         return self.state
     def get_state_size(self):
         return len(self.state)
     def get_action_size(self):
         return self.word_env.get_action_size()
     def check_word_status(self,wordtochange):
         # Find the list where value is wordtochange and then replace the index flag value to False
         #print(wordtochange)
         if(type(self.word_state)!=list):
             index=self.word_state.tolist().index(wordtochange)
         else:
             index=self.word_state.index(wordtochange) 
        
         if(self.state_flag[index]==True):
             return True
         else:
             return False
     def update_word_status(self,wordtochange):
         # Find the list where value is wordtochange and then replace the index flag value to False
         if(type(self.word_state)!=list):
             index=self.word_state.tolist().index(wordtochange)
         else:
             index=self.word_state.index(wordtochange) 
         self.state_flag[index]=self.state_flag[index]+1
     def numberofAdvGenerated(self):
         count=0
         for key in self.score.keys():
             if(key not in self.indexofalreadyAdv):
                     if(self.score[key]<=0.4):
                         count=count+1
             
         print("Total Adversarial Examples generated in the episode",count,"Total Examples ", len(self.score))
         return count      
     def reset(self):
         
         global Episodes
         print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@EpisodeStarts", Episodes)
         Episodes+=1
         #self.corpus=self.original.copy()
         self.word_env.reset()
         self.word_state=self.word_env.get_state()
         self.modelpredictions=self.originalpredictions.copy()
         #self.score=self.original_score.copy()
         #self.queries=1
         #totalexamples=list(range(0,len(self.score)))
         #self.word_changes=dict.fromkeys(totalexamples, 0)
         #self.wordreplacements=dict.fromkeys(totalexamples, [])
         #self.replacementactions=dict.fromkeys(totalexamples, [])
         self.state=self.word_state.copy() # No word is changed
         self.state_flag=np.zeros(len(self.word_state),dtype=np.int32)
         self.valid_actions=np.ones(len(self.actions),dtype=bool)
         self.observation_space=spaces.Box(low=0, high=20000,
                                        shape=(1, len(self.state)), dtype=np.bool)
         #self.actions,self.Word_Action_Mapping=self.word_env.get_action_space()
            ########According to Agent
         #self.action_space=spaces.Discrete(len(self.actions))
         return self.state
     def update_valid_actions(self,wordindex):
         action_ids=self.Word_Action_Mapping[wordindex]
         for i in action_ids:
             self.valid_actions[int(i)]=False
             
     def step(self,actionid):
            #Game ends in two cases either all words are change or new_score of corpus > 0.9
         previous_score=self.spamCorpusScore() # Previous scores of the corpus Average confidence of the classifer on its prediction 80% or 90%
         new_score=previous_score #intially new score is equal to previous_score  
         log=open(logfile, 'a+' ,encoding='utf-8',newline='')
         logwriter= csv.writer(log)                  
         try:
             assert True in self.valid_actions
         except:
             #print("#####################################################################Done here" ,len(self.state))
             done=True
             return self.state,-100,done,{'action_mask':list(self.valid_actions)}

         reward=-100 # Invalid
         done=False
         change=0
         [wordtochange,actiontoperform,replacementindex]=self.actions['action'+str(actionid)]
         #If wordindex is already changed by the action all other actions on it are invalid
         oldword=self.word_env.worddict[wordtochange]
         Previously_Changed=self.check_word_status(wordtochange)
#         #If word is not changed already
         if(Previously_Changed==False):
             Exampleindices=self.word_env.get_examples_for_word(wordtochange)
             Exampleindicies_tochange=self.check_status(Exampleindices,oldword) #Select only those example whose spam score >0.3
             #Writing Logs
             oldword=''
             newtoken=''
             
             
             if(Exampleindicies_tochange!=[] ):
                     
                     oldword=self.word_env.worddict[wordtochange]
                     newtoken=replacementindex
                     # invalid action
                     if(newtoken!='' and newtoken!=[]):
                             AdvExamples,actualImpacted,reward=self.calculate_effectiveness_of_candidate(wordtochange,oldword,Exampleindicies_tochange,newtoken,actiontoperform)
                             #update the state, new_score, 
                             new_score=self.spamCorpusScore() # new score updated 
                             self.update_word_status(wordtochange)
                             self.update_valid_actions(wordtochange) 
                             self.word_env.update_state(wordtochange,newtoken)
                             self.state=self.word_env.get_state()
                             if(reward>0):
                                
                                 
                                 print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@The reward is ", reward)
                                 logwriter.writerow([str(actiontoperform),str(replacementindex),str(oldword),str(newtoken),str(reward),str(len(Exampleindicies_tochange)),str(len(AdvExamples)),str(len(actualImpacted)),str(previous_score),str(new_score)])
                             else:
                                 self.valid_actions[actionid]=False
                                 reward=-1000
                                 #print("############################Invalid action ",wordtochange,actiontoperform,replacementindex)
             else: 
                 #print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$Stuck Problem")
                 self.update_word_status(wordtochange)
                 self.update_valid_actions(wordtochange) 
                 reward=0
         #else:
         #       print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$Stuck Previously changed problem ")
         totalwordschanged= (self.state_flag == True).sum()
         if(totalwordschanged==len(self.state_flag)):
             done=True
         #validactions=(self.valid_actions==True).sum()
            #Update valid actions position replaced
         print(Previously_Changed,"   @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@totalwordschange",totalwordschanged)
         if(new_score<=0.4) :
                 #print("totalwordschange",totalwordschanged,"uffffo", len(self.state_flag),"new_score ", new_score)
                 done=True  
                 reward=reward+1000

         log.close()
         return self.state,reward,done,{'action_mask':list(self.valid_actions)}
         
     def check_status(self,Examples,word):
         toupdate=[]
         for i in Examples:
             if(self.score[i]>0.5 and i not in self.indexofalreadyAdv and word not in self.immutalbe_doctokens[i]):
                 toupdate.append(i)
         return toupdate
                 
     def queries(self):
         return self.queries
     def check_semantic_similarity(self,sent1,sent2):
          semantic_sims = self.sim_predictor.semantic_sim([sent1],[sent2])[0]
          return semantic_sims[0]
    
     def compute_diff(self, sent1, sent2):
         org_word=dict()
         
         original=sent1.split()
         adversarial=sent2.split()
         for i, org in enumerate(original):
             if org!=adversarial[i]:
                 org_word[org]=adversarial[i]
         return org_word
                 
     def calculate_effectiveness_of_candidate(self,wordtochange,oldword,Exampleindicies_tochange,newtoken,action):
         AdvExamples=dict()
         actualimpacted=[]
         wordlength=dict()
         replacement_scores=dict()
         spelling_errors=dict()
         grammar_errors=dict()
         #spellingchange=dict()
         #grammarchange=dict()
         label=[]
         df = pd.DataFrame(index=Exampleindicies_tochange,
                 data=self.corpus,
                 columns=['text','label'])
#         for d in list(df['text']):
#             df[l]
         df['text']=df['text'].str.replace(str(oldword),str(newtoken))
         file1 = open(path+model+"_logging_final.txt","a+") 
         print("$$$$$$$$$$$$$$$$$$$$$ length of change ", len(Exampleindicies_tochange))
         start = time.time()
         newspellingerrors=0
         for i in Exampleindicies_tochange:
                 newexample=df.loc[i]['text']#.replace(str(oldword),str(newtoken))
                 oldexample=self.corpus.loc[i]['text']
                 replacement_score=self.check_semantic_similarity(oldexample,newexample)
                 wordlength[i]=len(self.doctokens[i])
                 grammarerrorsinnewexample=tool.check(newexample)
                 newgrammarscore=len(grammarerrorsinnewexample)/wordlength[i]
                 #oldgrammarscore=self.grammers[i]/wordlength[i]
#                 if(oldgrammarscore!=0):
#                     change_grammer_score=(newgrammarscore-oldgrammarscore)/oldgrammarscore# NewGrammererrors-oldgrammererrors
#                 else:
#                     change_grammer_score=(newgrammarscore-oldgrammarscore)
#                     
                 #oldspellingerrors=self.spellings[i]
                 newspellingerrors=spellingerrors(newexample)
                     
                 
#                 if(oldspellingerrors!=0):
#                         change_spelling_score=(newspellingerrors-oldspellingerrors)/oldspellingerrors# NewSpellingerrors-oldSpellingerrors
#                 else:
#                         change_spelling_score=(newspellingerrors-oldspellingerrors)
#                         
                 if(replacement_score>=self.sim_score_threshold and newgrammarscore<=0.50 and newspellingerrors<=0.50):
                          AdvExamples[i]=newexample
                          label.append(df.loc[i]['label'])
                          replacement_scores[i]=replacement_score
                          spelling_errors[i]=newspellingerrors
                          grammar_errors[i]=newgrammarscore
                          #spellingchange[i]=change_spelling_score
                          #grammarchange[i]=change_grammer_score
                          
         
         reward=0
         end = time.time()
         print(f"Runtime of the program is {end - start}")  
         if(AdvExamples):
               #print("\nAdvExamples are present that satisfy post const "+str(len(AdvExamples))+'\n')
               advcorpus=pd.DataFrame(list(zip(AdvExamples.values(),label)), columns =['text', 'label']) 
               allpredictions,allscores,sum_accuracy=self.target_classifier.get_score(advcorpus)  
               newpredictions=list(itertools.chain(*allpredictions)) 
               allscores=list(itertools.chain(*allscores)) 
               self.queries+=1  
               scores=self.get_tg_score(allscores,AdvExamples)
               count=0
               adv=open(AdversarialFile, 'a+' ,encoding='utf-8',newline='')
               ad= csv.writer(adv)
               
               for i in AdvExamples.keys():
                     diff=round((self.score[i]-scores[i])/self.score[i],3)# difference in previous score of the examples
                     #print("\nAdvExamples ",diff,'\n')
                     if(diff>0):
                         
                         actualimpacted.append(i)
                         self.corpus.loc[i,'text']=AdvExamples[i] #Corpus is updated
                         self.word_env.mapping[wordtochange][i]=0 # update mapping
                         file1.write("\nExample is "+str(i)+" old score was: "+str(self.score[i]))
                         self.score[i]=scores[i] #Score is updated 
                         file1.write("\nExample is "+str(i)+" new score is: "+str(self.score[i]))
                         self.word_changes[i]+=1
                         previousvalue=self.wordreplacements[i].copy()
                         previousvalue.append([oldword,newtoken])
                         self.wordreplacements.update({i:previousvalue})
                         
                         previousvalueaction=self.replacementactions[i].copy()
                         previousvalueaction.append(action)
                         self.replacementactions.update({i:previousvalueaction})
                         
                         #print('change is in corpus')
                     global Episodes
                     if(scores[i]<=0.5 and self.original.iloc[i]['label']!=newpredictions[count]+1):
                         error=(spelling_errors[i]+grammar_errors[i])
                         if(error<0):
                             error=0
                         scorechange=((self.original_score[i]-scores[i])/self.original_score[i])*100
                         perturbrate=(self.word_changes[i]/wordlength[i])*100
                         reward+=((scorechange+(replacement_scores[i]*100))/(perturbrate+(error*100)))
                         
                         print("ADV GENERATED")
                         #ad.writerow(['original_id','Adversary','oldscore','newscore','changedict','numberofperturbedwords','previousoutput','newoutput','originallabel','semanticscore','beforegrammarerrors','aftergrammarerrors','beforespellingerrors','afterspellingerrors'])
                         ad.writerow([Episodes,str(i+1),AdvExamples[i],self.original_score[i],scores[i],self.replacementactions[i],self.wordreplacements[i],self.word_changes[i],str(self.modelpredictions[i]+1),str(newpredictions[count]),str(self.original.iloc[i]['label']),str(replacement_scores[i]),str(self.grammers[i]//wordlength[i]),str(grammar_errors[i]),str(self.spellings[i]),str(spelling_errors[i])])
                         file1.write("\nAdversarial Example against "+str(i+1)+" has been generated:"+"The original prediction of model was:"+ str(self.modelpredictions[i])+" new prediction is "+str(newpredictions[count])+"\n new  Original score was "+str(self.original_score[i])+" New score is "+str(scores[i])+'\n')
                         file1.write("\nNumber of total words perturbed: "+ str(self.word_changes[i])+"\n")
                     
                     else:
                         reward+=diff
                     count+=1
               adv.close()
         # Effect of the word replacement on examples containing the word replaced (local reward)
         
         file1.close()          
         return AdvExamples,actualimpacted,reward/len(Exampleindicies_tochange)
     def get_tg_score(self,scores,Exampleindices):
          Spam_score=dict()
          count=0
          for i in Exampleindices.keys():
              if(self.original.iloc[i]['label']==1):
                   score=scores[count]
                   Spam_score[i]=(score[0])
                   
              else:
                  score=scores[count]
                  Spam_score[i]=(score[1])
              count+=1
          return Spam_score

     def Spam_score_dict(self,scores):
          Spam_score=dict()
          allreadyAdv=0
          file1 = open(path+model+"_logging_final.txt","a+") 
          for i,score in enumerate(scores):
              if(self.original.iloc[i]['label']==1):
                  Spam_score[i]=(score[0])
                  if(score[0]<0.5):
                      file1.write(" Already Adversarial "+ str(i)+"\n")
                      allreadyAdv+=1
                      self.indexofalreadyAdv.append(i)
              else:
                  Spam_score[i]=(score[1])
                  if(score[1]<0.5):
                      file1.write(" Already Adversarial "+ str(i)+"\n")
                      allreadyAdv+=1
                      self.indexofalreadyAdv.append(i)
          file1.write(" Total Already Adversarial "+ str(allreadyAdv)+"\n")  
          file1.close()
          return Spam_score
     #def computeSemanticSimilarity(self):
     def spamCorpusScore(self):
          spamcorpus=0
          for s in self.score.keys():
              spamcorpus+=self.score[s]
          return spamcorpus/len(self.score)
     def get_reward(self):
         return self.reward
     def get_queries(self):
         return self.queries
     def get_valid_actions(self):
        """ Get a vector of valid actions """
        valid_actions = self.valid_actions
        
        return valid_actions

################################################
#
######################################################
task='TaskName' # For example'Email'
name='DatasetName' #For example 'Enron'
model='Modelname'#For example 'rcnn'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config) 
#keras.backend.set_session(sess)
path='Directory to save model and log files'
## clear logs
file1 = open(path+model+"test_logging_final.txt","a+") 
file1.close()
seedfile=path+'testseed.csv' # Path of the Corpus
###Called only onces
logfile=path+'test_AnalysisLog_'+task+model
log=open(logfile, 'a+' ,encoding='utf-8',newline='')
logwriter= csv.writer(log)
logwriter.writerow(['actiondone','replacementindex','oldword','newword','actual_reward','totalExwithword','totalExSatPost','totalExscoreimpacted','overallpreviousscore','currentscore'])
log.close()
AdversarialFile=path+'test_Adversarial'+task+model
adv=open(AdversarialFile, 'a+' ,encoding='utf-8',newline='')
ad= csv.writer(adv)
ad.writerow(['Episode','original_id','Adversary','oldscore','newscore','actions','changedict','numberofperturbedwords','previousoutput','newoutput','originallabel','semanticscore','beforegrammarerrors','aftergrammarerrors','beforespellingerrors','afterspellingerrors'])
adv.close()
mindf=0.001
maxdf=0.99
doctokens=[]
mode='test'
filename=path+mode+'reinforcebug.csv'
if(mode=='train'):
           if (not os.path.exists(path+"vectorizer.pickle") or not os.path.exists(filename)):
                dmad=open(filename, 'w' ,encoding='utf-8',newline='')
                DM= csv.writer(dmad)
                identities=1
                DM.writerow(['id','orignal_text','immutables','immutable_tokens','tokens','spelling_errors','grammer_errors'])

else:
                
                dmad=open(filename, 'w' ,encoding='utf-8',newline='')
                DM= csv.writer(dmad)
                identities=1
                DM.writerow(['id','orignal_text','immutables','immutable_tokens','tokens','spelling_errors','grammer_errors'])
                
        

num_envs = 8

env_id='Reinforce-v0'
num_cpu=2
import gym
from stable_baselines.common.vec_env import DummyVecEnv
#import torch
def register():
    env_name = env_id
    if env_name in gym.envs.registry.env_specs:
        del gym.envs.registry.env_specs[env_name]

    gym.envs.register(
        id=env_name,
        entry_point=Attacker_Corpus,
         max_episode_steps=3000,
    )
register()
batch_size = 8
num_envs = 8
#num_gpus = torch.cuda.device_count()
#print('Gpus',num_gpus)
print('Making Environment')
#def make_env(index):
#    return lambda: gym.make(env_id, device=torch.device('cuda', index=index % num_gpus))
print('Making Vec Environment')
env = DummyVecEnv([lambda: Attacker_Corpus(task,seedfile,filename,model,mode,name)])
print('Valid Actions Settings')
# env.get_valid_actions = lambda: np.array([e.get_valid_actions() for e in env.envs])
env.get_valid_actions = lambda: np.array(env.env_method('get_valid_actions'))
log_dir=path+'log/'
os.makedirs(log_dir, exist_ok=True)
from stable_baselines.common.vec_env import VecEnvWrapper
import json
class VecMonitor(VecEnvWrapper):
    EXT = "monitor.csv"
    
    def __init__(self, venv, filename=None, keep_buf=0, info_keywords=()):
        VecEnvWrapper.__init__(self, venv)
        print('init vecmonitor: ',filename)
        self.eprets = None
        self.eplens = None
        self.epcount = 0
        self.tstart = time.time()
        if filename:
            self.results_writer = ResultsWriter(filename, header={'t_start': self.tstart},
                extra_keys=info_keywords)
        else:
            self.results_writer = None
        self.info_keywords = info_keywords
        self.keep_buf = keep_buf
        if self.keep_buf:
            self.epret_buf = deque([], maxlen=keep_buf)
            self.eplen_buf = deque([], maxlen=keep_buf)

    def reset(self):
        obs = self.venv.reset()
        self.eprets = np.zeros(self.num_envs, 'f')
        self.eplens = np.zeros(self.num_envs, 'i')
        return obs

    def step_wait(self):
        obs, rews, dones, infos = self.venv.step_wait()
        self.eprets += rews
        self.eplens += 1

        newinfos = list(infos[:])
        for i in range(len(dones)):
            if dones[i]:
                info = infos[i].copy()
                ret = self.eprets[i]
                eplen = self.eplens[i]
                epinfo = {'r': ret, 'l': eplen, 't': round(time.time() - self.tstart, 6)}
                for k in self.info_keywords:
                    epinfo[k] = info[k]
                info['episode'] = epinfo
                if self.keep_buf:
                    self.epret_buf.append(ret)
                    self.eplen_buf.append(eplen)
                self.epcount += 1
                self.eprets[i] = 0
                self.eplens[i] = 0
                if self.results_writer:
                    self.results_writer.write_row(epinfo)
                newinfos[i] = info
        return obs, rews, dones, newinfos
        
class ResultsWriter(object):
    def __init__(self, filename, header='', extra_keys=()):
        print('init resultswriter')
        self.extra_keys = extra_keys
        assert filename is not None
        if not filename.endswith(VecMonitor.EXT):
            if os.path.exists(filename):
                filename = os.path.join(filename, VecMonitor.EXT)
            else:
                filename = filename #   + "." + VecMonitor.EXT
        self.f = open(filename, "wt")
        if isinstance(header, dict):
            header = '# {} \n'.format(json.dumps(header))
        self.f.write(header)
        self.logger = csv.DictWriter(self.f, fieldnames=('r', 'l', 't')+tuple(extra_keys))
        self.logger.writeheader()
        self.f.flush()

    def write_row(self, epinfo):
        if self.logger:
            self.logger.writerow(epinfo)
            self.f.flush()       
Episodes=1
env = VecMonitor(env, log_dir)
print('Creating CustomLSTMPolicy')
#env = Monitor(env, path, allow_early_resets=True)
#policy_kwargs = dict(act_fun=tf.nn.relu,net_arch=[64,64])
agent = PPO2(policy='MlpPolicy', env=env, n_steps=3072, nminibatches=8,
                 lam=0.90, gamma=0.99, noptepochs=4, ent_coef=0,
                 learning_rate=lambda f: f * 2.5e-5, cliprange=lambda f: f * 0.1, verbose=1,tensorboard_log=path)

agentname=log_dir+model+'ReinforceBug'
print('Reloading')
agent=PPO2.load(agentname)
print('Reset Environment')
agent.set_env(env)
print('learning')
#agent.learn(agentname,total_timesteps=1000*30000) #Change done
#agent.save(agentname)
print('testing')
obs = env.reset()
state = None
total_rewards = 0
done = [False for _ in range(env.num_envs)]
print('testing 1 2 3')
for i in range(1000000):
    action, _states = agent.predict(obs, state=state, mask=done)
    obs, rewards, done, info = env.step(action)
    total_rewards += rewards
    if done:
            file1 = open(path+name+model+"logging_final.txt","a+") 
            file1.write('EP: '+ str(i))
            file1.write(' Episode reward: '+str(total_rewards)+'\n')
            file1.write(' Episode queries: '+ str(env.get_queries()))
            file1.write(' Episode misses: '+ str(env.miss)+'\n')
            file1.close()
            break
print(total_rewards)
