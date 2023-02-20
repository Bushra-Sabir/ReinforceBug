# -*- coding: utf-8 -*-
"""
Created on Wed May 13 14:43:03 2020

@author: bushra
"""

import tensorflow as tf
import argparse
from trans import *
import csv
Datasets=['Enron','Twitter','Yelp','Toxic']
Models=["char_cnn","word_cnn","word_rnn","rcnn"]
attackname=['Reinforce']
model_path='//fast//users//a1735399//TrainedModel//'
BATCH_SIZE = 128
WORD_MAX_LEN = 300
CHAR_MAX_LEN = 1014
for attack in attackname:
    FoldDetails=open("//fast//users//a1735399//project2//Datasets//"+attack+"//transferibility_train.csv", 'a+' ,encoding='utf-8',newline='')
    details = csv.writer(FoldDetails)   
    details.writerow(['dataset','model','avg accuracy'])    
    
    for data in Datasets:
        #datasource(attack,data)
        for model in Models:
            IndivAccuracies=[]
            allpredictions=[]
            allscores=[]
            modelname=model_path+data+'//'+data+'_'+model
            print(modelname)
            if model == "char_cnn":
                test_x, test_y, alphabet_size = build_char_dataset("test", "char_cnn", CHAR_MAX_LEN,attack,data)
            else:
                word_dict = build_word_dict(data)
                test_x, test_y = build_word_dataset("test", word_dict, WORD_MAX_LEN,attack,data)
            
            checkpoint_file = tf.train.latest_checkpoint(modelname)
            graph = tf.Graph()
            with graph.as_default():
                with tf.Session() as sess:
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
                    details.writerow([data,model,str(sum_accuracy/cnt)])
    FoldDetails.close()
