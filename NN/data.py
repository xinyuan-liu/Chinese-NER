# -*- coding: utf-8 -*-
import numpy as np
import pickle
X_train=pickle.load(open("pkl/X_train.pkl",'rb'))
Y_train=pickle.load(open("pkl/Y_train.pkl",'rb'))
index_word=pickle.load(open("pkl/index_word.pkl",'rb'))
dict_word=pickle.load(open("pkl/dict_word.pkl",'rb'))
index_tag=pickle.load(open("pkl/index_tag.pkl",'rb'))
dict_tag=pickle.load(open("pkl/dict_tag.pkl",'rb'))
from keras.preprocessing.sequence import pad_sequences

def get_X(filepath):
    f=open(filepath,'r')
    x_sen=[]
    word_sen=[]
    X_test=[]
    X_word=[]
    for line in f:
        #print(line)
        line=line.strip()
        if line == "":
            X_test.append(x_sen)
            X_word.append(word_sen)
            x_sen=[]
            word_sen=[]
            continue
        line=line.split(' ')
        word_sen.append(line[0])
        if line[0] in dict_word:
            x_sen.append(dict_word[line[0]])
        else:
            x_sen.append(1)
    X_test_cut=[]
    X_test_len=[]
    max_sen_len=100
    for i in range(len(X_test)):
        if len(X_test[i])<=max_sen_len:
            X_test_cut.append(X_test[i])
            X_test_len.append(len(X_test[i]))
            continue
        while len(X_test[i])>max_sen_len:
            flag=False
            for j in reversed(range(100)):
                if X_test[i][j]==dict_word['，'] or X_test[i][j]==dict_word['、']:
                    X_test_cut.append(X_test[i][:j+1])
                    X_test_len.append(j+1)
                    X_test[i]=X_test[i][j+1:]
                    break
                if j==0:
                    flag=True
            if flag:
                X_test_cut.append(X_test[i][:100])
                X_test[i]=X_test[i][100:]
                X_test_len.append(100)
    X_test_cut=pad_sequences(X_test_cut,maxlen=max_sen_len,padding='post')
    f.close()
    return X_test_cut,X_test_len,X_word

def write_result_to_file(filepath,Y_pred,X_test_len,X_word):
    f=open(filepath,'w')
    i2=0
    for i1 in range(len(X_word)):
        j2=0
        for j1 in range(len(X_word[i1])):
            f.write(X_word[i1][j1]+' ')
            tags=Y_pred[i2][j2]
            tag=0
            for i in range(8):
                if(tags[i]==1):
                    tag=i
            if tag==0:
                tag=1
            f.write(index_tag[tag]+'\n')
            j2+=1
            if j2 == X_test_len[i2]:
                j2=0
                i2+=1
        f.write('\n')
    f.close()
