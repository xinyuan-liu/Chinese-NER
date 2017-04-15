#coding=utf-8
import numpy as np
from keras.preprocessing.sequence import pad_sequences
import pickle
max_sen_len=100
f=open('ner_trn','r')
X_train=[]
Y_train=[]
x_sen=[]
y_sen=[]
dict_word={}
dict_tag={}
dict_word['PAD']=0
dict_word['UNK']=1
dict_tag['PAD']=0
for line in f:
    line=line.strip()
    if line == "":
        X_train.append((x_sen))
        x_sen=[]
        Y_train.append((y_sen))
        y_sen=[]
        continue
    line=line.split(' ')
    if line[0] in dict_word:
        x_sen.append(dict_word[line[0]])
    else:
        index=len(dict_word)
        dict_word[line[0]]=index
        x_sen.append(index)

    if line[1] in dict_tag:
        y_sen.append(dict_tag[line[1]])
    else:
        index=len(dict_tag)
        dict_tag[line[1]]=index
        y_sen.append(index)
index_word={}
for word in dict_word:
    index_word[dict_word[word]]=word

X_train_cut=[]
Y_train_cut=[]
for i in range(len(X_train)):
    if len(X_train[i])<=max_sen_len:
        X_train_cut.append(X_train[i])
        Y_train_cut.append(Y_train[i])
        continue
    while len(X_train[i])>max_sen_len:
        flag=False
        for j in reversed(range(100)):
            if X_train[i][j]==dict_word['，'] or X_train[i][j]==dict_word['、']:
                X_train_cut.append(X_train[i][:j+1])
                Y_train_cut.append(Y_train[i][:j+1])
                X_train[i]=X_train[i][j+1:]
                Y_train[i]=Y_train[i][j+1:]
                break
            if j==0:
                flag=True    
        if flag:
            X_train_cut.append(X_train[i][:100])
            Y_train_cut.append(Y_train[i][:100])
            X_train[i]=X_train[i][100:]
            Y_train[i]=Y_train[i][100:]
X_train=pad_sequences(X_train_cut,maxlen=max_sen_len,padding='post')
print(X_train.shape)
Y_train=pad_sequences(Y_train_cut,maxlen=max_sen_len,padding='post')
print(Y_train.shape)

index_tag={}
for tag in dict_tag:
    index_tag[dict_tag[tag]]=tag

pickle.dump(index_word,open('index_word.pkl','wb'),2)
pickle.dump(index_tag,open('index_tag.pkl','wb'),2)
pickle.dump(dict_word,open('dict_word.pkl','wb'),2)
pickle.dump(dict_tag,open('dict_tag.pkl','wb'),2)
pickle.dump(X_train,open('X_train.pkl','wb'),2)
pickle.dump(Y_train,open('Y_train.pkl','wb'),2)
