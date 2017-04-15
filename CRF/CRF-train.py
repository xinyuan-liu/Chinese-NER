import sklearn
import sys
import scipy.stats
from sklearn.metrics import make_scorer
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import RandomizedSearchCV

import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
#import data

X_train = []
Y_train=[]
f=open(sys.argv[1],'r')
trn_list=[]
sen=[]
for line in f:
    if line=='\n' or not line:
        trn_list.append(sen)
        sen=[]
        continue
    sen.append(line.strip())
for s in trn_list:
    l=len(s)
    fea_s=[]
    y_s=[]
    for i in range(l):
        fea_w={}
        for j in range(-3,3):
            k=i+j
            if k in range(l):
                fea_w[str(j)]=s[k].split(' ')[0]
            else:
                fea_w[str(j)]='<PAD>'
        fea_s.append(fea_w)
        y_s.append(s[i].split(' ')[1])
    X_train.append(fea_s)
    Y_train.append(y_s)
#print X_train
#print Y_train
#y_train = list(data.Y_train)
f.close()
crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.1,
    c2=0.1,
    max_iterations=100,
    all_possible_transitions=True
)
crf.fit(X_train, Y_train)
f=open(sys.argv[2],'r')
dev_list=[]
X_dev=[]
Y_dev=[]
for line in f:
    if line=='\n' or not line:
        dev_list.append(sen)
        sen=[]
        continue
    sen.append(line.strip())
for s in dev_list:
    l=len(s)
    fea_s=[]
    y_s=[]
    for i in range(l):
        fea_w={}
        for j in range(-3,3):
            k=i+j
            if k in range(l):
                fea_w[str(j)]=s[k].split(' ')[0]
            else:
                fea_w[str(j)]=' '
        fea_s.append(fea_w)
    X_dev.append(fea_s)

y_pred = crf.predict(X_dev)

labels = list(crf.classes_)

f=open('ner_pre','w')
labels.remove('O')
for i in range(len(X_dev)):
    for j in range(len(X_dev[i])):
        f.write( X_dev[i][j]['0']+' '+y_pred[i][j]+'\n')
    f.write('\n')

