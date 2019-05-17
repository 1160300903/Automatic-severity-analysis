import numpy as np
import math 
import argparse
import json
import nltk
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from utils import *
from sklearn.model_selection import train_test_split
from sklearn import metrics
class Knn():
    def __init__(self,model,in_file,out_file):
        print(in_file)
        self.model = model
        self.in_file = in_file
        self.out_file = out_file
        self.stop_words = stopwords.words("english")
        self.train_corpus = None
        self.train_tag = None
        self.test_corpus = None
        self.test_tag = None
        self.train_array = None
        self.test_array = None
        self.corpus = None
        self.tag = None
        self.BATCH_SIZE = 1000
        self.nn_sev = None
        self.nn_sim = None
    def sent2vec(self,s): 
        words = s 
        M = [] 
        for w in words: 
            try: 
                M.append(self.model[w]) 
            except:
                continue 
        M = np.array(M) 
        v = M.sum(axis=0) 
        sum = np.sqrt((v ** 2).sum())
        #单位化
        return v / sum
    def init_corpus(self):
        severity_index = {'blocker':0, 'critical':1,'major':2, 'minor':3, 'trivial':4}
        self.corpus=[]
        self.tag=[]
        with open(self.in_file, encoding='utf-8') as fin:
            for line in fin.readlines():
                temp = json.loads(line.strip())
                self.corpus.append(temp["summary"])
                self.tag.append(severity_index[temp["severity"]])
    def split_corpus(self,total):
        train_x, test_x, train_y, test_y = train_test_split(self.corpus, self.tag, test_size=1/total)
        self.train_corpus = train_x
        self.train_tag = train_y
        self.test_corpus = test_x
        self.test_tag =test_y
    def train(self):
        self.train_array = []
        i=0
        for sentence in self.train_corpus:
            i+=1
            if not i%1000:
                print("training:"+str(i))
            sentence_vector = self.sent2vec(sentence)
            self.train_array.append(sentence_vector)
        self.train_array = np.array(self.train_array)
    def predict(self):
        self.test_array = []
        for sentence in self.test_corpus:
            sentence_vector = self.sent2vec(sentence)
            self.test_array.append(sentence_vector)
        self.test_array = np.array(self.test_array)
        total = len(self.test_corpus)
        self.nn_sev = []
        self.nn_sim = []
        train_tag_array = np.array(self.train_tag)
        for i in range(0,total , self.BATCH_SIZE):
            print("predicting:batch"+str(i))
            j = min(i + self.BATCH_SIZE, total)
            similarities = self.test_array[i:j,:].dot(self.train_array.T)
            for i in range(similarities.shape[0]):
                row = similarities[i,:]
                #找到前25的下标
                ind = np.argpartition(row, -15)[-25:]
                ind = ind[np.argsort(-row[ind])]
                self.nn_sim.append(row[ind])
                self.nn_sev.append(train_tag_array[ind])
        """with open(self.out_file,"w") as r:
            for i in range(len(self.test_corpus)):
                for j in range(25):
                    r.write(str(self.nn_sev[i][j])+" ")
                r.write("\n")
                for j in range(25):
                    r.write(str(self.nn_sim)+" ")
                r.write("\n")"""  
    def get_tag(self,k):
        result = []
        for i in range(len(self.test_corpus)):
            a, b = 0, 0
            for j in range(k):
                #sim_temp = (self.nn_sim[i][j]+1)/2
                a += self.nn_sev[i][j]*self.nn_sim[i][j]
                b += self.nn_sim[i][j]
            #temp = min( 6, int(a/b+0.5) )
            #temp = max(0,temp)
            result.append( int(a/b+0.5) )
        return result
if __name__=="__main__":
    model = load_word_embedding("wordEmbedding/svd.model.txt")
    k = Knn(model,"data/full_data/gcc_full.json","knn/knn.txt")
    k.init_corpus()
    k.split_corpus(5)
    print("training")
    k.train()
    print("predicting")
    k.predict()
    for i in range(5,26,5):
        tag = k.get_tag(i)
        print(i,metrics.accuracy_score(k.test_tag, tag, 5))