import numpy as np
from collections import Counter
from math import log2
import re
import jieba
from gensim.corpora.dictionary import Dictionary
from sklearn.model_selection import train_test_split
from collections import defaultdict
import json
class NaiveBayes():
    def __init__(self,smooth=0.5,class_num=7):
        self.prior = {}
        self.condition = {}
        self.smooth = max(0,smooth)#平滑参数，输入负数或者0等于不平滑
        self.class_num = class_num
    def trainer(self,data,):
        Y = data[:,-1]
        priorCounter = Counter(Y)
        freq = {a[0]:a[1] for a in priorCounter.most_common()}
        for tag in range(self.class_num):
            freq[tag] = freq[tag] + self.smooth if tag in freq else self.smooth
        print("begin to compute prior probability")
        #初始化先验概率
        total = len(Y)+self.smooth*self.class_num#总共7类，加7次平滑值
        self.prior = {elem : log2(freq[elem])-log2(total) for elem in freq}
        #初始化条件概率
        for state in freq:
            self.condition[state] = {}
            for i in range(data.shape[1]-1):
                self.condition[state][i] = {i:self.smooth for i in range(self.class_num)}
        print("begin to comput conditional probability")
        for i in range(data.shape[0]):
            y = data[i][-1]#获得第i个用例的分类
            for j in range(data.shape[1]-1):
                self.condition[y][j][data[i][j]] = self.condition[y][j][data[i][j]]+1
        print("begin to normalize")
        #normalize
        for y in self.condition:
            for x in self.condition[y]:
                total = freq[y]+self.smooth*2
                #       获得y类的所有对象     对y类x属性的每个值进行平滑，每个属性2个属性值
                for a in self.condition[y][x]:
                    #print(self.condition[y][x][a],total)
                    #print(y,x,a)
                    self.condition[y][x][a] = log2(self.condition[y][x][a])-log2(total)
                
    def predictor(self,data):
        Y = data[:,-1]
        res = []
        for i in range(Y.shape[0]):
            result = None
            max = -float("inf")
            for y in self.prior:
                prob = self.prior[y]
                for x in self.condition[y]:
                    #print(y,x,data[i,x])    
                    prob += self.condition[y][x][data[i,x]]
                (max, result) = (prob, y) if prob>max else (max, result)
                #print("\n"+str(i))
                #print("class:"+str(y)+" probability"+str(prob),end="")
            assert result!=None
            res.append(result)
        return res
def get_corpus(path,train=True):
    severity_index = {'blocker':0, 'critical':1,'major':2, 'minor':3, 'trivial':4}
    split_corpus=[]
    tag=[]
    with open(path, encoding='utf-8') as fin:
        for line in fin.readlines():
            temp = json.loads(line.strip())
            split_corpus.append(temp["summary"])
            tag.append(severity_index[temp["severity"]])
    frequency = defaultdict(int)
    for line in split_corpus:
        for token in line:
            frequency[token] += 1
    split_corpus = [[word for word in x if frequency[word] >= 5] for x in split_corpus]
    word_dict = Dictionary(split_corpus)
    word_index = {word:index for index, word in word_dict.items()}
    print(len(word_index))
    feature_matrix = np.zeros((len(split_corpus),len(word_index)),dtype=int)#特征向量，最后一行是分类标签
    for i in range(len(split_corpus)):
        for word in split_corpus[i]:
            feature_matrix[i,word_index[word]] = 1
    return feature_matrix,tag
def k_():
    nb = NaiveBayes()
    all_data, tag = get_corpus("data/full_data/eclipse_full.json")
    train_x, test_x, train_y, test_y = train_test_split(all_data, tag, test_size=0.2)
    train_y = np.reshape(train_y,(len(train_y),1))
    nb.trainer(np.hstack((train_x,train_y)))
    test_y = np.reshape(test_y,(len(test_y),1))
    test = np.hstack((test_x,test_y))
    res = nb.predictor(test)
    print(test_y.shape)
    count = 0
    for i in range(test_y.shape[0]):
        count = count+1 if test_y[i][0]==res[i] else count
    print("test accuracy:"+str(count/test_y.shape[0]))
if __name__ == "__main__":
    k_()
        
