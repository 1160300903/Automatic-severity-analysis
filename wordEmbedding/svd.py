import numpy as np
import re
import nltk
from math import log2
from nltk.corpus import stopwords
import json
import collections
TOP_WORD_SIZE = 100
VECTOR_LENGTH = 300
WORD_FREQ = 5
WINDOW_LENGTH = 3
filepaths = ["eclipse_nohttp.json","mozilla_nohttp.json","gcc_nohttp.json"]
stop_words = stopwords.words("english")
def get_matrix():
    split_corpus = []
    for i in range(len(filepaths)):
        lines = []
        filepath = "../data/nohttp_data/"+filepaths[i]
        print(filepath)
        with open(filepath,"r") as fin:
            for line in fin.readlines():
                temp  = json.loads(line.strip())
                lines+=temp["summary"]
        split_corpus += [[word for word in nltk.word_tokenize(x) if word not in stop_words] for x in lines]
    temp_corpus = []
    for line in split_corpus:
        for word in line:
            temp_corpus.append(word)
    freq = collections.Counter(temp_corpus)
    all_word_dict = {a:freq[a] for a in freq if freq[a]>=WORD_FREQ}
    context_word_dict = {a:freq[a] for a in freq if freq[a]>=10}
    del temp_corpus
    del freq
    all_words_index = {}
    context_words_index = {}
    i, j = 0, 0
    for word in all_word_dict:
        all_words_index[word] = i
        i+=1
    for word in context_word_dict:
        context_words_index[word] = j
        j+=1
    del all_word_dict
    del context_word_dict
    print("length of text: "+str(len(split_corpus)))
    print("length of wordDist: "+str(len(all_words_index)))
    print("length of contextDist: "+str(len(context_words_index)))
    x = np.zeros((len(all_words_index),len(context_words_index)),dtype="float32")
    for words in split_corpus:
        length  = len(words)
        for j in range(length):
            for i in range(1,WINDOW_LENGTH+1):
                if j-i >= 0 and words[j] in all_words_index and words[j-i] in context_words_index:
                    x[all_words_index[words[j]]][context_words_index[words[j - i]]] += 1
                if j+i < length and words[j] in all_words_index and words[j+i] in context_words_index:
                    x[all_words_index[words[j]]][context_words_index[words[j + i]]] += 1
    return x, all_words_index
def compute_pmi(x,all_words_index):
    print(x.dtype)
    x += 1e-8
    total = x.sum()
    word_sum = x.sum(axis=1)
    context_sum = x.sum(axis=0)
    #for i in range(x.shape[0]):
        #print(word_sum[i],context_sum[i])
        #assert word_sum[i]==context_sum[i]
    x = np.log2(x)
    x += log2(total)
    for r in range(x.shape[0]):
        x[r,:] -= log2(word_sum[r])
    for c in range(x.shape[1]):
        x[:,c] -= log2(context_sum[c])
    return x
def get_reduced_dimension(x):
    la = np.linalg
    U, X = la.svd(x,full_matrices=0)[:2]
    length = VECTOR_LENGTH if X.shape[0]>VECTOR_LENGTH else X.shape[0]
    U = U[:,:length]
    X = X[:length]
    for i in range(length):
        U[:,i]*= X[i]
    return U


x, all_words_index = get_matrix()#统计数据，并存储

x = compute_pmi(x,all_words_index)#给矩阵计算pmi
U = get_reduced_dimension(x)#降维
output = open("svd.model.txt","w",encoding="utf-8")
output.write(str(len(all_words_index))+" "+str(VECTOR_LENGTH)+"\n")
for word in all_words_index:
    array = U[all_words_index[word]]
    output.write(word)
    for i in range(min(VECTOR_LENGTH,len(all_words_index))):
        output.write(" %.8f"%array[i])
    output.write("\n")
output.close()
