import re
import nltk
from nltk.corpus import stopwords
import numpy as np
from keras.preprocessing import sequence
from keras.utils import to_categorical
from gensim.corpora.dictionary import Dictionary
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Dropout
from keras.layers.core import Activation
from sklearn.model_selection import train_test_split
from utils import *
import yaml
import argparse
import csv
import json
from sklearn import metrics
import collections
def get_corpus(path):
    severity_index = { 'blocker':0, 'critical':1,'major':2, 'minor':3, 'trivial':4}
    a=[]
    tag=[]
    with open(path, encoding='utf-8') as fin:
        for line in fin.readlines():
            temp = json.loads(line.strip())
            a.append(" ".join(temp["summary"]))
            tag.append(severity_index[temp["severity"]])
    stop_words = stopwords.words("english")
    split_corpus = [[word for word in nltk.word_tokenize(x) if word not in stop_words] for x in a]
    split_corpus = np.array(split_corpus)
    tag = np.array(tag)
    tag = to_categorical(tag,num_classes=5) 
    return split_corpus,tag
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='binary sentiment analysis')
    parser.add_argument("-v","--vector_length",type=int,default=300,help="the length of word embeddings")
    parser.add_argument("-s","--sentence_length",type=int,default=50,help="the length of a sentence")
    parser.add_argument("-f","--file_name",default="data/full_data/gcc_full.json",help="the data")
    args = parser.parse_args()
    embedding_length = args.vector_length
    sentence_length = args.sentence_length
    print(args.file_name)
    model = load_word_embedding('wordEmbedding/svd.model.txt')
    corpus, tag = get_corpus(args.file_name)
    word_dict = Dictionary()
    word_dict.doc2bow(model.keys(), allow_update=True)
    word_index = {word:index+1 for index, word in word_dict.items()}
    index_form_corpus = []
    for sentence in corpus:
        index_form_sentence = []
        for word in sentence:
            if word in word_index:
                index_form_sentence.append(word_index[word])
            else:
                index_form_sentence.append(0)
        index_form_corpus.append(index_form_sentence)
    index_form_corpus = sequence.pad_sequences(index_form_corpus,maxlen=sentence_length)
    index_vectors_matrix = np.zeros((len(word_index)+1,embedding_length))
    for word,index in word_index.items():
        index_vectors_matrix[index,:] = model[word]
    lstm_model = Sequential()
    lstm_model.add(Embedding(output_dim=embedding_length,
    input_dim=len(word_index)+1,
    mask_zero=True,
    weights = [index_vectors_matrix],
    input_length=sentence_length))
    lstm_model.add(LSTM(activation="tanh",units=50))
    lstm_model.add(Dropout(0.5))
    lstm_model.add(Dense(5))
    lstm_model.add(Activation("softmax"))
    print("begin training")
    lstm_model.compile(loss="categorical_crossentropy", optimizer="adam",metrics=["accuracy"])
    #分割数据集
    train_x, test_x, train_y, test_y = train_test_split(index_form_corpus, tag, test_size=0.2)
    lstm_model.fit(train_x, train_y, batch_size=32, epochs=5, verbose=1)
    #result = lstm_model.to_yaml()
    #with open("lstm.yml","w") as output:
        #output.write(yaml.dump(result,default_flow_style=True))
    #lstm_model.save_weights("weights.h5")
    test_y = one_hot_to_num(test_y,5)
    result = lstm_model.predict_classes(test_x, batch_size=32)
    print(collections.Counter(test_y))
    print(collections.Counter(result))
    print(metrics.accuracy_score(test_y,result))
    print(metrics.f1_score(test_y, result, average='macro'))
    #print("accuracy:",accuracy) 
